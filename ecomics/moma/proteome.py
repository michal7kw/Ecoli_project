"""The proteome module: an ensemble of network-neighbour LASSO regressions.

Paper (paper.md:170, 79-81):

    "The protein expression values of 1001 proteins were predicted using LASSO
     constraint regression through a consensus of the transcriptional
     regulatory, PPI, co-expression network and other pathway information...
     The protein level of a target gene in a novel condition is predicted by
     LASSO regression of the expression levels of genes that are connected
     through a regulatory link to the target gene."

The finding this implements
---------------------------
Predicting a protein from ITS OWN mRNA works poorly: PCC 0.34 overall, and
0.18 +/- 0.51 for the 50 most variable proteins -- a standard deviation larger
than the mean, i.e. useless. But predicting it from the expression of its
FUNCTIONAL NEIGHBOURS reaches R^2 = 0.79.

Two reasons, and both matter:
  * Biology -- a protein's abundance is set by the physiological state of the
    module it belongs to (its regulon, complex, pathway), and the transcriptome
    reports that state redundantly through many correlated genes, whereas its
    own mRNA is filtered through gene-specific translation efficiency and
    protein half-life, which vary by orders of magnitude.
  * Statistics -- averaging over neighbours averages away the independent
    measurement noise on each. With only a handful of paired profiles, that
    matters a great deal.

Note this is, in modern terms, a hand-built graph neural network: predict a
node's value from its neighbours' values, over four graphs, then combine. The
inductive bias was the right one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ecomics.moma._lasso import fit_lasso
from ecomics.networks import Network

__all__ = ["NeighbourLassoPredictor", "ProteomeEnsemble"]


@dataclass
class NeighbourLassoPredictor:
    """One network's predictor: LASSO from a target's neighbours' expression."""

    network: Network
    alpha: float | None = 1e-3       # fixed; see the note in ProteomeEnsemble
    max_neighbours: int = 200
    models: dict[str, object] = field(default_factory=dict, repr=False)
    used: dict[str, list[str]] = field(default_factory=dict, repr=False)
    # Failed fits, kept apart from "this network has no neighbours for it".
    # Coverage is a headline number here, so conflating the two would misreport
    # the paper's central proteome claim -- see `_lasso.fit_lasso`.
    errors: dict[str, str] = field(default_factory=dict, repr=False)

    @property
    def name(self) -> str:
        return self.network.name

    def fit(self, X: np.ndarray, x_cols: list[str],
            Y: np.ndarray, y_cols: list[str]) -> "NeighbourLassoPredictor":
        """X: (n_samples x n_genes) transcript levels. Y: (n_samples x n_proteins)."""
        xi = {c: i for i, c in enumerate(x_cols)}
        for j, target in enumerate(y_cols):
            nb = [g for g in self.network.of(target) if g in xi]
            if not nb:
                continue
            # Cap the neighbourhood: with only a handful of training samples a
            # 500-neighbour design is pure noise, and LASSO's selection becomes
            # arbitrary. Keep the most variable neighbours.
            if len(nb) > self.max_neighbours:
                cols = [xi[g] for g in nb]
                var = np.nanvar(X[:, cols], axis=0)
                nb = [nb[k] for k in np.argsort(-var)[:self.max_neighbours]]

            cols = [xi[g] for g in nb]
            Xd = np.nan_to_num(X[:, cols])
            yd = Y[:, j]
            ok = np.isfinite(yd)
            if ok.sum() < 3:
                continue
            m, err = fit_lasso(Xd[ok], yd[ok], alpha=self.alpha)
            if m is None:
                self.errors[target] = err
                continue
            self.models[target] = m
            self.used[target] = nb
        return self

    def predict(self, X: np.ndarray, x_cols: list[str], y_cols: list[str]
                ) -> np.ndarray:
        """Returns (n_samples x n_proteins); NaN where this network has no model."""
        xi = {c: i for i, c in enumerate(x_cols)}
        out = np.full((X.shape[0], len(y_cols)), np.nan)
        for j, target in enumerate(y_cols):
            m = self.models.get(target)
            if m is None:
                continue
            cols = [xi[g] for g in self.used[target]]
            out[:, j] = m.predict(np.nan_to_num(X[:, cols]))
        return out

    @property
    def coverage(self) -> int:
        return len(self.models)


@dataclass
class ProteomeEnsemble:
    """Average the four network predictors, per protein, over whoever covers it.

    NOTE ON alpha: a fixed penalty is used rather than LassoCV. With only 5
    shared transcriptome/proteome conditions, an inner cross-validation over 3
    folds of 4 samples selects noise, and it multiplies runtime by ~40x for
    4 networks x 589 proteins x 5 outer folds. A fixed, documented alpha is the
    more honest choice at this sample size.

    Averaging only over the networks that HAVE a model for a given protein is
    what produces the paper's coverage result: TRN alone reaches 250 proteins,
    KEGG 547, CPN 847, PPI 1,000, but the union covers all 1,001.
    """

    networks: dict[str, Network]
    alpha: float | None = 1e-3
    # Forwarded to every NeighbourLassoPredictor. It previously was not, so the
    # ensemble path silently ignored the caller's cap and always used 200.
    max_neighbours: int = 200
    predictors: dict[str, NeighbourLassoPredictor] = field(default_factory=dict)

    def fit(self, X: np.ndarray, x_cols: list[str], Y: np.ndarray,
            y_cols: list[str], verbose: bool = False) -> "ProteomeEnsemble":
        for name, net in self.networks.items():
            p = NeighbourLassoPredictor(net, alpha=self.alpha,
                                        max_neighbours=self.max_neighbours)
            p.fit(X, x_cols, Y, y_cols)
            self.predictors[name] = p
            if verbose:
                print(f"    {name:<5s} fitted {p.coverage:>4d}/{len(y_cols)} proteins")
        return self

    def predict_each(self, X: np.ndarray, x_cols: list[str], y_cols: list[str]
                     ) -> dict[str, np.ndarray]:
        return {n: p.predict(X, x_cols, y_cols) for n, p in self.predictors.items()}

    def predict(self, X: np.ndarray, x_cols: list[str], y_cols: list[str]
                ) -> np.ndarray:
        """Ensemble prediction: the mean over networks that cover each protein."""
        each = self.predict_each(X, x_cols, y_cols)
        if not each:
            return np.full((X.shape[0], len(y_cols)), np.nan)
        stack = np.stack(list(each.values()))          # (n_nets, n_samples, n_prot)
        with np.errstate(invalid="ignore"):
            return np.nanmean(stack, axis=0)

    def coverage(self, y_cols: list[str] | None = None) -> dict[str, int]:
        """Proteins each network successfully modelled, plus the union.

        `y_cols` restricts the count to a given target set; passing None counts
        everything modelled. It used to be a required argument that the body
        never read, which is worse than either -- a caller could pass the wrong
        target list and get a plausible number back. `scripts/04` does not use
        this method at all; it recomputes coverage from finite predictions,
        which is the same quantity measured after the LOCO folds rather than
        before.
        """
        keep = set(y_cols) if y_cols is not None else None

        def count(models) -> set[str]:
            names = set(models)
            return names if keep is None else names & keep

        cov = {n: len(count(p.models)) for n, p in self.predictors.items()}
        union: set[str] = set()
        for p in self.predictors.values():
            union |= count(p.models)
        cov["ENSEMBLE"] = len(union)
        return cov


def own_mrna_baseline(X_train: np.ndarray, Y_train: np.ndarray,
                      X_test: np.ndarray, x_cols: list[str],
                      y_cols: list[str]) -> np.ndarray:
    """The paper's baseline: predict each protein from its OWN transcript.

    Reported at PCC 0.34 +/- 0.18 overall, and 0.18 +/- 0.51 on the 50 most
    variable proteins -- a standard deviation larger than the mean. Implemented
    as a per-protein univariate least-squares fit on the TRAINING fold only,
    applied to the test fold, which is the most favourable honest reading of
    "linear protein inference from transcriptional level".
    """
    xi = {c: i for i, c in enumerate(x_cols)}
    out = np.full((X_test.shape[0], len(y_cols)), np.nan)
    for j, target in enumerate(y_cols):
        i = xi.get(target)
        if i is None:
            continue
        x, y = X_train[:, i], Y_train[:, j]
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 3 or x[ok].std() < 1e-12:
            continue
        b, a = np.polyfit(x[ok], y[ok], 1)
        out[:, j] = a + b * np.nan_to_num(X_test[:, i])
    return out

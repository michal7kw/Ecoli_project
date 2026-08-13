"""The phenome module: growth rate as a performance-weighted consensus.

Paper (paper.md:180):

    "Final growth rate prediction was calculated by the weighted sum of
     predicted growth rates from each individual layer, with the weight
     proportional to the performance of the respective layer during
     cross-validation, and each layer predicts growth rate from concentration
     of molecules in the layer as well as extra-cellular information based on
     LASSO constrained regression."

    mu_hat = sum_k  (rho_k / sum_j rho_j) * mu_hat_k

Two things worth appreciating in the paper's own numbers (Fig. 5f, 5g):

  * The INPUT LAYER ALONE reaches PCC ~0.59 -- the condition description
    already says most of what there is to say about growth rate, which is
    unsurprising since medium richness dominates growth. All five layers
    together reach ~0.65. The molecular layers add ~0.06.
  * Individual layers UNDERPERFORM the input layer (proteome ~0.39, metabolome
    ~0.45) yet still help in the consensus. That is the ensemble signature: a
    weak predictor with decorrelated errors adds information even when its solo
    accuracy is poor. Weighting by CV performance stops the weak ones dragging
    the consensus down.

A caveat the paper does not flag: weighting by cross-validated performance uses
the CV data to set a model parameter, so unless the weights are fitted in a
nested inner loop the reported CV score is mildly optimistic. `fit` here takes
the weights from an INNER split of the training fold only.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.model_selection import KFold

from ecomics.moma._lasso import coefficients, fit_lasso

__all__ = ["LayerGrowthPredictor", "PhenomeConsensus"]


@dataclass
class LayerGrowthPredictor:
    """LASSO from one layer's molecular levels (+ the condition features)."""

    name: str
    alpha: float | None = None
    model: object | None = field(default=None, repr=False)
    n_selected: int = 0
    fit_error: str | None = None

    def fit(self, F: np.ndarray, y: np.ndarray) -> "LayerGrowthPredictor":
        ok = np.isfinite(y)
        Xd = np.nan_to_num(F[ok])
        if ok.sum() < 4 or Xd.shape[1] == 0:
            return self
        # `fit_error` records WHY on failure. A bare `self.model = None` makes a
        # failed layer indistinguishable from an absent one: the proteome layer
        # failed to fit for every fold and contributed nothing, silently, while
        # looking like a layer that was simply not covered.
        self.model, self.fit_error = fit_lasso(
            Xd, y[ok], alpha=self.alpha, cv_max=4, min_for_cv=1)
        if self.model is not None:
            self.n_selected = int((np.abs(coefficients(self.model)) > 1e-10).sum())
        return self

    def predict(self, F: np.ndarray) -> np.ndarray:
        """Predict, returning NaN where this layer has no data for a sample.

        The distinction that matters is between a row with SOME values missing
        and a row with NO data at all. `nan_to_num` treats them alike, which is
        correct for the first (impute the gaps) and badly wrong for the second:
        a condition that was never measured on this layer gets predicted as
        though every molecule were at zero, and the result looks like a real
        prediction. Under a consensus that renormalizes over available layers,
        that fabricated value is then given weight.

        Measured cost of getting this wrong: the metabolome layer covers 49 of
        226 conditions but emitted predictions for all 226, received the largest
        consensus weight (0.581) on the strength of them, and dragged the
        five-layer consensus from 0.405 to 0.078.
        """
        if self.model is None:
            return np.full(F.shape[0], np.nan)
        out = self.model.predict(np.nan_to_num(F))
        out[np.isnan(F).all(axis=1)] = np.nan
        return out

    def selected(self, names: list[str]) -> list[tuple[str, float]]:
        """Non-zero coefficients, largest first -- the paper's Table 1 analysis.

        Read these as PREDICTIVE importance within a joint model, not as causal
        claims: the paper shows that restricting to the top ten collapses
        performance from PCC 0.65 to 0.08.
        """
        if self.model is None:
            return []
        coef = coefficients(self.model)
        idx = np.flatnonzero(np.abs(coef) > 1e-10)
        return [(names[i], float(coef[i])) for i in idx[np.argsort(-np.abs(coef[idx]))]]


@dataclass
class PhenomeConsensus:
    """Weighted consensus of per-layer growth predictors."""

    layers: dict[str, LayerGrowthPredictor] = field(default_factory=dict)
    weights: dict[str, float] = field(default_factory=dict)

    def fit(self, features: dict[str, np.ndarray], y: np.ndarray,
            n_splits: int = 3, seed: int = 0, verbose: bool = False
            ) -> "PhenomeConsensus":
        """Fit each layer, then set weights from an INNER CV of the training data.

        The inner split is what keeps the consensus weights out of the outer
        evaluation -- see the module docstring.
        """
        finite_y = np.isfinite(y)

        for name, F in features.items():
            # Per-LAYER availability, not one shared mask. The omics layers cover
            # different conditions -- 179 transcriptome, 30 proteome, 46
            # metabolome -- and their strict intersection is 5. Requiring every
            # layer on every row would throw away almost all the data; instead
            # each layer trains on the rows where it exists, and `predict`
            # renormalizes the weights per sample over the layers that are
            # present. A row missing a layer is passed as NaN.
            ok = finite_y & np.isfinite(F).all(axis=1)
            if ok.sum() < 4:
                continue                        # nothing to fit this layer on
            y_ok = y[ok]
            F_ok = F[ok]
            # inner CV to score this layer
            scores = []
            if ok.sum() >= 2 * n_splits:
                for tr, te in KFold(n_splits=n_splits, shuffle=True,
                                    random_state=seed).split(F_ok):
                    p = LayerGrowthPredictor(name).fit(F_ok[tr], y_ok[tr])
                    pred = p.predict(F_ok[te])
                    if np.isfinite(pred).all() and np.std(pred) > 1e-12 \
                            and np.std(y_ok[te]) > 1e-12:
                        scores.append(np.corrcoef(pred, y_ok[te])[0, 1])
            rho = float(np.nanmean(scores)) if scores else 0.0
            self.weights[name] = max(rho, 0.0)     # a negatively-scoring layer
            #                                        contributes nothing
            self.layers[name] = LayerGrowthPredictor(name).fit(F_ok, y_ok)
            if verbose:
                print(f"    {name:<14s} rho={rho:+.3f}  "
                      f"{self.layers[name].n_selected} features selected")

        total = sum(self.weights.values())
        if total <= 0:
            n = max(len(self.weights), 1)
            self.weights = {k: 1.0 / n for k in self.weights}
        else:
            self.weights = {k: v / total for k, v in self.weights.items()}
        return self

    def predict(self, features: dict[str, np.ndarray]) -> np.ndarray:
        n = next(iter(features.values())).shape[0]
        num = np.zeros(n)
        den = np.zeros(n)
        for name, F in features.items():
            p = self.layers[name].predict(F) if name in self.layers else None
            if p is None:
                continue
            w = self.weights.get(name, 0.0)
            good = np.isfinite(p)
            num[good] += w * p[good]
            den[good] += w
        return np.divide(num, den, out=np.full(n, np.nan), where=den > 0)

    def predict_each(self, features: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {n: self.layers[n].predict(F)
                for n, F in features.items() if n in self.layers}

"""The proteome ensemble on the paper's OWN networks (Supplementary Data 2).

A SECOND wiring of the same model, kept beside `proteome.py` rather than folded
into it behind a flag:

    proteome.py        the ensemble, on graphs fetched from the web in 2026
    proteome_paper.py  the same ensemble, on the graphs the paper published

WHAT IS AND IS NOT A SECOND IMPLEMENTATION
------------------------------------------
Only the GRAPHS differ. `ProteomeEnsemble` already takes
`networks: dict[str, Network]`, so it never knew where its edges came from, and
copying `NeighbourLassoPredictor` here would produce ~200 lines that drift the
first time either copy is fixed. This module therefore supplies graphs and
policy and delegates the regression -- which also means alpha, the neighbour cap
and the LASSO itself are provably identical across the comparison, so any
difference in the result is the graphs.

That is the opposite of the choice `transcriptome_paper.py` makes, and for the
opposite reason: there the paper specifies a different MODEL (plain SGD, fixed
epochs, L1), so a separate class is what keeps the comparison interpretable.
Here the paper specifies the same model on different data.

THE SIX ARMS
------------
The paper ran six network predictors; `proteome.py` runs four. Data 2 supplies
the two that were never implemented here at all -- a sigma-factor network and a
small-RNA network -- alongside its own TRN, PPI and KEGG pathway graphs.

    TRN PPI KEGG SIGMA SRNA   from Data 2, external knowledge, no leakage
    CPN                       built HERE from held-out proteome conditions

Read `SRNA`'s number next to its coverage, always: it reaches 5 of our 583
proteome targets, so it is an ABSENT predictor rather than a failed one, and a
PCC alone will not say which.

THE LEAKAGE RULE, AND WHY IT IS ENFORCED BY NAME
------------------------------------------------
Data 2 also ships the paper's CPN. That one is not external knowledge: it is a
correlation network computed over the FULL compendium, so every condition a LOCO
fold holds out is already inside its edges. Averaging it in would score the
model partly on its memory of the test fold.

`ProteomeEnsemble.predict` nanmeans whatever it was given. This class does not:
it drops any arm whose name ends in `networks_paper.LEAKY_SUFFIX` from the mean,
while still fitting and reporting it. So the leak can be MEASURED -- which is
worth knowing -- without any path by which it reaches a headline number. The
guard lives in the arm's name rather than in a config flag because names survive
being copied into a results file and a table; flags do not.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ecomics.moma.proteome import ProteomeEnsemble
from ecomics.networks import Network, build_cpn
from ecomics.networks_paper import LEAKY_SUFFIX, ParseReport, load_paper_networks

__all__ = ["PaperProteomeEnsemble", "build_paper_ensemble_networks", "is_leaky"]


def is_leaky(name: str) -> bool:
    """Whether an arm carries the compendium-wide CPN, by its name alone."""
    return name.endswith(LEAKY_SUFFIX)


def build_paper_ensemble_networks(
        cpn_values: np.ndarray | None = None,
        cpn_columns: list[str] | None = None,
        include_leaky_cpn: bool = False,
        verbose: bool = True,
) -> tuple[dict[str, Network], dict[str, ParseReport]]:
    """The paper's five graphs, plus a CPN built here from `cpn_values`.

    `cpn_values` must be proteome profiles HELD OUT of evaluation, exactly as
    `scripts/04_reproduce.py:eval_proteome` builds them. Passing the evaluation
    matrix instead is leakage of the same kind this module refuses for the
    paper's own CPN, and nothing downstream can detect it -- the shape is right
    either way.
    """
    nets, reports = load_paper_networks(include_leaky_cpn=include_leaky_cpn,
                                        verbose=verbose)
    if cpn_values is not None and cpn_columns is not None:
        cpn = build_cpn(cpn_values, cpn_columns)
        nets["CPN"] = cpn
        if verbose:
            print(f"  {'CPN':<16s} {cpn.n_nodes:>5d} nodes  {cpn.n_edges:>7d} edges"
                  f"   (built here, held-out conditions)")
    return nets, reports


@dataclass
class PaperProteomeEnsemble:
    """`ProteomeEnsemble` with the leaky-arm rule, over the paper's networks.

    Every hyper-parameter is passed straight through, and the defaults match
    `ProteomeEnsemble`'s, so a comparison run varies the graphs and nothing else.

    NOTE ON `max_neighbours`: the paper's PPI gives a MEDIAN of 286 usable
    neighbours per proteome target against 4 training samples per LOCO fold, so
    the cap binds on most proteins and the variance ranking inside
    `NeighbourLassoPredictor.fit` -- not the graph -- chooses the final design
    matrix. Keep it at 200 for comparability with `proteome.py`; sweeping it is
    a separate experiment, and sweeping it in the same run would confound the
    two.
    """

    networks: dict[str, Network]
    alpha: float | None = 1e-3
    max_neighbours: int = 200
    inner: ProteomeEnsemble = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.inner = ProteomeEnsemble(self.networks, alpha=self.alpha,
                                      max_neighbours=self.max_neighbours)

    @property
    def predictors(self):
        return self.inner.predictors

    @property
    def leaky_arms(self) -> list[str]:
        return [n for n in self.networks if is_leaky(n)]

    @property
    def scored_arms(self) -> list[str]:
        """The arms that enter the ensemble mean."""
        return [n for n in self.networks if not is_leaky(n)]

    def fit(self, X: np.ndarray, x_cols: list[str], Y: np.ndarray,
            y_cols: list[str], verbose: bool = False) -> "PaperProteomeEnsemble":
        self.inner.fit(X, x_cols, Y, y_cols, verbose=verbose)
        return self

    def predict_each(self, X: np.ndarray, x_cols: list[str], y_cols: list[str]
                     ) -> dict[str, np.ndarray]:
        """Every arm INCLUDING the leaky one -- reporting is not averaging."""
        return self.inner.predict_each(X, x_cols, y_cols)

    def predict(self, X: np.ndarray, x_cols: list[str], y_cols: list[str]
                ) -> np.ndarray:
        """Mean over the non-leaky arms that cover each protein."""
        each = {n: p for n, p in self.predict_each(X, x_cols, y_cols).items()
                if not is_leaky(n)}
        if not each:
            return np.full((X.shape[0], len(y_cols)), np.nan)
        stack = np.stack(list(each.values()))
        with np.errstate(invalid="ignore"):
            return np.nanmean(stack, axis=0)

    def coverage(self, y_cols: list[str] | None = None) -> dict[str, int]:
        """Per-arm coverage, and the union over the arms that are averaged.

        `ProteomeEnsemble.coverage` unions everything it fitted, which would
        credit the ensemble with proteins only the leaky arm reaches. The
        per-arm counts are passed through unchanged so the leaky arm's own
        coverage is still visible.
        """
        cov = {n: len(p.models) if y_cols is None
               else len(set(p.models) & set(y_cols))
               for n, p in self.predictors.items()}
        union: set[str] = set()
        for n, p in self.predictors.items():
            if is_leaky(n):
                continue
            names = set(p.models)
            union |= names if y_cols is None else names & set(y_cols)
        cov["ENSEMBLE"] = len(union)
        return cov

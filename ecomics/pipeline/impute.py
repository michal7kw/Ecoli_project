"""Step 4 -- missingness filter and k-nearest-neighbour imputation.

Paper (paper.md:142):

    "genes that had more than 70% of their expression values missing were
     removed from the compendium. Similarly, we removed profiles where more
     than 70% of gene values were missing. Through this process, 502 genes and
     2 profiles were excluded from Ecomics. For the rest of the profiles/genes,
     we imputed their values by applying for each gene a method that is based
     on the k-nearest neighbours (k is 3 here) algorithm (R package impute)."

Neighbours are found AMONG GENES, using the profile axis as the feature space.
Co-regulation is the strongest structure in an expression matrix: a gene's
missing value in a given condition is best predicted by its regulon partners
measured in that same condition. Neighbouring PROFILES would also work, but
requires that similar conditions exist in the compendium -- and given the
sampling sparsity (only 11 of 649 conditions have two or more omics layers),
often they do not.

A caveat the paper does not state
--------------------------------
A 70% threshold is extremely permissive: a gene observed in 31% of profiles is
kept, and 69% of its row is then invented by k-NN. Those imputed values enter
both training and the held-out LOCO test folds, so a model can score well by
rediscovering the smooth low-rank structure the imputation itself created.
`missingness_filter` therefore returns per-gene observed fractions, and
`knn_impute` returns an `imputed` mask, so downstream evaluation can stratify
or mask. Ecomics ships fully imputed, so this cannot be checked retroactively
-- but it can be tracked for anything reprocessed here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["missingness_filter", "knn_impute", "FilterResult"]


@dataclass
class FilterResult:
    """Outcome of the missingness filter."""

    matrix: np.ndarray
    keep_genes: np.ndarray          # boolean mask over rows
    keep_profiles: np.ndarray       # boolean mask over columns
    gene_observed_frac: np.ndarray  # per RETAINED gene, fraction observed

    @property
    def n_dropped_genes(self) -> int:
        return int((~self.keep_genes).sum())

    @property
    def n_dropped_profiles(self) -> int:
        return int((~self.keep_profiles).sum())


def missingness_filter(mat: np.ndarray, threshold: float = 0.70,
                       verbose: bool = False) -> FilterResult:
    """Drop genes then profiles exceeding `threshold` missingness.

    Order matters and follows the paper: genes first, then profiles evaluated
    on what remains.
    """
    mat = np.asarray(mat, dtype=np.float64)
    missing = ~np.isfinite(mat)

    keep_genes = missing.mean(axis=1) <= threshold
    sub = mat[keep_genes]
    keep_profiles = (~np.isfinite(sub)).mean(axis=0) <= threshold
    out = sub[:, keep_profiles]

    observed = np.isfinite(out).mean(axis=1)
    if verbose:
        print(f"  dropped {int((~keep_genes).sum())} gene(s) and "
              f"{int((~keep_profiles).sum())} profile(s) at >{threshold:.0%} missing")
        print(f"  remaining {out.shape[0]} x {out.shape[1]}, "
              f"{(~np.isfinite(out)).mean():.2%} of cells still missing")
    return FilterResult(out, keep_genes, keep_profiles, observed)


def knn_impute(mat: np.ndarray, k: int = 3, verbose: bool = False,
               ) -> tuple[np.ndarray, np.ndarray]:
    """Impute missing values from the k most similar GENES.

    Similarity is Euclidean distance over the profiles where both genes are
    observed, normalized by the overlap count so genes with little overlap are
    not spuriously close. Returns (imputed matrix, boolean mask of what was
    imputed).

    Small k (3) preserves local co-expression structure; large k smooths toward
    the population mean and destroys exactly the condition-specific signal the
    model needs.
    """
    mat = np.asarray(mat, dtype=np.float64).copy()
    missing = ~np.isfinite(mat)
    if not missing.any():
        return mat, missing

    n_genes = mat.shape[0]
    obs = np.isfinite(mat)
    filled = np.where(obs, mat, 0.0)
    row_mean = np.divide(filled.sum(1), obs.sum(1),
                         out=np.zeros(n_genes), where=obs.sum(1) > 0)

    rows_with_gaps = np.flatnonzero(missing.any(axis=1))
    for i in rows_with_gaps:
        gaps = np.flatnonzero(missing[i])
        # Candidate neighbours are those observed at ANY of row i's gaps, and
        # each gap is then filled from the neighbours that cover THAT gap.
        #
        # This required a candidate observed at every gap simultaneously
        # (`.all(axis=1)`), which `impute::impute.knn` -- the method
        # paper.md:142 names -- does not: it averages the non-missing neighbour
        # entries per COLUMN. The difference is not cosmetic. A gene with
        # scattered gaps could find `cand.size == 0` and fall back to
        # `row_mean[i]`, an imputation carrying no neighbour information at all,
        # counted in the `imputed` mask exactly like a real one.
        cand = np.flatnonzero(obs[:, gaps].any(axis=1))
        cand = cand[cand != i]
        if cand.size == 0:
            mat[i, gaps] = row_mean[i]
            continue

        both = obs[i] & obs[cand]                    # (n_cand, n_profiles)
        n_common = both.sum(axis=1)
        diff = np.where(both, mat[cand] - mat[i], 0.0)
        d2 = (diff ** 2).sum(axis=1)
        dist = np.sqrt(np.divide(d2, n_common, out=np.full_like(d2, np.inf),
                                 where=n_common > 0))

        usable = np.isfinite(dist)
        if not usable.any():
            mat[i, gaps] = row_mean[i]
            continue
        cand, dist = cand[usable], dist[usable]
        order = np.argsort(dist)[:k]
        nb, nd = cand[order], dist[order]

        # UNWEIGHTED mean over the neighbours that actually observed each gap.
        # This was inverse-distance weighted, which impute.knn is not -- it
        # "imputes by averaging those (non-missing) elements of its neighbors".
        # Per-gap masking is the other half of matching it: a neighbour that
        # covers gap A but not gap B contributes to A only.
        block = mat[np.ix_(nb, gaps)]
        seen = obs[np.ix_(nb, gaps)]
        n_seen = seen.sum(axis=0)
        filled_block = np.where(seen, block, 0.0).sum(axis=0)
        vals = np.divide(filled_block, n_seen,
                         out=np.full(len(gaps), row_mean[i]),
                         where=n_seen > 0)
        mat[i, gaps] = vals

    if verbose:
        print(f"  imputed {int(missing.sum()):,} value(s) "
              f"({missing.mean():.2%} of the matrix) with k={k}")
    return mat, missing

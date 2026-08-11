"""Step 2 -- platform-bias correction, and the loess primitive.

Paper (paper.md:138):

    "we first performed quantile-normalization (using R package preprocessCore)
     for each platform and then we transformed the expressed genes on the same
     scale for each profile, by using loess fit between expression levels of
     before and after quantile-normalization. Then finally, we applied a
     z-score transformation for each platform data set"

Three sub-steps, in order:

  (i)   quantile-normalize within each platform
  (ii)  loess back-projection: fit post-quantile against pre-quantile values on
        the expressed genes, then apply that smooth map to every gene
  (iii) per-platform z-score

Why (ii) exists at all
----------------------
Quantile normalization is a RANK-based step function: two genes with nearly
identical raw values can be pushed apart simply because one ranked above the
other, and the mapping is a function of rank, not of intensity. Refitting the
same correction as a smooth monotone loess curve keeps the distributional
alignment while making the transform continuous in intensity -- and, crucially,
applicable to genes that were not part of the fit.

A note on (iii)
---------------
Figure 2's box 2 writes the z-score as (g_i - mu_i)/sigma_i with i in
{p_A, p_B}, over the density of a SINGLE gene g -- which reads as a per-gene,
per-platform standardization. But a per-gene z-score removes each gene's
characteristic abundance, and step 3's loess calibration needs exactly that
information to map relative levels onto absolute copy numbers (it fits ONE
monotone curve per profile, so it cannot restore per-gene offsets that have been
centred away). The reading that composes with step 3 is a per-platform
dataset-level standardization, which is the default here; `per_gene=True`
selects the literal reading, and pipeline/run.py shows what it costs.
"""

from __future__ import annotations

import numpy as np

__all__ = ["quantile_normalize", "loess_fit_predict", "platform_zscore",
           "correct_platform_bias"]


# --------------------------------------------------------------------------
# loess: tricube-weighted local linear regression
# --------------------------------------------------------------------------
def loess_fit_predict(x_train: np.ndarray, y_train: np.ndarray,
                      x_query: np.ndarray, span: float = 0.35,
                      degree: int = 1, clip: bool = True) -> np.ndarray:
    """Locally weighted polynomial regression, evaluated at `x_query`.

    span   fraction of training points in each local neighbourhood
    degree 1 (local linear) or 0 (local constant)
    clip   clamp queries to the fitted x-range before predicting. Loess is only
           defined over the span of its training data and extrapolates badly;
           the paper does not discuss this, but step 3 fits on a few hundred
           reference genes and applies the result to all 4,096, so queries do
           fall outside. Clipping degrades gracefully to the boundary fit
           instead of diverging.
    """
    x_train = np.asarray(x_train, dtype=np.float64).ravel()
    y_train = np.asarray(y_train, dtype=np.float64).ravel()
    x_query = np.asarray(x_query, dtype=np.float64).ravel()

    ok = np.isfinite(x_train) & np.isfinite(y_train)
    x_train, y_train = x_train[ok], y_train[ok]
    n = x_train.size
    if n == 0:
        return np.full(x_query.shape, np.nan)
    if n < 4:
        return np.full(x_query.shape, float(np.mean(y_train)))

    if clip:
        x_query = np.clip(x_query, x_train.min(), x_train.max())

    k = max(3, int(np.ceil(span * n)))
    order = np.argsort(x_train)
    xs, ys = x_train[order], y_train[order]

    out = np.empty(x_query.shape, dtype=np.float64)
    # The neighbourhood is the k NEAREST points, chosen by DISTANCE.
    #
    # This used to centre a k-wide window on the insertion index and clamp it
    # to `n - k`:
    #
    #     lo = max(0, min(p - k // 2, n - k));  hi = lo + k
    #
    # which is only the k nearest points when x is roughly uniform. On the
    # shape expression data actually has -- a dense low-abundance mass plus a
    # long high-abundance tail -- it is not. Measured on 200 points in [0, 1]
    # plus 20 in [8, 10], span 0.2, querying x = 0.99:
    #
    #     index window   x from 0.927 to 9.825   (width 8.90)
    #     true 44-NN     all within 0.142
    #
    # The bandwidth h = d.max() then becomes ~8.8, the tricube weights go flat
    # across every genuinely local point, and the "local" linear fit is a
    # global one. This primitive drives BOTH `correct_platform_bias` (step 2)
    # and `absolute_quantify` (step 3, the paper's most important step), so a
    # silently global smoother there is not a detail.
    #
    # Grow the window outward from the insertion point, always taking whichever
    # side is nearer. With xs sorted the k nearest to x0 are contiguous, so this
    # yields exactly them, and `lo` stays monotone in x0 -- which matters:
    # `absolute_quantify` requires the fitted calibration curve to be monotone
    # per profile, and a window that can jump between adjacent queries breaks it.
    pos = np.searchsorted(xs, x_query)
    for i, (x0, p) in enumerate(zip(x_query, pos)):
        lo = hi = int(p)
        while hi - lo < k:
            if lo == 0:
                hi += 1
            elif hi == n:
                lo -= 1
            elif (x0 - xs[lo - 1]) <= (xs[hi] - x0):
                lo -= 1
            else:
                hi += 1
        xw, yw = xs[lo:hi], ys[lo:hi]
        d = np.abs(xw - x0)
        h = d.max()
        u = d / h if h > 0 else np.zeros_like(d)
        w = (1.0 - u ** 3) ** 3                       # tricube kernel
        sw = w.sum()
        if sw <= 0:
            out[i] = yw.mean()
            continue
        mx = (w * xw).sum() / sw
        my = (w * yw).sum() / sw
        if degree == 0:
            out[i] = my
            continue
        sxx = (w * (xw - mx) ** 2).sum()
        sxy = (w * (xw - mx) * (yw - my)).sum()
        b = sxy / sxx if sxx > 1e-12 else 0.0
        out[i] = my + b * (x0 - mx)
    return out


# --------------------------------------------------------------------------
# (i) quantile normalization
# --------------------------------------------------------------------------
def quantile_normalize(mat: np.ndarray) -> np.ndarray:
    """Force every column of a (features x samples) matrix to one distribution.

    Sort each column, replace by the row-means of the sorted matrix, un-sort.
    NaNs are held out of the ranking and returned as NaN.

    TIED VALUES ARE AVERAGED, matching `preprocessCore::normalize.quantiles`
    -- the package paper.md:138 names. They were not: on [[1,4],[1,3],[3,2],
    [4,1]] the two tied 1s came out 1.0 and 1.5, where R gives both 1.25, so
    sort order alone put a rank difference between two equal measurements. Ties
    are not rare here -- RMA summarisation of low-expression probesets and
    htseq zero counts both produce large tied blocks.

    ⚠ `crosscheck_quantile` reported "exact, max diff 0.0" against R throughout,
    and could not see this: it draws lognormal random data, which has no ties.
    A cross-check is only as strong as the fixture it runs on, and a clean
    result from a fixture that cannot exercise the failure is not evidence.
    `crosscheck_quantile(with_ties=True)` now exercises it.
    """
    mat = np.asarray(mat, dtype=np.float64)
    if mat.ndim != 2:
        raise ValueError("expected a 2-D (features x samples) matrix")
    out = np.full_like(mat, np.nan)

    finite_cols = []
    for j in range(mat.shape[1]):
        col = mat[:, j]
        ok = np.isfinite(col)
        order = np.argsort(col[ok], kind="mergesort")
        finite_cols.append((ok, order, col[ok][order]))

    # Interpolate every column's sorted values onto a common grid, so columns
    # with different numbers of observed values can still be averaged.
    m = max(len(s) for _, _, s in finite_cols)
    grid = np.linspace(0, 1, m)
    stacked = np.vstack([
        np.interp(grid, np.linspace(0, 1, len(s)), s) if len(s) else np.full(m, np.nan)
        for _, _, s in finite_cols
    ])
    target = np.nanmean(stacked, axis=0)

    for j, (ok, order, sorted_vals) in enumerate(finite_cols):
        n = len(sorted_vals)
        if n == 0:
            continue
        tgt = np.interp(np.linspace(0, 1, n), grid, target)
        # TIES: give every member of a tied run the target evaluated at the run's
        # AVERAGE RANK. Without this the tie is broken by sort order -- on
        # [[1,4],[1,3],[3,2],[4,1]] the two 1s came out 1.0 and 1.5 where R gives
        # both 1.25 -- which puts a rank difference between two measurements that
        # were equal. Ties are the normal case here: RMA summarisation of
        # low-expression probesets and htseq zero counts both produce large tied
        # blocks.
        #
        # ▶ Interpolating at the average rank is NOT the same as averaging the
        # targets over the run, and the difference is measurable. Against
        # `preprocessCore::normalize.quantiles` on a tied fixture:
        #
        #     sort order (what this was)        max |diff|  1.70
        #     mean of the target over the run   max |diff|  0.092
        #     target at the average rank        max |diff|  0.00000000
        #
        # The two agree only where the target is locally linear. R does the
        # latter, so this does too -- and the check is now exact rather than
        # merely close.
        #
        # `sorted_vals` is ascending, so tied runs are contiguous.
        if n > 1:
            starts = np.flatnonzero(np.r_[True, sorted_vals[1:] != sorted_vals[:-1]])
            ends = np.r_[starts[1:], n]
            ranks = np.arange(n)
            base = tgt.copy()
            for a, b in zip(starts, ends):
                if b - a > 1:
                    tgt[a:b] = np.interp((a + b - 1) / 2.0, ranks, base)
        col = np.empty(n)
        col[order] = tgt
        out[ok, j] = col
    return out


# --------------------------------------------------------------------------
# (iii) z-score
# --------------------------------------------------------------------------
def platform_zscore(mat: np.ndarray, per_gene: bool = False) -> np.ndarray:
    """Standardize one platform's block.

    per_gene=False (default): one mean/sd for the whole platform block. Removes
        the platform's gain and offset while preserving each gene's relative
        level, which step 3 needs.
    per_gene=True: the literal reading of Fig. 2 box 2 -- centre each gene
        across that platform's profiles. Removes gene identity; step 3 can then
        no longer recover absolute levels. Provided for comparison.
    """
    mat = np.asarray(mat, dtype=np.float64)
    if per_gene:
        mu = np.nanmean(mat, axis=1, keepdims=True)
        sd = np.nanstd(mat, axis=1, keepdims=True)
    else:
        mu = np.nanmean(mat)
        sd = np.nanstd(mat)
    return (mat - mu) / (np.asarray(sd) + 1e-9)


# --------------------------------------------------------------------------
# the whole of step 2
# --------------------------------------------------------------------------
def correct_platform_bias(mat: np.ndarray, platform: np.ndarray,
                          expressed: np.ndarray | None = None,
                          span: float = 0.35, per_gene_zscore: bool = False,
                          ) -> np.ndarray:
    """Run sub-steps (i)-(iii) on a (genes x profiles) matrix.

    platform   (n_profiles,) platform label per profile
    expressed  optional boolean mask over genes, the subset the loess is fitted
               on ("the expressed genes" in the paper's wording). Defaults to
               all genes with a finite value in that profile.
    """
    mat = np.asarray(mat, dtype=np.float64)
    platform = np.asarray(platform)
    out = np.full_like(mat, np.nan)

    for p in np.unique(platform):
        cols = np.flatnonzero(platform == p)
        block = mat[:, cols]

        qn = quantile_normalize(block)                              # (i)

        lo = np.full_like(block, np.nan)                            # (ii)
        for jj, _ in enumerate(cols):
            raw, tgt = block[:, jj], qn[:, jj]
            ok = np.isfinite(raw) & np.isfinite(tgt)
            if expressed is not None:
                ok &= expressed
            if ok.sum() < 8:
                lo[:, jj] = tgt
                continue
            q = np.isfinite(block[:, jj])
            lo[q, jj] = loess_fit_predict(raw[ok], tgt[ok], block[q, jj], span=span)

        out[:, cols] = platform_zscore(lo, per_gene=per_gene_zscore)  # (iii)
    return out

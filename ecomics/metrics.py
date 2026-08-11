"""Metrics, separated from the evaluation harness that uses them.

Why this module exists
----------------------
Two reasons, both structural.

1. `moma/transcriptome.py` early-stops on validation per-gene PCC, so it needs
   the metric. Importing it from `evaluate.py` made the *model* depend on the
   *evaluation harness* -- the only cycle-shaped edge in the dependency graph.
   Both now import from here, and neither imports the other.

2. The per-gene PCC is called once per training epoch, so its cost is not an
   evaluation detail -- it is a term in the training budget. Measured on the
   transcriptome's validation split (537 x 4096), the original Python loop cost
   **0.247 s per call**, which was 71% of a 0.349 s GPU epoch and made a 6.3x
   device speedup present as 2.5x.

   Measured speedups of the vectorized form (float64, this machine):

       537 x 4096, no NaN     0.308 s -> 0.026 s   11.8x
       537 x 4096, 20% NaN    0.256 s -> 0.041 s    6.3x
       3578 x 4096, no NaN    1.722 s -> 0.250 s    6.9x

   Not the 100x one might hope for: the work is memory-bandwidth bound, not
   arithmetic bound, so the ceiling is set by how many passes are made over the
   array rather than by removing the Python interpreter. That is why the dense
   path below is separate (it makes 5 passes instead of 9) and why `einsum` is
   used for the second moments (it fuses multiply-and-reduce, allocating no
   input-sized temporary). Agreement with the loop is ~1e-15 and is pinned by
   `tests/test_metrics.py`.

Pairwise-complete correlation
-----------------------------
Every column (or row) has its own missing-value mask, which is why the naive
implementation was a loop. The vectorized form reproduces it by zero-filling
*after* centering on the masked mean, so masked entries contribute nothing to
any of the three sums:

    n_j    = sum_i  ok_ij
    mu_j   = sum_i  ok_ij * x_ij / n_j
    cov_j  = sum_i  ok_ij * (p_ij - mu^p_j)(t_ij - mu^t_j)
    r_j    = cov_j / sqrt(var^p_j * var^t_j)

Two guards are inherited verbatim from the loop, because changing either would
silently change every reported number:

    n_j < 3                     -> NaN (a correlation over two points is 1 or -1)
    population sd < 1e-12       -> NaN (a constant column has no correlation)

Note the second is the *population* sd (`ndarray.std()`, ddof=0), i.e.
`sqrt(var_j / n_j)`, not the sample sd. `tests/test_metrics.py` pins this.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import ranksums

__all__ = ["pcc_per_row", "pcc_per_column", "calibration_slope",
           "evaluate_predictions", "wilcoxon",
           "MIN_CONDITIONS_FOR_COLUMN_PCC"]

MIN_N = 3            # fewer paired observations than this -> NaN
SD_FLOOR = 1e-12     # population sd below this -> NaN (constant vector)

# Below this many CONDITIONS the per-molecule axis is not estimable, and is
# reported as "n/a (n cond)" rather than as a number. 15 is the conventional
# floor for a correlation to be worth interpreting; the proteome's 5 shared
# conditions give a 5-point correlation whose null distribution is so wide that
# ~a third of molecules exceed |r| = 0.7 by chance.
#
# It lives HERE, beside the `min_n_for_column_pcc` parameter it is passed to,
# rather than in the one script that first needed it: `scripts/04_reproduce.py`
# sets the reproduction table's suppressions and `ecomics/plots.py` has to draw
# them, and a threshold restated in two places is a threshold that will
# eventually disagree with itself.
MIN_CONDITIONS_FOR_COLUMN_PCC = 15

# Column-block size for the vectorized correlation. Bounds peak memory at
# roughly 4 * n_rows * _BLOCK * 8 bytes (~117 MB at 3,578 x 4,096) while keeping
# the inner work large enough that numpy's per-call overhead is negligible.
_BLOCK = 1024


def _pcc_block(p: np.ndarray, t: np.ndarray, axis: int) -> tuple:
    """Sums for one block. Returns (n, cov, var_p, var_t) reduced over `axis`."""
    keep = {"axis": axis, "keepdims": True}
    ok = np.isfinite(p) & np.isfinite(t)

    if ok.all():
        # Fast path. The published compendium is fully imputed (0.0000% NaN),
        # so this is what the per-epoch early-stopping call actually takes.
        # `einsum` fuses multiply-and-reduce, so the three second-moment sums
        # allocate nothing the size of the input.
        n = np.full(p.shape[1 - axis], p.shape[axis])
        pc = p - p.mean(**keep)
        tc = t - t.mean(**keep)
    else:
        n = ok.sum(axis=axis)
        # Zero-fill so masked entries drop out of every sum. Centre on the
        # MASKED mean, then re-mask: (x - mu) is nonzero at masked entries too.
        safe_n = np.where(n >= MIN_N, n, 1)
        safe_k = np.expand_dims(safe_n, axis)
        pc = np.where(ok, p, 0.0)
        tc = np.where(ok, t, 0.0)
        pc = np.where(ok, pc - pc.sum(**keep) / safe_k, 0.0)
        tc = np.where(ok, tc - tc.sum(**keep) / safe_k, 0.0)

    sub = "ij,ij->j" if axis == 0 else "ij,ij->i"
    return (n,
            np.einsum(sub, pc, tc),
            np.einsum(sub, pc, pc),
            np.einsum(sub, tc, tc))


def _pcc_along(pred: np.ndarray, true: np.ndarray, axis: int) -> np.ndarray:
    """Pairwise-complete PCC reducing over `axis`. See the module docstring.

    `axis=0` gives one value per column (per molecule); `axis=1` one per row.
    Reducing directly rather than transposing keeps both arrays C-contiguous,
    which matters: `einsum` over a transposed view is several times slower.
    """
    n_out = pred.shape[1 - axis]
    out = np.full(n_out, np.nan)

    for s0 in range(0, n_out, _BLOCK):
        sl = slice(s0, s0 + _BLOCK)
        idx = (slice(None), sl) if axis == 0 else (sl, slice(None))
        n, cov, var_p, var_t = _pcc_block(pred[idx], true[idx], axis)

        # The loop's guard is `a.std() < 1e-12` with ddof=0, i.e.
        # sqrt(var/n) < SD_FLOOR  <=>  var < n * SD_FLOOR**2.
        floor = np.maximum(n, 1) * SD_FLOOR ** 2
        good = (n >= MIN_N) & (var_p >= floor) & (var_t >= floor)
        if not good.any():
            continue

        r = cov[good] / np.sqrt(var_p[good] * var_t[good])
        out[sl][good] = np.clip(r, -1.0, 1.0)

    return out


def pcc_per_row(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """PCC per SAMPLE, across molecules. 'Did I get this profile's shape right?'

    Beware: this is dominated by the mean expression profile. Genes span orders
    of magnitude in typical abundance, so simply knowing "ribosomal genes are
    high, cryptic prophage genes are low" scores ~0.58 on Ecomics in EVERY
    condition. A trivial mean-profile predictor therefore looks strong here.
    Use `pcc_per_column` for the condition-specific question.
    """
    pred = np.asarray(pred, dtype=float)
    true = np.asarray(true, dtype=float)
    return _pcc_along(pred, true, axis=1)


def pcc_per_column(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """PCC per MOLECULE, across conditions. 'Do I track this gene's response?'

    THIS IS *NOT* THE PAPER'S AXIS -- it is this repository's primary one, and
    the difference cost most of a reproduction to find.

    The paper's axis is `pcc_per_row`. Supplementary Methods 3.3.3 ("Model
    evaluation") states it directly: PCC "between predicted expression levels
    and average of known expression levels for profiles belonging to the test
    condition" -- one value per held-out CONDITION, across genes. 3.3.4 says the
    same for the proteome.

    This docstring used to claim the opposite, inferring the axis from Fig. 5b's
    boxplot labels (`TFs (176)`, `All (4096)`) on the reasoning that those
    parentheticals count the values in each box. The inference is reasonable and
    it is wrong: the article body never states the axis, and the Supplementary
    Methods do. The two differ by ~0.3 PCC on identical predictions, which is
    why "ours 0.295 against the paper's 0.54" stood as the headline failure of
    this reproduction until the supplement was read -- on the paper's axis the
    same model scores 0.578. See `DISCREPANCIES.md` section 3.

    Why it is nonetheless primary here: a condition-blind predictor scores ~0 on
    this axis BY CONSTRUCTION, so no choice of representation can inflate a
    model's margin over its baselines. On the row axis a constant mean profile
    scores ~0.58. Both are reported everywhere; only the row axis is comparable
    to a number from the paper.
    """
    pred = np.asarray(pred, dtype=float)
    true = np.asarray(true, dtype=float)
    return _pcc_along(pred, true, axis=0)


def calibration_slope(pred: np.ndarray, true: np.ndarray) -> float:
    """Slope of true ~ pred. 1.0 is well calibrated.

    This is what PCC cannot see. The paper's Fig. 5h predicts growth over
    ~0.05-0.30 while the measured range is ~0-0.40 -- a slope well above 1,
    i.e. the predictions are squeezed toward the mean.

    Read the direction from the identity, which holds exactly:

        slope = r * sd(true) / sd(pred)

        slope > 1   predictions UNDER-dispersed (squeezed toward the mean)
        slope < 1   predictions OVER-dispersed, OR simply poorly correlated

    The second row is the trap, and it has bitten this codebase twice. A slope
    below 1 does NOT license "predictions span 1/slope times the truth's range"
    -- that inference needs r = 1. An early RNN run scored slope 0.242 while its
    predictions carried 93.6% of the truth's per-gene variance: the low slope
    came from r, not from dispersion.

    A second caveat, specific to the wide layers. This flattens both arrays, so
    on a profiles x genes matrix the dominant variance is gene-to-gene, not
    condition-to-condition -- the same mean-profile domination that inflates
    per-profile PCC. Measured on the transcriptome, the CONSTANT mean-profile
    baseline scores 0.987 (near-perfect calibration, and it has no
    condition-specific information at all) while the model scores 0.881. For a
    wide layer, read this as calibration of the overall expression scale, not
    of the condition response. It is unambiguous for the phenome, where the
    target is one scalar per condition.
    """
    p, t = np.asarray(pred, float).ravel(), np.asarray(true, float).ravel()
    ok = np.isfinite(p) & np.isfinite(t)
    if ok.sum() < MIN_N or p[ok].std() < SD_FLOOR:
        return float("nan")
    return float(np.polyfit(p[ok], t[ok], 1)[0])


def evaluate_predictions(pred: np.ndarray, true: np.ndarray,
                         min_n_for_column_pcc: int = 1,
                         n_effective: int | None = None) -> dict:
    """Both PCC axes, RMSE, and calibration slope.

    `pcc_mean` is the PER-MOLECULE (across conditions) correlation -- this
    repository's primary number, because a condition-blind predictor scores ~0
    on it by construction. `pcc_row_mean` is the per-profile correlation, and
    that is **the paper's axis** (Supplementary Methods 3.3.3); use it for any
    paper-vs-ours comparison. The gap between the two is itself diagnostic: a
    predictor with a high row score and a ~0 column score has learned the mean
    expression profile and nothing condition-specific.

    min_n_for_column_pcc
        Suppress the per-molecule axis entirely when fewer than this many
        independent observations are available to correlate across. Per-molecule
        PCC over 5 conditions is noise, not a weak result, and reporting it
        invites the reader to compare `-0.383` against the paper's `0.55` as
        though the two were the same quantity. `n_conditions_available` is always
        reported so the caller can say WHY it was suppressed. Default 1 keeps the
        historical behaviour; `scripts/04_reproduce.py` raises it.
    n_effective
        What counts as an independent observation. Defaults to the number of
        ROWS, but replicates of one condition are not independent -- 18 proteome
        profiles over 5 conditions give 5 points, not 18 -- so `run_loco` passes
        the number of distinct conditions instead.
    """
    pred = np.asarray(pred, float)
    true = np.asarray(true, float)
    n_avail = int(pred.shape[0] if n_effective is None else n_effective)
    suppressed = n_avail < min_n_for_column_pcc

    per_col = (np.full(pred.shape[1], np.nan) if suppressed
               else pcc_per_column(pred, true))
    per_row = pcc_per_row(pred, true)
    col = per_col[np.isfinite(per_col)]
    row = per_row[np.isfinite(per_row)]
    ok = np.isfinite(pred) & np.isfinite(true)
    rmse = float(np.sqrt(np.mean((pred[ok] - true[ok]) ** 2))) if ok.any() else np.nan

    def stats(v):
        return ((float(v.mean()), float(v.std()),
                 float(v.std() / np.sqrt(v.size)), int(v.size))
                if v.size else (np.nan, np.nan, np.nan, 0))

    cm, csd, csem, cn = stats(col)
    rm, rsd, rsem, rn = stats(row)
    return {
        # primary: per molecule, across conditions (the paper's axis)
        "pcc_mean": cm, "pcc_sd": csd, "pcc_sem": csem, "n": cn,
        "pcc_per_molecule": per_col,
        "frac_above_0.3": float((col > 0.3).mean()) if col.size else np.nan,
        "frac_above_0.5": float((col > 0.5).mean()) if col.size else np.nan,
        # secondary: per profile, across molecules
        "pcc_row_mean": rm, "pcc_row_sd": rsd, "n_rows": rn,
        "pcc_per_sample": per_row,
        "rmse": rmse,
        "calibration_slope": calibration_slope(pred, true),
        # provenance for the per-molecule axis
        "n_conditions_available": n_avail,
        "pcc_column_suppressed": bool(suppressed),
    }


def wilcoxon(a, b) -> float:
    """Wilcoxon rank-sum p-value, the paper's test for every comparison."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if a.size < MIN_N or b.size < MIN_N:
        return float("nan")
    return float(ranksums(a, b).pvalue)

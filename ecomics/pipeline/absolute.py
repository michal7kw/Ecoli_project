"""Step 3 -- absolute-level quantification. The most important step.

Paper (paper.md:140):

    "Due to fluctuations in the total RNA/cell, it is assumed that uniform
     expression distributions across different conditions can lead to
     inaccurate downstream analysis. To avoid this issue, we converted the
     relative expression measurements to absolute RNA copies per cell, by
     applying loess regression between the measured expression level and the
     absolute expression level for each profile. In cases where we have
     relative and absolute expression levels for some genes (the 'shared
     genes'), we trained a loess regression model, which was applied to the
     rest of the genes, and the process was repeated for all profiles."

Why this exists
---------------
Standard practice compares profiles as fold-changes, or aligns their
distributions. Both assume total RNA per cell is constant across conditions. It
is not: RNA/cell scales with growth rate (Schaechter 1958; Klumpp, Zhang & Hwa
2009), and Ecomics' conditions differ in exactly the factors -- medium richness,
stress, knockouts -- that set growth rate. So the assumption fails precisely
where the compendium is most interesting.

The fix is calibration, not normalization: anchor each profile onto external
measurements of absolute copy number (Taniguchi et al. 2010 for transcripts,
Lu et al. 2007 / APEX for proteins) using the genes present in both, and apply
the fitted curve to the rest.

Two properties matter:
  * PER PROFILE. Each profile gets its own curve, which is what allows two
    profiles to end up with genuinely different total RNA content -- the very
    thing quantile normalization destroys.
  * NON-PARAMETRIC. The intensity/abundance map saturates at the top and floors
    at the bottom; loess recovers that shape without being told it.

The alternative the paper rejected: "Focusing only on housekeeping genes as a
reference produced inferior results" -- unsurprising, since ribosomal and
translation genes are the MOST growth-dependent things in the cell, and a
handful of genes gives a single scale factor rather than a curve.
"""

from __future__ import annotations

import numpy as np

from ecomics.pipeline.platform import loess_fit_predict

__all__ = ["fit_absolute_loess", "absolute_quantify", "AbsoluteCalibration"]


class AbsoluteCalibration:
    """A fitted relative -> absolute map for one profile."""

    __slots__ = ("x_ref", "y_ref", "span", "n_shared", "x_lo", "x_hi")

    def __init__(self, x_ref: np.ndarray, y_ref: np.ndarray, span: float):
        self.x_ref = np.asarray(x_ref, dtype=np.float64)
        self.y_ref = np.asarray(y_ref, dtype=np.float64)
        self.span = span
        self.n_shared = self.x_ref.size
        self.x_lo = float(self.x_ref.min()) if self.n_shared else np.nan
        self.x_hi = float(self.x_ref.max()) if self.n_shared else np.nan

    def predict(self, x: np.ndarray) -> np.ndarray:
        return loess_fit_predict(self.x_ref, self.y_ref, x, span=self.span, clip=True)

    def extrapolated(self, x: np.ndarray) -> np.ndarray:
        """Mask of queries outside the reference range (clipped, not fitted)."""
        x = np.asarray(x, dtype=np.float64)
        return (x < self.x_lo) | (x > self.x_hi)


def fit_absolute_loess(relative: np.ndarray, absolute: np.ndarray,
                       shared: np.ndarray | None = None,
                       span: float = 0.5, log_absolute: bool = True,
                       ) -> AbsoluteCalibration:
    """Fit one profile's calibration from the shared genes.

    relative  (n_genes,) the profile after platform-bias correction
    absolute  (n_genes,) reference copy numbers; NaN where unknown
    shared    optional boolean mask; defaults to genes finite in both
    log_absolute  fit against log10(absolute). Copy numbers span ~5 orders of
                  magnitude, so a linear-scale fit is dominated by the handful
                  of ribosomal transcripts.
    """
    relative = np.asarray(relative, dtype=np.float64).ravel()
    absolute = np.asarray(absolute, dtype=np.float64).ravel()
    ok = np.isfinite(relative) & np.isfinite(absolute)
    if shared is not None:
        ok &= np.asarray(shared, dtype=bool)
    if ok.sum() < 8:
        raise ValueError(f"only {ok.sum()} shared genes; need at least 8")

    y = absolute[ok]
    if log_absolute:
        y = np.log10(np.maximum(y, 1e-3))
    return AbsoluteCalibration(relative[ok], y, span)


def absolute_quantify(mat: np.ndarray, reference: np.ndarray,
                      shared: np.ndarray | None = None, span: float = 0.5,
                      log_absolute: bool = True, return_log: bool = False,
                      verbose: bool = False,
                      ) -> tuple[np.ndarray, list[AbsoluteCalibration | None]]:
    """Calibrate every profile of a (genes x profiles) matrix onto absolute scale.

    reference  (n_genes,) absolute copy numbers, NaN where unknown. This is the
               role Taniguchi et al. 2010 plays in the paper.
    return_log if True return log10(copies/cell), else copies/cell.
    """
    mat = np.asarray(mat, dtype=np.float64)
    out = np.full_like(mat, np.nan)
    cals: list[AbsoluteCalibration | None] = []
    n_extrap = 0

    for j in range(mat.shape[1]):
        col = mat[:, j]
        try:
            cal = fit_absolute_loess(col, reference, shared, span, log_absolute)
        except ValueError:
            out[:, j] = np.nan
            cals.append(None)
            continue
        q = np.isfinite(col)
        pred = cal.predict(col[q])
        n_extrap += int(cal.extrapolated(col[q]).sum())
        out[q, j] = pred
        cals.append(cal)

    if log_absolute and not return_log:
        out = np.power(10.0, out)

    if verbose:
        good = [c for c in cals if c is not None]
        print(f"  calibrated {len(good)}/{mat.shape[1]} profiles "
              f"(mean {np.mean([c.n_shared for c in good]):.0f} shared genes)")
        print(f"  {n_extrap:,} query value(s) fell outside the reference range "
              f"and were clipped to the boundary fit")
    return out, cals


def synthetic_reference(mat: np.ndarray, n_shared: int = 300,
                        seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Build a stand-in absolute reference, for demos and tests.

    The genuine article is Taniguchi et al. 2010, which is not redistributable
    here. This produces a plausible log-normal copy-number distribution
    correlated with mean observed level, so the pipeline can be exercised
    end-to-end. Anything computed from it is a DEMONSTRATION, not a
    reproduction of Ecomics' absolute scale.
    """
    rng = np.random.default_rng(seed)
    mean_level = np.nanmean(mat, axis=1)
    rank = np.argsort(np.argsort(mean_level)) / max(len(mean_level) - 1, 1)
    log_copies = 0.3 + 3.2 * rank + rng.normal(0, 0.25, len(rank))
    absolute = np.power(10.0, log_copies)
    idx = rng.choice(len(absolute), size=min(n_shared, len(absolute)),
                     replace=False)
    shared = np.zeros(len(absolute), dtype=bool)
    shared[idx] = True
    ref = np.full(len(absolute), np.nan)
    ref[shared] = absolute[shared]
    return ref, shared

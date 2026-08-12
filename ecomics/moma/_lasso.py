"""One LASSO fit, shared by the proteome, metabolome and phenome modules.

Why this exists
---------------
Three modules were fitting `Lasso`/`LassoCV` with the same two defects, and
fixing them in three places invites them diverging again.

1. **Nothing was standardized.** scikit-learn's `Lasso` does not centre or scale
   `X`; glmnet -- the tool the paper used (Supplementary Methods §3.3, ref 123)
   -- standardizes by default. An L1 penalty is a single budget shared across
   coefficients, so on unscaled predictors it is spent almost entirely on
   whichever features happen to carry the largest units. That is not a
   regularizer, it is a unit-dependent feature filter.

   The symptom was loud and had been running for a while: **415
   `ConvergenceWarning`s** in `results/reproduce.log` and **2,794** in
   `results/five_layer_phenome.log`. Coordinate descent on badly-conditioned
   columns hits `max_iter` before the duality gap closes, so the stored
   coefficients are not the LASSO solution -- they are wherever the optimizer
   ran out of iterations.

   This repository's own course says so, at `docs/lasso/08-practice-and-pitfalls.md`
   §"the checklist" ("Are the columns standardized, or already on a genuinely
   common scale?") and §2 ("`StandardScaler` inside the pipeline, not before").
   Inside the pipeline matters: fitted on the training split only, so the test
   fold's scale never leaks into the fit.

2. **A bare `except Exception: continue`.** A target whose fit raised became
   indistinguishable from one with no neighbours in the network -- and coverage
   is a headline number for the proteome ("no single network covers every
   protein, the union does"). `phenome.LayerGrowthPredictor` already got this
   right and explains why; the other two did not. `fit_lasso` returns the
   exception instead of swallowing it, so callers can count failures apart from
   absences.

Reading coefficients
--------------------
Because the estimator is now a `Pipeline`, `model.coef_` no longer exists.
`coefficients()` reaches the final step. Note they are on the STANDARDIZED
scale, which is the right scale for ranking predictive importance (it is what
makes two features comparable) and the wrong one for reading off an effect per
original unit.
"""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso, LassoCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

__all__ = ["fit_lasso", "coefficients", "MAX_ITER"]

# Raised from 5,000/8,000 alongside standardization. Standardizing removes most
# of the conditioning problem; the higher ceiling covers the rest, and is cheap
# because a converged fit exits early and never reaches it.
MAX_ITER = 50_000


def fit_lasso(X: np.ndarray, y: np.ndarray, alpha: float | None = None,
              cv_max: int = 3, min_for_cv: int = 5, random_state: int = 0):
    """Fit a standardized LASSO. Returns (model, error) -- exactly one is None.

    alpha       None selects it by cross-validation (`LassoCV`) when there are
                at least `min_for_cv` samples, otherwise falls back to a fixed
                small penalty. These layers run on 5-49 conditions, so CV is
                often not available.
    cv_max      upper bound on CV folds; the real bound is the sample count.

    Convergence failures are promoted to errors rather than warnings. A LASSO
    that hit `max_iter` has not solved its objective, and silently reporting its
    coefficients is how 2,794 warnings went unread.
    """
    n = int(np.asarray(y).size)
    if alpha is None and n >= min_for_cv:
        inner = LassoCV(cv=min(cv_max, n), max_iter=MAX_ITER,
                        random_state=random_state)
    else:
        inner = Lasso(alpha=alpha if alpha is not None else 1e-3,
                      max_iter=MAX_ITER)
    model = make_pipeline(StandardScaler(), inner)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            model.fit(X, y)
    except Exception as e:                             # noqa: BLE001
        return None, f"{type(e).__name__}: {e}"
    return model, None


def coefficients(model) -> np.ndarray:
    """Coefficients of a model returned by `fit_lasso`, on the scaled basis."""
    if model is None:
        return np.zeros(0)
    return np.asarray(model[-1].coef_ if hasattr(model, "steps")
                      else model.coef_)

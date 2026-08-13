"""Step 1b -- noise removal by an anchored two-component Gaussian mixture.

Paper (paper.md:136):

    "Noise in gene expression was estimated from three distinct sources,
     phantom genes, negative-control probes designed by array manufacturers and
     non-K12 strain genes. Means and variances of both the signal and the noise
     were estimated using the Expectation Maximization (EM) algorithm
     implemented in R mclust, with the intensity below mean intensity of the
     three noise sources considered as noise during initialization."

Model, per profile:

    p(o) = pi_N N(o | mu_N, s_N^2) + pi_S N(o | mu_S, s_S^2)

fitted by EM, then S_i = O_i - E[noise contribution | O_i].

The anchoring is the point
--------------------------
EM converges to a LOCAL optimum of a likelihood that is invariant under
relabelling the two components, so "which component is noise" is otherwise
arbitrary. Seeding the low component from features that are noise BY
CONSTRUCTION -- annotated non-genes, manufacturer negative controls, genes
absent from the K-12 genome -- pins it to the physically correct location.
That is the "semi-supervised" ingredient the abstract refers to.

Subtraction is SOFT: we remove P(noise | observed) * mu_noise rather than
hard-classifying genes as expressed/not, because downstream regression needs
continuous values.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["GaussianMixtureNoise", "remove_noise"]

_SQRT_2PI = np.sqrt(2.0 * np.pi)


def _normal_pdf(x: np.ndarray, mu: float, sd: float) -> np.ndarray:
    sd = max(float(sd), 1e-9)
    z = (x - mu) / sd
    return np.exp(-0.5 * z * z) / (sd * _SQRT_2PI)


@dataclass
class GaussianMixtureNoise:
    """A fitted 2-component mixture: component 0 is noise, 1 is signal."""

    mu_noise: float
    sd_noise: float
    pi_noise: float
    mu_signal: float
    sd_signal: float
    pi_signal: float
    n_iter: int
    loglik: float
    converged: bool

    def responsibility(self, x: np.ndarray) -> np.ndarray:
        """P(noise | observed) for each value."""
        x = np.asarray(x, dtype=np.float64)
        a = self.pi_noise * _normal_pdf(x, self.mu_noise, self.sd_noise)
        b = self.pi_signal * _normal_pdf(x, self.mu_signal, self.sd_signal)
        tot = a + b
        out = np.divide(a, tot, out=np.full_like(a, 0.5), where=tot > 0)
        return out

    def denoise(self, x: np.ndarray, floor: float | None = 0.0) -> np.ndarray:
        """S = O - P(noise|O) * mu_noise.

        floor: clamp the result at this value, or None to allow negatives.
        Subtraction CAN drive low-expression genes below zero; the paper does
        not say how it handles that, so the behaviour is explicit here and the
        count is reported by `remove_noise`.
        """
        x = np.asarray(x, dtype=np.float64)
        out = x - self.responsibility(x) * self.mu_noise
        return out if floor is None else np.maximum(out, floor)


def fit_mixture(values: np.ndarray, anchor_idx: np.ndarray | None = None,
                max_iter: int = 200, tol: float = 1e-6,
                ) -> GaussianMixtureNoise:
    """Fit the 2-component mixture to one profile by EM.

    anchor_idx: indices of features known a priori to be noise. Their mean
    seeds the noise component; everything above it seeds the signal component.
    Without anchoring we fall back to a median split, which is what makes the
    result arbitrary -- see the module docstring.
    """
    x = np.asarray(values, dtype=np.float64).ravel()
    x = x[np.isfinite(x)]
    if x.size < 10:
        raise ValueError("need at least 10 finite values to fit a mixture")

    if anchor_idx is not None and len(anchor_idx) > 0:
        anchor_vals = np.asarray(values, dtype=np.float64).ravel()[anchor_idx]
        anchor_vals = anchor_vals[np.isfinite(anchor_vals)]
        lo = float(anchor_vals.mean()) if anchor_vals.size else float(np.median(x))
    else:
        lo = float(np.median(x))
    hi_vals = x[x > lo]
    hi = float(hi_vals.mean()) if hi_vals.size else float(x.mean() + x.std())

    mu = np.array([lo, hi], dtype=np.float64)
    sd = np.array([max(x.std() / 2, 1e-6)] * 2, dtype=np.float64)
    pi = np.array([0.5, 0.5], dtype=np.float64)

    prev_ll, converged, it = -np.inf, False, 0
    for it in range(1, max_iter + 1):
        # E-step: responsibilities
        dens = np.vstack([pi[k] * _normal_pdf(x, mu[k], sd[k]) for k in range(2)])
        tot = dens.sum(axis=0)
        tot = np.where(tot > 0, tot, 1e-300)
        gamma = dens / tot

        ll = float(np.log(tot).sum())
        if np.abs(ll - prev_ll) < tol * max(1.0, abs(prev_ll)):
            converged = True
            prev_ll = ll
            break
        prev_ll = ll

        # M-step: weighted MLE
        nk = gamma.sum(axis=1)
        nk = np.where(nk > 1e-12, nk, 1e-12)
        pi = nk / x.size
        mu = (gamma * x).sum(axis=1) / nk
        sd = np.sqrt(((gamma * (x[None, :] - mu[:, None]) ** 2).sum(axis=1) / nk))
        sd = np.maximum(sd, 1e-6)

    noise_k = int(np.argmin(mu))          # noise is the LOW-intensity component
    sig_k = 1 - noise_k
    return GaussianMixtureNoise(
        mu_noise=float(mu[noise_k]), sd_noise=float(sd[noise_k]),
        pi_noise=float(pi[noise_k]),
        mu_signal=float(mu[sig_k]), sd_signal=float(sd[sig_k]),
        pi_signal=float(pi[sig_k]),
        n_iter=it, loglik=float(prev_ll), converged=converged,
    )


def remove_noise(mat: np.ndarray, anchor_idx: np.ndarray | None = None,
                 floor: float | None = 0.0, verbose: bool = False,
                 ) -> tuple[np.ndarray, list[GaussianMixtureNoise]]:
    """Fit and subtract noise PER PROFILE across a (genes x profiles) matrix.

    Per profile, not globally: each array or sequencing run has its own noise
    floor set by its own optical background, hybridization stringency and
    depth. One global mixture would over-subtract from clean profiles and
    under-subtract from dirty ones.
    """
    mat = np.asarray(mat, dtype=np.float64)
    out = np.full_like(mat, np.nan)
    fits: list[GaussianMixtureNoise] = []
    n_negative = 0

    for j in range(mat.shape[1]):
        col = mat[:, j]
        try:
            fit = fit_mixture(col, anchor_idx)
        except ValueError:
            out[:, j] = col
            fits.append(None)  # type: ignore[arg-type]
            continue
        raw = fit.denoise(col, floor=None)
        n_negative += int(np.nansum(raw < 0))
        out[:, j] = raw if floor is None else np.maximum(raw, floor)
        fits.append(fit)

    if verbose:
        good = [f for f in fits if f is not None]
        conv = sum(f.converged for f in good)
        print(f"  fitted {len(good)}/{mat.shape[1]} profiles "
              f"({conv} converged); mean mu_noise="
              f"{np.mean([f.mu_noise for f in good]):.4f}, "
              f"mean pi_noise={np.mean([f.pi_noise for f in good]):.3f}")
        print(f"  {n_negative:,} value(s) went negative before flooring")
    return out, fits

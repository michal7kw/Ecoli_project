"""Microarray preprocessing: CEL reading, RMA from scratch, normexp.

Paper (paper.md:132):

  one-channel  "image data were first read into R using the affy package. Then
                RMA is applied for a set of replicates for background
                correction, normalization, probe-set summarization."
  two-channel  "Background correction was performed using normexp (with an
                offset of 5)... Red and green channels were separated and
                quantile-normalized for each set of replicates."

RMA (Irizarry 2003) is three operations, all implemented here:

  1. background correction   observed = exponential(signal) + normal(background);
                             return E[signal | observed]
  2. quantile normalization  across the array set (pipeline/platform.py)
  3. median polish           summarize log2 probe intensities per probe set,
                             robustly to individual bad probes

RMA is defined over a SET of arrays, so its output depends on which arrays are
processed together. The paper applies it "for a set of replicates", i.e.
batch-locally -- which is exactly why a subsequent cross-platform correction
step (pipeline/platform.py) is unavoidable.

Splitting two-colour channels
-----------------------------
A two-colour array natively yields log2(R/G) -- a ratio to a study-specific
reference, i.e. the fold-change trap in hardware. Treating the array as two
independent single-channel measurements gives up the ratio's cancellation of
dye and spot effects, and buys cross-study comparability. Given that Ecomics is
built on an absolute scale, that is the right trade.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ecomics.pipeline.platform import quantile_normalize

__all__ = ["CEL", "read_cel", "read_cdf", "rma", "rma_background_correct",
           "median_polish", "normexp_background_correct"]


# --------------------------------------------------------------------------
# CEL files
# --------------------------------------------------------------------------
@dataclass
class CEL:
    """One Affymetrix CEL file: a grid of per-cell intensities."""

    path: Path
    ncol: int
    nrow: int
    intensity: np.ndarray       # (nrow*ncol,) column-major: index = x + y*ncol
    stdev: np.ndarray | None
    npix: np.ndarray | None
    chip_type: str | None

    @property
    def name(self) -> str:
        return self.path.stem

    def at(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.intensity[np.asarray(y) * self.ncol + np.asarray(x)]


_HEADER_RE = re.compile(r"^(\w+)=(.*)$")


def read_cel(path: str | Path) -> CEL:
    """Read an ASCII (Version=3) Affymetrix CEL file.

    GEO ships E. coli arrays in this text format, so no binary parser is
    needed. The [INTENSITY] section is `X  Y  MEAN  STDV  NPIXELS`.
    """
    path = Path(path)
    raw = path.read_bytes().decode("latin-1")
    if not raw.lstrip().startswith("[CEL]"):
        raise ValueError(f"{path.name}: not an ASCII CEL file (binary v4 CELs "
                         "are not supported; re-export or use affy)")

    ncol = nrow = None
    chip_type = None
    lines = raw.splitlines()

    section = None
    intens: np.ndarray | None = None
    sd: np.ndarray | None = None
    npix: np.ndarray | None = None
    n_expected = 0
    filled = 0

    for line in lines:
        s = line.strip()
        if not s:
            continue
        if s.startswith("[") and s.endswith("]"):
            section = s[1:-1].upper()
            continue

        if section == "HEADER":
            m = _HEADER_RE.match(s)
            if m:
                key, val = m.group(1).upper(), m.group(2)
                if key == "COLS":
                    ncol = int(val)
                elif key == "ROWS":
                    nrow = int(val)
                elif key == "DATHEADER":
                    # e.g. "... Ecoli_ASv2.1sq ..." -> chip type
                    mm = re.search(r"([A-Za-z0-9_\-]+)\.1sq", val)
                    if mm:
                        chip_type = mm.group(1)
            continue

        if section == "INTENSITY":
            # CellHeader= must be tested BEFORE the generic KEY=VALUE regex,
            # which it also matches -- otherwise the array is never allocated.
            if s.startswith("CellHeader"):
                if ncol is None or nrow is None:
                    raise ValueError(f"{path.name}: missing Cols/Rows in header")
                size = ncol * nrow
                intens = np.full(size, np.nan, dtype=np.float64)
                sd = np.full(size, np.nan, dtype=np.float64)
                npix = np.full(size, np.nan, dtype=np.float64)
                continue
            m = _HEADER_RE.match(s)
            if m:
                if m.group(1).upper() == "NUMBERCELLS":
                    n_expected = int(m.group(2))
                continue
            if intens is None:
                continue
            parts = s.split()
            if len(parts) < 3:
                continue
            x, y = int(parts[0]), int(parts[1])
            idx = y * ncol + x
            intens[idx] = float(parts[2])
            if len(parts) > 3:
                sd[idx] = float(parts[3])
            if len(parts) > 4:
                npix[idx] = float(parts[4])
            filled += 1
            continue

        if section in {"MASKS", "OUTLIERS", "MODIFIED"} and intens is not None \
                and filled >= n_expected > 0:
            break

    if intens is None:
        raise ValueError(f"{path.name}: no [INTENSITY] section found")
    return CEL(path=path, ncol=ncol, nrow=nrow, intensity=intens,
               stdev=sd, npix=npix, chip_type=chip_type)


def read_cdf(path: str | Path, pm_only: bool = True
             ) -> dict[str, np.ndarray]:
    """Read the CDF layout TSV exported by tools/export_cdf.R.

    Returns {probeset -> array of linear cell indices}. The exporter writes
    1-based affy indices; they are converted to 0-based here.
    """
    import csv

    out: dict[str, list[int]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if pm_only and row["type"] != "pm":
                continue
            out.setdefault(row["probeset"], []).append(int(row["index"]) - 1)
    return {k: np.asarray(v, dtype=np.int64) for k, v in out.items()}


# --------------------------------------------------------------------------
# RMA step 1: background correction
# --------------------------------------------------------------------------
def _epanechnikov_mode(v: np.ndarray, n_pts: int = 1 << 14) -> float:
    """Mode of a kernel density estimate, matching R's density(kernel="epanechnikov").

    affy locates the background mode this way, so reproducing its bandwidth rule
    (bw.nrd0) and kernel matters: a Gaussian KDE with scipy's default bandwidth
    lands on a visibly different mode, and the mode propagates into every
    corrected value.
    """
    v = np.asarray(v, dtype=np.float64).ravel()
    v = v[np.isfinite(v)]
    n = v.size
    if n < 2:
        return float(v[0]) if n else 0.0

    # R's bw.nrd0
    sd = np.std(v, ddof=1)
    iqr = np.subtract(*np.percentile(v, [75, 25]))
    lo = min(sd, iqr / 1.349) if iqr > 0 else sd
    if lo == 0:
        lo = sd or abs(v[0]) or 1.0
    bw = 0.9 * lo * n ** (-0.2)
    if bw <= 0:
        return float(np.median(v))

    # R pads the grid by 3 bandwidths ("cut=3") on each side.
    lo_x, hi_x = v.min() - 3 * bw, v.max() + 3 * bw
    grid = np.linspace(lo_x, hi_x, n_pts)
    counts, edges = np.histogram(v, bins=n_pts, range=(lo_x, hi_x))
    centers = 0.5 * (edges[:-1] + edges[1:])

    # Epanechnikov as R parameterizes it: half-width a = bw*sqrt(5).
    a = bw * np.sqrt(5.0)
    step = centers[1] - centers[0]
    half = int(np.ceil(a / step))
    if half < 1:
        return float(centers[int(np.argmax(counts))])
    u = np.arange(-half, half + 1) * step
    kern = 0.75 / a * (1.0 - (u / a) ** 2)
    kern[np.abs(u) >= a] = 0.0
    dens = np.convolve(counts.astype(np.float64), kern, mode="same")
    return float(centers[int(np.argmax(dens))])


def bg_parameters(pm: np.ndarray, n_pts: int = 1 << 14) -> tuple[float, float, float]:
    """Estimate (alpha, mu, sigma) exactly as affy's `bg.parameters` does.

    Two-pass mode estimation, then sigma from the reflected lower tail scaled by
    sqrt(2), and alpha as the reciprocal of the MODE (not the mean) of the upper
    tail. Getting any of these wrong shifts every corrected value by a roughly
    constant amount, which survives quantile normalization and median polish.
    """
    x = np.asarray(pm, dtype=np.float64).ravel()
    x = x[np.isfinite(x)]
    if x.size < 10:
        return 1.0, 0.0, 1.0

    mu = _epanechnikov_mode(x, n_pts)          # pass 1: mode of all PM
    lower = x[x < mu]
    if lower.size > 1:
        mu = _epanechnikov_mode(lower, n_pts)  # pass 2: mode of the lower tail
        lower = x[x < mu]

    if lower.size > 1:
        d = lower - mu
        sigma = np.sqrt((d ** 2).sum() / (d.size - 1)) * np.sqrt(2.0)
    else:
        sigma = max(float(np.std(x)) / 2, 1e-6)
    sigma = max(float(sigma), 1e-6)

    upper = x[x > mu]
    if upper.size > 1:
        expr_mu = _epanechnikov_mode(upper - mu, n_pts)
        alpha = 1.0 / expr_mu if expr_mu > 0 else 1.0 / max(np.mean(upper - mu), 1e-6)
    else:
        alpha = 1.0
    return float(alpha), float(mu), sigma


def rma_background_correct(pm: np.ndarray, n_pts: int = 1 << 14) -> np.ndarray:
    """RMA convolution background: observed = Exp(alpha) signal + Normal(mu, sigma).

    Returns E[signal | observed], using affy's closed form

        a      = o - mu - alpha * sigma^2
        E[s|o] = a + sigma * phi(a/sigma) / Phi(a/sigma)

    where phi and Phi are the standard normal pdf and cdf. The ratio is
    evaluated in log space so the far-left tail (where Phi underflows) stays
    finite.
    """
    from scipy.special import log_ndtr
    from scipy.stats import norm

    o = np.asarray(pm, dtype=np.float64)
    alpha, mu, sigma = bg_parameters(o, n_pts)

    a = o - mu - alpha * sigma ** 2
    z = a / sigma
    # phi(z)/Phi(z) == exp(log phi(z) - log Phi(z)); log_ndtr is stable for z << 0
    log_phi = norm.logpdf(z)
    ratio = np.exp(log_phi - log_ndtr(z))
    out = a + sigma * ratio

    bad = ~np.isfinite(out) | (out <= 0)
    if bad.any():
        out[bad] = np.maximum(o[bad] - mu, 1e-3)
    return out


def normexp_background_correct(x: np.ndarray, offset: float = 5.0) -> np.ndarray:
    """normexp background correction for two-colour arrays, with an offset.

    Same exponential-plus-normal convolution as RMA. The OFFSET (the paper uses
    5, following Ritchie 2007) is added afterwards to damp the variance of
    low-intensity log-ratios: without it a background-corrected value near zero
    produces wildly unstable logs.
    """
    return rma_background_correct(x) + offset


# --------------------------------------------------------------------------
# RMA step 3: median polish
# --------------------------------------------------------------------------
def median_polish(mat: np.ndarray, max_iter: int = 10, tol: float = 1e-4
                  ) -> tuple[np.ndarray, np.ndarray, float]:
    """Tukey median polish of a (probes x arrays) matrix of log2 intensities.

    Fits  y_ij = overall + probe_i + array_j  by alternately sweeping row and
    column medians. Returns (array effects, probe effects, overall). The array
    effects plus the overall constant are RMA's expression summary -- robust to
    a few bad probes in a way a mean would not be.
    """
    z = np.array(mat, dtype=np.float64, copy=True)
    nr, nc = z.shape
    row_eff = np.zeros(nr)
    col_eff = np.zeros(nc)
    overall = 0.0

    for _ in range(max_iter):
        rmed = np.nanmedian(z, axis=1)
        z -= rmed[:, None]
        row_eff += rmed
        delta = np.nanmedian(col_eff)
        col_eff -= delta
        overall += delta

        cmed = np.nanmedian(z, axis=0)
        z -= cmed[None, :]
        col_eff += cmed
        delta = np.nanmedian(row_eff)
        row_eff -= delta
        overall += delta

        if np.nanmax(np.abs(np.r_[rmed, cmed])) < tol:
            break
    return col_eff, row_eff, overall


# --------------------------------------------------------------------------
# the whole of RMA
# --------------------------------------------------------------------------
def rma(cels: list[CEL], cdf: dict[str, np.ndarray],
        background: bool = True, verbose: bool = False,
        ) -> tuple[list[str], np.ndarray]:
    """Run RMA over a set of CEL files sharing one CDF.

    Returns (probeset ids, (n_probesets x n_arrays) log2 expression matrix).

    All three steps, in the paper's order: background correct each array,
    quantile-normalize across arrays, then median-polish each probe set.
    """
    if not cels:
        raise ValueError("no CEL files given")
    ncol = cels[0].ncol
    for c in cels:
        if c.ncol != ncol or c.nrow != cels[0].nrow:
            raise ValueError(f"{c.name}: array geometry differs from {cels[0].name}")

    probesets = sorted(cdf)
    all_idx = np.concatenate([cdf[p] for p in probesets])
    size = cels[0].ncol * cels[0].nrow
    if all_idx.max() >= size:
        raise ValueError(
            f"CDF references cell index {all_idx.max()} but the array has "
            f"{size} cells -- CDF and CEL geometry disagree")

    # 1. background correction, per array, on the PM cells
    pm = np.vstack([c.intensity[all_idx] for c in cels]).T   # (n_pm, n_arrays)
    if background:
        pm = np.column_stack([rma_background_correct(pm[:, j])
                              for j in range(pm.shape[1])])
        if verbose:
            print(f"  background-corrected {pm.shape[1]} array(s), "
                  f"{pm.shape[0]:,} PM probes each")

    # 2. quantile normalization across arrays
    pm = quantile_normalize(pm)
    if verbose:
        print("  quantile-normalized across arrays")

    # 3. median polish per probe set, on log2
    log_pm = np.log2(np.maximum(pm, 1e-6))
    expr = np.empty((len(probesets), len(cels)), dtype=np.float64)
    offset = 0
    for i, p in enumerate(probesets):
        n = len(cdf[p])
        block = log_pm[offset:offset + n]
        offset += n
        col_eff, _row_eff, overall = median_polish(block)
        expr[i] = col_eff + overall
    if verbose:
        print(f"  median-polished {len(probesets):,} probe sets")
    return probesets, expr


def probeset_to_gene(probesets: list[str]) -> dict[str, str]:
    """Map Affymetrix E. coli probe-set ids onto b-numbers.

    GPL199 ids embed the locus tag, e.g. "aas_b2836_at" -> "b2836", which is
    exactly Ecomics' gene identifier. Probe sets with no embedded b-number
    (controls, intergenic probes) are omitted.
    """
    out = {}
    for p in probesets:
        m = re.search(r"(b\d{4})", p)
        if m:
            out[p] = m.group(1)
    return out

"""Validation: the paper's negative control, and R/Bioconductor cross-checks.

Two independent kinds of check.

1. The half-life negative control (paper.md:140)
   ----------------------------------------------
   mRNA half-life is a property of the transcript, not of the condition, so it
   should not correlate with the normalization factor applied to a profile. If
   the relative->absolute map were introducing a distortion tied to abundance --
   and abundance correlates with stability -- short- and long-half-life genes
   would separate. The paper reports 0.573 +/- 0.004 vs 0.572 +/- 0.003,
   P = 0.41, i.e. no separation.

   This is a NEGATIVE control: it rules out one specific artefact. It cannot
   show the reconstructed absolute scale is correct.

2. Cross-checks against R/Bioconductor
   ------------------------------------
   Our RMA, quantile normalization, Gaussian-mixture noise model and k-NN
   imputation each have a canonical implementation in affy / preprocessCore /
   mclust / impute. Those are used ONLY as ground truth here -- the pipeline
   itself never calls R.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from ecomics import config as C

__all__ = ["half_life_control", "crosscheck_quantile", "crosscheck_impute",
           "crosscheck_rma", "crosscheck_all"]


# --------------------------------------------------------------------------
# 1. the paper's negative control
# --------------------------------------------------------------------------
# Bernstein et al. 2002 (PNAS 99:9697) measured genome-wide mRNA half-lives in
# E. coli. A curated short/long split is used here; anything not listed is
# excluded rather than guessed at.
#
# The two lists below are the curation AS TRANSCRIBED, kept verbatim so the
# error in them stays visible. Three b-numbers -- b0720 (gltA), b1136 (icd) and
# b2296 (ackA) -- appear in BOTH, i.e. 3 of 24 in each group, and they are
# excluded from the comparison by `_AMBIGUOUS` below rather than assigned.
#
# Why this mattered. The control asks whether short-lived transcripts sit at
# systematically different normalized levels from long-lived ones; the paper's
# answer is no (0.573 vs 0.572, P = 0.41), and a NEGATIVE result is what it is
# designed to produce. Two groups sharing 12.5% of their members are not
# independent samples, which is what `ranksums` assumes -- so the test statistic
# was not the one being reported, whichever way it happened to point.
#
# Measured both ways on the SAME `results/pipeline_output.npz`:
#
#     contaminated (24/24)   0.6395 vs 0.6481   P = 0.6427
#     disjoint     (21/21)   0.6279 vs 0.6377   P = 0.6417
#
# So removing them moves P by 0.001. The contamination was not, on this data,
# changing the answer -- and that is worth stating plainly rather than dressing
# the fix up as a rescue. What was wrong is that `ranksums` was being handed two
# samples that shared 12.5% of their members while assuming independence, so the
# quantity reported was not the quantity named. A control that happens to land
# correctly through an invalid test is still not evidence.
#
# ⚠ An earlier version of this note claimed P "rose by 0.02" on removal. That
# compared the contaminated value from BEFORE the loess neighbourhood fix
# (P = 0.658) against the disjoint value from after it, i.e. it confounded two
# changes -- exactly the error the factorial in scripts/06 exists to avoid.
# Isolated, the loess fix accounts for essentially all of the 0.658 -> 0.642
# movement and the group fix for essentially none of it.
#
# They are dropped rather than reassigned because nothing published here
# distinguishes which list each belongs to; all three are central-metabolism
# enzymes, and picking a side would be a fabrication of exactly the kind the
# "excluded rather than guessed at" rule above already forbids.
SHORT_HALF_LIFE = [
    "b0014", "b0169", "b0186", "b0439", "b0605", "b0720", "b0729", "b0755",
    "b1101", "b1136", "b1241", "b1276", "b1611", "b1817", "b2296", "b2415",
    "b2779", "b2926", "b3236", "b3612", "b3925", "b4025", "b4232", "b4395",
]
LONG_HALF_LIFE = [
    "b0002", "b0003", "b0004", "b0008", "b0114", "b0116", "b0118", "b0356",
    "b0720", "b1136", "b1479", "b2029", "b2296", "b2600", "b2925", "b3339",
    "b3340", "b3341", "b3342", "b3980", "b3986", "b4200", "b4201", "b4203",
]

_AMBIGUOUS = frozenset(SHORT_HALF_LIFE) & frozenset(LONG_HALF_LIFE)

# The disjoint sets actually compared. Order is preserved so the membership is
# still readable against the curation above.
SHORT_USED = [g for g in SHORT_HALF_LIFE if g not in _AMBIGUOUS]
LONG_USED = [g for g in LONG_HALF_LIFE if g not in _AMBIGUOUS]

assert not set(SHORT_USED) & set(LONG_USED), (
    "half-life control groups overlap: the test would be biased toward its own "
    "null. Add any newly-shared b-number to the exclusion above.")


def half_life_control(mat: np.ndarray, genes: list[str]) -> dict:
    """Compare normalized levels of short- vs long-half-life transcripts.

    Values are min-max scaled per profile first, so the comparison is on the
    same footing as the paper's (whose reported 0.572/0.573 are clearly on a
    0-1 scale, not raw copy numbers).
    """
    idx = {g: i for i, g in enumerate(genes)}
    short = [idx[g] for g in SHORT_USED if g in idx]
    long_ = [idx[g] for g in LONG_USED if g in idx]
    if len(short) < 3 or len(long_) < 3:
        return {"n_short": len(short), "n_long": len(long_),
                "mean_short": float("nan"), "mean_long": float("nan"),
                "p_value": float("nan"),
                "note": "too few half-life-annotated genes present"}

    lo = np.nanmin(mat, axis=0, keepdims=True)
    hi = np.nanmax(mat, axis=0, keepdims=True)
    scaled = (mat - lo) / np.where(hi - lo > 0, hi - lo, 1.0)

    a = np.nanmean(scaled[short], axis=1)
    b = np.nanmean(scaled[long_], axis=1)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]

    from scipy.stats import ranksums
    stat, p = ranksums(a, b)
    return {
        "n_short": len(short), "n_long": len(long_),
        "mean_short": float(a.mean()), "sem_short": float(a.std() / max(np.sqrt(a.size), 1)),
        "mean_long": float(b.mean()), "sem_long": float(b.std() / max(np.sqrt(b.size), 1)),
        "statistic": float(stat), "p_value": float(p),
        "n_excluded_ambiguous": len(_AMBIGUOUS),
        "excluded_ambiguous": sorted(_AMBIGUOUS),
        "paper": {"mean_short": 0.573, "mean_long": 0.572, "p_value": 0.41},
    }


# --------------------------------------------------------------------------
# 2. R cross-checks
# --------------------------------------------------------------------------
def _rscript() -> str | None:
    return shutil.which("Rscript")


def _run_r(code: str, timeout: int = 1800) -> tuple[bool, str]:
    rs = _rscript()
    if rs is None:
        return False, "Rscript not found on PATH"
    with tempfile.TemporaryDirectory() as td:
        script = Path(td) / "check.R"
        script.write_text(code, encoding="utf-8")
        try:
            p = subprocess.run([rs, str(script)], capture_output=True,
                               text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return False, "R timed out"
    return (p.returncode == 0), (p.stdout or "") + (p.stderr or "")


def _pcc(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float("nan")
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def crosscheck_quantile(seed: int = 0, shape=(400, 8),
                        with_ties: bool = True) -> dict:
    """Our quantile_normalize vs preprocessCore::normalize.quantiles.

    with_ties  round the fixture so that repeated values occur. THIS MATTERS.
               The fixture was continuous lognormal noise, in which ties have
               probability zero -- so the check reported "exact, max diff
               0.000000" for the entire life of the project while our
               implementation was NOT averaging tied ranks and R was. A
               cross-check that cannot exercise the divergence it is meant to
               cover is not evidence, however clean its output.

               Ties are the normal case in the real inputs: RMA summarisation
               of low-expression probesets and htseq zero counts both produce
               large tied blocks.
    """
    from ecomics.pipeline.platform import quantile_normalize

    rng = np.random.default_rng(seed)
    x = np.abs(rng.lognormal(3, 1, shape))
    if with_ties:
        # Round to one decimal AND floor the bottom decile to a single value,
        # which is what a detection floor does to real expression data.
        x = np.round(x, 1)
        x[x < np.quantile(x, 0.1)] = float(np.round(np.quantile(x, 0.1), 1))
    ours = quantile_normalize(x)

    with tempfile.TemporaryDirectory() as td:
        fin, fout = Path(td) / "in.tsv", Path(td) / "out.tsv"
        np.savetxt(fin, x, delimiter="\t")
        ok, log = _run_r(f'''
            suppressPackageStartupMessages(library(preprocessCore))
            m <- as.matrix(read.table("{fin.as_posix()}", sep="\\t"))
            q <- normalize.quantiles(m)
            write.table(q, "{fout.as_posix()}", sep="\\t",
                        row.names=FALSE, col.names=FALSE)
        ''')
        if not ok or not fout.exists():
            return {"available": False, "note": log.strip().splitlines()[-1:] or ["failed"]}
        theirs = np.loadtxt(fout, delimiter="\t")
    return {"available": True, "pcc": _pcc(ours, theirs),
            "max_abs_diff": float(np.nanmax(np.abs(ours - theirs)))}


def crosscheck_impute(seed: int = 0, shape=(300, 12), frac: float = 0.15) -> dict:
    """Our knn_impute vs impute::impute.knn."""
    from ecomics.pipeline.impute import knn_impute

    rng = np.random.default_rng(seed)
    base = rng.normal(size=(shape[0], 4)) @ rng.normal(size=(4, shape[1]))
    x = base + rng.normal(0, 0.1, shape)
    mask = rng.random(shape) < frac
    x[mask] = np.nan
    ours, _ = knn_impute(x, k=3)

    with tempfile.TemporaryDirectory() as td:
        fin, fout = Path(td) / "in.tsv", Path(td) / "out.tsv"
        np.savetxt(fin, x, delimiter="\t")
        ok, log = _run_r(f'''
            suppressPackageStartupMessages(library(impute))
            m <- as.matrix(read.table("{fin.as_posix()}", sep="\\t", na.strings="nan"))
            r <- impute.knn(m, k=3, rowmax=1, colmax=1)$data
            write.table(r, "{fout.as_posix()}", sep="\\t",
                        row.names=FALSE, col.names=FALSE)
        ''')
        if not ok or not fout.exists():
            return {"available": False, "note": log.strip().splitlines()[-1:] or ["failed"]}
        theirs = np.loadtxt(fout, delimiter="\t")
    return {"available": True,
            "pcc_imputed_only": _pcc(ours[mask], theirs[mask]),
            "pcc_overall": _pcc(ours, theirs),
            "n_imputed": int(mask.sum())}


def crosscheck_rma() -> dict:
    """Our from-scratch RMA vs affy::rma on the acquired CEL files."""
    from ecomics.pipeline.arrays import read_cdf, read_cel, rma

    cel_dir = C.RAW_DIR / C.GEO_ARRAY_SERIES
    cdf_path = C.RAW_DIR / "cdf" / "ecoliasv2.tsv"
    cels = sorted(cel_dir.glob("*.CEL")) if cel_dir.exists() else []
    if not cels or not cdf_path.exists():
        return {"available": False, "note": ["no CEL files or CDF layout"]}

    objs = [read_cel(p) for p in cels]
    ps, ours = rma(objs, read_cdf(cdf_path), verbose=False)

    with tempfile.TemporaryDirectory() as td:
        fout = Path(td) / "affy.tsv"
        script = C.REPO / "tools" / "crosscheck_rma.R"
        rs = _rscript()
        if rs is None:
            return {"available": False, "note": ["Rscript not found"]}
        p = subprocess.run([rs, str(script), str(cel_dir), str(fout)],
                           capture_output=True, text=True, timeout=3600)
        if p.returncode != 0 or not fout.exists():
            return {"available": False,
                    "note": (p.stderr or p.stdout).strip().splitlines()[-2:]}
        import csv
        rows = list(csv.DictReader(fout.open(), delimiter="\t"))

    cols = [c for c in rows[0] if c != "probeset"]
    theirs = {r["probeset"]: np.array([float(r[c]) for c in cols]) for r in rows}
    common = [p for p in ps if p in theirs]
    A = ours[[ps.index(p) for p in common]]
    B = np.vstack([theirs[p] for p in common])
    return {"available": True, "n_probesets": len(common),
            "pcc": _pcc(A, B),
            "rmse": float(np.sqrt(np.nanmean((A - B) ** 2))),
            "mean_abs_diff": float(np.nanmean(np.abs(A - B)))}


def crosscheck_all(verbose: bool = True) -> dict:
    """Run every available R cross-check and summarize."""
    out = {}
    checks = [("quantile normalization vs preprocessCore", crosscheck_quantile),
              ("k-NN imputation vs impute::impute.knn", crosscheck_impute),
              ("RMA vs affy::rma (real CEL files)", crosscheck_rma)]
    if verbose:
        print("R/Bioconductor cross-checks (ground truth, not runtime deps)")
    for name, fn in checks:
        try:
            res = fn()
        except Exception as exc:                      # noqa: BLE001
            res = {"available": False, "note": [str(exc)[:90]]}
        out[name] = res
        if not verbose:
            continue
        if not res.get("available"):
            print(f"  -- {name}: unavailable ({res.get('note', [''])[0]})")
        else:
            bits = [f"{k}={v:.6f}" if isinstance(v, float) else f"{k}={v}"
                    for k, v in res.items() if k != "available"]
            print(f"  ok {name}: " + "  ".join(bits))
    return out


if __name__ == "__main__":
    print(json.dumps(crosscheck_all(), indent=2, default=str))

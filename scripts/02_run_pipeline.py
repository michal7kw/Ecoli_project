#!/usr/bin/env python
"""Run the Ecomics normalization pipeline end-to-end on real raw data.

    python scripts/02_run_pipeline.py            # full run + validation
    python scripts/02_run_pipeline.py --no-check # skip the R cross-checks

Inputs (fetched by scripts/00_acquire.py):
    GSE12411   6 raw Affymetrix CEL files (GPL199), processed by our own RMA
    GSE73673   87 htseq count tables -- the paper's own 16-knockout RNA-Seq

Two genuinely different platforms measuring the same organism is exactly the
problem step 2 exists to solve, so this is a real test rather than a rehearsal.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ecomics import config as C                      # noqa: E402
from ecomics.pipeline.run import load_demo_inputs, run_pipeline  # noqa: E402
from ecomics.pipeline.validate import (              # noqa: E402
    crosscheck_all, half_life_control,
)


# The five stages of Fig. 2, in order, as they exist on `PipelineResult`.
_CASCADE = [("0. input", "raw"), ("1b. noise removed", "denoised"),
            ("2. platform-corrected", "corrected"),
            ("3. absolute (log10 copies)", "absolute"),
            ("4. filtered + imputed", "final")]

# How each R cross-check's own return dict maps onto the one number the figure
# shows. They differ because they check different things: quantile
# normalization is deterministic and should be bit-exact, while impute.knn uses
# a different neighbour-selection scheme and is expected to differ.
_CHECK_SPEC = {
    "quantile normalization vs preprocessCore": (
        "quantile normalization", "pcc", "preprocessCore::normalize.quantiles",
        "deterministic — anything below 1.0 would be a bug"),
    "k-NN imputation vs impute::impute.knn": (
        "k-NN imputation (k=3)", "pcc_imputed_only", "impute::impute.knn",
        "on imputed cells; a different neighbour scheme, so <1 is expected"),
    "RMA vs affy::rma (real CEL files)": (
        "RMA from scratch", "pcc", "affy::rma",
        "residual is grid discretization in the mode estimate"),
}


def write_validation_json(res, crosschecks: dict, hl: dict) -> Path:
    """Serialize what this run measured, so the figure atlas can plot it.

    Without this the numbers exist only in prose and in a gitignored log, which
    means `scripts/15_figures.py` would have to hard-code them -- exactly what
    "a number needs a provenance" forbids. The values are not recomputed here;
    they are the ones printed above.
    """
    cascade = []
    for label, attr in _CASCADE:
        m = getattr(res, attr)
        v = m[np.isfinite(m)]
        cascade.append({"step": label, "shape": f"{m.shape[0]}x{m.shape[1]}",
                        "median": float(np.median(v)),
                        "q1": float(np.percentile(v, 25)),
                        "q3": float(np.percentile(v, 75)),
                        "missing": float(1 - np.isfinite(m).mean())})

    checks = {}
    for key, (label, pcc_key, reference, note) in _CHECK_SPEC.items():
        got = crosschecks.get(key, {})
        if got.get("available"):
            checks[label] = {"pcc": float(got[pcc_key]), "reference": reference,
                             "note": note, "detail": {
                                 k: v for k, v in got.items() if k != "available"}}

    # Platform labels are the SOURCE keys ("affy_ecoliasv2", "rnaseq_gse73673"),
    # not a generic "array"/"rnaseq" -- two genuinely different platforms
    # measuring the same organism is the problem step 2 exists to solve, so the
    # labels name the actual platform.
    out = C.RESULTS / "pipeline_validation.json"
    platforms = {p: int(n) for p, n in
                 zip(*np.unique(np.asarray(res.platform), return_counts=True))}
    payload = {
        "platforms": platforms,
        "n_cel": int(sum(n for p, n in platforms.items() if p.startswith("affy"))),
        "n_htseq": int(sum(n for p, n in platforms.items()
                           if not p.startswith("affy"))),
        "n_genes": len(res.genes),
        "crosschecks": checks,
        "cascade": cascade,
        "half_life_control": hl,
        "synthetic_reference_warning":
            "step 3 uses a SYNTHETIC reference unless real copy numbers were "
            "supplied; Taniguchi et al. 2010 is not redistributable",
    }
    out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-check", action="store_true",
                    help="skip R/Bioconductor cross-checks")
    ap.add_argument("--per-gene-zscore", action="store_true",
                    help="use Fig. 2's literal per-gene z-score (see platform.py)")
    args = ap.parse_args()

    print("=" * 74)
    print("loading raw inputs")
    print("=" * 74)
    try:
        mat, genes, samples, platform = load_demo_inputs()
    except FileNotFoundError as exc:
        print(f"  {exc}")
        return 1

    print(f"\n  combined matrix: {mat.shape[0]} genes x {mat.shape[1]} samples "
          f"across {len(set(platform))} platforms")

    res = run_pipeline(mat, genes, samples, platform,
                       per_gene_zscore=args.per_gene_zscore)

    print("\n" + "=" * 74)
    print("summary")
    print("=" * 74)
    print(res.report())
    print(f"\n  imputed cells: {int(res.imputed_mask.sum()):,} "
          f"({res.imputed_mask.mean():.2%} of the final matrix)")

    print("\n" + "=" * 74)
    print("validation")
    print("=" * 74)
    hl = half_life_control(res.final, res.genes)
    # A NaN P-value means the control could not RUN (too few annotated genes
    # present), which is a different thing from the control failing. Testing
    # `p > 0.05` alone reports NaN as "SIGNIFICANT -- investigate", i.e. it
    # announces a normalization regression when the real problem is that the
    # half-life gene set is missing from the matrix. Check finiteness first and
    # surface the note the function already returns for exactly this case.
    if not np.isfinite(hl["p_value"]):
        print(f"  half-life negative control: NOT RUN -- "
              f"{hl.get('note', 'no P-value')} "
              f"(n_short={hl['n_short']}, n_long={hl['n_long']}; need >= 3 of each)")
    else:
        print(f"  half-life negative control: short={hl['mean_short']:.4f} "
              f"long={hl['mean_long']:.4f}  P={hl['p_value']:.3f}  "
              f"({'not significant, as in the paper' if hl['p_value'] > 0.05 else 'SIGNIFICANT -- investigate'})")

    crosschecks = {}
    if not args.no_check:
        print()
        crosschecks = crosscheck_all(verbose=True)

    out = C.RESULTS / "pipeline_output.npz"
    C.RESULTS.mkdir(parents=True, exist_ok=True)
    # `imputed_mask` is saved because ecomics/pipeline/impute.py's caveat turns
    # on it: a gene observed in 31% of profiles is kept and 69% of its row is
    # invented, and the only available mitigation is letting downstream work
    # stratify by what was real. Printing the percentage and dropping the mask
    # left that mitigation unavailable to anything reading this file.
    np.savez_compressed(out, final=res.final, genes=np.array(res.genes),
                        samples=np.array(res.samples), platform=res.platform,
                        imputed_mask=res.imputed_mask)
    print(f"\nwrote {out}")

    val = write_validation_json(res, crosschecks, hl)
    print(f"wrote {val}"
          + ("" if crosschecks else "  (no R cross-checks: --no-check was given)"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

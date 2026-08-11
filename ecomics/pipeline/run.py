"""End-to-end driver: raw platform data -> absolute-scale expression matrix.

Runs the four steps of paper Fig. 2 in order, on real data:

    1a  platform preprocessing   RMA (arrays) or htseq counts (RNA-Seq)
    1b  noise removal            anchored 2-component Gaussian mixture
    2   platform-bias correction quantile -> loess -> per-platform z-score
    3   absolute quantification  loess against reference copy numbers
    4   QC                       70% missingness filter + k-NN imputation

Reports the shape and a diagnostic at every step, so the attrition can be
compared against the paper's own (3,842 -> 3,581 x 4,589 -> 3,579 x 4,096).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ecomics import config as C
from ecomics.pipeline import arrays, rnaseq
from ecomics.pipeline.absolute import absolute_quantify, synthetic_reference
from ecomics.pipeline.impute import knn_impute, missingness_filter
from ecomics.pipeline.noise import remove_noise
from ecomics.pipeline.platform import correct_platform_bias

__all__ = ["PipelineResult", "run_pipeline", "load_demo_inputs"]

# Floor for the cross-platform gene intersection. The demo inputs share ~4,069
# genes, so anything in the low hundreds means the identifier spaces diverged
# rather than that the overlap is genuinely small.
_MIN_SHARED_GENES = 100


@dataclass
class PipelineResult:
    """Everything the pipeline produced, plus per-step diagnostics."""

    genes: list[str]
    samples: list[str]
    platform: np.ndarray
    raw: np.ndarray
    denoised: np.ndarray
    corrected: np.ndarray
    absolute: np.ndarray
    final: np.ndarray
    imputed_mask: np.ndarray
    steps: list[dict] = field(default_factory=list)

    def report(self) -> str:
        lines = ["step                              shape        diagnostic"]
        lines.append("-" * 74)
        for s in self.steps:
            lines.append(f"{s['name']:<32s}  {s['shape']:<12s} {s['note']}")
        return "\n".join(lines)


def _note(mat: np.ndarray) -> str:
    finite = np.isfinite(mat)
    if not finite.any():
        return "all missing"
    v = mat[finite]
    return (f"median={np.median(v):9.3f}  IQR=[{np.percentile(v,25):.3f}, "
            f"{np.percentile(v,75):.3f}]  missing={1-finite.mean():.2%}")


def run_pipeline(mat: np.ndarray, genes: list[str], samples: list[str],
                 platform: np.ndarray,
                 reference: np.ndarray | None = None,
                 shared: np.ndarray | None = None,
                 anchor_idx: np.ndarray | None = None,
                 missing_threshold: float = C.PAPER["missing_threshold"],
                 k: int = C.PAPER["knn_k"],
                 per_gene_zscore: bool = False,
                 verbose: bool = True) -> PipelineResult:
    """Run steps 1b-4 on a (genes x samples) matrix.

    reference/shared  absolute copy numbers for step 3. If omitted, a synthetic
                      reference is generated and the result is a DEMONSTRATION
                      of the machinery, not Ecomics' actual absolute scale --
                      which is anchored to Taniguchi et al. 2010.
    anchor_idx        indices of known-noise features for step 1b. If omitted,
                      the mixture falls back to a median split; see noise.py.
    """
    steps: list[dict] = []

    def record(name: str, m: np.ndarray) -> None:
        steps.append({"name": name, "shape": f"{m.shape[0]}x{m.shape[1]}",
                      "note": _note(m)})
        if verbose:
            print(f"  {name:<32s} {m.shape[0]:>6d} x {m.shape[1]:<4d}  {_note(m)}")

    if verbose:
        print("\nEcomics normalization pipeline")
        print("=" * 74)
    record("0. input", mat)

    if verbose:
        print("\n1b. noise removal (Gaussian mixture, per profile)")
    denoised, _fits = remove_noise(mat, anchor_idx=anchor_idx, verbose=verbose)
    record("1b. noise removed", denoised)

    if verbose:
        print("\n2. platform-bias correction")
    corrected = correct_platform_bias(denoised, platform,
                                      per_gene_zscore=per_gene_zscore)
    record("2. platform-corrected", corrected)

    if verbose:
        print("\n3. absolute-level quantification")
    if reference is None:
        reference, shared = synthetic_reference(corrected)
        if verbose:
            print("  NOTE: using a SYNTHETIC reference (Taniguchi et al. 2010 "
                  "copy numbers\n        are not redistributable here). The "
                  "machinery is real; the\n        resulting scale is a "
                  "demonstration, not Ecomics' absolute scale.")
    absolute, _cals = absolute_quantify(corrected, reference, shared,
                                        return_log=True, verbose=verbose)
    record("3. absolute (log10 copies)", absolute)

    if verbose:
        print("\n4. quality control")
    filt = missingness_filter(absolute, missing_threshold, verbose=verbose)
    kept_genes = [g for g, keep in zip(genes, filt.keep_genes) if keep]
    kept_samples = [s for s, keep in zip(samples, filt.keep_profiles) if keep]
    final, imputed = knn_impute(filt.matrix, k=k, verbose=verbose)
    record("4. filtered + imputed", final)

    return PipelineResult(
        genes=kept_genes, samples=kept_samples,
        platform=platform[filt.keep_profiles],
        raw=mat, denoised=denoised, corrected=corrected,
        absolute=absolute, final=final, imputed_mask=imputed, steps=steps,
    )


def load_demo_inputs(verbose: bool = True
                     ) -> tuple[np.ndarray, list[str], list[str], np.ndarray]:
    """Assemble a real two-platform matrix from the acquired raw data.

    Platform A: 6 Affymetrix CEL arrays (GSE12411, GPL199), processed with the
                from-scratch RMA in pipeline/arrays.py.
    Platform B: 87 htseq count tables (GSE73673) -- the paper's OWN 16-knockout
                RNA-Seq experiment.

    Both are keyed by b-number, so the two platforms can be joined on genes and
    the pipeline has a genuine cross-platform problem to solve, which is the
    whole point of step 2.
    """
    cel_dir = C.RAW_DIR / C.GEO_ARRAY_SERIES
    cdf_path = C.RAW_DIR / "cdf" / "ecoliasv2.tsv"
    seq_dir = C.RAW_DIR / C.GEO_RNASEQ_SERIES

    blocks: list[tuple[list[str], list[str], np.ndarray, str]] = []

    if cel_dir.exists() and cdf_path.exists():
        cels = [arrays.read_cel(p) for p in sorted(cel_dir.glob("*.CEL"))]
        if cels:
            cdf = arrays.read_cdf(cdf_path)
            ps, expr = arrays.rma(cels, cdf, verbose=False)
            p2g = arrays.probeset_to_gene(ps)
            keep = [i for i, p in enumerate(ps) if p in p2g]
            genes = [p2g[ps[i]] for i in keep]
            # several probe sets can map to one gene; average them
            uniq = sorted(set(genes))
            gi = {g: i for i, g in enumerate(uniq)}
            acc = np.zeros((len(uniq), expr.shape[1]))
            cnt = np.zeros(len(uniq))
            for row, g in zip(expr[keep], genes):
                acc[gi[g]] += row
                cnt[gi[g]] += 1
            blocks.append((uniq, [c.name for c in cels],
                           acc / cnt[:, None], "affy_ecoliasv2"))
            if verbose:
                print(f"  platform affy_ecoliasv2: {len(uniq)} genes x "
                      f"{len(cels)} arrays (from raw CELs via our RMA)")

    if seq_dir.exists() and any(seq_dir.glob("*.htcount.txt")):
        g, s, counts = rnaseq.read_htseq_dir(seq_dir)
        blocks.append((g, s, rnaseq.counts_to_cpm(counts), "rnaseq_gse73673"))
        if verbose:
            print(f"  platform rnaseq_gse73673: {len(g)} genes x {len(s)} "
                  f"samples (htseq counts -> log2 CPM)")

    if not blocks:
        raise FileNotFoundError(
            "no raw inputs found -- run `python scripts/00_acquire.py` first")

    common = sorted(set.intersection(*(set(b[0]) for b in blocks)))
    if verbose:
        print(f"  genes shared by all platforms: {len(common)}")
    # Assert, do not merely print. The two platforms arrive here through
    # different identifier routes -- arrays via `probeset_to_gene`, RNA-Seq via
    # whatever htseq-count wrote -- so a change on either side can empty this
    # intersection. `np.hstack` on zero-row blocks then succeeds, every later
    # step "works" on a (0, n) matrix, and the summary prints a full run with
    # `missing=100%`. A pipeline that completes on no data is worse than one
    # that fails.
    if len(common) < _MIN_SHARED_GENES:
        raise ValueError(
            f"only {len(common)} genes shared across {len(blocks)} platform(s), "
            f"expected at least {_MIN_SHARED_GENES} -- the identifier spaces "
            f"have probably diverged. First ids per platform: "
            + "; ".join(f"{b[3]}={b[0][:3]}" for b in blocks))

    cols, samples, plats = [], [], []
    for genes, snames, mat, pname in blocks:
        idx = {g: i for i, g in enumerate(genes)}
        cols.append(mat[[idx[g] for g in common]])
        samples.extend(f"{pname}:{s}" for s in snames)
        plats.extend([pname] * mat.shape[1])
    return np.hstack(cols), common, samples, np.asarray(plats)

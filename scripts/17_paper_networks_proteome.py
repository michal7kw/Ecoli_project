#!/usr/bin/env python
"""What do the paper's two extra network arms buy the proteome?

    python scripts/17_paper_networks_proteome.py
    python scripts/17_paper_networks_proteome.py --no-leaky      # skip the leaky arm
    python scripts/17_paper_networks_proteome.py --out results/scratch.json

THE QUESTION. Supplementary Data 2 holds six networks. `ecomics/networks.py`
implemented four of them and never had the other two -- a sigma-factor network
(8,180 edges) and a small-RNA network (213). `scripts/04` now fits all six. Are
the two additions worth anything, or is the layer already saturated at four?

THE HYPOTHESIS. Near-inert. The small-RNA network reaches 5 of 589 proteome
targets, so it cannot move a mean over 589; the sigma-factor network covers 566
but is transcription-focused and largely redundant with the TRN.

WHAT WOULD FALSIFY IT. Either arm moving the per-profile PCC by more than
run-to-run noise, in either direction. A DROP would be as interesting as a
rise: the ensemble is an unweighted mean, so a weak arm with wide coverage
drags it, which is exactly how the scraped TRN used to cost this layer 0.044.

WHAT IS HELD CONSTANT. Same LOCO folds, same `alpha`, same `max_neighbours`,
same metrics, same locally-built CPN. Only the arm COUNT changes:

    paper_matched   TRN PPI KEGG + CPN          4 arms
    paper           + SIGMA + SRNA              6 arms

⚠ THIS SCRIPT USED TO HAVE A THIRD RUNG, AND IT IS GONE. `scraped` fitted the
same four slots from live 2026 databases, and the gap between it and
`paper_matched` -- **+0.1127**, against **-0.0000133** for the two extra arms --
is why Data 2 is the layer's only source. The scraped loaders were removed with
that decision, so the rung cannot be re-run; its recorded values are historical
and must not be re-derived as if they were live.

THE LEAKY ARM. Data 2 also ships the paper's CPN, computed across the full
compendium -- every LOCO test fold is inside its edges. It is fitted and
reported here as `CPN_paper_LEAKY` and excluded from the ensemble mean by
`moma.proteome_paper`, which measures what the leak is worth instead of hiding
it. Its number is not comparable to anything else in this file.

READ `SRNA` NEXT TO ITS COVERAGE. It reaches 5 of 589 proteome targets. Its PCC
is the highest single-arm number here and means nothing: an ABSENT predictor,
not a good one.

Nothing here changes a model, and nothing here touches `results/all_layers.json`
-- whose proteome block is the six-arm rung below, written by `scripts/04`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Windows consoles and redirected stdout default to cp1252, which cannot encode
# the marks this repo uses in reports. Without this, `--help` alone raises
# UnicodeEncodeError at the first "⚠" in the docstring above -- argparse prints
# it before main() runs, so the script cannot even describe itself. The same
# character in a run's output would discard the results after every fit.
# `scripts/04_reproduce.py` carries this guard and the reasoning that earned it.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ecomics import config as C                          # noqa: E402
from ecomics.db.api import Ecomics                       # noqa: E402
from ecomics.evaluate import evaluate_predictions, loco_splits   # noqa: E402
from ecomics.metrics import MIN_CONDITIONS_FOR_COLUMN_PCC  # noqa: E402
from ecomics.moma.proteome import own_mrna_baseline      # noqa: E402
from ecomics.moma.proteome_paper import (                # noqa: E402
    PaperProteomeEnsemble, build_paper_ensemble_networks, is_leaky,
)
# Shared with scripts/04, which prints the same suppressed cell. This was a
# byte-identical local copy whose docstring already said "as in scripts/04" --
# which stopped being true the moment the function moved to ecomics/reporting.py.
# Identical bodies with drifted docstrings is how the three copies of `_aligned`
# announced themselves too.
from ecomics.reporting import _pcc_str                   # noqa: E402

_ARRAY_KEYS = ("pcc_per_sample", "pcc_per_molecule")

# Anything below this is noise at 5 conditions. Used only for the verdict line;
# the numbers themselves are reported unrounded.
NOISE = 0.01


def _scalars(m: dict) -> dict:
    return {k: v for k, v in m.items() if k not in _ARRAY_KEYS}


def _sizes(nets: dict) -> dict:
    return {n: {"nodes": net.n_nodes, "edges": net.n_edges}
            for n, net in nets.items()}


def _run(ens_factory, nets, Xt, tcols, Yp, pcols, keys) -> dict:
    """Fit and predict over the LOCO folds; returns per-arm metrics."""
    preds = {n: np.full_like(Yp, np.nan) for n in list(nets) + ["ENSEMBLE"]}
    for tr, te in loco_splits(keys):
        ens = ens_factory(nets).fit(Xt[tr], tcols, Yp[tr], pcols)
        for n, p in ens.predict_each(Xt[te], tcols, pcols).items():
            preds[n][te] = p
        preds["ENSEMBLE"][te] = ens.predict(Xt[te], tcols, pcols)

    rows = {}
    for n, p in preds.items():
        m = evaluate_predictions(p, Yp, MIN_CONDITIONS_FOR_COLUMN_PCC,
                                 n_effective=len(keys))
        rows[n] = {**_scalars(m),
                   "coverage": int(np.isfinite(p).any(axis=0).sum()),
                   "in_ensemble_mean": not is_leaky(n) and n != "ENSEMBLE"}
    return rows


def _table(title: str, rows: dict, targets: int) -> None:
    print(f"\n  {title}")
    print(f"  {'arm':<16s} {'proteins':>9s} {'PCC/molecule':>16s} "
          f"{'PCC/prof':>9s} {'RMSE':>8s}")
    for n, r in rows.items():
        flag = "  <- excluded from mean (leaky)" if is_leaky(n) else ""
        print(f"  {n:<16s} {r['coverage']:>4d}/{targets:<4d} {_pcc_str(r)} "
              f"{r['pcc_row_mean']:>9.3f} {r['rmse']:>8.4f}{flag}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--alpha", type=float, default=1e-3,
                    help="LASSO penalty, identical on both sides")
    ap.add_argument("--max-neighbours", type=int, default=200,
                    help="neighbour cap, identical on both sides. The paper's "
                         "PPI has a median of 286, so this binds -- do not "
                         "move it inside a comparison run")
    ap.add_argument("--no-leaky", action="store_true",
                    help="do not fit the paper's compendium-wide CPN at all")
    ap.add_argument("--out", type=Path,
                    default=C.RESULTS / "paper_networks_proteome.json")
    args = ap.parse_args()

    if not C.DB_PATH.exists():
        print(f"missing database: {C.DB_PATH}\nRun: python scripts/01_build_db.py")
        return 1
    if not C.SUPPLEMENTARY["interactions"].exists():
        print(f"missing Supplementary Data 2: {C.SUPPLEMENTARY['interactions']}\n"
              f"Run: python scripts/00_acquire.py")
        return 1

    db = Ecomics()
    shared, mats = db.aligned("transcriptome", "proteome")
    Xt, tcols, keys = mats["transcriptome"]
    Yp, pcols, _ = mats["proteome"]
    print("=" * 78)
    print("PROTEOME -- what the paper's two extra network arms are worth")
    print("=" * 78)
    print(f"  {len(shared)} shared conditions, {len(pcols)} proteome targets, "
          f"{len(tcols)} transcripts")
    if len(shared) < 3:
        print("  too few to cross-validate")
        return 1

    P_all = db.matrix("proteome")
    held = ~np.isin(P_all.condition_keys, shared)
    print(f"  CPN built from {int(held.sum())} proteome conditions held OUT of "
          f"evaluation, and SHARED by both rungs")

    print("\n  paper networks (Supplementary Data 2):")
    paper, reports = build_paper_ensemble_networks(
        cpn_values=P_all.values[held], cpn_columns=P_all.columns,
        include_leaky_cpn=not args.no_leaky, verbose=True)

    def paper_factory(nets):
        return PaperProteomeEnsemble(nets, alpha=args.alpha,
                                     max_neighbours=args.max_neighbours)

    # The four arms `networks.py` used to implement. `paper` may also carry the
    # leaky CPN; `matched` never does -- it exists to isolate the two ADDED
    # arms, so it must differ from `paper` in exactly those two.
    matched = {n: paper[n] for n in ("TRN", "PPI", "KEGG", "CPN") if n in paper}

    rows_matched = _run(paper_factory, matched, Xt, tcols, Yp, pcols, keys)
    rows_paper = _run(paper_factory, paper, Xt, tcols, Yp, pcols, keys)

    base = np.full_like(Yp, np.nan)
    for tr, te in loco_splits(keys):
        base[te] = own_mrna_baseline(Xt[tr], Yp[tr], Xt[te], tcols, pcols)
    mb = evaluate_predictions(base, Yp, MIN_CONDITIONS_FOR_COLUMN_PCC,
                              n_effective=len(keys))

    _table("PAPER_MATCHED (Data 2, four arms)", rows_matched, len(pcols))
    _table("PAPER (Data 2, all six arms -- results/all_layers.json)",
           rows_paper, len(pcols))
    print(f"\n  {'own mRNA':<16s} {'-':>9s} {_pcc_str(mb)} "
          f"{mb['pcc_row_mean']:>9.3f} {mb['rmse']:>8.4f}   (paper: 0.34 +/- 0.18)")

    # ---- verdict, computed rather than asserted
    mtch = rows_matched["ENSEMBLE"]["pcc_row_mean"]
    b = rows_paper["ENSEMBLE"]["pcc_row_mean"]
    d_arms = b - mtch
    leaky = {n: r["pcc_row_mean"] for n, r in rows_paper.items() if is_leaky(n)}
    leak_gain = (max(leaky.values()) - rows_paper["CPN"]["pcc_row_mean"]
                 if leaky and "CPN" in rows_paper else None)
    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    print(f"  paper_matched, 4 arms      : {mtch:.4f}  (coverage "
          f"{rows_matched['ENSEMBLE']['coverage']}/{len(pcols)})")
    print(f"  paper, 6 arms              : {b:.4f}  (coverage "
          f"{rows_paper['ENSEMBLE']['coverage']}/{len(pcols)})")
    print(f"  SIGMA + SRNA are worth     : {d_arms:+.6f} PCC, "
          f"{rows_paper['ENSEMBLE']['rmse'] - rows_matched['ENSEMBLE']['rmse']:+.4f} RMSE")
    if abs(d_arms) <= NOISE:
        print(f"  -> WITHIN NOISE on PCC (|delta| <= {NOISE}). Read the RMSE "
              f"column beside it: PCC cannot see range compression.")
    else:
        print(f"  -> the two added arms move the layer by {d_arms:+.4f}.")
    if leak_gain is not None:
        print(f"  paper CPN (LEAKY) minus our held-out CPN: {leak_gain:+.4f}"
              f"   <- the value of the leak, excluded from every mean above")
    print(f"  SRNA coverage: {rows_paper['SRNA']['coverage']}/{len(pcols)} "
          f"-- an absent predictor, not a failed one")

    out = {
        "script": "scripts/17_paper_networks_proteome.py",
        "shared_conditions": len(shared),
        "proteome_targets": len(pcols),
        "cpn_held_out_conditions": int(held.sum()),
        "alpha": args.alpha,
        "max_neighbours": args.max_neighbours,
        "networks": {"paper": _sizes(paper),
                     "paper_matched": _sizes(matched)},
        "parse_reports": {n: vars(r) for n, r in reports.items()},
        "arms": {"paper_matched": rows_matched, "paper": rows_paper},
        "own_mrna_baseline": _scalars(mb),
        "excluded_from_mean": sorted(n for n in paper if is_leaky(n)),
        "retired_rung": {
            "name": "scraped",
            "note": "TRN/PPI/KEGG from live public databases; the loaders "
                    "were removed, so this rung cannot be re-run. Recorded "
                    "values below are historical.",
            "ensemble_pcc_row_mean": 0.1518,
            "delta_graph_source": 0.1127,
            "record": ["results/all_layers_scraped_superseded.json",
                       "the paper_networks_proteome.json written before removal"],
        },
        "verdict": {
            "ensemble_paper_matched": mtch,
            "ensemble_paper": b,
            "delta_extra_arms": d_arms,
            "rmse_delta_extra_arms": (rows_paper["ENSEMBLE"]["rmse"]
                                      - rows_matched["ENSEMBLE"]["rmse"]),
            "within_noise": bool(abs(d_arms) <= NOISE),
            "noise_threshold": NOISE,
            "leak_value_vs_held_out_cpn": leak_gain,
            "srna_coverage": rows_paper["SRNA"]["coverage"],
        },
    }
    args.out.write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    print(f"\nwrote {args.out}")
    db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

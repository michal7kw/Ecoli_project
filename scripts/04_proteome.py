#!/usr/bin/env python
"""Evaluate MOMA's proteome module -- the 4-source network ensemble -- under LOCO.

    python scripts/04_proteome.py
    python scripts/04_proteome.py --out results/proteome_loco.json

The question
------------
The paper's central proteome claim is that a protein's abundance is predicted
better from the expression of its FUNCTIONAL NEIGHBOURS than from its OWN
mRNA. This script tests exactly that: it fits one LASSO per target protein per
network over four graphs (TRN, PPI, KEGG, CPN), averages the networks that
cover each protein, and scores the result against the paper's own-mRNA
baseline under leave-one-condition-out cross-validation.

What to read, and what NOT to read
----------------------------------
The transcriptome and proteome layers share only **5 conditions**. Per-molecule
PCC correlates each molecule ACROSS conditions, so on 5 points it is noise
dressed as a result -- it is suppressed here and prints as `n/a (5 cond)`. The
two numbers that ARE meaningful:

  * the DIRECTION -- ensemble vs own-mRNA on the per-profile axis;
  * COVERAGE -- how many proteins each network can predict at all, and how many
    the union of the four can.

Upstream of record
------------------
This repository is a subset. The version of record for this evaluation is
`scripts/04_reproduce.py:eval_proteome` in the full research repository, which
runs all five layers into one table; the logic below is extracted from it
unchanged. If the two ever disagree, that file is right and this one has
drifted.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ecomics import config as C                          # noqa: E402
from ecomics.db.api import Ecomics                       # noqa: E402
from ecomics.evaluate import (                           # noqa: E402
    evaluate_predictions, loco_splits,
)
from ecomics.metrics import MIN_CONDITIONS_FOR_COLUMN_PCC  # noqa: E402
from ecomics.moma.proteome import ProteomeEnsemble, own_mrna_baseline  # noqa: E402
from ecomics.networks import load_all_networks           # noqa: E402


_ARRAY_KEYS = ("pcc_per_sample", "pcc_per_molecule")

# MIN_CONDITIONS_FOR_COLUMN_PCC (= 15) is imported from ecomics.metrics above.
#
# Below that many CONDITIONS, the per-molecule axis is not estimable and is
# reported as "n/a (n cond)" rather than as a number. Per-molecule PCC
# correlates each molecule ACROSS conditions, so the proteome's 5 shared
# conditions give a 5-point correlation: its sampling distribution under the
# null is so wide that roughly a third of molecules exceed |r| = 0.7 by chance.
# Printing -0.383 next to the paper's 0.55 invites a comparison that the sample
# size cannot support. 15 is the conventional floor for a correlation to be
# worth interpreting; the per-profile axis and coverage are reported instead,
# and are computable.


def _pcc_str(m: dict, width: int = 16) -> str:
    """Per-molecule PCC, or a reason it is not reported."""
    if m.get("pcc_column_suppressed"):
        return f"n/a ({m.get('n_conditions_available', '?')} cond)".rjust(width)
    return f"{m['pcc_mean']:>8.3f} +/-{m['pcc_sd']:<5.3f}"


def _scalars(m: dict) -> dict:
    """Drop per-element arrays so a metrics dict is JSON-serializable."""
    return {k: v for k, v in m.items() if k not in _ARRAY_KEYS}


def _aligned(db: Ecomics, layers: list[str]):
    """Condition-averaged matrices restricted to conditions shared by `layers`."""
    shared = db.shared_conditions(*layers)
    out = {}
    for l in layers:
        m = db.matrix(l).averaged_by_condition().subset_conditions(shared)
        order = np.argsort(m.condition_keys)
        out[l] = (m.values[order], m.columns, m.condition_keys[order])
    return shared, out


# --------------------------------------------------------------------------
def eval_proteome(db: Ecomics) -> dict | None:
    """LOCO-evaluate the ensemble, each network alone, and the own-mRNA baseline."""
    print("\n" + "=" * 74)
    print("PROTEOME  -- 4-source network ensemble, LOCO")
    print("=" * 74)
    shared, mats = _aligned(db, ["transcriptome", "proteome"])
    Xt, tcols, keys = mats["transcriptome"]
    Yp, pcols, _ = mats["proteome"]
    print(f"  {len(shared)} shared conditions (paper: 5 conditions / 18 profiles)")
    if len(shared) < 3:
        print("  too few to cross-validate")
        return None

    # The CPN is built ONLY from proteome conditions held OUT of evaluation.
    # Building it from the evaluation conditions would leak the answer into the
    # graph the model predicts from.
    P_all = db.matrix("proteome")
    held = ~np.isin(P_all.condition_keys, shared)
    print(f"  CPN built from {int(held.sum())} proteome conditions held OUT "
          "of evaluation")
    nets = load_all_networks(cpn_values=P_all.values[held],
                             cpn_columns=P_all.columns, verbose=False)

    preds = {n: np.full_like(Yp, np.nan) for n in list(nets) + ["ENSEMBLE"]}
    base = np.full_like(Yp, np.nan)
    for tr, te in loco_splits(keys):
        ens = ProteomeEnsemble(nets).fit(Xt[tr], tcols, Yp[tr], pcols)
        each = ens.predict_each(Xt[te], tcols, pcols)
        for n, p in each.items():
            preds[n][te] = p
        preds["ENSEMBLE"][te] = ens.predict(Xt[te], tcols, pcols)
        base[te] = own_mrna_baseline(Xt[tr], Yp[tr], Xt[te], tcols, pcols)

    cov = {n: int(np.isfinite(p).any(axis=0).sum()) for n, p in preds.items()}
    rows = {}
    print(f"\n  {'predictor':<12s} {'proteins':>9s} {'PCC/molecule':>16s} "
          f"{'PCC/prof':>9s} {'RMSE':>8s}")
    for n, p in preds.items():
        m = evaluate_predictions(p, Yp, MIN_CONDITIONS_FOR_COLUMN_PCC,
                                 n_effective=len(keys))
        rows[n] = {**_scalars(m), "coverage": cov[n]}
        print(f"  {n:<12s} {cov[n]:>9d} {_pcc_str(m)} "
              f"{m['pcc_row_mean']:>9.3f} {m['rmse']:>8.4f}")
    mb = evaluate_predictions(base, Yp, MIN_CONDITIONS_FOR_COLUMN_PCC,
                              n_effective=len(keys))
    rows["own_mRNA_baseline"] = _scalars(mb)
    print(f"  {'own mRNA':<12s} {'-':>9s} {_pcc_str(mb)} "
          f"{mb['pcc_row_mean']:>9.3f} {mb['rmse']:>8.4f}   (paper: 0.34 +/- 0.18)")

    ens_r, base_r = rows["ENSEMBLE"]["pcc_row_mean"], rows["own_mRNA_baseline"]["pcc_row_mean"]
    print(f"\n  neighbours vs own mRNA (per profile): {ens_r:.3f} vs {base_r:.3f}"
          f"{f' ({ens_r / base_r:.1f}x)' if base_r > 0 else ''}")
    print(f"  coverage: union of 4 networks {rows['ENSEMBLE']['coverage']}/{len(pcols)}, "
          f"best single network "
          f"{max(rows[n]['coverage'] for n in nets)}/{len(pcols)}")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path,
                    default=C.RESULTS / "proteome_loco.json",
                    help="where to write the metrics (default: "
                         "results/proteome_loco.json)")
    args = ap.parse_args()

    if not C.DB_PATH.exists():
        print(f"missing database: {C.DB_PATH}\n"
              f"Run: python scripts/01_build_db.py")
        return 1
    if not C.PARQUET_DIR.exists() or not any(C.PARQUET_DIR.glob("*.parquet")):
        print(f"missing Parquet matrices in {C.PARQUET_DIR}\n"
              f"Run: python scripts/01_build_db.py")
        return 1

    db = Ecomics()
    try:
        rows = eval_proteome(db)
    finally:
        db.close()
    if rows is None:
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"proteome": rows}, indent=2, default=float),
                        encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

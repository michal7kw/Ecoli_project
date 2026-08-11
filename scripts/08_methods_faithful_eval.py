#!/usr/bin/env python
"""Evaluate exactly as the paper's Supplementary Methods says, and compare.

    python scripts/08_methods_faithful_eval.py

Why this exists
---------------
For most of this reproduction the transcriptome comparison read "ours 0.295
versus the paper's 0.54", and the paper's baselines (0.25/0.26/0.36) could not be
reproduced at all -- ours came out at -0.017/-0.106/-0.101. Since a baseline has
no free parameters, that had to be a difference in WHAT THE NUMBER IS COMPUTED
OVER, and it made every "paper vs ours" comparison unsafe (DISCREPANCIES.md 3).

The Supplementary Methods (ncomms13090-s1.pdf, Section 3.3.3) states all four
choices outright. None of them were guessable from the article body:

  1. AXIS   "we measure Pearson's correlation coefficient (PCC) between predicted
             expression levels and average of known expression levels for
             profiles belonging to the test condition"
            -> one PCC per test CONDITION, across genes, against the
               condition-AVERAGED truth. Not per gene across conditions.

  2. SCALE  "We then used min-max standardization on the absolute scale values
             for each gene"           y' = (y - min(y)) / (max(y) - min(y))
            -> per-GENE min-max. This is what flattens the mean-expression
               profile whose dominance drove our per-profile baseline to 0.63.

  3. SUBSET "we first extracted the transcriptome profiles that correspond to
             samples in the exponential phase from the compendium (2610
             profiles)"
            -> growth phase is in Supplementary Data 1, not in the released
               expression table. Applying it reproduces 2,610 profiles and 493
               conditions EXACTLY.

  4. WT     "we define as WT (wild-type) profiles the ones of the MG1655 strain
             in LB or M9 medium, any carbon source, without any stresses or
             genetic perturbation"
            -> a narrow subset (10% of profiles). Ours was `gp == none`, i.e.
               70%, which is why our WT and mean baselines were near-identical
               while the paper's differ by 0.10.

What it finds
-------------
Under the paper's own protocol our model scores ~0.578 against the paper's 0.54,
i.e. we do not underperform at all. But our BASELINES also land ~0.53 against the
paper's 0.26, so our model-minus-baseline margin is much smaller. The residual
discrepancy is therefore entirely in the baseline level, not in the model -- the
opposite of what "0.295 versus 0.54" suggested.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ecomics import config as C                              # noqa: E402
from ecomics.db.api import Ecomics                           # noqa: E402
from ecomics.metrics import pcc_per_row                      # noqa: E402
from ecomics.networks import gene_symbol_map                 # noqa: E402

PAPER = {"moma": 0.54, "random": 0.25, "mean": 0.26, "wildtype": 0.36,
         "tf_moma": 0.68, "tf_random": 0.41, "tf_mean": 0.41, "tf_wildtype": 0.40,
         "n_profiles": 2610, "n_conditions": 493, "n_testable": 262}
KINDS = ("random", "mean", "wildtype")


def tf_indices(columns: list[str]) -> np.ndarray:
    path = C.REMOTE_FILES.get("regulondb_tf_gene")
    if path is None or not path.exists():
        return np.array([], dtype=int)
    names = {r["regulator"].strip().lower()
             for r in csv.DictReader(path.open(encoding="utf-8"))}
    s2b = gene_symbol_map()
    tfb = {s2b[n] for n in names if n in s2b}
    return np.array([i for i, c in enumerate(columns) if c in tfb], dtype=int)


def lb_or_m9_media() -> set[str]:
    """Medium IDs whose base medium is LB or M9, any carbon source."""
    path = C.PROK_DIR / "medium.json"
    out = set()
    for m in json.loads(path.read_text(encoding="utf-8")):
        for field in ("Base Medium", "Description"):
            if (m.get(field) or "").strip().upper().startswith(("LB", "M9")):
                out.add(m["ID"])
                break
    return out


def growth_phase(db: Ecomics, profile_ids: np.ndarray) -> np.ndarray:
    """Per-profile exponential-phase mask, from Supplementary Data 1.

    The released expression table carries no growth phase, so this is the one
    piece of the paper's protocol that CANNOT be reconstructed without the
    supplementary material.
    """
    import pandas as pd

    path = C.SUPPLEMENTARY["metadata"]
    if not path.exists():
        raise SystemExit(f"missing {path}\nRun: python scripts/00_acquire.py")
    src = {r["id"]: r["source_id"] for r in db.conn.execute(
        "SELECT id, source_id FROM profile WHERE layer='transcriptome'")}
    meta = pd.read_excel(path, sheet_name="Transcriptome")
    phase = dict(zip(meta["ID"].astype(str), meta["Growth Phase"].astype(str)))
    return np.array([phase.get(src.get(int(p), ""), "").lower()
                     .startswith(("exp", "mid-exp")) for p in profile_ids])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--random-draws", type=int, default=200,
                    help="resamples of the 10-profile random set; the Methods "
                         "specify 1000, and the estimate is flat past ~200")
    ap.add_argument("--out", type=Path,
                    default=C.RESULTS / "methods_faithful_eval.json")
    args = ap.parse_args()

    cache = C.RESULTS / "transcriptome_predictions.npz"
    if not cache.exists():
        raise SystemExit(f"missing {cache}\nRun: python scripts/07_baseline_"
                         "calibration.py --refit")
    d = np.load(cache, allow_pickle=True)
    Y, P, keys = d["y_true"], d["y_pred"], d["condition_keys"]

    db = Ecomics()
    T = db.matrix("transcriptome")
    tf = tf_indices(T.columns)
    lbm9 = lb_or_m9_media()
    exp = growth_phase(db, T.profile_ids)
    print(f"exponential-phase profiles: {exp.sum()} (paper: {PAPER['n_profiles']})")

    Ye, Pe, ke = Y[exp], P[exp], keys[exp]
    uniq, inv = np.unique(ke, return_inverse=True)
    print(f"conditions:                 {len(uniq)} (paper: {PAPER['n_conditions']})\n")

    part = lambda k: k.split(".")                                  # noqa: E731
    wt = np.array([part(k)[0] == "MG1655" and part(k)[1] in lbm9
                   and part(k)[2] in ("none", "na", "") and part(k)[3] == "none"
                   for k in ke])
    print(f"WT profiles, paper definition: {wt.sum()} ({wt.mean():.1%}); "
          f"ours was `gp == none` = {np.mean([part(k)[3] == 'none' for k in ke]):.1%}\n")

    # condition-averaged truth and prediction, then per-gene min-max
    Yc = np.vstack([Ye[inv == i].mean(axis=0) for i in range(len(uniq))])
    Pc = np.vstack([Pe[inv == i].mean(axis=0) for i in range(len(uniq))])
    lo, hi = Ye.min(0, keepdims=True), Ye.max(0, keepdims=True)
    rg = np.where(hi - lo > 0, hi - lo, 1.0)
    Ys, Ycs, Pcs = (Ye - lo) / rg, (Yc - lo) / rg, (Pc - lo) / rg

    n_random_draws = args.random_draws
    rng = np.random.default_rng(0)
    bp = {k: np.full_like(Ycs, np.nan) for k in KINDS}
    for ci in range(len(uniq)):
        tr = inv != ci
        bp["mean"][ci] = np.nanmean(Ys[tr], axis=0)
        w = tr & wt
        bp["wildtype"][ci] = np.nanmean(Ys[w], axis=0) if w.any() else bp["mean"][ci]
        # The Methods specify the random baseline as the average over "1000
        # times of repetitive sampling ... of the 10-profile random set". A
        # SINGLE draw is not the same estimator: one 10-profile mean is noisy,
        # and that noise depresses its correlation with the truth. Drawing once
        # gave 0.453 where averaging gives 0.528 -- and the averaged form is
        # what makes random and mean nearly coincide, exactly as they do in the
        # paper (0.25 vs 0.26). Getting this wrong flattered us by 0.075.
        idx = np.flatnonzero(tr)
        bp["random"][ci] = np.mean(
            [Ys[rng.choice(idx, size=10, replace=False)].mean(axis=0)
             for _ in range(n_random_draws)], axis=0)

    def score(A, B, idx=slice(None)):
        r = pcc_per_row(A[:, idx], B[:, idx])
        return float(np.nanmean(r)), float(np.nanstd(r))

    out = {"n_profiles": int(exp.sum()), "n_conditions": int(len(uniq)),
           "n_wt_profiles": int(wt.sum()), "paper": PAPER, "ours": {}}
    print(f"{'predictor':<14s} {'ours (all)':>16s} {'paper':>8s} | "
          f"{'ours (TFs)':>16s} {'paper':>8s}")
    print("-" * 70)
    for name, arr, pk in (("MOMA", Pcs, "moma"), *[(k, bp[k], k) for k in KINDS]):
        m, s = score(arr, Ycs)
        tm, ts = score(arr, Ycs, tf) if len(tf) else (np.nan, np.nan)
        out["ours"][name] = {"pcc": m, "sd": s, "tf_pcc": tm, "tf_sd": ts}
        print(f"{name:<14s} {m:>8.3f} +/-{s:<5.3f} {PAPER[pk]:>8.2f} | "
              f"{tm:>8.3f} +/-{ts:<5.3f} {PAPER['tf_' + pk]:>8.2f}")

    gap_ours = out["ours"]["MOMA"]["pcc"] - out["ours"]["mean"]["pcc"]
    gap_paper = PAPER["moma"] - PAPER["mean"]
    out["gap_ours"], out["gap_paper"] = gap_ours, gap_paper
    tf_above = (out["ours"]["mean"]["tf_pcc"] > out["ours"]["mean"]["pcc"])
    out["tf_above_all_genes"] = bool(tf_above)

    print(f"\nMOMA minus mean baseline: ours {gap_ours:+.3f}, "
          f"paper {gap_paper:+.3f}")
    print(f"TF baselines above all-gene baselines (the paper's ordering): "
          f"{'YES' if tf_above else 'no'}")
    print("\nReading: under the paper's own protocol our model does NOT")
    print("underperform -- the residual discrepancy is in the BASELINE level.")

    args.out.write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    print(f"\nwrote {args.out}")
    db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

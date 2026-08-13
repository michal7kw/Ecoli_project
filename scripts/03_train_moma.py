#!/usr/bin/env python
"""Train and evaluate MOMA's transcriptome module under LOCO cross-validation.

    python scripts/03_train_moma.py                      # 5-fold grouped LOCO
    python scripts/03_train_moma.py --folds 0            # true leave-one-condition-out
    python scripts/03_train_moma.py --epochs 40 --depth 2
    python scripts/03_train_moma.py --depth-sweep        # sweep depths 1,2,3,4
    python scripts/03_train_moma.py --depth-sweep 1,2,3,4,5,6,8 --out results/depth_sweep_loco.json

Conditions, not profiles, are held out: Ecomics averages 6.0 replicate profiles
per condition, so a random split lets a model score well by recognising a
condition's own replicates. See ecomics/evaluate.py.

Writes two things: the metrics to `--out`, and, for a single-depth run, the
out-of-fold predictions to `results/transcriptome_predictions.npz`. The second
is what lets `08_methods_faithful_eval.py` re-score this run on the paper's own
axis without refitting the model.

ALWAYS pass `--out` with `--depth-sweep`. It defaults to
`results/transcriptome_loco.json`, and a sweep writes one block per depth -- so
running a sweep bare replaces the single-depth result of record with a
differently-shaped file. The prediction cache is safe either way: the npz is
only written when exactly one depth was requested.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ecomics import config as C                     # noqa: E402
from ecomics.db.api import Ecomics                  # noqa: E402
from ecomics.evaluate import run_loco, summarize    # noqa: E402
from ecomics.features import build_encoder          # noqa: E402
from ecomics.moma.transcriptome import TranscriptomeModule  # noqa: E402


_ARRAY_KEYS = ("pcc_per_sample", "pcc_per_molecule")


def _scalars(m: dict) -> dict:
    """Drop the per-element arrays so the metrics dict is JSON-serializable."""
    return {k: v for k, v in m.items() if k not in _ARRAY_KEYS}


def load(db: Ecomics, medium_kind: str = "present"):
    """Load the transcriptome layer and its encoded conditions.

    `medium_kind` is exposed because the encoder default changed
    (240-wide medium -> the paper's 120) and the change cost real accuracy:
    PCC/molecule 0.286 -> 0.188 under otherwise identical hyperparameters.
    Reproducing that comparison, or separating it from the simultaneous stress
    change, needs the flag rather than an edit.
    """
    enc = build_encoder(db, medium_kind=medium_kind)
    T = db.matrix("transcriptome")
    X = enc.transform(T.condition_keys)
    Y = T.values
    is_wt = np.array([k.rsplit(".", 1)[-1] == "none" for k in T.condition_keys])
    return enc, T, X, Y, is_wt


def tf_indices(db: Ecomics, columns: list[str]) -> np.ndarray:
    """Column indices of transcription factors, for the paper's TF subset.

    The paper reports TF performance separately (PCC 0.68 vs 0.54 for all
    genes): TFs sit at the input end of the regulatory hierarchy, so the
    condition reaches them almost directly.

    Delegates to `paper_protocol.tf_indices`, which reads the paper's own
    179-regulator list from Supplementary Data 2. This used to be a third copy
    of a RegulonDB-scraping implementation; all three broke together when that
    scrape was removed, which is the argument for there being
    one. `db` is unused and kept for signature stability.
    """
    from ecomics.paper_protocol import tf_indices as _tf
    return _tf(columns)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folds", type=int, default=5,
                    help="grouped folds; 0 = true leave-one-condition-out. "
                         "5 is what every published run here used -- the "
                         "default was 10, so the '5-fold LOCO' reported in the "
                         "documentation was only reproducible by knowing to "
                         "pass it.")
    ap.add_argument("--epochs", type=int, default=600)
    ap.add_argument("--depth", type=int, default=C.PAPER["memory_depth"])
    ap.add_argument("--l1", type=float, default=0.0)
    ap.add_argument("--rank", type=int, default=64,
                    help="low-rank factorization of w_y; 0 = full matrix")
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--depth-sweep", nargs="?", const="1,2,3,4", default=None,
                    metavar="D,D,...",
                    help="sweep memory depth (slow). Bare flag sweeps 1,2,3,4; "
                         "the published curve also needs 5,6,8, which used to "
                         "require three separate --depth invocations that "
                         "nothing recorded. Example: --depth-sweep 1,2,3,4,5,6,8")
    ap.add_argument("--wy-weight-decay", type=float, default=0.0,
                    help="weight decay on w_y alone; 0 keeps the recurrence "
                         "alive")
    ap.add_argument("--device", default="cpu",
                    help="torch device: cpu | cuda | cuda:N")
    ap.add_argument("--medium-kind", default="present",
                    choices=("present", "amount", "both"),
                    help="medium encoding. 'present' (default, 603 features) is "
                         "the paper's 120-wide block; 'both' (746) doubles it to "
                         "presence AND amount. The default changed from 'both' "
                         "and it cost PCC/molecule 0.286 -> 0.188 "
                         "-- use this flag to reproduce or ablate that")
    ap.add_argument("--out", type=Path, default=C.RESULTS / "transcriptome_loco.json")
    args = ap.parse_args()

    if args.device.startswith("cuda"):
        import torch
        if not torch.cuda.is_available():
            print(f"error: --device {args.device} requested but torch reports no CUDA "
                  f"(torch {torch.__version__}). Install a CUDA build or use --device cpu.",
                  file=sys.stderr)
            return 1
        print(f"device: {args.device} "
              f"({torch.cuda.get_device_name(torch.device(args.device))})")

    db = Ecomics()
    enc, T, X, Y, is_wt = load(db, medium_kind=args.medium_kind)
    print(f"transcriptome: {Y.shape[0]} profiles x {Y.shape[1]} genes, "
          f"{len(set(T.condition_keys))} conditions")
    print(f"input encoding: {X.shape[1]} features")
    print(f"wild-type profiles: {int(is_wt.sum())} ({is_wt.mean():.1%})")

    tf_idx = tf_indices(db, T.columns)
    print(f"transcription factors identified: {len(tf_idx)}")

    n_folds = None if args.folds == 0 else args.folds
    depths = ([int(d) for d in args.depth_sweep.split(",")]
              if args.depth_sweep else [args.depth])
    results = {}

    for depth in depths:
        print(f"\n{'='*70}\nmemory depth {depth}  "
              f"({'LOCO' if n_folds is None else f'{n_folds}-fold grouped'})\n{'='*70}")
        mod = TranscriptomeModule(memory_depth=depth, epochs=args.epochs,
                                  l1=args.l1, lr=args.lr,
                                  rank=args.rank or None,
                                  weight_decay=args.weight_decay,
                                  wy_weight_decay=args.wy_weight_decay,
                                  device=args.device)
        t0 = time.time()
        res = run_loco(
            # keys_tr is forwarded so the model's inner early-stopping split
            # can hold out whole CONDITIONS; run_loco detects the parameter.
            lambda xt, yt, xe, keys_tr=None: mod.fit_predict(
                xt, yt, xe, keys_tr=keys_tr),
            X, Y, T.condition_keys, is_wt, n_folds=n_folds,
        )
        elapsed = time.time() - t0
        print(f"\n  all {Y.shape[1]} genes  ({elapsed:.0f}s)")
        print(summarize(res, "MOMA transcriptome",
                        paper=C.PAPER["pcc_transcriptome_all"]))

        entry = {
            "memory_depth": depth,
            # The full run configuration travels WITH the numbers. Without it,
            # "5-fold LOCO" was recoverable only from a gitignored *.log, so the
            # headline PCC could not be attributed to a run from the tracked
            # record -- which is exactly what this repo's own "a number needs a
            # provenance" rule forbids.
            "config": {**{k: (str(v) if isinstance(v, Path) else v)
                          for k, v in vars(args).items()},
                       "n_folds_effective": "LOCO" if n_folds is None else n_folds,
                       "n_profiles": int(Y.shape[0]),
                       "n_genes": int(Y.shape[1]),
                       "n_conditions": int(len(set(T.condition_keys))),
                       "n_features": int(X.shape[1]),
                       "wildtype_definition": "gp == none (broad, not the paper's)",
                       "n_wildtype": int(is_wt.sum())},
            "all_genes": _scalars(res.metrics),
            "baselines": {b: _scalars(m) for b, m in res.baseline_metrics.items()},
            "p_values": res.p_values,
            "seconds": elapsed,
        }

        if len(tf_idx):
            from ecomics.evaluate import evaluate_predictions
            tf_m = evaluate_predictions(res.y_pred[:, tf_idx], res.y_true[:, tf_idx])
            print(f"\n  {len(tf_idx)} transcription factors: "
                  f"PCC={tf_m['pcc_mean']:.3f} +/- {tf_m['pcc_sd']:.3f}   "
                  f"(paper: {C.PAPER['pcc_transcriptome_tf'][0]} +/- "
                  f"{C.PAPER['pcc_transcriptome_tf'][1]})")
            entry["tfs"] = _scalars(tf_m)
        results[f"depth_{depth}"] = entry

        # Write after EVERY depth, not once at the end. A sweep is hours long
        # (depths 5,6,8 alone are ~1.5 h on a 3090) and the end-only write meant
        # an interruption discarded every completed depth with it -- which is
        # how one such run left no trace at all, not even a
        # partial curve. Each depth is independent, so a truncated file is a
        # shorter curve rather than a corrupt one.
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2, default=float),
                            encoding="utf-8")

        # Cache the out-of-fold predictions themselves, not just the metrics
        # summarized above. `08_methods_faithful_eval.py` re-scores these SAME
        # predictions on the paper's axis (per condition, against the
        # condition-averaged truth, per-gene min-max), and `run_loco` has
        # already computed everything it needs. Without this the paper-axis
        # evaluation costs a second full LOCO fit of an identical model --
        # which is precisely what `07_baseline_calibration.py:refit` was doing.
        #
        # The baselines travel too: `figures/fig_transcriptome.py` reads
        # `baseline_mean` / `baseline_wildtype`, so a file missing them would
        # break two figures. (08 does NOT read them -- it recomputes all three
        # under the paper's protocol, which is the whole point of that script.)
        #
        # Single-depth runs only. Under `--depth-sweep` there is no one
        # prediction set, and quietly keeping the last depth's is how
        # `fig_transcriptome.py:NPZ_RUN` came to need a paragraph explaining
        # which run its npz actually held. `source_json` puts that provenance
        # inside the file so the question cannot arise again.
        if len(depths) == 1:
            cache = C.RESULTS / "transcriptome_predictions.npz"
            np.savez_compressed(
                cache, y_true=res.y_true, y_pred=res.y_pred,
                condition_keys=res.condition_keys,
                source_json=str(args.out.name),
                **{f"baseline_{k}": v for k, v in res.baseline_pred.items()})
            print(f"  cached out-of-fold predictions -> {cache}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2, default=float),
                        encoding="utf-8")
    print(f"\nwrote {args.out}")
    db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

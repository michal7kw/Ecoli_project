#!/usr/bin/env python
"""Evaluate every MOMA layer and write results/reproduction_table.md.

    python scripts/04_reproduce.py                 # proteome, metabolome, fluxome, phenome
    python scripts/04_reproduce.py --all           # ...and fail if the transcriptome is missing

The transcriptome module is NOT trained here -- it takes ~39 min on CPU and
lives in scripts/03_train_moma.py. This script reads the results that 03 wrote
to results/transcriptome_loco.json; `--all` turns their absence from a warning
into an error, so the table cannot silently ship without those rows.

Every layer is evaluated by leave-one-condition-out cross-validation against
the paper's three baselines, and reported with PCC (the paper's metric) plus
RMSE and calibration slope (which PCC cannot see).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Windows consoles and redirected stdout default to cp1252, which cannot encode
# the marks this repo uses in reports. Without this, a print of "⚠" raises
# UnicodeEncodeError *after* every layer has been fitted -- the results are
# computed and then thrown away, because `main()` writes results/all_layers.json
# only once every eval_* has returned. That is not hypothetical: it discarded a
# complete run once, at the first non-ASCII character this script had ever
# printed. This is the script that most needs the guard, because it prints last.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ecomics import config as C                          # noqa: E402
from ecomics.db.api import Ecomics                       # noqa: E402
from ecomics.evaluate import (                           # noqa: E402
    evaluate_predictions, loco_splits, out_of_fold_baselines,
)
from ecomics.features import build_encoder               # noqa: E402
from ecomics.metrics import (                            # noqa: E402
    MIN_CONDITIONS_FOR_COLUMN_PCC, wilcoxon,
)
from ecomics.moma.metabolome import (                    # noqa: E402
    MetabolomeModule, load_metabolite_enzymes,
)
from ecomics.moma.phenome import PhenomeConsensus        # noqa: E402
from ecomics.moma.proteome import own_mrna_baseline   # noqa: E402
from ecomics.moma.proteome_paper import (                # noqa: E402
    PaperProteomeEnsemble, build_paper_ensemble_networks,
)
# The table is written by ecomics/reporting.py, not here. It is reporting rather
# than evaluation -- it reads every layer and writes prose -- and it moved out so
# that it could be tested without a 25-minute run of the layers it describes.
# `_pcc_str` went with it because the table and the console print the same
# suppressed cell, and having two copies of that wording is how they drift.
from ecomics.reporting import _pcc_str, write_table      # noqa: E402


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
#
# It moved to `metrics.py` when `ecomics/plots.py` had to draw the suppressed
# cells too -- see the comment there. This script remains the only caller that
# raises the threshold above the default of 1.


def _scalars(m: dict) -> dict:
    """Drop per-element arrays so a metrics dict is JSON-serializable."""
    return {k: v for k, v in m.items() if k not in _ARRAY_KEYS}


# --------------------------------------------------------------------------
def eval_proteome(db: Ecomics, results: dict) -> None:
    """The neighbour-LASSO ensemble, on the paper's own networks.

    Six arms from Supplementary Data 2 -- TRN, PPI, KEGG, sigma factor, small
    RNA -- plus a CPN built here from HELD-OUT conditions. Before that
    three of those graphs were instead scraped from live 2026 databases; that
    path was worth 0.113 LESS per profile and has been deleted, with its last
    run kept as `results/all_layers_scraped_superseded.json`.

    Data 2 ships the paper's own CPN too, and this does not use it: it spans
    the whole compendium, so it contains every LOCO test fold.
    `networks_paper.load_paper_networks` will not return it unless asked, and
    this is not the place to ask.
    """
    print("\n" + "=" * 74)
    print("PROTEOME  -- neighbour-LASSO ensemble, LOCO, "
          "the paper's own networks (Data 2)")
    print("=" * 74)
    shared, mats = db.aligned("transcriptome", "proteome")
    Xt, tcols, keys = mats["transcriptome"]
    Yp, pcols, _ = mats["proteome"]
    print(f"  {len(shared)} shared conditions (paper: 5 conditions / 18 profiles)")
    if len(shared) < 3:
        print("  too few to cross-validate")
        return

    P_all = db.matrix("proteome")
    held = ~np.isin(P_all.condition_keys, shared)
    print(f"  CPN built from {int(held.sum())} proteome conditions held OUT "
          "of evaluation")
    nets, reports = build_paper_ensemble_networks(
        cpn_values=P_all.values[held], cpn_columns=P_all.columns,
        include_leaky_cpn=False, verbose=False)
    dropped = sum(r.dropped for r in reports.values())
    print(f"  {len(nets)} arms from Supplementary Data 2 + our CPN; "
          f"{dropped} rows dropped to unmappable tokens")
    for n, net in nets.items():
        print(f"    {n:<8s} {net.n_nodes:>5d} nodes  {net.n_edges:>7d} edges")

    preds = {n: np.full_like(Yp, np.nan) for n in list(nets) + ["ENSEMBLE"]}
    base = np.full_like(Yp, np.nan)
    for tr, te in loco_splits(keys):
        ens = PaperProteomeEnsemble(nets).fit(Xt[tr], tcols, Yp[tr], pcols)
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
    print("  SRNA covers a handful of proteins by design -- read every arm "
          "against its coverage column, never its PCC alone")
    # The source travels with the numbers: a proteome block WITHOUT this key is
    # a run on the scraped graphs, whatever its filename says.
    # It is written unconditionally now that there is only one source, because
    # its absence is the signal, not its value.
    rows["network_source"] = "supplementary_data_2"
    results["proteome"] = rows


def eval_metabolome(db: Ecomics, results: dict) -> None:
    print("\n" + "=" * 74)
    print("METABOLOME -- core from proteins, non-core from transcripts, LOCO")
    print("=" * 74)
    rows = {}

    # core: proteins -> metabolites, on the P & M conditions.
    #
    # Run TWICE, and the second arm is the point. §3.3.5 says "for metabolites
    # having known enzyme-substrate relations, we predict its concentrations
    # from the mRNA expression levels of the RELATED ENZYMES" -- prior knowledge
    # in place of L1 selection. `enzymes` is that map, from Supplementary Data
    # 1's `Metabolite` sheet: 67 of our 114 metabolites and ALL 10 core ones,
    # with a median of 4 measured enzymes each against 589 columns unrestricted.
    #
    # Data 2's `substrate-enzyme` sheet is the one the Methods point at more
    # directly and reaches only 3 of the 114 -- `load_substrate_enzyme` is kept
    # so that 23x disagreement stays re-measurable.
    shared, mats = db.aligned("proteome", "metabolome")
    print(f"  proteome & metabolome: {len(shared)} conditions (core training set)")
    if len(shared) >= 4:
        Xp, pcol, keys = mats["proteome"]
        Ym, mcols, _ = mats["metabolome"]
        core, noncore = MetabolomeModule.split_core(mcols)
        print(f"  core metabolites present: {len(core)}; non-core: {len(noncore)}")

        enzymes = load_metabolite_enzymes()
        restricted = [c for c in mcols
                      if set(enzymes.get(c, ())) & set(pcol)]
        print(f"  enzyme-substrate map covers {len(restricted)}/{len(mcols)} "
              f"metabolites ({len([c for c in core if c in restricted])}/"
              f"{len(core)} core)")

        preds = {}
        for arm, emap in (("all_features", None), ("enzyme_features", enzymes)):
            pred = np.full_like(Ym, np.nan)
            for tr, te in loco_splits(keys):
                mod = MetabolomeModule().fit(
                    Xp[tr], Xp[tr], Ym[tr], mcols,
                    protein_cols=pcol, transcript_cols=pcol, enzyme_map=emap)
                pred[te] = mod.predict(Xp[te], Xp[te], mcols)
            preds[arm] = pred

        ci = [mcols.index(c) for c in core] if core else []
        ni = [mcols.index(c) for c in noncore]
        for arm, pred in preds.items():
            # `core_from_protein` keeps its name on the unrestricted arm: it is
            # the number of record and every pinned figure reads it.
            suffix = "" if arm == "all_features" else "_enzyme_features"
            if ci:
                m = evaluate_predictions(pred[:, ci], Ym[:, ci],
                                         MIN_CONDITIONS_FOR_COLUMN_PCC,
                                         n_effective=len(shared))
                rows[f"core_from_protein{suffix}"] = _scalars(m)
                print(f"  core (from proteins, {arm:<15s}): per-molecule "
                      f"{_pcc_str(m)}  per-profile {m['pcc_row_mean']:.3f}"
                      + ("   (paper: 0.65 +/- 0.21)" if not suffix else ""))
            if ni:
                m = evaluate_predictions(pred[:, ni], Ym[:, ni],
                                         MIN_CONDITIONS_FOR_COLUMN_PCC,
                                         n_effective=len(shared))
                rows[f"noncore_from_protein{suffix}"] = _scalars(m)
        rows["enzyme_map_coverage"] = {
            "metabolites": len(restricted), "of": len(mcols),
            "core": len([c for c in core if c in restricted]), "core_of": len(core),
            "source": "supplementary_data_1:Metabolite",
        }

    # non-core: transcripts -> metabolites, on the T & M conditions
    shared, mats = db.aligned("transcriptome", "metabolome")
    print(f"\n  transcriptome & metabolome: {len(shared)} conditions")
    if len(shared) >= 4:
        Xt, _tc, keys = mats["transcriptome"]
        Ym, mcols, _ = mats["metabolome"]
        pred = np.full_like(Ym, np.nan)
        for tr, te in loco_splits(keys):
            mod = MetabolomeModule().fit(None, Xt[tr], Ym[tr], mcols)
            pred[te] = mod.predict(None, Xt[te], mcols)
        core, noncore = MetabolomeModule.split_core(mcols)
        ni = [mcols.index(c) for c in noncore]
        if ni:
            m = evaluate_predictions(pred[:, ni], Ym[:, ni],
                                     MIN_CONDITIONS_FOR_COLUMN_PCC,
                                     n_effective=len(shared))
            rows["noncore_from_transcript"] = _scalars(m)
            print(f"  non-core (from transcripts): per-molecule {_pcc_str(m)}  "
                  f"per-profile {m['pcc_row_mean']:.3f}   (paper: 0.87 +/- 0.15)")
        ci = [mcols.index(c) for c in core]
        if ci:
            m = evaluate_predictions(pred[:, ci], Ym[:, ci],
                                     MIN_CONDITIONS_FOR_COLUMN_PCC,
                                     n_effective=len(shared))
            rows["core_from_transcript"] = _scalars(m)
            print(f"  core (from transcripts):     per-molecule {_pcc_str(m)}  "
                  f"per-profile {m['pcc_row_mean']:.3f}   (paper: 0.47 +/- 0.26)")
    results["metabolome"] = rows


def eval_fluxome(db: Ecomics, results: dict) -> None:
    """FBA with omics-derived bounds vs plain FBA (paper Fig. 5e).

    Ablation, following the paper: plain FBA, then + medium-derived exchange
    bounds (the input layer), then + expression-thresholded lower bounds.

    Two limits, both properties of the published data rather than of the method:
      * only 22 of 120 Ecomics reactions carry a BiGG cross-reference that also
        exists in iJO1366, so 21 scoreable reactions (one has zero variance);
      * one flux profile is an all-zero placeholder and is excluded.
    """
    from ecomics.moma.fluxome import (
        GenomeScaleModel, FluxomeModule, load_paper_medium_bounds,
    )

    print("\n" + "=" * 74)
    print("FLUXOME -- FBA on iJO1366 with omics-derived bounds")
    print("=" * 74)

    F = db.matrix("fluxome")
    keep = ~(F.values == 0).all(axis=1)
    if (~keep).any():
        print(f"  excluding {int((~keep).sum())} all-zero placeholder profile(s)")
    Y = F.values[keep]
    keys = F.condition_keys[keep]

    gsm = GenomeScaleModel.from_bigg()
    fm = FluxomeModule(gsm)
    mapped = fm.mappable_reactions(db.reaction_bigg())
    cols = [(F.columns.index(r), j) for r, j in mapped.items() if r in F.columns]
    if len(cols) < 5:
        print("  too few mappable reactions to evaluate")
        return
    ec, mc = [c[0] for c in cols], [c[1] for c in cols]
    print(f"  {Y.shape[0]} profiles, {len(cols)} reactions mappable into iJO1366")

    # Supplementary Data 4: the authors' own exchange bounds per medium, and the
    # only medium source. REQUIRED -- if it is absent this raises, with the
    # "Run: python scripts/00_acquire.py" message load_paper_medium_bounds
    # carries, rather than quietly evaluating a different specification.
    paper_bounds = load_paper_medium_bounds()
    flux_media = sorted({k.split(".")[1] for k in keys})
    from_data4 = [m for m in flux_media if m in paper_bounds]
    unlisted = [m for m in flux_media if m not in paper_bounds]
    print(f"  medium bounds: {len(from_data4)}/{len(flux_media)} flux media from "
          f"Supplementary Data 4 ({', '.join(from_data4) or 'none'})")
    if unlisted:
        # Reported, not substituted. These keep iJO1366's own default bounds --
        # which is what the hand-curated MEDIUM_TO_BIGG table used to write back
        # anyway: every flux medium is glucose-minimal, EX_glc__D_e defaults to
        # -10 and every other carbon exchange to 0, so the curated path changed
        # nothing on any of them. It has been removed.
        print(f"  not in Data 4, so left at iJO1366 defaults: "
              f"{', '.join(unlisted)}")

    T = db.matrix("transcriptome").averaged_by_condition()
    tmap = {k: i for i, k in enumerate(T.condition_keys)}
    mean_expr = np.nanmean(T.values, axis=0)

    def expression_for(key):
        """Measured transcriptome for this condition, else the compendium mean.

        Only 3 flux conditions have a measured transcriptome, so most fall back
        to the mean. That makes the expression constraint weak here -- reported
        rather than hidden.
        """
        i = tmap.get(key)
        v = T.values[i] if i is not None else mean_expr
        return {g: float(x) for g, x in zip(T.columns, v) if np.isfinite(x)}, i is not None

    def run(use_medium: bool, use_expression: bool, threshold: float,
            want_biomass: bool = False):
        """Predicted fluxes, optionally with the biomass each solution carried.

        `want_biomass` exists because a threshold can produce a beautifully
        correlated prediction from a model that is not growing -- see the
        viability guard on the sweep below.
        """
        pred = np.full((len(keys), len(cols)), np.nan)
        biomass = np.full(len(keys), np.nan)
        for i, key in enumerate(keys):
            mid = key.split(".")[1]
            expr, _ = expression_for(key) if use_expression else ({}, False)
            fm.threshold = threshold
            lb, ub = fm.bounds_from_expression(expr)
            if use_medium:
                # The authors' OWN exchange bounds, keyed by the same MD id the
                # condition key already carries, so this join involves no name
                # matching at all. A medium Data 4 does not list is a no-op here
                # and keeps iJO1366's defaults -- reported above, not swapped for
                # a guess, because a medium we cannot constrain should look
                # unconstrained rather than look like a different medium.
                #
                # This call site once used the hand-curated MEDIUM_TO_BIGG path
                # alone while docs/diagrams/05 5.4 and 13.5 both described Data 4
                # as the medium source, so the reported numbers took the path the
                # documentation said had been superseded (17 section 3.10). That
                # table has since been removed outright.
                lb = fm.apply_paper_bounds(lb, mid, paper_bounds)
            # MTF, not plain solve. Supplementary Methods 3.3.6 asks for "the
            # flux distribution that minimizes total the total absolute flux
            # (MTF) with the same objective value", and until it existed this
            # took whichever optimal vertex HiGHS landed on. On iJO1366 that
            # vertex carries 16,701 units of total flux against MTF's 699 --
            # the excess is thermodynamically infeasible internal loops, which
            # the measured 13C fluxes do not contain. Removing them moves
            # PCC/profile 0.745 -> 0.843 and RMSE 39.0 -> 28.6.
            v, obj = gsm.solve_mtf(lb, ub)
            if v is None:
                continue
            biomass[i] = obj
            # Measured fluxes are normalized to glucose uptake = 100, so the
            # predicted vector must be put on the same scale before comparison.
            gj = gsm.index().get("EX_glc__D_e")
            scale = abs(v[gj]) if gj is not None and abs(v[gj]) > 1e-9 else None
            pred[i] = (v[mc] / scale * 100.0) if scale else v[mc]
        return (pred, biomass) if want_biomass else pred

    n_measured = sum(1 for k in keys if k in tmap)
    n_media = len({k.split(".")[1] for k in keys})
    print(f"  flux conditions with a measured transcriptome: {n_measured}/{len(keys)}"
          f"  (the rest fall back to the compendium mean)")
    print(f"  distinct media across those profiles: {n_media}")
    print("\n  CAVEAT: all of these media are glucose-minimal and differ only in")
    print("  trace salts. For PLAIN FBA there is no medium at all, so the solution")
    print("  is identical for every profile, a constant has zero variance per")
    print("  reaction, and per-reaction PCC is undefined (NaN) -- not poor,")
    print("  undefined.")
    print("  Once Supplementary Data 4 supplies MD004's own bounds, the medium-only")
    print("  configuration is NO LONGER constant across profiles -- MD004 differs")
    print("  from the three media Data 4 does not list -- so per-reaction PCC")
    print("  becomes defined for it. It is defined on 4 media, which is thin.")

    truth = Y[:, ec]
    rows = {}
    metrics_by_name: dict[str, dict] = {}
    configs = [("plain FBA", False, False), ("+ medium (input layer)", True, False),
               ("+ medium + expression", True, True)]
    print(f"\n  {'configuration':<26s} {'PCC/reaction':>14s} "
          f"{'PCC/profile':>13s} {'solved':>8s}")
    for name, um, ue in configs:
        # 0.04 is the paper's own choice; this call site passed 0.1, which is
        # the value the paper's sweep range was expressly chosen to stay BELOW
        # ("determined to be below 0.1 as the mean expression level of genes
        # was 0.10"). At 0.1 biomass collapses to zero for every condition, so
        # the "+ medium + expression" row reported until now was computed from
        # a model that was not growing.
        pred = run(um, ue, threshold=0.04)
        m = evaluate_predictions(pred, truth)
        rows[name] = _scalars(m)
        metrics_by_name[name] = m
        solved = int(np.isfinite(pred).any(axis=1).sum())
        print(f"  {name:<26s} {m['pcc_mean']:>8.3f} +/-{m['pcc_sd']:<4.2f} "
              f"{m['pcc_row_mean']:>13.3f} {solved:>6d}/{len(keys)}")

    # ---------------------------------------------------- the paper's baselines
    # ⚠ THIS BLOCK IS WHY THE NUMBERS ABOVE ARE INTERPRETABLE, and it did not
    # exist. The fluxome was the ONLY layer evaluated without
    # baselines, because it is the only one that does not go through `run_loco`
    # -- FBA needs no training data, so there is no fit to hold out and the
    # harness that carries the baselines with it was never invoked. The cost of
    # that omission was large: `pcc_row_mean` 0.843 reads as comfortably beating
    # the paper's 0.72 until you ask what a CONSTANT scores on the same 22
    # reactions, which is 0.902. FBA is BELOW the mean baseline here.
    #
    # The paper reports exactly this number for exactly this figure -- Fig. 5e,
    # "mean baseline (PCC of 0.50 +/- 0.11, P < 10^-8)" -- so the comparison is
    # the paper's own, not an extra bar invented here. Its FBA clears its mean
    # baseline by +0.22; ours trails ours by -0.06.
    #
    # Held out per CONDITION, as everywhere else in this repo -- the mechanics
    # live in `evaluate.out_of_fold_baselines`, which is also where the reason
    # this layer needs its own call is written down: FBA has no fit to hold out,
    # so it never enters `run_loco` and never inherits the baselines from it.
    #
    # ⚠ The wild-type baseline is thin here and should be read as such: only
    # 3 of 31 flux conditions are unperturbed, so it is a mean over ~4 profiles.
    # `evaluate.py`'s module docstring records the broader caveat on
    # `is_wildtype` -- it means "no genetic perturbation", not the Methods'
    # much narrower MG1655/LB-or-M9 definition.
    is_wt = np.array([str(k).split(".")[3] == "none" for k in keys])
    base_pred = out_of_fold_baselines(truth, is_wt, keys)

    headline = "+ medium (input layer)"
    hm = metrics_by_name[headline]
    print(f"\n  the paper's three baselines, held out per condition"
          f"   (Fig. 5e reports mean = 0.50 +/- 0.11)")
    print(f"  {'baseline':<26s} {'PCC/reaction':>14s} {'PCC/profile':>13s} "
          f"{'margin':>8s} {'p':>10s}")
    base_rows = {}
    for kind, bp in base_pred.items():
        bm = evaluate_predictions(bp, truth)
        p = wilcoxon(hm["pcc_per_sample"], bm["pcc_per_sample"])
        margin = hm["pcc_row_mean"] - bm["pcc_row_mean"]
        base_rows[kind] = _scalars(bm) | {"margin_vs_headline_per_profile": float(margin),
                                          "wilcoxon_p_per_profile": float(p)}
        print(f"  {kind:<26s} {bm['pcc_mean']:>8.3f} +/-{bm['pcc_sd']:<4.2f} "
              f"{bm['pcc_row_mean']:>13.3f} {margin:>+8.3f} {p:>10.2e}")
    base_rows["_note"] = (
        "Wilcoxon and margin are on the PER-PROFILE axis, because that is the "
        "axis Fig. 5e reports ('evaluated over 32 different conditions', mean "
        "baseline 0.50). run_loco tests on the per-molecule axis instead; this "
        "layer differs deliberately, and the axis is named in the key.")
    rows["baselines"] = base_rows
    print(f"\n  ⚠ FBA does NOT clear the mean baseline on this data: "
          f"{hm['pcc_row_mean']:.3f} against "
          f"{evaluate_predictions(base_pred['mean'], truth)['pcc_row_mean']:.3f}.")
    print("  All four flux media are glucose-minimal and the 22 scoreable")
    print("  reactions are the BiGG-cross-referenced core (glycolysis, TCA), so")
    print("  every measured profile is nearly the same profile and a constant")
    print("  correlates at ~0.9. The paper's FBA clears ITS mean baseline by")
    print("  +0.22 over 120 fluxes; ours trails by -0.06 over 22.")

    # The paper tunes the expression threshold t empirically, over
    # {0.02, 0.04, 0.06, 0.08}. Its own range, not a wider one: the point of
    # the sweep is to reproduce the paper's choice, and a grid that reaches 0.4
    # only explores territory the paper explicitly excluded.
    #
    # Expect this to find nothing. Gene mean levels here run 0.058-0.407 about
    # a median of 0.094, so t <= 0.06 changes NO bounds (the prediction is then
    # constant across profiles, and a constant has no variance to correlate,
    # so PCC/reaction is NaN) while t = 0.08 collapses biomass to zero. There is
    # no value in the paper's range that both fires and leaves the model alive.
    # `best_t = None` is the honest answer and is recorded as such.
    # ⚠ THE SELECTION CRITERION NEEDS A VIABILITY GUARD, and this is not
    # hypothetical. Maximizing per-reaction PCC alone picks t = 0.08 with a PCC
    # of 0.312 -- at which ALL 42 profiles have zero biomass. A dead model still
    # returns *a* degenerate vertex, different profiles land on different ones,
    # and correlating those against measured flux yields a respectable-looking
    # number from pure artefact. The criterion rewards the broken regime,
    # because that is where spurious across-profile variance is largest. Any t
    # that kills most of the models is disqualified before it is scored.
    ALIVE_FRACTION = 0.5
    sweep, best, best_pcc = {}, None, -np.inf
    for t in (0.02, 0.04, 0.06, 0.08):
        pred_t, biomass_t = run(True, True, t, want_biomass=True)
        m = evaluate_predictions(pred_t, truth)
        alive = int((np.nan_to_num(biomass_t) > 1e-6).sum())
        viable = alive >= ALIVE_FRACTION * len(keys)
        sweep[str(t)] = {"pcc_reaction": float(m["pcc_mean"]),
                         "pcc_profile": float(m["pcc_row_mean"]),
                         "n_alive": alive, "viable": viable}
        if viable and np.isfinite(m["pcc_mean"]) and m["pcc_mean"] > best_pcc:
            best, best_pcc = t, m["pcc_mean"]
    print(f"\n  {'t':<7}{'alive':>9}{'PCC/reaction':>14}{'PCC/profile':>13}")
    for k, v in sweep.items():
        flag = "" if v["viable"] else "   <- DISQUALIFIED, the model is dead"
        print(f"  {k:<7}{v['n_alive']:>5}/{len(keys):<3}{v['pcc_reaction']:>14.4f}"
              f"{v['pcc_profile']:>13.4f}{flag}")
    if best is None:
        print("\n  expression threshold sweep: NO viable value in the paper's "
              "range yields a defined")
        print("  per-reaction PCC. Below 0.08 the rule barely fires, so the "
              "prediction is near-constant")
        print("  and PCC/reaction is undefined; at 0.08 every model is dead. "
              "The rule cannot both")
        print("  fire and leave the model alive on this compendium.")
    else:
        print(f"\n  expression threshold sweep: best t = {best} "
              f"(PCC/reaction {best_pcc:.3f})")
    rows["threshold_sweep"] = {
        "best_t": best,
        "pcc": float(best_pcc) if np.isfinite(best_pcc) else None,
        "grid": sweep,
    }
    rows["_caveats"] = {
        "n_profiles": int(len(keys)), "n_media": n_media,
        "n_with_measured_transcriptome": int(n_measured),
        "n_scoreable_reactions": len(cols),
        "verdict": ("the layer does not clear its own mean baseline. On the "
                    "paper's axis -- per profile, which is what Fig. 5e reports "
                    "('evaluated over 32 different conditions') -- FBA scores "
                    "0.843 and a CONSTANT mean flux profile scores 0.902, a "
                    "margin of -0.06. The paper's FBA clears its mean baseline "
                    "by +0.22 (0.72 vs 0.50) over 120 fluxes. Ours is scored on "
                    "the 22 reactions with a BiGG cross-reference, which are "
                    "core glycolysis and TCA, over 4 glucose-minimal media -- so "
                    "every measured profile is nearly the same profile and a "
                    "constant already correlates at ~0.9. Per-reaction PCC is "
                    "separately near-undefined, because plain FBA's prediction "
                    "is constant across profiles; it becomes defined for the "
                    "medium-only configuration once Supplementary Data 4 supplies "
                    "MD004's own bounds, but on 4 media that is thin. The public "
                    "flux data cannot support the paper's Fig. 5e comparison on "
                    "either axis"),
    }
    print(f"\n  paper reports: FBA+omics 0.72 +/- 0.24, plain FBA 0.65 +/- 0.39,")
    print(f"  mean baseline 0.50 +/- 0.11, over 32 conditions and 120 fluxes.")
    print(f"  Ours is scored on {len(cols)} reactions over {n_media} glucose-minimal media.")
    print(f"\n  ⚠ Do NOT read PCC/profile "
          f"{rows['plain FBA']['pcc_row_mean']:.3f} as beating the paper's 0.72.")
    print(f"  It is measured against a different, easier reaction set, and it is")
    print(f"  BELOW the mean baseline on that set. The paper's own margin over its")
    print(f"  mean baseline (+0.22) is the quantity to compare, and ours is negative.")
    print(f"\n  Nor did the paper establish that OMICS constrained FBA usefully:")
    print(f"  Fig. 5e puts every layer ablation at P < 10^-1 -- 0.72 vs 0.65 vs 0.67")
    print(f"  vs 0.70 are within noise over 32 conditions. Only the mean-baseline")
    print(f"  comparison reaches P < 10^-8. (The body text at paper.md:83 quotes")
    print(f"  P < 10^-8 for the plain-FBA comparison, contradicting its own caption.)")
    results["fluxome"] = rows


def eval_phenome(db: Ecomics, results: dict,
                 medium_kind: str = "present") -> None:
    """Growth rate from a weighted consensus over the layers, under LOCO.

    `medium_kind` is exposed for the same reason scripts/03 exposes it: the
    encoder default changed (240-wide medium -> the paper's 120)
    and this layer moved the OPPOSITE way from the transcriptome -- consensus
    0.465 -> 0.603 and input layer 0.588 -> 0.620, while the transcriptome lost
    0.098. That comparison was 756-vs-626 and so conflates the medium change
    with the simultaneous stress change; `--medium-kind both` isolates it.
    """
    print("\n" + "=" * 74)
    print("PHENOME -- growth rate from a weighted layer consensus, LOCO")
    print("=" * 74)
    enc = build_encoder(db, medium_kind=medium_kind)
    # Recorded so `write_table` can DERIVE the feature-count sentence instead of
    # stating it. The literal version printed "626 features (... 296
    # perturbation)" and survived the 296 -> 273 correction untouched, because a
    # hardcoded sentence in a generated file ages while every computed number
    # beside it stays current. It also gives all_layers.json the encoder width
    # that 28 files in results/ still lack (DOC-AUDIT T15).
    results["encoder"] = {
        "medium_kind": medium_kind,
        "strain": len(enc.strain_features),
        "medium": len(enc.medium_features),
        "stress": len(enc.stress_features),
        "perturbation": len(enc.pert_features),
        "n_features": (len(enc.strain_features) + len(enc.medium_features)
                       + len(enc.stress_features) + len(enc.pert_features)),
    }
    growth = db.growth_rate()

    T = db.matrix("transcriptome").averaged_by_condition()
    keys = np.array([k for k in T.condition_keys if k in growth])
    print(f"  conditions with transcriptome AND growth rate: {len(keys)} "
          f"(paper: 179)")
    if len(keys) < 10:
        print("  too few to evaluate")
        return

    sel = np.isin(T.condition_keys, keys)
    order = np.argsort(T.condition_keys[sel])
    keys = T.condition_keys[sel][order]
    Xt = T.values[sel][order]
    y = np.array([growth[k] for k in keys])
    Xin = enc.transform(keys)

    feats = {"input": Xin, "transcriptome": Xt}
    pred = np.full(len(y), np.nan)
    per_layer = {n: np.full(len(y), np.nan) for n in feats}

    for tr, te in loco_splits(keys, n_folds=10):
        cons = PhenomeConsensus().fit({n: F[tr] for n, F in feats.items()}, y[tr])
        pred[te] = cons.predict({n: F[te] for n, F in feats.items()})
        for n, p in cons.predict_each({n: F[te] for n, F in feats.items()}).items():
            per_layer[n][te] = p

    rows = {}
    print(f"\n  {'predictor':<20s} {'PCC':>8s} {'RMSE':>9s} {'slope':>8s}")
    for n, p in per_layer.items():
        m = evaluate_predictions(p.reshape(-1, 1), y.reshape(-1, 1))
        r = np.corrcoef(p[np.isfinite(p)], y[np.isfinite(p)])[0, 1]
        rows[n] = {"pcc": float(r), "rmse": m["rmse"]}
        print(f"  {n:<20s} {r:>8.3f} {m['rmse']:>9.4f}")
    ok = np.isfinite(pred)
    r = float(np.corrcoef(pred[ok], y[ok])[0, 1])
    slope = float(np.polyfit(pred[ok], y[ok], 1)[0])
    rmse = float(np.sqrt(np.mean((pred[ok] - y[ok]) ** 2)))
    rows["consensus"] = {"pcc": r, "rmse": rmse, "calibration_slope": slope}
    print(f"  {'CONSENSUS':<20s} {r:>8.3f} {rmse:>9.4f} {slope:>8.3f}   "
          f"(paper: 0.65 +/- 0.01)")
    print(f"\n  calibration slope {slope:.2f}: "
          f"{'predictions compressed toward the mean' if slope > 1.2 else 'well scaled'}")
    results["phenome"] = rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true",
                    help="fail instead of continuing if the saved transcriptome "
                         "results from scripts/03 are missing")
    ap.add_argument("--out", type=Path, default=C.RESULTS / "all_layers.json",
                    help="where to write the per-layer JSON. Anything other "
                         "than the default is treated as a SIDE RUN and leaves "
                         "results/reproduction_table.md alone")
    ap.add_argument("--medium-kind", default="present",
                    choices=("present", "amount", "both"),
                    help="medium encoding for the phenome layer's condition "
                         "features. 'present' (default) is the paper's 120-wide "
                         "block; 'both' doubles it to presence AND amount. The "
                         "default changed from 'both' and the "
                         "phenome consensus rose 0.465 -> 0.603 -- use this to "
                         "isolate that from the simultaneous stress change")
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
    results: dict = {}

    # The transcriptome is NOT run here -- it takes ~39 min and lives in
    # scripts/03. This reads what 03 already wrote. `--all` used to claim it
    # would "also run the transcriptome RNN"; `args` was then never read, so the
    # flag did nothing at all, and a missing transcriptome_loco.json silently
    # dropped four rows from the table. It now controls whether that absence is
    # fatal, which is the only honest thing the flag can mean here.
    prev = C.RESULTS / "transcriptome_loco.json"
    if prev.exists():
        saved = json.loads(prev.read_text())
        key = next(iter(saved))
        results["transcriptome"] = saved[key]
        print(f"loaded saved transcriptome results ({key})")
    elif args.all:
        print(f"missing {prev}\nRun: python scripts/03_train_moma.py")
        return 1
    else:
        print(f"NOTE: {prev.name} absent -- the transcriptome rows will be "
              f"omitted from the table. Run scripts/03_train_moma.py first, "
              f"or pass --all to make this an error.")

    eval_proteome(db, results)
    eval_metabolome(db, results)
    eval_fluxome(db, results)
    eval_phenome(db, results, medium_kind=args.medium_kind)

    C.RESULTS.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2, default=float),
                        encoding="utf-8")
    print(f"\nwrote {args.out}")
    # The table describes the RUN OF RECORD, so a side experiment must not
    # rewrite it. Without --out this script overwrote both files unconditionally,
    # which made `--medium-kind both` unusable as an isolating run: measuring the
    # variant destroyed the baseline it was supposed to be compared against.
    if args.out.name == "all_layers.json":
        write_table(results, C.RESULTS / "reproduction_table.md")
        print(f"wrote {C.RESULTS / 'reproduction_table.md'}")
    else:
        print("  (side run: reproduction_table.md left untouched)")
    db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

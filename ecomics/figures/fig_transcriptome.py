"""B — The transcriptome, and the trap that shaped this whole reproduction.

The same out-of-fold predictions score **-0.10 per molecule** and **+0.58 per
profile**. For most of this reproduction the transcriptome was reported as
"ours 0.287 vs the paper's 0.54", which compared this repository's per-molecule
number against the paper's per-profile one. Nothing about the model was wrong;
the comparison was. On the paper's own axis and protocol the identical model
scores 0.544 against the paper's 0.54 (`scripts/08`).

So the first figure here is deliberately two panels of the same four
predictors, and the second is the distribution behind the mean that neither
axis shows. The quickest refutation of the old reading needs no supplement at
all and is visible in the left panel: the paper's baselines are POSITIVE
(0.25-0.36), while a condition-blind predictor scores ~0 per molecule by
construction -- so they cannot be per-molecule numbers.
"""

from __future__ import annotations

import numpy as np

from .. import config as C
from .. import plots as P
from . import figure, results_json, results_npz

PREDICTORS = ["MOMA", "random", "mean", "wildtype"]

# Which run `results/transcriptome_predictions.npz` actually holds.
#
# It IS `transcriptome_loco.json`, the headline run: recomputing from the npz
# gives 0.18626 / 15.0% above 0.3 / slope 0.9583, which is that file exactly.
#
# It used to be a different run, and the difference was a bug rather than a
# seed: `scripts/07 --refit` called `fit_predict` without `keys_tr`, so y^(0)
# fell back to the whole-fold mean instead of the paper's WT-mean -- the same
# defect audited out of `scripts/03` and never fixed here. That cache scored
# 0.29498 / 44.2% / 0.8812, which is `transcriptome_loco_recurrent.json`, and
# the figures drawn from it had to cite that run to avoid contradicting
# the results reference. Once corrected, the cache and the
# headline run are the same model.
#
# The constant stays, and so does the citation in the footer. The npz is
# gitignored and rewritten by any `scripts/03` run, so the day it stops
# matching is the day this matters again --
# `test_saved_predictions_belong_to_the_run_the_figures_cite` is what notices.
# Anything read from the npz is recomputed from the npz, never borrowed from a
# JSON.
NPZ_RUN = "transcriptome_loco"


def _loco():
    return results_json("transcriptome_loco")["depth_2"]


@figure(
    name="transcriptome-two-axes",
    title="The same predictions on both PCC axes",
    group="transcriptome",
    requires=("results/transcriptome_loco.json",),
    tier=1,
    caveat="TWO PANELS, never two y-scales. The panels are not comparable to "
           "each other and their heights must not be read across.",
)
def transcriptome_two_axes(theme=P.LIGHT):
    d = _loco()
    vals = {
        "MOMA": d["all_genes"],
        "random": d["baselines"]["random"],
        "mean": d["baselines"]["mean"],
        "wildtype": d["baselines"]["wildtype"],
    }
    fig, axes = P.panels(
        1, 2, figsize=(11.0, 4.6), theme=theme,
        title="One model, one set of predictions, two numbers ~0.7 apart",
        subtitle=f"5-fold grouped leave-one-condition-out, "
                 f"{d['config']['n_profiles']:,} profiles / "
                 f"{d['config']['n_conditions']} conditions / "
                 f"{d['config']['n_genes']:,} genes. Error bars are 1 SD "
                 "across the units being correlated.")

    xs = np.arange(len(PREDICTORS))
    colors = [theme.series[0]] + [theme.muted] * 3

    # -- left: per molecule. Baselines are NEGATIVE here, and must be.
    ax = axes[0]
    heights = [vals[k]["pcc_mean"] for k in PREDICTORS]
    sds = [vals[k]["pcc_sd"] for k in PREDICTORS]
    P.bars(ax, xs, heights, color=colors, theme=theme)
    ax.errorbar(xs, heights, yerr=sds, fmt="none", ecolor=theme.baseline,
                elinewidth=1.0, capsize=3, zorder=3)
    ax.axhline(0, color=theme.baseline, linewidth=1.0)
    P.pcc_axis(ax, P.PER_MOLECULE)
    ax.set_xticks(xs, PREDICTORS)
    ax.set_title("This repository's primary axis", loc="left")
    P.caption(ax, "a near-constant prediction has near-zero variance across "
                  "conditions, so all three baselines sit below 0", theme=theme)
    P.grid(ax, "y", theme)

    # -- right: per profile. Everything is ~0.58, including the baselines.
    ax = axes[1]
    heights = [vals[k]["pcc_row_mean"] for k in PREDICTORS]
    sds = [vals[k].get("pcc_row_sd", 0.0) for k in PREDICTORS]
    P.bars(ax, xs, heights, color=colors, theme=theme)
    ax.errorbar(xs, heights, yerr=sds, fmt="none", ecolor=theme.baseline,
                elinewidth=1.0, capsize=3, zorder=3)
    P.pcc_axis(ax, P.PER_PROFILE)
    ax.set_xticks(xs, PREDICTORS)
    ax.set_ylim(0, 0.95)
    ax.set_title("The paper's axis", loc="left")
    P.caption(ax, "dominated by the mean expression profile: the model beats "
                  "the mean baseline here, but only just", theme=theme)
    P.grid(ax, "y", theme)

    P.finish(fig, script="scripts/15_figures.py",
                 sources=["results/transcriptome_loco.json"], theme=theme,
                 note="The paper's 0.54 belongs on the RIGHT panel's axis and "
                      "under its protocol; see figure `methods-faithful`.")
    return fig


@figure(
    name="transcriptome-distribution",
    title="Per-gene PCC distribution — what the mean hides",
    group="transcriptome",
    requires=("results/transcriptome_predictions.npz",),
    tier=2,
    caveat="Recomputed from the saved out-of-fold predictions; the per-element "
           "arrays are stripped before the JSON is written "
           "(scripts/04_reproduce.py:_ARRAY_KEYS).",
)
def transcriptome_distribution(theme=P.LIGHT):
    from ..metrics import pcc_per_column
    z = results_npz("transcriptome_predictions")
    true = z["y_true"]
    series = {
        "MOMA": (z["y_pred"], theme.series[0]),
        "mean baseline": (z["baseline_mean"], theme.series[1]),
        "wild-type baseline": (z["baseline_wildtype"], theme.series[2]),
    }

    fig, axes = P.panels(
        1, 2, figsize=(11.0, 4.4), theme=theme,
        title="0.287 is not “every gene at 0.29”",
        subtitle="Per-gene PCC across 596 conditions, for all 4,096 genes. "
                 "A mean is consistent with a uniform smear or with a subset "
                 "of genes tracked well and the rest not; these are the "
                 "latter.")

    curves = {}
    for name, (pred, color) in series.items():
        r = pcc_per_column(pred, true)
        curves[name] = (r[np.isfinite(r)], color)

    ax = axes[0]
    for name, (r, color) in curves.items():
        ax.hist(r, bins=60, histtype="step", linewidth=1.8, color=color,
                label=name)
    ax.axvline(0.3, color=theme.muted, linewidth=1.0)
    ax.text(0.3, 0.98, " 0.3", transform=ax.get_xaxis_transform(), ha="left",
            va="top", fontsize=7.5, color=theme.muted)
    P.pcc_axis(ax, P.PER_MOLECULE, on="x")
    ax.set_ylabel("genes")
    ax.set_title("Distribution", loc="left")
    ax.legend(loc="upper left")
    P.grid(ax, "y", theme)

    ax = axes[1]
    for name, (r, color) in curves.items():
        xs = np.sort(r)
        ax.plot(xs, np.linspace(0, 1, xs.size), color=color, label=name)
    frac = float((curves["MOMA"][0] > 0.3).mean())
    ax.axvline(0.3, color=theme.muted, linewidth=1.0)
    ax.plot([0.3], [1 - frac], marker="o", markersize=7,
            color=theme.series[0], markeredgecolor=theme.surface,
            markeredgewidth=2, zorder=5)
    ax.annotate(f"{frac:.1%} of genes above 0.3", (0.3, 1 - frac),
                textcoords="offset points", xytext=(12, -4), fontsize=8,
                color=theme.ink_secondary)
    P.pcc_axis(ax, P.PER_MOLECULE, on="x")
    ax.set_ylabel("cumulative fraction of genes")
    ax.set_title("Cumulative", loc="left")
    P.grid(ax, "y", theme)

    P.finish(fig, bottom=0.02, script="scripts/15_figures.py",
             sources=[f"results/transcriptome_predictions.npz  ({NPZ_RUN}.json)"],
             theme=theme,
             note="The random baseline is omitted: it lies under the mean "
                  "baseline to within the line width.")
    return fig


@figure(
    name="transcriptome-calibration",
    title="Calibration — the range compression PCC cannot see",
    group="transcriptome",
    requires=("results/transcriptome_predictions.npz",),
    tier=2,
    caveat="On a wide layer the dominant variance is gene-to-gene, so read "
           "this as calibration of the overall expression scale, not of the "
           "condition response (metrics.py:calibration_slope).",
)
def transcriptome_calibration(theme=P.LIGHT):
    from ..metrics import calibration_slope
    z = results_npz("transcriptome_predictions")
    true, pred = z["y_true"], z["y_pred"]

    # Computed from the npz itself, NOT read from transcriptome_loco.json --
    # see NPZ_RUN below. Borrowing the slope from a different run's JSON is the
    # provenance error this repository keeps catching in prose.
    ok = np.isfinite(true) & np.isfinite(pred)
    slope = calibration_slope(pred, true)
    rmse = float(np.sqrt(np.mean((pred[ok] - true[ok]) ** 2)))

    rng = np.random.default_rng(0)
    idx = rng.choice(int(ok.sum()), size=min(400_000, int(ok.sum())),
                     replace=False)
    t, p = true[ok][idx], pred[ok][idx]

    fig, ax = P.panels(
        figsize=(6.4, 5.6), theme=theme,
        title="Predicted against measured expression",
        subtitle=f"{idx.size:,} of {int(ok.sum()):,} out-of-fold predictions, "
                 "sampled. PCC is shift- and scale-invariant and so is blind "
                 "to everything this panel shows.")

    hb = ax.hexbin(p, t, gridsize=70, cmap=theme.cmap(), bins="log",
                   linewidths=0)
    lo, hi = float(min(p.min(), t.min())), float(max(p.max(), t.max()))
    ax.plot([lo, hi], [lo, hi], color=theme.muted, linewidth=1.2,
            label="identity")
    inter = float(np.mean(t) - slope * np.mean(p))
    ax.plot([lo, hi], [slope * lo + inter, slope * hi + inter],
            color=theme.series[1], linewidth=1.8,
            label=f"fitted slope {slope:.3f}")
    ax.set_xlabel("predicted")
    ax.set_ylabel("measured")
    ax.legend(loc="upper left")
    ax.set_aspect("equal", adjustable="box")
    cb = fig.colorbar(hb, ax=ax, fraction=0.045, pad=0.03)
    cb.set_label("predictions per cell (log)", fontsize=8)
    cb.outline.set_visible(False)

    ax.text(0.98, 0.02,
            f"RMSE {rmse:.4f}   ·   slope {slope:.3f}",
            transform=ax.transAxes, ha="right", fontsize=8,
            color=theme.ink_secondary)

    P.finish(fig, bottom=0.02, script="scripts/15_figures.py",
             sources=[f"results/transcriptome_predictions.npz  ({NPZ_RUN}.json)"],
             theme=theme,
             note="slope and RMSE are recomputed from the saved predictions, "
                  "not read from a JSON")
    return fig


@figure(
    name="methods-faithful",
    title="On the paper's own protocol: the model matches, the baselines do not",
    group="transcriptome",
    requires=("results/methods_faithful_eval.json",),
    tier=1,
    caveat="This is the ONE like-for-like comparison in the atlas: same axis, "
           "same scaling, same subset, same wild-type definition "
           "(Supplementary Methods 3.3.3).",
)
def methods_faithful(theme=P.LIGHT):
    d = results_json("methods_faithful_eval")
    rows = [("MOMA", "moma"), ("random", "random"),
            ("mean", "mean"), ("wildtype", "wildtype")]
    ours = [d["ours"][k]["pcc"] for k, _ in rows]
    theirs = [d["paper"][pk] for _, pk in rows]
    status = ["good", "critical", "critical", "critical"]

    fig, ax = P.panels(
        figsize=(8.6, 4.8), theme=theme,
        title="Applying all four of the Supplementary Methods' choices",
        subtitle=f"{d['n_profiles']:,} exponential-phase profiles / "
                 f"{d['n_conditions']} conditions, per-gene min-max scaling, "
                 f"{d['n_wt_profiles']} wild-type profiles (11.9%, not the 70% "
                 "that gp==none gives). No code in the model changed.")

    ys = np.arange(len(rows))[::-1]
    for y, o, t, s in zip(ys, ours, theirs, status):
        ax.plot([t, o], [y, y], color=theme.grid, linewidth=3, zorder=1,
                solid_capstyle="round")
        ax.scatter([t], [y], s=52, facecolor=theme.surface,
                   edgecolor=theme.paper, linewidth=1.6, zorder=3)
        ax.scatter([o], [y], s=52, color=theme.series[0], zorder=4)
        ax.text(max(o, t) + 0.035, y, P.STATUS_ICON[s], color=P.STATUS[s],
                fontsize=11, va="center", ha="left", fontweight="bold")
        ax.text(min(o, t) - 0.02, y, f"Δ {abs(o - t):.2f}", ha="right",
                va="center", fontsize=7.5, color=theme.muted)

    ax.set_yticks(ys, [r[0] for r in rows])
    P.pcc_axis(ax, P.PER_PROFILE, on="x")
    ax.set_xlim(0.15, 0.72)
    P.grid(ax, "x", theme)

    handles = [
        P.legend_proxy(ax, s=52, color=theme.series[0],
                       label="this reproduction"),
        P.legend_proxy(ax, s=52, facecolor=theme.surface,
                       edgecolor=theme.paper, linewidth=1.6, label="paper"),
    ]
    ax.legend(handles=handles, loc="lower right")
    ax.text(0.99, 0.97,
            f"{P.STATUS_ICON['good']} reproduced    "
            f"{P.STATUS_ICON['critical']} did not reproduce",
            transform=ax.transAxes, ha="right", va="top", fontsize=7.5,
            color=theme.ink_secondary)

    # Read from `d`, never written as a literal. This sentence carried "0.578"
    # long after the value it describes became 0.544 -- a stale number inside a
    # rendered figure, which nothing checks and every reader trusts. It is the
    # same failure `reporting.py` records for the table, and the same fix.
    fig.text(0.02, 0.045,
             f"MOMA reproduces ({ours[0]:.3f} vs {theirs[0]:.2f}). The BASELINES do not "
             f"(~0.52 vs 0.25/0.26/0.36), so the margin over baseline is "
             f"{d['gap_ours']:.3f} where the paper reports {d['gap_paper']:.2f}. "
             "A baseline has no free parameters, so that gap cannot be a "
             "modelling difference — it is a difference in what the "
             "baseline is computed over.",
             fontsize=7.5, color=theme.ink_secondary, wrap=True)

    P.finish(fig, bottom=0.1, script="scripts/15_figures.py",
                 sources=["results/methods_faithful_eval.json"], theme=theme)
    return fig


@figure(
    name="baseline-sweep",
    title="No representation reproduces the paper's baselines",
    group="transcriptome",
    requires=("results/baseline_calibration.json",),
    tier=1,
    caveat="The sweep moves ALL curves together, including MOMA's: no value "
           "of `a` opens the paper's gap.",
)
def baseline_sweep(theme=P.LIGHT):
    d = results_json("baseline_calibration")
    sweep = d["sweep"]
    a = [s["a"] for s in sweep]
    fitted, paper = d["fitted"], d["paper"]["all"]
    a_fit = fitted["a"]
    gap_paper = paper["moma"] - paper["mean"]

    fig, axes = P.panels(
        1, 2, figsize=(11.4, 4.6), theme=theme,
        title="Sweeping the representation, looking for the paper's baselines",
        subtitle="`a` interpolates raw values (a=0) to per-gene z-scores (a=1). "
                 "Hollow markers are the paper's four reported values, placed "
                 "at the `a` that best fits its RANDOM baseline.",
        gridspec_kw={"width_ratios": [1.3, 1.0]})

    ax = axes[0]
    keys = [("moma", "MOMA"), ("random", "random"),
            ("mean", "mean"), ("wildtype", "wild-type")]
    for i, (key, label) in enumerate(keys):
        ax.plot(a, [s[key] for s in sweep], color=theme.series[i], label=label)
        ax.scatter([a_fit], [paper[key]], s=54, facecolor=theme.surface,
                   edgecolor=theme.series[i], linewidth=1.6, zorder=5)
    ax.axvline(a_fit, color=theme.muted, linewidth=1.0, zorder=1)
    ax.text(a_fit, 0.99, f" a={a_fit}", transform=ax.get_xaxis_transform(),
            ha="left", va="top", fontsize=7.5, color=theme.muted)
    P.pcc_axis(ax, P.PER_PROFILE)
    ax.set_xlabel("representation parameter  a")
    ax.set_title("Every predictor moves together", loc="left")
    ax.legend(loc="lower left")
    P.grid(ax, "y", theme)
    P.caption(ax,
            f"at a={a_fit} our random baseline lands on the paper's "
            f"({fitted['random']:.3f} vs {paper['random']:.2f}) — but MOMA "
            f"falls with it, to {fitted['moma']:.3f} against {paper['moma']:.2f}", theme=theme)

    ax = axes[1]
    gaps = [s["gap"] for s in sweep]
    ax.plot(a, gaps, color=theme.series[0], label="ours")
    ax.scatter([a_fit], [fitted["gap"]], s=54, color=theme.series[0], zorder=5)
    P.reference_rule(ax, gap_paper, f"paper {gap_paper:.2f}", theme=theme)
    ax.set_xlabel("representation parameter  a")
    ax.set_ylabel("MOMA − mean baseline\n(per profile, across molecules)",
                  fontsize=8)
    ax.set_title("The margin that has to close", loc="left")
    ax.legend(loc="upper left")
    P.grid(ax, "y", theme)
    P.caption(ax,
            f"at the fitted a the margin is {fitted['gap']:.3f}, not "
            f"{gap_paper:.2f}. Where it exceeds {gap_paper:.2f} (a→1) the "
            "baselines have gone negative, so the axis no longer means the "
            "same thing.", theme=theme)

    P.finish(fig, script="scripts/15_figures.py",
                 sources=["results/baseline_calibration.json"], theme=theme,
                 note="scripts/07_baseline_calibration.py")
    return fig

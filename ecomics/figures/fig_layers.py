"""D — Proteome, metabolome, fluxome, phenome, and the prospective test.

Four layers where the sample size, not the model, is usually the binding
constraint. The proteome has **5** shared conditions and the metabolome 6–25,
so per-molecule PCC is not estimable and these figures draw the `n/a (5 cond)`
slot rather than a short bar: an absent measurement must not look like a weak
result.

What each figure is actually allowed to claim:

  proteome    coverage (arithmetic, 588 of 589 by union) and the ensemble
              beating own-mRNA. NOT the ordering between networks — scripts/12
              shows CPN and KEGG swap across the penalty.
  metabolome  nothing settled. The penalty sweep walks core-from-protein across
              the paper's entire band while per-molecule PCC collapses
              negative, which is why the layer is ⚫ undecidable.
  fluxome     the shape of the flux distribution (0.843 per profile). The
              per-reaction number is UNDEFINED, not poor — all four flux media
              are glucose-minimal, so FBA returns an identical solution for all
              42 profiles and a constant has no variance to correlate.
  phenome     the input layer alone reproduces the paper's ~0.59. Fig. 5g's
              additivity claim does not reproduce: the curve is non-monotone.
"""

from __future__ import annotations

import numpy as np

from .. import config as C
from .. import plots as P
from . import figure, results_json

def _proteome_arms(d: dict) -> list[str]:
    """Arm names in `all_layers.json` order, ENSEMBLE last.

    Read from the file rather than hard-coded: the arm SET changed on
    2026-08-12 when the layer moved to Supplementary Data 2's graphs, which
    added `SIGMA` and `SRNA`. A hard-coded list would have kept rendering the
    old five and silently dropped the two new ones — the exact failure the
    `requires` declaration exists to prevent one level up.
    """
    skip = {"own_mRNA_baseline", "network_source"}
    return [k for k, v in d.items() if isinstance(v, dict) and k not in skip]


@figure(
    name="proteome-networks",
    title="Proteome: coverage is arithmetic, the ordering is not",
    group="layers",
    requires=("results/all_layers.json",),
    tier=1,
    caveat="Per-molecule PCC is suppressed at 5 conditions. The per-network "
           "ORDERING is unsettled — see figure `alpha-sensitivity`. Every "
           "accuracy bar carries its coverage in the tick label, because SRNA "
           "reaches 5 proteins and would otherwise read as the best arm.",
)
def proteome_networks(theme=P.LIGHT):
    d = results_json("all_layers")["proteome"]
    base = d["own_mRNA_baseline"]
    arms = _proteome_arms(d)
    source = d.get("network_source", "scraped")

    fig, axes = P.panels(
        1, 2, figsize=(11.6, 4.8), theme=theme,
        title="A protein is better predicted from its neighbours than from its own mRNA",
        subtitle="Leave-one-condition-out over the 5 conditions shared by the "
                 "transcriptome and proteome. 33 condition-averaged profiles × "
                 "589 proteins — the paper trained on 71 × 1,001, never "
                 "released. Networks: "
                 + ("the paper's own, Supplementary Data 2"
                    if source == "supplementary_data_2" else "scraped (superseded)")
                 + ".")

    ax = axes[0]
    xs = np.arange(len(arms))
    cov = [d[n]["coverage"] for n in arms]
    colors = [theme.muted] * (len(arms) - 1) + [theme.series[0]]
    P.bars(ax, xs, cov, color=colors, theme=theme, fmt="{:.0f}")
    # at=0.99 rather than 0.01: with PPI at 582 the left edge now collides with
    # that bar's own value label.
    P.reference_rule(ax, 589, "589 in the release", theme=theme, at=0.99)
    ax.set_xticks(xs, arms, fontsize=7.5)
    ax.set_ylabel("proteins covered")
    ax.set_ylim(0, 760)
    ax.set_title("Coverage — no single network suffices", loc="left")
    P.grid(ax, "y", theme)
    P.caption(ax,
            f"{cov[-1]} of 589 by union: the paper's central coverage "
            "argument, reproduced as arithmetic rather than statistics", theme=theme)

    ax = axes[1]
    vals = [d[n]["pcc_row_mean"] for n in arms]
    P.bars(ax, xs, vals, color=colors, theme=theme, fmt="{:.3f}")
    P.reference_rule(ax, base["pcc_row_mean"],
                     f"own-mRNA baseline {base['pcc_row_mean']:.3f}", theme=theme,
                     color=theme.series[1], at=0.01)
    P.paper_band(ax, *C.PAPER["pcc_proteome_all"], theme=theme,
                 label="paper (ensemble)")
    # Coverage is welded into the tick label of the ACCURACY panel, not left to
    # the panel beside it. `SRNA` reaches 5 proteins and scores 0.664 -- the
    # tallest bar here -- so a reader who looks only at this panel would rank an
    # absent predictor first. The left panel already carries coverage; this is
    # the same number a second time, on purpose.
    P.pcc_axis(ax, P.PER_PROFILE)
    ax.set_xticks(xs, [f"{n}\n{c}" for n, c in zip(arms, cov)], fontsize=7.5)
    ax.set_xlabel("arm, and the proteins it covers", fontsize=8)
    ax.set_ylim(0, 0.90)
    ax.set_title("Accuracy — read every bar against its coverage", loc="left")
    P.grid(ax, "y", theme)
    ax.text(len(arms) / 2 - 0.5, 0.80, "per-molecule axis", ha="center",
            va="bottom", fontsize=7.5, color=theme.muted)
    P.not_applicable(ax, len(arms) / 2 - 0.5,
                     d["ENSEMBLE"]["n_conditions_available"], theme=theme, y=0.66)

    P.finish(fig, script="scripts/15_figures.py",
                 sources=["results/all_layers.json"], theme=theme,
                 note="scripts/04_reproduce.py")
    return fig


PAPER_ARMS = ["TRN", "PPI", "KEGG", "SIGMA", "SRNA", "CPN", "CPN_paper_LEAKY"]


@figure(
    name="paper-networks",
    title="Proteome: the graphs were the problem, not the model",
    group="layers",
    requires=("results/paper_networks_proteome.json",),
    tier=1,
    caveat="Read every arm against its COVERAGE: SRNA scores 0.664 over 5 of "
           "589 proteins, so it is an absent predictor and not the best one. "
           "Per-molecule PCC is suppressed at 5 conditions.",
)
def paper_networks(theme=P.LIGHT):
    """Panel B is a scatter rather than a bar chart, and that is the whole point.

    As bars ordered by PCC, `SRNA` is simply the tallest -- 0.664, above every
    other arm and above both ensembles. Putting coverage on its own axis moves
    it to the far left with nothing under it, which is what it is: a predictor
    for five proteins. No arrangement of bars makes that legible without the
    reader choosing to look at a second series.
    """
    d = results_json("paper_networks_proteome")
    v, arms = d["verdict"], d["arms"]
    n_prot = d["proteome_targets"]

    fig, axes = P.panels(
        1, 2, figsize=(11.6, 4.8), theme=theme,
        title="The paper's own networks are worth +0.113 to the proteome layer",
        subtitle="Identical model, folds, LASSO penalty, neighbour cap and "
                 "locally-built CPN on every rung — only the graphs change. "
                 f"LOCO over the {d['shared_conditions']} conditions shared by "
                 "the transcriptome and proteome. The scraped rung is HISTORY: "
                 "its loaders were removed and it cannot be re-run.")

    # -- A: the three rungs, with the attribution between them
    ax = axes[0]
    # The first rung is HISTORY: its code was deleted on 2026-08-12, so the
    # value is read from the `retired_rung` block the script records rather than
    # recomputed. Drawn in the muted "not comparable" neutral, and labelled as
    # retired, so nobody reads three live measurements off this panel.
    retired = d.get("retired_rung", {})
    rungs = [(f"scraped\n4 arms\n(retired)", retired.get("ensemble_pcc_row_mean")),
             ("Data 2\n4 arms", v["ensemble_paper_matched"]),
             ("Data 2\n6 arms", v["ensemble_paper"])]
    rungs = [(lbl, val) for lbl, val in rungs if val is not None]
    xs = np.arange(len(rungs))
    P.bars(ax, xs, [r[1] for r in rungs],
           color=[theme.paper] + [theme.series[0]] * (len(rungs) - 1),
           theme=theme, fmt="{:.3f}")
    P.reference_rule(ax, d["own_mrna_baseline"]["pcc_row_mean"],
                     f"own-mRNA baseline {d['own_mrna_baseline']['pcc_row_mean']:.3f}",
                     theme=theme, color=theme.series[1], at=0.01)
    P.paper_band(ax, *C.PAPER["pcc_proteome_all"], theme=theme,
                 label="paper (ensemble)")
    P.pcc_axis(ax, P.PER_PROFILE)
    ax.set_xticks(xs, [r[0] for r in rungs])
    ax.set_ylim(0, 0.90)
    ax.set_title("Three rungs, so the source is separable from the arms",
                 loc="left")
    P.grid(ax, "y", theme)
    # The attribution is the finding; a reader should not have to subtract.
    if retired.get("delta_graph_source") is not None:
        ax.annotate(f"graph source\n{retired['delta_graph_source']:+.3f}", (0.5, 0.34),
                    ha="center", fontsize=8, color=theme.ink_secondary)
    ax.annotate(f"+SIGMA +SRNA\n{v['delta_extra_arms']:+.5f}",
                (len(rungs) - 1.5, 0.34),
                ha="center", fontsize=8, color=theme.muted)
    P.caption(ax, "coverage is 588/589 on every rung, so none of the gain is "
                  "coverage — it had to show up as PCC or not at all",
              theme=theme)

    # -- B: every arm against its coverage, with the three shared arms' MOVE
    ax = axes[1]
    # Hand-placed so seven labels in one dense cluster stay readable. Automatic
    # placement was tried and put `CPN (paper, leaky)` on top of the reference
    # rule's own label.
    OFFSET = {"SRNA": (0, 10), "TRN": (0, 10), "KEGG": (0, 10),
              "PPI": (20, 1), "SIGMA": (-34, -10), "CPN": (-30, 3),
              "CPN_paper_LEAKY": (52, -3)}

    # The scraped arms used to be plotted here with an arrow to their Data 2
    # counterpart -- TRN moving UP while moving LEFT was the panel's second
    # finding. Those loaders were deleted, so `arms` has one side now. The
    # retired values are in the JSON's `retired_rung` block and in
    # docs/scripts/17.

    for name in PAPER_ARMS:
        a = arms["paper"][name]
        leaky = name.endswith("_LEAKY")
        # `theme.paper` is the reserved "not comparable" neutral, and the
        # diamond plus the word `leaky` carry the same meaning -- colour never
        # does this alone here, per `plots.status_marker`.
        ax.scatter(a["coverage"], a["pcc_row_mean"], s=46, zorder=3,
                   color=theme.paper if leaky else theme.series[0],
                   marker="D" if leaky else "o")
        dx, dy = OFFSET[name]
        ax.annotate("CPN paper — leaky" if leaky else name,
                    (a["coverage"], a["pcc_row_mean"]),
                    textcoords="offset points", xytext=(dx, dy),
                    ha="center", fontsize=7.5,
                    color=theme.paper if leaky else theme.ink_secondary)

    P.reference_rule(ax, arms["paper_matched"]["ENSEMBLE"]["pcc_row_mean"],
                     "4-arm ensemble", theme=theme, at=0.01)
    P.pcc_axis(ax, P.PER_PROFILE)
    ax.set_xlabel(f"proteins covered (of {n_prot})")
    ax.set_xlim(-45, n_prot + 200)
    ax.set_ylim(-0.03, 0.82)
    ax.set_title("Every arm against its coverage", loc="left")
    P.grid(ax, "y", theme)
    ax.text(0.34, 0.975,
            "● Data 2 arm        ◆ leaky, excluded from the mean",
            transform=ax.transAxes, fontsize=7.5, color=theme.muted)
    P.caption(ax, "SRNA sits top-LEFT: 0.664 over 5 proteins is an absent "
                  "predictor, not the best one", theme=theme)

    P.finish(fig, script="scripts/15_figures.py",
             sources=["results/paper_networks_proteome.json"], theme=theme,
             note="scripts/17_paper_networks_proteome.py")
    return fig


@figure(
    name="alpha-sensitivity",
    title="How much of the proteome and metabolome result is the LASSO penalty?",
    group="layers",
    requires=("results/alpha_sensitivity.json",),
    tier=1,
    caveat="The metabolome panel is the reason that layer is ⚫ undecidable "
           "rather than merely unreproduced: no penalty makes both axes "
           "respectable at once.",
)
def alpha_sensitivity(theme=P.LIGHT):
    d = results_json("alpha_sensitivity")
    alphas = d["alphas"]
    prot, meta = d["proteome"], d["metabolome"]
    v = d["verdict"]

    fig, axes = P.panels(
        1, 2, figsize=(11.4, 4.8), theme=theme,
        title="Refitting at seven penalties: what survives, and what does not",
        subtitle="`alpha` was never re-selected after the audit standardized "
                 "the LASSO, and with 4 training conditions it is decisive. "
                 "So the question is which claims hold across the whole sweep.")

    ax = axes[0]
    ax.plot(alphas, [p["ensemble"] for p in prot], color=theme.series[0],
            marker="o", label="ensemble")
    ax.plot(alphas, [p["own_mrna"] for p in prot], color=theme.series[1],
            marker="o", label="own-mRNA baseline")
    ax.set_xscale("log")
    P.pcc_axis(ax, P.PER_PROFILE)
    ax.set_xlabel("LASSO penalty  α")
    ax.set_title("Proteome — the claim survives", loc="left")
    ax.legend(loc="upper left")
    P.grid(ax, "y", theme)
    P.caption(ax,
            f"the ensemble beats own-mRNA at every penalty tested, by "
            f"{v['ratio_min']:.2f}× to {v['ratio_max']:.2f}×. The per-network "
            f"ordering does not: {v['n_distinct_orderings']} distinct orderings "
            "across seven penalties.", theme=theme)

    ax = axes[1]
    ax.plot(alphas, [m["per_profile"] for m in meta], color=theme.series[0],
            marker="o", label="per profile, across molecules")
    ax.plot(alphas, [m["per_molecule"] for m in meta], color=theme.series[1],
            marker="o", label="per molecule, across conditions")
    ax.axhline(0, color=theme.baseline, linewidth=1.0)
    P.paper_band(ax, *C.PAPER["pcc_metabolome_core"], theme=theme,
                 label="paper", text_x=0.35)
    ax.set_xscale("log")
    # Both axes appear on this panel BY DESIGN -- their divergence is the
    # finding -- so this is the one documented use of pcc_axis_both().
    P.pcc_axis_both(ax)
    ax.set_xlabel("LASSO penalty  α")
    ax.set_title("Metabolome, core from protein — it does not", loc="left")
    ax.legend(loc="lower left")
    P.grid(ax, "y", theme)
    P.caption(ax,
            "every penalty that lands per-profile inside the paper's band "
            "drives per-molecule to zero and then sharply negative — the "
            "signature of predictions collapsing onto a constant, which "
            "per-profile PCC rewards and per-molecule PCC cannot fake", theme=theme)

    P.finish(fig, script="scripts/15_figures.py",
                 sources=["results/alpha_sensitivity.json"], theme=theme,
                 note="scripts/12_alpha_sensitivity.py")
    return fig


_META_ROWS = [
    ("core_from_protein", "core\nfrom protein", "pcc_metabolome_core"),
    ("noncore_from_protein", "non-core\nfrom protein", None),
    ("noncore_from_transcript", "non-core\nfrom transcript", "pcc_metabolome_noncore"),
    ("core_from_transcript", "core\nfrom transcript", None),
]


@figure(
    name="metabolome-layers",
    title="Metabolome: 25 conditions is marginal and 6 is not enough for anything",
    group="layers",
    requires=("results/all_layers.json",),
    tier=1,
    caveat="core-from-transcript is NaN BY CONSTRUCTION, not from low variance: "
           "scripts/04 passes proteins=None on this join, so the core branch of "
           "MetabolomeModule.fit never runs. All 72 values are finite.",
)
def metabolome_layers(theme=P.LIGHT):
    d = results_json("all_layers")["metabolome"]

    fig, ax = P.panels(
        figsize=(8.4, 4.8), theme=theme,
        title="Predicting metabolites from proteins and from transcripts",
        subtitle="12 core and 102 non-core metabolites are present in the "
                 "published 114. The per-profile axis is the interpretable one "
                 "at this sample size.")

    xs = np.arange(len(_META_ROWS))
    vals, colors = [], []
    for key, _, _ in _META_ROWS:
        v = d[key]["pcc_row_mean"]
        vals.append(0.0 if v != v else v)
        colors.append(theme.muted if v != v else theme.series[0])
    P.bars(ax, xs, vals, color=colors, theme=theme, fmt="{:.3f}", labels=False)

    for x, (key, _, _) in zip(xs, _META_ROWS):
        m = d[key]
        v = m["pcc_row_mean"]
        n = m["n_conditions_available"]
        if v != v:
            P.not_applicable(ax, x, n, theme=theme, y=0.12)
            ax.text(x, 0.10, "NaN: core branch never fitted\n(proteins=None on this join)",
                    ha="center", va="top", fontsize=6.5, color=theme.muted)
        else:
            ax.text(x, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8,
                    color=theme.ink_secondary)
            P.annotate_n(ax, x, v, n, theme=theme)

    for x, (_, _, paper_key) in zip(xs, _META_ROWS):
        if paper_key:
            mean, sd = C.PAPER[paper_key]
            ax.plot([x - 0.34, x + 0.34], [mean, mean], color=theme.paper,
                    linewidth=1.4, zorder=5)
            ax.add_patch(__import__("matplotlib").patches.Rectangle(
                (x - 0.34, mean - sd), 0.68, 2 * sd, color=theme.paper,
                alpha=0.14, zorder=0))
            ax.text(x + 0.36, mean, f"paper {mean:.2f}±{sd:.2f}", fontsize=7,
                    va="center", color=theme.ink_secondary)

    ax.set_xticks(xs, [r[1] for r in _META_ROWS], fontsize=8)
    P.pcc_axis(ax, P.PER_PROFILE)
    ax.set_ylim(-0.12, 1.1)
    ax.axhline(0, color=theme.baseline, linewidth=1.0)
    P.grid(ax, "y", theme)

    fig.text(0.02, 0.04,
             "core-from-protein at 0.420 falls OUTSIDE the paper's 0.65 ± 0.21 "
             "at the default penalty — but figure `alpha-sensitivity` walks it "
             "across the paper's entire band, so neither number is settled.",
             fontsize=7.5, color=theme.ink_secondary, wrap=True)

    P.finish(fig, bottom=0.1, script="scripts/15_figures.py",
                 sources=["results/all_layers.json"], theme=theme)
    return fig


@figure(
    name="fluxome",
    title="Fluxome: the NaN is the result",
    group="layers",
    requires=("results/all_layers.json", "results/fluxome_spec_completion.json"),
    tier=1,
    caveat="Per-reaction PCC is UNDEFINED, not poor. A bar of height zero "
           "would be a lie, so the middle panel is text. The per-profile "
           "number is NOT the salvageable result either: a constant mean flux "
           "profile beats every arm (0.896 vs 0.843, p = 5e-5).",
)
def fluxome(theme=P.LIGHT):
    d = results_json("all_layers")["fluxome"]
    spec = results_json("fluxome_spec_completion")
    configs = ["plain FBA", "+ medium (input layer)", "+ medium + expression"]

    fig, axes = P.panels(
        1, 3, figsize=(13.0, 4.4), theme=theme,
        title="FBA captures the shape every flux profile shares — which a constant captures better",
        subtitle="43 profiles × 120 reactions, of which 22 map to iJO1366. "
                 "All four flux media are glucose-minimal, so the medium "
                 "constraint returns an IDENTICAL solution for every profile.",
        gridspec_kw={"width_ratios": [1.0, 0.95, 1.05]})

    ax = axes[0]
    xs = np.arange(len(configs))
    vals = [d[c]["pcc_row_mean"] for c in configs]
    P.bars(ax, xs, vals, color=theme.series[0], theme=theme, fmt="{:.3f}")
    # The baseline is the whole panel. FBA needs no training data, so this layer
    # never enters `run_loco` and never inherited the three baselines from it --
    # they arrived separately, via `evaluate.out_of_fold_baselines`, on
    # 2026-08-13. Until they were drawn this panel showed 0.843 under the
    # heading "what is measurable" with nothing to read it against, and a
    # caption calling the magnitudes "broadly right". They are: every profile
    # has the same broadly-right magnitudes, which is why a CONSTANT scores
    # higher. The rule sits above every bar on purpose.
    base = d["baselines"]
    mean_b = base["mean"]
    P.reference_rule(ax, mean_b["pcc_row_mean"],
                     f"constant mean flux profile {mean_b['pcc_row_mean']:.3f}",
                     theme=theme, color=P.STATUS["critical"], at=0.99)
    P.pcc_axis(ax, P.PER_PROFILE)
    ax.set_xticks(xs, ["plain\nFBA", "+ medium", "+ medium\n+ expression"],
                  fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_title("What is measurable, and what beats it", loc="left")
    P.grid(ax, "y", theme)
    P.caption(ax,
            f"{d['plain FBA']['n_rows']}/42 solved. Every arm scores BELOW the "
            f"constant ({mean_b['margin_vs_headline_per_profile']:+.3f}, p = "
            f"{mean_b['wilcoxon_p_per_profile']:.0e}); wild-type "
            f"{base['wildtype']['pcc_row_mean']:.3f} is indistinguishable",
            theme=theme)

    ax = axes[1]
    ax.axis("off")
    ax.set_title("What is not", loc="left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.5, 0.82, "undefined", ha="center", va="center", fontsize=28,
            color=theme.muted)
    ax.text(0.5, 0.68, "PCC per reaction, across profiles", ha="center",
            va="center", fontsize=8.5, color=theme.ink, fontweight="bold")
    ax.text(0.5, 0.58,
            "A constant prediction has zero variance\n"
            "per reaction, so the correlation is 0/0.\n"
            "Only the expression constraint can vary,\n"
            "and only 3 of 42 conditions have a\n"
            "measured transcriptome to vary it with.",
            ha="center", va="top", fontsize=8, color=theme.ink_secondary,
            linespacing=1.7)
    P.status_marker(ax, 0.16, 0.04, "serious", theme=theme,
                    label="undecidable on public data")

    ax = axes[2]
    scan = spec["threshold_scan"]
    ts = sorted(float(t) for t in scan)
    biomass = [scan[f"{t:g}"]["biomass"] for t in ts]
    changed = [scan[f"{t:g}"]["bounds_changed"] for t in ts]
    ax.plot(ts, biomass, color=theme.series[0], marker="o", label="biomass flux")
    ax.set_xlabel("expression threshold")
    ax.set_ylabel("biomass flux at the optimum")
    ax.set_title("Where the constraint kills the cell", loc="left")
    ax.set_ylim(-0.16, 1.26)
    P.grid(ax, "y", theme)
    first_dead = next((t for t, b in zip(ts, biomass) if b <= 0), None)
    if first_dead is not None:
        ax.axvline(first_dead, color=P.STATUS["critical"], linewidth=1.2)
        ax.text(first_dead, 0.42, f" infeasible from {first_dead:g}",
                transform=ax.get_xaxis_transform(), fontsize=7.5,
                color=P.STATUS["critical"], va="center")
    for t, b, c in zip(ts, biomass, changed):
        ax.annotate(f"{c}", (t, b), textcoords="offset points",
                    xytext=(0, 9 if b > 0.5 else -13), ha="center",
                    fontsize=6.5, color=theme.muted)
    P.caption(ax, "labels = reaction bounds changed at that threshold",
              theme=theme)
    ax.legend(loc="upper right")

    P.finish(fig, script="scripts/15_figures.py",
                 sources=["results/all_layers.json",
                          "results/fluxome_spec_completion.json"], theme=theme,
                 note="scripts/04_reproduce.py, scripts/14_fluxome_spec_completion.py")
    return fig


@figure(
    name="phenome-additivity",
    title="Fig. 5g's additivity claim: the curve is not monotone",
    group="layers",
    requires=("results/five_layer_phenome.json",),
    tier=1,
    caveat="The condition set MOVES as layers are added (254→179→207→226), so "
           "any rise is partly confounded with sample. On strictly shared "
           "conditions the question cannot be asked at all: 5 conditions carry "
           "both transcriptome and proteome, and 1 carries all four.",
)
def phenome_additivity(theme=P.LIGHT):
    d = results_json("five_layer_phenome")
    union = [c for c in d["combos"] if c["mode"] == "union"]
    strict = [c for c in d["combos"] if c["mode"] == "strict"]

    fig, axes = P.panels(
        1, 2, figsize=(11.4, 4.8), theme=theme,
        title="Does each added omic layer improve growth-rate prediction?",
        subtitle="The paper's Fig. 5g says yes, monotonically. Here the "
                 "transcriptome COSTS 0.087 and the four-layer model does not "
                 "beat the input layer alone.",
        gridspec_kw={"width_ratios": [1.2, 1.0]})

    ax = axes[0]
    xs = np.arange(len(union))
    vals = [c["pcc"] for c in union]
    ax.plot(xs, vals, color=theme.series[0], marker="o", zorder=3)
    for x, c in zip(xs, union):
        P.annotate_n(ax, x, c["pcc"], c["n"], theme=theme)
    P.paper_band(ax, *C.PAPER["pcc_growth_seen"], theme=theme, label="paper",
                 text_x=0.45)
    P.reference_rule(ax, vals[0], "", theme=theme, color=theme.grid)
    ax.set_xticks(xs, ["input", "+ transcript", "+ protein", "+ metabolite"],
                  fontsize=8)
    P.pcc_axis(ax, P.PER_PROFILE)
    ax.set_ylabel("PCC of predicted vs measured growth rate\n"
                  "(one value per condition)", fontsize=8)
    ax.set_ylim(0.42, 0.72)
    ax.set_title("Union mode — each layer used where available", loc="left")
    P.grid(ax, "y", theme)
    dip = vals[0] - vals[1]
    ax.annotate(f"−{dip:.3f}", (0.5, (vals[0] + vals[1]) / 2),
                ha="center", fontsize=8, color=P.STATUS["critical"],
                fontweight="bold")

    ax = axes[1]
    ys = np.arange(len(strict))[::-1]
    for y, c in zip(ys, strict):
        if c["pcc"] is None:
            ax.text(0.02, y, f"n = {c['n']}  →  not computable", va="center",
                    fontsize=8.5, color=theme.muted)
        else:
            ax.barh([y], [c["n"]], 0.5, color=theme.series[2], linewidth=0)
            ax.text(c["n"] + 4, y, f"n = {c['n']}", va="center", fontsize=8,
                    color=theme.ink_secondary)
    ax.set_yticks(ys, ["input", "+ transcript", "+ protein", "+ metabolite"],
                  fontsize=8)
    ax.set_xlabel("conditions carrying ALL the listed layers")
    ax.set_xlim(0, 300)
    ax.set_title("Strict mode — why union mode is used", loc="left")
    P.grid(ax, "x", theme)
    P.caption(ax,
            "only 5 conditions carry both transcriptome and proteome, and 1 "
            "carries all four — so the strictly-shared question is unanswerable", theme=theme)

    P.finish(fig, script="scripts/15_figures.py",
                 sources=["results/five_layer_phenome.json"], theme=theme,
                 note="scripts/11_five_layer_phenome.py; growth for the "
                      "proteome and metabolome conditions comes from "
                      "Supplementary Data 5")
    return fig


_KO_ARMS = [("non_specific", "non-specific", "good"),
            ("specific", "specific", "good"),
            ("same_batch", "same-batch", "critical")]
_PAPER_GAIN = {"non_specific": 0.27, "specific": 0.41, "same_batch": 0.25}


@figure(
    name="prospective-ko",
    title="Prospective validation on 16 knockouts the model never saw",
    group="layers",
    requires=("results/prospective_ko.json",),
    tier=1,
    caveat="The same-batch row is INVALID and the fault is ours: quantile-"
           "mapping every GEO sample onto one marginal makes wild-type and "
           "knockout profiles from that batch artificially similar, so its "
           "baseline scores 0.963 where the paper's scores ~0.60.",
)
def prospective_ko(theme=P.LIGHT):
    d = results_json("prospective_ko")

    fig, axes = P.panels(
        1, 2, figsize=(11.6, 5.0), theme=theme,
        title="GSE73673 — held-out data the model was never tuned against",
        subtitle="The paper's own 16 knockouts, scored on Fig. 6a's "
                 "per-condition axis. Three initializations of y⁽⁰⁾, which "
                 "enters the model only through w_y · y⁽⁰⁾.",
        gridspec_kw={"width_ratios": [1.25, 1.0]})

    ax = axes[0]
    per = d["non_specific"]["per_condition"]
    genes = sorted(per, key=per.get)
    ys = np.arange(len(genes))
    for i, (key, label, _) in enumerate(_KO_ARMS[:2]):
        vals = [d[key]["per_condition"][g] for g in genes]
        ax.scatter(vals, ys, s=44, color=theme.series[i], label=label,
                   zorder=3, edgecolor=theme.surface, linewidth=1.5)
    for i, (key, label, _) in enumerate(_KO_ARMS[:2]):
        P.reference_rule(ax, d[key]["baseline_per_profile"],
                         f"{label} baseline", theme=theme,
                         color=theme.series[i], orient="v")
    ax.set_yticks(ys, genes, fontsize=7.5)
    P.pcc_axis(ax, P.PER_PROFILE, on="x")
    ax.set_title("Per knockout", loc="left")
    ax.legend(loc="lower right")
    P.grid(ax, "x", theme)

    ax = axes[1]
    xs = np.arange(len(_KO_ARMS))
    width = 0.34
    moma = [d[k]["moma_per_profile"] for k, _, _ in _KO_ARMS]
    base = [d[k]["baseline_per_profile"] for k, _, _ in _KO_ARMS]
    # A 2px surface gap between adjacent bars, not a border.
    ax.bar(xs - width / 2 - 0.012, base, width, color=theme.muted,
           linewidth=0, label="baseline")
    ax.bar(xs + width / 2 + 0.012, moma, width, color=theme.series[0],
           linewidth=0, label="MOMA")
    for x, (key, _, status) in zip(xs, _KO_ARMS):
        top = max(moma[int(x)], base[int(x)])
        gain = d[key]["gain_vs_baseline"]
        ax.text(x, top + 0.04,
                f"{gain:+.1%}\npaper {_PAPER_GAIN[key]:+.0%}", ha="center",
                va="bottom", fontsize=7.5, color=theme.ink_secondary,
                linespacing=1.4)
        # Only the invalid arm is flagged. Marking the two valid ones as well
        # would spend the status channel on "nothing to report".
        if status == "critical":
            P.status_marker(ax, x - 0.62, top + 0.26, status, theme=theme,
                            label="invalid — see below")
    ax.set_xticks(xs, [a[1] for a in _KO_ARMS])
    ax.set_xlabel("initialization of  y⁽⁰⁾")
    P.pcc_axis(ax, P.PER_PROFILE)
    ax.set_ylim(0, 1.45)
    ax.legend(loc="upper left")
    ax.set_title("MOMA against its baseline", loc="left")
    P.grid(ax, "y", theme)
    P.caption(ax, "same-batch is invalid and the fault is ours: quantile-"
                  "mapping every GEO sample onto one marginal makes wild-type "
                  "and knockout profiles from that batch artificially similar",
              theme=theme)

    spread = max(moma) - min(moma)
    fig.text(0.02, 0.045,
             f"The three initializations now differ (spread {spread:.4f}) where "
             "before the recurrence fix they were 0.3457 / 0.3456 / 0.3458 — "
             "effectively one number, which is the observable that predicted "
             "the inert recurrence. A falsifiable prediction of the fix, tested "
             "on held-out data and confirmed.",
             fontsize=7.5, color=theme.ink_secondary, wrap=True)

    P.finish(fig, bottom=0.12, script="scripts/15_figures.py",
                 sources=["results/prospective_ko.json"], theme=theme,
                 note="scripts/05_prospective_ko.py")
    return fig

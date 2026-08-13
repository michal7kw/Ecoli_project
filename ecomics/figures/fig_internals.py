"""C — Model internals: the recurrence, the depth, the tuning, the machine.

The finding that organizes this group is that the recurrent term was **dead**.
`y⁽⁰⁾` enters the transcriptome model only through `w_y · y⁽⁰⁾`, so three
different initializations scoring 0.3457 / 0.3456 / 0.3458 could only mean
`w_y ≈ 0` -- and it was, by a factor of 2.7e7 against `w_x`. Weight decay was
the whole cause.

That diagnosis generated a falsifiable prediction rather than just an
explanation, which is what makes it worth a figure: fix `w_y` and those three
numbers *must* separate. They did, on held-out GEO data the model was never
tuned against.

⚠ The depth figure is deliberately TWO panels because its two sources are on
different axes. `scripts/03` reports per-molecule PCC; `scripts/10` reports the
paper's per-profile axis. Plotting 0.19 beside 0.60 on one scale would be the
exact mistake this atlas exists to prevent.

⚠ Its left panel plots `depth_sweep_loco.json` ALONE, which
carries all seven depths at one configuration. It used to append
`depth_{5,6,8}_loco.json` — earlier runs on the superseded encoding with no
`config` block to mark them, a splice of two scales inside a single line, which
is the same error as one scale across two axes. Re-run properly, the curve
**peaks at n=5 and falls after it**; the spliced tail had asserted a rise to
n=8. Those three files are retired.
"""

from __future__ import annotations

import numpy as np

from .. import plots as P
from . import figure, results_json

ARM_LABEL = {
    "zero_init": "w_y = 0\n+ weight decay",
    "warm_init": "w_y = 0.05\n+ weight decay",
    "no_decay": "w_y = 0\nno decay on w_y",
    "warm_no_decay": "w_y = 0.05\nno decay on w_y",
}


@figure(
    name="recurrence-revival",
    title="The recurrence was dead, and weight decay was the whole cause",
    group="internals",
    requires=("results/recurrence_experiment.json",),
    tier=1,
    caveat="Single held-out fold, so the PCC values are not the headline LOCO "
           "numbers. What is being compared is the arms against each other.",
)
def recurrence_revival(theme=P.LIGHT):
    d = results_json("recurrence_experiment")
    arms = d["arms"]
    names = [ARM_LABEL.get(a["label"], a["label"]) for a in arms]
    xs = np.arange(len(arms))
    alive = [a["w_y"] > 1e-6 for a in arms]
    colors = [theme.series[0] if a else theme.muted for a in alive]

    fig, axes = P.panels(
        1, 2, figsize=(11.0, 4.6), theme=theme,
        title="Excluding w_y from weight decay revives the recurrent term",
        subtitle="Four arms crossing initialization (0 or 0.05) with whether "
                 "w_y is subject to weight decay. Initialization changes "
                 "nothing; the decay changes everything.")

    ax = axes[0]
    P.bars(ax, xs, [a["pcc"] for a in arms], color=colors, theme=theme,
           fmt="{:.3f}")
    P.pcc_axis(ax, P.PER_MOLECULE)
    ax.set_xticks(xs, names, fontsize=7.5)
    ax.set_title("Accuracy", loc="left")
    ax.set_ylim(0, 0.36)
    P.grid(ax, "y", theme)

    ax = axes[1]
    wy = [a["w_y"] for a in arms]
    P.bars(ax, xs, wy, color=colors, theme=theme, labels=False)
    for x, v in zip(xs, wy):
        ax.text(x, v, f" {v:.1e}", ha="center", va="bottom", fontsize=7.5,
                color=theme.ink_secondary, rotation=0)
    P.reference_rule(ax, float(np.mean([a["w_x"] for a in arms])),
                     "mean |w_x|", theme=theme)
    ax.set_yscale("log")
    ax.set_ylim(1e-12, 1e-1)
    ax.set_ylabel("mean |w_y|  (log scale)")
    ax.set_xticks(xs, names, fontsize=7.5)
    ax.set_title("The learned gene–gene weights", loc="left")
    P.grid(ax, "y", theme)
    P.caption(ax,
            "nine orders of magnitude between a decayed w_y and a free one — "
            "the model had silently reduced to a feed-forward map", theme=theme)

    P.finish(fig, script="scripts/15_figures.py",
                 sources=["results/recurrence_experiment.json"], theme=theme,
                 note="scripts/06_recurrence_experiment.py")
    return fig


@figure(
    name="memory-depth",
    title="Memory depth — a knee at 2, a maximum at 5, and a decline after it",
    group="internals",
    requires=("results/depth_sweep_loco.json",
              "results/trn_seeded_recurrence.json"),
    tier=1,
    caveat="TWO PANELS BY AXIS. The left panel is per molecule (scripts/03); "
           "the right is the paper's per-profile axis (scripts/10). The "
           "heights must not be read across -- that is why this is two panels "
           "and not one. Both are now measured at 603 features: the right "
           "panel's sweep was re-run, and it moved the shape as "
           "well as the scale (756-era: monotone to n=6; 603: a peak at n=3).",
)
def memory_depth(theme=P.LIGHT):
    trn = results_json("trn_seeded_recurrence")

    # ONLY depth_sweep_loco.json, which now carries ALL SEVEN depths at 603
    # features under one configuration (5 folds, 600 epochs, rank 64) -- 5, 6
    # and 8 were re-run and merged.
    #
    # It used to splice in depth_{5,6,8}_loco.json, which were measured under
    # the superseded encoding AND carry no `config` block at all, so nothing in
    # them marked the difference. Worse than stale: they ran 3-4x FASTER than
    # the re-runs on a SMALLER input, which no encoding change explains, so they
    # were never on this curve under any encoding. The spliced tail rose to
    # 0.319 at n=8 and made this figure assert "no maximum under full LOCO".
    # Measured properly the curve PEAKS AT 5 and falls to 0.179 by 8. Those
    # three files are retired; do not splice anything into this curve again.
    loco = results_json("depth_sweep_loco")
    depths, pccs = [], []
    for k, v in sorted(loco.items(), key=lambda kv: kv[1]["memory_depth"]):
        depths.append(v["memory_depth"])
        pccs.append(v["all_genes"]["pcc_mean"])

    fig, axes = P.panels(
        1, 2, figsize=(11.0, 4.6), theme=theme,
        title="Does the paper's optimal memory depth of 2 reproduce?",
        subtitle="The paper found n = 2 by cross-validation and noted "
                 "separately that 75% of cycles in E. coli's TRN are shorter "
                 "than 3. As a knee it reproduces; as a strict maximum it does "
                 "not.")

    ax = axes[0]
    ax.plot(depths, pccs, color=theme.series[0], marker="o",
            label="5-fold LOCO (scripts/03)")
    ax.axvline(2, color=theme.muted, linewidth=1.0, zorder=1)
    ax.text(2, 0.99, " paper: n=2", transform=ax.get_xaxis_transform(),
            ha="left", va="top", fontsize=7.5, color=theme.muted)
    P.pcc_axis(ax, P.PER_MOLECULE)
    ax.set_xlabel("memory depth  n")
    ax.set_title("Per molecule", loc="left")
    ax.legend(loc="lower right")
    P.grid(ax, "y", theme)
    # Measured to the PEAK, not to the last depth. Once 5/6/8 were re-run the
    # curve turned over, so `pccs[-1] - pccs[0]` is the gain to a point on the
    # way back DOWN -- and the fraction it produced was 105%, a share of a total
    # that is larger than the total. A denominator has to be the maximum, not
    # the endpoint, on any curve that is not monotone.
    peak = max(range(len(pccs)), key=lambda i: pccs[i])
    span = pccs[peak] - pccs[0]
    knee = (pccs[1] - pccs[0]) / span if len(pccs) > 1 and span else float("nan")
    P.caption(ax,
            f"n=1 collapses to {pccs[0]:.3f}; n=2 captures {knee:.0%} of the "
            f"gain to the peak at n={depths[peak]} ({pccs[peak]:.3f}), after "
            f"which depth HURTS -- n={depths[-1]} falls to {pccs[-1]:.3f}, "
            f"below n=3. All {len(depths)} depths on one encoding",
            theme=theme)

    ax = axes[1]
    # Computed, not asserted. This caption used to read "it rises monotonically
    # to n=6" as a string literal, and a later sweep it described was
    # re-run: the sentence survived the data by a day. The left panel's caption
    # had already been through the same lesson -- see `peak`/`knee` above, and
    # `reporting.py`, where every verdict is derived for exactly this reason.
    sweep = trn["depth_sweep"]
    t_depths = [s["depth"] for s in sweep]
    t_vals = [s["all_genes"] for s in sweep]
    ax.plot(t_depths, t_vals, color=theme.series[2], marker="o",
            label="w_y seeded with the paper's TRN")
    unseeded = next(a for a in trn["arms"] if a["label"] == "unseeded")
    ax.scatter([unseeded["depth"]], [unseeded["all_genes"]], s=60,
               facecolor=theme.surface, edgecolor=theme.series[2],
               linewidth=1.8, zorder=5, label="unseeded, n=2")
    P.paper_band(ax, trn["paper"]["all"], theme=theme, label="paper")
    ax.axvline(2, color=theme.muted, linewidth=1.0, zorder=1)
    P.pcc_axis(ax, P.PER_PROFILE)
    ax.set_xlabel("memory depth  n")
    ax.set_title("Per profile — the paper's axis", loc="left")
    ax.legend(loc="lower right")
    P.grid(ax, "y", theme)
    # Deliberately silent rather than defensive-with-a-fallback when the sweep
    # is empty: `plots.audit_empty_series` already names that failure precisely,
    # and computing `max()` over nothing would raise a ValueError that pre-empts
    # it with a worse message. Let the guard do the talking.
    if t_vals:
        t_peak = t_depths[max(range(len(t_vals)), key=lambda i: t_vals[i])]
        shape = ("rising monotonically"
                 if all(b >= a for a, b in zip(t_vals, t_vals[1:]))
                 else "and the curve is not monotone")
        P.caption(ax,
                f"a biologically-seeded w_y does not reproduce the paper's "
                f"optimum either: it peaks at n={t_peak}, {shape}. The paper's "
                f"n=2 is {'' if t_peak == 2 else 'not '}the maximum here",
                theme=theme)

    P.finish(fig, script="scripts/15_figures.py",
                 sources=["results/depth_sweep_loco.json",
                          "results/trn_seeded_recurrence.json"], theme=theme)
    return fig


@figure(
    name="hyperparameter-tuning",
    title="Tuning, against a ridge control that has since been re-measured",
    group="internals",
    requires=("results/rnn_tuning2.json", "results/rnn_tuning.json",
              "results/discrepancy_analysis.json",
              "results/ridge_control.json"),
    tier=1,
    caveat="EVERY NUMBER HERE IS 756-FEATURE, SINGLE-FOLD, from an earlier "
           "audit -- the sweeps and the control alike. They are comparable to "
           "each other and to nothing else. The committed control is "
           "`ridge-control`, where ridge reaches 0.421 against the model's "
           "0.186 on matched folds.",
)
def hyperparameter_tuning(theme=P.LIGHT):
    second = results_json("rnn_tuning2")
    first = results_json("rnn_tuning")
    # ⚠ 0.236 is a SINGLE held-out fold at 756 features, produced by an audit
    # pass with no committed script -- a later ridge control replaced it,
    # exists because this number was quoted against the 5-fold 603-feature
    # headline in four documents and the comparison was withdrawn from all four
    # It is kept HERE because the two sweeps it is drawn against
    # are from the same audit, same fold, same encoder: within this figure the
    # comparison is the one that was actually run. The label carries the
    # protocol so the rule cannot be lifted out of the panel and requoted.
    ridge = results_json("discrepancy_analysis")["ridge_control"]["per_gene_pcc"]
    ridge_best = max(ridge.values())
    committed = max(r["pcc_mean"] for r in
                    results_json("ridge_control")["rungs"].values())
    rule_label = f"ridge, 756-feat 1 fold {ridge_best:.3f}"

    fig, axes = P.panels(
        1, 2, figsize=(11.4, 4.6), theme=theme,
        title="One configuration beat an early control; the committed one reverses that",
        subtitle="A linear ridge is the honest control for a model whose "
                 "recurrence was inert. Both sweeps and this control are "
                 "756-feature single-fold numbers, comparable to each other "
                 f"and to nothing else: measured on matched folds the ridge "
                 f"reaches {committed:.3f} and the model does not beat it at "
                 "any penalty (see `ridge-control`).",
        gridspec_kw={"width_ratios": [1.15, 1.0]})

    # Scatter uses the ALL-PAIRS separation test, which only the first three
    # categorical slots clear -- so colour encodes learning rate (3 values) and
    # nothing else. A fourth dimension would become a facet, not a fourth hue.
    ax = axes[0]
    lrs = sorted({lbl.split()[0] for lbl in second}, reverse=True)
    for i, lr in enumerate(lrs[:P.SERIES_ALL_PAIRS_CAP]):
        pts = [(v["frac03"], v["pcc"], k) for k, v in second.items()
               if k.startswith(lr)]
        ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=70,
                   color=theme.series[i], label=lr, zorder=3,
                   edgecolor=theme.surface, linewidth=2)
        for fr, pc, k in pts:
            # Flip the label to the left of a point in the right half, so a
            # long config string cannot run off the panel.
            right = fr > 0.33
            ax.annotate(k.split(" ", 1)[1], (fr, pc), fontsize=6.5,
                        textcoords="offset points",
                        xytext=(-9 if right else 9, -3),
                        ha="right" if right else "left", color=theme.muted)
    P.reference_rule(ax, ridge_best, rule_label, theme=theme)
    P.pcc_axis(ax, P.PER_MOLECULE)
    ax.set_xlabel("fraction of genes above PCC 0.3")
    ax.set_title("Second sweep, coloured by learning rate", loc="left")
    ax.set_xlim(0.10, 0.47)
    ax.set_ylim(0, 0.31)
    ax.legend(loc="lower right", title="lr")
    P.grid(ax, "y", theme)

    ax = axes[1]
    items = sorted(first.items(), key=lambda kv: kv[1]["pcc"])
    ys = np.arange(len(items))
    P.bars(ax, ys, [v["pcc"] for _, v in items], color=theme.muted,
           theme=theme, fmt="{:.3f}", horizontal=True)
    ax.set_yticks(ys, [k.replace("depth", "d") for k, _ in items], fontsize=6.5)
    # Shorter text and a wider axis on this panel only: a vertical rule's label
    # runs rightward from the rule with clipping off, so the long form reaches
    # past the figure edge and is silently cut at save time.
    P.reference_rule(ax, ridge_best, f"756-feat 1 fold  {ridge_best:.3f}",
                     theme=theme, orient="v")
    P.pcc_axis(ax, P.PER_MOLECULE, on="x")
    ax.set_xlim(0, max(ridge_best, max(v["pcc"] for _, v in items)) * 1.45)
    ax.set_title("First sweep — every config below the control", loc="left")
    P.grid(ax, "x", theme)

    P.finish(fig, bottom=0.02, script="scripts/15_figures.py",
                 sources=["results/rnn_tuning.json", "results/rnn_tuning2.json",
                          "results/discrepancy_analysis.json",
                          "results/ridge_control.json"], theme=theme)
    return fig


@figure(
    name="ridge-control",
    title="The committed ridge control: a linear map beats the RNN at every penalty",
    group="internals",
    requires=("results/ridge_control.json", "results/transcriptome_loco.json"),
    tier=1,
    caveat="TWO PANELS BY AXIS -- the heights must not be read across. Both "
           "compare against the SAME model run under the SAME folds, encoder "
           "and targets, which is the entire reason this figure exists and the "
           "0.236 in `hyperparameter-tuning` cannot be used the same way.",
)
def ridge_control(theme=P.LIGHT):
    """The control the repository can re-run, drawn on both axes.

    This is the figure `hyperparameter-tuning` could not be. That panel's ridge
    is a single held-out fold at 756 features from an audit pass with no
    committed script; quoting it against the 5-fold 603-feature headline put a
    comparison in four documents that had never been run, and it was withdrawn
    A dedicated ridge control replaces it with one whose config block
    is identical to the model's, and `ridge_control.json` records the model's
    own number under `model_for_comparison` so the two cannot drift apart.

    The sweep is the point. A control fixed at one penalty makes its own
    strength an unexamined choice; swept, the claim survives its weakest rung.
    """
    d = results_json("ridge_control")
    cfg, rungs = d["config"], d["rungs"]
    model_pm = d["model_for_comparison"]["pcc_mean"]
    model_pp = results_json("transcriptome_loco")["depth_2"]["all_genes"]["pcc_row_mean"]

    keys = sorted(rungs, key=lambda k: rungs[k]["alpha"])
    labels = [f"{rungs[k]['alpha']:g}" for k in keys]
    xs = np.arange(len(keys))

    fig, axes = P.panels(
        1, 2, figsize=(11.4, 4.6), theme=theme,
        title="A closed-form ridge beats the relaxation RNN, on the same folds, at every penalty",
        subtitle=f"Identical protocol on both sides: {cfg['n_folds_effective']} "
                 f"folds, {cfg['n_profiles']:,} profiles, "
                 f"{cfg['n_conditions']} conditions, {cfg['n_features']} "
                 f"features, same encoder and targets. The penalty is SWEPT so "
                 f"the control's strength is not an unexamined choice.")

    # The two axes do NOT tell the same story, and a shared caption template
    # would hide that. Per molecule the separation is the result; per profile
    # every predictor here scores ~0.6 because the axis is dominated by the
    # mean expression profile every condition shares, so the margin is real in
    # sign and negligible in size. Saying so is the point of drawing both.
    panels = (
        (axes[0], "pcc_mean", model_pm, P.PER_MOLECULE, "Per molecule", 0.99,
         "the separation is the result: the worst rung (α = {a}) still clears "
         "the model by {m:+.3f}, so it does not depend on tuning the control"),
        (axes[1], "pcc_row_mean", model_pp, P.PER_PROFILE,
         "Per profile — the paper's axis", 0.45,
         "the sign holds but the size does not: the worst rung (α = {a}) "
         "clears the model by only {m:+.3f}. This axis compresses everything "
         "toward the shared mean profile"),
    )
    for ax, key, model, which, name, at, cap in panels:
        vals = [rungs[k][key] for k in keys]
        P.bars(ax, xs, vals, color=theme.series[0], theme=theme, fmt="{:.3f}")
        P.reference_rule(ax, model, f"relaxation RNN {model:.3f}", theme=theme,
                         color=P.STATUS["critical"], at=at)
        P.pcc_axis(ax, which)
        ax.set_xticks(xs, labels, fontsize=8)
        ax.set_xlabel("ridge penalty  α")
        ax.set_ylim(0, max(max(vals), model) * 1.28)
        ax.set_title(name, loc="left")
        P.grid(ax, "y", theme)
        # The margin quoted is the WORST rung's, not the best's: a control
        # tuned to win proves nothing.
        worst = min(vals)
        P.caption(ax, cap.format(a=labels[vals.index(worst)],
                                 m=worst - model), theme=theme)

    P.finish(fig, script="scripts/15_figures.py",
             sources=["results/ridge_control.json",
                      "results/transcriptome_loco.json"], theme=theme,
             note="scripts/18_ridge_control.py, scripts/03_train_moma.py")
    return fig


# Metrics that are unitless and directly comparable between the two runs.
_DEVICE_METRICS = [
    ("PCC per molecule", lambda d: d["all_genes"]["pcc_mean"]),
    ("genes above 0.3", lambda d: d["all_genes"]["frac_above_0.3"]),
    ("TF subset PCC", lambda d: d["tfs"]["pcc_mean"]),
    ("RMSE", lambda d: d["all_genes"]["rmse"]),
    ("calibration slope", lambda d: d["all_genes"]["calibration_slope"]),
]


@figure(
    name="device-replicate",
    title="The same LOCO on two machines, two torch builds, two BLAS",
    group="internals",
    requires=("results/transcriptome_loco_wydecay_superseded.json",
              "results/transcriptome_loco_gpu_replicate.json"),
    tier=1,
    caveat="Measured on the PRE-RECURRENCE configuration (headline 0.235), "
           "not the current one. Kept because what it establishes — the number "
           "is a property of the model, not of one machine — does not depend "
           "on which configuration was used.",
)
def device_replicate(theme=P.LIGHT):
    cpu = results_json("transcriptome_loco_wydecay_superseded")["depth_2"]
    gpu = results_json("transcriptome_loco_gpu_replicate")["depth_2"]

    fig, ax = P.panels(
        figsize=(8.2, 4.4), theme=theme,
        title="Agreement between an independent CPU and GPU run",
        subtitle="CPU: torch 2.12.1+cpu, Windows, 14 threads. GPU: torch "
                 "2.13.0+cu130, RTX 3090 under WSL2. Nothing but the numeric "
                 "stack differs.")

    ys = np.arange(len(_DEVICE_METRICS))[::-1]
    for y, (name, get) in zip(ys, _DEVICE_METRICS):
        a, b = get(cpu), get(gpu)
        ax.plot([a, b], [y, y], color=theme.grid, linewidth=3, zorder=1,
                solid_capstyle="round")
        ax.scatter([a], [y], s=52, color=theme.series[0], zorder=3)
        ax.scatter([b], [y], s=52, color=theme.series[1], zorder=3)
        ax.text(max(a, b) + 0.02, y, f"Δ {abs(a - b):.5f}", va="center",
                fontsize=7.5, color=theme.muted)

    ax.set_yticks(ys, [m[0] for m in _DEVICE_METRICS])
    ax.set_xlabel("value (all five are unitless)")
    ax.set_xlim(0, 1.15)
    P.grid(ax, "x", theme)
    handles = [P.legend_proxy(ax, s=52, color=theme.series[0], label="CPU"),
               P.legend_proxy(ax, s=52, color=theme.series[1], label="GPU")]
    ax.legend(handles=handles, loc="lower right")

    fig.text(0.02, 0.045,
             "The residual is floating-point reduction order: CUDA does not "
             "promise to sum in the same sequence as a CPU BLAS. Agreement to "
             "1e-3 on the headline metric — well inside the 0.122 spread "
             "across genes — means the number characterizes the model. Wall "
             "clock is deliberately NOT plotted: the two runs were concurrent "
             "and competed for CPU, so the ratio is not a device benchmark.",
             fontsize=7.5, color=theme.ink_secondary, wrap=True)

    P.finish(fig, bottom=0.14, script="scripts/15_figures.py",
                 sources=["results/transcriptome_loco_wydecay_superseded.json",
                          "results/transcriptome_loco_gpu_replicate.json"],
                 theme=theme)
    return fig


@figure(name="epochs-trajectory", group="internals",
        title="More training does not rescue the paper's architecture",
        requires=("results/paper_faithful_epochs600.json",
                  "results/paper_faithful_architecture.json"),
        tier=1)
def epochs_trajectory(theme=P.LIGHT):
    """The rung that separated architecture from undertraining.

    Drawn as ONE panel with a floor line, because the finding is a relationship
    between two quantities and not a value: the curve flattens *below* the line.
    A bar chart of endpoint-versus-floor would state the same numbers and lose
    the reason they settle anything, which is the SHAPE of the approach.
    """
    tr = results_json("paper_faithful_epochs600")
    floor = results_json("paper_faithful_architecture")["baselines"]["mean"]["pcc"]

    pts = sorted((int(e), s["pcc"]) for e, s in tr["trajectory"].items())
    xs = [e for e, _ in pts]
    ys = [v for _, v in pts]

    fig, ax = P.panels(
        1, 1, figsize=(7.6, 4.8), theme=theme,
        title="Is the paper's fixed 100 epochs simply too few?",
        subtitle="Supplementary Methods 3.3.3 fixes 100 epochs and reports that "
                 "convergence is always reached within them -- a claim made on "
                 "178 transcription factors, not 4,096 genes. If the "
                 "architecture's deficit were undertraining, this curve would "
                 "cross the floor.")

    ax.plot(xs, ys, color=theme.series[0], marker="o", zorder=3,
            label="paper's architecture (measured)")

    # The floor is OUR measured baseline on the same folds, not the paper's
    # value, so it is drawn as a plain reference line rather than paper_band.
    ax.axhline(floor, color=theme.series[3], linestyle="--", linewidth=1.4,
               zorder=2, label=f"mean profile, no parameters ({floor:.3f})")

    # Everything more training can still buy, drawn where it would land. The
    # bracket sitting entirely below the line IS the finding.
    a = tr["analysis"]
    top = ys[-1] + a["sum_of_all_remaining_increments"]
    ax.annotate("", xy=(xs[-1], top), xytext=(xs[-1], ys[-1]),
                arrowprops=dict(arrowstyle="<->", color=theme.series[2], lw=1.3))
    # Anchored at the TOP of the bracket and growing downward: centred, the
    # three-line label was tall enough to overlap the floor line it exists to
    # stay below -- which inverted what the figure is showing.
    ax.text(xs[-1] * 1.06, top,
            f"all remaining\ntraining:\n+{a['sum_of_all_remaining_increments']:.4f}",
            fontsize=7.5, color=theme.series[2], va="top", ha="left")

    ax.set_xscale("log", base=2)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(x) for x in xs])
    ax.set_xlabel("training epochs  (log scale)")
    ax.axvline(100, color=theme.muted, linewidth=1.0, zorder=1)
    ax.text(100, 0.02, " paper: 100", transform=ax.get_xaxis_transform(),
            ha="left", va="bottom", fontsize=7.5, color=theme.muted)
    P.pcc_axis(ax, P.PER_PROFILE)
    P.grid(ax, "y", theme)
    # Lower LEFT: the 100-epoch rule and the budget bracket both sit on the
    # right, and a legend there hid the "paper: 100" marker entirely.
    # The four occupied regions are the rising curve (left), the floor line
    # (top), the 100-epoch rule (centre) and the budget bracket (right edge).
    # Lower right is what is left, and it only became free once xlim was
    # widened to make room for the bracket label.
    ax.legend(loc="lower right", fontsize=8)
    ax.set_xlim(xs[0] * 0.85, xs[-1] * 1.55)
    ax.set_ylim(top=floor + 0.035)      # headroom, so the floor is a line and
                                        # not the top edge of the panel

    P.caption(ax,
              f"Measured to epoch {xs[-1]} on {tr['trajectory'][str(xs[-1])]['n_conditions']} "
              f"conditions. Doubling the paper's budget buys "
              f"+{a['deltas'][str(xs[-1])]:.4f} and leaves it "
              f"{a['gap_to_floor_at_200']:.4f} below the floor; increments decay "
              f"~{sum(a['delta_ratio_per_doubling'])/2:.2f} per doubling, so every "
              f"one remaining is worth {a['sum_of_all_remaining_increments']:.4f} "
              f"-- under half the gap. 400 and 600 were not reached",
              theme=theme)
    P.finish(fig, script="scripts/15_figures.py",
             sources=["results/paper_faithful_epochs600.json",
                      "results/paper_faithful_architecture.json"], theme=theme,
             note="scripts/16_paper_faithful_architecture.py --rungs epochs_600")
    return fig

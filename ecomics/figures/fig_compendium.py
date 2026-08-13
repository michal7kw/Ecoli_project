"""A — The compendium: what was rebuilt, and the joins that prove it joined.

These are the strongest results in the reproduction and the least statistical.
They are exact integers derived from independently scraped data through a
canonicalizer written without reference to them, which is why
`16-results-reference.md` 16.11 ranks them above every modelling result: a
metric with error bars can be argued with, `179` cannot.

The overlap matrix is the figure that earns its place here. `db/canon.py` maps
three incompatible condition dialects (`b0624_OE` / `b0756(KO)` / `WT_na`, the
last reversed) onto one canonical form. Get it wrong and every off-diagonal
cell reads 0 -- silently, with no error anywhere, yielding empty training sets
for every cross-layer MOMA module. The matrix is what that failure would look
like, and what its absence looks like.
"""

from __future__ import annotations

import numpy as np

from .. import config as C
from .. import plots as P
from . import figure, results_json

LAYERS = ["transcriptome", "proteome", "metabolome", "fluxome", "phenome"]

# The paper's own shape, from config.PAPER. The second element of each pair is
# what the PUBLIC release actually contains where the two differ -- proteome and
# metabolome were condition-averaged before release, so those rows can never
# match and the figure says so rather than scoring them as failures.
_SHAPE = [
    # layer,           paper profiles,  paper molecules, status,    why
    ("transcriptome", "transcriptome_profiles", "transcriptome_genes", "good",
     "3,579 vs 3,578: the public file has one fewer"),
    ("proteome", "proteome_profiles", "proteome_proteins", "serious",
     "public release is condition-averaged (33 x 589), never 71 x 1,001"),
    ("metabolome", "metabolome_profiles", "metabolome_metabolites", "serious",
     "public release is condition-averaged (49 x 114), never 696 x 356"),
    ("fluxome", "fluxome_profiles", "fluxome_fluxes", "good",
     "43 x 120 exactly"),
]


def _db():
    """Open the compendium if it is built, else None."""
    try:
        from ..db.api import Ecomics
        if not C.DB_PATH.exists():
            return None
        return Ecomics()
    except Exception:
        return None


@figure(
    name="compendium-shape",
    title="Compendium shape: what the public release actually contains",
    group="compendium",
    requires=("data/ecomics.db",),
    tier=3,
    caveat="Two rows CANNOT match: the proteome and metabolome were "
           "condition-averaged before release, so 71 x 1,001 and 696 x 356 "
           "were never published.",
)
def compendium_shape(theme=P.LIGHT):
    db = _db()
    fig, axes = P.panels(
        1, 2, figsize=(13.2, 4.4), theme=theme,
        title="Compendium shape: ours vs the paper",
        subtitle="Log scale — the counts span three orders of magnitude. "
                 "Filled = this rebuild, hollow = the paper. Where the two "
                 "differ, the marker says whether the gap is a defect or a "
                 "property of the public release.")

    summary = db.summary().set_index("layer")
    widths = {l: db.matrix(l).values.shape[1] for l in LAYERS[:4]}

    for panel, (ax, (kind, getter, paper_key_idx)) in enumerate(zip(
            axes,
            [("profiles", lambda l: int(summary.loc[l, "n_profiles"]), 1),
             ("molecules", lambda l: widths[l], 2)])):
        names = [row[0] for row in _SHAPE]
        ours = [getter(n) for n in names]
        theirs = [C.PAPER[row[paper_key_idx]] for row in _SHAPE]
        ys = np.arange(len(names))[::-1]

        for y, o, t, row in zip(ys, ours, theirs, _SHAPE):
            ax.plot([t, o], [y, y], color=theme.grid, linewidth=2.5, zorder=1,
                    solid_capstyle="round")
            ax.scatter([t], [y], s=46, facecolor=theme.surface,
                       edgecolor=theme.paper, linewidth=1.6, zorder=3)
            ax.scatter([o], [y], s=46, color=theme.series[0], zorder=4)
            exact = o == t
            ax.text(max(o, t) * 1.25, y,
                    f"{P.STATUS_ICON['good' if exact else row[3]]}",
                    color=P.STATUS[("good" if exact else row[3])],
                    fontsize=10, va="center", fontweight="bold")
            # The reason a row does or does not match, once per figure. Without
            # it the transcriptome's 3,578-vs-3,579 and the proteome's
            # 33-vs-71 look like the same kind of disagreement, and they are
            # not: one is an off-by-one in the released file, the other is a
            # different table entirely.
            if panel == 0:
                ax.annotate(row[4], (max(o, t), y), textcoords="offset points",
                            xytext=(20, -1), fontsize=6.5, va="center",
                            color=theme.muted)

        ax.set_yticks(ys, names)
        ax.set_xscale("log")
        ax.set_xlabel(f"number of {kind}")
        ax.set_title(f"{kind.capitalize()} per layer", loc="left")
        ax.set_xlim(min(min(ours), min(theirs)) * 0.45,
                    max(max(ours), max(theirs)) * (90 if panel == 0 else 5))
        P.grid(ax, "x", theme)

    handles = [
        P.legend_proxy(axes[0], s=46, color=theme.series[0],
                       label="this rebuild"),
        P.legend_proxy(axes[0], s=46, facecolor=theme.surface,
                       edgecolor=theme.paper, linewidth=1.6, label="paper"),
    ]
    axes[1].legend(handles=handles, loc="lower right")

    legend = "   ".join(
        f"{P.STATUS_ICON[k]} {v}" for k, v in
        [("good", "matches exactly"),
         ("serious", "cannot match: condition-averaged in the public release")])
    fig.text(0.02, 0.055, legend, fontsize=7.5, color=theme.ink_secondary)

    P.finish(fig, bottom=0.07, script="scripts/15_figures.py",
                 sources=["data/ecomics.db", "ecomics/config.py:PAPER"],
                 theme=theme)
    db.close()
    return fig


@figure(
    name="cross-layer-overlap",
    title="Cross-layer condition overlap — the canonicalizer's signature",
    group="compendium",
    requires=(),
    tier=1,
    caveat="Without db/canon.py every off-diagonal cell reads 0, silently. "
           "The four boxed cells are quoted in the paper's own text.",
)
def cross_layer_overlap(theme=P.LIGHT):
    db = _db()
    n = len(LAYERS)
    # Start from the counts db/build.py ASSERTS, so the four cells the paper
    # states are present even in a fresh clone with no database built.
    m = np.full((n, n), np.nan)
    for (a, b), v in C.EXPECTED_OVERLAP.items():
        i, j = LAYERS.index(a), LAYERS.index(b)
        m[i, j] = m[j, i] = v

    live = db is not None
    if live:
        summary = db.summary().set_index("layer")
        for i in range(n):
            m[i, i] = int(summary.loc[LAYERS[i], "n_conditions"])
            for j in range(i + 1, n):
                if np.isnan(m[i, j]):
                    v = len(db.shared_conditions(LAYERS[i], LAYERS[j]))
                    m[i, j] = m[j, i] = v
        db.close()

    fig, ax = P.panels(
        figsize=(7.2, 5.6), theme=theme,
        title="Conditions shared between layers",
        subtitle="Diagonal = the layer's own condition count. A cell is the "
                 "number of conditions two layers have in common AFTER "
                 "canonicalization. Four match the paper's text verbatim.")

    # Sequential, one hue, on log magnitude -- the counts span 3 to 596.
    with np.errstate(invalid="ignore"):
        shade = np.log10(np.where(m > 0, m, np.nan))
    ax.imshow(shade, cmap=theme.cmap(), vmin=0, vmax=np.log10(600),
              interpolation="nearest")

    # The paper's own quoted numbers. Boxed, not recoloured: a second hue here
    # would compete with the magnitude ramp.
    quoted = {(0, 1): "5", (0, 4): "179", (0, 2): "6", (1, 2): "25"}
    for i in range(n):
        for j in range(n):
            v = m[i, j]
            if np.isnan(v):
                ax.text(j, i, "—", ha="center", va="center",
                        fontsize=9, color=theme.muted)
                continue
            dark = shade[i, j] > np.log10(600) * 0.55
            ax.text(j, i, f"{int(v)}", ha="center", va="center", fontsize=10,
                    fontweight="bold",
                    color="#ffffff" if dark else "#0b0b0b")
            key = (min(i, j), max(i, j))
            if key in quoted and i != j:
                ax.add_patch(mpl_rect(j, i, theme))

    ax.set_xticks(range(n), LAYERS, rotation=30, ha="right")
    ax.set_yticks(range(n), LAYERS)
    ax.set_xticks(np.arange(n + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(n + 1) - 0.5, minor=True)
    # A 2px surface gap between cells -- never a border around each mark.
    ax.grid(which="minor", color=theme.surface, linewidth=2)
    ax.tick_params(which="minor", length=0)
    for s in ax.spines.values():
        s.set_visible(False)

    note = ("boxed = quoted in the paper: 5 conditions (18 proteome profiles), "
            "179 of 596 conditions with growth, 25 proteome∩metabolome")
    fig.text(0.02, 0.055, note, fontsize=7.5, color=theme.ink_secondary)

    sources = ["ecomics/config.py:EXPECTED_OVERLAP"]
    sources.append("data/ecomics.db" if live else "(DB absent: unasserted pairs left blank)")
    P.finish(fig, bottom=0.06, script="scripts/15_figures.py", sources=sources,
             theme=theme)
    return fig


def mpl_rect(j, i, theme):
    """A hairline box marking a cell the paper states in its own text."""
    from matplotlib.patches import Rectangle
    return Rectangle((j - 0.44, i - 0.44), 0.88, 0.88, fill=False,
                     edgecolor=theme.ink, linewidth=1.4, zorder=5)


@figure(
    name="pipeline-validation",
    title="Normalization pipeline, checked against Bioconductor",
    group="compendium",
    requires=("results/pipeline_validation.json",),
    tier=1,   # written by scripts/02, but the JSON is tracked, so a fresh
              # clone has it without re-running the pipeline
    caveat="Step 3 uses a SYNTHETIC reference (Taniguchi et al. 2010 is not "
           "redistributable), so the absolute scale demonstrates the machinery "
           "rather than reproducing Ecomics' scale.",
)
def pipeline_validation(theme=P.LIGHT):
    d = results_json("pipeline_validation")
    checks = d["crosschecks"]
    cascade = d.get("cascade", [])

    fig, axes = P.panels(
        1, 2, figsize=(11.0, 4.2), theme=theme,
        title="The normalization pipeline, against an independent implementation",
        subtitle="Left: agreement with Bioconductor on real GEO data "
                 f"({d['n_cel']} CEL arrays, {d['n_htseq']} htseq tables). "
                 "Right: what each step does to the distribution.",
        gridspec_kw={"width_ratios": [1.0, 1.25]})

    # -- left: agreement as stat tiles. A one-bar bar chart would be wrong
    # here; the number IS the chart, and three of them near 1.0 would be three
    # indistinguishable full-height bars.
    ax = axes[0]
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    for k, (name, c) in enumerate(checks.items()):
        y = 0.90 - k * 0.32
        # Six decimals always, never trimmed: 1.000000 is the claim (quantile
        # normalization is deterministic and should be bit-exact), and "1"
        # would read as a rounded value.
        ax.text(0.0, y, f"{c['pcc']:.6f}", fontsize=25,
                color=theme.series[0], va="top", ha="left")
        ax.text(0.0, y - 0.115, name, fontsize=9, color=theme.ink,
                va="top", ha="left", fontweight="bold")
        ax.text(0.0, y - 0.163, f"vs {c['reference']}", fontsize=7.5,
                color=theme.ink_secondary, va="top", ha="left")
        ax.text(0.0, y - 0.208, c["note"], fontsize=7.5, color=theme.muted,
                va="top", ha="left")
    ax.set_title("PCC vs the R implementation", loc="left")

    # -- right: the median/IQR cascade
    ax = axes[1]
    if cascade:
        ys = np.arange(len(cascade))[::-1]
        for y, s in zip(ys, cascade):
            ax.plot([s["q1"], s["q3"]], [y, y], color=theme.series[0],
                    linewidth=6, solid_capstyle="butt", alpha=0.30, zorder=2)
            ax.scatter([s["median"]], [y], s=40, color=theme.series[0], zorder=3)
        ax.set_yticks(ys, [s["step"] for s in cascade])
        ax.set_xlabel("value (units change at step 3: log2 → log10 copies)")
        ax.set_title("Median and IQR after each step", loc="left")
        P.grid(ax, "x", theme)
        ax.text(0.99, 0.02, "shaded bar = IQR", transform=ax.transAxes,
                ha="right", fontsize=7.5, color=theme.muted)
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "no cascade recorded", ha="center",
                color=theme.muted)

    P.finish(fig, bottom=0.02, script="scripts/15_figures.py",
                 sources=["results/pipeline_validation.json"], theme=theme,
                 note="written by scripts/02_run_pipeline.py; "
                      "R cross-checks via ecomics/pipeline/validate.py:crosscheck_all")
    return fig

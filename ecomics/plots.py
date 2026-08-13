"""House style for the figure atlas, and the primitives that encode this repo's rules.

Nothing here draws a specific result -- `ecomics/figures/*.py` does that. This
module exists so that the *correct* chart is the cheap one to write and the
incorrect one is awkward, because the failure modes for figures in this
repository are not aesthetic. They are the same four that `metrics.py`,
`evaluate.py` and `scripts/04_reproduce.py` already guard against in numbers:

  1. A PCC drawn without its AXIS. The same predictions score -0.10 per molecule
     and +0.58 per profile. A bar labelled only "PCC" is
     not an incomplete figure, it is a wrong one -- so `pcc_axis()` is the only
     supported way to label a PCC scale, and `audit_pcc_panels()` fails any
     figure that labels one some other way.

  2. Ours drawn beside the paper's as though they were the same measurement.
     `paper_band()` draws the paper's value as a recessive band spanning its own
     standard deviation, never as a bar next to ours. A band reads as "the
     region the paper reports"; a neighbouring bar reads as "these two numbers
     are comparable", which for most rows in this repository they are not.

  3. A number with no provenance. `provenance()` stamps the producing script and
     every file read into the figure's own footer, so a figure lifted into a
     slide still says where it came from.

  4. A suppressed metric drawn as a value. `not_applicable()` draws the
     `n/a (5 cond)` slot that `reporting.py:_pcc_str` prints. A 5-point
     correlation is noise; plotting it as a short bar makes it look like a weak
     result rather than an absent one.

A fifth rule comes from outside: NO DUAL-AXIS CHARTS, ever. Two y-scales on one
plot invent a correlation the data does not contain, and the alignment of the
two scales is arbitrary. Two measures of different scale get two panels.
`has_dual_axis()` detects the violation and the figure tests assert it
over every builder.

A sixth was learned rather than imported. A figure's declared inputs are checked
for EXISTENCE, which is not the same as containing anything: `memory-depth` drew
a legend entry, a paper band and a caption around an empty curve for a day,
because the file it names kept its name and lost its contents. `audit_empty_series()`
fails a series that came back empty; `legend_proxy()` marks the one case where
an empty artist is the point, so the guard never has to guess.

Palette
-------
The categorical order below is the validated default from the `dataviz` skill's
reference instance, adopted unchanged. Its measured separations, which any edit
has to re-earn by re-running that skill's `scripts/validate_palette.js`:

    adjacent pairlist (bars, lines, stacks -- the default)
        worst CVD dE            9.1 light / 8.4 dark   (target >= 8)
        worst normal-vision dE 19.6 light / 19.3 dark  (floor  >= 15)

    all-pairs pairlist (scatter, matrices, small multiples)
        only the FIRST THREE SLOTS clear the floors, in both modes
        (worst CVD dE 9.2 light / 9.4 dark; normal-vision 24.0 / 20.9).

That last line is a constraint on the figures, not a footnote: `fig_internals.py`'s
tuning scatter and `fig_compendium.py`'s overlap matrix cap at three categorical
series and facet beyond, rather than reaching for a fourth hue.

Status colours are a separate, reserved set -- never used for a series, and
never carrying meaning alone: `status_marker()` always emits an icon and a text
label beside the colour.

Import guard
------------
matplotlib is an OPTIONAL dependency (see requirements.txt). Importing this
module without it raises a message naming the install command rather than a
bare ImportError from three frames down.
"""

from __future__ import annotations

import textwrap
from contextlib import contextmanager
from pathlib import Path

try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
except ImportError as exc:                                    # pragma: no cover
    raise ImportError(
        "The figure atlas needs matplotlib, which is an optional dependency of "
        "this repository.\n"
        "    pip install 'matplotlib>=3.7'\n"
        "Everything in the 00->01->(02|03|04) chain runs without it."
    ) from exc

from . import config as C

__all__ = [
    "LIGHT", "DARK", "STATUS", "STATUS_ICON", "SERIES_ALL_PAIRS_CAP",
    "Theme", "style", "panels", "finish", "caption", "save", "provenance", "pcc_axis",
    "pcc_axis_both", "paper_band",
    "not_applicable", "status_marker", "reference_rule", "annotate_n",
    "has_dual_axis", "audit_pcc_panels", "audit_empty_series", "legend_proxy",
    "figures_dir",
    "PER_MOLECULE", "PER_PROFILE",
]

# --------------------------------------------------------------------------
# Palette
# --------------------------------------------------------------------------
# Eight categorical slots, in a FIXED order. The order is the colour-vision
# safety mechanism, not a preference -- reordering invalidates the separations
# quoted in the module docstring. A ninth series is never a generated hue; it
# folds into "other" or becomes a facet.
SERIES_LIGHT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SERIES_DARK = ["#3987e5", "#d95926", "#199e70", "#c98500",
               "#d55181", "#008300", "#9085e9", "#e66767"]

# Scatter/matrix forms use the all-pairs separation test, which only the first
# three slots clear. Past three: facet, do not add a hue.
SERIES_ALL_PAIRS_CAP = 3

# Reserved. `good` = reproduced, `warning` = partial, `serious` = undecidable on
# public data, `critical` = did not reproduce. Mode-invariant by design.
STATUS = {"good": "#0ca30c", "warning": "#fab219",
          "serious": "#ec835a", "critical": "#d03b3b"}

# Colour never carries status alone; these travel with it. All four are in
# DejaVu Sans, which is why it leads the font stack below.
STATUS_ICON = {"good": "✓", "warning": "△",
               "serious": "◆", "critical": "✗"}

STATUS_WORD = {"good": "reproduced", "warning": "partial",
               "serious": "undecidable", "critical": "not reproduced"}

# One hue, light -> dark. For magnitude only; never a rainbow.
SEQUENTIAL = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec",
              "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab",
              "#184f95", "#104281", "#0d366b"]

# Two hues that read as opposite, with a NEUTRAL midpoint. Used for
# ours-minus-paper deltas, where the sign is the point.
DIVERGING = ("#2a78d6", "#f0efec", "#d03b3b")
DIVERGING_DARK = ("#3987e5", "#383835", "#e66767")


class Theme:
    """Resolved colours for one surface. Built by `style()`; read by builders."""

    def __init__(self, mode: str):
        dark = mode == "dark"
        self.mode = mode
        self.series = SERIES_DARK if dark else SERIES_LIGHT
        self.status = dict(STATUS)
        self.sequential = list(reversed(SEQUENTIAL)) if dark else list(SEQUENTIAL)
        self.diverging = DIVERGING_DARK if dark else DIVERGING
        self.surface = "#1a1a19" if dark else "#fcfcfb"
        self.ink = "#ffffff" if dark else "#0b0b0b"
        self.ink_secondary = "#c3c2b7" if dark else "#52514e"
        self.muted = "#898781"
        self.grid = "#2c2c2a" if dark else "#e1e0d9"
        self.baseline = "#383835" if dark else "#c3c2b7"
        # The paper's reported region, and any "not comparable" chrome. Neutral
        # on purpose: it must not read as one of our series.
        self.paper = "#898781"

    def cmap(self):
        """Sequential colormap, for magnitude only."""
        return mpl.colors.LinearSegmentedColormap.from_list(
            "ecomics_seq", self.sequential)


LIGHT, DARK = Theme("light"), Theme("dark")


# --------------------------------------------------------------------------
# The two PCC axes. Never a bare "PCC".
# --------------------------------------------------------------------------
PER_MOLECULE = "molecule"
PER_PROFILE = "profile"

_AXIS_TEXT = {
    PER_MOLECULE: "PCC per molecule, across conditions",
    PER_PROFILE: "PCC per profile, across molecules",
}
_AXIS_NOTE = {
    # Why each axis is the one being shown -- the sentence a reader needs in
    # order not to compare the two panels' heights.
    PER_MOLECULE: "a condition-blind predictor scores ~0 here by construction",
    PER_PROFILE: "the paper's axis (Suppl. Methods 3.3.3)",
}


def pcc_axis(ax, which: str, *, on: str = "y", note: bool = False):
    """Label a PCC scale, and register that it was labelled.

    This is the ONLY supported way to put a PCC on an axis. `audit_pcc_panels`
    walks a finished figure and fails any axis whose label mentions PCC without
    having come through here, which is what stops a later edit from
    reintroducing an unlabelled "PCC" bar chart -- the single mistake that made
    the transcriptome read "ours 0.287 vs the paper's 0.54" for most of this
    reproduction.
    """
    if which not in _AXIS_TEXT:
        raise ValueError(f"axis must be {PER_MOLECULE!r} or {PER_PROFILE!r}, "
                         f"not {which!r}")
    label = _AXIS_TEXT[which]
    if note:
        label += f"\n({_AXIS_NOTE[which]})"
    (ax.set_ylabel if on == "y" else ax.set_xlabel)(label, fontsize=8)
    registered = getattr(ax.figure, "_ecomics_pcc_axes", None)
    if registered is None:
        registered = ax.figure._ecomics_pcc_axes = set()
    registered.add(id(ax))
    return ax


def pcc_axis_both(ax, *, on: str = "y"):
    """Label a panel that deliberately shows BOTH axes at once.

    The documented exception, and there is exactly one class of it: a panel
    whose finding IS the divergence between the two axes. The metabolome
    penalty sweep is the case -- per-profile PCC walks up into the paper's band
    while per-molecule PCC collapses negative, and separating them into two
    panels would destroy the observation.

    It registers the same way `pcc_axis` does, so the exception is explicit and
    greppable rather than a silent bypass of the audit. The series legend must
    name which curve is which axis; the label alone is not enough.
    """
    (ax.set_ylabel if on == "y" else ax.set_xlabel)(
        "PCC — BOTH axes plotted\n(the divergence is the finding; see legend)",
        fontsize=8)
    registered = getattr(ax.figure, "_ecomics_pcc_axes", None)
    if registered is None:
        registered = ax.figure._ecomics_pcc_axes = set()
    registered.add(id(ax))
    return ax


def audit_pcc_panels(fig) -> list[str]:
    """Axis labels that mention PCC but did not come through `pcc_axis`.

    Returns a list of human-readable offences; empty means the figure is clean.
    """
    registered = getattr(fig, "_ecomics_pcc_axes", set())
    bad = []
    for i, ax in enumerate(fig.axes):
        if id(ax) in registered:
            continue
        for name, text in (("y", ax.get_ylabel()), ("x", ax.get_xlabel())):
            if "PCC" in text and "across" not in text:
                bad.append(f"axes[{i}] {name}-label {text!r} bypassed pcc_axis()")
    return bad


def audit_empty_series(fig) -> list[str]:
    """Series that were asked for and came back with nothing in them.

    `Figure.requires` checks that a declared input EXISTS. It cannot check that
    the input still has anything in it, and `memory-depth` once fell
    through exactly that gap: `scripts/10` was re-run without `--depth-sweep`,
    so `trn_seeded_recurrence.json`'s `depth_sweep` became `[]` while the file
    itself stayed present. The builder went on drawing the legend entry, the
    paper band and the caption around an `ax.plot([], [])`, and nothing
    complained -- the render test only asserts the
    PNG exceeds 5 kB, which an empty panel comfortably does. The figure would
    have shipped asserting a curve it no longer had.

    ⚠ An axes with NO series at all is fine and sometimes required: the
    fluxome's middle panel is text precisely because a bar of height zero would
    be a lie. The offence is not "this panel is empty", it is "this panel asked
    for a series and drew nothing", which is what an empty artist records.

    Only `Line2D`, `PathCollection` (scatter) and bar containers are inspected.
    `PolyCollection` is deliberately skipped: `hexbin` produces one whose
    `get_offsets()` is legitimately empty, so including it would fail
    `transcriptome-calibration` for drawing correctly. Artists built by
    `legend_proxy` are skipped too -- they are empty on purpose, and marking
    them is the only way to tell that apart from this function's quarry.

    Returns a list of human-readable offences; empty means the figure is clean.
    """
    bad = []
    for i, ax in enumerate(fig.axes):
        for ln in ax.lines:
            if _is_proxy(ln) or len(ln.get_xydata()):
                continue
            bad.append(f"axes[{i}] line {ln.get_label()!r} has no points")
        for col in ax.collections:
            if _is_proxy(col) or not isinstance(
                    col, mpl.collections.PathCollection):
                continue
            if len(col.get_offsets()) == 0:
                bad.append(f"axes[{i}] scatter {col.get_label()!r} has no points")
        for cont in getattr(ax, "containers", ()):
            if len(cont) == 0:
                bad.append(f"axes[{i}] bars {cont.get_label()!r} are empty")
    return bad


def _is_proxy(artist) -> bool:
    return getattr(artist, "_ecomics_legend_proxy", False)


def legend_proxy(ax, *, label: str, **kw):
    """An intentionally empty artist, drawn only to become a legend handle.

    Several panels draw their marks one row at a time inside a loop -- one
    `scatter` call per layer, per metric, per condition -- so no single call can
    carry the series label. The standard idiom is an empty scatter that exists
    only to be a handle.

    It has to be built HERE rather than as a bare `ax.scatter([], [])`, because
    `audit_empty_series` fails any series that came back empty and this is
    precisely a series that came back empty. Nothing in the artist distinguishes
    "empty on purpose" from "asked for data and got none" -- the second is the
    bug that guard exists to catch -- so the intent is recorded rather than
    guessed. Same enforcement style as `pcc_axis`, which registers the axes it
    labelled instead of trying to infer the label's provenance afterwards.
    """
    art = ax.scatter([], [], label=label, **kw)
    art._ecomics_legend_proxy = True
    return art


def has_dual_axis(fig) -> bool:
    """True if any two axes overlap exactly -- the signature of `twinx`/`twiny`.

    Two y-scales on one plot make the alignment of the scales arbitrary, so the
    chart asserts a relationship the data does not contain. Two measures of
    different scale get two panels here, always.
    """
    boxes = [tuple(round(v, 6) for v in ax.get_position().bounds)
             for ax in fig.axes]
    return len(boxes) != len(set(boxes))


# --------------------------------------------------------------------------
# Style and layout
# --------------------------------------------------------------------------
_RC = {
    "font.family": "sans-serif",
    # DejaVu Sans FIRST, deliberately. It ships with matplotlib, so every
    # machine renders the atlas identically -- and it is the only font in the
    # default stack that carries the status glyphs below (Segoe UI has none of
    # U+2713 / U+2717 / U+25B3 / U+25C6, and matplotlib does not fall back
    # per-glyph across the list, it warns and draws a box).
    "font.sans-serif": ["DejaVu Sans", "Segoe UI", "Helvetica Neue", "Arial"],
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "axes.axisbelow": True,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "legend.frameon": False,
    "legend.fontsize": 8,
    # Solid hairline grid, one shade off the surface. Never dashed -- dashing
    # reads as "threshold" or "projection" when it is only a grid.
    "grid.linestyle": "-",
    "grid.linewidth": 0.6,
    "lines.linewidth": 2.0,
    "lines.markersize": 5,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,          # embed as TrueType, so text stays selectable
}


@contextmanager
def style(mode: str = "light"):
    """Apply the house style for one surface; yields the `Theme`."""
    theme = DARK if mode == "dark" else LIGHT
    rc = dict(_RC)
    rc.update({
        "figure.facecolor": theme.surface,
        "axes.facecolor": theme.surface,
        "savefig.facecolor": theme.surface,
        "text.color": theme.ink,
        "axes.labelcolor": theme.ink_secondary,
        "axes.edgecolor": theme.baseline,
        "xtick.color": theme.muted,
        "ytick.color": theme.muted,
        "xtick.labelcolor": theme.ink_secondary,
        "ytick.labelcolor": theme.ink_secondary,
        "grid.color": theme.grid,
    })
    with mpl.rc_context(rc):
        yield theme


def panels(nrows: int = 1, ncols: int = 1, *, figsize=None, title: str = "",
           subtitle: str = "", theme: Theme = LIGHT, **kw):
    """A figure with the house title block. Returns `(fig, axes)`.

    `subtitle` is where the caveat that must travel with the figure goes -- the
    axis, the n, or the reason a panel is an annotation rather than a bar.
    """
    figsize = figsize or (4.6 * ncols, 3.6 * nrows + 0.6)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, **kw)
    lines = 0
    if title:
        fig.suptitle(title, x=0.02, y=0.99, ha="left", va="top",
                     fontsize=12, fontweight="bold", color=theme.ink)
        lines += 1
    if subtitle:
        wrapped = textwrap.wrap(subtitle, int(11.5 * figsize[0]))
        fig.text(0.02, 0.99 - 0.24 / figsize[1], "\n".join(wrapped),
                 ha="left", va="top", fontsize=8.5, color=theme.ink_secondary)
        lines += len(wrapped)
    # Remembered so `finish()` can reserve exactly the space the block uses,
    # instead of every builder hand-tuning a `rect` that drifts as soon as the
    # subtitle is reworded. Two of these were already wrong by ~5% of the
    # figure height before this was centralized.
    fig._ecomics_title_lines = lines
    return fig, axes


def finish(fig, *, script: str, sources, theme: Theme = LIGHT, note: str = "",
           bottom: float = 0.0, extra_top: float = 0.0):
    """Lay the figure out under its title block and stamp its provenance.

    `bottom` reserves room for a figure-level footnote drawn with `fig.text`;
    panels whose captions sit in axes coordinates need none, because a tight
    bounding box already grows to include them.
    """
    lines = getattr(fig, "_ecomics_title_lines", 0)
    # ~17 pt for the title line, ~12 pt per wrapped subtitle line, plus a
    # 14 pt gap, converted to a fraction of the figure height.
    block_pt = (15 if lines else 0) + 11 * max(0, lines - 1) + 6
    top = 1.0 - block_pt / (72.0 * fig.get_figheight()) - extra_top
    fig.tight_layout(rect=(0, bottom, 1, max(0.55, top)))
    return provenance(fig, script=script, sources=sources, theme=theme,
                      note=note)


def caption(ax, text: str, *, theme: Theme = LIGHT, dy: float = -34.0):
    """The sentence that must be read with the panel, placed under it.

    Wrapped to the PANEL's width, not the figure's -- matplotlib's `wrap=True`
    measures against the figure, so on a two-panel figure the left caption runs
    straight through the right one. Offset in points below the axes, so it
    clears the x tick labels and the x-axis label whatever their height.

    These carry the caveats: what the axis means, why a value is undefined, why
    a rise is confounded with sample size. They are not decoration -- most of
    them are the difference between a figure that informs and one that misleads.
    """
    width_in = ax.get_position().width * ax.figure.get_figwidth()
    wrapped = "\n".join(textwrap.wrap(text, max(30, int(width_in * 15))))
    ax.annotate(wrapped, xy=(0.5, 0), xycoords="axes fraction",
                xytext=(0, dy), textcoords="offset points",
                ha="center", va="top", fontsize=7.5,
                color=theme.ink_secondary, linespacing=1.5)
    return ax


def grid(ax, axis: str = "y", theme: Theme = LIGHT):
    """Recessive hairline grid on one axis only."""
    ax.grid(True, axis=axis, color=theme.grid, linewidth=0.6, linestyle="-")
    ax.set_axisbelow(True)
    return ax


# --------------------------------------------------------------------------
# Provenance, the visual form of "a number needs a provenance"
# --------------------------------------------------------------------------
def provenance(fig, *, script: str, sources, theme: Theme = LIGHT,
               note: str = ""):
    """Stamp the producing script and every file read into the figure footer.

    `results/*.json` paths are printed relative to the repo root so the footer
    is a working pointer, not decoration. A figure lifted out of the repo into a
    slide still says which run produced it -- which is the whole point, since
    two bugs in this reproduction were found purely by asking where a number
    came from.
    """
    src = ", ".join(str(s) for s in sources)
    text = f"{script}  ←  {src}"
    if note:
        text += f"\n{note}"
    fig.text(0.02, -0.005, text, ha="left", va="top", fontsize=6.5,
             color=theme.muted, family="monospace")
    return fig


def figures_dir() -> Path:
    """`results/figures/`, created on demand."""
    d = C.RESULTS / "figures"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save(fig, name: str, *, formats=("png",), mode: str = "light",
         close: bool = True, outdir: Path | None = None) -> list[Path]:
    """Write `<name>.png` into `results/figures/`.

    **PNG only.** Every figure here has exactly one raster of record; a PDF
    beside it is a second copy of the same claim that has to be regenerated in
    lockstep, and the copy that gets forgotten is the one a reader opens. That
    is the same failure this repo keeps hitting in prose ("summaries drift
    while the text under them is maintained"), in binary form.

    Asking for `pdf` raises rather than quietly writing an extra file, so the
    rule cannot be reintroduced by a default argument somewhere downstream --
    the enforcement style of `audit_pcc_panels` and `has_dual_axis`.
    """
    if any(str(ext).lower().lstrip(".") == "pdf" for ext in formats):
        raise ValueError(
            f"{name}: figures are rendered as PNG only -- PDF output is not "
            "supported. Drop 'pdf' from "
            "formats=.")
    outdir = outdir or figures_dir()
    stem = name if mode == "light" else f"{name}-dark"
    written = []
    for ext in formats:
        p = Path(outdir) / f"{stem}.{ext}"
        fig.savefig(p, format=ext)
        written.append(p)
    if close:
        plt.close(fig)
    return written


# --------------------------------------------------------------------------
# The comparison primitives
# --------------------------------------------------------------------------
def paper_band(ax, mean: float, sd: float | None = None, *, theme: Theme = LIGHT,
               label: str = "paper", orient: str = "h", text_x: float = 0.99):
    """The paper's reported value, as a recessive band -- never a bar beside ours.

    A band spanning mean +/- sd reads as "the region the paper reports". A bar
    drawn next to ours reads as "these two numbers are the same measurement",
    which for most rows in this reproduction is false: different axis, different
    protocol, different n. Where the comparison IS like-for-like (scripts/08),
    the caption says so.
    """
    draw_span = ax.axhspan if orient == "h" else ax.axvspan
    draw_line = ax.axhline if orient == "h" else ax.axvline
    if sd:
        draw_span(mean - sd, mean + sd, color=theme.paper, alpha=0.13, zorder=0)
    draw_line(mean, color=theme.paper, linewidth=1.2, linestyle="-", zorder=1)
    txt = f"{label} {mean:.2f}" + (f" ± {sd:.2f}" if sd else "")
    if orient == "h":
        ax.text(text_x, mean, f" {txt}", transform=ax.get_yaxis_transform(),
                ha="right" if text_x > 0.5 else "left", va="bottom",
                fontsize=7.5, color=theme.ink_secondary)
    else:
        ax.text(mean, text_x, f" {txt}", transform=ax.get_xaxis_transform(),
                ha="left", va="top", fontsize=7.5, color=theme.ink_secondary)
    return ax


def reference_rule(ax, y: float, label: str, *, theme: Theme = LIGHT,
                   color: str | None = None, orient: str = "h",
                   at: float = 0.01):
    """A baseline or control drawn as a rule, so every mark is read against it.

    "Read every PCC against a baseline" is a repo convention; on a chart the
    cheapest way to enforce it is to make the baseline part of the chrome
    rather than one more bar the eye has to find.
    """
    color = color or theme.muted
    (ax.axhline if orient == "h" else ax.axvline)(
        y, color=color, linewidth=1.0, zorder=1)
    if not label:
        return ax
    # A surface-coloured backing, not a border: the label often crosses a mark,
    # and knocking the mark back is what keeps it readable without adding a box.
    back = {"facecolor": theme.surface, "edgecolor": "none", "pad": 1.0,
            "alpha": 0.85}
    if orient == "h":
        ax.text(at, y, f" {label} ", transform=ax.get_yaxis_transform(),
                ha="right" if at > 0.5 else "left", va="bottom",
                fontsize=7.5, color=color, bbox=back, zorder=6)
    else:
        ax.text(y, at, f" {label} ", transform=ax.get_xaxis_transform(),
                ha="left", va="top", fontsize=7.5, color=color, bbox=back,
                zorder=6)
    return ax


def not_applicable(ax, x, n_conditions: int, *, theme: Theme = LIGHT,
                   y: float = 0.0, vertical: bool = True):
    """Draw the `n/a (5 cond)` slot instead of a value.

    Mirrors `reporting.py:_pcc_str`. Per-molecule PCC over five conditions
    has a null distribution wide enough that ~a third of molecules exceed
    |r| = 0.7 by chance, so plotting it as a short bar would present an absent
    measurement as a weak one.
    """
    ax.text(x, y, f"n/a\n({n_conditions} cond)", ha="center",
            va="bottom" if vertical else "center",
            fontsize=7.5, color=theme.muted, linespacing=1.3)
    return ax


def annotate_n(ax, x, y, n: int, *, theme: Theme = LIGHT, dy: float = 0.0):
    """`n = 179` beside a mark. The sample size travels with the number."""
    # Clears a value label already sitting at the mark: those are drawn
    # va="bottom" at ~8 pt, so 9 pt of offset lands the two on top of each other.
    ax.annotate(f"n={n}", (x, y), textcoords="offset points",
                xytext=(0, 21 + dy), ha="center", fontsize=7,
                color=theme.muted)
    return ax


def status_marker(ax, x, y, role: str, *, theme: Theme = LIGHT,
                  label: str | None = None, size: float = 9):
    """A status mark: colour PLUS icon PLUS word. Never colour alone.

    Roles map to the four ways a discrepancy can resolve, because they
    demand different responses: reproduced / partial / undecidable /
    not reproduced.
    """
    if role not in STATUS:
        raise ValueError(f"status must be one of {sorted(STATUS)}, not {role!r}")
    ax.text(x, y, STATUS_ICON[role], color=STATUS[role], fontsize=size,
            ha="center", va="center", fontweight="bold")
    if label is not False:
        # Offset in POINTS, not data units: the same call has to work on an
        # axis scaled 0-1 and on one scaled 0-600.
        ax.annotate(label or STATUS_WORD[role], (x, y),
                    textcoords="offset points", xytext=(10, 0), fontsize=7.5,
                    ha="left", va="center", color=theme.ink_secondary)
    return ax


def bars(ax, xs, heights, *, color, theme: Theme = LIGHT, width: float = 0.62,
         labels: bool = True, fmt: str = "{:+.3f}", horizontal: bool = False):
    """Thin bars with a 2px surface gap and selective end labels.

    No border is drawn around a mark to separate it -- the gap does that. Values
    are labelled at the data end rather than on a grid, and only here, where the
    exact number is the point.
    """
    colors = color if isinstance(color, (list, tuple)) else [color] * len(xs)
    draw = ax.barh if horizontal else ax.bar
    art = draw(xs, heights, width, color=colors, linewidth=0.0, zorder=2)
    if labels:
        for rect, h in zip(art, heights):
            if h is None or h != h:
                continue
            if horizontal:
                ax.text(h, rect.get_y() + rect.get_height() / 2,
                        f" {fmt.format(h)}", va="center",
                        ha="left" if h >= 0 else "right",
                        fontsize=7.5, color=theme.ink_secondary)
            else:
                ax.text(rect.get_x() + rect.get_width() / 2, h,
                        f"{fmt.format(h)}", ha="center",
                        va="bottom" if h >= 0 else "top",
                        fontsize=7.5, color=theme.ink_secondary)
    return art

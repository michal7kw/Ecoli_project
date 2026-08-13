#!/usr/bin/env python
"""Render the figure atlas to results/figures/.

    python scripts/15_figures.py                     # everything available
    python scripts/15_figures.py --list              # names, tiers, availability
    python scripts/15_figures.py --figure fluxome    # one figure (repeatable)
    python scripts/15_figures.py --group layers      # one group
    python scripts/15_figures.py --format png,svg    # default: png (never pdf)
    python scripts/15_figures.py --dark              # dark-surface variants too

NOT part of the 00 -> 01 -> (02|03|04) chain. This script computes nothing: it
reads the results of record as they stand, so it can be re-run at any time
without touching a model. Every figure declares the files it reads
(`ecomics/figures/__init__.py:Figure.requires`), which is what lets `--list`
say WHY a figure is unavailable instead of failing halfway through drawing it.

Three availability tiers, matching how the repository tracks its artefacts:

    1  committed results/*.json   -- available in a fresh clone
    2  gitignored results/*.npz   -- rerun scripts/02 or 03
    3  built data/ecomics.db      -- rerun scripts/00 + 01

A figure whose inputs are missing is SKIPPED with the missing path named. It is
never drawn in a degraded form, for the same reason a database test skips
wholesale rather than passing vacuously: a figure that quietly drew nothing
would be read as a result.

Every figure carries its own provenance footer (producing script and every file
read) and, where it shows a PCC, an explicit axis label. Those are enforced by
`ecomics/plots.py` and its tests, not by convention -- the axis
confusion they guard against is what made the transcriptome read "ours 0.287 vs
the paper's 0.54" for most of this reproduction.

Each figure's own `caveat` records what it is and is not allowed to claim.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib                                        # noqa: E402
matplotlib.use("Agg")                                    # headless by default

from ecomics import config as C                          # noqa: E402
from ecomics import plots as P                           # noqa: E402
from ecomics import figures as F                         # noqa: E402

TIER_HINT = {
    1: "committed results/*.json",
    2: "gitignored results/*.npz -- rerun scripts/02 or scripts/03",
    3: "data/ecomics.db -- rerun scripts/00 then scripts/01",
}


def cmd_list() -> int:
    total = avail = 0
    for group, figs in F.by_group().items():
        print(f"\n{F.GROUPS[group]}")
        for f in figs:
            total += 1
            missing = f.missing()
            if missing:
                mark, why = "  skip", f"missing: {', '.join(missing)}"
            else:
                avail += 1
                mark, why = "    ok", f"tier {f.tier}"
            print(f"{mark}  {f.name:<26s} {why}")
            print(f"        {f.title}")
    print(f"\n{avail} of {total} figures available.")
    for tier, hint in TIER_HINT.items():
        print(f"  tier {tier}: {hint}")
    return 0


def render(fig_spec, formats, modes, outdir) -> list[Path]:
    written = []
    for mode in modes:
        theme = P.DARK if mode == "dark" else P.LIGHT
        with P.style(mode):
            figure = fig_spec.builder(theme)
            problems = P.audit_pcc_panels(figure)
            if problems:
                raise AssertionError(
                    f"{fig_spec.name}: PCC axis label bypassed plots.pcc_axis() "
                    f"-- {'; '.join(problems)}")
            if P.has_dual_axis(figure):
                raise AssertionError(
                    f"{fig_spec.name}: two axes share a position, which is the "
                    "signature of twinx(). Two y-scales on one plot invent a "
                    "correlation the data does not contain -- use two panels.")
            empty = P.audit_empty_series(figure)
            if empty:
                raise AssertionError(
                    f"{fig_spec.name}: a declared input is PRESENT but carries "
                    f"no data -- {'; '.join(empty)}. `requires` checks that a "
                    "file exists, not that the series inside it survived a "
                    "re-run; drawing the legend and the caption around nothing "
                    "is how a figure ships a claim it no longer has.")
            written += P.save(figure, fig_spec.name, formats=formats,
                              mode=mode, outdir=outdir)
    return written


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--list", action="store_true",
                    help="report names, tiers and availability; render nothing")
    ap.add_argument("--figure", action="append", default=[],
                    help="render one figure by name (repeatable)")
    ap.add_argument("--group", choices=sorted(F.GROUPS),
                    help="render one group")
    ap.add_argument("--format", default="png",
                    help="comma-separated: png, svg (default png). PDF output "
                         "is refused by ecomics.plots.save -- PNG is the only "
                         "figure of record")
    ap.add_argument("--dark", action="store_true",
                    help="also render dark-surface variants (<name>-dark.png)")
    ap.add_argument("--outdir", type=Path, default=None,
                    help="override results/figures/")
    args = ap.parse_args(argv)

    if args.list:
        return cmd_list()

    if args.figure:
        try:
            selected = [F.get(n) for n in args.figure]
        except KeyError as exc:
            print(exc, file=sys.stderr)
            return 2
    elif args.group:
        selected = F.by_group().get(args.group, [])
    else:
        selected = [f for figs in F.by_group().values() for f in figs]

    formats = tuple(x.strip() for x in args.format.split(",") if x.strip())
    modes = ["light"] + (["dark"] if args.dark else [])
    outdir = args.outdir or P.figures_dir()
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Figure atlas -> {outdir}")
    print(f"  formats {', '.join(formats)}   modes {', '.join(modes)}\n")

    made, skipped, failed = [], [], []
    for f in selected:
        missing = f.missing()
        if missing:
            skipped.append((f, missing))
            print(f"  SKIP  {f.name:<26s} needs {', '.join(missing)}")
            continue
        try:
            written = render(f, formats, modes, outdir)
        except Exception as exc:                          # noqa: BLE001
            failed.append((f, exc))
            print(f"  FAIL  {f.name:<26s} {type(exc).__name__}: {exc}")
            traceback.print_exc(limit=3)
            continue
        made.append(f)
        print(f"  ok    {f.name:<26s} {', '.join(p.name for p in written)}")

    print(f"\n{len(made)} rendered, {len(skipped)} skipped, {len(failed)} failed.")
    if skipped:
        print("\nSkipped, and what each needs:")
        for f, missing in skipped:
            print(f"  {f.name:<26s} tier {f.tier}: {TIER_HINT[f.tier]}")
    if failed:
        return 1
    if not made:
        print("\nNothing rendered. Run --list to see what each figure needs.")
        return 1
    print(f"\nEach figure carries its own caveat and provenance footer.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

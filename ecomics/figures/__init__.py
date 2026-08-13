"""The figure atlas: one builder per figure, and the registry that finds them.

Every builder is a function `(theme) -> matplotlib.figure.Figure` registered by
the `@figure` decorator with the files it reads. That declaration is what makes
`scripts/15_figures.py --list` able to say *why* a figure is unavailable rather
than failing halfway through drawing it, and it is what lets
The figure tests render the whole tier-1 set from committed results with no
data download.

Three availability tiers, because the repository's artefacts are tracked
differently on purpose:

    1  committed `results/*.json`      -- always available in a fresh clone
    2  gitignored `results/*.npz`      -- regenerable by scripts/02 and 03
    3  built `data/ecomics.db`         -- regenerable by scripts/00 + 01

A tier-2 or tier-3 figure SKIPS with the missing path named. It never renders a
degraded version, for the same reason a database test skips wholesale rather
than passing vacuously: a figure that quietly drew nothing would be read as a
result.

Groups follow the order results are reported in, so the atlas and
the results reference can be read side by side.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable

from .. import config as C

__all__ = ["Figure", "figure", "REGISTRY", "get", "by_group", "GROUPS",
           "results_json", "results_npz", "have"]

GROUPS = {
    "compendium": "A — The compendium",
    "transcriptome": "B — Transcriptome, and the two axes",
    "internals": "C — Model internals",
    "layers": "D — The other layers",
    "summary": "E — Summary",
}


@dataclass
class Figure:
    name: str
    title: str
    group: str
    builder: Callable
    requires: tuple[str, ...] = ()
    tier: int = 1
    caveat: str = ""
    _order: int = field(default=0, repr=False)

    def missing(self) -> list[str]:
        """Declared inputs that are not on disk, repo-relative."""
        return [r for r in self.requires if not (C.REPO / r).exists()]

    def available(self) -> bool:
        return not self.missing()


REGISTRY: dict[str, Figure] = {}
_counter = [0]


def figure(*, name: str, title: str, group: str, requires=(), tier: int = 1,
           caveat: str = ""):
    """Register a builder. `requires` paths are relative to the repo root."""
    if group not in GROUPS:
        raise ValueError(f"group must be one of {sorted(GROUPS)}, not {group!r}")

    def deco(fn):
        if name in REGISTRY:
            raise ValueError(f"duplicate figure name {name!r}")
        _counter[0] += 1
        REGISTRY[name] = Figure(name=name, title=title, group=group, builder=fn,
                                requires=tuple(requires), tier=tier,
                                caveat=caveat, _order=_counter[0])
        return fn
    return deco


def get(name: str) -> Figure:
    if name not in REGISTRY:
        raise KeyError(f"no figure {name!r}; known: {', '.join(sorted(REGISTRY))}")
    return REGISTRY[name]


def by_group() -> dict[str, list[Figure]]:
    """Registered figures, grouped and in registration order."""
    out: dict[str, list[Figure]] = {g: [] for g in GROUPS}
    for f in sorted(REGISTRY.values(), key=lambda f: f._order):
        out[f.group].append(f)
    return {g: v for g, v in out.items() if v}


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def results_json(stem: str) -> dict:
    """Read `results/<stem>.json`."""
    p = C.RESULTS / f"{stem}.json"
    if not p.exists():
        raise FileNotFoundError(f"{p} -- run the script that writes it")
    with p.open(encoding="utf-8") as fh:
        return json.load(fh)


def results_npz(stem: str):
    """Read `results/<stem>.npz` (gitignored; tier 2)."""
    import numpy as np
    p = C.RESULTS / f"{stem}.npz"
    if not p.exists():
        raise FileNotFoundError(
            f"{p} is gitignored and regenerable -- run the script that writes it")
    return np.load(p, allow_pickle=True)


def have(*paths: str) -> bool:
    return all((C.REPO / p).exists() for p in paths)


def _load_all():
    """Import every builder module, populating REGISTRY.

    The `fig_` prefix is not decoration. `tools/verify_docs.py:resolve_source`
    resolves a doc's `some_file.py:symbol` reference by BASENAME when the path
    is not repo-relative, and accepts it only when exactly one file matches --
    so an unprefixed `figures/transcriptome.py` would make every existing
    `transcriptome.py:...` reference in the docs ambiguous, and those
    references live in files this work never touched. The prefix keeps every
    basename in the repository unique.
    """
    from . import (fig_compendium, fig_internals, fig_layers,   # noqa: F401
                   fig_scorecard, fig_transcriptome)


_load_all()

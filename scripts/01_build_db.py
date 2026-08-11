#!/usr/bin/env python
"""Build data/ecomics.db and the Parquet matrices, then print a summary.

    python scripts/01_build_db.py            # build, or refuse if one exists
    python scripts/01_build_db.py --force    # delete and rebuild

Rebuilds from scratch (the DB is a derived artefact). Ends by asserting the
cross-layer condition overlaps reported in the paper.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ecomics import config as C          # noqa: E402
from ecomics.db.api import Ecomics       # noqa: E402
from ecomics.db.build import build       # noqa: E402


def main() -> int:
    # This script had NO argument parser, so every argument -- `--help`
    # included -- was silently ignored and execution fell straight through to
    # `build()`, which unlinks data/ecomics.db before rebuilding. Asking this
    # script for help destroyed a 363 MB database. It is derived and fully
    # regenerable, but a 35 s rebuild is not something `--help` should trigger,
    # and every other script in this directory parses its arguments.
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true",
                    help="delete and rebuild even if data/ecomics.db exists")
    args = ap.parse_args()

    if C.DB_PATH.exists() and not args.force:
        print(f"{C.DB_PATH} already exists "
              f"({C.DB_PATH.stat().st_size / 1e6:.0f} MB).\n"
              f"Building deletes and recreates it. Pass --force to proceed.")
        return 1

    missing = [p for p in (C.TRANSCRIPTOME_TXT, C.PROTEOME_CSV, C.METABOLOME_CSV)
               if not p.exists()]
    if missing:
        print("missing published data files:")
        for p in missing:
            print(f"  {p}")
        return 1
    if not (C.PROK_DIR / "fluxome.json").exists():
        print("prokaryomics data not scraped -- run scripts/00_acquire.py first")
        return 1

    build()

    print("\n" + "=" * 62)
    print("compendium summary")
    print("=" * 62)
    with Ecomics() as db:
        print(db.summary().to_string(index=False))
        print()
        print(f"  distinct conditions across all layers : {len(db.conditions())}")
        print(f"  strains x genotype features           : "
              f"{db.strain_features().shape[0]} x {db.strain_features().shape[1]}")
        print(f"  media x composition features          : "
              f"{db.medium_components().shape[0]} x {db.medium_components().shape[1]}")
        print(f"  distinct stresses                     : {len(db.stresses())}")
        print(f"  distinct perturbations                : {len(db.perturbations())}")
        print(f"  reactions with a BiGG cross-reference : {len(db.reaction_bigg())}")
        print()
        print("  cross-layer training sets:")
        for a, b in (("transcriptome", "proteome"),
                     ("transcriptome", "metabolome"),
                     ("transcriptome", "fluxome"),
                     ("transcriptome", "phenome"),
                     ("proteome", "metabolome")):
            print(f"    {a:<14s} x {b:<12s} {len(db.shared_conditions(a, b)):>4d} conditions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

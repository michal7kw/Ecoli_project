#!/usr/bin/env python
"""Acquire everything the replication needs beyond the four published files.

    python scripts/00_acquire.py            # fetch anything missing
    python scripts/00_acquire.py --force    # re-fetch everything
    python scripts/00_acquire.py --verify   # check artefacts, download nothing

What this collects and why:

  prokaryomics.com   the ONLY public source for the fluxome (43 x 120) and
                     phenome (253 conditions) layers, plus the meta-data
                     ontology (65 strains x 152 features, 112 media x 120)
  RegulonDB TRN      proteome module, TRN predictor
  STRING PPI         proteome module, PPI predictor
  KEGG pathways      proteome module, pathway predictor
  BiGG iJO1366       fluxome module, FBA
  GEO GSE73673       the paper's own 16-knockout RNA-Seq, as htseq counts
  GEO GSE12411       raw Affymetrix CEL files, for the from-scratch RMA
  CDF layout         probe-set -> array-cell map, exported once via R
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ecomics import config as C            # noqa: E402
from ecomics.acquire import fetch, scrape  # noqa: E402

CDF_PKG = "ecoliasv2cdf"                   # Affymetrix E. coli Antisense v2 (GPL199)
CDF_TSV = C.RAW_DIR / "cdf" / "ecoliasv2.tsv"
CDF_NCOL = 544


def export_cdf(force: bool = False, verbose: bool = True) -> Path | None:
    """Dump the Affymetrix CDF probe-set layout to TSV, once, via R.

    RMA's median-polish step needs to know which array cells belong to which
    probe set. That layout ships as R-serialized data inside a Bioconductor CDF
    package, so this one acquisition step shells out to Rscript. Everything
    downstream reads the resulting TSV and is pure Python.
    """
    if CDF_TSV.exists() and not force:
        if verbose:
            n = sum(1 for _ in CDF_TSV.open()) - 1
            print(f"  CDF layout already exported ({n:,} probe-cell rows)")
        return CDF_TSV

    rscript = shutil.which("Rscript")
    if rscript is None:
        if verbose:
            print("  Rscript not found -- skipping CDF export. RMA will run at "
                  "cell level only (no probe-set summarization).")
        return None

    script = C.REPO / "tools" / "export_cdf.R"
    cmd = [rscript, str(script), CDF_PKG, str(CDF_TSV), str(CDF_NCOL)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if proc.returncode != 0 or not CDF_TSV.exists():
        if verbose:
            tail = (proc.stderr or proc.stdout).strip().splitlines()[-3:]
            print(f"  CDF export failed; install with:\n"
                  f'    Rscript -e \'BiocManager::install("{CDF_PKG}")\'')
            for line in tail:
                print(f"    {line}")
        return None
    if verbose:
        for line in proc.stdout.strip().splitlines():
            if line.strip() and not line.startswith("Warning"):
                print(f"  {line.strip()}")
    return CDF_TSV


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="re-fetch even if present")
    ap.add_argument("--verify", action="store_true", help="check only, no downloads")
    ap.add_argument("--skip-raw", action="store_true",
                    help="skip the ~62 MB GEO raw-array download")
    args = ap.parse_args()

    C.ensure_dirs()

    if args.verify:
        print("verifying acquired artefacts\n")
        ok = fetch.verify_all()
        print()
        scrape.summarize()
        print(f"\n{'all artefacts present' if ok else 'SOME ARTEFACTS MISSING'}")
        return 0 if ok else 1

    print("=" * 70)
    print("1. prokaryomics.com")
    print("   (live at UC Davis; HTTPS cert expired, so fetched over HTTP)")
    print("=" * 70)
    scrape.scrape(force=args.force)

    print("\n" + "=" * 70)
    print("2. interaction networks and metabolic model")
    print("=" * 70)
    fetch.fetch_networks(force=args.force)

    print("\n" + "=" * 70)
    print(f"3. RNA-Seq counts: {C.GEO_RNASEQ_SERIES} (the paper's own 16 knockouts)")
    print("=" * 70)
    fetch.fetch_rnaseq_counts(force=args.force)

    if not args.skip_raw:
        print("\n" + "=" * 70)
        print(f"4. raw Affymetrix arrays: {C.GEO_ARRAY_SERIES} (GPL199, ~62 MB)")
        print("=" * 70)
        fetch.fetch_raw_arrays(force=args.force)

        print("\n" + "=" * 70)
        print("5. CDF probe-set layout (one-time export via R)")
        print("=" * 70)
        export_cdf(force=args.force)

    print("\n" + "=" * 70)
    print("6. the paper's supplementary material (Europe PMC, CC-BY, ~31 MB)")
    print("   Supplementary Methods 3.3.3 states the evaluation protocol;")
    print("   Supplementary Data 1 carries the growth phase the released")
    print("   expression table omits. See DISCREPANCIES.md 3.")
    print("=" * 70)
    fetch.fetch_supplementary(force=args.force)

    print("\n" + "=" * 70)
    print("acquisition complete")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

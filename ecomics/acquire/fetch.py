"""Download the external resources MOMA needs but Ecomics does not ship.

  KEGG       gene list: symbol -> b-number       -> db/build.py, TF subsets
  BiGG       iJO1366 genome-scale model          -> fluxome FBA module
  GEO        raw .CEL arrays + GSE73673 counts   -> preprocessing pipeline demo

This used to fetch three interaction networks too -- RegulonDB via SBRG,
STRING v12, the KEGG pathway REST endpoint -- for the proteome module's TRN,
PPI and KEGG predictors. Those downloads are deliberately absent, because the
paper's own edge lists in Supplementary Data 2 became the layer's only source
(worth +0.113 per-profile PCC); a fourth, the KEGG pathway LIST, turned out to
have been fetched and read by nothing at all. `kegg_gene_list` stays and is
not a network -- `db/build.py` asserts against it while normalizing
perturbations to b-numbers.

Every fetch is idempotent and checksummed (see acquire/cache.py). Sources that
have moved hosts over the years declare several candidate URLs; the first that
returns usable content wins, and the one actually used is recorded in the
sidecar metadata.
"""

from __future__ import annotations

import gzip
import io
import re
from pathlib import Path

from ecomics import config as C
from ecomics.acquire.cache import DownloadError, download, fetch_bytes, verify

# --------------------------------------------------------------------------
# Networks and models
# --------------------------------------------------------------------------
def fetch_networks(force: bool = False, verbose: bool = True) -> dict[str, Path]:
    """Download the KEGG gene list and the BiGG model. Returns {key: path}.

    Failures are collected rather than raised. That tolerance was written for
    the interaction networks, where a missing PPI file degraded the proteome
    ensemble from four predictors to three -- a real result worth reporting,
    not a reason to abort. Those are gone; what remains is less forgiving, and
    the guard has moved downstream: a missing `kegg_gene_list` makes
    `gene_symbol_map` return `{}`, and `db/build.py` ASSERTS on that rather
    than building a database with three collapsed cross-layer joins.
    """
    C.ensure_dirs()
    got: dict[str, Path] = {}
    failed: dict[str, str] = {}

    for key, urls in C.REMOTE.items():
        dest = C.REMOTE_FILES[key]
        try:
            download(urls, dest, force=force, min_bytes=1000)
            got[key] = dest
            if verbose:
                print(f"  {key:<22s} {dest.stat().st_size:>12,d} B  {dest.name}")
        except DownloadError as exc:
            failed[key] = str(exc).splitlines()[0]
            if verbose:
                print(f"  {key:<22s} FAILED  {failed[key][:80]}")

    if failed and verbose:
        print(f"\n  {len(failed)} resource(s) unavailable; dependent modules will "
              "report reduced coverage rather than fail.")
    return got


# --------------------------------------------------------------------------
# GEO: raw samples for the preprocessing pipeline
# --------------------------------------------------------------------------
_HREF_RE = re.compile(r'href="([^"?][^"]*)"')


def _list_ftp_dir(url: str) -> list[str]:
    """List filenames in a GEO FTP-over-HTTPS directory."""
    try:
        html = fetch_bytes(url).decode("utf-8", errors="replace")
    except DownloadError:
        return []
    names = [h for h in _HREF_RE.findall(html) if not h.startswith(("/", "http"))]
    return [n for n in names if n not in ("../",)]


def fetch_geo_supplementary(
    gse: str,
    dest_dir: Path,
    *,
    pattern: str = r".*",
    limit: int | None = None,
    force: bool = False,
    verbose: bool = True,
) -> list[Path]:
    """Download files from a GEO series' suppl/ directory matching `pattern`."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    base = C.geo_series_dir(gse, "suppl")
    names = [n for n in _list_ftp_dir(base) if re.search(pattern, n, re.I)]
    if not names:
        if verbose:
            print(f"  {gse}: no suppl files matching /{pattern}/ at {base}")
        return []
    if limit:
        names = names[:limit]

    out = []
    for n in names:
        dest = dest_dir / n
        try:
            download(base + n, dest, force=force, min_bytes=100)
            out.append(dest)
            if verbose:
                print(f"    {n}  ({dest.stat().st_size:,} B)")
        except DownloadError as exc:
            if verbose:
                print(f"    {n}  FAILED: {str(exc).splitlines()[0][:70]}")
    return out


def fetch_raw_arrays(force: bool = False, verbose: bool = True) -> list[Path]:
    """Fetch a handful of raw Affymetrix .CEL files for the RMA demo.

    GEO ships per-series supplementary data as one RAW.tar containing the CELs.
    We download the tar and extract only the first few members, because the
    point is to exercise the CEL reader and RMA on genuine data, not to
    reprocess a whole series.
    """
    import tarfile

    gse = C.GEO_ARRAY_SERIES
    dest_dir = C.RAW_DIR / gse
    dest_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted(dest_dir.glob("*.CEL")) + sorted(dest_dir.glob("*.cel"))
    if existing and not force:
        if verbose:
            print(f"  {gse}: {len(existing)} CEL file(s) already present")
        return existing

    tars = fetch_geo_supplementary(gse, dest_dir, pattern=r"RAW\.tar$",
                                   limit=1, force=force, verbose=verbose)
    if not tars:
        return []

    out: list[Path] = []
    with tarfile.open(tars[0]) as tf:
        members = [m for m in tf.getmembers()
                   if re.search(r"\.cel(\.gz)?$", m.name, re.I)]
        for m in members[: C.GEO_ARRAY_MAX_SAMPLES]:
            data = tf.extractfile(m)
            if data is None:
                continue
            payload = data.read()
            name = Path(m.name).name
            if name.lower().endswith(".gz"):
                payload = gzip.decompress(payload)
                name = name[:-3]
            p = dest_dir / name
            p.write_bytes(payload)
            out.append(p)
            if verbose:
                print(f"    extracted {name}  ({len(payload):,} B)")
    return out


def _extract_tar(tar_path: Path, dest_dir: Path, pattern: str,
                 limit: int | None = None, verbose: bool = True) -> list[Path]:
    """Extract members matching `pattern`, gunzipping .gz members in place."""
    import tarfile

    out: list[Path] = []
    with tarfile.open(tar_path) as tf:
        members = [m for m in tf.getmembers() if re.search(pattern, m.name, re.I)]
        for m in members[:limit] if limit else members:
            fh = tf.extractfile(m)
            if fh is None:
                continue
            payload = fh.read()
            name = Path(m.name).name
            if name.lower().endswith(".gz"):
                payload = gzip.decompress(payload)
                name = name[:-3]
            p = dest_dir / name
            p.write_bytes(payload)
            out.append(p)
    if verbose:
        print(f"    extracted {len(out)} file(s) matching /{pattern}/ "
              f"from {tar_path.name}")
    return out


def fetch_rnaseq_counts(force: bool = False, verbose: bool = True) -> list[Path]:
    """Fetch GSE73673 -- this paper's own 16-knockout RNA-Seq experiment.

    GEO exposes only GSE73673_RAW.tar in suppl/; the per-sample files live
    inside it and are `.htcount.txt.gz` -- i.e. **htseq-count output**, exactly
    the product of the paper's Trimmomatic -> TopHat/bowtie -> htseq-count
    pipeline (paper.md:134). So this is the genuine article at the stage where
    the Ecomics normalization pipeline picks up.

    We stop at counts rather than realigning from FASTQ because that needs
    bowtie2/TopHat/samtools plus a K-12 index, none of which are installed
    here. pipeline/rnaseq.py wraps the alignment path and runs it when the
    binaries are present.
    """
    gse = C.GEO_RNASEQ_SERIES
    dest_dir = C.RAW_DIR / gse
    dest_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted(dest_dir.glob("*.htcount.txt"))
    if existing and not force:
        if verbose:
            print(f"  {gse}: {len(existing)} count file(s) already present")
        return existing

    # filelist.txt is the manifest; keep it for provenance.
    fetch_geo_supplementary(gse, dest_dir, pattern=r"^filelist\.txt$",
                            force=force, verbose=False)
    tars = fetch_geo_supplementary(gse, dest_dir, pattern=r"RAW\.tar$",
                                   limit=1, force=force, verbose=verbose)
    if not tars:
        return []
    return _extract_tar(tars[0], dest_dir, r"\.htcount\.txt(\.gz)?$", verbose=verbose)


# --------------------------------------------------------------------------
def fetch_all(force: bool = False, verbose: bool = True) -> dict:
    """Everything except the prokaryomics scrape (see acquire/scrape.py)."""
    if verbose:
        print("networks and models:")
    nets = fetch_networks(force=force, verbose=verbose)

    if verbose:
        print(f"\nraw microarrays ({C.GEO_ARRAY_SERIES}):")
    cels = fetch_raw_arrays(force=force, verbose=verbose)

    if verbose:
        print(f"\nRNA-Seq count tables ({C.GEO_RNASEQ_SERIES}):")
    counts = fetch_rnaseq_counts(force=force, verbose=verbose)

    return {"networks": nets, "cel_files": cels, "rnaseq": counts}


def verify_all(verbose: bool = True) -> bool:
    """Check every declared artefact against its sidecar metadata."""
    ok = True
    targets: list[tuple[str, Path]] = [
        (name, C.PROK_DIR / f"{name}.json")
        for name in list(C.PROK_JSON.values()) + list(C.PROK_EMBEDDED.values())
    ]
    targets += [(k, p) for k, p in C.REMOTE_FILES.items()]

    for name, path in targets:
        if path.suffix == ".json" and path.parent == C.PROK_DIR:
            # normalized scrape output: no sidecar, check it parses instead
            status = "ok" if path.exists() else "MISSING"
            good = path.exists()
        else:
            good, status = verify(path)
        # This was `ok &= good or "MISSING" not in status`, which could never be
        # False for a downloaded artefact: `verify()` returns LOWERCASE statuses
        # ("missing", "sha256 mismatch", "size N != recorded M"), so
        # `"MISSING" not in status` was always true and short-circuited every
        # failure away. verify_all() reported success regardless of what it
        # printed, and `--verify` exited 0 on a corrupt cache.
        ok &= good
        if verbose:
            flag = "ok " if good else "!! "
            print(f"  {flag}{name:<22s} {status}")
    return ok


if __name__ == "__main__":
    fetch_all()


def fetch_supplementary(force: bool = False, verbose: bool = True) -> list[Path]:
    """The paper's own supplementary material, and the files inside it.

    Fetched from Europe PMC rather than PMC's own /bin/ paths: PMC answers a
    non-browser client with a "Preparing to download..." interstitial instead of
    the file, while Europe PMC serves every supplementary file as one ZIP with no
    such gate. The article is CC-BY, so redistribution is permitted; the archive
    lands under untracked `data/` regardless, per the repo convention that data
    is regenerable and never committed.

    Worth having because it settles things the article body does not:
    Supplementary Methods 3.3.3 states the evaluation axis and data scale, and
    Supplementary Data 1 carries the per-profile growth phase that defines the
    paper's 2,610-profile exponential subset -- which is absent from the released
    expression table, and therefore not reconstructable without this.
    """
    import zipfile

    dest = C.REMOTE_FILES["supplementary_zip"]
    dest.parent.mkdir(parents=True, exist_ok=True)
    download(C.REMOTE["supplementary_zip"], dest, force=force,
             min_bytes=1_000_000)

    written = []
    with zipfile.ZipFile(dest) as z:
        for info in z.infolist():
            name = Path(info.filename).name
            if not name.startswith("ncomms13090-s"):
                continue                       # figures; the article carries them
            out = dest.parent / name
            if force or not out.exists() or out.stat().st_size != info.file_size:
                out.write_bytes(z.read(info.filename))
            written.append(out)
    if verbose:
        print(f"  {dest.name}: {dest.stat().st_size / 1e6:.1f} MB")
        for p in sorted(written):
            print(f"    {p.name:<24s} {p.stat().st_size:>9,d} bytes")
        missing = [k for k, v in C.SUPPLEMENTARY.items() if not v.exists()]
        print(f"  expected files present: "
              f"{len(C.SUPPLEMENTARY) - len(missing)}/{len(C.SUPPLEMENTARY)}"
              + (f"  MISSING: {missing}" if missing else ""))
    return written

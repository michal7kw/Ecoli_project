"""Idempotent, checksummed downloads.

Deliberately depends only on the standard library: `requests` is not installed
in this environment and adding a hard dependency for four GET requests is not
worth it. `urllib` handles gzip, redirects and timeouts perfectly well.

Every download writes a sidecar `<file>.meta.json` recording the URL, size,
SHA-256 and timestamp, so `--verify` can check artefacts without re-fetching
and provenance is auditable.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import shutil
import ssl
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Sequence

USER_AGENT = "ecomics-replication/0.1 (+https://doi.org/10.1038/ncomms13090)"
TIMEOUT = 120


class DownloadError(RuntimeError):
    """Raised when every candidate URL for a resource fails."""


def _opener(insecure: bool = False):
    """Build a urllib opener.

    insecure=True disables certificate verification. This is needed for exactly
    one host: prokaryomics.com, whose TLS certificate has expired. We only ever
    talk to it over plain http anyway, so this is belt-and-braces.
    """
    handlers: list[Any] = []
    if insecure:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        handlers.append(urllib.request.HTTPSHandler(context=ctx))
    return urllib.request.build_opener(*handlers)


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def _meta_path(dest: Path) -> Path:
    return dest.with_suffix(dest.suffix + ".meta.json")


def write_meta(dest: Path, url: str) -> dict:
    meta = {
        "url": url,
        "bytes": dest.stat().st_size,
        "sha256": sha256(dest),
        "fetched_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    _meta_path(dest).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def read_meta(dest: Path) -> dict | None:
    p = _meta_path(dest)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def fetch_bytes(url: str, insecure: bool = False, retries: int = 3) -> bytes:
    """GET a URL, transparently decompressing gzip Content-Encoding."""
    last: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with _opener(insecure).open(req, timeout=TIMEOUT) as resp:
                raw = resp.read()
                if resp.headers.get("Content-Encoding") == "gzip":
                    raw = gzip.decompress(raw)
                return raw
        except (urllib.error.URLError, TimeoutError, ssl.SSLError, OSError) as exc:
            last = exc
            if attempt < retries - 1:
                time.sleep(1.5 * (attempt + 1))
    raise DownloadError(f"{url}: {last}")


def download(
    urls: str | Sequence[str],
    dest: Path,
    *,
    insecure: bool = False,
    force: bool = False,
    min_bytes: int = 1,
) -> Path:
    """Download the first URL that works, into `dest`.

    Skips the download if `dest` already exists with a valid sidecar, unless
    `force`. Writes atomically via a .part file so an interrupted run never
    leaves a truncated artefact that a later run would treat as complete.
    """
    if isinstance(urls, str):
        urls = [urls]
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and not force:
        meta = read_meta(dest)
        if meta and dest.stat().st_size >= min_bytes:
            return dest

    errors = []
    for url in urls:
        try:
            payload = fetch_bytes(url, insecure=insecure)
        except DownloadError as exc:
            errors.append(str(exc))
            continue
        if len(payload) < min_bytes:
            errors.append(f"{url}: only {len(payload)} bytes (< {min_bytes})")
            continue
        tmp = dest.with_suffix(dest.suffix + ".part")
        tmp.write_bytes(payload)
        tmp.replace(dest)
        write_meta(dest, url)
        return dest

    raise DownloadError(f"all sources failed for {dest.name}:\n  " + "\n  ".join(errors))


def verify(dest: Path) -> tuple[bool, str]:
    """Check an artefact against its recorded sidecar metadata."""
    if not dest.exists():
        return False, "missing"
    meta = read_meta(dest)
    if meta is None:
        return False, "no sidecar metadata"
    if dest.stat().st_size != meta.get("bytes"):
        return False, f"size {dest.stat().st_size} != recorded {meta.get('bytes')}"
    if sha256(dest) != meta.get("sha256"):
        return False, "sha256 mismatch"
    return True, f"ok ({meta['bytes']:,} bytes)"


def save_json(obj: Any, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(obj, indent=1), encoding="utf-8")
    return dest


def load_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8", errors="replace"))


def gunzip(src: Path, dest: Path | None = None) -> Path:
    """Decompress a .gz file next to itself."""
    dest = dest or src.with_suffix("")
    with gzip.open(src, "rb") as fi, open(dest, "wb") as fo:
        shutil.copyfileobj(fi, fo)
    return dest


"""Scrape prokaryomics.com, the Ecomics web resource.

The site (Tagkopoulos lab, UC Davis) is LIVE at 128.120.136.57. Its TLS
certificate has expired, so https:// fails; plain http:// returns 200. This is
the only public source for three things the published Dropbox files omit:

  1. the fluxome layer      43 profiles x 120 fluxes (R0001..R0120)
  2. the phenome layer      253 conditions x lag time / growth rate / final OD
  3. the meta-data ontology 65 strains x 152 genotype features
                            112 media  x 120 composition features

Two different delivery mechanisms are in play:

  * Some pages back a bootstrap-table with `data-url="strain.v5.json"`, so the
    JSON can be fetched directly from the site root.
  * The fluxome and phenome pages have NO such endpoint. Their entire dataset
    is inlined into the page source as a JavaScript literal:
        var x = [{"_id":"...","Strain":"BW25113",...}, ...];
    so it has to be regexed out of the HTML. That is what `extract_embedded`
    does, and it is the only way to get those two layers.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from ecomics import config as C
from ecomics.acquire.cache import download, load_json, save_json

# Matches `var x = [ ... ];` non-greedily up to the first `];` that is followed
# by whitespace/newline. The payload is strict JSON, so json.loads validates it.
_EMBEDDED_RE = re.compile(r"var\s+x\s*=\s*(\[.*?\])\s*;", re.S)


def _strip_ids(records: list[dict]) -> list[dict]:
    """Drop MongoDB `_id` fields — they are storage artefacts, not data."""
    return [{k: v for k, v in r.items() if k != "_id"} for r in records]


def merge_duplicate_columns(pairs: list[tuple[str, object]]) -> dict:
    """`object_pairs_hook` that collapses a repeated column instead of dropping it.

    JSON permits a repeated key and `json.loads` keeps the LAST one, silently.
    That is a data loss, not a formatting detail: `strain.v5.json` serves 159
    columns per record, three of them twice — b4348 (hsdS), b0243 (proA) and
    b4346 (mcrB) — so a plain parse yields 156 keys and six cells that read
    "marker absent" when the source says otherwise.

    The duplication is upstream, not a scrape artefact: Supplementary Data 1's
    own Strain sheet has the same three names twice (155 non-meta column
    positions, 152 distinct). It records those loci in two places, once as a
    point allele and once as part of a multi-gene deletion —

        JA122   b4348 = ['hss1',   'no']        MC1000  b4346 = ['mcrB1', 'no']
        D31     b0243 = ['proA23', 'no']        TOP10   b4348 = ['no', 'hsdS-']
        AB1157  b0243 = ['proA-',  'proA-']     TOP10   b4346 = ['no', 'mcrB-']

    — and that sheet's `Comments` column names the deletions the right-hand
    spellings come from: TOP10 `delta(mrr-hsdRMS-mcrBC)` spans hsdS and mcrB,
    AB1157 `delta(gpt-proA)62` spans proA. Both spellings assert the same
    thing, that the locus is disrupted, which is what makes merging correct
    rather than convenient: they are two routes to one fact, not two facts.

    So take the single informative value, and RAISE if the repeats disagree on
    what it is. Choosing between two genuine alleles is exactly the judgement
    this must not make silently — the same failure, one layer up.
    """
    out: dict = {}
    repeated: dict[str, list] = {}
    for key, value in pairs:
        if key in out:
            repeated.setdefault(key, [out[key]]).append(value)
        else:
            out[key] = value

    for key, values in repeated.items():
        informative = [v for v in values
                       if str(v).strip().lower() not in C.ABSENT_TOKENS]
        distinct = {str(v).strip() for v in informative}
        if len(distinct) > 1:
            raise ValueError(
                f"column {key!r} appears {len(values)} times with conflicting "
                f"values {sorted(distinct)}. The merge rule assumes repeated "
                "columns record one locus two ways; it cannot choose between "
                "two different alleles. Inspect the source before proceeding."
            )
        # All-absent repeats collapse to the value already there.
        out[key] = informative[0] if informative else values[0]
    return out


def extract_embedded(html: str, page: str) -> list[dict]:
    """Pull the `var x = [...]` dataset out of a prokaryomics page."""
    m = _EMBEDDED_RE.search(html)
    if not m:
        raise ValueError(
            f"no `var x = [...]` payload found in /{page}. The site layout may "
            "have changed; inspect the HTML by hand before trusting any fallback."
        )
    try:
        records = json.loads(m.group(1), object_pairs_hook=merge_duplicate_columns)
    except json.JSONDecodeError as exc:
        raise ValueError(f"/{page}: embedded payload is not valid JSON: {exc}") from exc
    if not isinstance(records, list) or not records:
        raise ValueError(f"/{page}: embedded payload is empty or not a list")
    return _strip_ids(records)


def _normalize_table(payload) -> list[dict]:
    """Coerce a bootstrap-table endpoint response into a list of dicts."""
    if isinstance(payload, dict):
        for key in ("rows", "data", "records"):
            if key in payload:
                payload = payload[key]
                break
        else:
            payload = next(iter(payload.values()))
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"expected a non-empty list, got {type(payload).__name__}")
    if not isinstance(payload[0], dict):
        # The legacy /strain.json and /medium.json endpoints return positional
        # arrays with no column names. Unusable without a header, and the .v5
        # endpoints supersede them -- see the note in config.PROK_JSON.
        raise ValueError(
            f"records are {type(payload[0]).__name__}, not dicts: this looks like "
            "the legacy headerless schema, which is not supported"
        )
    return _strip_ids(payload)


def scrape(force: bool = False, verbose: bool = True) -> dict[str, Path]:
    """Fetch every prokaryomics resource into data/external/prokaryomics/.

    Returns {logical name -> written path}. Raises if a resource comes back
    with an unexpected record count, so a silent upstream change cannot
    corrupt the database downstream.
    """
    C.ensure_dirs()
    out: dict[str, Path] = {}
    # Raw responses are kept separately from the normalized output, so a raw
    # filename can never collide with a normalized one (e.g. the endpoint
    # `strain.v5.json` normalizes to `strain.json`, which is also a live URL).
    raw_dir = C.PROK_DIR / "_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    def log(msg: str) -> None:
        if verbose:
            print(msg)

    # --- direct JSON endpoints -------------------------------------------
    for filename, name in C.PROK_JSON.items():
        dest_raw = raw_dir / filename
        download(f"{C.PROK_BASE}/{filename}", dest_raw, insecure=True,
                 force=force, min_bytes=200)
        records = _normalize_table(
            load_json(dest_raw, object_pairs_hook=merge_duplicate_columns))
        dest = C.PROK_DIR / f"{name}.json"
        save_json(records, dest)
        out[name] = dest
        log(f"  {name:<12s} {len(records):>5d} records  -> {dest.name}")

    # --- pages with the dataset inlined in the HTML ----------------------
    for page, name in C.PROK_EMBEDDED.items():
        dest_html = raw_dir / f"{page}.html"
        download(f"{C.PROK_BASE}/{page}", dest_html, insecure=True,
                 force=force, min_bytes=2000)
        html = dest_html.read_text(encoding="utf-8", errors="replace")
        records = extract_embedded(html, page)
        dest = C.PROK_DIR / f"{name}.json"
        save_json(records, dest)
        out[name] = dest
        n_fields = len(records[0])
        log(f"  {name:<12s} {len(records):>5d} records x {n_fields} fields  "
            f"-> {dest.name}  (extracted from inline HTML)")

    _assert_expected(out)
    return out


def _assert_expected(paths: dict[str, Path]) -> None:
    """Fail loudly if the site returned an unexpected number of records."""
    problems = []
    for name, expected in C.PROK_EXPECTED.items():
        if name not in paths:
            problems.append(f"{name}: not fetched")
            continue
        got = len(load_json(paths[name]))
        if got != expected:
            problems.append(f"{name}: got {got} records, expected {expected}")

    for name, expected in C.PROK_EXPECTED_COLUMNS.items():
        if name not in paths:
            continue
        records = load_json(paths[name])
        got = len(records[0]) if records else 0
        if got != expected:
            problems.append(f"{name}: got {got} columns, expected {expected}")
    if problems:
        raise AssertionError(
            "prokaryomics returned unexpected data:\n  " + "\n  ".join(problems)
            + "\n(the site may have been updated; verify before proceeding)"
        )


def summarize(verbose: bool = True) -> dict[str, dict]:
    """Describe what was scraped, without re-fetching."""
    info: dict[str, dict] = {}
    for name in list(C.PROK_JSON.values()) + list(C.PROK_EMBEDDED.values()):
        p = C.PROK_DIR / f"{name}.json"
        if not p.exists():
            continue
        rec = load_json(p)
        fields = list(rec[0].keys()) if rec else []
        info[name] = {"records": len(rec), "fields": len(fields), "path": str(p)}
        if verbose:
            print(f"  {name:<12s} {len(rec):>5d} x {len(fields):>3d}  {fields[:6]}")
    return info


if __name__ == "__main__":
    scrape()

"""Acquisition of everything not shipped in data/.

  scrape.py  prokaryomics.com  -> strain/medium/molecule/reaction JSON,
                                  plus the fluxome and phenome datasets that
                                  are embedded inline in the page HTML
  fetch.py   RegulonDB / STRING / KEGG / BiGG / GEO
  cache.py   idempotent, checksummed downloads
"""

from ecomics.acquire.cache import download, load_json, save_json  # noqa: F401

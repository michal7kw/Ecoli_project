"""Paths and remote resource definitions.

Everything the project reads or writes is declared here, so a reader can see the
full data footprint in one place and no module hard-codes a path.

Layout under REPO:
    data/                        the four published Ecomics files (NEVER modified)
    data/external/prokaryomics/  scraped from prokaryomics.com
    data/external/networks/      the KEGG gene list (symbol -> b-number) only;
                                 the interaction graphs come from Data 2
    data/external/models/        BiGG iJO1366 genome-scale metabolic model
    data/external/raw/           raw microarray / RNA-Seq samples for the pipeline demo
    data/ecomics.db              the built SQLite compendium
    data/parquet/                wide numeric matrices, for fast model loading
    curation/                    hand-curated inputs -- TRACKED, since nothing
                                 regenerates them (see curation/README.md)
    results/                     metrics, reproduction table, figures
"""

from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
REPO = Path(os.environ.get("ECOMICS_REPO", Path(__file__).resolve().parent.parent))

DATA = REPO / "data"
EXTERNAL = DATA / "external"
PROK_DIR = EXTERNAL / "prokaryomics"
NET_DIR = EXTERNAL / "networks"
MODEL_DIR = EXTERNAL / "models"
RAW_DIR = EXTERNAL / "raw"
SUPP_DIR = EXTERNAL / "supplementary"

DB_PATH = DATA / "ecomics.db"
PARQUET_DIR = DATA / "parquet"

RESULTS = REPO / "results"
DOCS = REPO / "docs"
PAPER_MD = REPO / "paper" / "paper.md"

# The four published files, as downloaded by the user from the Ecomics Dropbox.
# Treated as read-only inputs.
TRANSCRIPTOME_TXT = DATA / "Ecomics.transcriptome.no_avg.v8.txt"
PROTEOME_CSV = DATA / "Ecomics.proteome.v5.csv"
METABOLOME_CSV = DATA / "Ecomics.metabolome.v3.csv"
DATA_README = DATA / "README.txt"

# --------------------------------------------------------------------------
# curation/ -- hand-curated inputs, tracked rather than under data/
# --------------------------------------------------------------------------
# Everything under data/ is untracked and regenerable from 00_acquire +
# 01_build_db. These are neither: they arrived from outside the project and no
# script reproduces them, so an untracked copy would be unrecoverable and any
# number read from one would have no provenance. They live here instead.
#
# The workbook is the source of record; the TSV is what documentation and code
# should read, because a binary cannot be diffed or quoted. `tools/extract_
# catalog.py` regenerates the second from the first and `--check` asserts they
# agree.
CURATION = REPO / "curation"
PERTURBATION_CATALOG = CURATION / "Ecoli_K12_perturbation_omics_catalog.xlsx"
PERTURBATION_CATALOG_TSV = CURATION / "catalog_entries.tsv"

ALL_DIRS = [EXTERNAL, PROK_DIR, NET_DIR, MODEL_DIR, RAW_DIR, SUPP_DIR,
            PARQUET_DIR, RESULTS]


def ensure_dirs() -> None:
    """Create every output directory. Safe to call repeatedly."""
    for d in ALL_DIRS:
        d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# Source-data conventions
# --------------------------------------------------------------------------
# Cell values in the scraped ontology tables meaning "this marker/component is
# not present". Shared by two pipeline stages that must agree on it: the scraper
# (acquire/scrape.py, merging repeated columns) and the loader (db/build.py,
# binarizing a cell to `present`). It lives here rather than in either one
# because a stage-01 constant imported by stage 00 would invert the pipeline's
# dependency direction, and a second copy would drift.
ABSENT_TOKENS = {"no", "none", "na", "", "-", "?"}



# --------------------------------------------------------------------------
# prokaryomics.com
# --------------------------------------------------------------------------
# The site is live at UC Davis (128.120.136.57) but its TLS certificate has
# EXPIRED, so https:// fails with SEC_E_CERT_EXPIRED. Plain http:// returns 200.
# This is why every URL below is http and not https.
PROK_BASE = "http://prokaryomics.com"

# Endpoints serving JSON directly (referenced as bootstrap-table data-url attrs).
PROK_JSON = {
    "strain.v5.json": "strain",          # 65 strains x 152 genotype features
    "medium.v5.json": "medium",          # 112 media x 120 composition features
    "molecules.filtered.json": "molecule",
    "reactions.json": "reaction",        # 115 reactions; only 26 carry a BiGG
                                         # cross-reference, of which 22 are in iJO1366
}
# Note: /strain.json and /medium.json also resolve (200) but serve an OLDER
# schema shaped as {"data": [[...row...], ...]} -- headerless positional arrays
# with no column names, and only 74 of the 112 media. They are strictly less
# useful than the .v5 endpoints and are deliberately not fetched.

# Pages that embed their whole dataset inline as `var x = [ ... ];`.
# This is the only public source for the fluxome and phenome layers.
PROK_EMBEDDED = {
    "fluxome": "fluxome",   # 43 profiles x 120 fluxes (R0001..R0120)
    "phenome": "phenome",   # 253 conditions x LT / GR / FOD
}

# Assertions that must hold after a successful scrape. If the site changes
# shape, acquisition fails loudly here rather than silently downstream.
PROK_EXPECTED = {
    "strain": 65,
    "medium": 112,
    "reaction": 115,
    "fluxome": 43,
    "phenome": 253,
}

# Columns per record, AFTER acquire/scrape.py:merge_duplicate_columns. Record
# counts alone were the only shape assertion here for a long time, and they are
# the axis nothing went wrong on: the strain endpoint lost three columns to a
# repeated-key collapse while still returning exactly 65 records. A width is as
# much a part of the contract as a length.
PROK_EXPECTED_COLUMNS = {
    "strain": 156,      # 4 meta + 152 genotype features (159 raw positions,
                        # three of them repeats -- see merge_duplicate_columns)
    "medium": 126,      # 6 meta + 120 composition features
    "molecule": 3,
    "reaction": 3,
    "fluxome": 125,     # 5 meta + 120 reactions R0001..R0120
    "phenome": 8,
}

# --------------------------------------------------------------------------
# External network / model resources
# --------------------------------------------------------------------------
# ⚠ THERE ARE NO INTERACTION-NETWORK DOWNLOADS HERE, DELIBERATELY.
#
# The proteome module's TRN, PPI and KEGG graphs could be scraped from RegulonDB,
# STRING and the KEGG pathway API. They are not. The paper's OWN edge lists ship
# in Supplementary Data 2, already on disk, and using them instead is worth
# +0.113 per-profile PCC -- so a scraped graph would be a second, worse way to
# build the same thing. `ecomics/networks_paper.py` is the only graph loader.
#
# `kegg_gene_list` STAYS, and is not a network: `networks.gene_symbol_map()`
# reads it to map gene SYMBOLS onto b-numbers, which `db/build.py` asserts
# against when normalizing perturbations, `paper_protocol.py` needs for the
# paper's TF subset, and `networks_paper.py` uses to repair Data 2's handful of
# non-b-number tokens. Removing it breaks the build, not the proteome.
REMOTE = {
    # KEGG gene list -- gene NAMES (rpoD, crp, arcA) -> b-numbers, so every
    # source in the repo shares one identifier space.
    "kegg_gene_list": ["https://rest.kegg.jp/list/eco"],
    # --- The paper's own supplementary material ---
    # CC-BY, open access. Europe PMC serves every supplementary file as one ZIP
    # and, unlike PMC's own /bin/ paths, does not gate on User-Agent -- PMC
    # answers a non-browser client with a "Preparing to download..." interstitial
    # rather than the file. The ZIP also carries the figures, which the /bin/
    # paths do not.
    #
    # Worth having because it settles things the article body does not:
    # Supplementary Methods 3.3.3 states the evaluation axis and the data scale
    # and Supplementary Data 1 carries the per-profile growth phase that defines
    # the paper's 2,610-profile exponential subset -- a column the released
    # expression table omits entirely.
    "supplementary_zip": [
        "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC5059772/supplementaryFiles",
    ],
    # --- Genome-scale metabolic model for FBA ---
    "bigg_ijo1366": [
        "http://bigg.ucsd.edu/static/models/iJO1366.json",
        "https://raw.githubusercontent.com/opencobra/cobratoolbox/master/test/models/json/iJO1366.json",
    ],
}

# Local filenames for the above.
REMOTE_FILES = {
    "kegg_gene_list": NET_DIR / "kegg_eco_gene_list.tsv",
    "bigg_ijo1366": MODEL_DIR / "iJO1366.json",
    "supplementary_zip": SUPP_DIR / "europepmc_supplementary.zip",
}

# Files extracted from `supplementary_zip`. The ZIP names them s1..s10 by
# position; each xlsx also states its own identity in its first cell, and the
# two agree on a clean offset -- sN.xlsx IS "Supplementary Data N-1", for all
# eight, with s1 the Methods PDF and s10 the peer-review file:
#
#   s2 Data 1 metadata      s4 Data 3 new functional terms  s6 Data 5 growth
#   s3 Data 2 interactions  s5 Data 4 FBA reaction bounds   s7 Data 6 stress pairs
#                                                           s8 Data 7 growth molecules
#                                                           s9 Data 8 per-group PCC
#
# The journal's LANDING PAGE disagrees, and it is the one that is wrong: its
# eight captions are permuted relative to its own download links (the byte sizes
# it lists -- 812/12674/112/12/79/175/61/84 kB -- match s2..s9 in order, so the
# links are fine and the captions are not). Read that page and you conclude the
# metadata file is "Data 6" and the FBA bounds are "Data 8". Each file's own
# first cell is authoritative; check a citation against that, not the listing.
SUPPLEMENTARY = {
    "methods_pdf": SUPP_DIR / "ncomms13090-s1.pdf",        # Figs 1-28, Tables, METHODS
    "metadata": SUPP_DIR / "ncomms13090-s2.xlsx",          # Data 1: compendium meta-data
    "interactions": SUPP_DIR / "ncomms13090-s3.xlsx",      # Data 2: molecular interactions
    "fba_bounds": SUPP_DIR / "ncomms13090-s5.xlsx",        # Data 4: FBA reaction bounds
    "growth": SUPP_DIR / "ncomms13090-s6.xlsx",            # Data 5: growth dynamics
    "growth_molecules": SUPP_DIR / "ncomms13090-s8.xlsx",  # Data 7: growth-predictive molecules
    "group_pcc": SUPP_DIR / "ncomms13090-s9.xlsx",         # Data 8: per-GO-group MOMA PCC
    "peer_review": SUPP_DIR / "ncomms13090-s10.pdf",
}

# --------------------------------------------------------------------------
# Raw samples for the preprocessing demo
# --------------------------------------------------------------------------
# GSE73673 is this paper's own RNA-Seq: the 16 knockouts selected by the
# GO-coverage analysis and used as the prospective validation set.
GEO_RNASEQ_SERIES = "GSE73673"

# A small E. coli K-12 Affymetrix E. coli Antisense v2 series (GPL199), used to
# exercise the CEL reader and the from-scratch RMA implementation.
# NOT the E. coli Genome 2.0 array (GPL3154), which this comment used to name:
# the CDF must match the array, and GPL199's is `ecoliasv2cdf` (544 x 544 grid,
# see scripts/00_acquire.py CDF_PKG/CDF_NCOL). Installing the Genome 2.0 CDF on
# these files gives a probe-set mismatch, not an error.
GEO_ARRAY_SERIES = "GSE12411"
GEO_ARRAY_MAX_SAMPLES = 6

GEO_FTP = "https://ftp.ncbi.nlm.nih.gov/geo/series"


def geo_series_dir(gse: str, kind: str) -> str:
    """Build the NCBI GEO FTP directory URL for a series.

    GEO buckets series into directories by masking the last three digits,
    e.g. GSE73673 -> GSE73nnn. `kind` is 'suppl' or 'matrix'.
    """
    stub = gse[:-3] + "nnn" if len(gse) > 6 else gse
    return f"{GEO_FTP}/{stub}/{gse}/{kind}/"


# --------------------------------------------------------------------------
# Reference values from the paper, used as assertions across the codebase
# --------------------------------------------------------------------------
PAPER = {
    # Compendium shape
    "profiles_total": 4389,
    "conditions_total": 649,
    "transcriptome_profiles": 3579,   # the public file has 3578; see docs/
    "transcriptome_genes": 4096,
    "transcriptome_conditions": 596,
    "proteome_profiles": 71,          # public file is condition-averaged: 33
    "proteome_proteins": 1001,        # public file: 589
    "metabolome_profiles": 696,       # public file is condition-averaged: 49
    "metabolome_metabolites": 356,    # public file: 114
    "fluxome_profiles": 43,
    "fluxome_fluxes": 120,
    # Meta-data ontology
    "n_strains": 65,
    "n_media": 112,
    "n_stresses": 52,
    "n_perturbations": 286,
    "n_features": 612,
    "strain_features": 154,           # scraped strain.v5.json yields 152
    "medium_features": 120,           # scraped medium.v5.json yields 120 exactly
    # Model
    "memory_depth": 2,
    "knn_k": 3,
    "missing_threshold": 0.70,
    "cpn_corr_threshold": 0.7,
    # Headline performance, for the reproduction table
    "pcc_transcriptome_all": (0.54, 0.15),
    "pcc_transcriptome_tf": (0.68, 0.14),
    "pcc_proteome_all": (0.55, 0.26),
    "pcc_proteome_var50": (0.77, 0.27),
    "pcc_metabolome_core": (0.65, 0.21),
    "pcc_metabolome_noncore": (0.87, 0.15),
    "pcc_fluxome": (0.72, 0.24),
    "pcc_growth_seen": (0.65, 0.01),
    "pcc_growth_novel": (0.76, None),
}

# Cross-layer condition overlaps, verified against the published files.
# db/build.py asserts these; they are the signature that the GP canonicalizer
# in ecomics.db.canon is working.
EXPECTED_OVERLAP = {
    ("transcriptome", "proteome"): 5,
    ("transcriptome", "metabolome"): 6,
    ("transcriptome", "fluxome"): 3,
    ("transcriptome", "phenome"): 179,
    ("proteome", "metabolome"): 25,
    # These three fluxome pairs are asserted because a real join bug once hid
    # in their absence: the fluxome writes perturbations as
    # upper-case gene symbols (TALB(KO)) while every other layer writes
    # b-numbers (b0008(KO)), so only the wild-type conditions -- the ones with
    # no gene name to disagree about -- ever matched. The five pairs above were
    # unaffected, so nothing failed.
    #
    # `db/build.py` now normalizes symbols to b-numbers via
    # networks.gene_symbol_map(). Coverage first, fix second: an unasserted
    # invariant is one nothing can regress against.
    ("proteome", "fluxome"): 22,      # was 1 before normalization
    ("metabolome", "fluxome"): 23,    # was 2
    ("fluxome", "phenome"): 25,       # was 3
}
EXPECTED_T_PROFILES_WITH_PHENOME = 1991  # paper reports 1992

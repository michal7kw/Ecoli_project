"""Paths and remote resource definitions.

Everything the project reads or writes is declared here, so a reader can see the
full data footprint in one place and no module hard-codes a path.

Layout under REPO:
    data/                        the four published Ecomics files (NEVER modified)
    data/external/prokaryomics/  scraped from prokaryomics.com
    data/external/networks/      RegulonDB (TRN), STRING (PPI), KEGG (pathways)
    data/external/models/        BiGG iJO1366 genome-scale metabolic model
    data/external/raw/           raw microarray / RNA-Seq samples for the pipeline demo
    data/ecomics.db              the built SQLite compendium
    data/parquet/                wide numeric matrices, for fast model loading
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

ALL_DIRS = [EXTERNAL, PROK_DIR, NET_DIR, MODEL_DIR, RAW_DIR, SUPP_DIR,
            PARQUET_DIR, RESULTS]


def ensure_dirs() -> None:
    """Create every output directory. Safe to call repeatedly."""
    for d in ALL_DIRS:
        d.mkdir(parents=True, exist_ok=True)


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

# --------------------------------------------------------------------------
# External network / model resources
# --------------------------------------------------------------------------
# NCBI taxonomy 511145 = Escherichia coli str. K-12 substr. MG1655.
STRING_TAXON = "511145"

REMOTE = {
    # --- Transcriptional regulatory network ---
    # The paper used RegulonDB directly. As of 2026 regulondb.ccg.unam.mx is a
    # single-page app that answers EVERY path -- including the old
    # /menu/download/datasets/files/network_tf_gene.txt -- with a 1,653-byte
    # HTML shell, so the historical bulk endpoints are unusable from a script.
    # We therefore use SBRG's PRECISE mirror of the RegulonDB TF-gene network:
    # 8,325 interactions over 237 regulators, targets already keyed by
    # b-number, which is exactly Ecomics' gene identifier. Scale is comparable
    # to the paper's TRN (3,809 TF events + 8,381 sigma-factor events).
    "regulondb_tf_gene": [
        "https://raw.githubusercontent.com/SBRG/precise-db/master/data/TRN.csv",
    ],
    # KEGG gene list, used to map TRN regulator gene NAMES (rpoD, crp, arcA)
    # onto b-numbers so regulators and targets share one identifier space.
    "kegg_gene_list": ["https://rest.kegg.jp/list/eco"],
    # --- Protein-protein interactions ---
    "string_links": [
        f"https://stringdb-downloads.org/download/protein.links.v12.0/{STRING_TAXON}.protein.links.v12.0.txt.gz",
        f"https://stringdb-static.org/download/protein.links.v11.5/{STRING_TAXON}.protein.links.v11.5.txt.gz",
    ],
    "string_info": [
        f"https://stringdb-downloads.org/download/protein.info.v12.0/{STRING_TAXON}.protein.info.v12.0.txt.gz",
        f"https://stringdb-static.org/download/protein.info.v11.5/{STRING_TAXON}.protein.info.v11.5.txt.gz",
    ],
    # --- KEGG pathway membership (REST API, plain text, two columns) ---
    "kegg_pathway_gene": ["https://rest.kegg.jp/link/eco/pathway"],
    "kegg_pathway_list": ["https://rest.kegg.jp/list/pathway/eco"],
    # --- The paper's own supplementary material ---
    # CC-BY, open access. Europe PMC serves every supplementary file as one ZIP
    # and, unlike PMC's own /bin/ paths, does not gate on User-Agent -- PMC
    # answers a non-browser client with a "Preparing to download..." interstitial
    # rather than the file. The ZIP also carries the figures, which the /bin/
    # paths do not.
    #
    # Worth having because it settles things the article body does not:
    # Supplementary Methods 3.3.3 states the evaluation axis and the data scale
    # (see DISCREPANCIES.md 3), and Supplementary Data 1 carries the per-profile
    # growth phase that defines the paper's 2,610-profile exponential subset.
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
    "regulondb_tf_gene": NET_DIR / "regulondb_trn.csv",
    "kegg_gene_list": NET_DIR / "kegg_eco_gene_list.tsv",
    "string_links": NET_DIR / f"string_{STRING_TAXON}.protein.links.txt.gz",
    "string_info": NET_DIR / f"string_{STRING_TAXON}.protein.info.txt.gz",
    "kegg_pathway_gene": NET_DIR / "kegg_eco_pathway_gene.tsv",
    "kegg_pathway_list": NET_DIR / "kegg_eco_pathway_list.tsv",
    "bigg_ijo1366": MODEL_DIR / "iJO1366.json",
    "supplementary_zip": SUPP_DIR / "europepmc_supplementary.zip",
}

# Files extracted from `supplementary_zip`, by what they actually contain. The
# PMC listing mislabels these by one -- it calls ncomms13090-s2.xlsx
# "Supplementary Data 1" but also calls s4.xlsx "Data 3" when its own header
# says otherwise -- so these names come from each file's first cell.
SUPPLEMENTARY = {
    "methods_pdf": SUPP_DIR / "ncomms13090-s1.pdf",        # Figs 1-28, Tables, METHODS
    "metadata": SUPP_DIR / "ncomms13090-s2.xlsx",          # Data 1: compendium meta-data
    "interactions": SUPP_DIR / "ncomms13090-s3.xlsx",      # Data 2: molecular interactions
    "fba_bounds": SUPP_DIR / "ncomms13090-s5.xlsx",        # Data 4: FBA reaction bounds
    "growth": SUPP_DIR / "ncomms13090-s6.xlsx",            # Data 5: growth dynamics
    "group_pcc": SUPP_DIR / "ncomms13090-s9.xlsx",         # Data 8: per-GO-group MOMA PCC
    "peer_review": SUPP_DIR / "ncomms13090-s10.pdf",
}

# --------------------------------------------------------------------------
# Raw samples for the preprocessing demo
# --------------------------------------------------------------------------
# GSE73673 is this paper's own RNA-Seq: the 16 knockouts selected by the
# GO-coverage analysis and used as the prospective validation set.
GEO_RNASEQ_SERIES = "GSE73673"

# A small E. coli K-12 Affymetrix E. coli Genome 2.0 series, used to exercise
# the CEL reader and the from-scratch RMA implementation.
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
}
EXPECTED_T_PROFILES_WITH_PHENOME = 1991  # paper reports 1992

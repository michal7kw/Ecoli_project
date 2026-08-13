"""Ecomics + MOMA: a faithful reimplementation of Kim et al. 2016.

Kim M., Rai N., Zorraquino V. & Tagkopoulos I. (2016)
"Multi-omics integration accurately predicts cellular state in unexplored
conditions for Escherichia coli", Nature Communications 7:13090.
doi:10.1038/ncomms13090

Package layout
--------------
  ecomics.config      paths and remote resource definitions
  ecomics.acquire     scrape prokaryomics.com; download networks / models / raw data
  ecomics.pipeline    the normalization pipeline (paper Fig. 2)
  ecomics.db          the Ecomics compendium as SQLite + Parquet
  ecomics.features    the 603-feature condition encoder (paper: 612)
  ecomics.networks    TRN / PPI / CPN / KEGG neighbour sets
  ecomics.moma        the five predictive modules and the full cascade
  ecomics.evaluate    LOCO cross-validation, baselines, metrics

See docs/ for the theory this implements.
"""

__version__ = "0.1.0"
__paper__ = "10.1038/ncomms13090"

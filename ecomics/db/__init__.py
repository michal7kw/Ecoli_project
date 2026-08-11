"""The Ecomics compendium as a queryable database.

  canon.py   condition-key canonicalization (the load-bearing utility)
  schema.sql SQLite schema
  build.py   parse every source -> data/ecomics.db + Parquet matrices
  api.py     query interface used by the model and evaluation code
"""

from ecomics.db.canon import (  # noqa: F401
    ConditionKey,
    canonical_gp,
    parse_transcriptome_cond,
    split_gp,
)

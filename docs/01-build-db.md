
# 01 - `01_build_db.py` - the compendium

**Layer:** all five. 
**Reads:** the three published Ecomics files + `data/external/**`. 
**Writes:** `data/ecomics.db` and `data/parquet/*.parquet`. 

Assembles five omic layers and three ontologies from three sources that disagree about how to write a condition, into one SQLite database plus a set of wide Parquet matrices.

---

## 1.1 What it does

Three data sources write the same experimental condition three incompatible ways:

| Source                | Wild type            | Knockout of *b0756* | Overexpression of *b0624* |
| --------------------- | -------------------- | ------------------- | ------------------------- |
| transcriptome table   | `WT_na` *(reversed)* | `b0756_KO`          | `b0624_OE`                |
| proteome / metabolome | `na`                 | `b0756(KO)`         | `b0624(OE)`               |
| prokaryomics          | *(empty)*            | `b0756 KO`          | `b0624 OE`                |

`ecomics/db/canon.py` maps all dialects onto one canonical form — wild type is the *absence* of a perturbation, written `none`; anything else is a lexicographically sorted `;`-joined list of `GENE(TYPE)`. Everything downstream joins on that.

The build then writes two representations of the same data on purpose:

- **SQLite** (`data/ecomics.db`) — a normalized `condition` / `profile` / `measurement` schema for provenance and ad-hoc queries.
- **Parquet** (`data/parquet/*.parquet`) — the wide numeric matrices models actually train on.

## 1.2 Usage and flags

```
# scripts/01_build_db.py
    python scripts/01_build_db.py            # build, or refuse if one exists
    python scripts/01_build_db.py --force    # delete and rebuild
```

| Flag | Effect |
|---|---|
| `--force` | Delete and rebuild even when `data/ecomics.db` exists. Without it, an existing database causes a refusal and exit `1`. |

## 1.3 What it reads

| Input                                                            | Supplied by                                                        |
| ---------------------------------------------------------------- | ------------------------------------------------------------------ |
| `data/Ecomics.transcriptome.no_avg.v8.txt`                       | the published release                                              |
| `data/Ecomics.proteome.v5.csv`, `data/Ecomics.metabolome.v3.csv` | the published release, condition-**averaged**                      |
| `data/external/prokaryomics/*.json`                              | `scripts/00_acquire.py` — fluxome, phenome, and all three ontologies |

## 1.4 What it writes

`data/ecomics.db` holds `condition`, `profile`, `measurement`, `molecule`, `strain_feature`, `medium_component` and `reaction_bigg` tables; 
`data/parquet/` holds one wide matrix per layer. 

Both are consumed through `ecomics/db/api.py:Ecomics`, which exposes SQLite via `.conn`, `.conditions()` and `.condition_info()`, and Parquet via `.matrix(layer)` → `LayerMatrix`.

`LayerMatrix` carries `condition_keys` alongside `values`, which is what makes LOCO grouping and cross-layer alignment possible at all: `condition_index()`, `averaged_by_condition()`, `subset_conditions()`.

## 1.5 Results

The compendium as built:

| Layer | Profiles | Conditions |
|---|---|---|
| transcriptome | 3,578 | 596 |
| phenome | 253 | 253 |
| metabolome | 49 | 49 |
| proteome | 33 | 33 |
| fluxome | 43 | 31 |
| **distinct conditions, all layers** | | **674** |

Ontologies: **65 strains × 152** genotype features, **112 media × 120** composition features, 69 distinct stresses, 273 distinct perturbations, 26 reactions with a BiGG cross-reference.

Every figure above is printed by the build itself, so a re-run is the check. The condition total
and the perturbation count both moved on 2026-08-11 — 696 → 674 and 296 → 273 — when stress
encoding changed from one column per observed `;`-joined *combination* to one per atomic stressor.
Older prose still quoting 696 or 296 predates that change.

## 1.6 Check

The six lines the build ends with:

```
  ok transcriptome x proteome: 5 conditions (expected 5)
  ok transcriptome x metabolome: 6 conditions (expected 6)
  ok transcriptome x fluxome: 3 conditions (expected 3)
  ok transcriptome x phenome: 179 conditions (expected 179)
  ok proteome x metabolome: 25 conditions (expected 25)
  ok transcriptome profiles with growth data: 1991 (expected 1991)
```

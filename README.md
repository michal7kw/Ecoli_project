# Ecomics / MOMA — a from-scratch reproduction

## Headline results

| | Ours | Paper | |
|---|---|---|---|
| transcriptome, the paper's own protocol | **0.544 ± 0.165** | 0.54 ± 0.15 | 🟢 reproduces |
| proteome, neighbours vs own mRNA | **0.152 vs 0.086** (1.8×) | 0.55 vs 0.34 | 🟢 direction reproduces, level does not |
| proteome coverage, union of 4 networks | **588 / 589** | 1001 / 1001 | 🟢 no single network suffices |
| normalization pipeline vs Bioconductor | **PCC 0.999897** (RMA) | — | 🟢 independent agreement |

> ⚠ **Two numbers are called "PCC" and they are not the same measurement.** Per *molecule* across conditions, and per *profile* across molecules. They differ by roughly 0.3 on identical predictions. `03` prints the first (0.186); the paper reports the second, and on the paper's own protocol the same model with no code change scores **0.544**. Any comparison to the paper must go through `08`. See [`docs/08-methods-faithful-eval.md`](docs/08-methods-faithful-eval.md).

## Install

```bash
python -m venv .venv && source .venv/bin/activate   # .venv/Scripts/activate on Windows
pip install -r requirements.txt
```

**GPU is optional and only `03` uses it.** On Windows the default PyPI torch wheel is CPU-only, and `torch.cuda.is_available()` returns False even with a working driver. For CUDA there:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu130
```

Then check the **build tag**, not the version — it must read `+cu130`.

## Running it

|        | Command                                         | Writes                                               |
| ------ | ----------------------------------------------- | ---------------------------------------------------- |
| **00** | `python scripts/00_acquire.py`                  | `data/external/**`                                   |
| **01** | `python scripts/01_build_db.py`                 | `data/ecomics.db` + `data/parquet/*.parquet`         |
| **02** | `python scripts/02_run_pipeline.py`             | `results/pipeline_{output.npz,validation.json}`      |
| **03** | `python scripts/03_train_moma.py --device cuda` | `results/transcriptome_loco.json` + prediction cache |
| **04** | `python scripts/04_proteome.py`                 | `results/proteome_loco.json`                         |
| **08** | `python scripts/08_methods_faithful_eval.py`    | `results/methods_faithful_eval.json`                 |

`00` and `01` are prerequisites for everything. After that, `02`, `03 → 08` and `04` are independent of one another. `08` refits nothing — it re-scores the out-of-fold predictions `03` cached, under the paper's protocol, so it costs seconds.

```bash
python scripts/00_acquire.py --verify        # checksum every artefact, download nothing
python scripts/01_build_db.py --force        # required to rebuild; deletes the existing DB
python scripts/03_train_moma.py --folds 0    # true leave-one-condition-out, slow
python scripts/03_train_moma.py --depth-sweep 1,2,3,4,5,6,8
```

## Documentation

One page per script, each ending in a "how to tell it worked" section:

- [`docs/01-build-db.md`](docs/01-build-db.md) — the compendium, and the assertions that prove it
- [`docs/02-run-pipeline.md`](docs/02-run-pipeline.md) — the normalization pipeline, on real raw data
- [`docs/03-train-moma.md`](docs/03-train-moma.md) — the transcriptome module (relaxation RNN)
- [`docs/04-proteome.md`](docs/04-proteome.md) — the proteome module (4-network LASSO ensemble)
- [`docs/08-methods-faithful-eval.md`](docs/08-methods-faithful-eval.md) — the paper's own protocol

# High level architecture

![High level architecture of the E. coli multi-omics pipeline](attachments/architecture-overview.png)

# Prediction objectives

![Prediction objectives across the modelled omics layers](attachments/prediction-objectives.png)

# Transcription data - RNN - model

![RNN model architecture for transcriptome prediction](attachments/transcriptome-rnn-model.png)

See [`docs/03-train-moma.md`](docs/03-train-moma.md). Conditions, not profiles, are held out —
Ecomics averages ~6 replicates per condition, so a random split would measure replicate
reproducibility rather than generalization.

# Proteome data - ensemble model

![Ensemble model architecture for proteome prediction](attachments/proteome-ensemble-model.png)

# DATA-ATLAS

## 1 - The two tracks

The repo contains **two data tracks that never touch each other**.

- The **compendium track** takes the *already-normalized* published Ecomics tables plus a web scrape, builds a database, and trains MOMA on it. This is the track that produces every headline result.
- The **pipeline track** takes *raw* CEL and htseq files and re-implements the normalization procedure of the paper's Fig. 2 (but this implementation for now is not deeply tested/validated). Its output (`results/pipeline_output.npz`) is **never** read by `db/build.py` or by any model.

![The compendium track and the pipeline track, with no data path between them](attachments/two-data-tracks.png)

## 2 - Master artifact table

Everything on disk, what produces it and what consumes it.

| Artifact                                     |          Bytes | Format            | Produced by             | Consumed by                             |
| -------------------------------------------- | -------------: | ----------------- | ----------------------- | --------------------------------------- |
| `data/Ecomics.transcriptome.no_avg.v8.txt`   |    270,852,299 | TSV               | *(published, external)* | `01`                                    |
| `data/Ecomics.proteome.v5.csv`               |        115,705 | CSV               | *(published, external)* | `01`                                    |
| `data/Ecomics.metabolome.v3.csv`             |         34,812 | CSV               | *(published, external)* | `01`                                    |
| `data/README.txt`                            |          1,641 | text              | *(published, external)* | humans                                  |
| `data/external/prokaryomics/*.json`          |        6 files | JSON array        | `00`                    | `01`                                    |
| `data/external/networks/*`                   |        6 files | CSV / TSV / gz    | `00`                    | `01`, `03`, `04`, `07`–`14`             |
| `data/external/models/iJO1366.json`          |      2,948,407 | BiGG JSON         | `00`                    | `04`, `13`, `14`                        |
| `data/external/raw/GSE12411/*.CEL`           |        6 files | Affymetrix CEL v3 | `00`                    | `02`                                    |
| `data/external/raw/cdf/ecoliasv2.tsv`        |     11,556,271 | TSV               | `tools/export_cdf.R`    | `02`                                    |
| `data/external/raw/GSE73673/*.htcount.txt`   |       87 files | TSV               | `00`                    | `02`, `05`                              |
| `data/external/supplementary/ncomms13090-s*` |       10 files | PDF / XLSX        | `00`                    | `03`, `04`, `08`–`14`, `16`             |
| `data/ecomics.db`                            |    363,659,264 | SQLite            | `01`                    | `03`, `04`, `08`, `09`, `11`–`16`       |
| `data/parquet/*.parquet`                     | 5 files, 81 MB | Parquet           | `01`                    | `03`, `04`, `05`, `06`, `10`–`14`, `16` |
| `results/transcriptome_predictions.npz`      |         ~95 MB | npz, 6 arrays     | `03`                    | `07`, `08`                              |
| `results/pipeline_output.npz`                |      2,058,765 | npz, 4 arrays     | `02`                    | `15`                                    |

## 3 - Primary sources

Three tables released by the Ecomics authors. They are the paper's *output*: measurements that have already been through the entire normalization pipeline of Fig. 2. Everything the models train on descends from these three files.

| Path                                  |       Bytes | Format | Shape                                          | Averaged?                       |
| ------------------------------------- | ----------: | ------ | ---------------------------------------------- | ------------------------------- |
| `Ecomics.transcriptome.no_avg.v8.txt` | 270,852,299 | TSV    | 3,578 rows × 4,098 cols (2 meta + 4,096 genes) | no — individual profiles        |
| `Ecomics.proteome.v5.csv`             |     115,705 | CSV    | 33 rows × 594 cols (5 meta + 589 proteins)     | **yes** — one row per condition |
| `Ecomics.metabolome.v3.csv`           |      34,812 | CSV    | 49 rows × 119 cols (5 meta + 114 metabolites)  | **yes** — one row per condition |

### Formats

```
# head -2 data/Ecomics.transcriptome.no_avg.v8.txt   (tab-delimited, columns elided)
ID	Cond	m.b4412	m.b1994	m.b2861	m.b4428	m.b0264	...
T0568	MG1655.MD026.RP-overexpress.na_WT	0.0855630708766258	0.130231463088323	...
```

```
# head -2 data/Ecomics.proteome.v5.csv   (columns elided)
Strain,MediumID,Medium,Stress,GP,m.b0002,m.b0003,m.b0004,m.b0008,...
N3433,MD004,MOPS+Glu(0.4%),none,none,1,0,1,0.451,0.318,...,0.971,0.041,NA,0,...
```

```
# head -2 data/Ecomics.metabolome.v3.csv   (columns elided)
Strain,MediumID,Medium,Stress,GP,m.X2HGTA,m.APAME,m.BA4A2H,m.C00009,...
MG1655,MD002,M9+Gly(40%),none,none,0.246,0.197,0.162,0.015,...
```

### Numbers

Values are on a **[0, 1] scale**, not raw copy numbers — the released compendium is min–max scaled, so a value of `0.451` means "45% of the way between this molecule's minimum and maximum across the compendium".

### ID conventions

- **Column ids** carry an `m.` prefix, stripped at load. Underneath: 
	- transcriptome and proteome use **b-numbers** (`b0002`); 
	- metabolome uses **KEGG compound ids for 101 of its 114 columns**, with 13 exceptions — `X2HGTA`, `APAME`, `BA4A2H`, `D08266`, `EMANTTE`, `HMDB02259`, `HMDB02356`, `HMDB06029`, `HMDB11739`, `INBG`, `IPTG`, `PY3H`, `TETRC`. 
	- proteome columns, 583 (ff the 589) are strict `b\d{4}`.
- **Row ids**: transcriptome profiles are `T####`; the averaged tables have no profile id at all, because one row is one condition.
- **Condition** is written two incompatible ways *within these three files alone*: 
	- the transcriptome packs it into one dotted string `{STRAIN}.{MEDIUM_ID}.{STRESS}.{GP}`, 
	- the two CSVs spread it over five columns. 
	- the transcriptome writes wild type as **`na_WT`** while the fluxome (§4) writes **`WT_na`** - reversed. 


> [!warning] 
> And the published proteome and metabolome are *condition-averaged*: the paper trained on 71 proteome and 696 metabolome profiles, which were never released.
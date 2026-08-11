
# 02 - [[scripts/02_run_pipeline.py]] - the normalization pipeline, on real raw data

**Layer:** transcriptome. 
**Reads:** [GSE12411](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE12411) (6 CEL files), [GSE73673](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi) (87 htseq tables). 
**Writes:** `results/pipeline_output.npz`. 

A **separate track** from the model. This demonstrates how the Ecomics compendium was built - Fig. 2 of the paper — and it does **not** feed `01-build-db.md`, which loads the published tables directly. 

---

## 2.1 What it does

Five steps, in the order of the paper's Fig. 2, driven by [[ecomics/pipeline/run.py]]:

| Step | Module                                                         | What it does                                                                         |
| ---- | -------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| 1    | [[ecomics/pipeline/arrays.py]], [[ecomics/pipeline/rnaseq.py]] | platform preprocessing - RMA **from scratch** on CEL files; htseq counts for RNA-Seq |
| 2    | [[ecomics/pipeline/noise.py]]                                  | GMM noise removal - fit a two-component mixture, drop the noise component            |
| 3    | [[ecomics/pipeline/platform.py]]                               | quantile normalization → loess → z-score, to put two platforms on one scale          |
| 4    | [[ecomics/pipeline/absolute.py]]                               | loess absolute quantification against a reference                                    |
| 5    | [[ecomics/pipeline/impute.py]]                                 | missingness filter, then k-NN imputation                                             |

`pipeline/validate.py` then cross-checks steps 1, 3 and 5 against Bioconductor.

## 2.2 Usage and flags

```
# scripts/02_run_pipeline.py
    python scripts/02_run_pipeline.py            # full run + validation
    python scripts/02_run_pipeline.py --no-check # skip the R cross-checks
```

| Flag                | Effect                                                                                                                                                                  |
| ------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--no-check`        | Skip the R/Bioconductor cross-checks. Use when R is unavailable — the pipeline itself is pure Python.                                                                   |
| `--per-gene-zscore` | Use Fig. 2's *literal* per-gene z-score instead of the default. The paper's figure and its text differ on this point; `ecomics/pipeline/platform.py` argues the choice. |

## 2.3 What it reads

| Input | Supplied by | Shape |
|---|---|---|
| GSE12411 CEL files | [`00`](00-acquire.md) | 6 raw Affymetrix arrays, GPL199 |
| GSE73673 htseq tables | [`00`](00-acquire.md) | 87 count tables |
| `data/external/raw/cdf/ecoliasv2.tsv` | [`00`](00-acquire.md) step 5 | 7,312 probe sets, 283,258 probe-cell rows |

## 2.4 What it writes

**Two** files.

`results/pipeline_output.npz`, compressed, with five arrays:

| Key | Meaning |
|---|---|
| `final` | the normalized, imputed expression matrix |
| `genes` | row labels |
| `samples` | column labels |
| `platform` | per-sample platform tag, the grouping step 3 normalizes across |
| `imputed_mask` | boolean, `True` where the value was invented by k-NN rather than measured |

`results/pipeline_validation.json`, written by `scripts/02_run_pipeline.py:write_validation_json`:

| Key | Meaning |
|---|---|
| `platforms`, `n_cel`, `n_htseq`, `n_genes` | what went in |
| `cascade` | median / q1 / q3 / missing at each of the five stages |
| `crosschecks` | the three R agreements, headline PCC plus full detail |
| `half_life_control` | the negative control, with the paper's own targets alongside |
| `synthetic_reference_warning` | travels with the numbers, so a plot cannot lose it |

It exists so the figure atlas can plot these results without hard-coding them. 
`results/figures/pipeline-validation.{png,pdf}` are rendered from it.

## 2.5 Results

Here an independent implementation exists, so agreement can be measured rather than argued:

| Our implementation                 | Checked against                       | Agreement                                             |
| ---------------------------------- | ------------------------------------- | ----------------------------------------------------- |
| RMA from scratch, 6 real CEL files | `affy::rma`                           | **PCC 0.999897**, mean absolute difference 0.017 log₂ |
| quantile normalization             | `preprocessCore::normalize.quantiles` | **exact** — max difference 0.0, including tied values |
| k-NN imputation                    | `impute::impute.knn`                  | **PCC 0.995** on imputed cells                        |
| half-life negative control         | the paper's own control               | **P = 0.642**, not significant (paper: P = 0.41)      |

## 2.6 Check — how to tell it worked

```
  combined matrix: <n> genes x 93 samples across 2 platforms
  half-life negative control: short=… long=…  P=0.642  (not significant, as in the paper)
```

- **`across 2 platforms`** — if it reads 1, only one input loaded and step 3 has nothing to do.
- **`not significant, as in the paper`** — the script says this itself; a `SIGNIFICANT -- investigate` means normalization introduced structure it should not have.
- The R cross-check block must print all four agreements above unless `--no-check` was passed.

## 2.7 Comments

- ⚠ **The absolute scale is synthetic, and the script says so loudly.** Ecomics is calibrated against Taniguchi et al. 2010 copy numbers, which are not redistributable, so step 4 falls back to `synthetic_reference()`. **The machinery is reproduced; the scale is not.** Do not present these as Ecomics' absolute values.
- **This is a demonstration, not the compendium's provenance.** The Ecomics tables in `data/` were produced by the original authors' pipeline, not this one. Nothing else depends on this script's output.
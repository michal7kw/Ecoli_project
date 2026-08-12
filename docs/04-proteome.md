
# 04 - `04_proteome.py` - the proteome module

**Layer:** proteome. 
**Reads:** `data/parquet/{transcriptome,proteome}.parquet` and the three network files under `data/external/networks/`. 
**Writes:** [`results/proteome_loco.json`](../results/proteome_loco.json). 

MOMA's second layer. Where the transcriptome module ([`03`](03-train-moma.md)) predicts expression from a **condition**, this one predicts protein abundance from **expression** — the first step of the paper's cascade, in which each layer's output becomes the next layer's input.

Its model is chosen in inverse proportion to its data. The transcriptome has 3,578 profiles and gets a recurrent network; the proteome shares **5 conditions** with the transcriptome and gets an ensemble of LASSO regressions with a fixed penalty.

---

## 4.1 What it does

For each target protein, regress its abundance on the expression of its **functional neighbours** — not on its own mRNA. Neighbourhoods come from four graphs, and one LASSO is fitted per protein per graph:

| Network | What an edge means | Source |
| ------- | ------------------ | ------ |
| `TRN`   | a regulatory link, TF → target gene | RegulonDB |
| `PPI`   | a protein–protein interaction | STRING, *E. coli* K-12 (511145) |
| `KEGG`  | membership of the same pathway | KEGG `eco` pathway–gene table |
| `CPN`   | co-expression above threshold | built from the proteome itself |

The ensemble prediction for a protein is the **mean over the networks that have a model for it** — not over all four. Averaging only the covering networks is what produces the paper's coverage result: no single graph reaches every protein, but their union nearly does.

Everything is evaluated by leave-one-**condition**-out cross-validation, against the paper's own baseline: predicting each protein from its **own transcript**, fitted per protein on the training fold only.

> ⚠ **The CPN is built only from proteome conditions held OUT of evaluation.** Co-expression is estimated from the 28 proteome conditions that are *not* among the 5 being predicted. Building it from the evaluation conditions would leak the answer into the graph the model predicts from, and it would inflate the CPN row specifically — which is the strongest single network here, so the effect would look like a finding.

## 4.2 Usage and flags

```
python scripts/04_proteome.py
```

| Flag | Default | Effect |
|---|---|---|
| `--out` | `results/proteome_loco.json` | Output path. |

## 4.3 What it reads

| Input | Supplied by | Used for |
|---|---|---|
| `data/parquet/transcriptome.parquet` | [`01`](01-build-db.md) | the predictors, condition-averaged |
| `data/parquet/proteome.parquet` | [`01`](01-build-db.md) | the targets: 589 proteins over 33 conditions |
| `data/external/networks/regulondb_trn.csv` | `scripts/00_acquire.py` | TRN edges |
| `data/external/networks/string_511145.protein.links.txt.gz` | `scripts/00_acquire.py` | PPI edges |
| `data/external/networks/kegg_eco_pathway_gene.tsv` | `scripts/00_acquire.py` | pathway membership |

Only **5 conditions** carry both a transcriptome and a proteome measurement, so the LOCO evaluation runs over those 5; the other 28 proteome conditions are used to build the CPN.

## 4.4 What it writes

`results/proteome_loco.json`, one block per predictor — the four networks alone, their `ENSEMBLE`, and `own_mRNA_baseline`. Each carries the standard metrics dict, plus:

| Key | Meaning |
|---|---|
| `coverage` | proteins this predictor produced any finite prediction for |
| `pcc_row_mean` | PCC per profile, across proteins — **the axis to read here** |
| `pcc_mean` | PCC per protein, across conditions — **suppressed**, see below |
| `pcc_column_suppressed` | `true`, because 5 conditions is below the floor of 15 |
| `n_conditions_available` | `5` — the reason for the suppression |

## 4.5 Results

| predictor | proteins | PCC / molecule | PCC / profile | RMSE |
|---|---|---|---|---|
| TRN | 471 | n/a (5 cond) | 0.009 | 1.0505 |
| PPI | 575 | n/a (5 cond) | 0.131 | 0.7049 |
| KEGG | 426 | n/a (5 cond) | 0.155 | 0.7831 |
| CPN | 574 | n/a (5 cond) | 0.196 | 0.5461 |
| **ENSEMBLE** | **588** | n/a (5 cond) | **0.152** | 0.5512 |
| own mRNA | — | n/a (5 cond) | 0.086 | 0.5757 |

Two of the paper's claims reproduce.

- 🟢 **Neighbours beat own mRNA.** 0.152 against 0.086 per profile — a factor of 1.8. This is the paper's central proteome claim, and the direction holds. The *level* does not: the paper reports 0.55 ± 0.26 for the ensemble and 0.34 ± 0.18 for the own-mRNA baseline, both far above ours.
- 🟢 **No single network covers every protein; the union does.** Best single network 575/589 (PPI); union 588/589. The paper reports 250 / 547 / 847 / 1,000 for TRN / KEGG / CPN / PPI against a union of 1,001.

### Why the levels cannot match ?

The paper trained on **71 proteome profiles × 1,001 proteins**. Only **33 condition-averaged rows × 589 proteins** were ever published. Averaging replicates away before release removes the within-condition variance the model was originally fitted against, and it is what reduces 1,001 proteins to 589. The gap in absolute PCC is therefore not evidence about the method; it is what is left after the published release.

## 4.6 Comments

- **The LASSO penalty is fixed at `alpha=1e-3`, not cross-validated.** With 5 conditions an inner CV over 3 folds of 4 samples selects noise, and it multiplies runtime by ~40×. The reasoning is in `ecomics/moma/proteome.py:ProteomeEnsemble`.
- **Neighbourhoods are capped at 200 genes**, keeping the most variable, for the same reason: a 500-neighbour design on a handful of samples makes LASSO's selection arbitrary.

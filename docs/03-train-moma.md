
# 03 - `03_train_moma.py` - the transcriptome module

**Layer:** transcriptome. 
**Reads:** `data/parquet/transcriptome.parquet` via [`ecomics/db/api.py`](../ecomics/db/api.py). 
**Writes:** [`results/transcriptome_loco.json`](../results/transcriptome_loco.json) and `results/transcriptome_predictions.npz`

Trains MOMA's relaxation RNN — the only one of the five modules with enough data to be trained at all — and evaluates it by leave-one-**condition**-out cross-validation against the paper's three baselines.

> ⚠ **The headline this script prints is not comparable to the paper's headline.** `03` reports PCC **per molecule across conditions**; the paper reports PCC **per profile across molecules**, under a further three protocol choices. For the like-for-like number, run [`08`](08-methods-faithful-eval.md). Nothing about the model differs between them.

---

## 3.1 What it does

The transcriptome module implements the paper's relaxation recurrence

```
y⁽ⁱ⁾ = σ(w_x·x + w_y·y⁽ⁱ⁻¹⁾)      for i = 1 … n
```

where `x` is the encoded condition (603 features: strain, medium, stress, perturbation) and `y` is the 4,096-gene expression profile. `w_y` is factorized to rank 64 by default rather than held as a full 4,096 × 4,096 matrix. Memory depth `n` defaults to `C.PAPER["memory_depth"]` = 2, the paper's value.

Evaluation runs through `ecomics/evaluate.py:run_loco`, which is the same contract every other MOMA layer is scored under: hold out whole conditions, predict them, and score against random, mean and wild-type baselines with a Wilcoxon p-value, reporting RMSE and calibration slope alongside PCC.

**Conditions, not profiles, are held out.** Ecomics averages ~6 replicate profiles per condition, so a random profile split lets a model score well by recognising a condition's own replicates — it would measure replicate reproducibility, not generalization.

## 3.2 Usage and flags

```
# scripts/03_train_moma.py
    python scripts/03_train_moma.py                      # 5-fold grouped LOCO
    python scripts/03_train_moma.py --folds 0            # true leave-one-condition-out
    python scripts/03_train_moma.py --epochs 40 --depth 2
    python scripts/03_train_moma.py --depth-sweep        # sweep depths 1,2,3,4
    python scripts/03_train_moma.py --depth-sweep 1,2,3,4,5,6,8   # the published curve
```

| Flag                | Default                           | Effect                                                                                                                                |
| ------------------- | --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| `--folds`           | `5`                               | Grouped folds; `0` means true leave-one-condition-out (596 fits, slow). **Every published run here used 5**                           |
| `--epochs`          | `600`                             | Maximum epochs; early stopping on validation PCC usually halts sooner.                                                                |
| `--depth`           | `C.PAPER["memory_depth"]` (2)     | Memory depth `n` of the relaxation.                                                                                                   |
| `--l1`              | `0.0`                             | L1 penalty on the weights.                                                                                                            |
| `--rank`            | `64`                              | Low-rank factorization of `w_y`; `0` uses the full matrix.                                                                            |
| `--weight-decay`    | `1e-4`                            | L2, applied to `w_x`. Biases are excluded - re-including them undoes the bias initialization, which is worth ~0.15 PCC.               |
| `--lr`              | `3e-4`                            | Adam learning rate.                                                                                                                   |
| `--depth-sweep`     | *(off)*                           | Sweep depths. Bare flag sweeps `1,2,3,4`;                                                                                             |
| `--wy-weight-decay` | `0.0`                             | Weight decay on `w_y` alone. **Zero is load-bearing** — any positive value kills the recurrence outright.                             |
| `--device`          | `cpu`                             | `cpu`, `cuda`, or `cuda:N`. Validated against `torch.cuda.is_available()` and exits 1 with a readable message if CUDA is unavailable. |
| `--out`             | `results/transcriptome_loco.json` | Output path.                                                                                                                          |

## 3.3 What it reads

| Input | Supplied by | Used for |
|---|---|---|
| `data/parquet/transcriptome.parquet` | [`01`](01-build-db.md) | 3,578 profiles × 4,096 genes over 596 conditions |
| the condition ontology in `data/ecomics.db` | [`01`](01-build-db.md) | `ecomics/features.py:build_encoder` → the 603-feature input vector |
| `C.REMOTE_FILES["regulondb_tf_gene"]` | `scripts/00_acquire.py` | the TF subset the paper reports separately (200 TFs here) |

If the RegulonDB file is absent, `tf_indices` returns an empty array and the `tfs` block is simply omitted from the output rather than failing the run.

## 3.4 What it writes

`results/transcriptome_loco.json` — one top-level key per depth trained, named `depth_<n>`. With the default flags that is a single key, `depth_2`.

| Key | Type | Meaning |
|---|---|---|
| `memory_depth` | int | the depth this entry was trained at |
| `config` | object | **every argparse value**, plus the derived counts below |
| `all_genes` | object | metrics over all 4,096 genes |
| `baselines` | object | the same metric block for `random`, `mean`, `wildtype` |
| `p_values` | object | Wilcoxon p, model vs each baseline |
| `seconds` | float | wall-clock for this depth |
| `tfs` | object | the same metric block restricted to TF columns; absent if no TFs resolved |

Every metric block — `all_genes`, each baseline, `tfs` — has the same shape:

| Key | Meaning |
|---|---|
| `pcc_mean`, `pcc_sd`, `pcc_sem` | **per molecule across conditions** — this repo's primary axis |
| `n` | molecules correlated (4,096 genes; 200 for `tfs`) |
| `frac_above_0.3`, `frac_above_0.5` | fraction of molecules clearing that PCC |
| `pcc_row_mean`, `pcc_row_sd` | **per profile across molecules** — the paper's axis |
| `n_rows` | profiles scored |
| `rmse` | root mean squared error, which PCC cannot see |
| `calibration_slope` | slope of truth on prediction; < 1 means range compression |
| `n_conditions_available` | independent observations behind the per-molecule axis |
| `pcc_column_suppressed` | true when `n_conditions_available` fell below the caller's floor |

## 3.5 Results

From [`results/transcriptome_loco.json`](../results/transcriptome_loco.json), depth 2, 5-fold grouped LOCO, `lr=3e-4`, `rank=64`, `wy_weight_decay=0`:

| Predictor | PCC / molecule | > 0.3 | PCC / profile | RMSE | p |
|---|---|---|---|---|---|
| **MOMA transcriptome** | **0.186 ± 0.107** | 15.0% | 0.600 | 0.096 | — |
| baseline: random | −0.100 | 0.0% | 0.580 | 0.099 | 0.0 |
| baseline: mean | −0.106 | 0.0% | 0.580 | 0.099 | 0.0 |
| baseline: wildtype | −0.101 | 0.0% | 0.578 | 0.099 | 0.0 |
| 200 transcription factors | 0.176 ± 0.108 | 13.5% | 0.567 | 0.087 | — |

**The baselines are the thing to look at.** On the per-molecule axis a condition-blind predictor scores ≈ 0 by construction, so all three land slightly negative and no representation choice can inflate the model's margin. That is exactly why this axis is the repo's primary one. It is also why these baselines cannot be compared to the paper's positive 0.25 / 0.26 / 0.36 — those must be on the other axis.

### Against the paper

| | Ours | Paper | |
|---|---|---|---|
| PCC, all genes | 0.186 per molecule | 0.54 ± 0.15 | **not comparable** — different axes |
| PCC, all genes, paper's protocol | **0.544 ± 0.165** ([`08`](08-methods-faithful-eval.md)) | 0.54 ± 0.15 | 🟢 reproduces |
| PCC, TFs | 0.176 per molecule / 0.550 paper-protocol | 0.68 ± 0.14 | 🔴 the +0.14 TF advantage does not appear |
| optimal memory depth | knee at 2, plateau by 3 | 2 | 🟡 depends on the criterion |

Every number in the two tables above comes from `results/transcriptome_loco.json` and `results/methods_faithful_eval.json` in this repository. If they disagree, the JSON is right.

### The depth curve

`python scripts/03_train_moma.py --depth-sweep 1,2,3,4 --out results/depth_sweep_loco.json`, from [`results/depth_sweep_loco.json`](../results/depth_sweep_loco.json) — same configuration as §3.5, so these are directly comparable with the table above:

| depth              | 1     | 2         | 3      | 4      |
| ------------------ | ----- | --------- | ------ | ------ |
| PCC / molecule     | 0.042 | **0.186** | 0.225  | 0.227  |
| genes > 0.3        | 1.9%  | 15.0%     | 25.4%  | 25.6%  |
| PCC / profile      | 0.591 | 0.600     | 0.612  | 0.610  |

Depth 2 captures **78% of the gain available up to depth 4**, and depth 3 captures **99%** — after which the curve is flat, 3 → 4 adding only +0.002. So depth 2 is a defensible **knee**.

Two things are worth stating precisely, because they are easy to overstate.

- **Depth 1 collapses.** At 0.042 the model is barely better than the condition-blind baselines (≈ −0.10). The recurrence is not a refinement here; without at least one relaxation step the architecture does not work at all.

> ⚠ **A sweep needs an explicit `--out`.** `--out` defaults to `results/transcriptome_loco.json`, and a sweep writes one block per depth, so running it without `--out` overwrites the single-depth result of record with a differently-shaped file. The prediction cache is safe — `03` only writes the npz when exactly one depth was requested.
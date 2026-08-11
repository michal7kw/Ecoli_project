
# 08 - `08_methods_faithful_eval.py` - the paper's own protocol

**Layer:** transcriptome. **Reads:** `results/transcriptome_predictions.npz`, Supplementary Data 1. **Writes:** [`results/methods_faithful_eval.json`](../../results/methods_faithful_eval.json).

---

## 8.1 What the Supplementary Methods actually say

Supplementary Methods §3.3.3 states **four** choices outright.

|               | What §3.3.3 specifies                                                                  | What was done in script 3           |
| ------------- | -------------------------------------------------------------------------------------- | ----------------------------------- |
| **axis**      | one PCC per test **condition**, across genes, against the condition-**averaged** truth | one PCC per gene, across conditions |
| **scale**     | per-**gene** min–max, `y' = (y − min)/(max − min)`                                     | the published absolute scale        |
| **subset**    | exponential-phase profiles only                                                        | all 3,578 profiles                  |
| **wild type** | MG1655 in LB or M9, any carbon source, no stress, no genetic perturbation              | `gp == none` — 70% of profiles      |

## 8.2 Method

Re-score the cached out-of-fold predictions under all four choices at once.

The random baseline is averaged over `--random-draws` independent draws.

## 8.3 Usage and flags

```
python scripts/08_methods_faithful_eval.py
```

| Flag | Default | Effect |
|---|---|---|
| `--random-draws` | `200` | Draws to average the random baseline over. The Methods specify 1,000; the estimate is flat past ~200. |
| `--out` | `results/methods_faithful_eval.json` | Output path. |

## 8.4 What it reads

| Input                                                              | Supplied by              | Hard requirement?                                           |
| ------------------------------------------------------------------ | ------------------------ | ----------------------------------------------------------- |
| `results/transcriptome_predictions.npz`                            | [`03`](docs/scripts/03-train-moma.md) | **yes**                                                     |
| Supplementary Data 1, sheet `Transcriptome`, column `Growth Phase` | [`00`](00-acquire.md)    | **yes** — the subset filter is unreconstructable without it |
| `data/prokaryomics/medium.json`                                    | [`00`](00-acquire.md)    | for the LB/M9 medium IDs in the wild-type definition        |

## 8.5 Results

### The subset reproduces exactly

Applying the exponential-phase filter gives **2,610 profiles over 493 conditions** — the paper's own stated numbers, to the profile.

### The model reproduces, and slightly exceeds

| | Ours | Paper |
|---|---|---|
| **MOMA** | **0.578 ± 0.211** | 0.54 ± 0.15 |
| random | 0.528 | 0.25 |
| mean | 0.529 | 0.26 |
| wild type | 0.514 | 0.36 |
| **model − baseline** | **0.049** | **0.280** |
| TFs (MOMA) | 0.580 | 0.68 |

🔴 **What remains open is the baseline level, not the model.** Baselines land near 0.53 where the paper reports 0.25/0.26/0.36, so margin over baseline is **0.049 against the paper's 0.280**.
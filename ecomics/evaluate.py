"""Evaluation: LOCO cross-validation, the paper's baselines, and metrics.

Leave-one-condition-out
-----------------------
Ecomics averages 6.0 replicate profiles per condition. Under random k-fold,
replicate 1 of a condition lands in training while replicate 2 lands in test,
so the model only has to reproduce something it has effectively already seen --
the score then measures replicate reproducibility, not generalization. LOCO
holds out EVERY profile of a condition at once, which is the "unexplored
conditions" claim in the paper's title.

The residual leak LOCO does not close: "M9 + glucose" and "M9 + glucose + IPTG"
are different conditions sharing 119 of 120 medium features, so they can still
land on opposite sides of a split. Group by medium or by strain to tighten it;
`loco_splits(group_by=...)` supports that.

The three baselines (paper.md:73, specified in Supplementary Methods §3.3.3)
----------------------------------------------------------------------------
    random     mean over a 10-profile random draw, averaged over 1000 draws
    mean       mean expression over all training profiles
    wild-type  mean over WILD-TYPE training profiles

Wild-type is the meaningful bar: it encodes "assume the perturbation changed
nothing", which for the many genes genuinely unaffected is the correct answer.

CAVEAT on the wild-type baseline, and it is a large one. `is_wildtype` is
supplied by the caller, and every caller here derives it from
`canon.ConditionKey.is_wildtype`, which means "no genetic perturbation" and
nothing else -- 2,510 of 3,578 transcriptome profiles, 70.2%. The Methods define
it far more narrowly: "the MG1655 strain in LB or M9 medium, any carbon source,
without any stresses or genetic perturbation" -- roughly 12%. At 70% the
"wild-type mean" IS approximately the overall mean, which is why our wild-type
and mean baselines sit on top of each other while the paper's separate by +0.10.
Use `db.api.Ecomics.wildtype_mask(strict=True)` for the paper's definition; the
broad one is kept as the default because most callers want "unperturbed", not
"the paper's baseline reference set".

Metrics
-------
PCC is the paper's metric and is reported for comparability, but it is scale-
and shift-invariant: a prediction uniformly ten-fold too high scores 1.0. For a
compendium whose entire point is getting absolute levels right, that hides real
error, so RMSE and the calibration slope are reported alongside. Fig. 5h shows
exactly the range compression the slope catches and PCC does not.

The metric functions themselves live in `ecomics/metrics.py` and are re-exported
here for callers that expect them. They were moved because `moma/transcriptome.py`
early-stops on validation PCC, which made the model import the evaluation harness.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Callable, Iterator, Sequence

import numpy as np

from ecomics.metrics import (calibration_slope, evaluate_predictions,  # noqa: F401
                             pcc_per_column, pcc_per_row, wilcoxon)

__all__ = ["pcc_per_row", "pcc_per_column", "calibration_slope", "evaluate_predictions",
           "wilcoxon", "loco_splits", "Baselines", "out_of_fold_baselines",
           "CVResult", "run_loco", "summarize"]


# --------------------------------------------------------------------------
# splits
# --------------------------------------------------------------------------
def loco_splits(condition_keys: Sequence[str], n_folds: int | None = None,
                group_by: Callable[[str], str] | None = None,
                seed: int = 0) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield (train_idx, test_idx), holding out whole conditions.

    n_folds  None = true leave-ONE-condition-out (as many folds as conditions).
             An integer buckets conditions into that many folds, which is much
             cheaper and is what you want for an expensive model.
    group_by optional function condition_key -> group label, to hold out whole
             media / strains / studies instead of single conditions. Tightens
             the residual leak noted in the module docstring.
    """
    keys = np.asarray(condition_keys)
    labels = np.asarray([group_by(k) for k in keys]) if group_by else keys
    uniq = np.unique(labels)

    if n_folds is None or n_folds >= len(uniq):
        for u in uniq:
            test = np.flatnonzero(labels == u)
            yield np.flatnonzero(labels != u), test
        return

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(uniq))
    for f in range(n_folds):
        held = set(uniq[order[f::n_folds]])
        mask = np.isin(labels, list(held))
        yield np.flatnonzero(~mask), np.flatnonzero(mask)


# --------------------------------------------------------------------------
# baselines
# --------------------------------------------------------------------------
@dataclass
class Baselines:
    """The paper's three baselines, fitted on a training fold."""

    mean_profile: np.ndarray
    wildtype_profile: np.ndarray
    random_profiles: list[np.ndarray] = field(default_factory=list)

    @classmethod
    def fit(cls, y_train: np.ndarray, is_wildtype: np.ndarray,
            n_random: int = 1000, subset: int = 10, seed: int = 0) -> "Baselines":
        """Fit the three baselines on a training fold.

        The random baseline follows Supplementary Methods §3.3.3 literally:
        "the average expression level for each gene calculated from **10 random
        profiles** ... we report the random baseline calculated from **1000
        times of repetitive sampling** without replacement of the 10-profile
        random set."

        So `subset=10` profiles per draw, `n_random=1000` draws, and `predict`
        returns their AVERAGE -- not one sampled draw.

        This was 10 draws of 20, with `predict` picking one draw at random per
        test row. That injects the sampling noise of a single small mean into
        the baseline itself, and noise depresses correlation with the truth:
        measured in `scripts/08`, drawing once understated the random baseline
        by 0.075 PCC and made our baseline spread look wider than it is. It also
        stops random and mean converging, when the Methods make them nearly
        identical -- which is exactly the relationship the paper reports
        (0.25 vs 0.26).

        `scripts/08_methods_faithful_eval.py` already did this correctly, in
        isolation. Putting it in the harness means every layer in
        `results/all_layers.json` now inherits it.
        """
        rng = np.random.default_rng(seed)
        mean_p = np.nanmean(y_train, axis=0)
        wt = (np.nanmean(y_train[is_wildtype], axis=0)
              if is_wildtype.any() else mean_p)
        k = min(subset, y_train.shape[0])
        # Accumulate rather than keep 1000 profiles: at 4,096 genes that is
        # 32 MB per fold for a quantity only ever used as its mean.
        acc = np.zeros_like(mean_p, dtype=float)
        for _ in range(n_random):
            idx = rng.choice(y_train.shape[0], size=k, replace=False)
            acc += np.nanmean(y_train[idx], axis=0)
        return cls(mean_p, wt, [acc / n_random])

    def predict(self, kind: str, n: int, seed: int = 0) -> np.ndarray:
        if kind == "mean":
            return np.tile(self.mean_profile, (n, 1))
        if kind == "wildtype":
            return np.tile(self.wildtype_profile, (n, 1))
        if kind == "random":
            # `random_profiles` holds the resampling AVERAGE as its single
            # entry. `seed` is kept for signature compatibility and unused: the
            # estimator is no longer stochastic at predict time.
            return np.tile(self.random_profiles[0], (n, 1))
        raise ValueError(f"unknown baseline {kind!r}")


def out_of_fold_baselines(Y: np.ndarray, is_wildtype: np.ndarray,
                          condition_keys: Sequence[str], *, seed: int = 0,
                          min_train: int = 2) -> dict[str, np.ndarray]:
    """The three baselines, predicted out of fold, for a layer with no model.

    `run_loco` already does this as a side effect of cross-validating a model.
    This exists for the layer that has no model to cross-validate: FBA needs no
    training data, so `moma/fluxome.py` never enters the harness and would
    otherwise be reported with no baseline at all. The cost of that omission is
    large -- `pcc_row_mean` 0.843 reads as beating the paper's 0.72 until you
    ask what a CONSTANT scores on the same 22 reactions, which is 0.896.

    Held out per CONDITION, like everything else here: each row's baseline is
    built from the rows of the OTHER conditions, so a condition's replicates
    never inform its own baseline. Without that the mean baseline contains its
    own test row and the bar is set too high.

    `min_train` skips a fold that would leave fewer than that many training
    rows. A mean over one profile is not a baseline.

    ⚠ **The seed convention here differs from `run_loco`'s, deliberately.**
    `run_loco` passes `seed + i` so each fold draws a different random sample;
    this passes one `seed` to every fold. The difference is small -- 
    `Baselines.fit` returns the AVERAGE of 1000 draws, not one draw
    -- but it is not zero, so the two are not interchangeable at the last
    decimal. This convention is the one the committed fluxome numbers were
    computed with, and it is kept so that extracting this function moved no
    reported value. Do not "unify" them without re-running and regenerating
    `results/all_layers.json`.

    `run_loco` does NOT call this: it must fit the model and the baselines in
    one pass over the folds, and calling out here would iterate them twice.

    ⚠ **`Y` is NOT upcast here, and that is a decision, not an oversight.**
    `run_loco` opens with `Y = np.asarray(Y, float)`, and `db.matrix(...).values`
    is **float32** for every layer -- so every layer evaluated through the
    harness is scored in float64 while this one is scored in float32. Adding the
    cast here changes each baseline by ~1.6e-7 relative (7e-6 absolute on fluxes
    normalized to glucose = 100), which is invisible at three decimals but is a
    real change to every float in `results/all_layers.json`'s `fluxome.baselines`
    block. Extracting this function was required to move no reported number, so
    the float32 arithmetic the committed run used is preserved verbatim. If the
    asymmetry is ever resolved, resolve it in one deliberate change that
    regenerates the record -- not as a side effect of touching this line.
    """
    Y = np.asarray(Y)
    keys = np.asarray([str(k) for k in condition_keys])
    out = {kind: np.full_like(Y, np.nan)
           for kind in ("random", "mean", "wildtype")}
    for tr, te in loco_splits(keys):
        if tr.size < min_train or te.size == 0:
            continue
        base = Baselines.fit(Y[tr], np.asarray(is_wildtype)[tr], seed=seed)
        for kind, arr in out.items():
            arr[te] = base.predict(kind, len(te))
    return out


# --------------------------------------------------------------------------
# the CV driver
# --------------------------------------------------------------------------
@dataclass
class CVResult:
    """Out-of-fold predictions and metrics for a model and the baselines."""

    y_true: np.ndarray
    y_pred: np.ndarray
    condition_keys: np.ndarray
    baseline_pred: dict[str, np.ndarray]
    metrics: dict = field(default_factory=dict)
    baseline_metrics: dict = field(default_factory=dict)
    p_values: dict = field(default_factory=dict)

    def summary_table(self, label: str = "model") -> str:
        head = (f"{'predictor':<22s} {'PCC/molecule':>16s} {'>0.3':>6s} "
                f"{'PCC/profile':>12s} {'RMSE':>8s} {'p':>10s}")
        rows = [head, "-" * len(head)]

        def line(name, m, p=None):
            ps = f"{p:.1e}" if p is not None and np.isfinite(p) else ""
            return (f"{name:<22s} {m['pcc_mean']:>8.3f} +/-{m['pcc_sd']:<5.3f} "
                    f"{m['frac_above_0.3']:>6.1%} {m['pcc_row_mean']:>12.3f} "
                    f"{m['rmse']:>8.3f} {ps:>10s}")

        rows.append(line(label, self.metrics))
        for name, bm in self.baseline_metrics.items():
            rows.append(line("baseline: " + name, bm, self.p_values.get(name)))
        rows.append("")
        rows.append("  PCC/molecule = across conditions (this repo's primary axis)")
        rows.append("  NOTE: the PAPER's axis is per profile (Suppl. Methods 3.3.3);")
        rows.append("        see scripts/08_methods_faithful_eval.py to compare like for like")
        rows.append("  PCC/profile  = across molecules (inflated by the mean profile)")
        return "\n".join(rows)


def run_loco(fit_predict: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
             X: np.ndarray, Y: np.ndarray, condition_keys: Sequence[str],
             is_wildtype: np.ndarray, n_folds: int | None = None,
             group_by=None, seed: int = 0, verbose: bool = True,
             min_n_for_column_pcc: int = 1) -> CVResult:
    """Leave-one-condition-out CV of `fit_predict(X_tr, Y_tr, X_te) -> Y_hat`.

    `fit_predict` may optionally declare a fourth parameter `keys_tr`, in which
    case it receives the training fold's condition keys. Models that make an
    inner split need them: without keys, an inner validation split can only be
    random over profiles, which re-introduces the replicate leak that holding
    out whole conditions exists to avoid. Detected by signature rather than by
    try/except, so a genuine TypeError inside the model is not swallowed.

    min_n_for_column_pcc
        Suppress the per-molecule axis when fewer than this many CONDITIONS are
        available to correlate across. The proteome has 5 and the metabolome
        6-25; a correlation over 5 points is noise, and reporting it next to the
        paper's 0.55 invites a false comparison. See `metrics.evaluate_predictions`.
    """
    keys = np.asarray(condition_keys)
    Y = np.asarray(Y, float)
    pred = np.full_like(Y, np.nan)
    bpred = {k: np.full_like(Y, np.nan) for k in ("random", "mean", "wildtype")}

    try:
        wants_keys = "keys_tr" in inspect.signature(fit_predict).parameters
    except (TypeError, ValueError):                    # builtins, C callables
        wants_keys = False

    folds = list(loco_splits(keys, n_folds=n_folds, group_by=group_by, seed=seed))
    for i, (tr, te) in enumerate(folds, 1):
        if tr.size == 0 or te.size == 0:
            continue
        pred[te] = (fit_predict(X[tr], Y[tr], X[te], keys_tr=keys[tr])
                    if wants_keys else fit_predict(X[tr], Y[tr], X[te]))
        base = Baselines.fit(Y[tr], is_wildtype[tr], seed=seed + i)
        for kind in bpred:
            bpred[kind][te] = base.predict(kind, len(te), seed=seed + i)
        if verbose and (i % max(1, len(folds) // 10) == 0 or i == len(folds)):
            print(f"    fold {i}/{len(folds)}")

    res = CVResult(Y, pred, keys, bpred)
    # Suppression is decided on the number of distinct CONDITIONS, not profiles:
    # 18 proteome profiles over 5 conditions still give only 5 points to
    # correlate across, because replicates of one condition are not independent.
    n_cond = int(len(np.unique(keys)))
    kw = {"min_n_for_column_pcc": min_n_for_column_pcc, "n_effective": n_cond}
    res.metrics = evaluate_predictions(pred, Y, **kw)
    for kind, bp in bpred.items():
        res.baseline_metrics[kind] = evaluate_predictions(bp, Y, **kw)
        # Tested on the PRIMARY axis (per molecule), matching the paper's
        # Wilcoxon rank-sum comparisons.
        res.p_values[kind] = wilcoxon(res.metrics["pcc_per_molecule"],
                                      res.baseline_metrics[kind]["pcc_per_molecule"])
    return res


def summarize(res: CVResult, label: str, paper: tuple[float, float] | None = None
              ) -> str:
    out = [res.summary_table(label)]
    if paper is not None:
        mu, sd = paper
        sd_s = f" +/- {sd}" if sd is not None else ""
        out.append(f"\n  paper reports: PCC {mu}{sd_s}")
    return "\n".join(out)

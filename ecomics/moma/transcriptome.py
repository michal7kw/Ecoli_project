"""The transcriptome module: a fixed-point (relaxation) recurrent network.

Paper (paper.md:164-168):

    y^(i) = h( w_x . x  +  w_y . y^(i-1) )        for 1 <= i <= n

    x       612 condition features (strain, medium, stress, perturbation)
    y^(0)   INITIALIZATION -- a real, measured expression profile
    w_x     (n_genes x n_features)  condition -> gene
    w_y     (n_genes x n_genes)     gene -> gene: the learned regulatory network
    h       sigmoid
    n       MEMORY DEPTH, found optimal at 2

    "During the training phase, W is adjusted by minimizing the residual sum of
     squares between observed y and predicted y for all training data based on
     stochastic gradient descent... The optimal memory depth was 2 and the
     cycles having length less than 3 account for 75% of all cycles in the TRN
     in E. coli."

This is NOT a sequence RNN
--------------------------
There is no time axis anywhere in Ecomics -- every profile is a single
steady-state snapshot. The index i is an iteration of a FIXED-POINT RELAXATION,
and the cited reference (Pineda 1987) is about recurrent backpropagation for
networks that settle, not about Elman/LSTM sequence modelling. Read it as:
"gene expression is a self-consistent state; start from a known cellular state,
apply the condition, let the regulatory network propagate the consequences a
couple of rounds, and read off where it settles."

What memory depth buys, exactly
-------------------------------
After n iterations an influence has travelled n-1 edges through w_y and stops.
Anything needing a longer path is structurally invisible, however many
parameters the model has. So the paper's n=2 -- chosen purely by
cross-validation -- agreeing with the independent observation that 75% of
E. coli TRN cycles are shorter than 3 is evidence the recurrence models real
propagation rather than just adding capacity.

Weights are TIED across iterations (one w_x, one w_y, applied n times), which is
what makes this a relaxation rather than a 2-layer feed-forward net.

Keeping the recurrence alive: w_y must NOT be weight-decayed
-----------------------------------------------------------
The recurrent term was originally INERT here. After training, |w_y| averaged
1.2e-11 against |w_x| at 3.2e-4 -- a ratio of 2.7e7 -- so y^(i) was effectively
sigmoid(w_x x + b) and the model had degenerated to a feed-forward map. Memory
depth did nothing, and the paper's three y^(0) strategies gave IDENTICAL
predictions in scripts/05_prospective_ko.py.

Two causes were plausible: w_y starts at exactly zero, and weight decay gives it
no reason to grow once w_x already fits the training data. A 2x2 factorial
(scripts/06_recurrence_experiment.py, one held-out fold) separated them:

    w_y init   decay on w_y    PCC     >0.3    final |w_y|   epochs
    zero       yes            0.1975   20.5%     1.2e-11       49
    small      yes            0.1971   20.6%     2.0e-11       49
    zero       NO             0.2932   44.9%     1.1e-02      583
    small      NO             0.2486   33.1%     4.5e-03      139

WEIGHT DECAY WAS THE WHOLE CAUSE. A non-zero init changed nothing -- decay
pulled it straight back to zero. Removing decay from w_y alone is necessary and
sufficient, and it works from an exactly-zero start, so the zero init (which is
what makes the first forward pass reproduce the mean profile) is kept.

Hence wy_weight_decay defaults to 0.0. This is the same fix already applied to
the bias, for the same reason: the parameter holds something shrinkage destroys.

Under full 5-fold LOCO the fix moved the transcriptome from 0.235 to 0.295.
(The current headline is 0.287: threading condition keys into fit_predict
switched y^(0) from an unintended grand mean to the paper's non_specific
strategy, which had never actually run under LOCO. See `initialization`
and open issue 3.8.)

Memory depth: a knee at 2, not a maximum at 2
--------------------------------------------
With the recurrence alive, depth becomes measurable -- it was meaningless while
w_y was dead, because every depth was the same feed-forward model. Under full
5-fold LOCO (results/depth_sweep_loco.json, depth_{5,6,8}_loco.json):

    depth   1       2       3       4       5       6       8
    PCC     0.2344  0.2950  0.3014  0.3074  0.3079  0.3176  0.3185
    delta   --     +0.0606 +0.0065 +0.0059 +0.0005 +0.0097 +0.0009

DEPTH 2 DELIVERS 72% OF THE TOTAL GAIN available up to depth 8 (+0.0606 of
+0.0841). Everything after is an order of magnitude smaller, non-monotone within
noise, and 2 -> 8 costs 2.2x the training time for +0.024 PCC.

Whether this reproduces the paper depends on a criterion the paper never states:
as a strict maximum it does not (8 wins); as the point past which returns
collapse it does, exactly. The second reading is the interesting one -- the
paper's biological argument (75% of TRN cycles shorter than 3) and this
cross-validation curve independently agree about where useful structure stops.

A single held-out fold showed a clean interior PEAK at 2 (0.1979 / 0.2932 /
0.2431 / 0.2523, with 3 and 4 clearly worse) and was briefly written up as
reproducing the paper. Under full CV 3 and 4 are clearly BETTER: the single fold
reversed the shape of the curve, not just its magnitudes. Same mistake, in the
same codebase, as choosing lr on one fold and quoting its score.

Depth 1 remains instructive: it applies w_y once but no influence travels an edge
beyond y^(0), and the optimizer responds by leaving |w_y| two orders of magnitude
smaller than at depth 2. The model declines to use a recurrence that cannot
propagate.

Whether the paper's w_y was non-trivial is still unknowable from the
publication: it reports the architecture and the optimal memory depth but no
weight statistics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

# `metrics`, not `evaluate`: the model must not depend on the evaluation
# harness. Early stopping needs the metric, not the CV driver.
from ecomics.metrics import pcc_per_column

try:
    import torch
    import torch.nn as nn
    _HAVE_TORCH = True
except ImportError:                                   # pragma: no cover
    _HAVE_TORCH = False

__all__ = ["RelaxationRNN", "TranscriptomeModule", "TargetScaler",
           "INIT_STRATEGIES", "trn_adjacency"]

# The three initializations the paper reports (paper.md:184, Fig. 6a).
INIT_STRATEGIES = ("non_specific", "specific", "same_batch")


class TargetScaler:
    """Map expression onto (0,1), invertibly, so a sigmoid output can reach it.

    Why this is needed: the model's output layer is a sigmoid, bounded to (0,1)
    -- which matches the paper's stated architecture and the biological fact
    that transcription has a floor and a ceiling. Ecomics' published proteome
    and metabolome tables are already exactly [0,1]. The transcriptome table is
    99.9% within [0, 0.955] but carries a thin tail out to 59.87:

        > 1 :  12,050 values (0.082%), spread over 710 of 4,096 genes
        > 2 :     552 values (0.004%)

    Left unscaled, a sigmoid can never reach those values and the loss is
    dominated by a handful of unreachable targets.

    The default rescales by a high percentile and clips, which keeps 99.9% of
    the data untouched and folds the tail onto the ceiling. `n_clipped` records
    exactly how many values that affected, so the cost is reported rather than
    hidden.

    NOT THE PAPER'S SCALING, and the difference is structural. Supplementary
    Methods 3.3.3 eq. (1) specifies per-GENE min-max,
    `y_i' = (y_i - min(y_i)) / (max(y_i) - min(y_i))`, "applied for each entry i
    of the first 4,096 entries". This applies ONE global scalar to every gene.

    Why that matters beyond fidelity: per-gene min-max removes each gene's
    characteristic abundance, and that abundance is precisely what makes a
    constant mean-profile predictor score ~0.58 on the per-profile axis. So the
    warning in `metrics.pcc_per_row` about that axis being "dominated by the
    mean expression profile" is partly an artefact of NOT applying the paper's
    scaling; `scripts/08_methods_faithful_eval.py` applies it at evaluation time
    and the row axis becomes the meaningful one there.

    Training on the paper's scale is a larger change than it looks -- it would
    invalidate the depth sweep, the recurrence factorial and every published
    per-molecule number at once -- so it is recorded here rather than made
    silently. See open issue 3.7.
    """

    def __init__(self, percentile: float = 99.9):
        self.percentile = percentile
        self.scale_: float = 1.0
        self.n_clipped: int = 0
        self.frac_clipped: float = 0.0

    def fit(self, Y: np.ndarray) -> "TargetScaler":
        finite = np.isfinite(Y)
        self.scale_ = float(np.nanpercentile(Y[finite], self.percentile)) or 1.0
        over = finite & (Y > self.scale_)
        self.n_clipped = int(over.sum())
        self.frac_clipped = float(over.sum() / max(finite.sum(), 1))
        return self

    def transform(self, Y: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(Y, float) / self.scale_, 0.0, 1.0)

    def inverse_transform(self, Y: np.ndarray) -> np.ndarray:
        return np.asarray(Y, float) * self.scale_

    def fit_transform(self, Y: np.ndarray) -> np.ndarray:
        return self.fit(Y).transform(Y)


class RelaxationRNN:
    """y^(i) = sigmoid(W_x x + W_y y^(i-1)), weights tied, trained by L1+SGD."""

    def __init__(self, n_features: int, n_genes: int, memory_depth: int = 2,
                 l1: float = 0.0, weight_decay: float = 1e-4,
                 rank: int | None = 64, lr: float = 3e-4, epochs: int = 600,
                 batch_size: int = 128, patience: int = 40, val_frac: float = 0.15,
                 early_stop_metric: str = "pcc",
                 wy_init: float = 0.0, wy_weight_decay: float | None = 0.0,
                 group_inner_split: bool = False,
                 wy_seed: np.ndarray | None = None, wy_seed_scale: float = 0.1,
                 seed: int = 0, device: str = "cpu", verbose: bool = False):
        """
        lr            THE dominant hyper-parameter, and the one that made the
                      first attempt fail. Measured on a held-out fold
                      (results/rnn_tuning2.json), against a ridge control of
                      0.236 per-gene PCC:

                          lr=5e-2  ->  0.028   (the original setting)
                          lr=1e-2  ->  see rnn_tuning2.json
                          lr=1e-3  ->  0.214
                          lr=3e-4  ->  0.267   <- beats the ridge control

                      At 5e-2 with Adam the model overfits inside one epoch --
                      early stopping fired at epoch 1-2 -- so nothing
                      condition-specific was ever learned. 3e-4 is ~3x slower to
                      converge (136 epochs) but is the only setting that beats a
                      plain ridge regression.

        rank          Factorize w_y as U @ V.T with this inner dimension, instead
                      of a full 4096 x 4096 matrix. None keeps the full matrix.

                      This is the single most important change from the naive
                      implementation. A full w_y is 16.8 M parameters fitted
                      against 596 unique conditions -- ~28,000 parameters per
                      condition -- and it duly fit noise: predictions carried
                      93.6% of the truth's variance with ~0 correlation to it.
                      Rank 64 costs 4096*64*2 = 524 K parameters, a 32x
                      reduction, and it is the biologically sensible constraint:
                      a few hundred transcription factors drive several thousand
                      genes, so the regulatory map IS low rank.

        l1            THE PAPER'S REGULARIZER, AND IT IS OFF BY DEFAULT (0.0).
                      Supplementary Methods 3.3.3, "Regularization", is specific:
                      "we used L1 regularization ... the optimal performance
                      (PCC = 0.76 +/- 0.12) is achieved when lambda approximates
                      the value of 0.005."

                      So this class's own summary line -- "trained by L1+SGD" --
                      describes something that does not run unless you ask for
                      it. What actually regularizes the fit is `weight_decay`
                      (L2) plus early stopping, and the optimizer is Adam rather
                      than the supplement's SGD at alpha = 0.01.

                      It is left at 0.0 rather than 0.005 because lambda there is
                      defined against the paper's own loss scaling, which is not
                      this one (`_loss` divides by the observed-cell count), so
                      transplanting the number would be arithmetic theatre.
                      Sweeping l1 on this loss is open issue 3.7's sibling and
                      has not been done -- but the divergence is now stated
                      rather than implied.

        weight_decay  L2 on top of -- in practice, INSTEAD OF -- the paper's L1.
                      At these parameter counts l1 alone does not control the fit.
        patience      Early stopping on an inner validation split, so the epoch
                      budget stops being a tuning parameter.
        """
        if not _HAVE_TORCH:
            raise ImportError("PyTorch is required for the transcriptome module")
        self.n_features = n_features
        self.n_genes = n_genes
        self.memory_depth = memory_depth
        self.l1 = l1
        self.weight_decay = weight_decay
        self.rank = rank
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.val_frac = val_frac
        self.early_stop_metric = early_stop_metric
        # Two knobs for the inert-recurrence question (docs/implementation 17 1.3).
        # wy_init         scale of the random init for w_y's free factor. 0.0 is
        #                 the default: w_y == 0 exactly, so the first forward
        #                 pass reproduces the mean profile. Measured to be
        #                 IRRELEVANT to whether the recurrence survives -- decay
        #                 pulls any non-zero start straight back to zero.
        # wy_weight_decay decay applied to w_y alone. None means "same as the
        #                 rest", which is what killed the recurrence; 0.0 is the
        #                 DEFAULT, and is the fix.
        #
        #                 It defaults on RelaxationRNN itself, not only on the
        #                 TranscriptomeModule wrapper, because callers construct
        #                 this class directly (scripts/05_prospective_ko.py does)
        #                 and would otherwise silently get the broken behaviour.
        self.wy_init = wy_init
        self.wy_weight_decay = wy_weight_decay
        # See the note in `fit`: measured to cost 0.059 PCC with `patience`
        # unchanged, so it is opt-in until the stopping rule is re-tuned.
        self.group_inner_split = group_inner_split
        # wy_seed  an (n_genes, n_genes) array whose [i, j] entry is "gene j
        #          regulates gene i", used as the STARTING POINT for w_y instead
        #          of zero/noise. Supplying the real TRN turns the recurrence
        #          from a free parameter block into a biological prior, which is
        #          what the paper's architecture claims it is.
        #          Low rank forces an approximation: a 0.02%-dense adjacency is
        #          not rank-64, so `_build` takes its truncated SVD and keeps the
        #          64 strongest regulatory directions.
        self.wy_seed = wy_seed
        self.wy_seed_scale = wy_seed_scale
        self.seed = seed
        self.device = torch.device(device)
        self.verbose = verbose
        self.y0: np.ndarray | None = None
        self.history: list[float] = []
        self.best_epoch: int = 0
        self._built = False

    def _build(self, mean_profile: np.ndarray | None = None) -> None:
        """Initialize the weights.

        The bias is the important part. With bias = 0 the sigmoid emits 0.5 for
        every gene, but Ecomics' median scaled expression is ~0.09, so the model
        starts far below the trivial "predict the mean profile" baseline and
        spends its entire optimization budget climbing back to it. Under a fixed
        epoch budget it never arrives, and the result scores BELOW the baseline
        (measured: PCC 0.38 vs a mean-profile baseline of 0.58).

        Initializing bias = logit(mean training profile) starts the model AT the
        baseline, so training is spent learning condition-specific deviations --
        which is what the model is for. This is the same role the paper's
        y^(0) initialization plays (paper.md:184, Fig. 6a): MOMA is a
        perturbation model, not a from-scratch predictor.

        w_y starts at exactly zero so the first forward pass reproduces the mean
        profile exactly, and the recurrence is learned rather than assumed. It
        also keeps the map contractive, so the truncated relaxation is stable.
        """
        torch.manual_seed(self.seed)
        g = torch.Generator().manual_seed(self.seed)
        self.Wx = nn.Parameter((torch.randn(self.n_genes, self.n_features,
                                            generator=g) * 0.01).to(self.device))
        # `wy_init` scales the FREE factor. At 0.0 (the default) w_y is exactly
        # zero; above it, the recurrence starts active. Scaling by 1/sqrt(rank)
        # keeps ||w_y|| independent of the rank, so a sweep varies one thing.
        seed_u, seed_v, seed_full = self._seed_factors()
        if self.rank is None:
            w0 = (torch.randn(self.n_genes, self.n_genes, generator=g)
                  * (self.wy_init / np.sqrt(self.n_genes)))
            if seed_full is not None:
                w0 = w0 + torch.tensor(seed_full, dtype=torch.float32)
            self.Wy = nn.Parameter(w0.to(self.device))
            self.Wy_u = self.Wy_v = None
        else:
            # w_y = U @ V.T. V starts at zero by default, so the product is
            # exactly zero: the first forward pass reproduces the mean profile
            # and the recurrence is learned rather than assumed.
            u0 = torch.randn(self.n_genes, self.rank, generator=g) * 0.01
            v0 = (torch.randn(self.n_genes, self.rank, generator=g)
                  * (self.wy_init / np.sqrt(self.rank)))
            if seed_u is not None:
                # Replace, not add: the seed IS the starting hypothesis, and
                # adding noise to it would dilute exactly the structure under
                # test.
                u0 = torch.tensor(seed_u, dtype=torch.float32)
                v0 = torch.tensor(seed_v, dtype=torch.float32)
            self.Wy_u = nn.Parameter(u0.to(self.device))
            self.Wy_v = nn.Parameter(v0.to(self.device))
            self.Wy = None
        if mean_profile is None:
            b0 = torch.zeros(self.n_genes)
        else:
            p = np.clip(np.nan_to_num(mean_profile, nan=0.5), 1e-4, 1 - 1e-4)
            b0 = torch.tensor(np.log(p / (1 - p)), dtype=torch.float32)
        self.bias = nn.Parameter(b0.to(self.device))

    def _seed_factors(self):
        """Factor `wy_seed` into (U, V, full) for whichever parametrization is in use.

        A regulatory adjacency is not low rank -- 3,444 edges over 4,096 genes is
        0.02% dense and its singular values decay slowly -- so a rank-64 model
        cannot represent it exactly. The truncated SVD keeps the 64 strongest
        regulatory directions, which is the closest thing the parametrization
        admits to "start from the TRN".

        Splitting the singular values as U*sqrt(s) and V*sqrt(s) keeps the two
        factors on the same scale; putting all of s on one side would make that
        factor's gradients ~|s| times larger than the other's and the optimizer
        would effectively train only one of them.
        """
        if self.wy_seed is None:
            return None, None, None
        A = np.asarray(self.wy_seed, dtype=np.float64)
        if A.shape != (self.n_genes, self.n_genes):
            raise ValueError(f"wy_seed must be {(self.n_genes, self.n_genes)}, "
                             f"got {A.shape}")
        # Scale so a PRESENT EDGE has magnitude wy_seed_scale. Normalizing by the
        # mean over all entries instead is a trap: the adjacency is 0.02% dense,
        # so that mean is ~2e-4 and the edges come out ~500x too large.
        peak = np.abs(A).max()
        if peak > 0:
            A = A * (self.wy_seed_scale / peak)
        if self.rank is None:
            return None, None, A

        from scipy.sparse.linalg import svds

        k = min(self.rank, min(A.shape) - 1)
        u, s, vt = svds(A, k=k, random_state=self.seed)
        order = np.argsort(-s)
        u, s, vt = u[:, order], s[order], vt[order]
        root = np.sqrt(np.maximum(s, 0.0))
        U = u * root[None, :]
        V = vt.T * root[None, :]
        if k < self.rank:                      # pad so shapes match the model
            pad = self.rank - k
            U = np.hstack([U, np.zeros((self.n_genes, pad))])
            V = np.hstack([V, np.zeros((self.n_genes, pad))])
        return U, V, None

    # ----------------------------------------------------------- mechanics
    def _forward(self, x: "torch.Tensor", y0: "torch.Tensor",
                 clamp: dict[int, float] | None = None) -> "torch.Tensor":
        y = y0.expand(x.shape[0], -1) if y0.dim() == 1 else y0
        for _ in range(self.memory_depth):
            # y @ w_y.T, computed without ever materializing w_y when low-rank:
            #   w_y = U V^T  =>  y @ w_y.T = (y @ V) @ U.T
            if self.Wy is not None:
                rec = y @ self.Wy.T
            else:
                rec = (y @ self.Wy_v) @ self.Wy_u.T
            y = torch.sigmoid(x @ self.Wx.T + rec + self.bias)
            if clamp:
                y = y.clone()
                for gene, val in clamp.items():
                    y[:, gene] = val
        return y

    def fit(self, X: np.ndarray, Y: np.ndarray,
            y0: np.ndarray | None = None,
            condition_keys: Sequence[str] | None = None) -> "RelaxationRNN":
        """Minimize residual sum of squares with an L1 penalty, by SGD.

        condition_keys  one key per row of X. When given, the inner
                        early-stopping split holds out whole CONDITIONS instead
                        of random profiles -- see the split below for why that
                        matters.
        """
        X_t = torch.tensor(np.asarray(X, np.float32), device=self.device)
        Y_t = torch.tensor(np.asarray(Y, np.float32), device=self.device)
        self.y0 = (np.nanmean(Y, axis=0) if y0 is None
                   else np.asarray(y0, np.float32))
        # Build now, so the bias can be seeded from the TRAINING mean profile.
        if not self._built:
            self._build(mean_profile=np.nanmean(Y, axis=0))
            self._built = True
        y0_t = torch.tensor(np.nan_to_num(self.y0).astype(np.float32),
                            device=self.device)

        params = [self.Wx, self.bias]
        params += [self.Wy] if self.Wy is not None else [self.Wy_u, self.Wy_v]
        for p in params:
            p.requires_grad_(True)
        # The bias is EXCLUDED from weight decay. It carries
        # logit(mean training profile); shrinking it toward 0 pulls the output
        # back to sigmoid(0) = 0.5 for every gene and undoes the whole point of
        # the initialization. Measured: with the bias decayed, per-gene PCC was
        # -0.006; without, see results/rnn_tuning.json.
        wy_params = ([self.Wy] if self.Wy is not None else [self.Wy_u, self.Wy_v])
        wy_wd = (self.weight_decay if self.wy_weight_decay is None
                 else self.wy_weight_decay)
        wy_ids = {id(q) for q in wy_params}
        opt = torch.optim.Adam(
            [{"params": [p for p in params
                         if p is not self.bias and id(p) not in wy_ids],
              "weight_decay": self.weight_decay},
             {"params": wy_params, "weight_decay": wy_wd},
             {"params": [self.bias], "weight_decay": 0.0}], lr=self.lr)

        n = X_t.shape[0]
        gen = torch.Generator().manual_seed(self.seed)
        # Inner split, so early stopping never sees the outer test fold.
        #
        # `group_inner_split` holds out whole CONDITIONS instead of random
        # profiles. A uniformly random split IS the leak `evaluate.py` uses to
        # justify LOCO in the first place: Ecomics averages ~6 replicates per
        # condition, so replicate 1 lands in inner-train while replicate 2 lands
        # in inner-val, and the stopping metric then measures replicate
        # reproducibility rather than generalization.
        #
        # IT IS OFF BY DEFAULT, and the reason is a measurement rather than a
        # preference. Under full 5-fold LOCO, everything else identical:
        #
        #     inner split         PCC/molecule   >0.3    slope
        #     random profiles (default)  0.286    41.4%    0.836
        #     whole conditions           0.227    24.8%    0.815
        #
        # Both rows use the paper's non_specific y0, so the split is the only
        # variable. Getting that clean took three full-LOCO runs; the obvious
        # comparison (0.295 vs 0.227) also changed y0 and was not one.
        #
        # Switching it on costs 0.059 PCC. That is not the leak being removed
        # from an inflated score -- the OUTER fold was never touched, so 0.295
        # was always a valid held-out number. What the grouped split changes is
        # the stopping SIGNAL: it is noisier (fewer, more homogeneous held-out
        # conditions), so with `patience` unchanged early stopping fires sooner
        # and the model is simply undertrained. The falling calibration slope
        # (0.917 -> 0.815, i.e. more range compression) is the tell.
        #
        # Re-tuning `patience` does NOT recover it, which was the obvious
        # explanation and is wrong. Swept on GPU: patience 40 / 120 / 250 gives
        # 0.2239 / 0.2374 / 0.2437, converging near 0.25 and still ~0.035 short,
        # with the calibration slope flat at ~0.79 rather than climbing toward
        # the ungrouped 0.875. The grouped split produces a WORSE model, not an
        # under-trained one.
        #
        # Why a leaky selection signal can beat a clean one: early stopping is a
        # selection problem, and its signal has bias AND variance. Leakage
        # biases the estimate optimistically but lowers its variance; grouping
        # removes the bias and leaves ~72 held-out conditions to estimate a
        # per-gene PCC from, which is noisy enough to pick a worse epoch. Here
        # the variance reduction is worth more than the bias costs.
        #
        # Neither choice affects validity -- the OUTER fold is untouched either
        # way. What the leak buys is a better-chosen stopping epoch, not a
        # better-looking score. Off by default as a measured choice, not a
        # deferral. See open issue 3.7(a).
        n_val = max(1, int(self.val_frac * n)) if self.val_frac > 0 else 0
        if n_val and condition_keys is not None and self.group_inner_split:
            keys = np.asarray(condition_keys)
            uniq = np.unique(keys)
            # Hold out whole conditions until the validation share is reached.
            order = uniq[torch.randperm(len(uniq), generator=gen).numpy()]
            held, taken = set(), 0
            for u in order:
                if taken >= n_val:
                    break
                held.add(u)
                taken += int((keys == u).sum())
            mask = np.isin(keys, list(held))
            # Degenerate guard: one condition covering everything would leave no
            # training rows, in which case fall back to the random split.
            if mask.any() and not mask.all():
                val_idx = torch.as_tensor(np.flatnonzero(mask))
                tr_idx = torch.as_tensor(np.flatnonzero(~mask))
            else:
                condition_keys = None
        if n_val == 0 or condition_keys is None or not self.group_inner_split:
            perm0 = torch.randperm(n, generator=gen)
            val_idx, tr_idx = perm0[:n_val], perm0[n_val:]
        observed = torch.isfinite(Y_t)

        def _loss(idx):
            xb, yb, ob = X_t[idx], Y_t[idx], observed[idx]
            pred = self._forward(xb, y0_t)
            diff = torch.where(ob, pred - torch.nan_to_num(yb),
                               torch.zeros_like(pred))
            return (diff ** 2).sum() / max(int(ob.sum()), 1)

        def _l1():
            reg = self.Wx.abs().sum()
            reg = reg + (self.Wy.abs().sum() if self.Wy is not None
                         else self.Wy_u.abs().sum() + self.Wy_v.abs().sum())
            return reg

        best_val, best_state, bad = float("inf"), None, 0
        for ep in range(1, self.epochs + 1):
            order = tr_idx[torch.randperm(len(tr_idx), generator=gen)]
            for s0 in range(0, len(order), self.batch_size):
                idx = order[s0:s0 + self.batch_size]
                loss = _loss(idx) + self.l1 * _l1()
                opt.zero_grad()
                loss.backward()
                opt.step()

            with torch.no_grad():
                vi = val_idx if n_val else tr_idx
                if self.early_stop_metric == "pcc":
                    # Stop on validation per-GENE PCC -- the reported metric.
                    # Validation MSE is dominated by each gene's mean level,
                    # which is fitted almost immediately, so it plateaus long
                    # before any condition-specific signal is learned and stops
                    # training far too early.
                    pv = self._forward(X_t[vi], y0_t).cpu().numpy()
                    tv = Y_t[vi].cpu().numpy()
                    pc = pcc_per_column(pv, tv)
                    pc = pc[np.isfinite(pc)]
                    val = -float(pc.mean()) if pc.size else 0.0
                else:
                    val = float(_loss(vi).item())
            self.history.append(val)

            if val < best_val - 1e-6:
                best_val, bad = val, 0
                self.best_epoch = ep
                best_state = [p.detach().clone() for p in params]
            else:
                bad += 1
                if self.patience and bad >= self.patience:
                    if self.verbose:
                        print(f"      early stop at epoch {ep} "
                              f"(best {self.best_epoch}, val {best_val:.5f})")
                    break
            if self.verbose and ep % max(1, self.epochs // 10) == 0:
                print(f"      epoch {ep:>4d}/{self.epochs}  val={val:.5f}")

        if best_state is not None:                     # restore the best epoch
            with torch.no_grad():
                for p, b in zip(params, best_state):
                    p.copy_(b)
        for p in params:
            p.requires_grad_(False)
        return self

    def predict(self, X: np.ndarray, y0: np.ndarray | None = None,
                knockouts: Sequence[int] | None = None) -> np.ndarray:
        """Predict expression. `knockouts` clamps those genes to 0 each iteration.

        Clamping is how the paper simulates a knock-out (paper.md:184): "we set
        the expression of that gene to zero in the RNN". Natural in a relaxation
        network, awkward in a feed-forward one.
        """
        X_t = torch.tensor(np.asarray(X, np.float32), device=self.device)
        base = self.y0 if y0 is None else np.asarray(y0, np.float32)
        y0_t = torch.tensor(np.nan_to_num(base).astype(np.float32),
                            device=self.device)
        clamp = {int(g): 0.0 for g in knockouts} if knockouts else None
        with torch.no_grad():
            return self._forward(X_t, y0_t, clamp=clamp).cpu().numpy()

    def propagation_radius(self, source: int, tol: float = 1e-4) -> int:
        """How many edges a perturbation at `source` travels. Should be n-1.

        A direct check that memory depth means what the module docstring says.
        """
        X = np.zeros((1, self.n_features), dtype=np.float32)
        y0 = np.nan_to_num(self.y0) if self.y0 is not None else np.zeros(self.n_genes)
        a = self.predict(X, y0=y0)
        b = self.predict(X, y0=y0, knockouts=[source])
        changed = np.flatnonzero(np.abs(a - b).ravel() > tol)
        return max(len(changed) - 1, 0)


# --------------------------------------------------------------------------
@dataclass
class TranscriptomeModule:
    """Wraps RelaxationRNN with the paper's initialization strategies."""

    memory_depth: int = 2
    l1: float = 0.0
    lr: float = 3e-4
    rank: int | None = 64
    weight_decay: float = 1e-4
    epochs: int = 600
    seed: int = 0
    scale_percentile: float = 99.9
    # 0.0, not `weight_decay`: with w_y decayed the recurrence dies (|w_y| ~1e-11)
    # and the model degenerates to a feed-forward map. Measured on a held-out
    # fold: decayed 0.198, undecayed 0.293. See results/recurrence_experiment.json.
    wy_weight_decay: float | None = 0.0
    # Hold out whole conditions in the inner early-stopping split. Off by
    # default: measured at 0.227 against 0.295, because the stopping rule was
    # never re-tuned for the noisier signal. See RelaxationRNN.fit.
    group_inner_split: bool = False
    device: str = "cpu"
    verbose: bool = False
    model: RelaxationRNN | None = field(default=None, repr=False)
    scaler: "TargetScaler | None" = field(default=None, repr=False)

    def initialization(self, strategy: str, Y_train: np.ndarray,
                       keys_train: np.ndarray,
                       target_key: str | None = None) -> np.ndarray:
        """Build y^(0) under one of the paper's three strategies.

        non_specific  mean over the most-represented condition in training
        specific      mean over training profiles sharing the target's strain
                      and medium
        same_batch    mean over training profiles of the exact target condition
                      -- an upper bound, and NOT available under LOCO by
                      construction, which is the point: it isolates how much of
                      Fig. 6a's performance comes from the starting state.

        ⚠ THIS WAS NOT REACHED UNDER LOCO AT ALL until the audit, and the
        discovery came sideways. `run_loco`'s contract is
        `fit_predict(X_tr, Y_tr, X_te)`, so `scripts/03` passed no condition
        keys, `keys_tr` was None, and `fit_predict` silently took its fallback
        branch: `y0 = mean of the ENTIRE training fold`. None of the paper's
        three strategies ran. Threading `keys_tr` through (done so the inner
        split could group by condition) switched it on, and the two starting
        states are not close -- correlation **0.71**, mean absolute difference
        0.051 on a [0, 1] scale. Measured cost under full 5-fold LOCO:
        **0.295 -> 0.287** per molecule, 44.2% -> 41.2% of genes above 0.3.

        `non_specific` NOW USES THE PAPER'S DEFINITION. It selects the strict
        wild-type set -- MG1655 in LB/M9, no stress, no genetic perturbation
        (Supplementary Methods 3.3.3), 10.3% of profiles -- via
        `db.api.Ecomics.wildtype_mask(strict=True)`.

        It previously took "whichever condition has the most training profiles",
        which on this compendium is `MG1655.MD102.lactose-shift.none` (187
        profiles): the right strain in the wrong state. `y^(0)` is meant to be
        the UNperturbed reference the model perturbs away from, so starting from
        a lactose shift asks the network to predict every other condition as a
        deviation from an already-shifted one.

        The proxy remains as a fallback for when the scraped medium ontology is
        absent, since the strict mask needs it to know which media are LB or M9.
        """
        if strategy == "non_specific":
            try:
                from ecomics.db.api import Ecomics
                wt = Ecomics.wildtype_mask(keys_train, strict=True)
            except Exception:                          # noqa: BLE001
                # The strict mask needs the scraped medium ontology to know
                # which medium IDs are LB or M9. Without it, fall back to the
                # proxy rather than failing a fit.
                wt = np.zeros(len(keys_train), dtype=bool)
            if wt.any():
                return np.nanmean(Y_train[wt], axis=0)
            uniq, counts = np.unique(keys_train, return_counts=True)
            top = uniq[np.argmax(counts)]
            return np.nanmean(Y_train[keys_train == top], axis=0)

        if strategy == "specific" and target_key:
            strain, medium = target_key.split(".")[0], target_key.split(".")[1]
            sel = np.array([k.split(".")[0] == strain and k.split(".")[1] == medium
                            for k in keys_train])
            if sel.any():
                return np.nanmean(Y_train[sel], axis=0)
            return np.nanmean(Y_train, axis=0)

        if strategy == "same_batch" and target_key:
            sel = keys_train == target_key
            if sel.any():
                return np.nanmean(Y_train[sel], axis=0)
        return np.nanmean(Y_train, axis=0)

    def fit_predict(self, X_tr: np.ndarray, Y_tr: np.ndarray, X_te: np.ndarray,
                    keys_tr: np.ndarray | None = None,
                    init: str = "non_specific") -> np.ndarray:
        """Train on a fold and predict it -- the callable run_loco expects.

        The target scaler is fitted on the TRAINING fold only, so no information
        about the held-out condition's scale leaks into the fit.
        """
        self.scaler = TargetScaler(self.scale_percentile).fit(Y_tr)
        Y_s = self.scaler.transform(Y_tr)
        y0 = (self.initialization(init, Y_s, keys_tr)
              if keys_tr is not None else np.nanmean(Y_s, axis=0))
        self.model = RelaxationRNN(
            n_features=X_tr.shape[1], n_genes=Y_tr.shape[1],
            memory_depth=self.memory_depth, l1=self.l1, lr=self.lr,
            rank=self.rank, weight_decay=self.weight_decay,
            epochs=self.epochs, seed=self.seed, device=self.device,
            wy_weight_decay=self.wy_weight_decay, verbose=self.verbose,
            group_inner_split=self.group_inner_split,
        ).fit(X_tr, Y_s, y0=y0, condition_keys=keys_tr)
        return self.scaler.inverse_transform(self.model.predict(X_te))


def trn_adjacency(columns, path=None) -> np.ndarray:
    """The paper's own TRN as an adjacency over `columns`, for seeding w_y.

    Supplementary Data 2 lists 3,489 TF->target edges over 179 regulators, keyed
    by b-number -- the same identifier space as the expression matrix, so no name
    mapping is involved. 3,444 of those edges have both endpoints among the 4,096
    genes.

    Orientation matters and is easy to get backwards. The forward pass computes
    `y @ w_y.T`, so `w_y[i, j]` multiplies gene j's level when producing gene i:
    the entry means "j regulates i". This returns A[target, regulator].
    """
    import openpyxl

    from ecomics import config as C

    path = Path(path) if path else C.SUPPLEMENTARY["interactions"]
    if not path.exists():
        raise FileNotFoundError(
            f"{path}\nRun: python scripts/00_acquire.py")
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    rows = [r for r in wb["TRN"].iter_rows(values_only=True)]
    wb.close()

    idx = {c: i for i, c in enumerate(columns)}
    A = np.zeros((len(columns), len(columns)), dtype=np.float32)
    n = 0
    for r in rows:
        if not r or not r[0] or str(r[0]).strip() == "TF":
            continue
        i, j = idx.get(str(r[1]).strip()), idx.get(str(r[0]).strip())
        if i is not None and j is not None:
            A[i, j] = 1.0                      # regulator j -> target i
            n += 1
    # An all-zero adjacency must NOT be returned quietly. Both endpoints go
    # through `idx.get`, so a change in the sheet's column order, or a switch
    # from b-numbers to gene symbols, matches nothing -- and every downstream
    # step still "works": `svds` of a zero matrix succeeds, `w_y` is seeded with
    # zeros, and `scripts/10_trn_seeded_recurrence.py` ends up comparing the
    # unseeded control against itself while reporting "the biology buys
    # nothing". A wrong answer that looks like a finding.
    if n == 0:
        sample = [str(c) for c in (rows[1][:2] if len(rows) > 1 else [])]
        raise ValueError(
            f"{path}: no TRN edge matched the expression matrix. Expected "
            f"b-numbers in column A (regulator) and B (target); first data "
            f"row reads {sample}.")
    return A

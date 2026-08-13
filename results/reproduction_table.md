# Reproduction table

MOMA re-implemented from the published Ecomics data plus the prokaryomics.com scrape.

All figures are leave-one-condition-out cross-validation. PCC in the table below is
**per molecule, across conditions** -- this repository's primary axis, because a
condition-blind predictor scores ~0 on it by construction and so cannot flatter a model.

> **This is NOT the paper's axis.** Supplementary Methods 3.3.3 measures PCC per test
> CONDITION across genes, against the condition-averaged truth. The two differ by
> roughly 0.3 on identical predictions, so **nothing in the `paper` column below is like-for-like**
> with the `this reproduction` column. For the paper-faithful comparison see
> `results/methods_faithful_eval.json` (`scripts/08`), where the same model scores 0.544
> against the paper's 0.54.

**Rows where the metric does not apply are marked `n/a`, not scored.** Per-molecule PCC
correlates each molecule ACROSS conditions, so it needs conditions. The transcriptome has
596 and the phenome 179; the proteome has **5** and the metabolome 6-25. A 5-point
correlation has a null distribution so wide that ~a third of molecules exceed |r| = 0.7 by
chance, so a number there is noise dressed as a result -- and printing it beside the
paper's 0.55 invites a comparison the sample size cannot support. Those cells now read
`n/a (n cond)`, with the per-profile axis and coverage reported instead. The threshold is
15 conditions (`ecomics/metrics.py:MIN_CONDITIONS_FOR_COLUMN_PCC`).

| layer | metric | this reproduction | paper | note |
|---|---|---|---|---|
| transcriptome | PCC, all genes | 0.186 | 0.54 +/- 0.15 | 3,578 profiles, 596 conditions |
| transcriptome | PCC, random baseline | -0.100 | 0.25 | a constant predictor scores ~0 on this axis; see below |
| transcriptome | PCC, mean baseline | -0.106 | 0.26 | a constant predictor scores ~0 on this axis; see below |
| transcriptome | PCC, wildtype baseline | -0.101 | 0.36 | a constant predictor scores ~0 on this axis; see below |
| transcriptome | PCC, TFs | 0.177 | 0.68 +/- 0.14 | 176 transcription factors |
| proteome | PCC, ensemble | n/a (5 cond) | 0.55 +/- 0.26 | 589 averaged proteins, not 1,001 individual profiles |
| proteome | coverage | 588/589 | 1001/1001 | union of 6 networks (the paper's own, Supplementary Data 2) |
| proteome | PCC, own-mRNA baseline | n/a (5 cond) | 0.34 +/- 0.18 | |
| metabolome | PCC, core from protein | 0.360 | 0.65 +/- 0.21 | 25 shared conditions |
| metabolome | PCC, non-core from transcript | n/a (6 cond) | 0.87 +/- 0.15 | only 6 shared conditions |
| phenome | PCC, consensus | 0.602 | 0.65 +/- 0.01 | 179 conditions with growth; the paper CV'd over 101 |
| phenome | PCC, input layer only | 0.618 | ~0.59 | |

## What reproduced, and what did not

**Reproduced.**

- **The proteome coverage argument**: no single network covers every protein, the union does (588/589 here; 1001/1001 in the paper, against the unpublished per-profile table).
- **The paper's central proteome claim** -- a protein is better predicted from its functional NEIGHBOURS than from its OWN mRNA. Per profile 0.264 vs 0.086 (3.1x): the direction holds.
- **Every cross-layer condition count** matches the paper exactly, including 179 conditions / 1,991 transcriptome profiles with growth data.

**Did not reproduce.**

- **Core metabolome from proteins**: 0.420 +/- 0.390 per PROFILE against the paper's 0.65 +/- 0.21 -- OUTSIDE one standard deviation. Per MOLECULE it is 0.360, on 25 conditions, which is too few for that axis to mean much. The two are not the same measurement and are not comparable to each other.

**The transcriptome, and which axis it is read on.**

- On the PUBLISHED scale, per molecule, MOMA reaches 0.186 against baselines of -0.100 / -0.106 / -0.101. A constant predictor scores ~0 on this axis by construction, so the margin is unambiguous -- but the number is NOT comparable to the paper's 0.54.
- **The paper's own axis is per profile**, against the condition-averaged truth, on per-gene min-max values, over exponential-phase profiles only (Supplementary Methods 3.3.3). None of those four choices is stated in the article body. Applying all four, the same model with no code change scores **0.544 against the paper's 0.54** -- see `scripts/08_methods_faithful_eval.py` and `results/methods_faithful_eval.json`.
- What remains open is the BASELINE LEVEL, not the model: under the paper's protocol ours come out near 0.53 where the paper reports 0.25/0.26/0.36. A baseline has no free parameters, so that gap cannot be a modelling difference -- it is a difference in what the baseline is computed over.

## Why some cells cannot match

- **Proteome and metabolome are condition-AVERAGED in the public release.** The paper trained on 71 proteome profiles x 1,001 proteins and 696 metabolome profiles x 356 metabolites; only 33 x 589 and 49 x 114 averaged rows were ever published.
- **The absolute-scale reference is not redistributable.** Ecomics is calibrated against Taniguchi et al. 2010 copy numbers; the pipeline here reproduces the machinery but demonstrates it on a synthetic reference.
- **Only 26 of 120 flux reactions carry a BiGG cross-reference** in prokaryomics `reaction.json`, and 22 of those exist in iJO1366, so the fluxome comparison covers 22 reactions rather than 120.
- **The 612-feature ontology reconstructs to 603 features** (152 strain + 120 medium + 58 stress + 273 perturbation, at medium_kind='present'): counts here are OBSERVED values, the paper's are its own categorization. Medium matches exactly; every remaining gap is a counting difference. This read 756 (240 medium / 68 stress) under an earlier encoding, and its perturbation block read 296 until the fluxome's gene symbols were normalized to b-numbers.

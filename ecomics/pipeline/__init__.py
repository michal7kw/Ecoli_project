"""The Ecomics normalization pipeline (paper Fig. 2, Methods paper.md:128-156).

Raw platform data -> a matrix of estimated molecules per cell, comparable
across every row and every column.

    arrays.py    CEL reader; RMA from scratch; normexp for two-channel arrays
    rnaseq.py    Trimmomatic/TopHat/htseq wrappers + htseq count-table reader
    noise.py     step 1b  anchored 2-component Gaussian mixture, fitted by EM
    platform.py  step 2   quantile normalization -> loess -> per-platform z-score
    absolute.py  step 3   loess calibration onto absolute copy numbers
    impute.py    step 4   70% missingness filter + k-NN imputation (k=3)
    run.py       the end-to-end driver
    validate.py  the half-life negative control + R/Bioconductor cross-checks

Pure Python + numpy/scipy throughout. R is used only by tools/export_cdf.R as a
one-time acquisition step, and by validate.py as an independent ground truth.
"""

from ecomics.pipeline.absolute import absolute_quantify, fit_absolute_loess  # noqa: F401
from ecomics.pipeline.impute import knn_impute, missingness_filter  # noqa: F401
from ecomics.pipeline.noise import GaussianMixtureNoise, remove_noise  # noqa: F401
from ecomics.pipeline.platform import (  # noqa: F401
    loess_fit_predict,
    platform_zscore,
    quantile_normalize,
)

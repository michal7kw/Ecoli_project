"""MOMA -- Multi-Omics Model and Analytics.

NOTE ON THE NAME: in the metabolic-modelling literature "MOMA" usually means
Minimization Of Metabolic Adjustment (Segre et al. 2002). This is not that.

Five modules, each matched to how much data its layer actually has -- the
paper's most transferable engineering lesson is choosing model rigidity in
INVERSE proportion to data availability:

    transcriptome  3,578 profiles   relaxation RNN        flexible
    proteome          33 conditions ensemble over prior networks
    metabolome     25/6 conditions  LASSO
    fluxome           43 profiles   FBA (mechanistic, needs no training data)
    phenome          253 conditions performance-weighted consensus
"""

from ecomics.moma.transcriptome import RelaxationRNN, TranscriptomeModule  # noqa: F401

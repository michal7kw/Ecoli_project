"""The paper's evaluation protocol, as Supplementary Methods 3.3.3 defines it.

Four choices the Methods state and the article body omits, plus the two subsets
they operate on. They are collected here because they are *protocol*, not model
and not script-local: `scripts/08` scores a cached run against them and
`scripts/16` scores two live implementations, and both must use the same
definitions or they are measuring different tasks.

    axis      one PCC per test CONDITION, against the condition-AVERAGED truth
    scale     per-GENE min-max standardization of the absolute values, eq. (1)
    subset    exponential-phase profiles only -- 2,610 profiles / 493 conditions
    testable  of those 493, the 262 whose four attributes each recur elsewhere
    wild type MG1655 in LB or M9, any carbon source, no stress, no perturbation

Getting the axis wrong is what made the transcriptome read "ours 0.287 vs the
paper's 0.54" for most of this reproduction; `ecomics/metrics.py` documents that
trap at length. Getting the SCALE wrong repeats it more quietly: correlating raw
values instead of min-max ones lets the mean-expression profile dominate every
row, and a parameter-free mean-profile baseline then measures 0.638 where the
paper reports 0.26.

Before this module existed, `scripts/16` reached into `scripts/08` with
`importlib.import_module("08_methods_faithful_eval")` after mutating `sys.path`
-- the only cross-script import in the repository, and fragile for two reasons:
a module name starting with a digit cannot be imported normally, and
`tools/verify_docs.py:resolve_source` resolves doc references by BASENAME, so a
package module sharing a name with a numbered script would silently break
references in documents nobody touched.
"""

from __future__ import annotations

import json

import re

import numpy as np

from ecomics import config as C
from ecomics.db.api import Ecomics
from ecomics.metrics import pcc_per_row

__all__ = ["PAPER", "BASELINE_KINDS", "tf_indices", "lb_or_m9_media",
           "growth_phase", "testable_conditions", "minmax_by_gene",
           "score_by_condition"]

# What the paper reports, for the protocol comparison. Distinct from
# `config.PAPER`, which carries the compendium's shape rather than its scores.
PAPER = {"moma": 0.54, "random": 0.25, "mean": 0.26, "wildtype": 0.36,
         "tf_moma": 0.68, "tf_random": 0.41, "tf_mean": 0.41, "tf_wildtype": 0.40,
         "n_profiles": 2610, "n_conditions": 493, "n_testable": 262}

BASELINE_KINDS = ("random", "mean", "wildtype")

_BNUM = re.compile(r"b\d{4}")


def tf_indices(columns: list[str]) -> np.ndarray:
    """Column indices of the PAPER'S transcription factors, for the TF subset.

    The regulators of Supplementary Data 2's TRN sheet: **179 TFs, keyed by
    b-number**, of which **176** are among the 4,096 genes. That is exactly the
    count Fig. 5b's boxplot is labelled with -- `TFs (176)` -- so this is the
    paper's own set rather than a reconstruction of it.

    ⚠ The alternative is a reconstruction: RegulonDB regulator NAMES from
    the scraped TRN, mapped through `gene_symbol_map`, giving **200**. That file
    was deleted with the rest of the scraped graph path, and because this
    function reached it through `REMOTE_FILES.get(...)` it degraded to an empty
    array rather than failing -- silently dropping the TF row from every caller.
    The lesson is in the shape, not the incident: a `.get()` fallback written to
    tolerate a MISSING FILE also tolerates a missing KEY, and the second is a
    code change rather than an environment one. Raising here would have caught
    it; instead `scripts/08` printing the count is what has to.

    Still returns an empty array when Data 2 is absent, so a caller degrades to
    all-genes scoring rather than failing -- but the file is now one the repo
    downloads for five other purposes, not an optional extra.
    """
    path = C.SUPPLEMENTARY["interactions"]
    if not path.exists():
        return np.array([], dtype=int)
    import openpyxl

    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        tfb = {str(r[0]).strip() for r in wb["TRN"].iter_rows(values_only=True)
               if r and r[0] and _BNUM.fullmatch(str(r[0]).strip())}
    finally:
        wb.close()
    return np.array([i for i, c in enumerate(columns) if c in tfb], dtype=int)


def lb_or_m9_media() -> set[str]:
    """Medium IDs whose base medium is LB or M9, any carbon source.

    Half of the paper's wild-type definition. Read from the scraped medium
    ontology rather than the database because it is a property of the ontology,
    not of any build.
    """
    path = C.PROK_DIR / "medium.json"
    out = set()
    for m in json.loads(path.read_text(encoding="utf-8")):
        for field in ("Base Medium", "Description"):
            if (m.get(field) or "").strip().upper().startswith(("LB", "M9")):
                out.add(m["ID"])
                break
    return out


def growth_phase(db: Ecomics, profile_ids: np.ndarray) -> np.ndarray:
    """Per-profile exponential-phase mask, from Supplementary Data 1.

    The released expression table carries no growth phase, so this is the one
    element of the paper's protocol that CANNOT be reconstructed from the public
    data alone. Applying it reproduces the paper's 2,610 profiles and 493
    conditions exactly.
    """
    import pandas as pd

    path = C.SUPPLEMENTARY["metadata"]
    if not path.exists():
        raise SystemExit(f"missing {path}\nRun: python scripts/00_acquire.py")
    src = {r["id"]: r["source_id"] for r in db.conn.execute(
        "SELECT id, source_id FROM profile WHERE layer='transcriptome'")}
    meta = pd.read_excel(path, sheet_name="Transcriptome")
    phase = dict(zip(meta["ID"].astype(str), meta["Growth Phase"].astype(str)))
    return np.array([phase.get(src.get(int(p), ""), "").lower()
                     .startswith(("exp", "mid-exp")) for p in profile_ids])


def testable_conditions(keys: np.ndarray) -> np.ndarray:
    """The paper's "testable" conditions, as a mask over `keys`.

    A condition is testable when each of its FOUR ATTRIBUTES -- strain, medium,
    stress, genetic perturbation -- also appears in another condition. Hold it
    out and the model has still seen every attribute value it needs.

    On the paper's own 493-condition subset this returns **262, exactly the
    paper's number**, and it also discriminates: the rival readings give 330
    (per feature) and ~all (the epochs paragraph's literal "one or more
    features"). So it identifies which criterion the authors computed, from a
    sentence that does not say. See `docs/ecomics/paper-protocol.md`.

    Reproduces the paper's 262 testable conditions exactly.
    """
    import collections

    parts = [str(k).split(".") for k in keys]
    if any(len(p) != 4 for p in parts):
        raise ValueError("condition keys must be strain.medium.stress.gp")
    counts = [collections.Counter(p[i] for p in parts) for i in range(4)]
    return np.array([all(counts[i][p[i]] > 1 for i in range(4)) for p in parts])


def minmax_by_gene(truth: np.ndarray, *arrays: np.ndarray):
    """Eq. (1), fitted on `truth` and applied to it and every array given.

    A gene constant across `truth` has range 0; it maps to a constant 0 rather
    than producing inf/nan.
    """
    lo = np.nanmin(truth, axis=0, keepdims=True)
    hi = np.nanmax(truth, axis=0, keepdims=True)
    rg = np.where(hi - lo > 0, hi - lo, 1.0)
    return tuple((a - lo) / rg for a in (truth, *arrays))


def score_by_condition(pred: np.ndarray, truth: np.ndarray, keys: np.ndarray,
                       tf_idx: np.ndarray | None = None) -> dict:
    """The paper's axis: one PCC per condition, against condition-averaged truth.

    "PCC between predicted expression levels and average of known expression
    levels for profiles belonging to the test condition" -- not per gene across
    conditions.

    Scored on the per-gene MIN-MAX scale, because eq. (1) standardizes before
    anything else. Omitting that is not cosmetic: the mean-expression profile
    then dominates every row and the mean-profile baseline reads 0.638 against
    the paper's 0.26.
    """
    truth_s, pred_s = minmax_by_gene(truth, pred)
    uniq, inv = np.unique(keys, return_inverse=True)
    tc = np.vstack([truth_s[inv == i].mean(axis=0) for i in range(len(uniq))])
    pc = np.vstack([pred_s[inv == i].mean(axis=0) for i in range(len(uniq))])
    r = pcc_per_row(pc, tc)
    out = {"pcc": float(np.nanmean(r)), "sd": float(np.nanstd(r)),
           "n_conditions": int(len(uniq))}
    if tf_idx is not None and len(tf_idx):
        rt = pcc_per_row(pc[:, tf_idx], tc[:, tf_idx])
        out["tf_pcc"] = float(np.nanmean(rt))
        out["tf_sd"] = float(np.nanstd(rt))
    return out

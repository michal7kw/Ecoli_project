"""The condition feature encoder -- MOMA's input vector.

The paper encodes an experimental condition as 612 features (paper.md:128):

    154  genetic       genotype markers of the K-12 strains
    120  medium        chemical composition
     52  stress        experimental settings
    286  perturbation  the specific knock-outs / over-expressions
    ---
    612

Reconstructed here from the scraped meta-data ontology. These are the block
sizes the encoder ACTUALLY emits:

    152  strain features    strain.v5.json has 156 columns, 4 of which are
                            descriptive (Strain Name, Link, PMID, Alternate
                            Names). The paper says 154, and its own ontology
                            does not support that -- see `FEATURE_COUNT_NOTE`.
    120  medium features    medium.v5.json has 126 columns, 6 descriptive
                            (ID, Base Medium, Description, Link, PMID,
                            Defined), leaving 120 components -- matching the
                            paper exactly. One binary presence feature each
                            (`medium_kind="present"`), which is the encoding
                            Supplementary Methods 3.3.2 illustrates.
     58  stress multi-hot   ATOMIC stressors, obtained by splitting the field
                            on ';' (`canon.split_stress`). Every one appears in
                            Supplementary Data 1's Stress sheet, which lists 62;
                            the 4 missing are ones the compendium never
                            observes. The paper's prose says 52.
    273  perturbation       distinct (gene, type) perturbations observed. This
                            read 296 before the fluxome's
                            genotypes -- written as upper-case gene SYMBOLS
                            where every other layer writes b-numbers -- were
                            normalized in `db/canon.py`, and 23 of the columns
                            turned out to be the same perturbation twice.
    ---
    603

Every remaining gap to the paper's 612 is now a COUNTING difference -- observed
values against the paper's own categorization -- not a design difference:
-2 strain (152 vs 154, and ours is the derivable one), +6 stress (58 vs 52),
-13 perturbation (273 vs 286), medium matching exactly.

⚠ Do not hard-code this block anywhere. It has been wrong in a generated file
once already: `scripts/04_reproduce.py` printed "626 features ... 296
perturbation" into `results/reproduction_table.md` as a string literal, and it
stayed there through the 296 -> 273 correction because nothing recomputes a
sentence. That script now derives the line from `build_encoder`.

History, because both previous values are quoted in older prose. This block
once read "120 medium / 69 stress" and summed to 637, matching nothing. It then
read 240 medium / 68 stress and summed to **756**, which is the number in
any results file written under the earlier encoding. 756 came from two
choices since reverted:
`medium_kind="both"` (presence AND amount, doubling the medium block) and
one-hot over ';'-joined stress combinations. See `fit` for why each changed.

Why decompose at all
--------------------
This is the design decision that licenses the paper's claim of covering ~10^8
unseen conditions. A one-hot over 65 strain NAMES can only ever recall strains
seen in training -- every column is zero for a novel strain. A genotype VECTOR
shares structure: a new strain that matches 150 of 152 markers lights up 150
features the model already has weights for. Same argument for encoding media by
their 120 chemical components rather than by name.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ecomics.db.api import Ecomics
from ecomics.db.canon import (ConditionKey, parse_transcriptome_cond, split_gp,
                              split_stress)

FEATURE_COUNT_NOTE = (
    "strain.v5.json yields 152 genotype features; the paper reports 154. "
    "RESOLVED, and ours is the correct count: Supplementary Data 1's Strain "
    "sheet has 160 columns, of which b0243, b4346 and b4348 each appear TWICE "
    "-- 157 distinct names, minus 5 metadata columns (Strain Name, Source, "
    "PMID, Alternate Names, Comments) = exactly 152 distinct genotype "
    "features. That set is identical to ours: `ours - theirs` and "
    "`theirs - ours` are both empty. The paper's 154 is not derivable from its "
    "own published ontology. Medium components match exactly at 120."
)

# Conditions that encode identically cannot be told apart by any model, and do
# not fully separate under leave-one-condition-out. Measured
# against a built database:
#
#                             transcriptome (596)      every layer (674)
#   medium_kind='present'     19 in  9 groups (3.2%)   23 in 11 groups (3.4%)
#   medium_kind='both'        11 in  5 groups (1.8%)   11 in  5 groups (1.6%)
#
# ALWAYS SAY WHICH CONDITION SET. This block once read
# "23 of 596 transcriptome conditions (3.9%) in 11 groups" -- the count from the
# all-layer row against the denominator from the transcriptome row, and a
# percentage matching neither. It propagated verbatim into six documents. The
# numbers were each individually real, which is exactly why nobody caught the
# pairing; a collision count is meaningless without the set it was counted over.
#
# The 'both' row is the trade the paper's 120-wide medium block costs: 120 extra
# features separate four media pairs differing only in concentration. Reverting
# to the paper's width gives those four back. Taken deliberately -- see `fit`.
#
# This block also once claimed every group was a "limit of the PUBLISHED
# ontology, not an encoder bug". Only one of the four causes is.
#
# 1. STRAIN BINARIZATION (encoder) -- the ontology records the difference:
#
#   rpoA14 == rpoA27 == rpoD3   RNA-polymerase point mutants differing in
#                               exactly one recorded cell: `Plasmid` reads
#                               pHACM-rpoA14 / -rpoA27 / -rpoD3. `db/build.py`
#                               reduces every strain cell to `value not in
#                               ABSENT`, so all three become 1. The 204 distinct
#                               allele strings in strain_feature.value are never
#                               read. (Same mechanism costs Lambda+ vs Lambda-,
#                               and Sex=F- -- absence of the F plasmid -- which
#                               all encode as presence.)
#   KSL2000 == KSL2009          NOT identical across the 152 markers, as this
#                               block used to say: `Plasmid` reads pBAD-RNA vs
#                               pBAD-NRNA. Same binarization, same loss.
#
# 2. CONCENTRATION ONLY (the documented cost of medium_kind="present"):
#
#   MD001 == MD106              M9+Glu, 0.4% vs 0.5% (the raw '40%' is a
#                               data-entry error; Supplementary Data 1 has
#                               0.004 for it)
#   MD004 == MD067              MOPS+Glu, 0.4% vs 0.1%
#   MD027 == MD116              LB+Gly, 0.4% vs 10%
#   MD042 == MD073              EZ vs MOPS+Glu+KH2PO4; Glu 20% vs 0.1%
#
#   These are real, and `medium_kind="both"` does separate them. It is not
#   worth 120 features and a doubled block the paper is explicit about; the
#   amount half is not even on a consistent scale (see `fit`).
#
# 3. '?' READ AS ABSENT (parser semantics -- probably a genuine BUG):
#
#   MD010 == MD022              modified MOPS+Glu vs MOPS+CA -- 20 amino acids
#   MD090 == MD091              M9+T vs M9+Gal -- Gal
#   MD018 == MD094              LB vs LB+Fru -- Fru
#
#   Each differs only in cells recorded '0' against '?', and parse_amount maps
#   BOTH to present=0. For these three pairs that is wrong: the medium NAMES
#   say the component is there (MD094 is 'LB+Fru', MD091 is 'M9+Gal', MD022 is
#   'MOPS+CA'), and Supplementary Methods 3.1.3 states the convention for the
#   last outright -- "for media with casamino acids we assume that all 20 amino
#   acids are present". So `LB+Fru` is currently encoded as plain `LB`.
#
#   ⚠ But do NOT "fix" this by mapping every '?' to present. Checked over all
#   194 '?' cells, in 15 media, the token means different things:
#
#       125  rich/complex media (dYT, TB, LB+, MOPS+CA, M63+CA, EZ) -- the
#            20-amino-acid or nucleobase block IS there, unquantified   PRESENT
#        63  DEFINED MINIMAL media (MD046 'minimal salt', MD043/MD079 'K') --
#            glucose plus salts and nothing else; amino acids are ABSENT
#         5  the component is named in the medium's own description        PRESENT
#         1  MD100 'W2' MgSO4                                             unclear
#
#   A blanket flip would corrupt the 63. '?' genuinely means UNRECORDED, and
#   resolving it needs a per-medium rule keyed on whether the base medium is
#   rich or defined-minimal -- a judgement this repo has not made. Left as-is,
#   with the three wrong collisions documented rather than half-fixed. Tracked
#   as a known limit of the published ontology.
#
#   `medium_kind="both"` never separated these either: `numeric()` imputes the
#   NULL onto the component median, and 116 of the 120 medians are 0.0.
#
# 4. ONTOLOGY limit -- the published data genuinely cannot separate these:
#
#   LJ110 == W3110              identical in all 152 raw cells, not merely in
#                               their encoded form. Nothing downstream can fix
#                               this without new annotation.
#
# The distinction matters because it implies opposite responses: an ontology
# limit is a documented limit, an encoder limit is closable, and a concentration
# collision is a priced trade-off.
#
# ConditionEncoder.collisions() recomputes this for any condition list.
KNOWN_COLLISION_CAUSES = {
    "strain binarization": [("rpoA14", "rpoA27", "rpoD3"), ("KSL2000", "KSL2009")],
    "concentration only": [("MD001", "MD106"), ("MD004", "MD067"),
                           ("MD027", "MD116"), ("MD042", "MD073")],
    "question mark read as absent": [("MD010", "MD022"), ("MD090", "MD091"),
                                     ("MD018", "MD094")],
    "identical genotypes": [("LJ110", "W3110")],
}

BLOCKS = ("strain", "medium", "stress", "perturbation")


@dataclass
class ConditionEncoder:
    """Encodes a condition key into MOMA's input vector.

    Fitted against the built database so the feature space is fixed and
    reproducible; `transform` then works for any condition, including ones
    never observed, as long as its parts were seen.
    """

    strain_features: list[str] = field(default_factory=list)
    medium_features: list[str] = field(default_factory=list)
    stress_features: list[str] = field(default_factory=list)
    pert_features: list[str] = field(default_factory=list)
    medium_kind: str = "present"          # 'present' | 'amount' | 'both'

    _strain_tbl: pd.DataFrame | None = None
    _medium_tbl: pd.DataFrame | None = None

    # ------------------------------------------------------------------ fit
    @classmethod
    def fit(cls, db: Ecomics, medium_kind: str = "present") -> "ConditionEncoder":
        """Fit the feature space against the built compendium.

        medium_kind:
          'present'  120 binary features, THE DEFAULT. One per component, which
                     is the width Supplementary Methods 3.3.2 specifies and the
                     encoding its only worked example shows ("x_122 = {0, 1}
                     represents K2HPO4, 0 and 1 denotes absence and presence").
          'amount'   120 numeric features. Available, but do not reach for it
                     without reading the unit warning below.
          'both'     240 features, presence AND amount. The former default; it
                     is what made this encoder 756 wide.

        Why 'both' is no longer the default
        -----------------------------------
        It bought the fewest collisions -- 11 groups under 'present', 7 under
        'amount', fewest under 'both' -- and colliding conditions leak across
        leave-one-condition-out folds, since holding one out leaves an
        identical input in the training set. That measurement stands.

        What did not stand is the reason to pay 120 features for it:

        1. The example this docstring used to give was wrong. It claimed
           'present' cannot separate M9+Glu(0.4%) from M9+Glu(40%). In
           Supplementary Data 1 both are 0.004 -- the '40%' in the scraped
           table is a data-entry error for 0.40% (40% w/v glucose is ~400 g/L).
           Those media do not differ in glucose, so nothing was being lost.
        2. The 'amount' half is not a concentration scale. Units are parsed,
           stored in medium_component.unit, and then never read: 72 of the 120
           components appear in more than one unit and 88% of quantified cells
           sit in such a component, so '400 mM' and '400 uM' both enter the
           model as the float 400.0. Adding that block adds noise with an
           arbitrary scale, not information.
        3. It doubled the block the paper is most explicit about, making every
           paper-vs-ours shape comparison need a caveat.

        The collisions 'present' reintroduces are real and are NOT swept up
        here: see KNOWN_COLLISION_CAUSES above, `collisions()`, and note that
        the three MDxxx media pairs collide under 'both' as well, because
        `numeric()` imputes '?' onto the component median and 116 of the 120
        medians are 0.0. 'both' never separated those either.
        """
        strain_tbl = db.strain_features()

        def numeric(tbl: pd.DataFrame) -> pd.DataFrame:
            # Unknown concentrations are NaN; a model cannot consume NaN, and 0
            # would assert "absent", a different claim. Impute with the
            # component's median over media that do report it.
            #
            # ⚠ That rationale does not survive contact with the data: the
            # matrix is 87% zeros (11,728 of 13,440 cells), so 116 of the 120
            # component medians ARE 0.0, and '?' is imputed to exactly the
            # value this is written to avoid. It is why the three MDxxx media
            # pairs in KNOWN_COLLISION_CAUSES collide under 'both' too, not
            # only under the 'present' default.
            #
            # This is now OFF the default path -- medium_kind defaults to
            # 'present', which never calls this. Left as-is rather than fixed
            # because the fix is an explicit missingness indicator, not a
            # different fill value, and because '?' should arguably not be
            # missing at all (see cause 3 in KNOWN_COLLISION_CAUSES).
            return tbl.fillna(tbl.median()).fillna(0.0)

        if medium_kind == "present":
            medium_tbl = db.medium_components("present")
        elif medium_kind == "amount":
            medium_tbl = numeric(db.medium_components("amount"))
        elif medium_kind == "both":
            pres = db.medium_components("present").add_suffix(".present")
            amt = numeric(db.medium_components("amount")).add_suffix(".amount")
            medium_tbl = pd.concat([pres, amt], axis=1)
        else:
            raise ValueError(f"medium_kind must be present/amount/both, "
                             f"got {medium_kind!r}")

        enc = cls(
            strain_features=list(strain_tbl.columns),
            medium_features=list(medium_tbl.columns),
            # ATOMIC stressors, not raw field values: the field is a set, and
            # 12 of the observed values are ';'-joined combinations. See
            # canon.split_stress for why one-hot over combinations is wrong.
            stress_features=sorted({a for s in db.stresses()
                                    for a in split_stress(s)}),
            pert_features=list(db.perturbations()),
            medium_kind=medium_kind,
        )
        enc._strain_tbl = strain_tbl
        enc._medium_tbl = medium_tbl
        return enc

    # ------------------------------------------------------------- geometry
    @property
    def block_sizes(self) -> dict[str, int]:
        return {
            "strain": len(self.strain_features),
            "medium": len(self.medium_features),
            "stress": len(self.stress_features),
            "perturbation": len(self.pert_features),
        }

    @property
    def n_features(self) -> int:
        return sum(self.block_sizes.values())

    @property
    def feature_names(self) -> list[str]:
        return ([f"strain:{f}" for f in self.strain_features]
                + [f"medium:{f}" for f in self.medium_features]
                + [f"stress:{f}" for f in self.stress_features]
                + [f"gp:{f}" for f in self.pert_features])

    def block_slice(self, block: str) -> slice:
        start = 0
        for b in BLOCKS:
            n = self.block_sizes[b]
            if b == block:
                return slice(start, start + n)
            start += n
        raise KeyError(block)

    # ------------------------------------------------------------ transform
    def transform_one(self, key: ConditionKey | str) -> np.ndarray:
        """Encode one condition. Unknown parts contribute zeros, not errors.

        A condition naming an unseen strain or medium is still encodable -- it
        simply carries no information in that block. That is the honest
        behaviour: the model then predicts from the parts it does recognise,
        which is exactly the compositional-generalization setting.
        """
        if isinstance(key, str):
            key = parse_transcriptome_cond(key)

        x = np.zeros(self.n_features, dtype=np.float32)
        off = 0

        # strain genotype
        n = len(self.strain_features)
        if self._strain_tbl is not None and key.strain in self._strain_tbl.index:
            x[off:off + n] = self._strain_tbl.loc[key.strain].to_numpy(np.float32)
        off += n

        # medium composition
        n = len(self.medium_features)
        if self._medium_tbl is not None and key.medium_id in self._medium_tbl.index:
            x[off:off + n] = self._medium_tbl.loc[key.medium_id].to_numpy(np.float32)
        off += n

        # stress multi-hot ('none' splits to [], so no-stress is all zeros).
        # Multi-hot, not one-hot: an unseen COMBINATION of two known stresses
        # must light up both columns, not fall off the vocabulary as an
        # unknown category and encode as no stress at all.
        n = len(self.stress_features)
        for s in split_stress(key.stress):
            j = self._stress_index.get(s)
            if j is not None:
                x[off + j] = 1.0
        off += n

        # perturbation multi-hot
        for p in split_gp(key.gp):
            j = self._pert_index.get(p.as_str())
            if j is not None:
                x[off + j] = 1.0
        return x

    def transform(self, keys) -> np.ndarray:
        """Encode many conditions -> (n, n_features)."""
        return np.vstack([self.transform_one(k) for k in keys]).astype(np.float32)

    # --------------------------------------------------------------- lookup
    @property
    def _stress_index(self) -> dict[str, int]:
        if not hasattr(self, "_si"):
            object.__setattr__(self, "_si",
                               {s: i for i, s in enumerate(self.stress_features)})
        return self._si

    @property
    def _pert_index(self) -> dict[str, int]:
        if not hasattr(self, "_pi"):
            object.__setattr__(self, "_pi",
                               {p: i for i, p in enumerate(self.pert_features)})
        return self._pi

    # -------------------------------------------------------------- reports
    def collisions(self, keys) -> list[list[str]]:
        """Groups of conditions that encode to the SAME vector.

        Worth checking whenever the encoding changes: colliding conditions are
        indistinguishable to any model, and they leak across LOCO folds because
        holding one out still leaves an identical input in the training set.
        """
        import collections as _c

        groups: dict[bytes, list[str]] = _c.defaultdict(list)
        for k in keys:
            groups[self.transform_one(k).tobytes()].append(
                k if isinstance(k, str) else k.as_str())
        return [g for g in groups.values() if len(g) > 1]

    def describe(self) -> str:
        b = self.block_sizes
        lines = [
            f"condition encoder (medium_kind={self.medium_kind!r})",
            f"  strain        {b['strain']:>4d}   (paper: 154)",
            f"  medium        {b['medium']:>4d}   (paper: 120)",
            f"  stress        {b['stress']:>4d}   (paper:  52)",
            f"  perturbation  {b['perturbation']:>4d}   (paper: 286)",
            f"  {'total':<13s} {self.n_features:>4d}   (paper: 612)",
            "",
            "  Counts differ from the paper because ours are OBSERVED values "
            "(58 atomic",
            "  stressors; 296 distinct perturbations appear in the published "
            "files, which",
            "  become 273 columns once the fluxome's gene SYMBOLS are "
            "normalized to",
            "  b-numbers and 23 prove to be duplicates) while the paper reports "
            "its own",
            "  categorization (52 stress categories, 286 perturbations).",
            "  All 58 appear in Supplementary Data 1's Stress sheet, which "
            "lists 62; the 4",
            "  it names that we never observe make up the difference. Every "
            "remaining gap",
            "  to 612 is a counting difference, not a design difference.",
            f"  {FEATURE_COUNT_NOTE}",
        ]
        return "\n".join(lines)


def build_encoder(db: Ecomics | None = None, **kw) -> ConditionEncoder:
    own = db is None
    db = db or Ecomics()
    try:
        return ConditionEncoder.fit(db, **kw)
    finally:
        if own:
            db.close()

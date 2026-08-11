"""The condition feature encoder -- MOMA's input vector.

The paper encodes an experimental condition as 612 features (paper.md:128):

    154  genetic       genotype markers of the K-12 strains
    120  medium        chemical composition
     52  stress        experimental settings
    286  perturbation  the specific knock-outs / over-expressions
    ---
    612

Reconstructed here from the scraped meta-data ontology. These are the block
sizes the encoder ACTUALLY emits, and they sum to the 756 quoted elsewhere:

    152  strain features    strain.v5.json has 156 columns, 4 of which are
                            descriptive (Strain Name, Link, PMID, Alternate
                            Names). The paper says 154, and its own ontology
                            does not support that -- see `FEATURE_COUNT_NOTE`.
    240  medium features    medium.v5.json has 126 columns, 6 descriptive
                            (ID, Base Medium, Description, Link, PMID,
                            Defined), leaving 120 components -- matching the
                            paper exactly. The encoder emits each TWICE, as
                            presence and as amount (`medium_kind="both"`), so
                            the block is 240. Presence alone collapses 11 groups
                            of conditions into ties that leak across CV folds.
     68  stress one-hots    distinct stresses actually present in the built
                            compendium (the paper's 52 counts its own
                            categorization; ours counts observed values)
    296  perturbation       distinct (gene, type) perturbations observed
    ---
    756

(This block previously read "120 medium / 69 stress" and summed to 637, which
matched neither the encoder nor the 756 stated in `README.md`,
`docs/11-reproduction.md` and `DISCREPANCIES.md`. Verified against a built
database: 152 + 240 + 68 + 296.)

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
from ecomics.db.canon import ConditionKey, parse_transcriptome_cond, split_gp

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

# With the default encoding, 15 of 596 transcriptome conditions (2.5%) fall into
# 7 groups that encode identically. These are limits of the PUBLISHED ontology,
# not encoder bugs, and they are worth knowing because such conditions cannot be
# told apart by any model and do not fully separate under leave-one-condition-out:
#
#   rpoA14 == rpoA27 == rpoD3   RNA-polymerase POINT MUTANTS. The strain
#                               ontology records gene presence/absence, so it
#                               has no column that can distinguish alleles of
#                               the same gene.
#   KSL2000 == KSL2009          identical across all 152 markers
#   LJ110   == W3110            identical across all 152 markers
#   MD010 == MD022, MD090 == MD091, MD018 == MD094
#                               media indistinguishable in the 120 published
#                               components (they differ in something unrecorded)
#
# ConditionEncoder.collisions() recomputes this for any condition list.
KNOWN_COLLISION_CAUSES = {
    "point mutants": ["rpoA14", "rpoA27", "rpoD3"],
    "identical genotypes": [("KSL2000", "KSL2009"), ("LJ110", "W3110")],
    "identical media": [("MD010", "MD022"), ("MD090", "MD091"), ("MD018", "MD094")],
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
    medium_kind: str = "both"             # 'present' | 'amount' | 'both'

    _strain_tbl: pd.DataFrame | None = None
    _medium_tbl: pd.DataFrame | None = None

    # ------------------------------------------------------------------ fit
    @classmethod
    def fit(cls, db: Ecomics, medium_kind: str = "both") -> "ConditionEncoder":
        """Fit the feature space against the built compendium.

        medium_kind:
          'present'  120 binary features -- faithful to the paper's count, but
                     it discards concentration, so M9+Glu(0.4%) and M9+Glu(40%)
                     encode IDENTICALLY. That collapses 11 groups of distinct
                     conditions onto the same input vector.
          'amount'   120 numeric features -- separates concentrations but loses
                     the non-numeric readings ('yes', '?'). 7 collision groups.
          'both'     240 features, presence AND amount. Fewest collisions, and
                     the default: two conditions sharing an input vector cannot
                     be told apart by any model, and worse, they leak across
                     leave-one-condition-out folds.
        """
        strain_tbl = db.strain_features()

        def numeric(tbl: pd.DataFrame) -> pd.DataFrame:
            # Unknown concentrations are NaN; a model cannot consume NaN, and 0
            # would assert "absent", a different claim. Impute with the
            # component's median over media that do report it.
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
            stress_features=[s for s in db.stresses() if s != "none"],
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

        # stress one-hot ('none' is the absence of a stress, so has no column)
        n = len(self.stress_features)
        if key.stress in self._stress_index:
            x[off + self._stress_index[key.stress]] = 1.0
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
            "(68 distinct",
            "  stresses, 296 distinct perturbations in the published files) "
            "while the paper",
            "  reports its own categorization (52 stress categories, 286 "
            "perturbations).",
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

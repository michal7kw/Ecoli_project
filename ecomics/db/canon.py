"""Canonicalization of Ecomics condition keys.

THE most load-bearing module in this package. A condition in Ecomics is the
4-tuple (strain, medium ID, stress, genetic perturbations), but the published
files write that tuple in THREE mutually incompatible dialects:

    transcriptome (v8 TSV)   Cond = "MG1655.MD018.none.na_WT"
                                    "MG1655.MD018.Nx.b0624_OE"
                                    "MG1655.MD102.lactose-shift.b2741_KO"
    proteome / metabolome    GP column = "none"
                                         "b0756(KO)"
                                         "b0008(KO);b0688(KO)"
    fluxome (scraped)        GP column = "WT_na"          <- note: REVERSED
                                         "b1852(KO)"

Join on the raw strings and every cross-layer query returns ZERO rows. That is
not hypothetical -- it is what a naive join produces, and it silently yields an
empty training set for every cross-layer module in MOMA.

Canonicalizing all three dialects onto one form recovers exactly the overlaps
the paper reports:

    transcriptome & proteome     5 conditions   (paper: "18 profiles (5 conditions)")
    transcriptome & phenome    179 conditions   (paper: "1992 profiles ... 179 conditions")

Canonical form
--------------
Wild type is the empty perturbation, written "none". Otherwise a
semicolon-joined, lexicographically SORTED list of "GENE(TYPE)" terms, so that
"b0008(KO);b0688(KO)" and "b0688(KO);b0008(KO)" are the same condition.

Perturbation types seen in the compendium:
    KO   knock-out          OE   over-expression      IN   insertion
    VAR  variant/mutation   RM   removal              EX   other/experimental
    WT   wild type (the sentinel, normalized away)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Mapping, NamedTuple

# Tokens that all mean "no genetic perturbation".
WILDTYPE_TOKENS = frozenset({
    "", "none", "na", "wt", "na_wt", "wt_na", "nan", "null", "-",
})

# Known perturbation type codes, upper-cased.
GP_TYPES = frozenset({"KO", "OE", "IN", "VAR", "RM", "EX", "WT"})

NO_GP = "none"

# "b0756(KO)" / "mevalonate-pw(IN)"
_PAREN_RE = re.compile(r"^\s*(?P<gene>.+?)\s*\(\s*(?P<type>[A-Za-z]+)\s*\)\s*$")
# "b0624_OE" / "na_WT" / "WT_na"
_USCORE_RE = re.compile(r"^\s*(?P<a>.+?)_(?P<b>[A-Za-z]+)\s*$")


class ConditionKey(NamedTuple):
    """A canonical Ecomics condition."""

    strain: str
    medium_id: str
    stress: str
    gp: str  # canonical, sorted, ';'-joined, or "none"

    def as_str(self) -> str:
        return f"{self.strain}.{self.medium_id}.{self.stress}.{self.gp}"

    @property
    def is_wildtype(self) -> bool:
        return self.gp == NO_GP


@dataclass(frozen=True)
class Perturbation:
    """One genetic perturbation: a target and what was done to it."""

    gene: str      # b-number where available, otherwise a descriptive token
    type: str      # KO / OE / IN / VAR / RM / EX

    def as_str(self) -> str:
        return f"{self.gene}({self.type})"


def normalize_gene(gene: str, gene_map: Mapping[str, str] | None) -> str:
    """Map a gene SYMBOL to its b-number. Identity when no map is supplied.

    Why this exists, and why the map is INJECTED rather than imported here.

    The fluxome layer writes perturbations as upper-case gene symbols
    (`TALB(KO)`) while every other layer writes b-numbers (`b0008(KO)`). Those
    are the same knockout. Left unreconciled they become two feature columns,
    and -- worse -- the fluxome cannot join to the proteome, metabolome or
    phenome on ANY perturbed condition: measured, those overlaps were 1, 2 and
    3, against 22, 23 and 25 once normalized. `config.EXPECTED_OVERLAP` did not
    catch it because it asserts only the five transcriptome-anchored pairs.

    The map is a parameter, not an import, because `networks.gene_symbol_map()`
    returns an EMPTY DICT when its KEGG file is absent. Imported implicitly,
    that would make condition keys -- the join surface for everything -- depend
    on whether an untracked file happens to exist, which is a worse bug than
    the one being fixed. `db/build.py` loads it once and asserts it is
    non-empty; every other caller gets identity and is unaffected.
    """
    if not gene_map:
        return gene
    return gene_map.get(gene.lower(), gene)


def _parse_one(token: str,
               gene_map: Mapping[str, str] | None = None) -> Perturbation | None:
    """Parse a single perturbation token in any of the three dialects.

    Returns None for wild-type sentinels, which are dropped rather than
    represented -- "wild type" is the ABSENCE of perturbations, not a
    perturbation named WT. Conflating the two is what makes `na_WT` fail to
    match `none`.
    """
    tok = token.strip()
    if tok.lower() in WILDTYPE_TOKENS:
        return None

    m = _PAREN_RE.match(tok)
    if m:
        gene, typ = m.group("gene").strip(), m.group("type").strip().upper()
    else:
        m = _USCORE_RE.match(tok)
        if not m:
            # No type suffix at all; treat the whole token as a gene with an
            # unknown perturbation type rather than silently dropping it.
            return Perturbation(gene=tok, type="EX")
        a, b = m.group("a").strip(), m.group("b").strip().upper()
        # Handle both "b0624_OE" (gene_TYPE) and the fluxome's "WT_na"
        # (TYPE_gene), by checking which half looks like a type code.
        if b in GP_TYPES:
            gene, typ = a, b
        elif a.upper() in GP_TYPES:
            gene, typ = b, a.upper()
        else:
            gene, typ = a, b

    if typ == "WT" or gene.lower() in WILDTYPE_TOKENS:
        return None
    return Perturbation(gene=normalize_gene(gene, gene_map), type=typ)


def split_gp(raw: str | None,
             gene_map: Mapping[str, str] | None = None) -> list[Perturbation]:
    """Parse a raw GP field into a sorted list of perturbations.

    `gene_map` maps lower-case gene symbols to b-numbers; see `normalize_gene`.
    """
    if raw is None:
        return []
    perts = [p for tok in str(raw).split(";") if (p := _parse_one(tok, gene_map))]
    # Sort so that permutations of the same multi-KO condition collapse.
    return sorted(set(perts), key=lambda p: (p.gene, p.type))


def canonical_gp(raw: str | None,
                 gene_map: Mapping[str, str] | None = None) -> str:
    """Canonicalize any GP dialect to the shared form.

    >>> canonical_gp("na_WT"), canonical_gp("WT_na"), canonical_gp("none")
    ('none', 'none', 'none')
    >>> canonical_gp("b0624_OE")
    'b0624(OE)'
    >>> canonical_gp("b0688(KO);b0008(KO)")
    'b0008(KO);b0688(KO)'
    """
    perts = split_gp(raw, gene_map)
    return ";".join(p.as_str() for p in perts) if perts else NO_GP


def canonical_stress(raw: str | None) -> str:
    """Normalize the stress field. 'none' is the no-stress sentinel."""
    s = (raw or "").strip()
    return "none" if not s or s.lower() in {"none", "na", "nan", "null", "-"} else s


def split_stress(raw: str | None) -> list[str]:
    """Atomic stressors in a stress field, sorted. ';' joins co-applied ones.

    >>> split_stress("heat;osmotic")
    ['heat', 'osmotic']
    >>> split_stress("none")
    []

    The mirror of `split_gp`, and it exists for the same reason: the field is a
    SET of stressors, not a category. 12 of the 68 distinct stress strings in
    the compendium are ';'-joined, and treating each as its own one-hot column
    is what made the encoder's stress block 68 wide against the paper's 52.
    Splitting gives 58 atoms, every one of which appears in Supplementary Data
    1's Stress sheet; the 4 it lists that we never observe make up its 62.

    Encoding matters more than the count. Under one-hot, an UNSEEN combination
    of two well-sampled stresses is an unknown category and encodes as all
    zeros -- the model is told there is no stress rather than that there are
    two. Supplementary Methods 3.3.2 specifies the multi-hot reading: "a 52-by-1
    vector where each is a binary random variable and represents a stress".

    Measured cost: NONE. Ablated against the same encoder with the old 240-wide
    medium block (results/transcriptome_loco_ablation_746.json), one-hot 68 vs
    multi-hot 58 moves PCC/molecule 0.2862 -> 0.2878. A free correctness fix --
    unlike the medium change made the same day, which cost 0.0997.

    NOTE this deliberately does NOT change `canonical_stress` or `make_key`.
    Condition keys are the join surface for every cross-layer merge, and
    config.EXPECTED_OVERLAP asserts their behaviour; re-canonicalizing the
    stress field would move those counts. This is a read-side helper only.
    """
    s = canonical_stress(raw)
    if s == "none":
        return []
    return sorted({tok.strip() for tok in s.split(";") if tok.strip()})


def make_key(strain: str, medium_id: str, stress: str, gp: str | None,
             gene_map: Mapping[str, str] | None = None) -> ConditionKey:
    return ConditionKey(
        strain=(strain or "").strip(),
        medium_id=(medium_id or "").strip(),
        stress=canonical_stress(stress),
        gp=canonical_gp(gp, gene_map),
    )


def parse_transcriptome_cond(cond: str,
                            gene_map: Mapping[str, str] | None = None) -> ConditionKey:
    """Parse the transcriptome file's dotted `Cond` string.

    Format: "{STRAIN}.{MEDIUM_ID}.{STRESS}.{GP}". Stress values may themselves
    contain hyphens ("lactose-shift", "O2-starvation") but not dots, so a
    4-way split is safe; we split from the right to be robust to a strain name
    that ever contains a dot.
    """
    parts = cond.split(".")
    if len(parts) != 4:
        raise ValueError(f"expected 4 dot-separated fields, got {len(parts)}: {cond!r}")
    strain, medium_id, stress, gp = parts
    return make_key(strain, medium_id, stress, gp, gene_map)


def perturbed_genes(gp: str | None) -> list[str]:
    """The gene identifiers touched by a GP field (b-numbers where available)."""
    return [p.gene for p in split_gp(gp)]


def summarize_dialects(values: Iterable[str]) -> dict[str, int]:
    """Diagnostic: count how raw GP tokens map onto canonical forms."""
    from collections import Counter

    out: Counter[str] = Counter()
    for v in values:
        out[canonical_gp(v)] += 1
    return dict(out)

"""The fluxome module: FBA on iJO1366, with bounds informed by omics.

Paper (paper.md:174-178):

    "FBA was used to predict 2382 fluxes, while protein/transcript and
     extra-cellular information from three layers of transcriptome, proteome
     and input was used to inform bounds. More specifically, lower bounds of
     reactions changed by:

         l_i = 0      if mean(g_i) <= t
         l_i = -1000  otherwise

     where l_i is expression levels of enzymes in reaction i. t is empirically
     determined by finding optimal parameter with maximum predictive
     performance."

Why FBA at all, in a data-driven paper
--------------------------------------
43 flux profiles over 120 measured fluxes, against 2,382 reactions to predict.
No statistical model can be fitted from that. FBA needs NO training data -- it
derives fluxes from stoichiometric mass balance plus an optimality assumption --
so the scarce omics data is spent CONSTRAINING it rather than fitting it. Model
rigidity chosen in inverse proportion to data availability.

What the omics constraint actually does
---------------------------------------
Raising a lower bound from -1000 to 0 removes reverse flux for that reaction,
slicing a dimension off the feasible polytope. In the paper this moves the mean
from 0.65 to 0.72 -- but the more striking effect is on the SPREAD, which falls
from +/-0.39 to +/-0.24. Constraining the polytope does not make FBA cleverer,
it makes it less free to be wrong.

Honest limits of this reimplementation:
  * MOMA's rule is the crudest of its family (compare GIMME, iMAT, and E-Flux,
    which scales bounds continuously with expression). Implemented as specified.
  * Only 26 of the compendium's 120 flux reactions carry a published BiGG
    cross-reference in prokaryomics /reactions.json, so validation is limited
    to those. `mappable_reactions()` reports exactly which.

Solved with scipy.optimize.linprog (HiGHS); no cobra dependency.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, hstack, identity, vstack

from ecomics import config as C

__all__ = ["GenomeScaleModel", "FluxomeModule",
           "load_paper_medium_bounds", "GrowthCalibration"]

DEFAULT_BOUND = 1000.0

_GPR_TOKEN = re.compile(r"\(|\)|\band\b|\bor\b|[^\s()]+")


def _eval_gpr(rule: str, level_of):
    """Evaluate a BiGG gene-reaction rule with AND = min, OR = max.

    `(b0001 and b0002) or b0003` means "the complex of b0001+b0002, or else the
    isozyme b0003". A complex runs no faster than its scarcest subunit (min);
    isozymes are alternatives, so the most abundant one carries the reaction
    (max).

    Unmeasured genes are treated as ABSENT rather than as zero, which is the
    difference between "we did not look" and "it is not there":

      * inside an AND, an unmeasured subunit is skipped -- the complex is scored
        on the subunits that were measured, rather than being forced to 0 and
        closing a reaction on no evidence;
      * inside an OR, an unmeasured isozyme simply does not compete;
      * if NOTHING in the rule is measured, the result is None and the caller
        leaves the reaction's bounds alone.

    Never raises on a malformed rule. Unbalanced parentheses degrade to whatever
    operands were recovered (`((a` yields a's level) and an unparsable rule
    yields None, so the caller falls back to the flat mean instead of the layer
    going down mid-solve. Recursion is
    avoided in favour of an explicit stack: iJO1366 has rules with ~30 terms and
    deep nesting, and a blown stack in a solver loop is hard to attribute.
    """
    tokens = _GPR_TOKEN.findall(rule)
    if not tokens:
        return None

    # Shunting-yard into an operand/operator stack pair, evaluating on the way.
    # `and` binds tighter than `or`, as in BiGG's own convention.
    prec = {"or": 1, "and": 2}

    def apply(op, right, left):
        if left is None:
            return right
        if right is None:
            return left
        return min(left, right) if op == "and" else max(left, right)

    values: list = []
    ops: list[str] = []

    def reduce_once() -> bool:
        if not ops or len(values) < 2:
            return False
        op = ops.pop()
        right, left = values.pop(), values.pop()
        values.append(apply(op, right, left))
        return True

    try:
        for tok in tokens:
            low = tok.lower()
            if tok == "(":
                ops.append("(")
            elif tok == ")":
                while ops and ops[-1] != "(":
                    if not reduce_once():
                        break
                if ops and ops[-1] == "(":
                    ops.pop()
            elif low in prec:
                while (ops and ops[-1] != "("
                       and prec.get(ops[-1], 0) >= prec[low]):
                    if not reduce_once():
                        break
                ops.append(low)
            else:
                values.append(level_of(tok))
        while ops:
            op = ops.pop()
            if op == "(":
                continue
            if len(values) < 2:
                break
            right, left = values.pop(), values.pop()
            values.append(apply(op, right, left))
    except Exception:                                # noqa: BLE001
        return None

    if not values:
        return None
    out = values[0]
    return None if out is None else float(out)

def load_paper_medium_bounds(path=None) -> dict[str, dict[str, float]]:
    """The paper's OWN exchange bounds per medium, from Supplementary Data 4.

    THE medium source for this module, and the only one. It lists, for each
    Ecomics medium ID, the exact exchange reactions and lower bounds the authors
    used --

        MD001  M9+Glu   EX_o2(e),...,EX_glc(e)   -1000,...,-20

    keyed by the same MD id that appears in our condition keys, so no name
    matching is involved at all.

    This file is REQUIRED, not optional. It used to have a fallback:
    `MEDIUM_TO_BIGG`, a ~40-entry hand-curated guess at how Ecomics'
    120-component medium ontology maps onto BiGG exchange reactions, applied by
    a `close_unlisted_exchanges()` that switched carbon sources only. Both are
    gone. The guess was never defensible -- its own comment recorded that only
    9 of the 120 component names match a BiGG exchange id by string -- and it
    turned out to be doing nothing: on all four media the compendium's flux
    profiles actually use (MD004, MD066, MD120, MD121, every one of them
    glucose-minimal) it wrote back exactly iJO1366's own defaults, since
    EX_glc__D_e already sits at -10 and every other carbon exchange at 0.
    Removing it changed no reported number. See `apply_paper_bounds` for what
    happens to a medium Data 4 does not list.

    Two translations are needed. The paper writes the pre-2015 BiGG identifier
    style `EX_glc(e)`; iJO1366 as distributed today uses `EX_glc__D_e`, and the
    compartment suffix is not the only difference -- `glc` became `glc__D`
    because BiGG made stereochemistry explicit. Unresolvable ids are returned
    under their original name so the caller can see what was dropped rather than
    silently losing a constraint.

    Returns {medium_id: {reaction_id: lower_bound}}. Stress entries are returned
    under their stress name (e.g. "O2-starvation"), which is how Supplementary
    Data 4 keys them.
    """
    import openpyxl

    from ecomics import config as C

    path = Path(path) if path else C.SUPPLEMENTARY["fba_bounds"]
    if not path.exists():
        raise FileNotFoundError(
            f"{path}\nRun: python scripts/00_acquire.py")
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    rows = [r for r in wb["fba"].iter_rows(values_only=True)]
    wb.close()

    out: dict[str, dict[str, float]] = {}
    for r in rows:
        if (not r or not r[0]
                or str(r[0]).strip() == "Condition Type"
                or str(r[0]).startswith("Supplementary")):
            continue
        mid = str(r[2]).strip()
        rxns = [x.strip() for x in str(r[4] or "").split(",") if x.strip()]
        lbs = [x.strip() for x in str(r[5] or "").strip('"').split(",")
               if x.strip()]
        if len(rxns) != len(lbs):
            continue
        out[mid] = {_bigg_modern(x): float(v) for x, v in zip(rxns, lbs)}
    return out


def _bigg_modern(rid: str) -> str:
    """`EX_glc(e)` -> `EX_glc__D_e`, the identifier style iJO1366 ships with.

    Two independent changes happened when BiGG renumbered:

      compartment   `(e)`      -> `_e`
      stereochemistry `_L`/`_D` -> `__L`/`__D`, doubling the separator so the
                    stereo tag cannot be confused with a name part

    The stereo rule is applied generally rather than by table, because the amino
    acids alone account for 14 of the 59 exchanges here and enumerating them is
    how you end up silently dropping the fifteenth. Only `glc` needs a special
    case: it gained a stereo tag it did not previously carry.

    IDEMPOTENT, which it was not. An id already in modern form fell through to
    the final f-string and got wrapped a second time:

        'EX_glc(e)'   -> EX_glc__D_e      correct
        'EX_ac_e'     -> EX_EX_ac_e_e     corrupted
        'EX_glc__D_e' -> EX_EX_glc__D_e_e corrupted

    Nothing then complained: `apply_paper_bounds` looks the id up with
    `idx.get(rid)` and skips on None, so a mangled id silently dropped its
    constraint -- while this module's docstring promised unresolvable ids are
    surfaced rather than lost. Anything that is not the legacy `EX_*(e)` form is
    now returned untouched.
    """
    if not (rid.startswith("EX_") and rid.endswith("(e)")):
        return rid
    core = rid[3:-3]
    core = re.sub(r"(?<!_)_([LDRS])$",
                  lambda m: "__" + m.group(1), core)
    if core == "glc":
        core = "glc__D"
    return f"EX_{core}_e"


@dataclass
class GenomeScaleModel:
    """A BiGG genome-scale metabolic model, as an LP."""

    reactions: list[str]
    metabolites: list[str]
    S: csr_matrix
    lb: np.ndarray
    ub: np.ndarray
    objective: np.ndarray
    gene_rules: dict[str, str] = field(default_factory=dict)
    rxn_genes: dict[str, list[str]] = field(default_factory=dict)

    @classmethod
    def from_bigg(cls, path: Path | None = None) -> "GenomeScaleModel":
        path = path or C.REMOTE_FILES["bigg_ijo1366"]
        data = json.loads(Path(path).read_text(encoding="utf-8"))

        rxns = data["reactions"]
        rxn_ids = [r["id"] for r in rxns]
        met_ids = sorted({m for r in rxns for m in r["metabolites"]})
        mi = {m: i for i, m in enumerate(met_ids)}

        rows, cols, vals = [], [], []
        lb = np.empty(len(rxns))
        ub = np.empty(len(rxns))
        obj = np.zeros(len(rxns))
        rules, genes = {}, {}
        for j, r in enumerate(rxns):
            for m, c in r["metabolites"].items():
                rows.append(mi[m])
                cols.append(j)
                vals.append(float(c))
            lb[j] = float(r.get("lower_bound", -DEFAULT_BOUND))
            ub[j] = float(r.get("upper_bound", DEFAULT_BOUND))
            obj[j] = float(r.get("objective_coefficient", 0.0))
            rule = r.get("gene_reaction_rule", "") or ""
            rules[r["id"]] = rule
            genes[r["id"]] = sorted(set(
                g for g in rule.replace("(", " ").replace(")", " ")
                .replace(" and ", " ").replace(" or ", " ").split()
                if g not in {"and", "or"} and g))

        if not obj.any():
            # BiGG usually flags the biomass reaction; fall back to name matching
            for j, rid in enumerate(rxn_ids):
                if "BIOMASS" in rid.upper():
                    obj[j] = 1.0
                    break

        S = csr_matrix((vals, (rows, cols)), shape=(len(met_ids), len(rxns)))
        return cls(rxn_ids, met_ids, S, lb, ub, obj, rules, genes)

    # ----------------------------------------------------------------- solve
    def solve(self, lb: np.ndarray | None = None, ub: np.ndarray | None = None
              ) -> tuple[np.ndarray | None, float]:
        """max c'v subject to Sv = 0, lb <= v <= ub."""
        lb = self.lb if lb is None else lb
        ub = self.ub if ub is None else ub
        res = linprog(c=-self.objective, A_eq=self.S,
                      b_eq=np.zeros(self.S.shape[0]),
                      bounds=np.column_stack([lb, ub]), method="highs")
        if not res.success:
            return None, float("nan")
        return res.x, float(-res.fun)

    def solve_mtf(self, lb: np.ndarray | None = None,
                  ub: np.ndarray | None = None, frac_opt: float = 1.0
                  ) -> tuple[np.ndarray | None, float]:
        """Minimum-total-flux FBA: the sparsest optimal solution.

        Supplementary Methods 3.3.6: "Typically, a FBA solution is not one as
        multiple genome-wide fluxes might achieves the same objective. From the
        multiple solutions, we advocate the flux distribution that minimizes
        total the total absolute flux (MTF)."

        WHY THIS MATTERS more than it sounds
        ------------------------------------
        `solve` returns *an* optimal vertex, whichever one HiGHS lands on. FBA
        optima are routinely non-unique -- many flux distributions achieve the
        same biomass -- so without a secondary objective the individual reaction
        fluxes are **not identified**, and comparing them against measured
        fluxes is partly comparing against an arbitrary choice among equivalent
        optima. That is a stronger objection to per-reaction flux validation
        than any data limitation, and it is independent of them.

        MTF is the standard tie-break (also called parsimonious FBA): among all
        solutions achieving the optimum, take the one with least total flux. It
        is a proxy for enzyme cost -- a cell that can run a pathway two ways
        pays less for the shorter one.

        How the LP is built
        -------------------
        |v| is not linear, so an auxiliary t >= |v| is introduced and sum(t) is
        minimized. Since t is pushed down, t = |v| at the optimum:

            minimize    sum(t)
            subject to  S v = 0
                        c'v >= frac_opt * optimum        (near-optimality)
                        v - t <= 0   and   -v - t <= 0   (t >= |v|)
                        lb <= v <= ub,  t >= 0

        ⚠ The obvious alternative -- split v into non-negative forward and
        reverse parts, v = f - r with 0 <= f <= max(ub,0) and
        0 <= r <= max(-lb,0) -- is WRONG here and was tried first. It silently
        drops any bound that does not straddle zero: a reaction with lb > 0
        (iJO1366 forces flux through several) gets r_ub = 0 and f free from 0,
        so the solver may set it to 0 and violate its own lower bound; likewise
        for ub < 0. Keeping v as a variable with its original bounds cannot
        express that error. `test_mtf_respects_the_bounds_it_is_given` is the
        regression test, and it caught exactly this.

        `frac_opt = 1.0` pins the objective exactly, which is the paper's
        "with the same objective value". Values below 1 trade a little biomass
        for a sparser distribution; the argument exists because 1.0 can be
        numerically tight on a degenerate model, not because it should be tuned.

        Returns `(None, nan)` if either LP fails, so an infeasible model stays
        distinguishable from a zero-growth one -- same contract as `solve`.
        """
        lb = self.lb if lb is None else np.asarray(lb, dtype=float)
        ub = self.ub if ub is None else np.asarray(ub, dtype=float)

        _v, opt = self.solve(lb, ub)
        if not np.isfinite(opt):
            return None, float("nan")

        n = len(self.reactions)
        S = csr_matrix(self.S)
        eye = identity(n, format="csr")
        zero = csr_matrix((S.shape[0], n))

        A_eq = hstack([S, zero], format="csr")
        b_eq = np.zeros(S.shape[0])
        A_ub = vstack([
            hstack([eye, -eye], format="csr"),                     # v - t <= 0
            hstack([-eye, -eye], format="csr"),                    # -v - t <= 0
            csr_matrix(np.concatenate([-self.objective,
                                       np.zeros(n)])[None, :]),    # c'v >= frac*opt
        ], format="csr")
        b_ub = np.concatenate([np.zeros(2 * n), [-frac_opt * opt]])

        t_ub = np.maximum(np.abs(lb), np.abs(ub))
        bounds = np.vstack([np.column_stack([lb, ub]),
                            np.column_stack([np.zeros(n), t_ub])])

        res = linprog(c=np.concatenate([np.zeros(n), np.ones(n)]),
                      A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub,
                      bounds=bounds, method="highs")
        if not res.success:
            return None, float("nan")
        v = res.x[:n]
        # Report the biomass this solution actually carries, not the target.
        return v, float(self.objective @ v)

    def index(self) -> dict[str, int]:
        return {r: i for i, r in enumerate(self.reactions)}


@dataclass
class FluxomeModule:
    """FBA with expression-thresholded lower bounds."""

    model: GenomeScaleModel
    # The paper's own value. Supplementary Methods 3.3.6 sweeps
    # {0.02, 0.04, 0.06, 0.08} and selects 0.04, with the range "determined to
    # be below 0.1 as the mean expression level of genes was 0.10".
    #
    # ⚠ This default was 0.1 -- exactly the value the paper's range was chosen
    # to stay below -- and 0.1 is CATASTROPHIC on this compendium. Measured on
    # the condition-averaged transcriptome, whose gene means run 0.058 to 0.407
    # with a median of 0.094 (the paper's 0.10, reproduced):
    #
    #     t = 0.02 / 0.04 / 0.05   ->    0 bounds changed, biomass 0.982372
    #     t = 0.10                 ->  111 bounds changed, biomass 0.000000
    #     t = 0.20                 ->  216 bounds changed, biomass 0.000000
    #
    # At 0.1 the threshold sits on the median, closes reverse flux through half
    # the gene-associated network at once, and the LP returns zero growth --
    # a legal answer that raises nothing, exactly like the starvation bug
    # recorded in `apply_paper_bounds`. A spec-completion sweep
    # measures the collapse.
    threshold: float = 0.04
    exchange_prefix: str = "EX_"
    # "mean"  -- the paper's rule: mean over the reaction's measured enzymes,
    #            which discards the GPR's Boolean structure. THE DEFAULT,
    #            because this module reproduces the paper rather than improving
    #            on it (see the module docstring).
    # "logic" -- evaluate the GPR tree with AND = min, OR = max. Biochemically
    #            the right thing and what GIMME/iMAT do, but NOT what Kim et al.
    #            specify. Provided so the difference can be measured; see
    #            a spec-completion sweep.
    gpr_mode: str = "mean"

    def reaction_level(self, rid: str, expression: dict[str, float],
                       proteome: dict[str, float] | None = None
                       ) -> float | None:
        """One expression level for a reaction, or None if nothing is measured.

        Two things the specification asks for, and one it does not.

        **Source precedence (specified).** Supplementary Methods 3.3.6:
        "Expression levels of enzymes are interrogated from proteome layer if
        measured and from transcriptome layer otherwise." So `proteome` wins
        per GENE, not per reaction -- a reaction with two measured proteins and
        one transcript-only gene uses all three, taking the protein value where
        it exists. Reading it per reaction instead would discard the
        transcript-only genes of any partly-proteomic reaction.

        **Aggregation (specified as `mean`).** The paper thresholds
        `mean(g_i)`, which already throws the Boolean structure away, so
        `gpr_mode="mean"` is the faithful reading and the default.

        **`gpr_mode="logic"` is an improvement, not a fix.** AND means an enzyme
        complex, where the scarcest subunit is limiting, so `min`; OR means
        interchangeable isozymes, where the most abundant suffices, so `max`.
        Note the errors run in OPPOSITE directions -- averaging calls a complex
        open when a subunit is missing, and calls an isozyme set closed when one
        isozyme is abundant -- so they do not cancel across reactions. That is
        why the difference is worth measuring even though `mean` stays default.

        Returns None when no gene of the reaction is measured in either layer,
        which is what leaves the reaction's bounds untouched: absence of
        evidence is not evidence of absence.
        """
        def level_of(gene: str) -> float | None:
            if proteome is not None and gene in proteome:
                return float(proteome[gene])
            if gene in expression:
                return float(expression[gene])
            return None

        if self.gpr_mode == "logic":
            rule = self.model.gene_rules.get(rid, "")
            if rule:
                val = _eval_gpr(rule, level_of)
                if val is not None:
                    return val
            # Fall through to the flat mean when there is no parsable rule --
            # some reactions carry genes with no rule string.

        genes = self.model.rxn_genes.get(rid, [])
        levels = [v for v in (level_of(g) for g in genes) if v is not None]
        return float(np.mean(levels)) if levels else None

    def bounds_from_expression(self, expression: dict[str, float],
                               proteome: dict[str, float] | None = None
                               ) -> tuple[np.ndarray, np.ndarray]:
        """Apply the paper's rule, plus medium-derived exchange bounds.

            l_i = 0       if mean(expression of reaction i's enzymes) <= t
            l_i = -1000   otherwise

        Reactions whose genes are all unmeasured keep their original bounds --
        absence of evidence is not evidence of absence, and asserting
        irreversibility for an unmeasured reaction would be a fabrication.

        The -1000 branch is CLAMPED to the model's own lower bound, and that is
        a deliberate departure from the rule as literally written. iJO1366
        curates 1,947 reactions as thermodynamically irreversible (lb >= 0), of
        which 1,529 are gene-associated. Taking `l_i = -1000` at face value
        opens every one of those whose enzymes happen to be expressed, and the
        LP stops being a metabolic model:

            biomass, default bounds        0.9824
            biomass, rule applied blindly 25.8253   <- 26x, doubling time ~1.6 min

        A solution at 25.8 h^-1 is riding thermodynamically infeasible loops, so
        every flux it reports is meaningless. The rule is a way to *restrict*
        reactions whose enzymes are absent, not a licence to reverse reactions
        biochemistry says cannot run backwards -- which is the same principle the
        unmeasured case above already applies. `min(model.lb[j], 0.0)` keeps a
        reversible reaction reversible and leaves an irreversible one closed.
        """
        lb, ub = self.model.lb.copy(), self.model.ub.copy()
        for j, rid in enumerate(self.model.reactions):
            level = self.reaction_level(rid, expression, proteome)
            if level is None:
                continue
            lb[j] = (0.0 if level <= self.threshold
                     else min(float(self.model.lb[j]), 0.0))

        # NOTE: the medium is applied by apply_paper_bounds(), from the authors'
        # own Supplementary Data 4 list. Deriving the medium here instead, by
        # opening every nutrient an ontology lists to +/-1000, produced a
        # degenerate zero-growth LP -- see that method's docstring.
        return lb, ub

    # The carbon inputs iJO1366 exposes as exchange reactions. Nothing in this
    # module writes to them any more -- the medium arrives whole, from
    # Supplementary Data 4 -- but the set is the natural definition of "the
    # ways carbon can enter the model", and the tests close exactly
    # these to assert the LP's most basic contract: no carbon, no biomass.
    CARBON_EXCHANGES = frozenset({
        "EX_glc__D_e", "EX_glyc_e", "EX_lac__D_e", "EX_lcts_e", "EX_ac_e",
        "EX_fru_e", "EX_gal_e", "EX_mal__L_e", "EX_man_e", "EX_pyr_e",
        "EX_rmn_e", "EX_succ_e", "EX_xyl__D_e", "EX_fum_e", "EX_for_e",
        "EX_fuc__L_e",
    })

    def apply_paper_bounds(self, lb: np.ndarray, medium_id: str,
                           bounds: dict | None = None) -> np.ndarray:
        """Set exchange bounds from Supplementary Data 4, by Ecomics medium ID.

        The ONLY medium path. Data 4 gives the authors' own exchange list and
        lower bounds for each medium, keyed by the same MD id our condition keys
        carry, and all 59 reactions it names resolve against iJO1366.

        Why nothing derives the medium from the ontology instead
        -------------------------------------------------------
        An earlier version closed EVERY exchange not on a hand-written keep-list.
        That starved the model: growth fell to exactly 0 in all four flux media
        while glucose uptake pinned at -1000, i.e. a degenerate LP vertex with no
        biomass. The lesson is that iJO1366's DEFAULT bounds already encode a
        curated minimal medium, and re-deriving one by hand from a 120-component
        ontology is a good way to omit something essential.

        The surviving fragment of that attempt -- `MEDIUM_TO_BIGG`, restricted to
        carbon sources so it could not starve anything -- has since been removed
        too. It mapped 9 of 120 component names by string and the rest by hand,
        and on the four media the flux profiles use it wrote back exactly
        iJO1366's own defaults, so it was a no-op wearing the costume of a
        curation decision.

        Unknown medium ids leave `lb` untouched and are reported by the caller
        rather than silently falling back -- a medium we cannot constrain should
        look unconstrained, not look like a different medium. Since Data 4 lists
        MD004 but not MD066/MD120/MD121, that is the live case here, not a
        hypothetical one.
        """
        table = bounds if bounds is not None else load_paper_medium_bounds()
        spec = table.get(medium_id)
        if not spec:
            return lb
        idx = self.model.index()
        lb = lb.copy()
        for rid, val in spec.items():
            j = idx.get(rid)
            if j is not None:
                lb[j] = float(val)
        return lb

    def predict(self, expression: dict[str, float],
                medium_id: str | None = None,
                bounds: dict | None = None
                ) -> tuple[np.ndarray | None, float]:
        """Expression-constrained FBA, optionally on a specified medium.

        The medium is named by its Ecomics `MD###` id, not described as a
        component dict: Supplementary Data 4 is keyed by that id, so the caller
        has nothing to translate. Pass `bounds` to reuse a table already loaded
        -- reading the workbook once per condition is the whole cost.

        The parameter used to be a component dict, and before that it was
        accepted and never read at all -- `bounds_from_expression` binds it and
        ignores it, so `predict(expr, medium)` returned an answer with the medium
        unapplied that looked like a medium-specific one. It was masked only
        because the single caller in `scripts/04_reproduce.py` applied the medium
        itself afterwards.
        """
        lb, ub = self.bounds_from_expression(expression)
        if medium_id is not None:
            lb = self.apply_paper_bounds(lb, medium_id, bounds)
        return self.model.solve(lb, ub)

    # ------------------------------------------------------------- mapping
    def mappable_reactions(self, r_to_bigg: dict[str, str]) -> dict[str, int]:
        """Ecomics R0001..R0120 -> column index in the genome-scale model.

        Only reactions with a published BiGG cross-reference that also exists in
        iJO1366 can be compared against measured fluxes.
        """
        idx = self.model.index()
        out = {}
        for rid, bigg in r_to_bigg.items():
            if not bigg or bigg == "na":
                continue
            for cand in (bigg, bigg.upper(), bigg.replace("-", "_")):
                if cand in idx:
                    out[rid] = idx[cand]
                    break
        return out

    def flux_variability(self, lb: np.ndarray, ub: np.ndarray,
                         columns: list[int], frac_opt: float = 0.99
                         ) -> np.ndarray:
        """Range each listed flux can take among near-optimal solutions.

        The direct measure of how much the omics constraints shrink the
        feasible polytope -- which is what actually improves FBA here.
        """
        _v, opt = self.model.solve(lb, ub)
        if not np.isfinite(opt):
            return np.full(len(columns), np.nan)
        A_ub = -self.model.objective[None, :]
        b_ub = np.array([-frac_opt * opt])
        bounds = np.column_stack([lb, ub])

        widths = np.full(len(columns), np.nan)
        for k, j in enumerate(columns):
            c = np.zeros(len(self.model.reactions))
            c[j] = 1.0
            lo = linprog(c=c, A_eq=self.model.S, b_eq=np.zeros(self.model.S.shape[0]),
                         A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
            hi = linprog(c=-c, A_eq=self.model.S, b_eq=np.zeros(self.model.S.shape[0]),
                         A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
            if lo.success and hi.success:
                widths[k] = (-hi.fun) - lo.fun
        return widths


# --------------------------------------------------------------------------
# FBA biomass -> measured growth rate
# --------------------------------------------------------------------------
@dataclass
class GrowthCalibration:
    """Linear map from FBA biomass flux onto measured growth rate, in 1/h.

    Supplementary Methods 3.3.6, last sentence: "To have the predicted growth
    rate from FBA comparable to ones predicted from other layers, we built a
    linear transformation function between predicted growth rate from FBA and
    measured growth rate for training conditions."

    Why this is needed at all
    -------------------------
    FBA maximizes a biomass *pseudo-reaction* whose coefficients are chosen so
    that a flux of 1 corresponds to one gram of dry weight per gram of dry
    weight per hour. That is a modelling convention, not a measurement, and it
    depends on the biomass composition the model's authors assumed. The phenome
    consensus averages growth rates in 1/h from five layers, so a raw biomass
    flux cannot be averaged with them -- it is on the model's own scale, and
    the two agree only up to an affine transform.

    Fitted on the TRAINING conditions only, exactly as the sentence specifies,
    so this stays inside the fold like every other fitted quantity.

    Why a bare least-squares line, and not `_lasso.fit_lasso`
    ---------------------------------------------------------
    One predictor and no feature selection to do, so an L1 penalty has nothing
    to select and would only bias the slope toward zero. `np.polyfit(x, y, 1)`
    is the same estimator `proteome.own_mrna_baseline` uses for the same reason.

    WHAT THIS DOES NOT FIX
    ----------------------
    Implementing this does **not** let the fluxome join the phenome consensus,
    and the gap it closes is smaller than it looks. Of the 31 conditions with a
    flux profile, **25** carry a measured growth rate -- but only **1** of those
    is among the 179 conditions the phenome evaluation actually runs on, and
    Supplementary Data 5 -- which supplied the missing growth data for the
    proteome and metabolome layers -- has no fluxome sheet at all.
    `PhenomeConsensus` requires 4 usable rows to fit a layer and 6 to
    cross-validate its weight, so the fluxome layer is dropped for lack of data
    whether or not this calibration exists.

    ⚠ That **25** would read **3** without the b-number normalization: the
    fluxome wrote its knockout genotypes as uppercase gene symbols where every
    other layer wrote b-numbers, so 22 KO conditions silently failed to join to
    their own growth measurements. Fixing the join fits the line on 25 points
    instead of 3 and changes the conclusion not at all -- which is the useful
    part. The binding constraint was never the number of growth measurements; it
    is the overlap with the phenome evaluation set, and that is still 1 of 179.
    A join that returns *some* rows is worse than one that returns none.

    The fit is barely determined either way. FBA returns only **2** distinct
    biomass values across those 25 conditions, because MD066 and MD121 are
    absent from Supplementary Data 4 and so keep iJO1366's default bounds. A
    line through two distinct x values is an interpolation between two medium
    classes scored on itself -- do not quote its fit quality as evidence.

    So the fifth layer is blocked TWICE, and the binding constraint is the
    data, not this function. `fit` returns an unfitted object rather than
    raising when it is handed too few points, and `ok` reports which.
    """

    slope: float = float("nan")
    intercept: float = float("nan")
    n: int = 0
    min_points: int = 3

    @property
    def ok(self) -> bool:
        return np.isfinite(self.slope) and np.isfinite(self.intercept)

    def fit(self, biomass: np.ndarray, growth: np.ndarray) -> "GrowthCalibration":
        """Fit growth ~ a*biomass + b over the pairs where both are finite."""
        x = np.asarray(biomass, dtype=float).ravel()
        y = np.asarray(growth, dtype=float).ravel()
        if x.shape != y.shape:
            raise ValueError(f"biomass {x.shape} and growth {y.shape} differ")
        ok = np.isfinite(x) & np.isfinite(y)
        self.n = int(ok.sum())
        # Degenerate inputs give a meaningless line rather than an exception:
        # too few points, or an x with no spread (which is the usual case here,
        # since every flux medium is glucose-minimal and FBA returns the same
        # biomass for all of them). Leaving slope NaN makes `ok` False and
        # `predict` return NaN, which the consensus already treats as
        # "this layer has nothing to say" -- see phenome.PhenomeConsensus.
        if self.n < self.min_points or np.ptp(x[ok]) < 1e-12:
            self.slope = self.intercept = float("nan")
            return self
        self.slope, self.intercept = (float(v) for v in np.polyfit(x[ok], y[ok], 1))
        return self

    def predict(self, biomass: np.ndarray) -> np.ndarray:
        """Map biomass fluxes onto 1/h. All-NaN when the fit did not happen."""
        x = np.asarray(biomass, dtype=float).ravel()
        if not self.ok:
            return np.full(x.shape, np.nan)
        out = self.slope * x + self.intercept
        # A negative predicted growth rate is not a value the phenome layer can
        # hold -- growth rates in the compendium run 0.01 to 2.14 /h. Clamp at
        # zero rather than emit one, and keep NaN as NaN.
        return np.where(np.isfinite(x), np.maximum(out, 0.0), np.nan)

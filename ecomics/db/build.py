"""Build data/ecomics.db from every source, and mirror wide matrices to Parquet.

Sources
    data/Ecomics.transcriptome.no_avg.v8.txt   3,578 profiles x 4,096 genes
    data/Ecomics.proteome.v5.csv                  33 conditions x 589 proteins
    data/Ecomics.metabolome.v3.csv                49 conditions x 114 metabolites
    data/external/prokaryomics/fluxome.json       43 profiles x 120 fluxes
    data/external/prokaryomics/phenome.json      253 conditions x LT/GR/FOD
    data/external/prokaryomics/strain.json        65 strains x 152 features
    data/external/prokaryomics/medium.json       112 media x 120 components
    data/external/prokaryomics/{molecule,reaction}.json

The build ends by asserting the cross-layer overlaps the paper reports. Those
assertions are the regression test for db/canon.py: if GP canonicalization
regresses, the overlaps collapse to zero and the build fails loudly instead of
silently producing empty training sets.
"""

from __future__ import annotations

import csv
import json
import re
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np

from ecomics import config as C
from ecomics.db.canon import ConditionKey, make_key, parse_transcriptome_cond, split_gp

# Descriptive columns in the scraped meta-data tables; everything else is a feature.
STRAIN_META = {"Strain Name", "Link", "PMID", "Alternate Names"}
MEDIUM_META = {"ID", "Base Medium", "Description", "Link", "PMID", "Defined"}

# Values in the strain table meaning "marker absent".
ABSENT = {"no", "none", "na", "", "-", "?"}

# "328mM" / "0.40%" / "100ug/mL" / "1mg/L" / "8.2 mM"
_AMOUNT_RE = re.compile(r"^\s*([0-9]*\.?[0-9]+)\s*([a-zA-Z%/]*)\s*$")

# Two cells wrap an otherwise ordinary reading in a qualifier the regex above
# rejects: MD065 MgSO4 "low(10nM)" and MD076 NaCl "high(0.3M)". Without this
# they fall through to (None, None, True) -- present, amount unknown -- and the
# encoder's `numeric()` then imputes them onto the component median, which is
# 0.0 for 116 of the 120 components. That produced the compendium's only
# "present at zero concentration" cells. The number is in the file; only the
# wrapper is in the way.
#
# Deliberately narrow: it matches ONLY a leading word plus parentheses around
# the whole value. A looser rule (strip anything non-numeric) would silently
# invent readings from qualifiers that do not carry one.
_QUALIFIED_RE = re.compile(r"^\s*[a-zA-Z]+\s*\(\s*(?P<inner>[^()]+?)\s*\)\s*$")

csv.field_size_limit(1 << 31)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def parse_amount(raw: str | None) -> tuple[float | None, str | None, bool]:
    """Parse a medium component cell into (amount, unit, present).

    The published table mixes numeric concentrations ('328mM', '0.40%') with
    explicit unknowns ('?'). Rather than coercing everything to a number and
    inventing precision, we keep the numeric reading where one exists and
    always return a reliable `present` flag, which is what the feature encoder
    mostly uses.

    ⚠ The boolean branch ('yes'/'y'/'present') and the 'no'/'-'/'unknown'/'na'
    tokens are DEFENSIVE, not descriptive: counted over all 112 media, they
    never fire. Those words live only in the six metadata columns (Link='na',
    PMID='unknown', Defined='yes'), which MEDIUM_META excludes from `comps`
    and load_media consumes separately. The 120 component columns contain only
    '0' (11,725), a concentration (1,517), '?' (194), '' (2) and two malformed
    cells. Do not read this signature as a description of the file -- an
    earlier note did, and concluded the 'yes' branch fires twice.

    The two malformed cells are MD065 MgSO4 'low(10nM)' and MD076 NaCl
    'high(0.3M)'. Both carry a usable number that _AMOUNT_RE rejects for its
    wrapper text. They used to fall through to (None, None, True) and then
    impute to amount 0.0 -- the compendium's only "present at zero
    concentration" cells. _QUALIFIED_RE now unwraps them: they read (10.0,
    'nM', True) and (0.3, 'M', True). The presence bit is unchanged either way,
    so this moves the 'amount'/'both' encodings only, not the 'present'
    default.
    """
    if raw is None:
        return None, None, False
    s = str(raw).strip()
    if s == "" or s in {"0", "no", "-"}:
        return 0.0, None, False
    if s in {"?", "unknown", "na"}:
        return None, None, False        # unknown, treated as absent
    if s in {"yes", "y", "present"}:
        return None, None, True
    m = _AMOUNT_RE.match(s)
    if m:
        return float(m.group(1)), (m.group(2) or None), float(m.group(1)) != 0.0
    q = _QUALIFIED_RE.match(s)          # 'low(10nM)' -> '10nM'
    if q and (m := _AMOUNT_RE.match(q.group("inner"))):
        return float(m.group(1)), (m.group(2) or None), float(m.group(1)) != 0.0
    return None, None, True             # unparseable but non-empty -> present


def named_in_medium(component: str, base_medium: str | None,
                    description: str | None) -> bool:
    """Does the medium's own name announce this component?

    Resolves ONE narrow case of `?`. A `?` cell means the concentration is
    unrecorded, and `parse_amount` reads that as absent -- which is right for a
    defined minimal medium and wrong when the medium is literally called after
    the component. `MD094` is `LB+Fru` with `Fru = '?'`, and it was encoding as
    plain `LB`; likewise `M9+Gal`, `M9+T`, `TPM2+Glu`.

    Deliberately the CONSERVATIVE rule of the four considered. Counted over all
    194 `?` cells in 15 media, the token is not uniform:

        125  rich/complex media (dYT, TB, LB+, MOPS+CA, M63+CA, EZ)   present
         63  DEFINED MINIMAL media (MD046 'minimal salt', MD043/MD079
             'K') -- glucose plus salts and nothing else               ABSENT
          5  named in the medium's own description                    present
          1  MD100 'W2' MgSO4                                         unclear

    Flipping every `?` to present -- the obvious reading of Supplementary
    Methods 3.1.3 ("for media with casamino acids we assume that all 20 amino
    acids are present") -- would corrupt those 63 to fix 130. Keying on
    `medium.defined` fails too: MD022 (`MOPS+CA`) is defined=1 yet genuinely
    contains casamino acids.

    So this rule fires on 4 cells only, and leaves the amino-acid block for a
    curated pass. It closes 2 of the 3 collision groups in
    features.KNOWN_COLLISION_CAUSES; MD010 == MD022 stays, because "CA" does
    not name the individual amino acids.

    Token-boundary matched, so 'Glu' does not match 'Glucose' and 'T' does not
    match 'Tet'.
    """
    hay = f"{base_medium or ''} {description or ''}".lower()
    pat = r"(^|[^a-z0-9])" + re.escape(component.lower()) + r"([^a-z0-9]|$)"
    return re.search(pat, hay) is not None


def _num(v) -> float | None:
    if v is None:
        return None
    s = str(v).strip()
    if s == "" or s.upper() in {"NA", "NAN", "NULL"}:
        return None
    try:
        return float(s)
    except ValueError:
        return None


class Builder:
    def __init__(self, db_path: Path = C.DB_PATH, verbose: bool = True):
        self.db_path = db_path
        self.verbose = verbose
        self.conn: sqlite3.Connection
        self._cond_id: dict[ConditionKey, int] = {}
        self._next_profile = 1
        self.stats: dict[str, int] = {}
        self._gene_map = self._load_gene_map()

    @staticmethod
    def _load_gene_map() -> dict[str, str]:
        """Gene symbol -> b-number, for canonicalizing perturbation names.

        The fluxome writes `TALB(KO)` where every other layer writes
        `b0008(KO)` -- the same knockout. Unreconciled, that is 23 duplicate
        feature columns AND three cross-layer joins that silently return almost
        nothing: proteome/metabolome/phenome against fluxome were 1, 2 and 3
        conditions where the data supports 22, 23 and 25.

        ASSERTED non-empty on purpose. `networks.gene_symbol_map()` returns an
        empty dict when its KEGG file is missing, and an empty map here would
        silently produce DIFFERENT condition keys -- a build whose join surface
        depends on whether an untracked file happens to exist. Failing loudly
        is the whole point; see canon.normalize_gene.
        """
        from ecomics.networks import gene_symbol_map

        m = {k.lower(): v for k, v in gene_symbol_map().items()}
        if not m:
            raise RuntimeError(
                "gene_symbol_map() is empty -- "
                f"{C.REMOTE_FILES.get('kegg_gene_list')} is missing.\n"
                "Condition keys depend on it (fluxome writes gene SYMBOLS, every "
                "other layer writes b-numbers), so building without it would "
                "produce a database that silently fails to join.\n"
                "Run: python scripts/00_acquire.py")
        return m

    def log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    # ---------------------------------------------------------------- setup
    def open(self) -> None:
        C.ensure_dirs()
        if self.db_path.exists():
            self.db_path.unlink()
        self.conn = sqlite3.connect(self.db_path)
        self.conn.executescript((Path(__file__).parent / "schema.sql").read_text())
        self.conn.commit()

    def close(self) -> None:
        self.conn.commit()
        self.conn.close()

    # ------------------------------------------------------------ condition
    def condition_id(self, key: ConditionKey) -> int:
        """Intern a condition, creating it (and its perturbation links) once."""
        if key in self._cond_id:
            return self._cond_id[key]
        cur = self.conn.execute(
            "INSERT INTO condition (key, strain, medium_id, stress, gp, is_wildtype) "
            "VALUES (?,?,?,?,?,?)",
            (key.as_str(), key.strain, key.medium_id, key.stress, key.gp,
             int(key.is_wildtype)),
        )
        cid = cur.lastrowid
        self._cond_id[key] = cid
        for p in split_gp(key.gp, self._gene_map):
            self.conn.execute(
                "INSERT OR IGNORE INTO perturbation (id, gene, type) VALUES (?,?,?)",
                (p.as_str(), p.gene, p.type))
            self.conn.execute(
                "INSERT OR IGNORE INTO condition_perturbation VALUES (?,?)",
                (cid, p.as_str()))
        self.conn.execute("INSERT OR IGNORE INTO stress (name) VALUES (?)",
                          (key.stress,))
        return cid

    def add_profile(self, layer: str, cid: int, source_id: str | None,
                    averaged: bool) -> int:
        pid = self._next_profile
        self._next_profile += 1
        self.conn.execute(
            "INSERT INTO profile (id, source_id, layer, condition_id, is_averaged) "
            "VALUES (?,?,?,?,?)", (pid, source_id, layer, cid, int(averaged)))
        return pid

    # ------------------------------------------------------------- metadata
    def load_strains(self) -> None:
        recs = json.loads((C.PROK_DIR / "strain.json").read_text(encoding="utf-8"))
        feats = [k for k in recs[0] if k not in STRAIN_META]
        for r in recs:
            self.conn.execute(
                "INSERT OR REPLACE INTO strain VALUES (?,?,?,?)",
                (r.get("Strain Name"), r.get("Link"), r.get("PMID"),
                 r.get("Alternate Names")))
            rows = []
            for f in feats:
                v = r.get(f)
                present = int(str(v).strip().lower() not in ABSENT)
                rows.append((r.get("Strain Name"), f, v, present))
            self.conn.executemany(
                "INSERT OR REPLACE INTO strain_feature VALUES (?,?,?,?)", rows)
        self.stats["strains"] = len(recs)
        self.stats["strain_features"] = len(feats)
        self.log(f"  strains          {len(recs):>5d} x {len(feats)} features")

    def load_media(self) -> None:
        recs = json.loads((C.PROK_DIR / "medium.json").read_text(encoding="utf-8"))
        comps = [k for k in recs[0] if k not in MEDIUM_META]
        for r in recs:
            self.conn.execute(
                "INSERT OR REPLACE INTO medium VALUES (?,?,?,?,?,?)",
                (r.get("ID"), r.get("Base Medium"), r.get("Description"),
                 r.get("Link"), r.get("PMID"),
                 int(str(r.get("Defined", "")).strip().lower() == "yes")))
            rows = []
            for cname in comps:
                amount, unit, present = parse_amount(r.get(cname))
                # '?' means unrecorded, not absent, when the medium is named
                # after the component -- see named_in_medium.
                if (not present and str(r.get(cname)).strip() == "?"
                        and named_in_medium(cname, r.get("Base Medium"),
                                            r.get("Description"))):
                    present = True
                rows.append((r.get("ID"), cname, r.get(cname), amount, unit,
                             int(present)))
            self.conn.executemany(
                "INSERT OR REPLACE INTO medium_component VALUES (?,?,?,?,?,?)", rows)
        self.stats["media"] = len(recs)
        self.stats["medium_components"] = len(comps)
        self.log(f"  media            {len(recs):>5d} x {len(comps)} components")

    def load_molecules(self) -> None:
        recs = json.loads((C.PROK_DIR / "molecule.json").read_text(encoding="utf-8"))
        self.conn.executemany(
            "INSERT OR REPLACE INTO molecule VALUES (?,?,?,?)",
            [(r.get("Name"), r.get("Molecule"), r.get("Name"), r.get("Description"))
             for r in recs if r.get("Name")])
        recs_r = json.loads((C.PROK_DIR / "reaction.json").read_text(encoding="utf-8"))
        self.conn.executemany(
            "INSERT OR REPLACE INTO reaction VALUES (?,?,?)",
            [(r.get("Reaction"), r.get("Description"), r.get("BIGG"))
             for r in recs_r])
        self.stats["molecules"] = len(recs)
        self.stats["reactions"] = len(recs_r)
        self.log(f"  molecules        {len(recs):>5d}   reactions {len(recs_r)}")

    # ---------------------------------------------------------------- omics
    def load_transcriptome(self) -> tuple[list[str], list[int], np.ndarray]:
        path = C.TRANSCRIPTOME_TXT
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            header = next(reader)
            genes = [h[2:] for h in header[2:]]
            pids: list[int] = []
            rows: list[np.ndarray] = []
            batch: list[tuple] = []
            for row in reader:
                key = parse_transcriptome_cond(row[1], self._gene_map)
                cid = self.condition_id(key)
                pid = self.add_profile("transcriptome", cid, row[0], averaged=False)
                pids.append(pid)
                vals = np.array([_num(v) if v else np.nan for v in row[2:]],
                                dtype=np.float32)
                rows.append(vals)
                batch.extend((pid, g, float(v))
                             for g, v in zip(genes, vals) if not np.isnan(v))
                if len(batch) > 400_000:
                    self.conn.executemany(
                        "INSERT INTO measurement VALUES (?,?,?)", batch)
                    batch.clear()
            if batch:
                self.conn.executemany("INSERT INTO measurement VALUES (?,?,?)", batch)
        mat = np.vstack(rows)
        self.stats["transcriptome_profiles"] = len(pids)
        self.stats["transcriptome_genes"] = len(genes)
        self.log(f"  transcriptome    {mat.shape[0]:>5d} profiles x {mat.shape[1]} genes")
        return genes, pids, mat

    def load_wide_csv(self, path: Path, layer: str
                      ) -> tuple[list[str], list[int], np.ndarray]:
        """Load the condition-averaged proteome / metabolome CSVs.

        Columns: Strain, MediumID, Medium, Stress, GP, then m.<MOLECULE>.
        These are already averaged per condition, so one row == one profile
        == one condition, and is_averaged is set accordingly.
        """
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            mols = [h[2:] for h in header[5:]]
            pids, rows, batch = [], [], []
            for row in reader:
                key = make_key(row[0], row[1], row[3], row[4], self._gene_map)
                cid = self.condition_id(key)
                pid = self.add_profile(layer, cid, None, averaged=True)
                pids.append(pid)
                vals = np.array([_num(v) if v != "" else np.nan for v in row[5:]],
                                dtype=np.float32)
                rows.append(vals)
                batch.extend((pid, m, float(v))
                             for m, v in zip(mols, vals) if not np.isnan(v))
            self.conn.executemany("INSERT INTO measurement VALUES (?,?,?)", batch)
        mat = np.vstack(rows)
        self.stats[f"{layer}_profiles"] = len(pids)
        self.stats[f"{layer}_molecules"] = len(mols)
        self.log(f"  {layer:<16s} {mat.shape[0]:>5d} conditions x {mat.shape[1]}")
        return mols, pids, mat

    def load_fluxome(self) -> tuple[list[str], list[int], np.ndarray]:
        recs = json.loads((C.PROK_DIR / "fluxome.json").read_text(encoding="utf-8"))
        rxns = sorted(k for k in recs[0] if re.fullmatch(r"R\d+", k))

        # The fluxome carries 120 reaction columns but /reactions.json documents
        # only 115: R0080, R0081, R0109, R0111 and R0117 have flux values but no
        # description or BiGG cross-reference published. Register them so the
        # foreign key holds; they simply cannot be mapped into iJO1366 and the
        # FBA module reports them as unmappable rather than dropping them here.
        undocumented = [r for r in rxns if not self.conn.execute(
            "SELECT 1 FROM reaction WHERE id=?", (r,)).fetchone()]
        if undocumented:
            self.conn.executemany(
                "INSERT INTO reaction (id, description, bigg) VALUES (?,?,?)",
                [(r, "undocumented in prokaryomics /reactions.json", None)
                 for r in undocumented])
            self.log(f"  note: {len(undocumented)} flux reaction(s) undocumented "
                     f"({', '.join(undocumented)})")

        pids, rows, batch = [], [], []
        for r in recs:
            key = make_key(r["Strain"], r["MediumID"], r["Stress"], r["GP"],
                           self._gene_map)
            cid = self.condition_id(key)
            pid = self.add_profile("fluxome", cid, None, averaged=True)
            pids.append(pid)
            vals = np.array([_num(r.get(x)) if r.get(x) is not None else np.nan
                             for x in rxns], dtype=np.float32)
            rows.append(vals)
            batch.extend((pid, x, float(v))
                         for x, v in zip(rxns, vals) if not np.isnan(v))
        self.conn.executemany("INSERT INTO flux VALUES (?,?,?)", batch)
        mat = np.vstack(rows)
        self.stats["fluxome_profiles"] = len(pids)
        self.stats["fluxome_reactions"] = len(rxns)
        self.log(f"  fluxome          {mat.shape[0]:>5d} profiles x {mat.shape[1]} fluxes")
        return rxns, pids, mat

    def load_phenome(self) -> tuple[list[int], np.ndarray]:
        recs = json.loads((C.PROK_DIR / "phenome.json").read_text(encoding="utf-8"))
        pids, rows = [], []
        for r in recs:
            key = make_key(r["Strain"], r["MediumID"], r["Stress"], r["GP"],
                           self._gene_map)
            cid = self.condition_id(key)
            pid = self.add_profile("phenome", cid, None, averaged=True)
            pids.append(pid)
            lt, gr, fod = _num(r.get("LT")), _num(r.get("GR")), _num(r.get("FOD"))
            rows.append([lt, gr, fod])
            self.conn.execute("INSERT INTO phenotype VALUES (?,?,?,?)",
                              (pid, lt, gr, fod))
        mat = np.array(rows, dtype=np.float32)
        self.stats["phenome_profiles"] = len(pids)
        self.log(f"  phenome          {len(pids):>5d} conditions x 3 phenotypes")
        return pids, mat

    # ------------------------------------------------------------- parquet
    def write_parquet(self, name: str, pids: list[int], cols: list[str],
                      mat: np.ndarray) -> None:
        """Mirror a wide matrix to Parquet, keyed by profile id and condition key."""
        import pandas as pd

        keys = {cid: k for k, cid in self._cond_id.items()}
        cond_of = dict(self.conn.execute(
            "SELECT id, condition_id FROM profile").fetchall())
        df = pd.DataFrame(mat, columns=cols)
        df.insert(0, "condition_key", [keys[cond_of[p]].as_str() for p in pids])
        df.insert(0, "profile_id", pids)
        C.PARQUET_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(C.PARQUET_DIR / f"{name}.parquet", index=False)

    # ----------------------------------------------------------- assertions
    def check(self) -> list[str]:
        """Assert the cross-layer overlaps the paper reports."""
        problems: list[str] = []
        q = """
        SELECT COUNT(*) FROM (
            SELECT condition_id FROM profile WHERE layer=?
            INTERSECT
            SELECT condition_id FROM profile WHERE layer=?)
        """
        for (a, b), expected in C.EXPECTED_OVERLAP.items():
            got = self.conn.execute(q, (a, b)).fetchone()[0]
            mark = "ok " if got == expected else "!! "
            self.log(f"  {mark}{a} x {b}: {got} conditions (expected {expected})")
            if got != expected:
                problems.append(f"{a} x {b}: got {got}, expected {expected}")

        n = self.conn.execute("""
            SELECT COUNT(*) FROM profile p WHERE p.layer='transcriptome'
              AND p.condition_id IN (SELECT condition_id FROM profile WHERE layer='phenome')
        """).fetchone()[0]
        exp = C.EXPECTED_T_PROFILES_WITH_PHENOME
        mark = "ok " if n == exp else "!! "
        self.log(f"  {mark}transcriptome profiles with growth data: {n} (expected {exp})")
        if n != exp:
            problems.append(f"transcriptome profiles with phenome: {n} != {exp}")
        return problems

    # ----------------------------------------------------------------- run
    def run(self) -> dict:
        t0 = time.time()
        self.open()
        self.log("meta-data")
        self.load_strains()
        self.load_media()
        self.load_molecules()

        self.log("\nomics layers")
        genes, tpids, tmat = self.load_transcriptome()
        pmols, ppids, pmat = self.load_wide_csv(C.PROTEOME_CSV, "proteome")
        mmols, mpids, mmat = self.load_wide_csv(C.METABOLOME_CSV, "metabolome")
        rxns, fpids, fmat = self.load_fluxome()
        hpids, hmat = self.load_phenome()

        self.conn.execute(
            "UPDATE stress SET n_profiles = ("
            "  SELECT COUNT(*) FROM profile p JOIN condition c ON c.id=p.condition_id"
            "  WHERE c.stress = stress.name)")

        self.log("\nparquet mirrors")
        self.write_parquet("transcriptome", tpids, genes, tmat)
        self.write_parquet("proteome", ppids, pmols, pmat)
        self.write_parquet("metabolome", mpids, mmols, mmat)
        self.write_parquet("fluxome", fpids, rxns, fmat)
        self.write_parquet("phenome", hpids, ["lag_time", "growth_rate", "final_od"],
                           hmat)
        self.log(f"  wrote 5 files to {C.PARQUET_DIR}")

        self.stats["conditions"] = len(self._cond_id)
        for k, v in self.stats.items():
            self.conn.execute("INSERT OR REPLACE INTO build_info VALUES (?,?)",
                              (k, str(v)))
        self.conn.execute("INSERT OR REPLACE INTO build_info VALUES (?,?)",
                          ("built_utc", time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                      time.gmtime())))
        self.conn.commit()

        self.log(f"\nconditions (all layers): {len(self._cond_id)}")
        self.log("\ncross-layer overlap checks")
        problems = self.check()
        self.close()

        self.log(f"\nbuilt {self.db_path} in {time.time()-t0:.1f}s")
        if problems:
            raise AssertionError(
                "cross-layer overlaps do not match the paper:\n  "
                + "\n  ".join(problems)
                + "\n(this almost always means db/canon.py GP canonicalization broke)")
        return self.stats


def build(verbose: bool = True) -> dict:
    return Builder(verbose=verbose).run()


if __name__ == "__main__":
    sys.exit(0 if build() else 1)

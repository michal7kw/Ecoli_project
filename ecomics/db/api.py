"""Query interface over the built Ecomics database.

Two access paths, deliberately:

  * SQLite (`Ecomics.conn`) for provenance, integrity and ad-hoc questions
    -- "which conditions perturb b2741?", "what is in medium MD066?"
  * Parquet (`Ecomics.matrix`) for the wide numeric matrices the model trains
    on, because pulling 14.68 M rows out of the `measurement` table every time
    would dominate runtime.

Both are views of the same build; db/build.py writes them together.
"""

from __future__ import annotations

import functools
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ecomics import config as C
from ecomics.db.canon import ConditionKey, make_key, parse_transcriptome_cond


@functools.lru_cache(maxsize=1)
def lb_or_m9_media() -> frozenset[str]:
    """Medium IDs whose base medium is LB or M9, any carbon source.

    Half of the paper's wild-type definition (Supplementary Methods §3.3.3);
    see `Ecomics.wildtype_mask`. Read from the scraped medium ontology rather
    than the database because it is a property of the ontology, not of any
    build. Cached: the file is small but this is called per fold.
    """
    path = C.PROK_DIR / "medium.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path}\nThe paper's wild-type definition needs the medium "
            f"ontology. Run: python scripts/00_acquire.py")
    out = set()
    for m in json.loads(path.read_text(encoding="utf-8")):
        for field in ("Base Medium", "Description"):
            if (m.get(field) or "").strip().upper().startswith(("LB", "M9")):
                out.add(m["ID"])
                break
    return frozenset(out)

LAYERS = ("transcriptome", "proteome", "metabolome", "fluxome", "phenome")


@dataclass
class LayerMatrix:
    """A wide matrix for one layer, with aligned profile and condition labels."""

    layer: str
    values: np.ndarray            # (n_profiles, n_molecules), float32, NaN = missing
    columns: list[str]            # molecule / reaction / phenotype identifiers
    profile_ids: np.ndarray       # (n_profiles,)
    condition_keys: np.ndarray    # (n_profiles,) canonical key strings

    def __repr__(self) -> str:
        return (f"<LayerMatrix {self.layer} {self.values.shape[0]}x"
                f"{self.values.shape[1]} conds={len(set(self.condition_keys))}>")

    @property
    def n_profiles(self) -> int:
        return self.values.shape[0]

    def condition_index(self) -> dict[str, np.ndarray]:
        """condition key -> row indices belonging to it (the LOCO grouping)."""
        out: dict[str, list[int]] = {}
        for i, k in enumerate(self.condition_keys):
            out.setdefault(k, []).append(i)
        return {k: np.asarray(v) for k, v in out.items()}

    def averaged_by_condition(self) -> "LayerMatrix":
        """Collapse replicate profiles to one row per condition (nan-aware)."""
        idx = self.condition_index()
        keys = sorted(idx)
        vals = np.vstack([np.nanmean(self.values[idx[k]], axis=0) for k in keys])
        return LayerMatrix(self.layer, vals.astype(np.float32), self.columns,
                           np.arange(len(keys)), np.asarray(keys))

    def subset_conditions(self, keys) -> "LayerMatrix":
        keep = np.isin(self.condition_keys, np.asarray(list(keys)))
        return LayerMatrix(self.layer, self.values[keep], self.columns,
                           self.profile_ids[keep], self.condition_keys[keep])


class Ecomics:
    """Handle on the built compendium."""

    def __init__(self, db_path: Path = C.DB_PATH,
                 parquet_dir: Path = C.PARQUET_DIR):
        if not db_path.exists():
            raise FileNotFoundError(
                f"{db_path} not found -- run `python scripts/01_build_db.py` first")
        self.db_path = db_path
        self.parquet_dir = parquet_dir
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        # Per-instance memo for the expensive reads. See `matrix`.
        self._memo: dict = {}

    # ------------------------------------------------------------- matrices
    def matrix(self, layer: str) -> LayerMatrix:
        """Load one layer's wide matrix from Parquet.

        Memoized PER INSTANCE, in `self._memo`. It was `@functools.lru_cache`,
        which keys on `(self, layer)` and lives on the CLASS -- so the cache
        held a strong reference to every `Ecomics` ever constructed, and its
        SQLite connection, for the life of the process. `close()` could not
        release it. Worse for correctness than for memory: rebuild the database
        in the same process and this returned the pre-rebuild Parquet contents.
        """
        key = ("matrix", layer)
        if key in self._memo:
            return self._memo[key]
        if layer not in LAYERS:
            raise ValueError(f"unknown layer {layer!r}; expected one of {LAYERS}")
        df = pd.read_parquet(self.parquet_dir / f"{layer}.parquet")
        cols = [c for c in df.columns if c not in ("profile_id", "condition_key")]
        out = LayerMatrix(
            layer=layer,
            values=df[cols].to_numpy(dtype=np.float32),
            columns=cols,
            profile_ids=df["profile_id"].to_numpy(),
            condition_keys=df["condition_key"].to_numpy().astype(str),
        )
        self._memo[key] = out
        return out

    def growth_rate(self) -> dict[str, float]:
        """condition key -> maximum growth rate (h^-1)."""
        m = self.matrix("phenome")
        gr = m.values[:, m.columns.index("growth_rate")]
        return {k: float(v) for k, v in zip(m.condition_keys, gr) if np.isfinite(v)}

    # ----------------------------------------------------------- conditions
    def conditions(self, layer: str | None = None) -> list[str]:
        if layer is None:
            rows = self.conn.execute("SELECT key FROM condition ORDER BY key")
        else:
            rows = self.conn.execute(
                "SELECT DISTINCT c.key FROM condition c JOIN profile p "
                "ON p.condition_id=c.id WHERE p.layer=? ORDER BY c.key", (layer,))
        return [r[0] for r in rows]

    def shared_conditions(self, *layers: str) -> list[str]:
        """Conditions present in every named layer -- the cross-layer join."""
        if not layers:
            return []
        sets = [set(self.conditions(l)) for l in layers]
        return sorted(set.intersection(*sets))

    def aligned(self, *layers: str) -> tuple[list[str], dict]:
        """Condition-averaged matrices restricted to the conditions all `layers` share.

        Returns `(shared, {layer: (values, columns, condition_keys)})`, every
        layer's rows in the SAME order -- which is the whole point, because the
        caller then indexes two matrices with one fold's index array and a
        mismatch would be silent rather than an error.

        The sort is what guarantees that. `shared_conditions` returns a sorted
        list, but `subset_conditions` preserves each matrix's own row order, so
        two layers can hold the same conditions in different orders; sorting
        each by `condition_keys` puts them back in step.

        This lived as a private `_aligned` copy-pasted into `scripts/04`, `12`
        and `17`. The bodies were byte-identical and the docstrings had already
        drifted, which is the usual first sign. It belongs here rather than in
        `evaluate.py` because it is an operation on the data and adds no import
        edge -- `averaged_by_condition` and `subset_conditions` are already
        `LayerMatrix` methods.
        """
        shared = self.shared_conditions(*layers)
        out = {}
        for l in layers:
            m = self.matrix(l).averaged_by_condition().subset_conditions(shared)
            order = np.argsort(m.condition_keys)
            out[l] = (m.values[order], m.columns, m.condition_keys[order])
        return shared, out

    def condition_info(self, key: str) -> dict:
        row = self.conn.execute(
            "SELECT * FROM condition_layers WHERE key=?", (key,)).fetchone()
        if row is None:
            raise KeyError(key)
        d = dict(row)
        d["perturbations"] = [r[0] for r in self.conn.execute(
            "SELECT perturbation_id FROM condition_perturbation cp "
            "JOIN condition c ON c.id=cp.condition_id WHERE c.key=?", (key,))]
        return d

    def parse(self, cond: str) -> ConditionKey:
        """Parse a dotted condition string into a canonical key."""
        return parse_transcriptome_cond(cond)

    def key(self, strain: str, medium_id: str, stress: str, gp: str) -> str:
        return make_key(strain, medium_id, stress, gp).as_str()

    def wildtype_conditions(self, layer: str = "transcriptome") -> list[str]:
        return [r[0] for r in self.conn.execute(
            "SELECT DISTINCT c.key FROM condition c JOIN profile p "
            "ON p.condition_id=c.id WHERE p.layer=? AND c.is_wildtype=1", (layer,))]

    @staticmethod
    def wildtype_mask(keys, strict: bool = False) -> np.ndarray:
        """Boolean mask over condition keys: which count as wild type?

        TWO DEFINITIONS, and the difference is a factor of six.

        `strict=False` (default) is `gp == "none"` -- no genetic perturbation,
        any strain, any medium, any stress. That is what `canon.ConditionKey`
        means by wild type and what the database's `is_wildtype` column stores.
        On the transcriptome it selects **2,510 of 3,578 profiles (70.2%)**.

        `strict=True` is the paper's, from Supplementary Methods §3.3.3: "we
        define as WT (wild-type) profiles the ones of the **MG1655** strain in
        **LB or M9** medium, any carbon source, without any stresses or genetic
        perturbation." That selects roughly **12%**.

        Why it matters, and why both are kept. At 70% the "wild-type mean" is
        approximately the overall mean, so the wild-type and mean baselines
        collapse onto each other -- which is exactly what this reproduction
        observed, while the paper reports them separating by +0.10. The broad
        definition is not wrong, it just answers "is this strain unperturbed?"
        rather than "is this one of the paper's baseline reference profiles?".
        Anything compared against a number from the paper wants `strict=True`.
        """
        part = [str(k).split(".") for k in keys]
        if not strict:
            return np.array([len(p) > 3 and p[3] == "none" for p in part])
        media = lb_or_m9_media()
        return np.array([len(p) > 3 and p[0] == "MG1655" and p[1] in media
                         and p[2] in ("none", "na", "") and p[3] == "none"
                         for p in part])

    # ------------------------------------------------------------- metadata
    def strain_features(self) -> pd.DataFrame:
        """strains x features, 0/1 presence. Per-instance memo; see `matrix`."""
        if ("strain_features",) in self._memo:
            return self._memo[("strain_features",)]
        df = pd.read_sql(
            "SELECT strain, feature, present FROM strain_feature", self.conn)
        out = df.pivot(index="strain", columns="feature",
                       values="present").fillna(0).astype(np.float32)
        self._memo[("strain_features",)] = out
        return out

    def medium_components(self, kind: str = "present") -> pd.DataFrame:
        """media x components. kind='present' (0/1) or 'amount'. See `matrix`."""
        if ("medium_components", kind) in self._memo:
            return self._memo[("medium_components", kind)]
        col = "present" if kind == "present" else "amount"
        df = pd.read_sql(
            f"SELECT medium_id, component, {col} AS v FROM medium_component",
            self.conn)
        out = df.pivot(index="medium_id", columns="component",
                       values="v").astype(np.float32)
        self._memo[("medium_components", kind)] = out
        return out

    def stresses(self) -> list[str]:
        return [r[0] for r in self.conn.execute(
            "SELECT name FROM stress ORDER BY name")]

    def perturbations(self) -> list[str]:
        return [r[0] for r in self.conn.execute(
            "SELECT id FROM perturbation ORDER BY id")]

    def reaction_bigg(self) -> dict[str, str]:
        """R0001..R0120 -> BiGG reaction id, where one is published."""
        return {r[0]: r[1] for r in self.conn.execute(
            "SELECT id, bigg FROM reaction WHERE bigg IS NOT NULL AND bigg != 'na'")}

    # -------------------------------------------------------------- summary
    def summary(self) -> pd.DataFrame:
        return pd.read_sql("SELECT * FROM layer_summary ORDER BY layer", self.conn)

    def build_info(self) -> dict[str, str]:
        return {r[0]: r[1] for r in self.conn.execute("SELECT * FROM build_info")}

    def close(self) -> None:
        self._memo.clear()
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


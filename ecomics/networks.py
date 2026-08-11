"""Interaction networks -> functional-neighbour sets, for the proteome module.

The paper predicts each protein's level from the expression of the genes that
are FUNCTIONALLY RELATED to it, using four independent notions of "related"
(paper.md:170):

    TRN   RegulonDB       genes connected by a regulatory link
    PPI   Bacteriome/STRING  physically interacting proteins
    CPN   co-expression   built here, edges where pairwise r > 0.7, from
                          proteome profiles HELD OUT of evaluation
    KEGG  pathways        genes implicated in the same pathway

Why four, and why these four
----------------------------
Averaging predictors reduces error variance only in proportion to how
DECORRELATED they are: Var(mean) = rho*sigma^2 + (1-rho)*sigma^2/M. Four views
of the same regulatory network would share errors (rho near 1) and buy nothing.
These four encode genuinely different biology -- co-regulation, physical
complex membership, empirical co-variation, pathway co-membership -- so they
fail on different proteins.

They also differ in COVERAGE, which is a separate benefit: in the paper TRN
reaches only 250 proteins, KEGG 547, CPN 847, PPI 1,000, but their union
covers all 1,001.
"""

from __future__ import annotations

import csv
import gzip
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ecomics import config as C

__all__ = ["Network", "load_trn", "load_ppi", "load_kegg", "build_cpn",
           "load_all_networks", "gene_symbol_map"]


@dataclass
class Network:
    """An undirected neighbour map over gene/protein b-numbers."""

    name: str
    neighbours: dict[str, set[str]]

    @property
    def n_nodes(self) -> int:
        return len(self.neighbours)

    @property
    def n_edges(self) -> int:
        return sum(len(v) for v in self.neighbours.values()) // 2

    def of(self, node: str, exclude_self: bool = True) -> list[str]:
        nb = self.neighbours.get(node, set())
        return sorted(nb - {node}) if exclude_self else sorted(nb)

    def restrict(self, allowed: set[str]) -> "Network":
        """Keep only nodes and neighbours present in `allowed`."""
        out = {n: (v & allowed) for n, v in self.neighbours.items() if n in allowed}
        return Network(self.name, {k: v for k, v in out.items() if v})

    def __repr__(self) -> str:
        return f"<Network {self.name}: {self.n_nodes} nodes, {self.n_edges} edges>"


def _add(d: dict[str, set[str]], a: str, b: str) -> None:
    if a == b:
        return
    d.setdefault(a, set()).add(b)
    d.setdefault(b, set()).add(a)


# --------------------------------------------------------------------------
def gene_symbol_map() -> dict[str, str]:
    """Gene symbol (lower-case) -> b-number, from the KEGG eco gene list.

    Needed because the TRN names regulators by symbol (rpoD, crp, arcA) while
    everything else in Ecomics is keyed by b-number.
    """
    path = C.REMOTE_FILES.get("kegg_gene_list")
    out: dict[str, str] = {}
    if path is None or not path.exists():
        return out
    # KEGG's /list/eco is 4 columns, and the gene symbol lives in the LAST one:
    #   eco:b0001 <TAB> CDS <TAB> 190..255 <TAB> thrL; thr operon leader peptide
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 2:
            continue
        b = parts[0].split(":")[-1].strip()
        if not re.fullmatch(r"b\d{4}", b):
            continue
        desc = parts[-1]
        for sym in desc.split(";")[0].split(","):
            sym = sym.strip().lower()
            if sym and " " not in sym:
                out.setdefault(sym, b)
    return out


def load_trn(path: Path | None = None) -> Network:
    """Transcriptional regulatory network, from the RegulonDB-derived TRN.

    Regulator symbols are mapped onto b-numbers; regulators that are effector
    METABOLITES rather than proteins (FMN, L-tryptophan, TPP) have no b-number
    and are dropped, which is correct -- they are not genes whose expression
    could serve as a predictor.
    """
    path = path or C.REMOTE_FILES["regulondb_tf_gene"]
    nb: dict[str, set[str]] = {}
    if not path.exists():
        return Network("TRN", nb)
    sym2b = gene_symbol_map()
    with path.open(encoding="utf-8", errors="replace") as fh:
        for row in csv.DictReader(fh):
            reg = (row.get("regulator") or "").strip()
            tgt = (row.get("gene_id") or "").strip()
            if not re.fullmatch(r"b\d{4}", tgt):
                continue
            reg_b = reg if re.fullmatch(r"b\d{4}", reg) else sym2b.get(reg.lower())
            if reg_b:
                _add(nb, reg_b, tgt)
    return Network("TRN", nb)


def load_ppi(min_score: int = 700, path: Path | None = None,
             info_path: Path | None = None) -> Network:
    """Protein-protein interactions from STRING.

    min_score 700 is STRING's "high confidence" cutoff. The full network at
    score 0 is essentially complete and would make every protein everyone's
    neighbour, which defeats the purpose of a neighbourhood.
    """
    path = path or C.REMOTE_FILES["string_links"]
    info_path = info_path or C.REMOTE_FILES["string_info"]
    nb: dict[str, set[str]] = {}
    if not path.exists() or not info_path.exists():
        return Network("PPI", nb)

    # STRING protein id -> b-number, via the preferred name and the annotation
    sym2b = gene_symbol_map()
    sid2b: dict[str, str] = {}
    with gzip.open(info_path, "rt", encoding="utf-8", errors="replace") as fh:
        next(fh, None)
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            sid, pref = parts[0], parts[1].strip()
            m = re.search(r"(b\d{4})", " ".join(parts))
            if m:
                sid2b[sid] = m.group(1)
            elif pref.lower() in sym2b:
                sid2b[sid] = sym2b[pref.lower()]

    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
        next(fh, None)
        for line in fh:
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                score = int(parts[2])
            except ValueError:
                continue
            if score < min_score:
                continue
            a, b = sid2b.get(parts[0]), sid2b.get(parts[1])
            if a and b:
                _add(nb, a, b)
    return Network("PPI", nb)


def load_kegg(path: Path | None = None, max_pathway_size: int = 200) -> Network:
    """Pathway co-membership from KEGG.

    Very large pathways (metabolic maps with hundreds of genes) are skipped:
    they connect nearly everything to everything and add noise rather than
    functional specificity.
    """
    path = path or C.REMOTE_FILES["kegg_pathway_gene"]
    nb: dict[str, set[str]] = {}
    if not path.exists():
        return Network("KEGG", nb)

    members: dict[str, list[str]] = defaultdict(list)
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        pw, gene = parts[0].strip(), parts[1].split(":")[-1].strip()
        if re.fullmatch(r"b\d{4}", gene):
            members[pw].append(gene)

    for pw, genes in members.items():
        if len(genes) > max_pathway_size or len(genes) < 2:
            continue
        for i, a in enumerate(genes):
            for b in genes[i + 1:]:
                _add(nb, a, b)
    return Network("KEGG", nb)


def build_cpn(values: np.ndarray, columns: list[str],
              threshold: float = C.PAPER["cpn_corr_threshold"]) -> Network:
    """Co-expression network: an edge wherever pairwise correlation > threshold.

    IMPORTANT: build this from profiles that are HELD OUT of proteome
    evaluation. The paper is explicit that its CPN comes from 20 profiles "not
    used for proteome prediction" (paper.md:170) -- otherwise the graph encodes
    the correlations of the very data it is scored on, and the predictor is
    partly memorizing.

    SIGNED, not absolute. Supplementary Methods §3.3.4: "For two proteins to be
    considered co-expressed, their pairwise correlation should be **larger
    than** 0.7." This used `np.abs(corr) > threshold`, contradicting both the
    Methods and this function's own first line. Including anti-correlated pairs
    is defensible as a predictor -- a strong negative correlation is just as
    informative to a LASSO -- but it is a different graph from the paper's, with
    more edges and higher node degree, so the coverage counts it produces are
    not comparable to the paper's 847.
    """
    v = np.asarray(values, float)
    ok = np.isfinite(v).all(axis=0)
    v, cols = v[:, ok], [c for c, k in zip(columns, ok) if k]
    if v.shape[0] < 3 or v.shape[1] < 2:
        return Network("CPN", {})

    sd = v.std(axis=0)
    keep = sd > 1e-9
    v, cols = v[:, keep], [c for c, k in zip(cols, keep) if k]
    corr = np.corrcoef(v, rowvar=False)

    nb: dict[str, set[str]] = {}
    idx = np.argwhere(np.triu(corr > threshold, k=1))
    for i, j in idx:
        _add(nb, cols[i], cols[j])
    return Network("CPN", nb)


def load_all_networks(cpn_values: np.ndarray | None = None,
                      cpn_columns: list[str] | None = None,
                      verbose: bool = True, strict: bool = True
                      ) -> dict[str, Network]:
    """Load TRN, PPI, KEGG and (optionally) build the CPN.

    strict  raise if any network came back EMPTY. Default on, because the
            failure it catches is silent and expensive.

            Each loader returns `Network(name, {})` when its file is missing, so
            a failed download degrades the four-network ensemble to three, two
            or one with nothing raised: `ProteomeEnsemble.fit` simply builds no
            models for that network and `nanmean` averages over the rest. The
            paper's central proteome claim -- that no single network covers
            every protein but the union does (TRN 250 / KEGG 547 / CPN 847 /
            PPI 1000 / union 1001) -- then becomes unreproducible without any
            error. It is made worse by `acquire`: RegulonDB's app shell is a
            1,653-byte HTTP 200 that passes the download size check, so
            "TRN 0 nodes" is a realistic state, not a hypothetical one.
    """
    nets = {"TRN": load_trn(), "PPI": load_ppi(), "KEGG": load_kegg()}
    if cpn_values is not None and cpn_columns is not None:
        nets["CPN"] = build_cpn(cpn_values, cpn_columns)
    if verbose:
        for n in nets.values():
            print(f"  {n.name:<5s} {n.n_nodes:>5d} nodes  {n.n_edges:>8d} edges")
    empty = [n.name for n in nets.values() if n.n_edges == 0]
    if strict and empty:
        raise ValueError(
            f"network(s) loaded with zero edges: {', '.join(empty)}. "
            f"The file is missing or was downloaded as an error page; "
            f"run `python scripts/00_acquire.py --verify`. "
            f"Pass strict=False to proceed with a reduced ensemble.")
    return nets

"""The graph primitive, the co-expression network, and the symbol map.

This module holds what every graph in the repository is made of. It does NOT
load interaction networks -- `ecomics/networks_paper.py` does, from the paper's
own Supplementary Data 2.

    Network            an undirected neighbour map over b-numbers
    _add               the one insert: drops self-loops, keeps the map symmetric
    gene_symbol_map    gene SYMBOL -> b-number, from the KEGG eco gene list
    build_cpn          the co-expression network, from HELD-OUT profiles

⚠ Why this module does not scrape its graphs
--------------------------------------------
The TRN, PPI and KEGG graphs could be fetched from a current RegulonDB snapshot,
STRING v12 at score >= 700, and the KEGG pathway REST endpoint. Building them
that way was measured against the paper's own edge lists from Supplementary
Data 2, and lost by **+0.113** per-profile PCC (0.152 -> 0.264).

The instructive part is *why*: the scraped TRN was **larger** than the paper's
-- 7,450 edges against 3,190 -- and scored **0.009 against 0.154**. A 2026
snapshot of a curated database is not a better version of the 2016 one. It is a
different graph, and for these five conditions a worse one. More edges is not
more signal.

`gene_symbol_map` is not proteome-specific and is unaffected: `db/build.py`
asserts against it when normalizing perturbations to b-numbers,
`paper_protocol.py` needs it for the paper's TF subset, and `networks_paper.py`
uses it to repair Data 2's handful of non-b-number tokens.

Why several graphs rather than one
----------------------------------
Averaging predictors reduces error variance only in proportion to how
DECORRELATED they are: Var(mean) = rho*sigma^2 + (1-rho)*sigma^2/M. Several
views of the same regulatory network would share errors (rho near 1) and buy
nothing. The paper's six encode genuinely different biology -- co-regulation,
physical complex membership, empirical co-variation, pathway co-membership,
sigma-factor control, small-RNA regulation -- so they fail on different
proteins.

They also differ in COVERAGE, which is a separate benefit: in the paper TRN
reaches only 250 proteins, KEGG 547, CPN 847, PPI 1,000, but their union
covers all 1,001.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from ecomics import config as C

__all__ = ["Network", "build_cpn", "gene_symbol_map"]


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

    The one identifier bridge in the repository. Ecomics keys everything by
    b-number, but three sources do not: the fluxome writes perturbations as
    uppercase SYMBOLS, the paper's TF list names transcription factors by
    symbol, and Supplementary Data 2 carries a handful of symbol tokens among
    its b-number pairs. Callers, in rough order of how much depends on them:

        db/build.py         normalizes perturbations to b-numbers, and ASSERTS
                            this map is non-empty -- an empty one silently
                            collapses three cross-layer joins
        paper_protocol.py   the paper's 178-TF subset
        networks_paper.py   repairs `citB`, `ydbA_2` and friends in Data 2
        scripts/03        the TF subset reported alongside the all-gene result
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



"""RNA-Seq: the alignment pipeline, and the count-table entrypoint.

Paper (paper.md:134):

    "The publicly available RNA-Seq data were downloaded from Sequence Read
     Archive, first converted to fastq using fastq-dump and then processed to
     follow the process (that is, Trimmomatic, TopHat/bowtie2, htseq-count)
     below. The low qualities of raw reads were trimmed using Trimmomatic
     (v0.30) with default settings. Trimmed reads were aligned on the most
     recent reference genome of E. coli K-12 MG1655 (GenBank: U00096.3) by
     using TopHat (v2.0.10) coupled with bowtie (v1.0.0). The resulting SAM
     file is then processed to have gene-level read counts using htseq-count."

Two entrypoints:

  run_alignment()   the full FASTQ -> counts path. Requires external binaries
                    (trimmomatic, tophat/bowtie, htseq-count, samtools) and a
                    K-12 index. Runs only when they are present; raises a clear
                    error naming what is missing otherwise.
  read_htseq()      reads an htseq-count table directly. This is the documented
                    alternative and the one exercised here, because GSE73673 --
                    the paper's OWN 16-knockout experiment -- is published as
                    .htcount.txt files, i.e. the exact output of the pipeline
                    above. Starting from those is not a shortcut around the
                    paper's method; it is the paper's method's output.

A note on TopHat
----------------
TopHat is a SPLICE-AWARE aligner and E. coli has no introns, so its junction
discovery is inert here and plain bowtie would do. Harmless, and preserved for
fidelity, but worth knowing when reading the original Methods.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = ["read_htseq", "read_htseq_dir", "counts_to_cpm", "check_tools",
           "run_alignment", "HTSeqCounts"]

# htseq-count appends these summary rows after the per-gene counts.
HTSEQ_SPECIAL = ("__no_feature", "__ambiguous", "__too_low_aQual",
                 "__not_aligned", "__alignment_not_unique")

REQUIRED_TOOLS = {
    "trimmomatic": "read trimming (Trimmomatic v0.30)",
    "tophat": "spliced alignment (TopHat v2.0.10)",
    "bowtie": "alignment backend (bowtie v1.0.0)",
    "samtools": "SAM/BAM handling",
    "htseq-count": "gene-level counting",
}


@dataclass
class HTSeqCounts:
    """One htseq-count table."""

    sample: str
    genes: list[str]
    counts: np.ndarray
    summary: dict[str, int]

    @property
    def total_assigned(self) -> int:
        return int(self.counts.sum())

    @property
    def assignment_rate(self) -> float:
        tot = self.total_assigned + sum(self.summary.values())
        return self.total_assigned / tot if tot else float("nan")


def read_htseq(path: str | Path) -> HTSeqCounts:
    """Read one htseq-count output file (two columns: feature, count)."""
    path = Path(path)
    genes, counts, summary = [], [], {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        name, val = parts[0].strip(), parts[1].strip()
        try:
            n = int(float(val))
        except ValueError:
            continue
        if name.startswith("__"):
            summary[name] = n
        else:
            genes.append(name)
            counts.append(n)
    # GEO prefixes sample files with their GSM id: GSM1900470_WT_LB_1.htcount.txt
    sample = path.name.split(".")[0]
    return HTSeqCounts(sample, genes, np.asarray(counts, dtype=np.float64), summary)


def read_htseq_dir(directory: str | Path, pattern: str = "*.htcount.txt"
                   ) -> tuple[list[str], list[str], np.ndarray]:
    """Read a directory of htseq tables into a (genes x samples) matrix.

    Returns (gene ids, sample names, counts). Genes are the intersection across
    samples, in sorted order, so the matrix is rectangular and aligned.
    """
    files = sorted(Path(directory).glob(pattern))
    if not files:
        raise FileNotFoundError(f"no files matching {pattern} in {directory}")
    tables = [read_htseq(f) for f in files]

    common = sorted(set.intersection(*(set(t.genes) for t in tables)))
    idx = [{g: i for i, g in enumerate(t.genes)} for t in tables]
    mat = np.vstack([t.counts[[ix[g] for g in common]]
                     for t, ix in zip(tables, idx)]).T
    return common, [t.sample for t in tables], mat


def counts_to_cpm(counts: np.ndarray, log: bool = True,
                  prior: float = 1.0) -> np.ndarray:
    """Counts -> counts per million, optionally log2.

    Library-size scaling only. This is deliberately minimal: cross-platform
    reconciliation is the job of pipeline/platform.py, and putting a second
    normalization here would double-correct.
    """
    counts = np.asarray(counts, dtype=np.float64)
    lib = counts.sum(axis=0, keepdims=True)
    lib = np.where(lib > 0, lib, 1.0)
    cpm = counts / lib * 1e6
    return np.log2(cpm + prior) if log else cpm


def check_tools() -> dict[str, str | None]:
    """Which external aligner tools are available on PATH."""
    return {t: shutil.which(t) for t in REQUIRED_TOOLS}


def run_alignment(fastq: str | Path, out_dir: str | Path, index: str | Path,
                  gtf: str | Path, threads: int = 4) -> Path:
    """FASTQ -> gene counts, following the paper's tool chain.

    Raises RuntimeError naming the missing binaries if the environment cannot
    run it, rather than silently degrading -- an alignment that did not happen
    should never look like one that did.
    """
    missing = [t for t, p in check_tools().items() if p is None]
    if missing:
        raise RuntimeError(
            "cannot run the alignment pipeline; missing: "
            + ", ".join(f"{t} ({REQUIRED_TOOLS[t]})" for t in missing)
            + ".\nUse read_htseq()/read_htseq_dir() on published count tables "
              "instead -- GSE73673 ships the paper's own samples that way.")

    fastq, out_dir = Path(fastq), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trimmed = out_dir / (fastq.stem + ".trimmed.fastq")
    bam = out_dir / "accepted_hits.bam"
    counts = out_dir / (fastq.stem + ".htcount.txt")

    subprocess.run(["trimmomatic", "SE", "-threads", str(threads),
                    str(fastq), str(trimmed),
                    "LEADING:3", "TRAILING:3", "SLIDINGWINDOW:4:15", "MINLEN:36"],
                   check=True)
    subprocess.run(["tophat", "-p", str(threads), "-o", str(out_dir),
                    str(index), str(trimmed)], check=True)
    with open(counts, "w") as fh:
        subprocess.run(["htseq-count", "-f", "bam", "-s", "no",
                        str(bam), str(gtf)], check=True, stdout=fh)
    return counts

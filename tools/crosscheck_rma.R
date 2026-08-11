#!/usr/bin/env Rscript
# Ground-truth RMA via Bioconductor `affy`, for cross-checking the pure-Python
# implementation in ecomics/pipeline/arrays.py.
#
# This is a VALIDATION tool, not part of the pipeline. The replication runs
# entirely in Python; R exists here only to provide an independent answer.
#
# Usage: Rscript tools/crosscheck_rma.R <cel_dir> <out.tsv>

suppressPackageStartupMessages({
  library(affy)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) stop("usage: crosscheck_rma.R <cel_dir> <out.tsv>")
cel_dir <- args[1]
out <- args[2]

files <- list.files(cel_dir, pattern = "\\.CEL$", full.names = TRUE,
                    ignore.case = TRUE)
if (!length(files)) stop("no CEL files in ", cel_dir)
cat(sprintf("reading %d CEL file(s)\n", length(files)))

ab <- ReadAffy(filenames = files)
cat(sprintf("chip type: %s\n", annotation(ab)))

es <- affy::rma(ab)
m <- exprs(es)
cat(sprintf("affy rma: %d probe sets x %d arrays\n", nrow(m), ncol(m)))

df <- data.frame(probeset = rownames(m), m, check.names = FALSE)
write.table(df, out, sep = "\t", quote = FALSE, row.names = FALSE)
cat(sprintf("wrote %s\n", out))

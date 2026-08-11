#!/usr/bin/env Rscript
# Export an Affymetrix CDF probe-set layout to a plain TSV.
#
# RMA's third step -- median-polish summarization -- needs to know which cells
# of the array belong to which probe set, and which are PM vs MM. That mapping
# lives in a Bioconductor CDF package as R-serialized data, which Python cannot
# read directly.
#
# This script is therefore an ACQUISITION step, not a runtime dependency: it is
# run once to dump the layout, after which the whole pipeline is pure Python.
#
# Usage:  Rscript tools/export_cdf.R <cdfpkg> <outfile.tsv> [ncol]
#   e.g.  Rscript tools/export_cdf.R ecoliasv2cdf data/external/raw/cdf/ecoliasv2.tsv 544
#
# `ncol` is the array width in cells. It defaults to 544 (Affymetrix E. coli
# Antisense v2, GPL199) and is taken as an argument rather than read from the
# package's <pkg>dim object, which is a lazy-load stub that cannot be coerced
# outside AnnotationDbi. The CEL header carries the same value (Cols=), so the
# Python reader cross-checks it -- see pipeline/arrays.py.
#
# Output columns:
#   probeset   probe set identifier, e.g. "aas_b2836_at"
#   type       "pm" or "mm"
#   index      1-based cell index into the CEL intensity vector (column-major,
#              i.e. index = x + y * ncol + 1), matching affy's own convention
#   x, y       cell coordinates, 0-based

suppressPackageStartupMessages({
  library(methods)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) stop("usage: export_cdf.R <cdfpkg> <outfile.tsv> [ncol]")
pkg <- args[1]
out <- args[2]
ncol_arr <- if (length(args) >= 3) as.integer(args[3]) else 544L

if (!requireNamespace(pkg, quietly = TRUE))
  stop(sprintf("CDF package '%s' is not installed", pkg))
suppressPackageStartupMessages(library(pkg, character.only = TRUE))

env <- get(pkg, envir = asNamespace(pkg))
ids <- ls(env)
cat(sprintf("%s: %d probe sets, array width %d cells\n", pkg, length(ids), ncol_arr))

rows <- vector("list", length(ids))
for (i in seq_along(ids)) {
  m <- get(ids[i], envir = env)     # matrix with columns 'pm' and 'mm'
  pm <- m[, "pm"]
  mm <- if ("mm" %in% colnames(m)) m[, "mm"] else rep(NA_integer_, length(pm))
  idx <- c(pm, mm)
  typ <- c(rep("pm", length(pm)), rep("mm", length(mm)))
  keep <- !is.na(idx)
  rows[[i]] <- data.frame(
    probeset = ids[i],
    type     = typ[keep],
    index    = idx[keep],
    stringsAsFactors = FALSE
  )
}
df <- do.call(rbind, rows)

# affy indices are 1-based into the column-major cell vector.
df$x <- (df$index - 1L) %% ncol_arr
df$y <- (df$index - 1L) %/% ncol_arr

dir.create(dirname(out), recursive = TRUE, showWarnings = FALSE)
write.table(df, out, sep = "\t", quote = FALSE, row.names = FALSE)
cat(sprintf("wrote %s: %d rows (%d pm, %d mm)\n",
            out, nrow(df), sum(df$type == "pm"), sum(df$type == "mm")))

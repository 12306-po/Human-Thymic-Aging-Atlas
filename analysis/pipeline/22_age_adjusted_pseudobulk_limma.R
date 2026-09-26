#!/usr/bin/env Rscript
# R2 (figure audit 2026-09-16, section 6.2): donor-level pseudobulk age model.
#
# Formal donor-level inference replacing unadjusted mixture-level Spearman as
# the PRIMARY age-association analysis:
#     voom expression ~ age_z + sex,  n = 18 donors
#
# All 18 donors share platform GPL24676 and a single preparation group, so no
# batch term is estimable; this is recorded in the design-check output.
#
# Inputs:
#   02_pseudobulk/donor_matrix/raw_combined.csv.gz         (whole thymus)
#   02_pseudobulk/celltype_matrix/raw_counts/donor*_<CT>.csv.gz
#   01_raw_processing/metadata/05B_final_regression_cohort.csv
# Output: 03_feature_selection/limma_adjusted/

suppressPackageStartupMessages({
  library(limma)
  library(edgeR)
})

args <- commandArgs(trailingOnly = FALSE)
proj <- Sys.getenv("PROJ", "/data/zxy/projects/human_thymus_age_ML_DL")
out <- file.path(proj, "03_feature_selection", "limma_adjusted")
dir.create(out, recursive = TRUE, showWarnings = FALSE)

meta <- read.csv(file.path(proj, "01_raw_processing", "metadata",
                           "05B_final_regression_cohort.csv"),
                 stringsAsFactors = FALSE)
meta$age_z <- as.numeric(scale(meta$age_years))
meta$sex <- factor(meta$sex)

# ---- design diagnostics (rank, age-sex confounding, batch unestimable) ----
sink(file.path(out, "limma_age_sex_design_check.txt"))
cat("n donors:", nrow(meta), "\n")
cat("table(sex):\n"); print(table(meta$sex))
cat("Spearman age vs sex (Male=1):",
    suppressWarnings(cor(meta$age_years, as.numeric(meta$sex == "Male"),
                         method = "spearman")), "\n")
cat("mean age by sex:\n"); print(aggregate(age_years ~ sex, meta, mean))
cat("platform/preparation: single level across all 18 donors (GPL24676;",
    "combined_CD45pos_CD45neg_6to4) -> batch term not estimable\n")
sink()

fit_one <- function(counts, donors, tag, min_donors = 10) {
  m <- match(donors, meta$donor_id)
  md <- meta[m[!is.na(m)], ]
  counts <- counts[!is.na(m), , drop = FALSE]
  n_d <- nrow(counts)
  if (n_d < min_donors) {
    message(tag, ": only ", n_d, " donors -> skipped (<", min_donors, ")")
    return(invisible(NULL))
  }
  counts <- round(as.matrix(counts))
  storage.mode(counts) <- "integer"
  design <- model.matrix(~ age_z + sex, data = md)
  if (qr(design)$rank < ncol(design)) {
    message(tag, ": design rank deficient -> skipped"); return(invisible(NULL))
  }
  y <- DGEList(counts = t(counts))
  keep <- filterByExpr(y, design)
  y <- y[keep, , keep.lib.sizes = FALSE]
  y <- calcNormFactors(y, method = "TMM")
  v <- voom(y, design, plot = FALSE)
  fit <- eBayes(lmFit(v, design))
  tt <- topTable(fit, coef = "age_z", number = Inf, sort.by = "P")
  res <- data.frame(
    gene = rownames(tt),
    logFC_age_perSD = tt$logFC,
    AveExpr = tt$AveExpr,
    t = tt$t,
    P.Value = tt$P.Value,
    adj.P.Val = tt$adj.P.Val,
    n_donors = n_d,
    row.names = NULL)
  write.csv(res, file.path(out, paste0(tag, "_limma_age_sex.csv")),
            row.names = FALSE)
  data.frame(set = tag, n_donors = n_d, n_genes_tested = nrow(tt),
             n_sig_q05 = sum(tt$adj.P.Val < 0.05),
             n_up = sum(tt$adj.P.Val < 0.05 & tt$logFC > 0),
             n_down = sum(tt$adj.P.Val < 0.05 & tt$logFC < 0))
}

cov <- list()

# ---- whole thymus ----
wt <- read.csv(gzfile(file.path(
  proj, "02_pseudobulk", "donor_matrix", "raw_combined.csv.gz")),
  check.names = FALSE)
rownames(wt) <- wt[[1]]; wt[[1]] <- NULL
cov[[length(cov) + 1]] <- fit_one(wt, rownames(wt), "whole_thymus")

# ---- per cell type (only donor x CT files passing min_cells=20 exist) ----
rd <- file.path(proj, "02_pseudobulk", "celltype_matrix", "raw_counts")
files <- list.files(rd, pattern = "^donor\\d+_.+\\.csv\\.gz$")
tags <- sub("^donor\\d+_(.+)\\.csv\\.gz$", "\\1", files)
for (ct in sort(unique(tags))) {
  ff <- file.path(rd, files[tags == ct])
  tabs <- lapply(ff, function(f) {
    d <- read.csv(gzfile(f))
    setNames(d$count, d$gene)
  })
  donors_ct <- sub("^(donor\\d+)_.+$", "\\1", basename(ff))
  genes <- sort(unique(unlist(lapply(tabs, names))))
  mat <- matrix(0L, nrow = length(tabs), ncol = length(genes),
                dimnames = list(donors_ct, genes))
  for (i in seq_along(tabs)) mat[i, names(tabs[[i]])] <- tabs[[i]]
  cov[[length(cov) + 1]] <- fit_one(as.data.frame(mat), donors_ct, ct)
}

cov <- do.call(rbind, cov)
write.csv(cov, file.path(out, "limma_celltype_coverage.csv"), row.names = FALSE)
message("\nCoverage / significance summary:\n")
print(cov)
message("=== R2 limma COMPLETE ===")

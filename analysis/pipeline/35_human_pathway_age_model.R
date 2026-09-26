#!/usr/bin/env Rscript
# Step 35: signed donor-level Hallmark pathway age analysis using camera.
# This replaces unsigned candidate-list over-representation when a direction is
# required for cross-dataset pathway comparison.

suppressPackageStartupMessages({
  library(limma)
  library(edgeR)
})

args <- commandArgs(trailingOnly = TRUE)
project <- Sys.getenv("PROJ", unset = "/data/zxy/projects/human_thymus_age_ML_DL")
gmt <- Sys.getenv("MSIGDB_HALLMARK_GMT", unset = "")
if (length(args)) {
  i <- 1
  while (i <= length(args)) {
    if (args[i] == "--project" && i < length(args)) {
      project <- args[i + 1]; i <- i + 2
    } else if (args[i] == "--gmt" && i < length(args)) {
      gmt <- args[i + 1]; i <- i + 2
    } else {
      stop("Unknown or incomplete argument: ", args[i])
    }
  }
}
if (!nzchar(gmt) || !file.exists(gmt)) {
  stop("Hallmark GMT is required. Set MSIGDB_HALLMARK_GMT or pass --gmt FILE")
}

meta_file <- file.path(project, "01_raw_processing", "metadata",
                       "05B_final_regression_cohort.csv")
count_file <- file.path(project, "02_pseudobulk", "donor_matrix",
                        "raw_combined.csv.gz")
out <- file.path(project, "03_feature_selection")
dir.create(out, recursive = TRUE, showWarnings = FALSE)

meta <- read.csv(meta_file, stringsAsFactors = FALSE)
meta$donor_id <- as.character(meta$donor_id)
meta$age_z <- as.numeric(scale(meta$age_years))
meta$sex <- factor(meta$sex)
counts <- read.csv(gzfile(count_file), check.names = FALSE)
rownames(counts) <- as.character(counts[[1]])
counts[[1]] <- NULL
counts <- counts[meta$donor_id, , drop = FALSE]

design <- model.matrix(~ age_z + sex, data = meta)
if (qr(design)$rank < ncol(design)) stop("Age+sex design is rank deficient")
y <- DGEList(counts = t(round(as.matrix(counts))))
keep <- filterByExpr(y, design)
y <- y[keep, , keep.lib.sizes = FALSE]
y <- calcNormFactors(y, method = "TMM")
v <- voom(y, design, plot = FALSE)
fit <- eBayes(lmFit(v, design))
age_beta <- fit$coefficients[, "age_z"]

gmt_lines <- readLines(gmt, warn = FALSE)
gene_sets <- lapply(gmt_lines, function(line) {
  fields <- strsplit(line, "\t", fixed = FALSE)[[1]]
  if (length(fields) < 3) return(NULL)
  list(name = fields[1], description = fields[2], genes = unique(fields[-c(1, 2)]))
})
gene_sets <- Filter(Negate(is.null), gene_sets)
names(gene_sets) <- vapply(gene_sets, `[[`, character(1), "name")
index <- lapply(gene_sets, function(entry) which(rownames(v) %in% entry$genes))
index <- index[vapply(index, length, integer(1)) >= 10]
if (!length(index)) stop("No Hallmark set has at least 10 tested genes")

camera_result <- camera(v, index = index, design = design,
                        contrast = which(colnames(design) == "age_z"))
camera_result$program_id <- rownames(camera_result)
rows <- lapply(camera_result$program_id, function(program) {
  genes <- rownames(v)[index[[program]]]
  beta <- age_beta[genes]
  camera_row <- camera_result[program, ]
  data.frame(
    program_id = program,
    n_genes_tested = length(genes),
    effect_age = median(beta, na.rm = TRUE),
    mean_gene_beta_age = mean(beta, na.rm = TRUE),
    direction = as.character(camera_row$Direction),
    p_value = as.numeric(camera_row$PValue),
    q_value = as.numeric(camera_row$FDR),
    model = "camera competitive gene-set test on sex-adjusted donor-level voom expression",
    stringsAsFactors = FALSE
  )
})
result <- do.call(rbind, rows)
result <- result[order(result$q_value, result$p_value), ]
write.table(result, file.path(out, "program_age_effects.tsv"),
            sep = "\t", row.names = FALSE, quote = FALSE, na = "NA")

gmt_sha <- unname(tools::md5sum(gmt))
writeLines(c(
  paste0("GMT: ", normalizePath(gmt)),
  paste0("GMT_MD5: ", gmt_sha),
  "analysis unit: donor",
  "model: voom expression ~ standardized age + sex",
  "pathway test: limma camera competitive gene-set test",
  "effect_age: median member-gene beta_age; direction confirmed by camera Direction"
), file.path(out, "program_age_effects_contract.txt"))
cat("Signed pathway age results written to",
    file.path(out, "program_age_effects.tsv"), "\n")

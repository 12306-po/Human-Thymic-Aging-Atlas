#!/usr/bin/env Rscript
# Step 29: donor influence and age-shape sensitivity for whole-thymus expression.
# Primary inference remains Step 22 (sex-adjusted limma-voom, all 18 donors).
# These reruns quantify robustness to influential donors, age restriction,
# female-only analysis, and a quadratic age term.

suppressPackageStartupMessages({
  library(limma)
  library(edgeR)
})

args <- commandArgs(trailingOnly = TRUE)
project <- Sys.getenv("PROJ", unset = "/data/zxy/projects/human_thymus_age_ML_DL")
if (length(args) >= 2 && args[1] == "--project") project <- args[2]

meta_file <- file.path(project, "01_raw_processing", "metadata",
                       "05B_final_regression_cohort.csv")
count_file <- file.path(project, "02_pseudobulk", "donor_matrix",
                        "raw_combined.csv.gz")
primary_file <- file.path(project, "03_feature_selection", "limma_adjusted",
                          "whole_thymus_limma_age_sex.csv")
out <- file.path(project, "03_feature_selection", "expression_sensitivity")
dir.create(out, recursive = TRUE, showWarnings = FALSE)

for (path in c(meta_file, count_file, primary_file)) {
  if (!file.exists(path)) stop("Missing required input: ", path)
}

meta <- read.csv(meta_file, stringsAsFactors = FALSE)
meta$donor_id <- as.character(meta$donor_id)
if (nrow(meta) != 18 || anyDuplicated(meta$donor_id)) {
  stop("Expected 18 unique donors in frozen regression cohort")
}
counts <- read.csv(gzfile(count_file), check.names = FALSE)
rownames(counts) <- as.character(counts[[1]])
counts[[1]] <- NULL
counts <- counts[meta$donor_id, , drop = FALSE]
if (any(is.na(rownames(counts)))) stop("Donor/count alignment failed")

fit_subset <- function(donor_ids, tag, nonlinear = FALSE, female_only = FALSE) {
  md <- meta[match(donor_ids, meta$donor_id), , drop = FALSE]
  cts <- round(as.matrix(counts[donor_ids, , drop = FALSE]))
  storage.mode(cts) <- "integer"
  md$age_z <- as.numeric(scale(md$age_years))
  md$age_z_squared <- md$age_z^2
  md$sex <- factor(md$sex)
  if (nonlinear) {
    design <- if (female_only || nlevels(md$sex) < 2) {
      model.matrix(~ age_z + age_z_squared, data = md)
    } else {
      model.matrix(~ age_z + age_z_squared + sex, data = md)
    }
  } else {
    design <- if (female_only || nlevels(md$sex) < 2) {
      model.matrix(~ age_z, data = md)
    } else {
      model.matrix(~ age_z + sex, data = md)
    }
  }
  if (qr(design)$rank < ncol(design)) stop(tag, ": rank-deficient design")
  y <- DGEList(counts = t(cts))
  keep <- filterByExpr(y, design)
  y <- y[keep, , keep.lib.sizes = FALSE]
  y <- calcNormFactors(y, method = "TMM")
  voom_object <- voom(y, design, plot = FALSE)
  fit <- eBayes(lmFit(voom_object, design))
  age <- topTable(fit, coef = "age_z", number = Inf, sort.by = "none")
  result <- data.frame(
    analysis = tag,
    gene = rownames(age),
    beta_age_per_SD = age$logFC,
    age_P_Value = age$P.Value,
    age_adj_P_Val = age$adj.P.Val,
    n_donors = nrow(md),
    min_age = min(md$age_years),
    max_age = max(md$age_years),
    stringsAsFactors = FALSE
  )
  if (nonlinear) {
    quadratic <- topTable(fit, coef = "age_z_squared", number = Inf, sort.by = "none")
    result$beta_age_squared <- quadratic[match(result$gene, rownames(quadratic)), "logFC"]
    result$age_squared_P_Value <- quadratic[match(result$gene, rownames(quadratic)), "P.Value"]
    result$age_squared_adj_P_Val <- quadratic[match(result$gene, rownames(quadratic)), "adj.P.Val"]
  }
  result
}

ordered <- meta$donor_id[order(meta$age_years)]
subsets <- list(
  remove_youngest = setdiff(meta$donor_id, ordered[1]),
  remove_oldest = setdiff(meta$donor_id, tail(ordered, 1)),
  remove_both_extremes = setdiff(meta$donor_id, c(ordered[1], tail(ordered, 1))),
  restricted_age_18_64 = meta$donor_id[meta$age_years >= 18 & meta$age_years <= 64],
  female_only = meta$donor_id[tolower(meta$sex) == "female"]
)

results <- list()
for (tag in names(subsets)) {
  donors <- subsets[[tag]]
  if (length(donors) < 8) next
  results[[tag]] <- fit_subset(donors, tag, female_only = tag == "female_only")
}
results$nonlinear_age_quadratic_plus_sex <- fit_subset(
  meta$donor_id, "nonlinear_age_quadratic_plus_sex", nonlinear = TRUE
)
sensitivity <- do.call(rbind, results)
sensitivity_connection <- gzfile(
  file.path(out, "expression_age_shape_sensitivity.tsv.gz"), "wt"
)
write.table(sensitivity, sensitivity_connection, sep = "\t", row.names = FALSE,
            quote = FALSE, na = "NA", fileEncoding = "UTF-8")
close(sensitivity_connection)

# Leave-one-donor-out expression influence. Each rerun refits voom/limma.
influence <- lapply(meta$donor_id, function(excluded) {
  fit <- fit_subset(setdiff(meta$donor_id, excluded), paste0("exclude_", excluded))
  fit$excluded_donor <- excluded
  fit
})
influence <- do.call(rbind, influence)
influence_connection <- gzfile(
  file.path(out, "expression_leave_one_donor_out.tsv.gz"), "wt"
)
write.table(influence, influence_connection, sep = "\t", row.names = FALSE,
            quote = FALSE, na = "NA", fileEncoding = "UTF-8")
close(influence_connection)

primary <- read.csv(primary_file, stringsAsFactors = FALSE)
names(primary)[names(primary) == "logFC_age_perSD"] <- "primary_beta_age"
primary_keep <- primary[, c("gene", "primary_beta_age", "adj.P.Val")]
names(primary_keep)[names(primary_keep) == "adj.P.Val"] <- "primary_q_value"

summary_rows <- lapply(names(results), function(tag) {
  tab <- merge(primary_keep, results[[tag]], by = "gene", all = FALSE)
  data.frame(
    analysis = tag,
    n_donors = unique(tab$n_donors)[1],
    n_genes_compared = nrow(tab),
    spearman_beta_vs_primary = suppressWarnings(cor(
      tab$primary_beta_age, tab$beta_age_per_SD, method = "spearman"
    )),
    sign_concordance_all = mean(sign(tab$primary_beta_age) == sign(tab$beta_age_per_SD)),
    sign_concordance_primary_q05 = {
      selected <- tab$primary_q_value < 0.05
      if (sum(selected, na.rm = TRUE) > 0) {
        mean(sign(tab$primary_beta_age[selected]) == sign(tab$beta_age_per_SD[selected]))
      } else NA_real_
    },
    n_age_q05 = sum(tab$age_adj_P_Val < 0.05, na.rm = TRUE),
    n_quadratic_q05 = if ("age_squared_adj_P_Val" %in% names(tab)) {
      sum(tab$age_squared_adj_P_Val < 0.05, na.rm = TRUE)
    } else NA_integer_,
    stringsAsFactors = FALSE
  )
})
summary <- do.call(rbind, summary_rows)
write.table(summary, file.path(out, "expression_sensitivity_summary.tsv"),
            sep = "\t", row.names = FALSE, quote = FALSE, na = "NA")

# For every primary q<0.05 gene, report the range/sign stability across 18 LODO fits.
primary_sig <- primary_keep$gene[primary_keep$primary_q_value < 0.05]
influence_sig <- influence[influence$gene %in% primary_sig, ]
stability <- do.call(rbind, lapply(split(influence_sig, influence_sig$gene), function(tab) {
  primary_beta <- primary_keep$primary_beta_age[match(tab$gene[1], primary_keep$gene)]
  data.frame(
    gene = tab$gene[1],
    primary_beta_age = primary_beta,
    minimum_LODO_beta = min(tab$beta_age_per_SD, na.rm = TRUE),
    maximum_LODO_beta = max(tab$beta_age_per_SD, na.rm = TRUE),
    LODO_sign_stability_fraction = mean(sign(tab$beta_age_per_SD) == sign(primary_beta)),
    minimum_LODO_q = min(tab$age_adj_P_Val, na.rm = TRUE),
    maximum_LODO_q = max(tab$age_adj_P_Val, na.rm = TRUE),
    stringsAsFactors = FALSE
  )
}))
write.table(stability, file.path(out, "expression_primary_gene_donor_stability.tsv"),
            sep = "\t", row.names = FALSE, quote = FALSE, na = "NA")

contract <- c(
  "analysis unit: donor",
  "primary comparison source: Step 22 sex-adjusted limma-voom",
  "sensitivity subsets: remove youngest; remove oldest; remove both; age 18-64; female-only",
  "age-shape model: standardized linear age + squared standardized age + sex",
  "LODO influence: voom and limma are refit after excluding each donor",
  "interpretation: sensitivity analyses; not independent validation"
)
writeLines(contract, file.path(out, "29_analysis_contract.txt"))
cat("Expression sensitivity outputs written to", out, "\n")

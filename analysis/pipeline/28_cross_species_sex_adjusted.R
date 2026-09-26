#!/usr/bin/env Rscript
# Step 28: descriptive human-mouse comparison using sex-adjusted human limma beta_age.
#
# Mouse data contain one Young and one Aged animal. This script therefore emits
# counts and explicit assessability fields only; it performs no animal-level
# hypothesis test and never labels the result as validation or conservation.

args <- commandArgs(trailingOnly = TRUE)
project <- Sys.getenv("PROJ", unset = "/data/zxy/projects/human_thymus_age_ML_DL")
if (length(args) >= 2 && args[1] == "--project") project <- args[2]

limma_file <- file.path(project, "03_feature_selection", "limma_adjusted",
                        "whole_thymus_limma_age_sex.csv")
mouse_dir <- file.path(project, "08_mouse_validation")
ortholog_file <- file.path(mouse_dir, "orthologs",
                           "human_mouse_orthologs_reconciled.csv")
rna_file <- file.path(mouse_dir, "18_mouse_human_orthologs.csv")
atac_file <- file.path(mouse_dir, "19_scATAC_gene_level_join.csv")
peak_file <- file.path(mouse_dir, "19_scATAC_peak_validation.csv")
out <- file.path(mouse_dir, "sex_adjusted_descriptive")
dir.create(out, recursive = TRUE, showWarnings = FALSE)

required <- c(limma_file, ortholog_file, rna_file, atac_file, peak_file)
missing <- required[!file.exists(required)]
if (length(missing)) stop("Missing required inputs:\n", paste(missing, collapse = "\n"))

as_bool <- function(x) {
  if (is.logical(x)) return(x)
  tolower(as.character(x)) %in% c("true", "t", "1", "yes")
}

human <- read.csv(limma_file, stringsAsFactors = FALSE, check.names = FALSE)
needed_human <- c("gene", "logFC_age_perSD", "P.Value", "adj.P.Val")
if (!all(needed_human %in% names(human))) {
  stop("Limma table lacks required columns: ",
       paste(setdiff(needed_human, names(human)), collapse = ", "))
}
if (anyDuplicated(human$gene)) stop("Limma table must contain one row per human gene")
human$human_gene <- as.character(human$gene)
human$human_lower <- tolower(human$human_gene)
human$human_beta_age <- as.numeric(human$logFC_age_perSD)
human$human_p_value <- as.numeric(human$P.Value)
human$human_q_value <- as.numeric(human$adj.P.Val)
human$human_significant_q05 <- !is.na(human$human_q_value) & human$human_q_value < 0.05
human$human_direction <- ifelse(human$human_beta_age > 0, "Age_up",
                          ifelse(human$human_beta_age < 0, "Age_down", NA_character_))
human_keep <- human[, c("human_gene", "human_lower", "human_beta_age",
                        "human_p_value", "human_q_value",
                        "human_significant_q05", "human_direction")]
human_effect_keep <- human_keep[, setdiff(names(human_keep), "human_gene")]

orth <- read.csv(ortholog_file, stringsAsFactors = FALSE)
orth$is_best_ortholog <- as_bool(orth$is_best_ortholog)
orth <- orth[orth$is_best_ortholog %in% TRUE, ]
if (anyDuplicated(orth$human_gene)) stop("Best-ortholog table has duplicate human genes")
orth$human_lower <- tolower(orth$human_gene)
orth$mouse_lower <- tolower(orth$mouse_gene)
orth_keep <- orth[, intersect(
  c("human_gene", "human_lower", "mouse_gene", "mouse_lower",
    "ortholog_priority", "orthology_type", "in_ML_consensus", "in_DL_top100"),
  names(orth)
)]

classify_direction <- function(df, mouse_effect_column, detectable_column,
                               evidence_type) {
  df$mouse_effect <- as.numeric(df[[mouse_effect_column]])
  detectable <- as_bool(df[[detectable_column]])
  df$assessable <- detectable & !is.na(df$human_beta_age) &
    !is.na(df$mouse_effect) & df$human_beta_age != 0 & df$mouse_effect != 0
  df$concordant <- NA
  df$concordant[df$assessable] <-
    sign(df$human_beta_age[df$assessable]) == sign(df$mouse_effect[df$assessable])
  df$non_assessable_reason <- NA_character_
  df$non_assessable_reason[!detectable] <- "mouse feature not detectable"
  df$non_assessable_reason[detectable & is.na(df$human_beta_age)] <-
    "human gene absent from sex-adjusted limma result"
  df$non_assessable_reason[detectable & !is.na(df$human_beta_age) &
                             df$human_beta_age == 0] <- "human beta_age is zero"
  df$non_assessable_reason[detectable & !is.na(df$human_beta_age) &
                             is.na(df$mouse_effect)] <- "mouse effect missing"
  df$non_assessable_reason[detectable & !is.na(df$human_beta_age) &
                             !is.na(df$mouse_effect) & df$mouse_effect == 0] <-
    "mouse effect is zero"
  df$evidence_type <- evidence_type
  df$analysis_unit_human <- "donor"
  df$mouse_design <- "one Young and one Aged animal; descriptive only"
  df
}

# RNA: keep the frozen best-ortholog/mouse effect rows, replace the human metric.
rna <- read.csv(rna_file, stringsAsFactors = FALSE)
if (!"human_gene" %in% names(rna)) stop("18_mouse_human_orthologs.csv lacks human_gene")
rna$human_lower <- tolower(rna$human_gene)
drop_old <- intersect(c("WT_age_rho", "WT_rho_padj", "human_sign", "concordant",
                        "assessable"), names(rna))
rna[drop_old] <- NULL
rna <- merge(rna, human_effect_keep, by = "human_lower", all.x = TRUE, sort = FALSE)
rna <- classify_direction(rna, "RNA_adjusted_effect", "detectable", "mouse_scRNA")
write.table(rna, file.path(out, "28_human_mouse_RNA_sex_adjusted.tsv"),
            sep = "\t", row.names = FALSE, quote = FALSE, na = "NA")

# ATAC gene activity.
atac <- read.csv(atac_file, stringsAsFactors = FALSE)
if (!"human_gene" %in% names(atac)) {
  atac <- merge(atac, orth_keep[, c("human_lower", "human_gene")],
                by = "human_lower", all.x = TRUE, sort = FALSE)
}
atac$human_lower <- tolower(atac$human_gene)
drop_old <- intersect(c("WT_age_rho", "WT_rho_padj", "human_dir",
                        "expected_atac_sign", "atac_direction_ok"), names(atac))
atac[drop_old] <- NULL
atac <- merge(atac, human_effect_keep, by = "human_lower", all.x = TRUE, sort = FALSE)
atac <- classify_direction(atac, "ATAC_adjusted_effect", "atac_detectable",
                           "mouse_scATAC_gene_activity")
write.table(atac, file.path(out, "28_human_mouse_ATAC_sex_adjusted.tsv"),
            sep = "\t", row.names = FALSE, quote = FALSE, na = "NA")

# Linked peaks: map the human gene through the frozen best ortholog and use beta_age.
peaks <- read.csv(peak_file, stringsAsFactors = FALSE)
if (!"human_lower" %in% names(peaks)) stop("19_scATAC_peak_validation.csv lacks human_lower")
drop_old <- intersect(c("WT_age_rho", "human_dir", "expected_sign", "peak_direction_ok"),
                      names(peaks))
peaks[drop_old] <- NULL
peaks <- merge(peaks, orth_keep[, c("human_lower", "human_gene")],
               by = "human_lower", all.x = TRUE, sort = FALSE)
peaks <- merge(peaks, human_keep, by = c("human_lower", "human_gene"),
               all.x = TRUE, sort = FALSE)
peaks$peak_detectable <- !is.na(peaks$accessibility_effect_Age_minus_Young)
peaks <- classify_direction(peaks, "accessibility_effect_Age_minus_Young",
                            "peak_detectable", "mouse_scATAC_linked_peak")
write.table(peaks, file.path(out, "28_human_mouse_linked_peaks_sex_adjusted.tsv"),
            sep = "\t", row.names = FALSE, quote = FALSE, na = "NA")

summary_one <- function(df, evidence_type) {
  subsets <- list(
    all_mapped = rep(TRUE, nrow(df)),
    human_q_lt_0_05 = df$human_significant_q05 %in% TRUE,
    ML_consensus = if ("in_ML_consensus" %in% names(df)) as_bool(df$in_ML_consensus)
                   else rep(FALSE, nrow(df))
  )
  do.call(rbind, lapply(names(subsets), function(label) {
    z <- df[subsets[[label]], ]
    assessed <- z[z$assessable %in% TRUE, ]
    data.frame(
      evidence_type = evidence_type,
      subset = label,
      n_rows_total = nrow(z),
      n_assessable = nrow(assessed),
      n_concordant = sum(assessed$concordant %in% TRUE),
      concordance_fraction = ifelse(nrow(assessed) > 0,
                                    mean(assessed$concordant %in% TRUE), NA_real_),
      significance_status = "descriptive counts only; no animal-level test",
      stringsAsFactors = FALSE
    )
  }))
}

summary <- rbind(
  summary_one(rna, "mouse_scRNA"),
  summary_one(atac, "mouse_scATAC_gene_activity"),
  summary_one(peaks, "mouse_scATAC_linked_peak")
)
write.table(summary, file.path(out, "28_cross_species_descriptive_summary.tsv"),
            sep = "\t", row.names = FALSE, quote = FALSE, na = "NA")

common_columns <- c("human_gene", "mouse_gene", "human_beta_age", "human_p_value",
                    "human_q_value", "human_significant_q05", "human_direction",
                    "mouse_effect", "assessable", "concordant",
                    "non_assessable_reason", "evidence_type",
                    "analysis_unit_human", "mouse_design")
normalize_columns <- function(df, columns) {
  for (column in setdiff(columns, names(df))) df[[column]] <- NA
  df[, columns]
}
combined <- rbind(normalize_columns(rna, common_columns),
                  normalize_columns(atac, common_columns))
write.table(combined, file.path(out, "human_mouse_ortholog_comparison.tsv"),
            sep = "\t", row.names = FALSE, quote = FALSE, na = "NA")

contract <- c(
  "human effect: sex-adjusted limma logFC_age_perSD",
  "human significance: adj.P.Val < 0.05 reported as a status field",
  "directional assessability: nonzero human beta_age + nonzero mouse effect + detectable mouse feature",
  "mouse design: one Young and one Aged animal",
  "inference: none at the animal level; genes and peaks are not biological replicates",
  "interpretation: hypothesis-generating supplementary comparison only"
)
writeLines(contract, file.path(out, "28_analysis_contract.txt"))
cat("Sex-adjusted descriptive human-mouse outputs written to", out, "\n")

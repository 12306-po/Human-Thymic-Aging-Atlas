#!/usr/bin/env Rscript
# Step 18: Mouse scRNA cross-species validation.
# Join human candidates -> mouse orthologs -> mouse gene-level age direction
# (from the mouse thymus atlas pipeline), compare direction concordance.
#
# Inputs:
#   08_mouse_validation/orthologs/human_mouse_orthologs_reconciled.csv
#   /data/zxy/projects/mouse_thymus/mouse_thymus_atlas/tables/
#     07_all_shared_genes_phase_position_matched_effects.csv (gene-level)
#     08_cross_modal_robust_candidate_genes.csv (cross-modal robust)
#   06_interpretation/16_candidate_direction_table.csv (human WT age-rho)
# Outputs (08_mouse_validation/):
#   18_mouse_human_orthologs.csv
#   18_cross_species_validation.csv
#   18_cross_species_summary.txt
#   09_figures/18_cross_species_concordance.pdf
#   10_results/logs/18_mouse_scRNA_cross_species_validation.log
proj <- Sys.getenv("PROJ", unset = "")
if (!nzchar(proj)) stop("Required env var PROJ is not set. Run: source scripts/00_project_config.sh")
orth_dir <- file.path(proj, "08_mouse_validation")
tabs <- Sys.getenv("MOUSE_DERIVED", unset = "")
if (!nzchar(tabs)) stop("Required env var MOUSE_DERIVED is not set. Run: source scripts/00_project_config.sh")
tabs <- file.path(tabs, "tables")
outdir <- file.path(proj, "08_mouse_validation")
figdir <- file.path(proj, "09_figures")
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)
dir.create(figdir, recursive = TRUE, showWarnings = FALSE)

logf <- file.path(proj, "10_results/logs/18_mouse_scRNA_cross_species_validation.log")
sink(logf, split = TRUE)
cat("=== Step 18: Mouse scRNA cross-species validation ===\n")

# ---- 1. Load orthologs (human -> mouse) ----
orth <- read.csv(file.path(orth_dir, "orthologs/human_mouse_orthologs_reconciled.csv"),
                 stringsAsFactors = FALSE)
orth$is_best_ortholog <- as.logical(orth$is_best_ortholog)
orth$in_ML_consensus <- as.logical(orth$in_ML_consensus)
orth$in_DL_top100 <- as.logical(orth$in_DL_top100)
orth <- orth[orth$is_best_ortholog, ]
orth$mouse_lower <- tolower(orth$mouse_gene)
orth$human_lower <- tolower(orth$human_gene)
cat("Ortholog pairs (best-ortholog):", nrow(orth), "| human genes:", length(unique(orth$human_gene)), "\n")

# ---- 2. Load mouse gene-level age effects (table 07) ----
eff <- read.csv(file.path(tabs, "07_all_shared_genes_phase_position_matched_effects.csv"),
                stringsAsFactors = FALSE)
# P0-3: uniqueness assertion — table 07 must be one row per mouse gene.
# NEVER silently take the first row / dedup here; if duplicates exist, explicit
# pre-agreed stage/program aggregation is required (here: fail instead).
stopifnot(
  "mouse effect table has duplicated feature rows — must aggregate explicitly" =
    !anyDuplicated(eff$feature) > 0
)
eff$mouse_lower <- tolower(eff$feature)
cat("Mouse effect genes:", length(unique(eff$feature)),
    "| detectables:", sum(eff$detectable, na.rm = TRUE), "\n")

# ---- 3. Load human direction (WT age-rho from Step 16) ----
# NB: in_ML_consensus / in_DL_top100 membership flags are taken from the ortholog
# table (same gene universe) to avoid a duplicate-column collision on merge.
hum <- read.csv(file.path(proj, "06_interpretation/16_candidate_direction_table.csv"),
                stringsAsFactors = FALSE)
hum <- hum[!is.na(hum$WT_age_rho), c("gene", "WT_age_rho", "WT_rho_padj",
                                      "ml_best_median_pct")]
names(hum)[names(hum) == "gene"] <- "human_gene"
hum$human_lower <- tolower(hum$human_gene)
hum$human_gene <- NULL  # keep only join key + direction features

# ---- 4. Join: human gene + mouse effect + human direction ----
j <- merge(orth, eff, by = "mouse_lower", all.x = FALSE)
j <- merge(j, hum, by = "human_lower", all.x = TRUE)
cat("Matched human-mouse pairs:", nrow(j),
    "| unique human genes:", length(unique(j$human_gene)), "\n")

# P0: freeze mapping — Human->Mouse best ortholog comes from Step 18a ONLY.
# Any human gene in the candidate universe with NO best ortholog in the 18a
# table is marked ortholog_status = "NOT_MAPPED" and excluded from gene-level
# concordance (reported separately).
cand_genes <- unique(c(
  scan(file.path(proj, "04_machine_learning/ML_consensus_genes.txt"),
       what = character(), quiet = TRUE),
  scan(file.path(proj, "06_interpretation/16_DL_top100_genes.txt"),
       what = character(), quiet = TRUE)
))
mapped_human <- unique(j$human_gene)
unmapped <- setdiff(cand_genes, mapped_human)
cat("Candidate genes NOT mapped by Step 18a:", length(unmapped), "\n")
if (length(unmapped)) {
  notmapped <- data.frame(human_gene = unmapped, ortholog_status = "NOT_MAPPED",
                          stringsAsFactors = FALSE)
  write.csv(notmapped, file.path(outdir, "18_unmapped_human_genes.csv"), row.names = FALSE)
}

# ---- 5. Direction comparison ----
# P0 (2026-09-15, frozen rule): primary concordance is a DIRECT sign comparison
# between the human continuous-age association and the mouse RNA effect:
#     assessable = detectable ortholog + valid human WT_age_rho + valid mouse
#                  RNA_adjusted_effect
#     concordant = sign(WT_age_rho) == sign(RNA_adjusted_effect)
# Mouse-internal `direction` (Age_up / Young_up / Discordant) is used ONLY for
# the high-confidence SENSITIVITY subset, never to determine primary
# concordance, and a mouse "Discordant" never forces concordant=NA.
j$detectable <- as.logical(j$detectable)
j$assessable <- (!is.na(j$WT_age_rho)) &
                (!is.na(j$RNA_adjusted_effect)) &
                j$detectable %in% TRUE
j$human_sign <- sign(j$WT_age_rho)
j$mouse_rna_sign <- sign(j$RNA_adjusted_effect)
j$concordant <- NA
# TRUE/FALSE for all assessable pairs (an exact-zero effect carries no direction
# and is treated as non-concordant, not NA).
j$concordant[j$assessable] <-
  (j$human_sign[j$assessable] == j$mouse_rna_sign[j$assessable]) &
  (j$human_sign[j$assessable] != 0)

assess <- j[j$assessable %in% TRUE, ]
cat("Main assessable pairs (all detectable, valid effects):", nrow(assess), "\n")
# high-confidence sensitivity subset = mouse-internal Age/Young concordant
hiconf <- assess[assess$direction %in% c("Age_up_concordant", "Young_up_concordant"), ]
cat("High-confidence subset (mouse-internal concordant):", nrow(hiconf), "\n")

write.csv(j, file.path(outdir, "18_mouse_human_orthologs.csv"), row.names = FALSE)

# ---- 6. Cross-species validation table ----
# Exact binomial test against p=0.5 (P1, 2026-09-15). 1Y+1A mouse design is
# NOT biological-replicate inference; report as descriptive/supportive only.
binom_row <- function(k, n) {
  if (is.na(n) || n == 0) {
    return(c(k = NA_real_, n = NA_real_, rate = NA_real_,
             ci_low = NA_real_, ci_high = NA_real_, p_vs_0.5 = NA_real_))
  }
  bt <- tryCatch(binom.test(k, n, p = 0.5), error = function(e) NULL)
  if (is.null(bt)) {
    return(c(k = k, n = n, rate = k / n,
             ci_low = NA_real_, ci_high = NA_real_, p_vs_0.5 = NA_real_))
  }
  c(k = k, n = n, rate = unname(k / n),
    ci_low = unname(bt$conf.int[1]), ci_high = unname(bt$conf.int[2]),
    p_vs_0.5 = unname(bt$p.value))
}

conc_pairs <- sum(assess$concordant %in% TRUE)
conc_pairs_hi <- sum(hiconf$concordant %in% TRUE)
conc_genes <- length(unique(assess$human_gene[assess$concordant %in% TRUE]))
conc_genes_hi <- length(unique(hiconf$human_gene[hiconf$concordant %in% TRUE]))

n_assess <- nrow(assess)
n_hi <- nrow(hiconf)
n_assess_gene <- length(unique(assess$human_gene))
n_hi_gene <- length(unique(hiconf$human_gene))

val_tab <- rbind(
  c(level = "pair_main",             binom_row(conc_pairs, n_assess)),
  c(level = "pair_highconf",         binom_row(conc_pairs_hi, n_hi)),
  c(level = "gene_main",             binom_row(conc_genes, n_assess_gene)),
  c(level = "gene_highconf",         binom_row(conc_genes_hi, n_hi_gene))
)
val_tab <- as.data.frame(val_tab, stringsAsFactors = FALSE)
for (col in c("k", "n", "rate", "ci_low", "ci_high", "p_vs_0.5")) {
  val_tab[[col]] <- as.numeric(val_tab[[col]])
}
val_tab$rate <- round(val_tab$rate, 3)
val_tab$ci_low <- round(val_tab$ci_low, 3)
val_tab$ci_high <- round(val_tab$ci_high, 3)
val_tab$p_vs_0.5 <- signif(val_tab$p_vs_0.5, 3)

# ML-consensus subset at gene level
ml_genes <- unique(assess$human_gene[assess$in_ML_consensus %in% TRUE])
ml_conc <- sum(assess$in_ML_consensus %in% TRUE & assess$concordant %in% TRUE)
# NOTE: wrap the named vector in as.list() so as.data.frame produces a 1-row,
# 7-column table (as.data.frame of a bare vector would be 7 rows x 1 column).
val_tab_ml <- as.data.frame(
  as.list(c(level = "gene_ML_consensus",
            binom_row(ml_conc, length(ml_genes)))),
  stringsAsFactors = FALSE)
for (col in c("k", "n", "rate", "ci_low", "ci_high", "p_vs_0.5")) {
  val_tab_ml[[col]] <- as.numeric(val_tab_ml[[col]])
}
val_tab_ml$rate <- round(val_tab_ml$rate, 3)
val_tab_ml$ci_low <- round(val_tab_ml$ci_low, 3)
val_tab_ml$ci_high <- round(val_tab_ml$ci_high, 3)
val_tab_ml$p_vs_0.5 <- signif(val_tab_ml$p_vs_0.5, 3)
val_tab <- rbind(val_tab, val_tab_ml)
write.csv(val_tab, file.path(outdir, "18_cross_species_validation.csv"), row.names = FALSE)
cat("\nCross-species validation (exact binomial vs p=0.5, descriptive/supportive):\n")
print(val_tab)

# ---- 7. Per-celltype? Table 07 is T-lineage pooled; use table 19/20 program
#        direction as celltype-stage-level supplement ----
prog <- read.csv(file.path(tabs, "19_broad_stage_crossmodal_program_effects.csv"),
                 stringsAsFactors = FALSE)
prog_sig <- prog[prog$direction_concordant == TRUE, ]
cat("\nCross-modal stage-program directions (supplement):\n")
print(table(prog_sig$stage, prog_sig$direction_scRNA))

# ---- 8. Quadrant scatter: human WT rho vs mouse RNA effect (all assessable) ----
# P0-1 figure: x = human WT Spearman rho, y = mouse RNA Age-Young effect,
# quadrant labels + n + %. One point per human gene (best ortholog from 18a).
j2 <- assess[!duplicated(assess$human_gene), ]
j2 <- j2[!is.na(j2$RNA_adjusted_effect) & !is.na(j2$WT_age_rho), ]
cat("\nGenes with both effects (main assessable, gene-level):", nrow(j2), "\n")
if (nrow(j2) >= 5) {
  j2$quad <- with(j2, ifelse(WT_age_rho > 0 & RNA_adjusted_effect > 0, "concordant_age_up",
                     ifelse(WT_age_rho < 0 & RNA_adjusted_effect < 0, "concordant_age_down",
                     ifelse(WT_age_rho > 0, "discordant", "discordant"))))
  tab_q <- table(j2$quad)
  pdf(file.path(figdir, "18_cross_species_concordance.pdf"),
      width = 7, height = 6)
  par(mar = c(5, 5, 3, 1))
  plot(j2$WT_age_rho, j2$RNA_adjusted_effect,
       col = ifelse(j2$quad == "discordant", "firebrick", "steelblue"),
       pch = 16, cex = 0.8,
       xlab = "Human WT age Spearman rho", ylab = "Mouse RNA Age-Young effect",
       main = sprintf("Cross-species direction (n=%d genes)", nrow(j2)))
  abline(h = 0, v = 0, lty = 2, col = "grey50")
  q1 <- tab_q[["concordant_age_up"]]; q2 <- tab_q[["concordant_age_down"]]
  mtext(sprintf("Concordant: %d (%.0f%%)", sum(j2$quad != "discordant"),
                100 * sum(j2$quad != "discordant") / nrow(j2)),
        side = 3, line = 0.2, cex = 0.9)
  legend("topright", legend = c("concordant", "discordant"),
         col = c("steelblue", "firebrick"), pch = 16, cex = 0.8)
  dev.off()
  cat("Wrote quadrant scatter:", nrow(j2), "genes\n")
}

cat("\n=== Step 18 COMPLETE ===\n")
sink()

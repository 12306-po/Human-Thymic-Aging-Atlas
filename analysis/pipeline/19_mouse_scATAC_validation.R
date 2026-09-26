#!/usr/bin/env Rscript
# Step 19: Mouse scATAC cross-species regulatory validation.
# Reuses the mouse thymus atlas scATAC outputs (peak-level accessibility
# effects, motif/TF enrichment) to ask: for the human candidate genes,
# do their mouse orthologs' linked peaks show the same age direction, and do
# candidate TF motifs support the regulatory direction?
#
# Inputs:
#   08_mouse_validation/orthologs/human_mouse_orthologs_reconciled.csv
#   /data/zxy/projects/mouse_thymus/mouse_thymus_atlas/tables/
#     07_all_shared_genes_phase_position_matched_effects.csv
#     09_candidate_linked_peak_accessibility_effects.csv
#     10_motif_enrichment_with_TF_multiomic_evidence.csv
#     10_candidate_TF_motif_peak_target_network.csv
#   /data/zxy/projects/mouse_thymus/mouse_thymus_atlas/results/scATAC/
#     mouse_thymus_scATAC_final_metadata.csv
#   06_interpretation/16_candidate_direction_table.csv (human direction)
#   06_interpretation/16_TF_candidates.csv (human TF candidates)
#
# Outputs (08_mouse_validation/):
#   19_scATAC_gene_peak_validation.csv
#   19_scATAC_TF_validation.csv
#   19_scATAC_validation_summary.txt
#   09_figures/19_scATAC_regulatory_concordance.pdf
proj <- Sys.getenv("PROJ", unset = "")
if (!nzchar(proj)) stop("Required env var PROJ is not set. Run: source scripts/00_project_config.sh")
tabs <- Sys.getenv("MOUSE_DERIVED", unset = "")
if (!nzchar(tabs)) stop("Required env var MOUSE_DERIVED is not set. Run: source scripts/00_project_config.sh")
tabs <- file.path(tabs, "tables")
results <- file.path(Sys.getenv("MOUSE_DERIVED"), "results/scATAC")
outdir <- file.path(proj, "08_mouse_validation")
figdir <- file.path(proj, "09_figures")
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)
dir.create(figdir, recursive = TRUE, showWarnings = FALSE)

logf <- file.path(proj, "10_results/logs/19_mouse_scATAC_validation.log")
sink(logf, split = TRUE)
cat("=== Step 19: Mouse scATAC cross-species regulatory validation ===\n")

suppressMessages(library(grDevices))

# ---- 1. Orthologs (reconciled best) ----
orth <- read.csv(file.path(outdir, "orthologs/human_mouse_orthologs_reconciled.csv"),
                 stringsAsFactors = FALSE)
orth$is_best_ortholog <- as.logical(orth$is_best_ortholog)
orth$in_ML_consensus <- as.logical(orth$in_ML_consensus)
orth$in_DL_top100 <- as.logical(orth$in_DL_top100)
orth <- orth[orth$is_best_ortholog, ]
orth$mouse_lower <- tolower(orth$mouse_gene)
orth$human_lower <- tolower(orth$human_gene)
cat("Orthologs (best):", nrow(orth), "| human genes:", length(unique(orth$human_gene)), "\n")

# ---- 2. Human candidates (direction reference) ----
hum <- read.csv(file.path(proj, "06_interpretation/16_candidate_direction_table.csv"),
                stringsAsFactors = FALSE)
hum <- hum[!is.na(hum$WT_age_rho), c("gene", "WT_age_rho", "WT_rho_padj")]
names(hum)[names(hum) == "gene"] <- "human_gene"
# Zero-direction rule (2026-09-16): rho == 0 carries NO direction — it must
# not be bucketed as Age_down. NA = non-assessable directional evidence.
hum$human_dir <- ifelse(hum$WT_age_rho > 0, "Age_up",
                 ifelse(hum$WT_age_rho < 0, "Age_down", NA_character_))
hum$human_lower <- tolower(hum$human_gene)

# ---- 3. Mouse gene-level ATAC effects (table 07: ATAC columns) ----
eff <- read.csv(file.path(tabs, "07_all_shared_genes_phase_position_matched_effects.csv"),
                stringsAsFactors = FALSE)
# P0/P1 (2026-09-15): uniqueness assertion — one row per mouse gene, matching
# Step 18. Never silently dedup; Step 20 must not rely on drop_duplicates().
stopifnot(
  "Mouse ATAC effect table has duplicated gene rows; aggregate explicitly" =
    !anyDuplicated(eff$feature) > 0
)
eff$mouse_lower <- tolower(eff$feature)
cat("Mouse effect genes:", length(unique(eff$feature)), "\n")

# ---- 4. Join at the gene level ----
# keep only join keys + membership flags from orth; human_gene must NOT be
# carried into the merge (it collides with hum$human_gene -> orphaned .x/.y)
orth_join <- orth[, c("human_lower", "mouse_lower", "in_ML_consensus", "in_DL_top100")]
j <- merge(orth_join, eff, by = "mouse_lower", all.x = TRUE)
j <- merge(j, hum, by = "human_lower", all.x = TRUE)
j$atac_detectable <- j$detectable %in% TRUE & !is.na(j$ATAC_adjusted_effect)
# expected sign: human Age_up -> mouse ATAC Age_up (ATAC_effect_Age_minus_Young > 0)
j$expected_atac_sign <- ifelse(j$human_dir == "Age_up", 1,
                        ifelse(j$human_dir == "Age_down", -1, NA_real_))
j$atac_direction_ok <- (sign(j$ATAC_adjusted_effect) == j$expected_atac_sign)
# Zero-direction rule: neither a zero human rho nor a zero mouse ATAC effect
# has a direction -> non-assessable (NA), neither concordant nor discordant.
j$atac_direction_ok[
  is.na(j$expected_atac_sign) |
  is.na(j$ATAC_adjusted_effect) |
  j$ATAC_adjusted_effect == 0
] <- NA
# Authoritative tri-state column (2026-09-16): non-detectable genes are
# NON-ASSESSABLE, so the column must be NA (not TRUE/FALSE). Step 20 inherits
# this column directly as mouse_ATAC_concordant and applies no gate of its
# own, exactly as it does for Step 18 RNA concordance.
j$atac_direction_ok[!j$atac_detectable %in% TRUE] <- NA
j$human_gene <- orth$human_gene[match(j$human_lower, orth$human_lower)]
cat("Human genes joined:", length(unique(j$human_gene)),
    "| with mouse ATAC effect:", sum(j$atac_detectable), "\n")
write.csv(j, file.path(outdir, "19_scATAC_gene_level_join.csv"), row.names = FALSE)

# ---- 5. Peak-level validation (table 09) ----
# P0-1: MAIN analysis = ALL peaks with valid expected sign and non-NA
# accessibility effect (direction_concordant is mouse-internal QC, used only
# for the high-confidence sensitivity subset, NOT as the main denominator).
pk <- read.csv(file.path(tabs, "09_candidate_linked_peak_accessibility_effects.csv"),
               stringsAsFactors = FALSE)
pk$mouse_lower <- tolower(pk$feature)
pk <- merge(pk, orth_join[, c("mouse_lower", "human_lower", "in_ML_consensus", "in_DL_top100")],
            by = "mouse_lower", all.x = FALSE)
pk <- merge(pk, hum[, c("human_lower", "human_dir", "WT_age_rho")],
            by = "human_lower", all.x = TRUE)
pk$expected_sign <- ifelse(pk$human_dir == "Age_up", 1,
                    ifelse(pk$human_dir == "Age_down", -1, NA_real_))
pk$peak_direction_ok <- (sign(pk$accessibility_effect_Age_minus_Young) == pk$expected_sign)
# Zero-direction rule (2026-09-16): a zero human rho or a zero peak
# accessibility effect is non-assessable (NA), not discordant.
pk$peak_direction_ok[
  is.na(pk$expected_sign) |
  is.na(pk$accessibility_effect_Age_minus_Young) |
  pk$accessibility_effect_Age_minus_Young == 0
] <- NA
write.csv(pk, file.path(outdir, "19_scATAC_peak_validation.csv"), row.names = FALSE)

# P0-1: main assessable = valid expected sign + non-NA accessibility effect
assess_peak <- pk[!is.na(pk$peak_direction_ok) &
                    !is.na(pk$accessibility_effect_Age_minus_Young), ]
n_assess_peak <- nrow(assess_peak)
n_ok_peak <- sum(assess_peak$peak_direction_ok, na.rm = TRUE)
cat("Peaks assessable (main):", n_assess_peak, "| direction-ok:", n_ok_peak,
    "| rate:", round(n_ok_peak / max(1, n_assess_peak), 3), "\n")
# high-confidence sensitivity subset (mouse-internal concordant peaks)
assess_peak_hi <- assess_peak[assess_peak$direction_concordant %in% TRUE, ]
n_ok_peak_hi <- sum(assess_peak_hi$peak_direction_ok, na.rm = TRUE)
cat("Peaks high-confidence subset:", nrow(assess_peak_hi),
    "| direction-ok:", n_ok_peak_hi,
    "| rate:", round(n_ok_peak_hi / max(1, nrow(assess_peak_hi)), 3), "\n")

# ---- 6. Motif/TF validation (table 10) ----
# P1-2 (2026-09-15): candidate TF is defined uniformly as the TFs in the
# FROZEN Step 16 list (16_TF_candidates.csv) — NOT "any human candidate
# universe". This keeps a single TF-candidate definition across the full
# network (Section 6b) and the motif summary below.
tf16 <- read.csv(file.path(proj, "06_interpretation/16_TF_candidates.csv"),
                 stringsAsFactors = FALSE)
tf16_genes <- unique(as.character(tf16$TF))
mot <- read.csv(file.path(tabs, "10_motif_enrichment_with_TF_multiomic_evidence.csv"),
                stringsAsFactors = FALSE)
# map TF symbol (mouse) to human ortholog. Mouse symbols may be Title-cased
# (e.g. Arnt, Arid3a); normalize both sides to lowercase for the join, then
# recover the human symbol from the ortholog table.
mot$tf_mouse_lower <- tolower(mot$tf_symbol)
orth_tf <- orth[, c("human_lower", "mouse_lower", "human_gene")]
mot_tf <- merge(mot, orth_tf, by.x = "tf_mouse_lower", by.y = "mouse_lower", all.x = TRUE)
# restrict to TFs that are themselves human candidates (frozen Step 16 list,
# case-insensitive match on the human ortholog symbol)
mot_tf$is_human_candidate <- tolower(mot_tf$human_gene) %in% tolower(tf16_genes)
cat("Motif rows:", nrow(mot), "| TF rows mapped to human ortholog:", sum(!is.na(mot_tf$human_gene)),
    "| candidate TFs (Step 16):", sum(mot_tf$is_human_candidate, na.rm = TRUE), "\n")
write.csv(mot_tf, file.path(outdir, "19_scATAC_motif_TF_validation.csv"), row.names = FALSE)

# candidate-TF direction support: TF_RNA_direction_support / TF_ATAC_direction_support
mot_cand <- mot_tf[mot_tf$is_human_candidate %in% TRUE &
                   mot_tf$high_confidence_regulator %in% TRUE, ]
n_tf_highconf <- nrow(mot_cand)
n_tf_rna_support <- sum(mot_cand$TF_RNA_direction_support %in% TRUE, na.rm = TRUE)
n_tf_atac_support <- sum(mot_cand$TF_ATAC_direction_support %in% TRUE, na.rm = TRUE)
cat("Candidate TFs (high-confidence regulator):", n_tf_highconf,
    "| RNA direction support:", n_tf_rna_support,
    "| ATAC direction support:", n_tf_atac_support, "\n")

# ----------------------------------------------------------------------
# 6b. P0-2: TF -> motif -> peak -> target chain (network table)
#     10_candidate_TF_motif_peak_target_network.csv:
#       motif_id / motif_name / tf_symbol / reference_direction / peak /
#       target_gene / region_class / distance_to_tss /
#       accessibility_effect_Age_minus_Young / motif_p_adjust / motif_odds_ratio
#     For each human TF candidate, map: human_TF -> mouse_TF -> motif -> peak ->
#     mouse target gene -> human target gene (via 18a best ortholog), and score
#     each link component for direction support. Output the full chain table.
#     (Step 20 combines these layers via its FROZEN mouse_regulatory_support
#      rule; Step 19 only reports the evidence, it does NOT adjudicate.)
# ----------------------------------------------------------------------
net <- read.csv(file.path(tabs, "10_candidate_TF_motif_peak_target_network.csv"),
                stringsAsFactors = FALSE)
cat("\nNetwork file rows:", nrow(net), "\n")
net$tf_mouse_lower <- tolower(net$tf_symbol)
net$target_mouse_lower <- tolower(net$target_gene)

# reverse ortholog map: mouse TF symbol -> candidate human TF symbol (18a)
orth_tf2 <- orth[, c("mouse_lower", "human_lower", "human_gene")]
names(orth_tf2) <- c("tf_mouse_lower", "tf_human_lower", "human_TF")
net_tf <- merge(net, orth_tf2, by = "tf_mouse_lower", all.x = TRUE)
# restrict to human TF candidates (from Step 16 TF list; tf16_genes defined in
# section 6). Single source of TF candidate definition.
net_tf$is_human_TF_candidate <- tolower(net_tf$human_TF) %in% tolower(tf16_genes)

# human target gene via 18a best ortholog (mouse target -> human)
orth_target <- orth[, c("mouse_lower", "human_lower", "human_gene")]
names(orth_target) <- c("target_mouse_lower", "target_human_lower", "human_target_gene")
net_full <- merge(net_tf, orth_target, by = "target_mouse_lower", all.x = TRUE)

# human target age direction (from Step 16 direction table)
hum2 <- data.frame(gene = hum$human_gene, target_human_age_rho = hum$WT_age_rho,
                   stringsAsFactors = FALSE)
net_full <- merge(net_full, hum2, by.x = "human_target_gene", by.y = "gene", all.x = TRUE)

# motif direction support: TF reference_direction vs motif peak accessibility sign
net_full$peak_age_effect <- net_full$accessibility_effect_Age_minus_Young
net_full$peak_direction_support <- with(net_full, {
  exp_sgn <- ifelse(reference_direction == "Age_up_concordant", 1,
             ifelse(reference_direction == "Young_up_concordant", -1, NA_real_))
  !is.na(exp_sgn) & sign(peak_age_effect) == exp_sgn
})
# P0 (2026-09-16): DISTINCT from peak_direction_support above. That column
# compares the peak effect against the TF/motif regulatory reference_direction;
# this one compares the SAME peak effect against the human TARGET gene's WT
# age-rho direction. Step 20 linked-peak evidence MUST use this target-direction
# column, not the motif/TF reference-direction one.
# P1 (2026-09-16): zero effects have no direction — sign(0)==sign(0) would
# wrongly count (rho=0, effect=0) as concordant. Require both nonzero.
net_full$target_peak_direction_support <- with(net_full,
  !is.na(target_human_age_rho) &
  !is.na(peak_age_effect) &
  target_human_age_rho != 0 &
  peak_age_effect != 0 &
  sign(peak_age_effect) == sign(target_human_age_rho)
)
net_full$motif_direction_support <- net_full$peak_direction_support &
  (net_full$motif_p_adjust < 0.05)

# full chain: human TF candidate + motif direction support + target has human rho
# with matching direction (mature-state expectation is descriptive, not formal)
# Zero-direction rule (2026-09-16): do NOT use !target_human_age_up, which
# would mis-classify rho == 0 as Young_up/Age_down support. Judge directly by
# the sign of rho; rho == 0 can never support a full chain.
net_full$target_human_age_up <- net_full$target_human_age_rho > 0
# P1 (2026-09-15): column name aligned with out_cols and the Step 20 reader
# (full_chain_supported). Previously defined as `chain_supported`, which made
# the write.csv column selection fail with "undefined columns selected".
net_full$full_chain_supported <- with(net_full,
  is_human_TF_candidate & motif_direction_support %in% TRUE &
  !is.na(target_human_age_rho) &
  target_human_age_rho != 0 &
  ((reference_direction == "Age_up_concordant" & target_human_age_rho > 0) |
   (reference_direction == "Young_up_concordant" & target_human_age_rho < 0)))

out_cols <- c("human_TF", "tf_symbol", "motif_id", "motif_name",
              "reference_direction", "peak", "peak_age_effect",
              "peak_direction_support", "target_peak_direction_support",
              "target_gene", "human_target_gene",
              "target_human_age_rho", "motif_p_adjust", "motif_direction_support",
              "full_chain_supported", "is_human_TF_candidate")
write.csv(net_full[, out_cols],
          file.path(outdir, "19_TF_motif_peak_target_validation.csv"), row.names = FALSE)
cat("TF chain rows:", nrow(net_full),
    "| human TF candidates with motif:", sum(net_full$is_human_TF_candidate %in% TRUE),
    "| full-chain supported:", sum(net_full$full_chain_supported %in% TRUE), "\n")

# ---- 7. Summary ----
n_gene_orth <- length(unique(j$human_gene[!is.na(j$human_gene)]))
n_gene_atac <- length(unique(j$human_gene[j$atac_detectable]))
# gene-level concordance is assessed only on ATAC-detectable genes
assess_gene <- j[j$atac_detectable %in% TRUE, ]
gene_ok <- sum(assess_gene$atac_direction_ok %in% TRUE, na.rm = TRUE)
gene_assess <- sum(assess_gene$atac_direction_ok %in% c(TRUE, FALSE), na.rm = TRUE)

# P1-3 (2026-09-15): exact binomial CIs/p for each ATAC concordance level.
binom_line <- function(label, k, n) {
  if (is.na(n) || n == 0) {
    return(sprintf("%s: 0/0 = NA (exact 95%% CI NA-NA, binomial p vs 0.5 NA)", label))
  }
  bt <- tryCatch(binom.test(k, n, p = 0.5), error = function(e) NULL)
  if (is.null(bt)) {
    return(sprintf("%s: %d/%d = %.1f%% (exact CI unavailable)", label, k, n, 100*k/n))
  }
  sprintf("%s: %d/%d = %.1f%% (exact 95%% CI %.1f-%.1f%%, binomial p vs 0.5 = %.3g)",
          label, k, n, 100 * k / n,
          100 * unname(bt$conf.int[1]), 100 * unname(bt$conf.int[2]),
          unname(bt$p.value))
}

summary_lines <- c(
  sprintf("human candidate genes with mouse ortholog: %d", n_gene_orth),
  sprintf("human genes with mouse ATAC gene-level effect: %d", n_gene_atac),
  binom_line("gene-level ATAC concordance (detectable only)", gene_ok, gene_assess),
  binom_line("peak-level MAIN concordance", n_ok_peak, n_assess_peak),
  binom_line("peak-level HIGH-CONFIDENCE concordance", n_ok_peak_hi, nrow(assess_peak_hi)),
  sprintf("candidate TFs with high-confidence regulatory motif: %d (RNA support %d, ATAC support %d)",
          n_tf_highconf, n_tf_rna_support, n_tf_atac_support),
  sprintf("TF->motif->peak->target chain rows: %d (human TF candidates with motif: %d, full chain supported: %d)",
          nrow(net_full),
          sum(net_full$is_human_TF_candidate %in% TRUE, na.rm = TRUE),
          sum(net_full$full_chain_supported %in% TRUE, na.rm = TRUE))
)
writeLines(summary_lines, file.path(outdir, "19_scATAC_validation_summary.txt"))
cat("\nSummary:\n"); cat(paste(summary_lines, collapse = "\n"), "\n")

# ---- 8. Figure: peak direction concordance by human direction ----
ok_df <- assess_peak[!is.na(assess_peak$peak_direction_ok), ]
if (nrow(ok_df) > 0) {
  pdf(file.path(figdir, "19_scATAC_regulatory_concordance.pdf"),
      width = 8, height = 5)
  par(mar = c(5, 5, 3, 2))
  tab <- table(human_dir = ok_df$human_dir, peak_ok = ok_df$peak_direction_ok)
  if (all(c("Age_up", "Age_down") %in% rownames(tab)) &&
      ncol(tab) == 2) {
    barplot(t(tab), beside = TRUE,
            col = c("firebrick", "navy"),
            legend.text = c("concordant", "discordant"),
            main = "Mouse peak accessibility direction vs human candidate direction",
            xlab = "Human candidate direction (age-rho)", ylab = "Peaks")
  } else {
    cat("(table too sparse for barplot; skipping figure)\n")
  }
  dev.off()
  cat("Wrote peak-concordance figure\n")
}

cat("\n=== Step 19 COMPLETE ===\n")
sink()

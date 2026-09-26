#!/usr/bin/env Rscript
# Step 17: External human thymus atlas validation (GSE195812).
# GSE195812 has NO Young/Old design -> validate candidate localization across
# DN -> DP -> SP developmental states (NOT aging AUC). Per frozen design
# v2.1/2026-09-15, this is EXTERNAL DEVELOPMENTAL CONTEXT ONLY, NOT an aging
# replication layer; it does NOT contribute weight to the aging evidence score.
#
# Inputs:
#   04_machine_learning/ML_consensus_genes.txt (symbols, PRIMARY)
#   05_deep_learning/interpretation/gene_importance.csv (DL top-100)
#   06_interpretation/16_candidate_direction_table.csv (direction reference)
# Outputs (07_external_validation/):
#   17_external_metadata_probe.txt
#   17_candidate_gene_localization.csv
#   17_candidate_gene_top_states.csv
#   17_external_state_validation.csv
#   09_figures/17_external_localization_heatmap.pdf
#   10_results/logs/17_external_human_validation.log
proj <- Sys.getenv("PROJ", unset = "")
if (!nzchar(proj)) stop("Required env var PROJ is not set. Run: source scripts/00_project_config.sh")
extdir <- Sys.getenv("HUMAN_EXTERNAL_RAW", unset = "")
if (!nzchar(extdir)) stop("Required env var HUMAN_EXTERNAL_RAW is not set. Run: source scripts/00_project_config.sh")
outdir <- file.path(proj, "07_external_validation")
figdir <- file.path(proj, "09_figures")
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)
dir.create(figdir, recursive = TRUE, showWarnings = FALSE)

logf <- file.path(proj, "10_results/logs/17_external_human_validation.log")
sink(logf, split = TRUE)
cat("=== Step 17: External human validation (GSE195812) ===\n")

suppressMessages(library(Seurat))
suppressMessages(library(Matrix))

# P1: unified data accessor — Seurat v5 `layer=` vs v4 `slot=` compatibility.
get_data <- function(obj, assay = "RNA", layer = "data") {
  if (inherits(try(GetAssayData(obj, assay = assay, layer = layer), silent = TRUE),
               "try-error")) {
    return(GetAssayData(obj, assay = assay, slot = layer))
  }
  GetAssayData(obj, assay = assay, layer = layer)
}

# P0 (2026-09-16): expected-state-aware resolver. We must NOT default to
# orig.ident: in Seurat it frequently holds sample/project/sort-source labels,
# not developmental states. Score EVERY candidate (Idents + all metadata
# columns) by how many of the frozen expected developmental states it contains,
# and fail fast if none contains enough of them.
resolve_state_vector <- function(obj, expected_states, object_name) {
  candidates <- list(Idents = as.character(Idents(obj)))
  for (col in colnames(obj@meta.data)) {
    candidates[[col]] <- as.character(obj@meta.data[[col]])
  }
  scores <- vapply(candidates,
                   function(v) length(intersect(unique(v), expected_states)),
                   integer(1))
  best_name <- names(which.max(scores))
  best_score <- max(scores)
  if (best_score < 2) {
    stop(object_name,
         ": no identity/metadata column contains enough expected developmental ",
         "states (found ", best_score, ", need >=2). Expected: ",
         paste(expected_states, collapse = ", "),
         ". Inspect 17_external_metadata_probe.txt and freeze the correct column.")
  }
  state_vec <- candidates[[best_name]]
  present <- intersect(unique(state_vec), expected_states)
  unexpected <- setdiff(unique(state_vec), expected_states)
  list(state = state_vec, source = best_name, overlap = best_score,
       present = present, unexpected = unexpected, all_scores = scores)
}

# ---- candidates ----
ml_genes <- scan(file.path(proj, "04_machine_learning/ML_consensus_genes.txt"),
                 what = character(), quiet = TRUE)
# P0-2: DL top100 is read from the FROZEN Step 16 list (NOT head(100) of a
# re-computed gene_importance.csv — that file is not the authoritative list).
dl_top100_file <- file.path(proj, "06_interpretation/16_DL_top100_genes.txt")
if (!file.exists(dl_top100_file)) stop("Frozen DL top100 missing: ", dl_top100_file)
dl_top100 <- scan(dl_top100_file, what = character(), quiet = TRUE)
candidates <- unique(c(ml_genes, dl_top100))
cat("ML genes:", length(ml_genes), "| DL top100 (frozen):", length(dl_top100),
    "| union:", length(candidates), "\n")

probe_lines <- c("=== GSE195812 metadata probe ===")

# ---- Sort1 (DN123) probe ----
cat("\n=== Sort1 (DN123) ===\n")
s1 <- readRDS(file.path(extdir, "GSE195812_Sort1_DN123.rds.gz"))
cat("dims:", nrow(s1), "x", ncol(s1), "\n")
cat("Idents:", paste(sort(unique(as.character(Idents(s1)))), collapse = ", "), "\n")
cat("meta columns:", paste(colnames(s1@meta.data), collapse = ", "), "\n")
probe_lines <- c(probe_lines,
  sprintf("Sort1 dims: %d x %d", nrow(s1), ncol(s1)),
  sprintf("Sort1 Idents: %s", paste(sort(unique(as.character(Idents(s1)))), collapse = "; ")),
  sprintf("Sort1 meta: %s", paste(colnames(s1@meta.data), collapse = "; ")))
age_cols <- grep("age|donor|sample|patient|individual", colnames(s1@meta.data),
                 ignore.case = TRUE, value = TRUE)
cat("age/donor-like columns in Sort1:", paste(age_cols, collapse = ", "), "\n")
probe_lines <- c(probe_lines,
  sprintf("Sort1 age/donor cols: %s", paste(age_cols, collapse = "; ")))

# ---- Sort2 (ISP/DP/SP) ----
cat("\n=== Sort2 (ISP/DP/SP) ===\n")
s2 <- readRDS(file.path(extdir, "GSE195812_Sort2_ISP_DP_SP.rds.gz"))
cat("dims:", nrow(s2), "x", ncol(s2), "\n")
cat("meta columns:", paste(colnames(s2@meta.data), collapse = ", "), "\n")
probe_lines <- c(probe_lines,
  sprintf("Sort2 dims: %d x %d", nrow(s2), ncol(s2)),
  sprintf("Sort2 meta: %s", paste(colnames(s2@meta.data), collapse = "; ")))
age_cols2 <- grep("age|donor|sample|patient|individual", colnames(s2@meta.data),
                  ignore.case = TRUE, value = TRUE)
cat("age/donor-like columns in Sort2:", paste(age_cols2, collapse = ", "), "\n")
probe_lines <- c(probe_lines,
  sprintf("Sort2 age/donor cols: %s", paste(age_cols2, collapse = "; ")))

# P0-3: Sort1 and Sort2 are DIFFERENT sorting strategies + cell compositions +
# objects. Scheme B (MAIN): per-object gene-wise z-score, then compare RELATIVE
# localization within each object's developmental axis. Scheme A (sensitivity)
# would merge + joint-normalize; here we keep the per-object z-score as the
# official output and always retain source_object for downstream (Fig 4F is
# faceted by source_object, never raw-mean compared across objects).

# ---- helper: per-object localization with gene-wise z-scores ----
# state_vector is the EXPLICITLY resolved developmental-state vector (P0);
# this function never chooses an identity column on its own.
localize_object <- function(obj, obj_name, states, state_vector) {
  data_slot <- get_data(obj)
  if (is.null(data_slot) || length(data_slot) == 0) {
    stop("No normalized data available for ", obj_name)
  }
  sym_present <- candidates[candidates %in% rownames(data_slot)]
  cat("  ", obj_name, "candidate symbols present:", length(sym_present), "/",
      length(candidates), "\n")
  state_idx <- lapply(states, function(s) which(state_vector == s))
  names(state_idx) <- states
  state_idx <- state_idx[vapply(state_idx, length, integer(1)) > 0]
  if (length(state_idx) == 0) stop("No cells for any state in ", obj_name)

  res <- vector("list", length(sym_present))
  for (i in seq_along(sym_present)) {
    g <- sym_present[i]
    v <- as.numeric(data_slot[g, ])
    # gene-wise z-score across ALL cells of THIS object (Scheme B)
    z <- (v - mean(v)) / (sd(v) + 1e-9)
    res[[i]] <- do.call(rbind, lapply(names(state_idx), function(s) {
      cells <- state_idx[[s]]
      data.frame(
        gene = g,
        source_object = obj_name,
        developmental_stage = s,
        n_cells = length(cells),
        mean_expr = mean(v[cells]),
        median_expr = median(v[cells]),
        pct_expr = mean(v[cells] > 0) * 100,
        mean_zscore = mean(z[cells]),      # Scheme B primary
        median_zscore = median(z[cells]),
        stringsAsFactors = FALSE)
    }))
    if (i %% 100 == 0) cat("  ...", obj_name, i, "genes\n")
  }
  do.call(rbind, res)
}

# Sort1: DN1 -> DN2 -> DN3 developmental axis
# P1-2 (2026-09-15): use a frozen biological order, never alphabetical sort().
# P0 (2026-09-16): resolve the state column by expected-state overlap; do NOT
# assume orig.ident. Only frozen expected states enter the primary localization;
# extra identities are audit-only.
cat("\n=== Sort1 localization (DN1->DN2->DN3, Scheme B z-score) ===\n")
S1_BIOLOGICAL_ORDER <- c("DN1", "DN2", "DN3")
s1_resolved <- resolve_state_vector(s1, S1_BIOLOGICAL_ORDER, "Sort1")
cat("Sort1 state column:", s1_resolved$source,
    "| expected-state overlap:", s1_resolved$overlap, "\n")
cat("Sort1 expected states present:", paste(s1_resolved$present, collapse = ", "), "\n")
if (length(s1_resolved$unexpected) > 0)
  cat("Sort1 extra labels (audit-only, excluded from primary localization):",
      paste(head(s1_resolved$unexpected, 20), collapse = ", "), "\n")
s1_states <- S1_BIOLOGICAL_ORDER
loc1 <- localize_object(s1, "Sort1", s1_states, s1_resolved$state)
cat("Sort1 states used:", paste(unique(loc1$developmental_stage), collapse = ", "), "\n")
probe_lines <- c(probe_lines,
  sprintf("Sort1 selected state column: %s (overlap=%d)",
          s1_resolved$source, s1_resolved$overlap),
  sprintf("Sort1 expected states present: %s", paste(s1_resolved$present, collapse = "; ")),
  sprintf("Sort1 extra labels excluded: %s",
          paste(head(s1_resolved$unexpected, 50), collapse = "; ")))

# Sort2: ISP -> DP_CD3min -> DP_CD3plus -> CD4_SP / CD8_SP
# P1-2 (2026-09-15): frozen biological order (no alphabetical sort()).
# P0 (2026-09-16): same expected-state resolver; no Idents(s2) <- "orig.ident".
cat("\n=== Sort2 localization (ISP->DP->SP, Scheme B z-score) ===\n")
S2_BIOLOGICAL_ORDER <- c("ISP", "DP_CD3min", "DP_CD3plus", "CD4_SP", "CD8_SP")
s2_resolved <- resolve_state_vector(s2, S2_BIOLOGICAL_ORDER, "Sort2")
cat("Sort2 state column:", s2_resolved$source,
    "| expected-state overlap:", s2_resolved$overlap, "\n")
cat("Sort2 expected states present:", paste(s2_resolved$present, collapse = ", "), "\n")
if (length(s2_resolved$unexpected) > 0)
  cat("Sort2 extra labels (audit-only, excluded from primary localization):",
      paste(head(s2_resolved$unexpected, 20), collapse = ", "), "\n")
s2_states <- S2_BIOLOGICAL_ORDER
cat("Sort2 states:", paste(s2_states, collapse = ", "), "\n")
print(table(factor(s2_resolved$state, levels = s2_states)))
loc2 <- localize_object(s2, "Sort2", s2_states, s2_resolved$state)
probe_lines <- c(probe_lines,
  sprintf("Sort2 selected state column: %s (overlap=%d)",
          s2_resolved$source, s2_resolved$overlap),
  sprintf("Sort2 expected states present: %s", paste(s2_resolved$present, collapse = "; ")),
  sprintf("Sort2 extra labels excluded: %s",
          paste(head(s2_resolved$unexpected, 50), collapse = "; ")),
  sprintf("Sort2 states: %s", paste(s2_states, collapse = "; ")),
  sprintf("candidates matched Sort2: %d", length(unique(loc2$gene))))
writeLines(probe_lines, file.path(outdir, "17_external_metadata_probe.txt"))

loc <- rbind(loc1, loc2)
write.csv(loc, file.path(outdir, "17_candidate_gene_localization.csv"), row.names = FALSE)
cat("Wrote localization:", nrow(loc), "rows\n")

# ---- top state per gene per source_object (by mean_zscore, Scheme B) ----
top <- do.call(rbind, lapply(split(loc, list(loc$gene, loc$source_object)),
                             function(x) x[which.max(x$mean_zscore), ]))
rownames(top) <- NULL
# frozen immature set (defined here, before stage_class is computed below)
immature <- c("ISP", "DP_CD3min", "DN1", "DN2", "DN3")
top$stage_class <- ifelse(top$developmental_stage %in% immature, "immature", "mature")
write.csv(top, file.path(outdir, "17_candidate_gene_top_states.csv"), row.names = FALSE)

# P0-1 (2026-09-15): each gene may have two rows in the long `top` table (one
# per source_object: Sort1 + Sort2). Emit a UNIQUE gene-level wide table so
# Step 20 can merge one-row-per-gene without duplicating rows. If a gene is
# detected in only one object, the other object's columns are NA.
detected <- unique(top[, c("gene", "source_object")])
detected$detected <- TRUE
detected_wide <- reshape(detected, idvar = "gene", timevar = "source_object",
                         direction = "wide")
names(detected_wide) <- c("gene", "detected_Sort1", "detected_Sort2")

pivot_one <- function(df, obj) {
  sub <- df[df$source_object == obj, c("gene", "developmental_stage", "stage_class", "mean_zscore")]
  sub <- sub[!duplicated(sub$gene), ]  # one top-state row per gene already
  names(sub) <- c("gene", paste0("top_state_", obj), paste0("stage_class_", obj),
                  paste0("top_z_", obj))
  sub
}
context <- Reduce(function(a, b) merge(a, b, by = "gene", all = TRUE),
                  list(detected_wide,
                       pivot_one(top, "Sort1"),
                       pivot_one(top, "Sort2")))
write.csv(context, file.path(outdir, "17_candidate_gene_developmental_context.csv"),
          row.names = FALSE)
cat("Wrote unique gene-level developmental context:",
    nrow(context), "genes\n")

stopifnot("17 unique-gene context must be one row per gene" =
            !anyDuplicated(context$gene) > 0)

# ---- per-state summary = external_state_validation.csv (per source_object) ----
st_tab <- aggregate(cbind(mean_zscore, pct_expr) ~ developmental_stage + source_object,
                    loc, FUN = function(x) c(n_genes = length(x), mean = mean(x)))
state_df <- data.frame(
  source_object = st_tab$source_object,
  dev_stage = st_tab$developmental_stage,
  n_genes = st_tab$mean_zscore[, "n_genes"],
  mean_mean_zscore = st_tab$mean_zscore[, "mean"],
  mean_pct_expr = st_tab$pct_expr[, "mean"])
write.csv(state_df, file.path(outdir, "17_external_state_validation.csv"), row.names = FALSE)
print(state_df)

# ---- direction cross-check vs primary cohort (descriptive only) ----
dir_tab <- read.csv(file.path(proj, "06_interpretation/16_candidate_direction_table.csv"),
                    stringsAsFactors = FALSE)
# Python bools ("True"/"False") are read as character by read.csv; coerce.
for (bc in c("in_ML_consensus", "in_DL_top100")) {
  if (bc %in% colnames(dir_tab)) dir_tab[[bc]] <- as.logical(dir_tab[[bc]])
}
dir_tab <- dir_tab[dir_tab$gene %in% candidates, ]
# genes whose top external state is DP/SP-mature vs DN/ISP-immature, crossed with
# primary age-rho sign (old-high genes expected in mature states is NOT a formal
# test; reported as concordance count only) — per source_object separately
m <- merge(top[, c("gene", "source_object", "developmental_stage", "stage_class")],
           dir_tab, by = "gene")
m$rho_sign <- sign(m$WT_age_rho)
ct <- table(stage_class = m$stage_class, rho_sign = m$rho_sign,
            source_object = m$source_object)
cat("\nTop-state class x WT age-rho sign (descriptive, by source_object):\n")
print(ct)
write.csv(as.data.frame(ct), file.path(outdir, "17_topstate_x_rho_contingency.csv"),
          row.names = FALSE)

# ---- heatmap of top-40 ML candidates across states (faceted by source_object) ----
ml_ranked <- dir_tab[dir_tab$in_ML_consensus, ]
ml_ranked <- ml_ranked[order(ml_ranked$ml_best_median_pct), ]
top40 <- head(ml_ranked$gene, 40)
pdf(file.path(figdir, "17_external_localization_heatmap.pdf"), width = 10, height = 10)
par(mfrow = c(1, 2))
for (so in c("Sort1", "Sort2")) {
  loc_so <- loc[loc$source_object == so, ]
  if (so == "Sort1") {
    states_so <- S1_BIOLOGICAL_ORDER[S1_BIOLOGICAL_ORDER %in% unique(loc_so$developmental_stage)]
  } else {
    states_so <- S2_BIOLOGICAL_ORDER[S2_BIOLOGICAL_ORDER %in% unique(loc_so$developmental_stage)]
  }
  states_so <- c(states_so, setdiff(unique(loc_so$developmental_stage), states_so))
  mat <- sapply(states_so, function(s) {
    x <- loc_so[loc_so$developmental_stage == s, ]
    setNames(x$mean_zscore, x$gene)[top40]
  })
  mat <- as.matrix(mat)
  mat <- mat[complete.cases(mat), , drop = FALSE]
  cat("heatmap genes retained (", so, "):", nrow(mat), "\n")
  heatmap(mat, scale = "row", margins = c(8, 8),
          main = sprintf("Top ML candidates across %s dev states (gene-wise Z)",
                         so),
          xlab = "Developmental state", ylab = "Gene")
}
dev.off()
cat("Wrote heatmap (faceted by source_object)\n")

cat("\n=== Step 17 COMPLETE ===\n")
sink()

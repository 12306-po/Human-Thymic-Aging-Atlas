#!/usr/bin/env Rscript
# Prepare donor-level HRA007984 features that match the frozen GSE231906 schema.
# The script is deliberately fail-closed: no guessed metadata or cell labels.

suppressPackageStartupMessages({
  library(Seurat)
  library(Matrix)
})

args <- commandArgs(trailingOnly = TRUE)
get_arg <- function(flag, default = NA_character_) {
  hit <- which(args == flag)
  if (length(hit) == 0) return(default)
  if (hit[1] == length(args)) stop("Missing value after ", flag)
  args[hit[1] + 1]
}
required <- c("--rds", "--selected-dictionary", "--celltype-map", "--output")
missing_required <- required[is.na(vapply(required, get_arg, character(1)))]
if (length(missing_required)) stop("Missing arguments: ", paste(missing_required, collapse = ", "))

rds_path <- normalizePath(get_arg("--rds"), mustWork = TRUE)
dict_path <- normalizePath(get_arg("--selected-dictionary"), mustWork = TRUE)
map_path <- normalizePath(get_arg("--celltype-map"), mustWork = TRUE)
out_dir <- get_arg("--output")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
assay_name <- get_arg("--assay", "RNA")
min_cells <- as.integer(get_arg("--min-cells-per-context", "20"))

obj <- readRDS(rds_path)
if (!inherits(obj, "Seurat")) stop("The downloaded RDS is not a Seurat object")
if (!(assay_name %in% names(obj@assays))) stop("Assay not found: ", assay_name)
meta <- obj@meta.data
meta$cell_barcode <- rownames(meta)

audit <- data.frame(
  column = colnames(meta),
  class = vapply(meta, function(x) paste(class(x), collapse = ";"), character(1)),
  n_unique = vapply(meta, function(x) length(unique(x)), integer(1)),
  example_values = vapply(meta, function(x) paste(head(unique(as.character(x)), 6), collapse = " | "), character(1)),
  stringsAsFactors = FALSE
)
write.table(audit, file.path(out_dir, "metadata_column_audit.tsv"), sep = "\t", row.names = FALSE, quote = FALSE)

candidate_sets <- list(
  donor = c("donor_id", "donor", "sample", "sample_id", "orig.ident", "individual"),
  age = c("age_years", "age", "Age"),
  stage = c("age_group", "stage", "Stage", "group", "Group"),
  sex = c("sex", "Sex", "gender", "Gender"),
  health = c("health", "health_status", "disease", "condition", "status"),
  celltype = c("cell_type", "celltype", "CellType", "annotation", "cell_annotation", "subtype")
)
resolve_col <- function(kind, flag) {
  explicit <- get_arg(flag)
  if (!is.na(explicit)) {
    if (!(explicit %in% colnames(meta))) stop(flag, " column not found: ", explicit)
    return(explicit)
  }
  hits <- intersect(candidate_sets[[kind]], colnames(meta))
  if (length(hits) != 1) {
    stop("Cannot uniquely resolve ", kind, " metadata column. Candidates found: ",
         paste(hits, collapse = ", "), ". Review metadata_column_audit.tsv and pass ", flag)
  }
  hits
}

donor_col <- resolve_col("donor", "--donor-col")
age_col <- resolve_col("age", "--age-col")
stage_col <- resolve_col("stage", "--stage-col")
sex_col <- resolve_col("sex", "--sex-col")
health_col <- resolve_col("health", "--health-col")
celltype_col <- resolve_col("celltype", "--celltype-col")

mapping <- read.delim(map_path, check.names = FALSE, stringsAsFactors = FALSE, comment.char = "")
needed_map <- c("external_label", "broad_cell_type", "include", "review_status")
if (!all(needed_map %in% colnames(mapping))) stop("Cell-type map must contain: ", paste(needed_map, collapse = ", "))
mapping$include <- tolower(trimws(as.character(mapping$include))) %in% c("true", "1", "yes", "y")
observed_labels <- sort(unique(as.character(meta[[celltype_col]])))
map_observed <- data.frame(external_label = observed_labels, stringsAsFactors = FALSE)
map_observed <- merge(map_observed, mapping, by = "external_label", all.x = TRUE, sort = FALSE)
write.table(map_observed, file.path(out_dir, "observed_celltype_mapping_audit.tsv"), sep = "\t", row.names = FALSE, quote = FALSE)
if (any(is.na(map_observed$include)) || !any(map_observed$include)) {
  stop("Cell-type mapping is incomplete or has no author-approved include=TRUE rows. Review observed_celltype_mapping_audit.tsv")
}
approved <- mapping[mapping$include & toupper(mapping$review_status) %in% c("APPROVED", "AUTHOR_APPROVED"), ]
if (nrow(approved) == 0) stop("No cell-type mapping row is marked APPROVED or AUTHOR_APPROVED")

meta$donor_id <- as.character(meta[[donor_col]])
meta$age_years <- suppressWarnings(as.numeric(gsub("[^0-9.]", "", as.character(meta[[age_col]]))))
meta$stage <- as.character(meta[[stage_col]])
meta$sex <- as.character(meta[[sex_col]])
meta$health <- as.character(meta[[health_col]])
meta$external_celltype <- as.character(meta[[celltype_col]])
meta$broad_cell_type <- approved$broad_cell_type[match(meta$external_celltype, approved$external_label)]
meta$mapped <- !is.na(meta$broad_cell_type)

stage_ok <- grepl("adult|geriatric|elder|aged|old", meta$stage, ignore.case = TRUE)
health_bad <- grepl("tumou?r|cancer|thymoma|carcinoma|hyperplasia|disease", meta$health, ignore.case = TRUE)
meta$eligible <- stage_ok & !health_bad & meta$mapped
if (!any(meta$eligible)) stop("No mapped adult/geriatric non-diseased cells survived the audited filters")

constant_or_na <- function(x) {
  y <- unique(x[!is.na(x) & nzchar(as.character(x))])
  if (length(y) > 1) stop("Donor-level metadata is inconsistent within donor: ", paste(y, collapse = " | "))
  if (length(y) == 0) return(NA)
  y[1]
}
donors <- split(meta, meta$donor_id)
manifest_rows <- lapply(donors, function(d) data.frame(
  donor_id = constant_or_na(d$donor_id),
  age_years = constant_or_na(d$age_years),
  sex = constant_or_na(d$sex),
  stage = constant_or_na(d$stage),
  health = constant_or_na(d$health),
  n_cells_total = nrow(d),
  n_cells_mapped = sum(d$mapped),
  n_cells_eligible = sum(d$eligible),
  eligible = any(d$eligible),
  stringsAsFactors = FALSE
))
manifest <- do.call(rbind, manifest_rows)
manifest <- manifest[manifest$eligible, ]
if (anyDuplicated(manifest$donor_id)) stop("Donor IDs are not unique after aggregation")
if (nrow(manifest) < 4) warning("Fewer than four eligible external donors were identified")
write.table(manifest, file.path(out_dir, "hra007984_donor_manifest.tsv"), sep = "\t", row.names = FALSE, quote = FALSE)

eligible_cells <- rownames(meta)[meta$eligible & meta$donor_id %in% manifest$donor_id]
obj <- subset(obj, cells = eligible_cells)
meta <- meta[eligible_cells, , drop = FALSE]

counts <- tryCatch(
  GetAssayData(obj, assay = assay_name, layer = "counts"),
  error = function(e) GetAssayData(obj, assay = assay_name, slot = "counts")
)
if (nrow(counts) == 0 || ncol(counts) == 0) stop("Raw RNA counts are unavailable")
gene_symbols <- rownames(counts)

log_cpm <- function(cell_ids) {
  if (length(cell_ids) == 0) return(setNames(numeric(0), character(0)))
  sums <- Matrix::rowSums(counts[, cell_ids, drop = FALSE])
  total <- sum(sums)
  if (!is.finite(total) || total <= 0) stop("Zero library size in donor pseudobulk")
  setNames(log1p(as.numeric(sums) / total * 1e6), gene_symbols)
}

dict <- read.delim(dict_path, check.names = FALSE, stringsAsFactors = FALSE)
needed_dict <- c("feature", "feature_type", "gene", "celltype")
if (!all(needed_dict %in% colnames(dict))) stop("Selected dictionary lacks: ", paste(needed_dict, collapse = ", "))
if (anyDuplicated(dict$feature)) stop("Selected feature dictionary contains duplicate keys")

feature_matrix <- matrix(NA_real_, nrow = nrow(manifest), ncol = nrow(dict),
                         dimnames = list(manifest$donor_id, dict$feature))
coverage_rows <- list()
for (i in seq_len(nrow(manifest))) {
  donor <- manifest$donor_id[i]
  donor_cells <- rownames(meta)[meta$donor_id == donor]
  wt <- log_cpm(donor_cells)
  tab <- table(meta[donor_cells, "broad_cell_type"])
  total_mapped <- sum(tab)
  context_cache <- list()
  for (ct in unique(na.omit(dict$celltype[dict$feature_type == "gene_celltype"]))) {
    ct_cells <- donor_cells[meta[donor_cells, "broad_cell_type"] == ct]
    context_cache[[ct]] <- if (length(ct_cells) >= min_cells) log_cpm(ct_cells) else NULL
  }
  for (j in seq_len(nrow(dict))) {
    row <- dict[j, ]
    if (row$feature_type == "whole_thymus") {
      feature_matrix[i, j] <- unname(wt[row$gene])
    } else if (row$feature_type == "gene_celltype") {
      vec <- context_cache[[row$celltype]]
      if (!is.null(vec)) feature_matrix[i, j] <- unname(vec[row$gene])
    } else if (row$feature_type == "proportion") {
      feature_matrix[i, j] <- if (total_mapped > 0) unname(tab[row$celltype] / total_mapped) else NA_real_
    } else stop("Unknown feature_type: ", row$feature_type)
  }
  coverage_rows[[i]] <- data.frame(
    donor_id = donor, n_cells = length(donor_cells), n_mapped_cells = total_mapped,
    n_observed_selected_features = sum(!is.na(feature_matrix[i, ])),
    n_selected_features = ncol(feature_matrix), stringsAsFactors = FALSE
  )
}

con <- gzfile(file.path(out_dir, "external_feature_matrix.tsv.gz"), "wt")
write.table(data.frame(donor_id = rownames(feature_matrix), feature_matrix, check.names = FALSE),
            con, sep = "\t", row.names = FALSE, quote = FALSE, na = "NA")
close(con)
write.table(do.call(rbind, coverage_rows), file.path(out_dir, "pseudobulk_feature_coverage.tsv"),
            sep = "\t", row.names = FALSE, quote = FALSE)
audit_con <- gzfile(file.path(out_dir, "cell_metadata_audit.tsv.gz"), open = "wt")
write.table(meta[, c("cell_barcode", "donor_id", "age_years", "stage", "sex", "health",
                     "external_celltype", "broad_cell_type", "eligible")],
            audit_con, sep = "\t", row.names = FALSE, quote = FALSE)
close(audit_con)
cat("Prepared", nrow(feature_matrix), "external donors and", ncol(feature_matrix), "selected features\n")

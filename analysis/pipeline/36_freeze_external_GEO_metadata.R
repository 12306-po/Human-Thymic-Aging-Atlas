#!/usr/bin/env Rscript
# Step 36: download and freeze GEO sample metadata before writing adapters.
# The script does not infer age groups, biological units, pooling, or inclusion.

suppressPackageStartupMessages(library(GEOquery))

args <- commandArgs(trailingOnly = TRUE)
project <- Sys.getenv("PROJ", unset = "/data/zxy/projects/human_thymus_age_ML_DL")
if (length(args) >= 2 && args[1] == "--project") project <- args[2]
out <- file.path(project, "08_external_evidence", "metadata_freeze")
dir.create(out, recursive = TRUE, showWarnings = FALSE)

accessions <- c(
  "GSE288730", "GSE223049", "GSE221034", "GSE132136",
  "GSE15945", "GSE153562", "GSE240016"
)

summary_rows <- list()
for (accession in accessions) {
  message("Fetching GEO metadata: ", accession)
  gse <- getGEO(accession, GSEMatrix = TRUE, getGPL = FALSE)
  if (!is.list(gse)) gse <- list(gse)
  for (index in seq_along(gse)) {
    metadata <- Biobase::pData(gse[[index]])
    metadata$geo_accession_frozen <- rownames(metadata)
    metadata$series_accession_frozen <- accession
    metadata$platform_index_frozen <- index
    path <- file.path(out, sprintf("%s_platform%02d_sample_metadata.tsv", accession, index))
    write.table(metadata, path, sep = "\t", row.names = FALSE,
                quote = FALSE, na = "NA")

    template <- data.frame(
      dataset_id = accession,
      sample_id = rownames(metadata),
      include = NA,
      exclusion_reason = NA_character_,
      species = "Mus musculus",
      tissue_confirmed_thymus = NA,
      compartment = NA_character_,
      modality = NA_character_,
      age_value = NA_real_,
      age_unit = NA_character_,
      age_group = NA_character_,
      sex = NA_character_,
      treatment = NA_character_,
      biological_unit_id = NA_character_,
      biological_unit_type = NA_character_,
      animals_per_unit = NA_integer_,
      pairing_id_RNA_ATAC = NA_character_,
      batch = NA_character_,
      metadata_verified_by = NA_character_,
      metadata_verification_date = NA_character_,
      stringsAsFactors = FALSE
    )
    write.table(template,
                file.path(out, sprintf("%s_platform%02d_manual_inclusion_template.tsv",
                                       accession, index)),
                sep = "\t", row.names = FALSE, quote = FALSE, na = "NA")
    summary_rows[[length(summary_rows) + 1]] <- data.frame(
      accession = accession,
      platform_index = index,
      n_samples = nrow(metadata),
      metadata_file = path,
      freeze_time_UTC = format(Sys.time(), tz = "UTC", usetz = TRUE),
      stringsAsFactors = FALSE
    )
  }
}
summary <- do.call(rbind, summary_rows)
write.table(summary, file.path(out, "external_GEO_metadata_freeze_manifest.tsv"),
            sep = "\t", row.names = FALSE, quote = FALSE)
writeLines(c(
  "Manual review is mandatory before analysis.",
  "Do not infer biological replicates from cell count or GSM count.",
  "A pooled library is one biological pool unless source records establish otherwise.",
  "Freeze inclusion, age, sex, compartment, treatment, animal/pool IDs, and RNA/ATAC pairing."
), file.path(out, "README_metadata_freeze.txt"))
message("Frozen GEO metadata written to ", out)

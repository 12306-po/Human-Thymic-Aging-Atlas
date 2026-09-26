#!/usr/bin/env Rscript
## 17b_state_donor_counts.R
## One-off source-table step for Fig. 4 (review 3.3): count how many biological
## subjects contribute cells to each external developmental state, so that the
## fig4_dotplot_source.tsv export can carry n_donors.
##
## Input : HUMAN_EXTERNAL_RAW/GSE195812_Sort1_DN123.rds.gz and
##         .../GSE195812_Sort2_ISP_DP_SP.rds.gz (metadata only usage)
## Output: 07_external_validation/17_state_donor_counts.csv
##         columns: object, state, n_cells_state, n_donors
suppressMessages(library(Seurat))

proj <- Sys.getenv("PROJ", unset = "")
extdir <- Sys.getenv("HUMAN_EXTERNAL_RAW", unset = "")
if (!nzchar(proj) || !nzchar(extdir)) {
  stop("Required env vars PROJ and HUMAN_EXTERNAL_RAW must be set (source 00_project_config.sh)")
}
outdir <- file.path(proj, "07_external_validation")
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

# frozen state definitions (same as 17_external_human_validation.R / fig4.py)
jobs <- list(
  list(file = file.path(extdir, "GSE195812_Sort1_DN123.rds.gz"),
       object = "Sort1", states = c("DN1", "DN2", "DN3"),
       state_col = "orig.ident"),
  list(file = file.path(extdir, "GSE195812_Sort2_ISP_DP_SP.rds.gz"),
       object = "Sort2", states = c("ISP", "DP_CD3min", "DP_CD3plus"),
       state_col = "orig.ident"))

rows <- list()
for (j in jobs) {
  cat("[17b] loading", basename(j$file), "...\n")
  obj <- readRDS(j$file)
  md <- obj@meta.data
  if (!j$state_col %in% colnames(md)) stop(j$object, ": missing state column ", j$state_col)
  if (!"subjects" %in% colnames(md)) stop(j$object, ": missing 'subjects' metadata column")
  state <- as.character(md[[j$state_col]])
  subjects <- as.character(md$subjects)
  dat <- data.frame(state = state, subject = subjects, stringsAsFactors = FALSE)
  dat <- dat[dat$state %in% j$states, , drop = FALSE]
  if (nrow(dat) == 0) stop(j$object, ": no cells in frozen states")
  cnt <- aggregate(list(n_donors = dat$subject), dat["state"],
                   function(x) length(unique(x)))
  ncell <- table(dat$state)
  cnt$n_cells_state <- as.integer(ncell[cnt$state])
  cnt$object <- j$object
  cnt <- cnt[, c("object", "state", "n_cells_state", "n_donors")]
  cnt$n_donors <- as.integer(cnt$n_donors)
  rows[[j$object]] <- cnt
  cat("[17b]  ", j$object, "donor counts:\n")
  print(cnt)
  rm(obj); gc()
}
res <- do.call(rbind, rows)
write.csv(res, file.path(outdir, "17_state_donor_counts.csv"), row.names = FALSE)
cat("[17b] wrote", file.path(outdir, "17_state_donor_counts.csv"), "\n")
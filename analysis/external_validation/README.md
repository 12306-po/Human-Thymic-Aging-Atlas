# Frozen-model external feasibility workflows

This directory implements a fail-closed transfer test of the GSE231906
Elastic Net model in HRA007984 / Zenodo 13207776.  It does **not** merge the
cohorts and it never uses external ages for feature selection, imputation,
scaling, model fitting, calibration, or hyperparameter tuning.

## Inputs

1. The completed GSE231906 project, including
   `04_machine_learning/dataset/feature_matrix_all_combined.csv.gz`,
   `donor_metadata.csv`, and `feature_dictionary.csv`.
2. `thymus.sc.RDS` from Zenodo record 13207776 (published MD5:
   `9d323bb344796efc4d7aa859c112f2ba`).
3. An author-reviewed metadata mapping and cell-type mapping.  The preparation
   script prints candidate metadata columns and writes a mapping template when
   mappings are incomplete; it then stops without producing predictions.

## Run

```bash
project=/data/zxy/projects/human_thymus_age_ML_DL
repo="$project/Human-Thymic-Aging-Atlas"
out="$project/09_external_human/HRA007984_frozen_transfer"
rds="$project/09_external_human/raw/thymus.sc.RDS"

python "$repo/analysis/external_validation/41_freeze_primary_elasticnet.py" \
  --project "$project" --output "$out/frozen_model"

Rscript "$repo/analysis/external_validation/42_prepare_hra007984_seurat.R" \
  --rds "$rds" \
  --selected-dictionary "$out/frozen_model/selected_feature_dictionary.tsv" \
  --celltype-map "$repo/analysis/external_validation/hra007984_celltype_map.tsv" \
  --output "$out/prepared" \
  --donor-col donor_id --age-col age --stage-col age_group \
  --sex-col sex --health-col health --celltype-col cell_type

python "$repo/analysis/external_validation/43_score_hra007984_frozen.py" \
  --model "$out/frozen_model/frozen_elasticnet.json" \
  --external-matrix "$out/prepared/external_feature_matrix.tsv.gz" \
  --manifest "$out/prepared/hra007984_donor_manifest.tsv" \
  --primary-metadata "$project/04_machine_learning/dataset/donor_metadata.csv" \
  --output "$out/results"

python "$repo/analysis/external_validation/44_plot_hra007984_validation.py" \
  --results "$out/results" --output "$out/figures"
```

The convenience wrapper `run_hra007984_external_validation.sh` performs the
same sequence and writes SHA-256 checksums.  Edit the reviewed cell-type map
before running.  Exact column names in the published Seurat object must be
confirmed from `metadata_column_audit.tsv`; command-line examples above are
placeholders, not assumed facts.

## Interpretation gate

The run is labelled `completed_external_feasibility_test` only when at least
four eligible independent external donors have donor-resolved predictions and
numeric exact ages.  Even then, four donors are insufficient to establish a
validated clinical age clock.  If exact ages are unavailable, predictions are
retained but age-performance metrics are not calculated.

## Multi-cohort extension: Tabula Sapiens, Park and GSE147520

The extension reuses the exact `frozen_elasticnet.json` already used for
HRA007984. It never reselects features, refits coefficients, recalculates
scaling, or calibrates predictions in an external cohort.

The three new cohorts have different inferential roles:

| Cohort | Prespecified donors | Evaluation |
|---|---:|---|
| Tabula Sapiens v1 thymus | TSP2 (61 y), TSP14 (51 y) | exact-age frozen transfer |
| Park E-MTAB-8581 | A16 (20–25 y), A43 (35–40 y) | interval-age transfer; no midpoint MAE/R² |
| GSE147520 | one 25-y adult | stromal-feature sensitivity only |

The combined exact-age summary contains only the four HRA007984 adults plus
the two Tabula Sapiens adults. Park age bands and GSE147520 stromal output are
reported separately and never added to the six-donor MAE/correlation/R².

### New code

- `45_prepare_external_h5ad.py`: audited raw-count H5AD to donor pseudobulk.
- `46_score_external_frozen.py`: unchanged preprocessing/model transfer.
- `47_integrate_multicohort_results.py`: six-donor exact-age summary plus
  separate interval and stromal outputs.
- `48_plot_multicohort_external.py`: PDF/SVG vector figure and 600-dpi PNG.
- `cohort_configs/`: reviewed donor manifests and initial cell-type crosswalks.
- `run_multicohort_external_validation.sh`: complete server workflow.

### Required input files

Download and decompress, when necessary:

1. Tabula Sapiens v1 thymus H5AD (CELLxGENE dataset UUID
   `0ced5e76-6040-47ff-8a72-93847965afc0`; 33,664 cells). Use the original v1
   dataset for the prespecified TSP2/TSP14 analysis, not a later integrated
   atlas unless a new protocol is declared.
2. Park E-MTAB-8581 `HTA07.A01.v02.entire_data_raw_count.h5ad` plus the
   matching annotated H5AD or per-cell metadata from
   `thymus_annotated_matrix_files.zip` (Zenodo record 3711134). The files are
   aligned by observation/cell-barcode names.
3. GSE147520 `GSE147520_all_cells.h5ad` or a raw-count reconstruction of the
   adult GSM4466786 library. Do not use the combined Bautista/Park object for
   both cohorts because that would duplicate Park donors.

Raw counts are mandatory. For CELLxGENE downloads they are normally stored in
`adata.raw.X`; for a raw-count-only file they may be in `adata.X`. The code
checks that the chosen values are non-negative and integer-like and stops if a
normalized/integrated matrix is selected.

### First pass: metadata audit

Column names differ among public releases. Run an audit before scoring:

```bash
project=/data/zxy/projects/human_thymus_age_ML_DL
repo="$project/Human-Thymic-Aging-Atlas"

export PROJECT="$project"
export REPO="$repo"
export PYTHON_BIN=/data/zxy/environments/micromamba_envs/human_thymus_age_ml_dl/bin/python
export TABULA_H5AD=/data/zxy/external/tabula_sapiens/0ced5e76-6040-47ff-8a72-93847965afc0.h5ad
export PARK_H5AD=/data/zxy/external/park/E-MTAB-8581_raw_counts.h5ad
export PARK_METADATA_H5AD=/data/zxy/external/park/E-MTAB-8581_annotated.h5ad
export GSE147520_H5AD=/data/zxy/external/GSE147520/GSE147520_all_cells.h5ad
export AUDIT_ONLY=1

bash "$repo/analysis/external_validation/run_multicohort_external_validation.sh"
```

Review each cohort's `prepared/metadata_column_audit.tsv`,
`observed_donor_audit.tsv`, and `observed_celltype_mapping_audit.tsv`. If the
release uses different metadata columns, set, for example:

```bash
export PARK_DONOR_COL=Donor
export PARK_CELLTYPE_COL=celltype
export PARK_MATRIX_SOURCE=X

export GSE147520_DONOR_COL=Age
export GSE147520_CELLTYPE_COL=annotation
export GSE147520_MATRIX_SOURCE=raw
```

Every observed target-donor cell label must be explicitly marked APPROVED in
its TSV crosswalk, including labels deliberately excluded with `include=FALSE`.
This prevents silent guesses about biological correspondence.

### Full run

After the audits and crosswalks are resolved:

```bash
export AUDIT_ONLY=0
bash "$repo/analysis/external_validation/run_multicohort_external_validation.sh" \
  2>&1 | tee "$PROJECT/logs/multicohort_external_validation.log"
```

Outputs are written under
`09_external_human/multicohort_frozen_transfer/`. The main files are:

- `integrated/combined_six_donor_exact_age_predictions.tsv`
- `integrated/exact_age_cohort_metrics.tsv`
- `integrated/park_interval_age_predictions.tsv`
- `integrated/gse147520_stromal_sensitivity.tsv`
- `integrated/multicohort_external_summary.json`
- `figures/Figure_external_multicohort_validation.pdf`
- `SHA256SUMS.txt`

### Interpretation boundaries

- Tabula Sapiens donor clinical backgrounds must be reported; normal tissue is
  not synonymous with a prospectively recruited healthy-volunteer cohort.
- Park ages remain intervals. Midpoints are not substituted as exact ages.
- GSE147520 is stromal-enriched. Whole-thymus and composition features are
  deliberately left missing and receive the frozen primary-training medians.
  Its output is a feature-restricted sensitivity projection, not an external
  validation of the whole-thymus age model.
- Six exact-age adults remain a feasibility evaluation, not clinical
  validation of an age clock.


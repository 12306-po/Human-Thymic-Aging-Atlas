# HRA007984 frozen-model external feasibility test

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


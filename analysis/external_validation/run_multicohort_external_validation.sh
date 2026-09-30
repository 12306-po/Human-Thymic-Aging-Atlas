#!/usr/bin/env bash
set -euo pipefail

: "${PROJECT:?Set PROJECT to the human_thymus_age_ML_DL project directory}"
: "${TABULA_H5AD:?Set TABULA_H5AD to the Tabula Sapiens v1 thymus H5AD}"
: "${PARK_H5AD:?Set PARK_H5AD to the E-MTAB-8581 raw-count H5AD}"
: "${GSE147520_H5AD:?Set GSE147520_H5AD to the GSE147520 H5AD containing raw counts}"

REPO="${REPO:-$PROJECT/Human-Thymic-Aging-Atlas}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3)}"
HRA_OUT="${HRA_OUT:-$PROJECT/09_external_human/HRA007984_frozen_transfer}"
MODEL_DIR="${MODEL_DIR:-$HRA_OUT/frozen_model}"
OUT_ROOT="${OUT_ROOT:-$PROJECT/09_external_human/multicohort_frozen_transfer}"
CONFIG="$REPO/analysis/external_validation/cohort_configs"
SCRIPT="$REPO/analysis/external_validation"
AUDIT_ONLY="${AUDIT_ONLY:-0}"

TABULA_DONOR_COL="${TABULA_DONOR_COL:-donor_id}"
TABULA_CELLTYPE_COL="${TABULA_CELLTYPE_COL:-cell_type}"
TABULA_MATRIX_SOURCE="${TABULA_MATRIX_SOURCE:-raw}"
TABULA_GENE_SYMBOL_COLUMN="${TABULA_GENE_SYMBOL_COLUMN:-feature_name}"

PARK_DONOR_COL="${PARK_DONOR_COL:-donor}"
PARK_CELLTYPE_COL="${PARK_CELLTYPE_COL:-cell_type}"
PARK_MATRIX_SOURCE="${PARK_MATRIX_SOURCE:-X}"
PARK_GENE_SYMBOL_COLUMN="${PARK_GENE_SYMBOL_COLUMN:-}"
PARK_METADATA_H5AD="${PARK_METADATA_H5AD:-}"
PARK_CELL_METADATA="${PARK_CELL_METADATA:-}"
PARK_CELL_ID_COLUMN="${PARK_CELL_ID_COLUMN:-cell_barcode}"

GSE147520_DONOR_COL="${GSE147520_DONOR_COL:-donor}"
GSE147520_CELLTYPE_COL="${GSE147520_CELLTYPE_COL:-cell_type}"
GSE147520_MATRIX_SOURCE="${GSE147520_MATRIX_SOURCE:-raw}"
GSE147520_GENE_SYMBOL_COLUMN="${GSE147520_GENE_SYMBOL_COLUMN:-}"

mkdir -p \
  "$MODEL_DIR" \
  "$OUT_ROOT/tabula_sapiens/prepared" "$OUT_ROOT/tabula_sapiens/results" \
  "$OUT_ROOT/park_e_mtab_8581/prepared" "$OUT_ROOT/park_e_mtab_8581/results" \
  "$OUT_ROOT/gse147520/prepared" "$OUT_ROOT/gse147520/results" \
  "$OUT_ROOT/integrated" "$OUT_ROOT/figures"

if [[ ! -s "$MODEL_DIR/frozen_elasticnet.json" ]]; then
  "$PYTHON_BIN" "$SCRIPT/41_freeze_primary_elasticnet.py" \
    --project "$PROJECT" --output "$MODEL_DIR"
fi
MODEL="$MODEL_DIR/frozen_elasticnet.json"
DICT="$MODEL_DIR/selected_feature_dictionary.tsv"

gene_symbol_args=()
if [[ -n "$TABULA_GENE_SYMBOL_COLUMN" ]]; then
  gene_symbol_args=(--gene-symbol-column "$TABULA_GENE_SYMBOL_COLUMN")
fi
audit_args=()
if [[ "$AUDIT_ONLY" == "1" ]]; then audit_args=(--audit-only); fi
"$PYTHON_BIN" "$SCRIPT/45_prepare_external_h5ad.py" \
  --h5ad "$TABULA_H5AD" --selected-dictionary "$DICT" \
  --donor-manifest "$CONFIG/tabula_sapiens_donors.tsv" \
  --celltype-map "$CONFIG/tabula_sapiens_celltype_map.tsv" \
  --output "$OUT_ROOT/tabula_sapiens/prepared" --cohort "Tabula Sapiens" \
  --donor-col "$TABULA_DONOR_COL" --celltype-col "$TABULA_CELLTYPE_COL" \
  --matrix-source "$TABULA_MATRIX_SOURCE" --scope full_thymus \
  "${gene_symbol_args[@]}" "${audit_args[@]}"

gene_symbol_args=()
if [[ -n "$PARK_GENE_SYMBOL_COLUMN" ]]; then
  gene_symbol_args=(--gene-symbol-column "$PARK_GENE_SYMBOL_COLUMN")
fi
park_metadata_args=()
if [[ -n "$PARK_METADATA_H5AD" ]]; then
  park_metadata_args=(--metadata-h5ad "$PARK_METADATA_H5AD")
elif [[ -n "$PARK_CELL_METADATA" ]]; then
  park_metadata_args=(--cell-metadata "$PARK_CELL_METADATA" --cell-id-column "$PARK_CELL_ID_COLUMN")
fi
"$PYTHON_BIN" "$SCRIPT/45_prepare_external_h5ad.py" \
  --h5ad "$PARK_H5AD" --selected-dictionary "$DICT" \
  --donor-manifest "$CONFIG/park_e_mtab_8581_donors.tsv" \
  --celltype-map "$CONFIG/park_e_mtab_8581_celltype_map.tsv" \
  --output "$OUT_ROOT/park_e_mtab_8581/prepared" --cohort "Park E-MTAB-8581" \
  --donor-col "$PARK_DONOR_COL" --celltype-col "$PARK_CELLTYPE_COL" \
  --matrix-source "$PARK_MATRIX_SOURCE" --scope full_thymus \
  "${gene_symbol_args[@]}" "${park_metadata_args[@]}" "${audit_args[@]}"

gene_symbol_args=()
if [[ -n "$GSE147520_GENE_SYMBOL_COLUMN" ]]; then
  gene_symbol_args=(--gene-symbol-column "$GSE147520_GENE_SYMBOL_COLUMN")
fi
"$PYTHON_BIN" "$SCRIPT/45_prepare_external_h5ad.py" \
  --h5ad "$GSE147520_H5AD" --selected-dictionary "$DICT" \
  --donor-manifest "$CONFIG/gse147520_donors.tsv" \
  --celltype-map "$CONFIG/gse147520_celltype_map.tsv" \
  --output "$OUT_ROOT/gse147520/prepared" --cohort "GSE147520" \
  --donor-col "$GSE147520_DONOR_COL" --celltype-col "$GSE147520_CELLTYPE_COL" \
  --matrix-source "$GSE147520_MATRIX_SOURCE" --scope stromal_sensitivity \
  --compatible-celltypes "TEC,Fibroblast" \
  "${gene_symbol_args[@]}" "${audit_args[@]}"

if [[ "$AUDIT_ONLY" == "1" ]]; then
  echo "Metadata audits completed. Review every observed donor/cell-type mapping before scoring."
  exit 0
fi

PRIMARY_META="$PROJECT/04_machine_learning/dataset/donor_metadata.csv"
"$PYTHON_BIN" "$SCRIPT/46_score_external_frozen.py" \
  --model "$MODEL" \
  --external-matrix "$OUT_ROOT/tabula_sapiens/prepared/external_feature_matrix.tsv.gz" \
  --manifest "$OUT_ROOT/tabula_sapiens/prepared/external_donor_manifest.tsv" \
  --primary-metadata "$PRIMARY_META" --output "$OUT_ROOT/tabula_sapiens/results" \
  --cohort "Tabula Sapiens" --analysis-type exact_age

"$PYTHON_BIN" "$SCRIPT/46_score_external_frozen.py" \
  --model "$MODEL" \
  --external-matrix "$OUT_ROOT/park_e_mtab_8581/prepared/external_feature_matrix.tsv.gz" \
  --manifest "$OUT_ROOT/park_e_mtab_8581/prepared/external_donor_manifest.tsv" \
  --primary-metadata "$PRIMARY_META" --output "$OUT_ROOT/park_e_mtab_8581/results" \
  --cohort "Park E-MTAB-8581" --analysis-type interval_age

"$PYTHON_BIN" "$SCRIPT/46_score_external_frozen.py" \
  --model "$MODEL" \
  --external-matrix "$OUT_ROOT/gse147520/prepared/external_feature_matrix.tsv.gz" \
  --manifest "$OUT_ROOT/gse147520/prepared/external_donor_manifest.tsv" \
  --primary-metadata "$PRIMARY_META" --output "$OUT_ROOT/gse147520/results" \
  --cohort "GSE147520" --analysis-type stromal_sensitivity

"$PYTHON_BIN" "$SCRIPT/47_integrate_multicohort_results.py" \
  --model "$MODEL" --hra-results "$HRA_OUT/results" \
  --tabula-results "$OUT_ROOT/tabula_sapiens/results" \
  --park-results "$OUT_ROOT/park_e_mtab_8581/results" \
  --gse147520-results "$OUT_ROOT/gse147520/results" \
  --output "$OUT_ROOT/integrated"

"$PYTHON_BIN" "$SCRIPT/48_plot_multicohort_external.py" \
  --results "$OUT_ROOT/integrated" --output "$OUT_ROOT/figures"

find "$OUT_ROOT" -type f ! -name SHA256SUMS.txt -print0 \
  | sort -z | xargs -0 sha256sum > "$OUT_ROOT/SHA256SUMS.txt"
echo "Completed multi-cohort frozen-model evaluation: $OUT_ROOT"

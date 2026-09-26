#!/usr/bin/env bash
set -euo pipefail

: "${PROJECT:?Set PROJECT to the GSE231906 project directory}"
: "${HRA_RDS:?Set HRA_RDS to thymus.sc.RDS from Zenodo 13207776}"
REPO="${REPO:-$PROJECT/Human-Thymic-Aging-Atlas}"
OUT="${OUT:-$PROJECT/09_external_human/HRA007984_frozen_transfer}"
CELLTYPE_MAP="${CELLTYPE_MAP:-$REPO/analysis/external_validation/hra007984_celltype_map.tsv}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3)}"
RSCRIPT_BIN="${RSCRIPT_BIN:-$(command -v Rscript)}"

mkdir -p "$OUT"/{frozen_model,prepared,results,figures}
actual_md5="$(md5sum "$HRA_RDS" | awk '{print $1}')"
expected_md5="9d323bb344796efc4d7aa859c112f2ba"
if [[ "$actual_md5" != "$expected_md5" ]]; then
  echo "ERROR: HRA007984 RDS MD5 mismatch: $actual_md5" >&2
  exit 1
fi

"$PYTHON_BIN" "$REPO/analysis/external_validation/41_freeze_primary_elasticnet.py" \
  --project "$PROJECT" --output "$OUT/frozen_model"

"$RSCRIPT_BIN" "$REPO/analysis/external_validation/42_prepare_hra007984_seurat.R" \
  --rds "$HRA_RDS" \
  --selected-dictionary "$OUT/frozen_model/selected_feature_dictionary.tsv" \
  --celltype-map "$CELLTYPE_MAP" \
  --output "$OUT/prepared" "${@:1}"

"$PYTHON_BIN" "$REPO/analysis/external_validation/43_score_hra007984_frozen.py" \
  --model "$OUT/frozen_model/frozen_elasticnet.json" \
  --external-matrix "$OUT/prepared/external_feature_matrix.tsv.gz" \
  --manifest "$OUT/prepared/hra007984_donor_manifest.tsv" \
  --primary-metadata "$PROJECT/04_machine_learning/dataset/donor_metadata.csv" \
  --output "$OUT/results"

"$PYTHON_BIN" "$REPO/analysis/external_validation/44_plot_hra007984_validation.py" \
  --results "$OUT/results" --output "$OUT/figures"

find "$OUT" -type f ! -name SHA256SUMS.txt -print0 | sort -z | xargs -0 sha256sum > "$OUT/SHA256SUMS.txt"
echo "Completed HRA007984 frozen-model feasibility workflow: $OUT"

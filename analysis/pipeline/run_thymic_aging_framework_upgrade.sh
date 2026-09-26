#!/usr/bin/env bash
# Build the atlas/score layer and prepare the cross-dataset evidence layer.
set -Eeuo pipefail

PROJECT_DIR=${PROJECT_DIR:-/data/zxy/projects/human_thymus_age_ML_DL}
SCRIPT_DIR=${SCRIPT_DIR:-$PROJECT_DIR/figures-0920}
PY_BIN=${PY_BIN:-/data/zxy/environments/micromamba_envs/human_thymus_age_ml_dl/bin/python}
R_BIN=${R_BIN:-/data/zxy/environments/micromamba_envs/r_validation/bin/Rscript}
MODE=${1:---check}

case "$MODE" in
  --check|--execute-core|--execute-with-external) ;;
  *) printf 'Usage: bash %s [--check|--execute-core|--execute-with-external]\n' "$0" >&2; exit 2 ;;
esac

[[ -x "$PY_BIN" ]] || { printf 'ERROR: Python not executable: %s\n' "$PY_BIN" >&2; exit 2; }
[[ -x "$R_BIN" ]] || { printf 'ERROR: Rscript not executable: %s\n' "$R_BIN" >&2; exit 2; }
for script in \
  14b_strict_oof_HumanThymusFormer.py \
  31_build_thymic_aging_score.py \
  32_build_human_thymic_aging_atlas.py \
  33_build_human_reference_effects.py \
  34_cross_dataset_evidence_validation.py \
  35_human_pathway_age_model.R; do
  [[ -s "$SCRIPT_DIR/$script" ]] || { printf 'ERROR: missing %s/%s\n' "$SCRIPT_DIR" "$script" >&2; exit 2; }
done

required=(
  '04_machine_learning/fully_nested/oof_predictions_all_models.tsv'
  '04_machine_learning/fully_nested/model_metrics.tsv'
  '03_feature_selection/composition_adjusted/composition_results.tsv'
  '01_raw_processing/metadata/submission_manifest/donor_manifest.csv'
  '05_deep_learning/dataset/tokens.csv.gz'
)
for relative in "${required[@]}"; do
  [[ -s "$PROJECT_DIR/$relative" ]] || {
    printf 'ERROR: prerequisite missing or empty: %s/%s\n' "$PROJECT_DIR" "$relative" >&2
    printf 'Run Steps 25-30 first, and retain the Step 14b dataset/checkpoints.\n' >&2
    exit 2
  }
done

if [[ -z "${MSIGDB_HALLMARK_GMT:-}" || ! -s "${MSIGDB_HALLMARK_GMT:-}" ]]; then
  printf 'ERROR: set MSIGDB_HALLMARK_GMT to the exact Hallmark GMT file.\n' >&2
  exit 2
fi

"$PY_BIN" -c 'import numpy,pandas,scipy,sklearn,torch,captum; print("Python framework dependencies: OK")'
"$R_BIN" -e 'stopifnot(requireNamespace("limma", quietly=TRUE), requireNamespace("edgeR", quietly=TRUE)); cat("R framework dependencies: OK\n")'

printf 'Project: %s\nCode: %s\nHallmark GMT: %s\n' "$PROJECT_DIR" "$SCRIPT_DIR" "$MSIGDB_HALLMARK_GMT"
printf 'Core order:\n'
printf '  14b strict OOF HumanThymusFormer with signed IG\n'
printf '  31 internal Thymic Aging Score\n'
printf '  35 signed pathway age model\n'
printf '  33 signed human reference effects\n'
printf '  32 Human Thymic Aging Atlas packet\n'
printf '  34 cross-dataset evidence integration\n'
if [[ "$MODE" == "--check" ]]; then
  printf 'Preflight OK. No analysis was run.\n'
  exit 0
fi

export PROJ="$PROJECT_DIR"
log_dir="$PROJECT_DIR/10_results/logs/thymic_aging_framework_upgrade"
mkdir -p "$log_dir"
run_id=$(date -u +%Y%m%dT%H%M%SZ)

run_py() {
  local script=$1
  shift
  "$PY_BIN" "$SCRIPT_DIR/$script" --project "$PROJECT_DIR" "$@" 2>&1 \
    | tee "$log_dir/${run_id}_${script%.*}.log"
}
run_r() {
  local script=$1
  shift
  "$R_BIN" "$SCRIPT_DIR/$script" --project "$PROJECT_DIR" "$@" 2>&1 \
    | tee "$log_dir/${run_id}_${script%.*}.log"
}

# Step 14b uses PROJ directly and does not accept --project.
"$PY_BIN" "$SCRIPT_DIR/14b_strict_oof_HumanThymusFormer.py" 2>&1 \
  | tee "$log_dir/${run_id}_14b_strict_oof_HumanThymusFormer.log"
run_py 31_build_thymic_aging_score.py
run_r 35_human_pathway_age_model.R --gmt "$MSIGDB_HALLMARK_GMT"
run_py 33_build_human_reference_effects.py
run_py 32_build_human_thymic_aging_atlas.py
if [[ "$MODE" == "--execute-with-external" ]]; then
  run_py 34_cross_dataset_evidence_validation.py
else
  run_py 34_cross_dataset_evidence_validation.py --allow-empty
fi

printf 'Framework upgrade finished. Logs: %s\n' "$log_dir"

#!/usr/bin/env bash
# Run the post-review completion analyses without overwriting Steps 00-24.
set -Eeuo pipefail

PROJECT_DIR=${PROJECT_DIR:-/data/zxy/projects/human_thymus_age_ML_DL}
SCRIPT_DIR=${SCRIPT_DIR:-$PROJECT_DIR/figures-0920}
PY_BIN=${PY_BIN:-/data/zxy/environments/micromamba_envs/human_thymus_age_ml_dl/bin/python}
R_BIN=${R_BIN:-/data/zxy/environments/micromamba_envs/r_validation/bin/Rscript}
MODE=${1:---check}

if [[ "$MODE" != "--check" && "$MODE" != "--execute" ]]; then
  printf 'Usage: bash %s [--check|--execute]\n' "$0" >&2
  exit 2
fi

[[ -d "$PROJECT_DIR" ]] || { printf 'ERROR: missing project: %s\n' "$PROJECT_DIR" >&2; exit 2; }
[[ -d "$SCRIPT_DIR" ]] || { printf 'ERROR: missing code directory: %s\n' "$SCRIPT_DIR" >&2; exit 2; }
[[ -x "$PY_BIN" ]] || { printf 'ERROR: Python not executable: %s\n' "$PY_BIN" >&2; exit 2; }
[[ -x "$R_BIN" ]] || { printf 'ERROR: Rscript not executable: %s\n' "$R_BIN" >&2; exit 2; }

steps=(
  'py|25_build_submission_donor_manifest.py'
  'py|26_fully_nested_donor_ML.py'
  'py|27_composition_aware_age.py'
  'R|28_cross_species_sex_adjusted.R'
  'R|29_expression_age_shape_sensitivity.R'
  'py|30_build_submission_result_packet.py'
)
for entry in "${steps[@]}"; do
  IFS='|' read -r kind script <<< "$entry"
  [[ -f "$SCRIPT_DIR/$script" ]] || {
    printf 'ERROR: missing script: %s/%s\n' "$SCRIPT_DIR" "$script" >&2
    exit 2
  }
done

required=(
  '01_raw_processing/metadata/04G_final_cohort_freeze.csv'
  '01_raw_processing/metadata/05B_final_regression_cohort.csv'
  '02_pseudobulk/donor_celltype_proportion_all.csv'
  '04_machine_learning/dataset/feature_matrix_all_combined.csv.gz'
  '03_feature_selection/limma_adjusted/whole_thymus_limma_age_sex.csv'
  '08_mouse_validation/18_mouse_human_orthologs.csv'
  '08_mouse_validation/19_scATAC_gene_level_join.csv'
  '08_mouse_validation/19_scATAC_peak_validation.csv'
)
for relative in "${required[@]}"; do
  [[ -s "$PROJECT_DIR/$relative" ]] || {
    printf 'ERROR: prerequisite missing or empty: %s/%s\n' "$PROJECT_DIR" "$relative" >&2
    exit 2
  }
done

"$PY_BIN" -c 'import numpy,pandas,scipy,sklearn; print("Python analysis dependencies: OK")'
"$R_BIN" -e 'stopifnot(requireNamespace("limma", quietly=TRUE), requireNamespace("edgeR", quietly=TRUE)); cat("R analysis dependencies: OK\n")'

printf 'Project: %s\nCode: %s\nPython: %s\nRscript: %s\n' \
  "$PROJECT_DIR" "$SCRIPT_DIR" "$PY_BIN" "$R_BIN"
printf 'New steps (existing exploratory outputs are retained):\n'
for entry in "${steps[@]}"; do
  IFS='|' read -r kind script <<< "$entry"
  printf '  %-2s %s\n' "$kind" "$SCRIPT_DIR/$script"
done

if [[ "$MODE" == "--check" ]]; then
  printf 'Preflight OK. No analysis was run. Re-run with --execute.\n'
  exit 0
fi

export PROJ="$PROJECT_DIR"
log_dir="$PROJECT_DIR/10_results/logs/submission_completion_25_30"
mkdir -p "$log_dir"
run_id=$(date -u +%Y%m%dT%H%M%SZ)

run_step() {
  local kind=$1
  local script=$2
  local log="$log_dir/${run_id}_${script%.*}.log"
  printf '[%s] START %s\n' "$(date -u +%FT%TZ)" "$script"
  if [[ "$kind" == "py" ]]; then
    "$PY_BIN" "$SCRIPT_DIR/$script" --project "$PROJECT_DIR" 2>&1 | tee "$log"
  else
    "$R_BIN" "$SCRIPT_DIR/$script" --project "$PROJECT_DIR" 2>&1 | tee "$log"
  fi
  printf '[%s] DONE  %s\n' "$(date -u +%FT%TZ)" "$script"
}

for entry in "${steps[@]}"; do
  IFS='|' read -r kind script <<< "$entry"
  run_step "$kind" "$script"
done

printf 'Submission-completion analyses finished. Logs: %s\n' "$log_dir"

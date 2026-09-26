#!/usr/bin/env bash
# Run the frozen analysis pipeline from THIS figures-0920 code directory.
# Default is a read-only preflight. Add --execute to rerun analysis outputs.
# The script deliberately does NOT call the old scripts/ launcher or plot figures.
set -Eeuo pipefail

PROJECT_DIR=/data/zxy/projects/human_thymus_age_ML_DL
SCRIPT_DIR="$PROJECT_DIR/figures-0920"
PY_BIN="${PY_BIN:-/data/zxy/environments/micromamba_envs/human_thymus_age_ml_dl/bin/python}"
R_ENV_DIR="${R_ENV_DIR:-/data/zxy/environments/micromamba_envs/r_validation}"
R_BIN="${R_BIN:-$R_ENV_DIR/bin/Rscript}"
export PERM_N_JOBS="${PERM_N_JOBS:-8}"

usage() {
  printf 'Usage: bash %s [--check | --execute]\n' "$0"
  printf '  --check     validate files and print the exact run order (default)\n'
  printf '  --execute   rerun 00–24 and overwrite project-derived results\n'
}

mode=check
case "${1:---check}" in
  --check) mode=check ;;
  --execute) mode=execute ;;
  --help|-h) usage; exit 0 ;;
  *) usage >&2; exit 2 ;;
esac
if (( $# > 1 )); then
  usage >&2
  exit 2
fi

if [[ ! -d "$SCRIPT_DIR" ]]; then
  printf 'ERROR: code directory does not exist: %s\n' "$SCRIPT_DIR" >&2
  exit 2
fi
if [[ ! -f "$SCRIPT_DIR/00_project_config.sh" ]]; then
  printf 'ERROR: missing %s/00_project_config.sh\n' "$SCRIPT_DIR" >&2
  exit 2
fi
# The config establishes all raw-data paths and PROJ. It must come from the
# same code bundle, never from PROJECT_DIR/scripts/.
# shellcheck disable=SC1091
source "$SCRIPT_DIR/00_project_config.sh"
if [[ "$PROJ" != "$PROJECT_DIR" ]]; then
  printf 'ERROR: config PROJ=%s, expected %s\n' "$PROJ" "$PROJECT_DIR" >&2
  exit 2
fi

# Dependency order, including the letter-suffixed steps needed by the revised
# figures. 21–24 are run after the main 20-step analysis; their prerequisites
# (Step 08/10/11/12 etc.) have therefore already been rebuilt.
steps=(
  'py|00_check_environment.py'
  'py|01_inventory_human_data.py'
  'py|02_pair_GSE231906_10x_libraries.py'
  'py|03_parse_GSE231906_GEO_metadata.py'
  'py|04_build_final_thymus_donor_metadata.py'
  'py|05_define_age_tasks.py'
  'py|06_read_merge_primary_10x_and_QC.py'
  'py|07_broad_cell_annotation.py'
  'py|08_build_pseudobulk_and_composition.py'
  'py|09_age_feature_selection.py'
  'py|10_prepare_ML_matrix.py'
  'py|11_nested_donor_ML.py'
  'py|12_integrate_ML_features.py'
  'py|13_build_HumanThymusFormer_dataset.py'
  'py|14_train_HumanThymusFormer.py'
  'py|14b_strict_oof_HumanThymusFormer.py'
  'py|15_interpret_HumanThymusFormer.py'
  'py|16_pathway_and_TF_interpretation.py'
  'R|17_external_human_validation.R'
  'R|17b_state_donor_counts.R'
  'py|18a_ortholog_mapping.py'
  'R|18_mouse_scRNA_cross_species_validation.R'
  'R|19_mouse_scATAC_validation.R'
  'py|20_final_multiomics_integration.py'
  'py|21_ablation_nested_lodo.py'
  'R|22_age_adjusted_pseudobulk_limma.R'
  'py|23_permutation_null_stability.py'
  'py|24_doublet_removed_sensitivity.py'
)

[[ -x "$PY_BIN" ]] || { printf 'ERROR: Python not executable: %s\n' "$PY_BIN" >&2; exit 2; }
[[ -x "$R_BIN" ]] || { printf 'ERROR: Rscript not executable: %s\n' "$R_BIN" >&2; exit 2; }
for tool in sha256sum flock; do
  command -v "$tool" >/dev/null || {
    printf 'ERROR: required command not found: %s\n' "$tool" >&2
    exit 2
  }
done
for var in HUMAN_PRIMARY_RAW HUMAN_EXTERNAL_RAW MOUSE_SCRNA_RAW MOUSE_SCATAC_RAW MOUSE_DERIVED; do
  path="${!var:-}"
  [[ -d "$path" ]] || {
    printf 'ERROR: %s is not a readable directory: %s\n' "$var" "$path" >&2
    exit 2
  }
done
for entry in "${steps[@]}"; do
  IFS='|' read -r kind script_name <<< "$entry"
  [[ -f "$SCRIPT_DIR/$script_name" ]] || {
    printf 'ERROR: missing step file: %s/%s\n' "$SCRIPT_DIR" "$script_name" >&2
    exit 2
  }
done

printf 'Project: %s\nCode:    %s\nPython:  %s\nRscript: %s\n' \
  "$PROJECT_DIR" "$SCRIPT_DIR" "$PY_BIN" "$R_BIN"
printf 'Step 23 PERM_N_JOBS=%s\n' "$PERM_N_JOBS"
printf 'Steps to execute (%s):\n' "${#steps[@]}"
for entry in "${steps[@]}"; do
  IFS='|' read -r kind script_name <<< "$entry"
  printf '  %-2s  %s\n' "$kind" "$SCRIPT_DIR/$script_name"
done
if [[ "$mode" == check ]]; then
  printf '\nPreflight OK. No analysis was run. Use --execute after backing up frozen outputs.\n'
  exit 0
fi

# The analysis scripts write to PROJ and may overwrite older derived tables.
# Lock the project so two copies of this runner cannot run simultaneously.
log_root="$PROJECT_DIR/10_results/logs/figures_0920_full_rerun"
mkdir -p "$log_root"
exec 9>"$log_root/.run.lock"
if ! flock -n 9; then
  printf 'ERROR: another figures-0920 rerun holds %s/.run.lock\n' "$log_root" >&2
  exit 2
fi
run_dir="$log_root/$(date -u +%Y%m%dT%H%M%SZ)_$$"
mkdir -p "$run_dir"
master="$run_dir/master.log"
printf 'run_dir=%s\ncode_dir=%s\npython=%s\nrscript=%s\nPERM_N_JOBS=%s\n' \
  "$run_dir" "$SCRIPT_DIR" "$PY_BIN" "$R_BIN" "$PERM_N_JOBS" > "$run_dir/run_metadata.txt"
sha256sum "$SCRIPT_DIR/00_project_config.sh" > "$run_dir/code_sha256.txt"
for entry in "${steps[@]}"; do
  IFS='|' read -r kind script_name <<< "$entry"
  sha256sum "$SCRIPT_DIR/$script_name" >> "$run_dir/code_sha256.txt"
done

finish() {
  rc=$?
  if (( rc != 0 )); then
    printf 'FAILED (exit %s); inspect %s\n' "$rc" "$run_dir" | tee -a "$master"
    printf '%s\n' "$rc" > "$run_dir/FAILED"
  else
    printf 'COMPLETE; logs: %s\n' "$run_dir" | tee -a "$master"
    printf '0\n' > "$run_dir/COMPLETE"
  fi
}
trap finish EXIT

cd "$PROJECT_DIR"
r_library_path="$R_ENV_DIR/lib"
if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
  r_library_path="$r_library_path:$LD_LIBRARY_PATH"
fi
index=0
for entry in "${steps[@]}"; do
  IFS='|' read -r kind script_name <<< "$entry"
  index=$((index + 1))
  tag="${script_name%.*}"
  step_log="$run_dir/$(printf '%02d' "$index")_${tag}.log"
  printf '[%s] START %02d/%02d %s\n' "$(date -Is)" "$index" "${#steps[@]}" "$script_name" | tee -a "$master"
  if [[ "$kind" == py ]]; then
    if "$PY_BIN" "$SCRIPT_DIR/$script_name" > "$step_log" 2>&1; then
      :
    else
      rc=$?
      printf '[%s] FAILED %s (exit %s); %s\n' "$(date -Is)" "$script_name" "$rc" "$step_log" | tee -a "$master"
      exit "$rc"
    fi
  else
    if env LD_LIBRARY_PATH="$r_library_path" \
        "$R_BIN" "$SCRIPT_DIR/$script_name" > "$step_log" 2>&1; then
      :
    else
      rc=$?
      printf '[%s] FAILED %s (exit %s); %s\n' "$(date -Is)" "$script_name" "$rc" "$step_log" | tee -a "$master"
      exit "$rc"
    fi
  fi
  printf '[%s] DONE   %02d/%02d %s\n' "$(date -Is)" "$index" "${#steps[@]}" "$script_name" | tee -a "$master"
done

# These outputs are required by the revised Figure 1–5/S01–S09 scripts.
required_outputs=(
  '04_machine_learning/audit/donor_alignment_audit.tsv'
  '04_machine_learning/ablation_nested/fold_predictions_ablation.csv'
  '04_machine_learning/permutation_null/gene_recurrence_null.csv'
  '04_machine_learning/permutation_null/null_gene_fold_counts.npy'
  '04_machine_learning/permutation_null/null_summary.json'
  '05_deep_learning/strict_oof/oof_IG_gene_consistency.csv'
  '07_external_validation/17_state_donor_counts.csv'
  '03_feature_selection/limma_adjusted/whole_thymus_limma_age_sex.csv'
  '08_mouse_validation/18_mouse_human_orthologs.csv'
  '08_mouse_validation/19_scATAC_gene_level_join.csv'
  '10_results/final_candidate_panel.csv'
  '10_results/sensitivity_doublet_removed/doublet_sensitivity_summary.json'
)
for rel in "${required_outputs[@]}"; do
  if [[ ! -s "$PROJECT_DIR/$rel" ]]; then
    printf 'ERROR: missing or empty required output: %s\n' "$PROJECT_DIR/$rel" | tee -a "$master"
    exit 1
  fi
done
printf 'Required-output check passed. Analysis complete; figures are not yet redrawn.\n' | tee -a "$master"

#!/bin/bash
# Formal unified rerun Step00 -> Step20 (frozen code, 2026-09-16).
# - sources 00_project_config.sh ONCE at the shell layer
# - python scripts : human_thymus_age_ml_dl env python
# - R scripts 17/18/19 : r_validation env Rscript
# - stop on first error (set -e); per-step logs + master console
# Usage: nohup bash scripts/run_full_rerun_20260916.sh > /dev/null 2>&1 &

set -euo pipefail

PROJ=/data/zxy/projects/human_thymus_age_ML_DL
cd "$PROJ"
source scripts/00_project_config.sh

PYENV=/data/zxy/environments/micromamba_envs/human_thymus_age_ml_dl
RENV=/data/zxy/environments/micromamba_envs/r_validation
# Absolute interpreter paths. Do NOT put the R env lib/ on a GLOBAL
# LD_LIBRARY_PATH (it could shadow the Python env's libstdc++ for torch/scanpy);
# the R env's lib path is scoped to R steps only inside run().

LOGDIR=10_results/logs/rerun_20260916
mkdir -p "$LOGDIR"
MASTER="$LOGDIR/00_master.log"

PY="$PYENV/bin/python"
RS="$RENV/bin/Rscript"

run () {
  local kind="$1"; local script="$2"; local tag="$3"
  local log="$LOGDIR/${tag}.log"
  echo "================================================================" | tee -a "$MASTER"
  echo "[$(date '+%F %T')] START $tag ($script)" | tee -a "$MASTER"
  if [ "$kind" = "py" ]; then
    "$PY" "scripts/$script" > "$log" 2>&1
  else
    # scope R env lib path to this step only
    env LD_LIBRARY_PATH="$RENV/lib:${LD_LIBRARY_PATH:-}" \
      "$RS" "scripts/$script" > "$log" 2>&1
  fi
  echo "[$(date '+%F %T')] DONE  $tag" | tee -a "$MASTER"
}

echo "FULL RERUN START $(date '+%F %T')" | tee -a "$MASTER"

run py 00_check_environment.py                         00_check_environment
run py 01_inventory_human_data.py                     01_inventory
run py 02_pair_GSE231906_10x_libraries.py             02_pairing
run py 03_parse_GSE231906_GEO_metadata.py             03_GEO_metadata
run py 04_build_final_thymus_donor_metadata.py        04_donor_metadata
run py 05_define_age_tasks.py                         05_age_tasks
run py 06_read_merge_primary_10x_and_QC.py            06_QC
run py 07_broad_cell_annotation.py                    07_annotation
run py 08_build_pseudobulk_and_composition.py         08_pseudobulk
run py 09_age_feature_selection.py                    09_feature_selection
run py 10_prepare_ML_matrix.py                        10_prepare_ML
run py 11_nested_donor_ML.py                          11_nested_ML
run py 12_integrate_ML_features.py                    12_integrate_ML
run py 13_build_HumanThymusFormer_dataset.py          13_build_DL_dataset
run py 14_train_HumanThymusFormer.py                  14_train_DL
run py 15_interpret_HumanThymusFormer.py              15_interpret_DL
run py 16_pathway_and_TF_interpretation.py            16_pathway_TF
run R  17_external_human_validation.R                 17_external_human
run py 18a_ortholog_mapping.py                        18a_ortholog
run R  18_mouse_scRNA_cross_species_validation.R      18_mouse_scRNA
run R  19_mouse_scATAC_validation.R                   19_mouse_scATAC
run py 20_final_multiomics_integration.py             20_final_integration

echo "FULL RERUN COMPLETE $(date '+%F %T')" | tee -a "$MASTER"

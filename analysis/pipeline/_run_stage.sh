#!/bin/bash
# Generic gated-stage runner.
# Usage: _run_stage.sh <manifest_file> <log_subdir> <stage_label>
# manifest lines:  py|relative_script.py|tag
#                  R |relative_script.R |tag
# Sentinels: <log_subdir>/STAGE_<label>_RUNNING / _DONE / _FAILED
set -uo pipefail

PROJ=/data/zxy/projects/human_thymus_age_ML_DL
cd "$PROJ"
# shellcheck disable=SC1091
source scripts/00_project_config.sh >/dev/null

PY=/data/zxy/environments/micromamba_envs/human_thymus_age_ml_dl/bin/python
RS=/data/zxy/environments/micromamba_envs/r_validation/bin/Rscript
RENV=/data/zxy/environments/micromamba_envs/r_validation

MANIFEST="$1"; LOGDIR="10_results/logs/$2"; LABEL="$3"
mkdir -p "$LOGDIR"
MASTER="$LOGDIR/${LABEL}_master.log"
touch "$LOGDIR/STAGE_${LABEL}_RUNNING"

echo "STAGE $LABEL START $(date '+%F %T')" | tee -a "$MASTER"
while IFS='|' read -r kind script tag; do
  [ -z "${kind:-}" ] && continue
  case "$kind" in \#*) continue;; esac
  log="$LOGDIR/${tag}.log"
  echo "---- [$(date '+%T')] START $tag ($script)" | tee -a "$MASTER"
  if [ "$kind" = "py" ]; then
    "$PY" "scripts/$script" > "$log" 2>&1
  else
    env LD_LIBRARY_PATH="$RENV/lib:${LD_LIBRARY_PATH:-}" "$RS" "scripts/$script" > "$log" 2>&1
  fi
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "---- [$(date '+%T')] FAILED $tag rc=$rc" | tee -a "$MASTER"
    echo "$rc" > "$LOGDIR/STAGE_${LABEL}_FAILED"
    exit $rc
  fi
  echo "---- [$(date '+%T')] DONE  $tag" | tee -a "$MASTER"
done < "$MANIFEST"

rm -f "$LOGDIR/STAGE_${LABEL}_RUNNING"
echo "0" > "$LOGDIR/STAGE_${LABEL}_DONE"
echo "STAGE $LABEL COMPLETE $(date '+%F %T')" | tee -a "$MASTER"

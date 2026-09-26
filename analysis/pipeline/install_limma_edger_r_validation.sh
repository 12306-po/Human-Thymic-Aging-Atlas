#!/usr/bin/env bash
set -euo pipefail

# Install limma and edgeR into the R environment used by step 22.
# Usage: bash install_limma_edger_r_validation.sh [--check|--plan|--execute]

MODE="${1:---check}"
case "$MODE" in
  --check|--plan|--execute) ;;
  *) echo "Usage: bash $0 [--check|--plan|--execute]" >&2; exit 2 ;;
esac

R_ENV_DIR="/data/zxy/environments/micromamba_envs/r_validation"
R_BIN="$R_ENV_DIR/bin/Rscript"

if [[ ! -x "$R_BIN" ]]; then
  echo "Rscript not found or not executable: $R_BIN" >&2
  exit 1
fi

echo "Target Rscript: $R_BIN"
"$R_BIN" -e 'cat("R version:", as.character(getRversion()), "\n"); for (p in c("limma", "edgeR")) cat(p, "installed:", requireNamespace(p, quietly=TRUE), "\n")'

if [[ "$MODE" == --check ]]; then
  echo "Check only. Use --plan to preview changes, then --execute to install."
  exit 0
fi

if [[ -n "${MICROMAMBA_BIN:-}" ]]; then
  PACKAGE_MANAGER="$MICROMAMBA_BIN"
elif command -v micromamba >/dev/null 2>&1; then
  PACKAGE_MANAGER="$(command -v micromamba)"
elif command -v mamba >/dev/null 2>&1; then
  PACKAGE_MANAGER="$(command -v mamba)"
elif command -v conda >/dev/null 2>&1; then
  PACKAGE_MANAGER="$(command -v conda)"
else
  echo "micromamba, mamba or conda was not found on PATH." >&2
  echo "If micromamba is installed elsewhere, set MICROMAMBA_BIN=/absolute/path/to/micromamba." >&2
  exit 1
fi

if [[ ! -x "$PACKAGE_MANAGER" ]]; then
  echo "Package manager not executable: $PACKAGE_MANAGER" >&2
  exit 1
fi

echo "Package manager: $PACKAGE_MANAGER"
INSTALL_ARGS=(install -p "$R_ENV_DIR" -c conda-forge -c bioconda bioconductor-limma bioconductor-edger)

if [[ "$MODE" == --plan ]]; then
  "$PACKAGE_MANAGER" "${INSTALL_ARGS[@]}" --dry-run
  echo "No packages were installed. Review the proposed changes before using --execute."
  exit 0
fi

"$PACKAGE_MANAGER" "${INSTALL_ARGS[@]}" -y

"$R_BIN" -e '
for (p in c("limma", "edgeR")) {
  if (!requireNamespace(p, quietly=TRUE)) stop(paste("Package unavailable after installation:", p))
  cat(p, as.character(packageVersion(p)), find.package(p), "\n")
}
'
echo "Installation and verification completed."

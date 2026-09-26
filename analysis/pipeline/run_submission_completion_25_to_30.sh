#!/usr/bin/env bash
# Preferred entry point. The older filename is retained as a compatibility shim.
set -Eeuo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec bash "$script_dir/run_submission_completion_25_to_29.sh" "$@"

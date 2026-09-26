#!/usr/bin/env python3
"""Step 00: environment check + provenance capture.

Verifies every package the pipeline imports and writes a frozen environment
record to 10_results/environment/ so downstream results are reproducible:
  - python_version.txt
  - package_versions.csv
  - pip_freeze.txt

Exit code is non-zero if any required package cannot be imported.
"""
from __future__ import annotations

import csv
import os
import subprocess
import sys
from pathlib import Path

print("=" * 70)
print("Human Thymus Aging ML/DL Environment Check")
print("=" * 70)

# name used in `pip freeze` / import may differ; (check_name, import_name)
REQUIRED = [
    ("numpy", None),
    ("pandas", None),
    ("scipy", None),
    ("scikit-learn", "sklearn"),
    ("scanpy", None),
    ("anndata", None),
    ("xgboost", None),
    ("shap", None),
    ("matplotlib", None),
    ("statsmodels", None),
    ("h5py", None),
    ("pyarrow", None),
    ("torch", None),
    ("captum", None),
    ("tensorboard", None),
    # P1 (2026-09-15 review): previously missing checks
    ("scrublet", None),
    ("gseapy", None),
    ("openpyxl", None),
    ("requests", None),
    ("igraph", None),
    ("leidenalg", None),
    ("scikit-misc", "skmisc"),
]

packages: dict[str, tuple[str, str]] = {}


def check(check_name: str, import_name: str | None = None) -> None:
    mod_name = import_name or check_name
    try:
        mod = __import__(mod_name)
        version = getattr(mod, "__version__", "unknown")
        packages[check_name] = ("OK", version)
    except Exception as exc:  # noqa: BLE001
        packages[check_name] = ("FAILED", str(exc))


for check_name, import_name in REQUIRED:
    check(check_name, import_name)

print("Python :", sys.version.split()[0])
print()

failed: list[str] = []
for name, (status, version) in packages.items():
    print(f"{name:15s} {status:8s} {version}")
    if status != "OK":
        failed.append(name)

print()

import torch  # noqa: E402

print("PyTorch device information")
print("--------------------------")
print("PyTorch :", torch.__version__)
print("CUDA    :", torch.cuda.is_available())
print("Threads :", torch.get_num_threads())

print("=" * 70)

# ---------------------------------------------------------------------------
# Environment provenance (P1, 2026-09-15 review)
# ---------------------------------------------------------------------------
proj = os.environ.get("PROJ")
if proj:
    env_dir = Path(proj) / "10_results" / "environment"
    env_dir.mkdir(parents=True, exist_ok=True)

    (env_dir / "python_version.txt").write_text(
        sys.version + "\n\n" + sys.executable + "\n"
    )

    with open(env_dir / "package_versions.csv", "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["package", "status", "version"])
        for name, (status, version) in packages.items():
            writer.writerow([name, status, version])

    try:
        freeze = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            check=True, capture_output=True, text=True,
        )
        (env_dir / "pip_freeze.txt").write_text(freeze.stdout)
        print(f"Environment provenance written to {env_dir}")
    except Exception as exc:  # noqa: BLE001
        # do not fail the check solely because pip freeze could not run
        print(f"WARNING: pip freeze capture failed: {exc}")
else:
    print("PROJ not set; skipping environment provenance write "
          "(run: source scripts/00_project_config.sh)")

if failed:
    print("FAILED PACKAGES:")
    for x in failed:
        print(" -", x)
    raise SystemExit(1)

print("ENVIRONMENT CHECK PASSED")

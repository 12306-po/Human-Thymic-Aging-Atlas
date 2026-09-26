from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export server-only donor composition and optional authentic cell-level H5AD."
    )
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-cell-h5ad", action="store_true")
    args = parser.parse_args()

    project = args.project.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    sources = {
        "donor_composition": project / "02_pseudobulk" / "donor_celltype_proportion_all.csv",
        "donor_composition_qc": project / "02_pseudobulk" / "pseudobulk_qc_summary_all.csv",
    }
    if args.include_cell_h5ad:
        sources["cell_level_h5ad"] = (
            project / "01_raw_processing" / "filtered" / "07_GSE231906_thymus_annotated.h5ad"
        )
    missing = [str(path) for path in sources.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing server files:\n" + "\n".join(missing))

    rows = []
    for logical_name, source in sources.items():
        target = output / source.name
        shutil.copy2(source, target)
        rows.append(
            {
                "logical_name": logical_name,
                "source_path": str(source),
                "exported_file": target.name,
                "bytes": target.stat().st_size,
                "sha256": sha256(target),
            }
        )
    pd.DataFrame(rows).to_csv(output / "server_export_manifest.csv", index=False)
    print(output)


if __name__ == "__main__":
    main()

"""Package only the code and corrected metadata used by the core website."""
import argparse
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

FILES = ["app.py", "atlas_core_data.py", "requirements-core.txt", "docs/core_site.md",
         "release/core_v1/data/donor_metadata.csv", "release/core_v1/manifest.json",
         "tests/test_atlas_core_data.py", "atlas_gene_data.py", "build_gene_reference_bundle.py",
         "import_corrected_gene_evidence.py", "tests/test_atlas_gene_data.py",
         ".streamlit/config.toml", "release/core_v1/data/gene_index.csv",
         "release/core_v1/data/gene_expression_reference.csv.gz",
         "release/core_v1/data/gene_development_reference.csv.gz",
         "assets/ATTRIBUTIONS.md", "assets/healthy_human_t_cell.jpg", "assets/thymus_lobule_nih_bioart.svg"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    if args.output.exists():
        raise FileExistsError(args.output)
    for name in FILES:
        if not (root / name).is_file():
            raise FileNotFoundError(name)
    with ZipFile(args.output, "x", compression=ZIP_DEFLATED) as archive:
        for name in FILES:
            archive.write(root / name, arcname=name)
    print(f"Website update package: {args.output}")


if __name__ == "__main__":
    main()

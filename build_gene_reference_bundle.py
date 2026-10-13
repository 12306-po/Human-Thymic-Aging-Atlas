"""Export age-independent expression references for the restored gene browser."""
import argparse
from pathlib import Path

from atlas_gene_data import gene_references, reference_files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    index, expression, development = gene_references(args.root)
    files = reference_files(index, expression, development)
    for name, data in files.items():
        path = args.root / name
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing reference bundle: {path}")
    for name, data in files.items():
        path = args.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    print(f"Exported {len(index)} gene symbols, {len(expression)} expression-reference rows, "
          f"{len(development)} developmental-reference rows")


if __name__ == "__main__":
    main()

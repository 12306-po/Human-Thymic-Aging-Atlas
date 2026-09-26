#!/usr/bin/env python
"""Check exported page sizes and required files after the layout rebuild.

This is a mechanical check, not a substitute for visual and data review.
Usage: python 100_check_paper_figures.py --output /path/to/new/figures
"""
from __future__ import annotations

import argparse
from pathlib import Path
import struct


EXPECTED_MM = {
    "Figure1": (183, 285),
    "Figure2": (183, 290),
    "Figure3": (183, 265),
    "Figure4": (183, 285),
    "Figure5": (183, 285),
}
REQUIRED_AUDITS = (
    "fig2a_top_six_genes_audit.tsv",
    "fig4_candidate_selection_audit.tsv",
    "fig4_dotplot_source.tsv",
    "fig4_state_coverage_audit.tsv",
    "fig4_stage_delta_audit.tsv",
    "figure_source_manifest.tsv",
)


def png_size(path: Path) -> tuple[int, int]:
    """Read PNG dimensions without requiring an image-processing package."""
    with path.open("rb") as handle:
        header = handle.read(24)
    if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise ValueError(f"Not a valid PNG header: {path}")
    return struct.unpack(">II", header[16:24])


def check(output: Path) -> None:
    problems = []
    for name, (width_mm, height_mm) in EXPECTED_MM.items():
        png = output / f"{name}.png"
        pdf = output / f"{name}.pdf"
        if not png.is_file() or not pdf.is_file() or pdf.stat().st_size == 0:
            problems.append(f"{name}: PNG/PDF missing or empty")
            continue
        try:
            actual = png_size(png)
        except ValueError as exc:
            problems.append(str(exc))
            continue
        target = (round(width_mm / 25.4 * 300), round(height_mm / 25.4 * 300))
        if any(abs(a - b) > 2 for a, b in zip(actual, target)):
            problems.append(f"{name}: PNG size {actual}, expected about {target}")
        else:
            print(f"{name}: {actual[0]} × {actual[1]} px (300 dpi)")
    for name in REQUIRED_AUDITS:
        file = output / name
        if not file.is_file() or file.stat().st_size == 0:
            problems.append(f"Missing/empty audit: {name}")
    supplementary = output / "Supplementary"
    if not supplementary.is_dir():
        problems.append("Supplementary directory missing")
    else:
        for extension in ("png", "pdf"):
            count = len(list(supplementary.glob(f"Supplementary_Figure_*.{extension}")))
            if count != 10:  # S01-S09 plus S08b
                problems.append(f"Supplementary {extension}: {count} files, expected 10")
    if problems:
        raise SystemExit("FAILED:\n- " + "\n- ".join(problems))
    print("Mechanical export checks passed. Inspect every panel and source table manually.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    check(args.output.expanduser().resolve())


if __name__ == "__main__":
    main()

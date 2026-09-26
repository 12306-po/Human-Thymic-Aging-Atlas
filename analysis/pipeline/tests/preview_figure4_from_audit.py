"""Render Figure 4 layout locally from a prior real audit export.

This is visual QA only. The publication build must still read the frozen
project source tables on the analysis server via 99_make_paper_figures.py.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import figure_common  # noqa: E402
from fig_paper import fig4  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    source = pd.read_csv(args.audit_dir / "fig4_dotplot_source.tsv", sep="\t")
    audit = pd.read_csv(args.audit_dir / "fig4_candidate_selection_audit.tsv",
                        sep="\t")
    genes = audit.loc[audit.displayed_main.eq(True), "gene"].tolist()
    if len(genes) != 20:
        raise ValueError("Expected the prior audit's 20 displayed genes")
    dc = (source[["object", "state", "n_cells", "n_donors"]]
          .drop_duplicates()
          .rename(columns={"n_cells": "n_cells_state"}))
    loc = (source.drop(columns=["n_donors"])
           .rename(columns={"object": "source_object",
                            "state": "developmental_stage",
                            "avg_expr": "mean_expr"}))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with patch.object(fig4.pd, "read_csv", side_effect=[loc, dc]), \
         patch.object(fig4, "_preselect_genes", return_value=(genes, audit)), \
         patch.object(fig4, "FIGF", args.output_dir), \
         patch.object(figure_common, "FIGF", args.output_dir):
        fig4.build()
    print(args.output_dir / "Figure4.png")


if __name__ == "__main__":
    main()

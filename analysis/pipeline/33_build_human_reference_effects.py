#!/usr/bin/env python
"""Step 33: build signed human reference effects for external evidence testing."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project", type=Path,
        default=Path(os.environ.get("PROJ", "/data/zxy/projects/human_thymus_age_ML_DL")),
    )
    return parser.parse_args()


def main() -> None:
    project = parse_args().project.resolve()
    output = project / "08_external_evidence"
    output.mkdir(parents=True, exist_ok=True)
    frames = []

    limma_dir = project / "03_feature_selection" / "limma_adjusted"
    for path in sorted(limma_dir.glob("*_limma_age_sex.csv")):
        table = pd.read_csv(path)
        required = {"gene", "logFC_age_perSD", "P.Value", "adj.P.Val"}
        if not required.issubset(table.columns):
            continue
        context = path.name.replace("_limma_age_sex.csv", "")
        feature_type = "gene" if context == "whole_thymus" else "gene_celltype"
        feature_id = table["gene"].astype(str) if context == "whole_thymus" else (
            table["gene"].astype(str) + "@@" + context
        )
        frames.append(pd.DataFrame({
            "feature_type": feature_type,
            "feature_id": feature_id,
            "gene": table["gene"].astype(str),
            "cell_context": context,
            "human_effect_age": table["logFC_age_perSD"],
            "human_p_value": table["P.Value"],
            "human_q_value": table["adj.P.Val"],
            "human_model": "sex-adjusted donor-level limma-voom",
            "source_file": str(path),
        }))

        if context == "whole_thymus":
            tf_path = project / "06_interpretation" / "16_TF_candidates.csv"
            if tf_path.exists() and tf_path.stat().st_size:
                tf_table = pd.read_csv(tf_path)
                if "TF" in tf_table.columns:
                    tf_names = set(tf_table["TF"].dropna().astype(str))
                    tf_effects = table[table["gene"].astype(str).isin(tf_names)]
                    frames.append(pd.DataFrame({
                        "feature_type": "tf",
                        "feature_id": tf_effects["gene"].astype(str),
                        "gene": tf_effects["gene"].astype(str),
                        "cell_context": "whole_thymus",
                        "human_effect_age": tf_effects["logFC_age_perSD"],
                        "human_p_value": tf_effects["P.Value"],
                        "human_q_value": tf_effects["adj.P.Val"],
                        "human_model": "sex-adjusted donor-level limma-voom TF expression",
                        "source_file": str(path),
                    }))

    composition_path = (
        project / "03_feature_selection" / "composition_adjusted" / "composition_results.tsv"
    )
    composition = pd.read_csv(composition_path, sep="\t")
    composition = composition[composition["term"].eq("age_z")]
    frames.append(pd.DataFrame({
        "feature_type": "cell_state",
        "feature_id": composition["cell_type"].astype(str),
        "gene": pd.NA,
        "cell_context": composition["cell_type"].astype(str),
        "human_effect_age": composition["estimate"],
        "human_p_value": composition["p_value"],
        "human_q_value": composition["q_value_BH"],
        "human_model": "sex-adjusted donor-level CLR composition model",
        "source_file": str(composition_path),
    }))

    # Optional signed pathway/program table. The required contract is explicit;
    # an unsigned over-representation table is intentionally not substituted.
    program_path = project / "03_feature_selection" / "program_age_effects.tsv"
    if program_path.exists() and program_path.stat().st_size:
        programs = pd.read_csv(program_path, sep="\t")
        required_program = {"program_id", "effect_age", "p_value", "q_value"}
        if not required_program.issubset(programs.columns):
            raise ValueError(
                "program_age_effects.tsv must contain: "
                + ", ".join(sorted(required_program))
            )
        frames.append(pd.DataFrame({
            "feature_type": "pathway",
            "feature_id": programs["program_id"].astype(str),
            "gene": pd.NA,
            "cell_context": programs.get("cell_context", "whole_thymus"),
            "human_effect_age": programs["effect_age"],
            "human_p_value": programs["p_value"],
            "human_q_value": programs["q_value"],
            "human_model": programs.get("model", "signed donor-level pathway model"),
            "source_file": str(program_path),
        }))

    reference = pd.concat(frames, ignore_index=True)
    reference["human_direction"] = np.where(
        reference["human_effect_age"] > 0, "Age_up",
        np.where(reference["human_effect_age"] < 0, "Age_down", pd.NA),
    )
    duplicate = reference.duplicated(["feature_type", "feature_id"], keep=False)
    if duplicate.any():
        reference[duplicate].to_csv(
            output / "human_reference_duplicate_keys.tsv", sep="\t", index=False
        )
        raise RuntimeError("Duplicate human reference keys; see audit table")
    reference.to_csv(output / "human_reference_effects.tsv", sep="\t", index=False)
    print(f"Human signed reference effects written to {output}")


if __name__ == "__main__":
    main()

"""Export checked T05 results for the existing Streamlit research site."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DEFAULT_T05 = ROOT / "release" / "transformer_annotation_v1"


def read_stage_table(directory: Path, preferred_name: str,
                     required: set[str]) -> pd.DataFrame:
    """Find a stage table by schema, allowing minor release filename changes."""
    if not directory.is_dir():
        raise FileNotFoundError(f"T05 stage directory missing: {directory}")
    paths = sorted(directory.glob("*.tsv"))
    paths.sort(key=lambda path: path.name != preferred_name)
    matches = []
    for path in paths:
        frame = pd.read_csv(path, sep="\t")
        if required.issubset(frame.columns):
            matches.append((path, frame))
    if not matches:
        raise FileNotFoundError(
            f"No TSV in {directory} has all required columns: {sorted(required)}"
        )
    if len(matches) > 1 and matches[0][0].name != preferred_name:
        raise ValueError(f"Ambiguous T05 tables: {[path.name for path, _ in matches]}")
    return matches[0][1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--t05c", type=Path,
                        default=DEFAULT_T05 / "t05c_nested_lodo_age_prediction_v1")
    parser.add_argument("--t05d", type=Path,
                        default=DEFAULT_T05 / "t05d_crossfitted_immune_age_gap_v1")
    parser.add_argument("--t05f", type=Path,
                        default=DEFAULT_T05 / "t05f_full_pipeline_permutation_v1")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "release" / "age_prediction_v1")
    args = parser.parse_args()

    raw = read_stage_table(
        args.t05c, "primary_model_oof_predictions.tsv",
        {"donor_id", "chronological_age", "predicted_age", "absolute_error",
         "is_age_extrapolation"},
    )
    gap = read_stage_table(
        args.t05d, "primary_crossfitted_immune_age_gap.tsv",
        {"donor_id", "chronological_age", "raw_predicted_age",
         "residualized_immune_age_gap_years"},
    )
    permutation = read_stage_table(
        args.t05f, "full_pipeline_permutation_summary.tsv",
        {"scheme", "n_permutations", "observed_mae", "observed_baseline_mae",
         "observed_cv_r2", "permutation_p_mae_improvement", "permutation_p_cv_r2"},
    )

    model = "scgpt_global_plus_composition__ridge"
    for label, frame in (("T05C", raw), ("T05D", gap)):
        if len(frame) != 18 or not frame["donor_id"].is_unique:
            raise ValueError(f"{label} must contain 18 unique donors")
        if "model" in frame.columns and not frame["model"].eq(model).all():
            raise ValueError(f"{label} contains a different model")
    if set(raw["donor_id"]) != set(gap["donor_id"]):
        raise ValueError("T05C and T05D donor IDs differ")
    if set(permutation["scheme"]) != {"unrestricted", "within_sex"}:
        raise ValueError("Both full-pipeline permutation schemes are required")

    combined = raw.merge(
        gap[["donor_id", "chronological_age", "raw_predicted_age",
             "residualized_immune_age_gap_years"]],
        on="donor_id", validate="one_to_one", suffixes=("", "_gap"),
    )
    if not np.allclose(combined["chronological_age"], combined["chronological_age_gap"]):
        raise ValueError("Chronological ages differ between stages")
    if not np.allclose(combined["predicted_age"], combined["raw_predicted_age"], atol=1e-4):
        raise ValueError("T05C and T05D predicted ages differ")
    if not np.allclose(
        np.abs(combined["predicted_age"] - combined["chronological_age"]),
        combined["absolute_error"], atol=1e-4,
    ):
        raise ValueError("Absolute errors do not match predictions")
    if not np.isfinite(combined["residualized_immune_age_gap_years"]).all():
        raise ValueError("A corrected age residual is not finite")

    donors = combined[["donor_id", "chronological_age",
                       "predicted_age", "absolute_error",
                       "residualized_immune_age_gap_years",
                       "is_age_extrapolation"]].copy()
    if "heldout_fold" in combined.columns:
        donors.insert(1, "heldout_fold", combined["heldout_fold"])
    donors = donors.rename(columns={
        "absolute_error": "absolute_error_years",
        "residualized_immune_age_gap_years": "corrected_gap_years",
        "is_age_extrapolation": "age_range_extrapolation",
    })
    donors = donors.sort_values(
        "heldout_fold" if "heldout_fold" in donors.columns else "donor_id"
    )
    donors["age_range_extrapolation"] = (
        donors["age_range_extrapolation"].astype(str).str.lower().eq("true")
    )

    reference = permutation.iloc[0]
    if not np.allclose(permutation["observed_mae"], reference["observed_mae"], atol=1e-4):
        raise ValueError("Observed MAE differs across permutation schemes")
    if not np.isclose(donors["absolute_error_years"].mean(), reference["observed_mae"], atol=0.05):
        raise ValueError("Permutation MAE differs from exported donor MAE")

    summary = {
        "status": "research_only",
        "source": "T05C/T05D/T05F frozen scGPT donor-level analyses",
        "primary_model": {
            "name": model,
            "mae_years": float(reference["observed_mae"]),
            "cv_r2": float(reference["observed_cv_r2"]),
            "baseline_mae_years": float(reference["observed_baseline_mae"]),
        },
        "n_donors": 18,
        "external_multi_donor_age_validation": False,
        "gap_scope": "Exploratory, age-bias-corrected donor residual; not a clinical immune age",
    }

    args.output.mkdir(parents=True, exist_ok=True)
    donors.to_csv(args.output / "donor_age_predictions.tsv", sep="\t", index=False)
    permutation.to_csv(args.output / "permutation_summary.tsv", sep="\t", index=False)
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"T05 website research bundle exported: {args.output}")


if __name__ == "__main__":
    main()

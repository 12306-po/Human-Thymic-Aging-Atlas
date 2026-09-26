#!/usr/bin/env python
"""Step 31: build an internal, donor-level Thymic Aging Score.

Primary score source: Step 26 fully nested Elastic Net OOF predictions.
Interpretation source: Step 14b strict-OOF HumanThymusFormer signed IG.

The score is a proof-of-concept internal representation. It is not a clinical
clock and is not assigned external-validity or health-outcome meaning.
"""
from __future__ import annotations

import argparse
import json
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
    parser.add_argument("--top-contributions", type=int, default=10)
    return parser.parse_args()


def loo_age_bias_residual(observed: np.ndarray, predicted: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    expected = np.full(len(observed), np.nan)
    residual = np.full(len(observed), np.nan)
    for index in range(len(observed)):
        train = np.arange(len(observed)) != index
        design = np.column_stack([np.ones(train.sum()), observed[train]])
        coefficients, _, _, _ = np.linalg.lstsq(design, predicted[train], rcond=None)
        expected[index] = coefficients[0] + coefficients[1] * observed[index]
        residual[index] = predicted[index] - expected[index]
    return expected, residual


def percentile_rank(values: pd.Series) -> pd.Series:
    return values.rank(method="average", pct=True) * 100.0


def aggregate_attribution(ig: pd.DataFrame, group_column: str) -> pd.DataFrame:
    return (
        ig.groupby(["test_donor", group_column], as_index=False)
        .agg(
            signed_expression_IG_years=("IG_signed_years", "sum"),
            absolute_expression_IG_years=("IG_abs_years", "sum"),
            n_tokens=("feature", "nunique"),
        )
        .rename(columns={"test_donor": "donor_id"})
    )


def top_rows(table: pd.DataFrame, entity_column: str, n: int) -> pd.DataFrame:
    rows = []
    for donor, donor_table in table.groupby("donor_id"):
        positive = donor_table.nlargest(n, "signed_expression_IG_years")
        negative = donor_table.nsmallest(n, "signed_expression_IG_years")
        for direction, selected in (("older-directed", positive), ("younger-directed", negative)):
            selected = selected.copy()
            selected["contribution_direction"] = direction
            selected["rank_within_direction"] = np.arange(1, len(selected) + 1)
            rows.append(selected)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def main() -> None:
    args = parse_args()
    project = args.project.resolve()
    ml_file = project / "04_machine_learning" / "fully_nested" / "oof_predictions_all_models.tsv"
    dl_prediction_file = project / "05_deep_learning" / "strict_oof" / "dl_strict_oof_predictions.csv"
    ig_file = project / "05_deep_learning" / "strict_oof" / "oof_IG_token_long.csv"
    metadata_file = project / "04_machine_learning" / "dataset" / "donor_metadata.csv"
    output = project / "05_deep_learning" / "thymic_aging_score"
    output.mkdir(parents=True, exist_ok=True)

    for path in (ml_file, dl_prediction_file, ig_file, metadata_file):
        if not path.exists() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Required input missing or empty: {path}")

    ml = pd.read_csv(ml_file, sep="\t")
    primary = ml[
        ml["cohort"].eq("primary_18_donors") & ml["model"].eq("ElasticNet")
    ].copy()
    if len(primary) != 18 or primary["donor_id"].nunique() != 18:
        raise AssertionError("Expected 18 unique fully nested ElasticNet OOF predictions")
    primary = primary.rename(columns={
        "y_true": "chronological_age",
        "y_pred": "elasticnet_OOF_predicted_age",
    })
    observed = primary["chronological_age"].to_numpy(float)
    predicted = primary["elasticnet_OOF_predicted_age"].to_numpy(float)
    expected, deviation = loo_age_bias_residual(observed, predicted)
    primary["expected_OOF_prediction_given_age_LOO"] = expected
    primary["raw_age_gap_years"] = predicted - observed
    primary["thymic_age_deviation_years"] = deviation
    primary["internal_thymic_biological_age"] = observed + deviation
    primary["deviation_percentile_within_18_donors"] = percentile_rank(
        primary["thymic_age_deviation_years"]
    )

    metadata = pd.read_csv(metadata_file, index_col=0)
    metadata.index = metadata.index.astype(str)
    metadata.index.name = "donor_id"
    primary = primary.merge(
        metadata[[column for column in ("sex", "binary_task_group") if column in metadata]],
        left_on="donor_id", right_index=True, how="left",
    )

    dl = pd.read_csv(dl_prediction_file).rename(columns={
        "y_pred": "HumanThymusFormer_strict_OOF_predicted_age",
        "y_true": "HumanThymusFormer_y_true",
    })
    if len(dl) != 18 or dl["donor_id"].nunique() != 18:
        raise AssertionError("Expected 18 unique strict-OOF HumanThymusFormer predictions")
    primary = primary.merge(
        dl[["donor_id", "HumanThymusFormer_strict_OOF_predicted_age"]],
        on="donor_id", how="left", validate="one_to_one",
    )
    primary["model_prediction_difference_EN_minus_HTF"] = (
        primary["elasticnet_OOF_predicted_age"]
        - primary["HumanThymusFormer_strict_OOF_predicted_age"]
    )
    primary["score_scope"] = "internal proof-of-concept; not clinically validated"
    primary.to_csv(output / "thymic_aging_score_internal.tsv", sep="\t", index=False)

    ig = pd.read_csv(ig_file)
    needed = {"test_donor", "feature", "gene", "celltype", "IG_signed_years", "IG_abs_years"}
    if not needed.issubset(ig.columns):
        raise RuntimeError(
            "Step 14b must be rerun with signed IG enabled. Missing columns: "
            + ", ".join(sorted(needed.difference(ig.columns)))
        )
    if set(primary["donor_id"]) != set(ig["test_donor"].astype(str)):
        raise AssertionError("Strict-OOF IG donors do not match score donors")

    token_columns = [
        "fold", "test_donor", "feature", "gene", "celltype", "IG_signed_years",
        "IG_abs_years", "rank", "rank_pct",
    ]
    ig[token_columns].rename(columns={"test_donor": "donor_id"}).to_csv(
        output / "donor_token_expression_contributions.tsv", sep="\t", index=False
    )
    gene = aggregate_attribution(ig, "gene")
    celltype = aggregate_attribution(ig, "celltype")
    gene.to_csv(output / "donor_gene_contributions.tsv", sep="\t", index=False)
    celltype.to_csv(output / "donor_cellstate_contributions.tsv", sep="\t", index=False)
    top_rows(gene, "gene", args.top_contributions).to_csv(
        output / "donor_top_gene_contributions.tsv", sep="\t", index=False
    )
    top_rows(celltype, "celltype", args.top_contributions).to_csv(
        output / "donor_top_cellstate_contributions.tsv", sep="\t", index=False
    )

    contract = {
        "name": "Internal Thymic Aging Score (proof-of-concept)",
        "primary_prediction_model": "fully nested ElasticNet OOF",
        "age_deviation": (
            "ElasticNet OOF predicted age minus the expected OOF prediction at the same "
            "chronological age, with the calibration line fitted on the other 17 donors"
        ),
        "interpretation_model": "strict-OOF HumanThymusFormer",
        "attribution_scope": (
            "signed Integrated Gradients for the expression channel only; not a causal effect "
            "and not a complete decomposition of the model output"
        ),
        "prohibited_claims": [
            "clinical biological-age clock",
            "individual health or prognosis",
            "external validation",
            "causal gene or cell-state contribution",
        ],
        "required_upgrade_for_clock_claim": (
            "larger human training cohort plus independent human test cohort and association "
            "with external thymic function or health outcomes"
        ),
    }
    (output / "thymic_aging_score_contract.json").write_text(
        json.dumps(contract, indent=2), encoding="utf-8"
    )
    print(f"Internal Thymic Aging Score written to {output}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Apply the unchanged GSE231906 Elastic Net to one external cohort."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def exact_spearman_p(x: np.ndarray, y: np.ndarray) -> float:
    observed = abs(float(stats.spearmanr(x, y).statistic))
    if len(y) > 8:
        return float(stats.spearmanr(x, y).pvalue)
    permuted = (
        abs(float(stats.spearmanr(x, np.asarray(perm)).statistic))
        for perm in itertools.permutations(y.tolist())
    )
    return float(np.mean([value >= observed - 1e-12 for value in permuted]))


def interval_distance(predicted: pd.Series, lower: pd.Series, upper: pd.Series) -> pd.Series:
    return pd.Series(
        np.where(predicted < lower, predicted - lower,
                 np.where(predicted > upper, predicted - upper, 0.0)),
        index=predicted.index,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--external-matrix", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--primary-metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cohort", required=True)
    parser.add_argument(
        "--analysis-type",
        choices=["exact_age", "interval_age", "stromal_sensitivity"],
        required=True,
    )
    parser.add_argument("--max-missing-fraction", type=float)
    args = parser.parse_args()

    slug = slugify(args.cohort)
    bundle = json.loads(args.model.read_text(encoding="utf-8"))
    selected = bundle["selected_features_before_variance"]
    external = pd.read_csv(args.external_matrix, sep="\t", index_col=0)
    manifest = pd.read_csv(args.manifest, sep="\t")
    primary = pd.read_csv(args.primary_metadata, index_col=0)
    external.index = external.index.astype(str)
    manifest["donor_id"] = manifest.donor_id.astype(str)
    primary.index = primary.index.astype(str)
    if not external.index.is_unique or not manifest.donor_id.is_unique:
        raise AssertionError("External feature and manifest donor IDs must be unique")
    if set(external.index) != set(manifest.donor_id):
        raise AssertionError("External feature matrix and manifest contain different donors")
    overlap = sorted(set(external.index) & set(primary.index))
    if overlap:
        raise AssertionError(f"External-primary donor ID overlap: {overlap}")

    missing_columns = [feature for feature in selected if feature not in external.columns]
    aligned = external.reindex(columns=selected).astype(float)
    missing_fraction = aligned.isna().mean(axis=1)
    default_limit = 0.99 if args.analysis_type == "stromal_sensitivity" else 0.50
    missing_limit = default_limit if args.max_missing_fraction is None else args.max_missing_fraction
    if args.analysis_type != "stromal_sensitivity" and missing_limit > 0.50:
        raise ValueError("Full-thymus exact/interval validation cannot allow >50% missing features")
    if args.analysis_type == "stromal_sensitivity":
        scopes = set(manifest.get("scope", pd.Series(dtype=str)).dropna().astype(str))
        if scopes != {"stromal_sensitivity"}:
            raise RuntimeError("Stromal analysis requires scope=stromal_sensitivity in the manifest")

    coverage = pd.DataFrame({
        "cohort": args.cohort,
        "donor_id": aligned.index,
        "analysis_type": args.analysis_type,
        "n_required_features": len(selected),
        "n_observed_features": aligned.notna().sum(axis=1).to_numpy(),
        "missing_fraction": missing_fraction.to_numpy(),
        "observed_fraction": (1.0 - missing_fraction).to_numpy(),
        "n_columns_absent_from_external_export": len(missing_columns),
        "maximum_allowed_missing_fraction": missing_limit,
    })
    failed = coverage.missing_fraction > missing_limit
    if failed.any():
        donors = coverage.loc[failed, "donor_id"].tolist()
        raise RuntimeError(f"External feature coverage below the prespecified threshold: {donors}")

    # Only the external expression matrix enters prediction. Age is joined later.
    values = aligned.to_numpy(float)
    medians = np.asarray(bundle["training_imputation_median"], dtype=float)
    if values.shape[1] != medians.size:
        raise AssertionError("Frozen imputation vector does not match selected feature order")
    values = np.where(np.isnan(values), medians[None, :], values)
    support = np.asarray(bundle["variance_support"], dtype=bool)
    values = values[:, support]
    feature_names = np.asarray(selected, dtype=object)[support]
    mean = np.asarray(bundle["scaler_mean"], dtype=float)
    scale = np.asarray(bundle["scaler_scale"], dtype=float)
    coefficients = np.asarray(bundle["elasticnet_coefficients"], dtype=float)
    if not (values.shape[1] == mean.size == scale.size == coefficients.size):
        raise AssertionError("Frozen transform dimensions are inconsistent")
    if np.any(scale <= 0):
        raise AssertionError("Frozen scaler contains a non-positive scale")
    standardized = (values - mean[None, :]) / scale[None, :]
    contributions = standardized * coefficients[None, :]
    prediction = contributions.sum(axis=1) + float(bundle["elasticnet_intercept"])
    pred = pd.DataFrame({
        "cohort": args.cohort,
        "donor_id": aligned.index,
        "predicted_age": prediction,
        "prediction_delta_from_frozen_intercept": contributions.sum(axis=1),
    })

    result = pred.merge(manifest, on=["cohort", "donor_id"], how="left", validate="one_to_one")
    result["chronological_age"] = pd.to_numeric(result.get("age_years"), errors="coerce")
    result["age_lower"] = pd.to_numeric(result.get("age_lower"), errors="coerce")
    result["age_upper"] = pd.to_numeric(result.get("age_upper"), errors="coerce")
    result["analysis_type"] = args.analysis_type
    result["external_age_used_during_prediction"] = False
    result["external_refitting"] = False
    result["external_calibration"] = False
    result["frozen_model_sha256"] = sha256(args.model)

    metrics: dict[str, object] = {
        "cohort": args.cohort,
        "analysis_type": args.analysis_type,
        "n_external_donors_predicted": int(len(result)),
        "external_age_used_during_prediction": False,
        "external_refitting": False,
        "external_calibration": False,
        "n_selected_features": int(len(selected)),
        "n_model_features": int(coefficients.size),
        "model_sha256": sha256(args.model),
        "maximum_allowed_missing_fraction": float(missing_limit),
        "mean_observed_feature_fraction": float(coverage.observed_fraction.mean()),
    }

    if args.analysis_type == "exact_age":
        evaluable = result.dropna(subset=["chronological_age", "predicted_age"]).copy()
        if len(evaluable) != len(result):
            raise RuntimeError("Every exact-age validation donor must have age_years")
        result["prediction_error"] = result.predicted_age - result.chronological_age
        result["absolute_error"] = result.prediction_error.abs()
        y = evaluable.chronological_age.to_numpy(float)
        p = evaluable.predicted_age.to_numpy(float)
        metrics.update({
            "n_external_donors_with_exact_age": int(len(evaluable)),
            "MAE_years": float(mean_absolute_error(y, p)),
            "RMSE_years": float(np.sqrt(mean_squared_error(y, p))),
        })
        if len(evaluable) >= 2 and np.unique(y).size >= 2:
            metrics["Spearman_rho"] = float(stats.spearmanr(y, p).statistic)
            metrics["Spearman_exact_or_asymptotic_p"] = exact_spearman_p(y, p)
        if len(evaluable) >= 3 and np.unique(y).size >= 2:
            metrics["R2"] = float(r2_score(y, p))
        metrics["claim_status"] = "independent_exact_age_feasibility_predictions"
        metrics["claim_boundary"] = (
            "Independent frozen-model feasibility evidence; the small cohort does not validate a clinical clock"
        )
    elif args.analysis_type == "interval_age":
        if result[["age_lower", "age_upper"]].isna().any().any():
            raise RuntimeError("Every interval-age donor must have age_lower and age_upper")
        if (result.age_lower > result.age_upper).any():
            raise RuntimeError("An age interval has lower > upper")
        result["within_reported_age_interval"] = (
            (result.predicted_age >= result.age_lower) & (result.predicted_age <= result.age_upper)
        )
        result["signed_distance_to_interval"] = interval_distance(
            result.predicted_age, result.age_lower, result.age_upper
        )
        result["absolute_distance_to_interval"] = result.signed_distance_to_interval.abs()
        metrics.update({
            "n_external_donors_with_age_interval": int(len(result)),
            "n_predictions_within_reported_interval": int(result.within_reported_age_interval.sum()),
            "mean_absolute_distance_to_interval_years": float(result.absolute_distance_to_interval.mean()),
        })
        metrics["claim_status"] = "independent_interval_age_feasibility_predictions"
        metrics["claim_boundary"] = (
            "Age bands are not exact ages; midpoint MAE, correlation, and R-squared were not calculated"
        )
    else:
        if len(result) != 1:
            raise RuntimeError("The prespecified GSE147520 stromal sensitivity analysis expects one adult donor")
        if result.chronological_age.notna().any():
            result["descriptive_prediction_error"] = result.predicted_age - result.chronological_age
        metrics["n_stromal_sensitivity_donors"] = int(len(result))
        metrics["claim_status"] = "feature_restricted_stromal_sensitivity_projection"
        metrics["claim_boundary"] = (
            "Stromal-enriched data cannot validate the whole-thymus clock; no inferential age metric is reported"
        )

    args.output.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output / f"{slug}_frozen_predictions.tsv", sep="\t", index=False)
    coverage.to_csv(args.output / f"{slug}_feature_coverage.tsv", sep="\t", index=False)
    contribution_frame = pd.DataFrame(contributions, index=aligned.index, columns=feature_names)
    contribution_frame.index.name = "donor_id"
    contribution_frame.reset_index().melt(
        id_vars="donor_id", var_name="feature", value_name="prediction_contribution_years"
    ).assign(cohort=args.cohort).to_csv(
        args.output / f"{slug}_feature_contributions.tsv.gz", sep="\t", index=False, compression="gzip"
    )
    (args.output / f"{slug}_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

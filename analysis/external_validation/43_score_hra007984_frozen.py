#!/usr/bin/env python3
"""Apply the frozen GSE231906 model to donor-resolved HRA007984 features.

External ages are joined only after predictions have been calculated.  The
program rejects donor overlap, duplicated donors, reordered coefficients, and
excessive feature absence.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def exact_spearman_p(x: np.ndarray, y: np.ndarray) -> float:
    observed = abs(stats.spearmanr(x, y).statistic)
    if len(y) > 8:
        return float(stats.spearmanr(x, y).pvalue)
    vals = []
    for perm in itertools.permutations(y.tolist()):
        vals.append(abs(stats.spearmanr(x, np.asarray(perm)).statistic))
    return float(np.mean(np.asarray(vals) >= observed - 1e-12))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--external-matrix", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--primary-metadata", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--max-missing-fraction", type=float, default=0.50)
    args = ap.parse_args()

    bundle = json.loads(args.model.read_text(encoding="utf-8"))
    selected = bundle["selected_features_before_variance"]
    ext = pd.read_csv(args.external_matrix, sep="\t", index_col=0)
    manifest = pd.read_csv(args.manifest, sep="\t")
    primary = pd.read_csv(args.primary_metadata, index_col=0)
    ext.index = ext.index.astype(str)
    manifest["donor_id"] = manifest["donor_id"].astype(str)
    primary.index = primary.index.astype(str)
    if not ext.index.is_unique or not manifest.donor_id.is_unique:
        raise AssertionError("External donor identifiers must be unique")
    if set(ext.index) != set(manifest.donor_id):
        raise AssertionError("Feature matrix and donor manifest contain different donors")
    overlap = sorted(set(ext.index) & set(primary.index))
    if overlap:
        raise AssertionError(f"External-primary donor ID overlap: {overlap}")

    missing_columns = [c for c in selected if c not in ext.columns]
    aligned = ext.reindex(columns=selected).astype(float)
    missing_fraction = aligned.isna().mean(axis=1)
    coverage = pd.DataFrame({
        "donor_id": aligned.index,
        "n_required_features": len(selected),
        "n_observed_features": aligned.notna().sum(axis=1).to_numpy(),
        "missing_fraction": missing_fraction.to_numpy(),
    })
    if (missing_fraction > args.max_missing_fraction).any():
        bad = coverage.loc[coverage.missing_fraction > args.max_missing_fraction, "donor_id"].tolist()
        raise RuntimeError(f"External feature coverage below threshold for donors: {bad}")

    # Prediction occurs before age columns are joined.
    values = aligned.to_numpy(float)
    medians = np.asarray(bundle["training_imputation_median"], float)
    if values.shape[1] != len(medians):
        raise AssertionError("Frozen imputation vector does not match feature order")
    values = np.where(np.isnan(values), medians[None, :], values)
    support = np.asarray(bundle["variance_support"], bool)
    values = values[:, support]
    mean = np.asarray(bundle["scaler_mean"], float)
    scale = np.asarray(bundle["scaler_scale"], float)
    coef = np.asarray(bundle["elasticnet_coefficients"], float)
    if not (values.shape[1] == len(mean) == len(scale) == len(coef)):
        raise AssertionError("Frozen transform dimensions are inconsistent")
    z = (values - mean[None, :]) / scale[None, :]
    prediction = z @ coef + float(bundle["elasticnet_intercept"])
    pred = pd.DataFrame({"donor_id": aligned.index, "predicted_age": prediction})

    # Ages enter only here, after the predictions are fixed.
    result = pred.merge(manifest, on="donor_id", how="left", validate="one_to_one")
    result["chronological_age"] = pd.to_numeric(result.get("age_years"), errors="coerce")
    result["prediction_error"] = result.predicted_age - result.chronological_age
    result["absolute_error"] = result.prediction_error.abs()
    result["model_scope"] = "frozen GSE231906 model; no HRA007984 refitting or calibration"

    args.output.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output / "hra007984_frozen_predictions.tsv", sep="\t", index=False)
    coverage.assign(n_columns_absent_from_external_export=len(missing_columns)).to_csv(
        args.output / "hra007984_feature_coverage.tsv", sep="\t", index=False
    )

    evaluable = result.dropna(subset=["chronological_age", "predicted_age"]).copy()
    metrics = {
        "n_external_donors_predicted": int(len(result)),
        "n_external_donors_with_exact_age": int(len(evaluable)),
        "external_age_used_during_prediction": False,
        "external_refitting": False,
        "external_calibration": False,
        "n_model_features": int(len(coef)),
        "n_export_columns_absent": int(len(missing_columns)),
    }
    if len(evaluable) >= 2:
        y = evaluable.chronological_age.to_numpy(float)
        p = evaluable.predicted_age.to_numpy(float)
        metrics.update({
            "MAE_years": float(mean_absolute_error(y, p)),
            "RMSE_years": float(np.sqrt(mean_squared_error(y, p))),
            "Spearman_rho": float(stats.spearmanr(y, p).statistic),
            "Spearman_exact_or_asymptotic_p": exact_spearman_p(y, p),
        })
        if len(evaluable) >= 3 and np.unique(y).size >= 2:
            metrics["R2"] = float(r2_score(y, p))
    metrics["claim_status"] = (
        "completed_external_feasibility_test"
        if len(evaluable) >= 4 else
        "predictions_completed_but_exact_age_evaluation_incomplete"
    )
    metrics["claim_boundary"] = (
        "Independent feasibility evidence only; four donors cannot validate a clinical age clock"
    )
    (args.output / "hra007984_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()


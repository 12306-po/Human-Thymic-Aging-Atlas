#!/usr/bin/env python
"""Step 26: fully nested donor-level continuous-age machine learning.

Every inner split independently refits missingness filtering, variance filtering,
univariate feature screening, imputation, scaling, and the candidate model.
The held-out outer donor is never used for screening or tuning.

This script writes new results under ``04_machine_learning/fully_nested`` and
does not overwrite the exploratory Step 11 outputs.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR

try:
    from xgboost import XGBRegressor
except Exception:  # pragma: no cover - recorded as unavailable at runtime
    XGBRegressor = None


SEED = 371
TOP_K = 2000
NAN_FRAC_CUT = 0.50
INNER_SPLITS = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project", type=Path,
        default=Path(os.environ.get("PROJ", "/data/zxy/projects/human_thymus_age_ML_DL")),
    )
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--skip-sensitivities", action="store_true")
    return parser.parse_args()


def model_specs() -> dict[str, tuple[object, dict[str, list[object]]]]:
    specs: dict[str, tuple[object, dict[str, list[object]]]] = {
        "ElasticNet": (
            ElasticNet(max_iter=20000, random_state=SEED),
            {"alpha": [0.01, 0.1, 1.0], "l1_ratio": [0.1, 0.5, 0.9]},
        ),
        "LinearSVR": (
            LinearSVR(max_iter=30000, random_state=SEED),
            {"C": [0.01, 0.1, 1.0, 10.0], "epsilon": [0.0, 1.0, 3.0]},
        ),
        "RF": (
            RandomForestRegressor(
                n_estimators=300, random_state=SEED, n_jobs=-1
            ),
            {"max_depth": [3, 5], "min_samples_leaf": [2, 4]},
        ),
    }
    if XGBRegressor is not None:
        specs["XGB"] = (
            XGBRegressor(
                n_estimators=250, subsample=0.8, colsample_bytree=0.8,
                random_state=SEED, n_jobs=1, verbosity=0,
            ),
            {"max_depth": [2, 3], "learning_rate": [0.03, 0.1]},
        )
    return specs


def parameter_combinations(grid: dict[str, list[object]]) -> list[dict[str, object]]:
    keys = list(grid)
    return [dict(zip(keys, values)) for values in product(*(grid[key] for key in keys))]


def make_pipeline(estimator: object) -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("variance", VarianceThreshold(0.0)),
        ("scale", StandardScaler()),
        ("model", estimator),
    ])


def select_features(
    x_train: pd.DataFrame, y_train: np.ndarray, candidate_columns: list[str], top_k: int
) -> tuple[list[str], dict[str, int]]:
    x = x_train[candidate_columns]
    values = x.to_numpy(dtype=float, copy=False)
    n_before = values.shape[1]
    keep_missing = np.isnan(values).mean(axis=0) <= NAN_FRAC_CUT
    values = values[:, keep_missing]
    columns = np.asarray(candidate_columns, dtype=object)[keep_missing]
    if values.shape[1] == 0:
        return [], {"n_before": n_before, "n_after_missing": 0, "n_after_variance": 0, "n_selected": 0}

    medians = np.nanmedian(values, axis=0)
    medians = np.where(np.isnan(medians), 0.0, medians)
    values = np.where(np.isnan(values), medians, values)
    keep_variance = np.var(values, axis=0) > 0
    values = values[:, keep_variance]
    columns = columns[keep_variance]
    if values.shape[1] == 0:
        return [], {"n_before": n_before, "n_after_missing": int(keep_missing.sum()), "n_after_variance": 0, "n_selected": 0}

    ranked_features = np.apply_along_axis(stats.rankdata, 0, values)
    ranked_age = stats.rankdata(y_train)
    ranked_features -= ranked_features.mean(axis=0)
    ranked_age = ranked_age - ranked_age.mean()
    denominator = np.sqrt((ranked_features ** 2).sum(axis=0)) * np.sqrt((ranked_age ** 2).sum())
    denominator[denominator == 0] = 1.0
    absolute_rho = np.abs((ranked_features * ranked_age[:, None]).sum(axis=0) / denominator)
    n_select = min(top_k, len(columns))
    selected_index = np.argpartition(absolute_rho, -n_select)[-n_select:]
    selected_index = selected_index[np.argsort(-absolute_rho[selected_index], kind="stable")]
    selected = columns[selected_index].tolist()
    return selected, {
        "n_before": n_before,
        "n_after_missing": int(keep_missing.sum()),
        "n_after_variance": int(keep_variance.sum()),
        "n_selected": len(selected),
    }


def inner_plan(
    x_outer: pd.DataFrame,
    y_outer: np.ndarray,
    candidate_columns: list[str],
    top_k: int,
    outer_fold: int,
    block: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    n_splits = min(INNER_SPLITS, len(y_outer))
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=SEED + outer_fold)
    plans: list[dict[str, object]] = []
    audits: list[dict[str, object]] = []
    for inner_fold, (train_index, valid_index) in enumerate(splitter.split(x_outer), start=1):
        selected, counts = select_features(
            x_outer.iloc[train_index], y_outer[train_index], candidate_columns, top_k
        )
        if not selected:
            raise RuntimeError(f"No features selected: outer={outer_fold}, inner={inner_fold}, block={block}")
        plans.append({"train": train_index, "valid": valid_index, "selected": selected})
        audits.append({
            "outer_fold": outer_fold,
            "inner_fold": inner_fold,
            "feature_block": block,
            "train_donors": ";".join(x_outer.index[train_index]),
            "validation_donors": ";".join(x_outer.index[valid_index]),
            **counts,
            "screening_scope": "inner-training donors only",
        })
    return plans, audits


def tune_model(
    x_outer: pd.DataFrame,
    y_outer: np.ndarray,
    estimator: object,
    grid: dict[str, list[object]],
    plans: list[dict[str, object]],
) -> tuple[dict[str, object], float, list[dict[str, object]]]:
    best_parameters: dict[str, object] | None = None
    best_score = -np.inf
    tuning_rows: list[dict[str, object]] = []
    for parameters in parameter_combinations(grid):
        scores: list[float] = []
        for inner_fold, plan in enumerate(plans, start=1):
            selected = plan["selected"]
            model = clone(estimator).set_params(**parameters)
            pipeline = make_pipeline(model)
            train_index = plan["train"]
            valid_index = plan["valid"]
            pipeline.fit(x_outer.iloc[train_index][selected], y_outer[train_index])
            prediction = pipeline.predict(x_outer.iloc[valid_index][selected])
            score = -mean_absolute_error(y_outer[valid_index], prediction)
            scores.append(float(score))
            tuning_rows.append({
                "inner_fold": inner_fold,
                "parameters": json.dumps(parameters, sort_keys=True),
                "n_selected_features": len(selected),
                "negative_MAE": float(score),
            })
        mean_score = float(np.mean(scores))
        if mean_score > best_score:
            best_score = mean_score
            best_parameters = parameters
    if best_parameters is None:
        raise RuntimeError("All inner-CV parameter combinations failed")
    return best_parameters, best_score, tuning_rows


def fit_outer_model(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_test: pd.DataFrame,
    estimator: object,
    parameters: dict[str, object],
    candidate_columns: list[str],
    top_k: int,
) -> tuple[float, list[str], Pipeline, dict[str, int]]:
    selected, counts = select_features(x_train, y_train, candidate_columns, top_k)
    if not selected:
        raise RuntimeError("No features survived outer-training screening")
    pipeline = make_pipeline(clone(estimator).set_params(**parameters))
    pipeline.fit(x_train[selected], y_train)
    prediction = float(pipeline.predict(x_test[selected])[0])
    return prediction, selected, pipeline, counts


def metric_row(model: str, cohort: str, frame: pd.DataFrame) -> dict[str, object]:
    y_true = frame["y_true"].to_numpy(float)
    y_pred = frame["y_pred"].to_numpy(float)
    return {
        "cohort": cohort,
        "model": model,
        "n_donors": len(frame),
        "R2": r2_score(y_true, y_pred) if len(frame) >= 3 else np.nan,
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "Spearman_rho": stats.spearmanr(y_true, y_pred).statistic if len(frame) >= 3 else np.nan,
    }


def bootstrap_metrics(
    frame: pd.DataFrame, model: str, cohort: str, n_boot: int, seed: int
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    y_true = frame["y_true"].to_numpy(float)
    y_pred = frame["y_pred"].to_numpy(float)
    values = {"R2": [], "MAE": [], "RMSE": []}
    for _ in range(n_boot):
        index = rng.integers(0, len(frame), len(frame))
        if np.unique(y_true[index]).size < 2:
            continue
        values["R2"].append(r2_score(y_true[index], y_pred[index]))
        values["MAE"].append(mean_absolute_error(y_true[index], y_pred[index]))
        values["RMSE"].append(np.sqrt(mean_squared_error(y_true[index], y_pred[index])))
    row: dict[str, object] = {"cohort": cohort, "model": model, "n_bootstrap": n_boot}
    for metric, samples in values.items():
        row[f"{metric}_ci_low"] = np.quantile(samples, 0.025)
        row[f"{metric}_ci_high"] = np.quantile(samples, 0.975)
    return row


def paired_null_bootstrap(
    model_frame: pd.DataFrame, null_frame: pd.DataFrame, n_boot: int, seed: int
) -> dict[str, object]:
    merged = model_frame.merge(
        null_frame[["donor_id", "y_pred"]], on="donor_id", suffixes=("_model", "_null")
    )
    model_error = np.abs(merged["y_true"] - merged["y_pred_model"]).to_numpy()
    null_error = np.abs(merged["y_true"] - merged["y_pred_null"]).to_numpy()
    paired = model_error - null_error
    rng = np.random.default_rng(seed)
    samples = np.asarray([
        paired[rng.integers(0, len(paired), len(paired))].mean() for _ in range(n_boot)
    ])
    pvalue = 2 * min(np.mean(samples <= 0), np.mean(samples >= 0))
    return {
        "n_donors": len(merged),
        "delta_MAE_model_minus_null": paired.mean(),
        "ci_low": np.quantile(samples, 0.025),
        "ci_high": np.quantile(samples, 0.975),
        "bootstrap_two_sided_p": min(1.0, pvalue),
    }


def linear_age_error_pattern(frame: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    joined = frame.merge(metadata.reset_index(), on="donor_id", how="left")
    age_z = (joined["age_years"] - joined["age_years"].mean()) / joined["age_years"].std(ddof=0)
    sex_male = joined["sex"].astype(str).str.lower().eq("male").astype(float)
    design = np.column_stack([np.ones(len(joined)), age_z, age_z ** 2, sex_male])
    rows = []
    for outcome_name, outcome in {
        "signed_error": joined["y_pred"] - joined["y_true"],
        "absolute_error": np.abs(joined["y_pred"] - joined["y_true"]),
    }.items():
        beta, _, _, _ = np.linalg.lstsq(design, outcome.to_numpy(float), rcond=None)
        residual = outcome.to_numpy(float) - design @ beta
        df = len(joined) - design.shape[1]
        covariance = np.linalg.pinv(design.T @ design) * (residual @ residual / df)
        se = np.sqrt(np.diag(covariance))
        tvalue = beta / se
        pvalue = 2 * stats.t.sf(np.abs(tvalue), df)
        for term, estimate, standard_error, p in zip(
            ("intercept", "age_z", "age_z_squared", "sex_male"), beta, se, pvalue
        ):
            rows.append({
                "outcome": outcome_name, "term": term, "estimate": estimate,
                "standard_error": standard_error, "p_value": p, "n_donors": len(joined),
            })
    return pd.DataFrame(rows)


def run_nested(
    x: pd.DataFrame,
    ages: np.ndarray,
    blocks: dict[str, list[str]],
    models: dict[str, tuple[object, dict[str, list[object]]]],
    top_k: int,
    cohort: str,
    run_ablation: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prediction_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    inner_rows: list[dict[str, object]] = []
    recurrence_rows: list[dict[str, object]] = []
    ablation_rows: list[dict[str, object]] = []

    for outer_fold, (train_index, test_index) in enumerate(LeaveOneOut().split(x), start=1):
        x_train, x_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = ages[train_index], ages[test_index]
        donor = str(x.index[test_index[0]])
        plans, audits = inner_plan(
            x_train, y_train, blocks["all_features"], top_k, outer_fold, "all_features"
        )
        inner_rows.extend({"cohort": cohort, **row} for row in audits)

        for model_index, (name, (estimator, grid)) in enumerate(models.items()):
            parameters, inner_score, tuning = tune_model(x_train, y_train, estimator, grid, plans)
            for row in tuning:
                inner_rows.append({
                    "cohort": cohort, "outer_fold": outer_fold, "model": name,
                    "feature_block": "all_features", "record_type": "parameter_score", **row,
                })
            prediction, selected, pipeline, counts = fit_outer_model(
                x_train, y_train, x_test, estimator, parameters,
                blocks["all_features"], top_k,
            )
            prediction_rows.append({
                "cohort": cohort, "outer_fold": outer_fold, "donor_id": donor,
                "model": name, "y_true": float(y_test[0]), "y_pred": prediction,
                "absolute_error": abs(float(y_test[0]) - prediction),
            })
            fold_rows.append({
                "cohort": cohort, "outer_fold": outer_fold, "test_donor": donor,
                "model": name, "best_parameters": json.dumps(parameters, sort_keys=True),
                "inner_mean_negative_MAE": inner_score, **counts,
                "outer_training_donors": ";".join(x_train.index),
                "screening_scope": "inner-training for tuning; outer-training for final refit",
                "seed": SEED,
            })
            if name == "ElasticNet":
                for feature in selected:
                    recurrence_rows.append({
                        "cohort": cohort, "outer_fold": outer_fold,
                        "test_donor": donor, "feature": feature,
                    })

        null_prediction = float(np.mean(y_train))
        prediction_rows.append({
            "cohort": cohort, "outer_fold": outer_fold, "donor_id": donor,
            "model": "MeanAgeNull", "y_true": float(y_test[0]),
            "y_pred": null_prediction,
            "absolute_error": abs(float(y_test[0]) - null_prediction),
        })

        if run_ablation:
            elastic, grid = models["ElasticNet"]
            for block, columns in blocks.items():
                if block == "all_features":
                    main = prediction_rows[-(len(models) + 1)]
                    ablation_rows.append({**main, "feature_block": block})
                    continue
                if len(columns) < 2:
                    ablation_rows.append({
                        "cohort": cohort, "outer_fold": outer_fold, "donor_id": donor,
                        "model": "ElasticNet", "feature_block": block,
                        "y_true": float(y_test[0]), "y_pred": null_prediction,
                        "absolute_error": abs(float(y_test[0]) - null_prediction),
                        "status": "insufficient features; null prediction used",
                    })
                    continue
                block_plans, block_audits = inner_plan(
                    x_train, y_train, columns, top_k, outer_fold, block
                )
                inner_rows.extend({"cohort": cohort, **row} for row in block_audits)
                parameters, _, _ = tune_model(x_train, y_train, elastic, grid, block_plans)
                prediction, selected, _, _ = fit_outer_model(
                    x_train, y_train, x_test, elastic, parameters, columns, top_k
                )
                ablation_rows.append({
                    "cohort": cohort, "outer_fold": outer_fold, "donor_id": donor,
                    "model": "ElasticNet", "feature_block": block,
                    "y_true": float(y_test[0]), "y_pred": prediction,
                    "absolute_error": abs(float(y_test[0]) - prediction),
                    "n_selected_features": len(selected), "status": "estimated",
                })
    return (
        pd.DataFrame(prediction_rows), pd.DataFrame(fold_rows),
        pd.DataFrame(inner_rows), pd.DataFrame(recurrence_rows),
        pd.DataFrame(ablation_rows),
    )


def main() -> None:
    args = parse_args()
    project = args.project.resolve()
    dataset = project / "04_machine_learning" / "dataset"
    output = project / "04_machine_learning" / "fully_nested"
    output.mkdir(parents=True, exist_ok=True)

    features = pd.read_csv(dataset / "feature_matrix_all_combined.csv.gz", index_col=0)
    metadata = pd.read_csv(dataset / "donor_metadata.csv", index_col=0)
    dictionary = pd.read_csv(dataset / "feature_dictionary.csv")
    features.index = features.index.astype(str)
    metadata.index = metadata.index.astype(str)
    metadata.index.name = "donor_id"
    donors = metadata.index.tolist()
    features = features.loc[donors]
    ages = metadata["age_years"].to_numpy(float)
    if len(donors) != 18 or len(set(donors)) != 18:
        raise AssertionError("Fully nested primary analysis requires 18 unique frozen donors")

    feature_type = dictionary.set_index("feature")["feature_type"].to_dict()
    blocks = {
        "all_features": features.columns.tolist(),
        "gene_celltype_only": [c for c in features if feature_type.get(c) == "gene_celltype"],
        "whole_thymus_only": [c for c in features if feature_type.get(c) == "whole_thymus"],
        "composition_only": [c for c in features if feature_type.get(c) == "proportion"],
    }
    specs = model_specs()
    predictions, fold_audit, inner_audit, recurrence, ablation = run_nested(
        features, ages, blocks, specs, args.top_k, "primary_18_donors", True
    )

    sensitivity_predictions: list[pd.DataFrame] = []
    if not args.skip_sensitivities:
        order = np.argsort(ages)
        sensitivity_masks = {
            "remove_youngest": np.arange(len(ages)) != order[0],
            "remove_oldest": np.arange(len(ages)) != order[-1],
            "remove_both_extremes": ~np.isin(np.arange(len(ages)), [order[0], order[-1]]),
            "restricted_age_18_64": (ages >= 18) & (ages <= 64),
            "female_only": metadata["sex"].astype(str).str.lower().eq("female").to_numpy(),
        }
        for cohort, mask in sensitivity_masks.items():
            if int(mask.sum()) < 8:
                continue
            subset_models = {"ElasticNet": specs["ElasticNet"]}
            subset_predictions, _, _, _, _ = run_nested(
                features.loc[mask], ages[mask], blocks, subset_models,
                args.top_k, cohort, False,
            )
            sensitivity_predictions.append(subset_predictions)

    all_predictions = pd.concat([predictions, *sensitivity_predictions], ignore_index=True)
    all_predictions.to_csv(output / "oof_predictions_all_models.tsv", sep="\t", index=False)
    fold_audit.to_csv(output / "outer_fold_audit.tsv", sep="\t", index=False)
    inner_audit.to_csv(output / "inner_fold_screening_audit.tsv", sep="\t", index=False)
    recurrence.to_csv(output / "elasticnet_outer_feature_recurrence.tsv", sep="\t", index=False)
    ablation.to_csv(output / "nested_ablation_oof_predictions.tsv", sep="\t", index=False)

    metrics = []
    intervals = []
    for (cohort, model), frame in all_predictions.groupby(["cohort", "model"]):
        metrics.append(metric_row(model, cohort, frame))
        intervals.append(bootstrap_metrics(frame, model, cohort, args.bootstrap, SEED))
    pd.DataFrame(metrics).to_csv(output / "model_metrics.tsv", sep="\t", index=False)
    pd.DataFrame(intervals).to_csv(output / "model_metrics_bootstrap_ci.tsv", sep="\t", index=False)

    ablation_metrics = [
        metric_row(block, cohort, frame)
        for (cohort, block), frame in ablation.groupby(["cohort", "feature_block"])
    ]
    pd.DataFrame(ablation_metrics).rename(columns={"model": "feature_block"}).to_csv(
        output / "nested_ablation_results.tsv", sep="\t", index=False
    )

    primary_null = predictions[predictions["model"].eq("MeanAgeNull")]
    paired_rows = []
    for model, frame in predictions[~predictions["model"].eq("MeanAgeNull")].groupby("model"):
        paired_rows.append({
            "model": model,
            **paired_null_bootstrap(frame, primary_null, args.bootstrap, SEED + 17),
        })
    pd.DataFrame(paired_rows).to_csv(
        output / "paired_model_vs_null_MAE.tsv", sep="\t", index=False
    )

    influence_rows = []
    for model, frame in predictions.groupby("model"):
        full_mae = mean_absolute_error(frame["y_true"], frame["y_pred"])
        full_r2 = r2_score(frame["y_true"], frame["y_pred"])
        for donor in frame["donor_id"]:
            subset = frame[~frame["donor_id"].eq(donor)]
            influence_rows.append({
                "model": model, "excluded_donor": donor,
                "R2_without_donor": r2_score(subset["y_true"], subset["y_pred"]),
                "delta_R2_vs_full": r2_score(subset["y_true"], subset["y_pred"]) - full_r2,
                "MAE_without_donor": mean_absolute_error(subset["y_true"], subset["y_pred"]),
                "delta_MAE_vs_full": mean_absolute_error(subset["y_true"], subset["y_pred"]) - full_mae,
                "interpretation": "jackknife influence on pooled OOF metrics; models not refit",
            })
    pd.DataFrame(influence_rows).to_csv(output / "donor_influence.tsv", sep="\t", index=False)

    elastic_predictions = predictions[predictions["model"].eq("ElasticNet")]
    linear_age_error_pattern(elastic_predictions, metadata).to_csv(
        output / "age_error_patterns.tsv", sep="\t", index=False
    )

    audit = {
        "analysis_unit": "donor",
        "outer_cv": "leave-one-donor-out",
        "inner_cv": f"{INNER_SPLITS}-fold donor CV within each outer-training set",
        "feature_screening": "recomputed independently in every inner-training fold",
        "preprocessing": "imputation, variance filtering, scaling fit inside each inner-training fold",
        "selection_metric": "mean inner-fold MAE",
        "top_k": args.top_k,
        "random_seed": SEED,
        "candidate_models": list(specs),
        "xgboost_available": XGBRegressor is not None,
        "sensitivity_cohorts": sorted(all_predictions["cohort"].unique()),
        "note": "Internal validation only; no independent external adult-aging cohort.",
    }
    (output / "fully_nested_analysis_contract.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    print(f"Fully nested outputs written to {output}")


if __name__ == "__main__":
    main()

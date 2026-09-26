#!/usr/bin/env python3
"""Fit and serialize the final GSE231906 Elastic Net for external transfer.

Feature screening and tuning use only the 18 primary donors.  External data
are never read by this program.  The resulting JSON is deliberately portable:
it stores the selected feature order, training medians, variance mask,
standardization parameters, coefficients, intercept, and file hashes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 371
NAN_FRAC_CUT = 0.50


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def select_features(x: pd.DataFrame, y: np.ndarray, top_k: int) -> list[str]:
    values = x.to_numpy(float, copy=False)
    keep_missing = np.isnan(values).mean(axis=0) <= NAN_FRAC_CUT
    values = values[:, keep_missing]
    columns = np.asarray(x.columns, dtype=object)[keep_missing]
    if not len(columns):
        raise RuntimeError("No feature survived the primary missingness filter")
    medians = np.nanmedian(values, axis=0)
    medians = np.where(np.isnan(medians), 0.0, medians)
    values = np.where(np.isnan(values), medians, values)
    keep_variance = np.var(values, axis=0) > 0
    values = values[:, keep_variance]
    columns = columns[keep_variance]
    if not len(columns):
        raise RuntimeError("No feature survived the primary variance filter")
    ranked_x = np.apply_along_axis(stats.rankdata, 0, values)
    ranked_y = stats.rankdata(y)
    ranked_x -= ranked_x.mean(axis=0)
    ranked_y -= ranked_y.mean()
    denom = np.sqrt((ranked_x**2).sum(axis=0)) * np.sqrt((ranked_y**2).sum())
    denom[denom == 0] = 1.0
    rho = np.abs((ranked_x * ranked_y[:, None]).sum(axis=0) / denom)
    n = min(top_k, len(columns))
    idx = np.argpartition(rho, -n)[-n:]
    idx = idx[np.argsort(-rho[idx], kind="stable")]
    return columns[idx].tolist()


def make_pipeline(params: dict[str, float]) -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("variance", VarianceThreshold(0.0)),
        ("scale", StandardScaler()),
        ("model", ElasticNet(max_iter=20000, random_state=SEED, **params)),
    ])


def tune(x: pd.DataFrame, y: np.ndarray, top_k: int) -> tuple[dict, pd.DataFrame]:
    grid = [dict(alpha=a, l1_ratio=l) for a, l in product(
        [0.01, 0.1, 1.0], [0.1, 0.5, 0.9]
    )]
    folds = list(KFold(n_splits=3, shuffle=True, random_state=SEED).split(x))
    rows = []
    for params in grid:
        errors = []
        for fold, (tr, va) in enumerate(folds, start=1):
            selected = select_features(x.iloc[tr], y[tr], top_k)
            model = make_pipeline(params)
            model.fit(x.iloc[tr][selected], y[tr])
            pred = model.predict(x.iloc[va][selected])
            error = mean_absolute_error(y[va], pred)
            errors.append(error)
            rows.append({"fold": fold, **params, "n_selected": len(selected), "MAE": error})
        rows.append({"fold": "mean", **params, "n_selected": np.nan, "MAE": np.mean(errors)})
    audit = pd.DataFrame(rows)
    means = audit[audit.fold.eq("mean")].sort_values(["MAE", "alpha", "l1_ratio"])
    best = means.iloc[0]
    return {"alpha": float(best.alpha), "l1_ratio": float(best.l1_ratio)}, audit


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--top-k", type=int, default=2000)
    args = ap.parse_args()

    dataset = args.project.resolve() / "04_machine_learning" / "dataset"
    x_path = dataset / "feature_matrix_all_combined.csv.gz"
    m_path = dataset / "donor_metadata.csv"
    d_path = dataset / "feature_dictionary.csv"
    for path in (x_path, m_path, d_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    x = pd.read_csv(x_path, index_col=0)
    meta = pd.read_csv(m_path, index_col=0)
    dictionary = pd.read_csv(d_path)
    x.index = x.index.astype(str)
    meta.index = meta.index.astype(str)
    if len(meta) != 18 or not meta.index.is_unique:
        raise AssertionError("The frozen primary training set must contain 18 unique donors")
    x = x.loc[meta.index]
    y = meta["age_years"].to_numpy(float)
    if dictionary.feature.duplicated().any() or not x.columns.is_unique:
        raise AssertionError("Primary feature identifiers are not unique")

    best, tuning = tune(x, y, args.top_k)
    selected = select_features(x, y, args.top_k)
    pipeline = make_pipeline(best)
    pipeline.fit(x[selected], y)

    imp = pipeline.named_steps["impute"]
    var = pipeline.named_steps["variance"]
    scale = pipeline.named_steps["scale"]
    model = pipeline.named_steps["model"]
    variance_mask = var.get_support().astype(bool)
    transformed_features = np.asarray(selected, dtype=object)[variance_mask].tolist()
    if len(transformed_features) != len(model.coef_):
        raise AssertionError("Serialized coefficient dimension mismatch")

    args.output.mkdir(parents=True, exist_ok=True)
    selected_dict = (dictionary.set_index("feature").loc[selected].reset_index())
    selected_dict.to_csv(args.output / "selected_feature_dictionary.tsv", sep="\t", index=False)
    tuning.to_csv(args.output / "final_model_inner_cv.tsv", sep="\t", index=False)

    bundle = {
        "schema_version": "1.0",
        "model": "ElasticNet",
        "training_accession": "GSE231906",
        "training_n_donors": 18,
        "training_donor_ids": meta.index.tolist(),
        "outcome": "chronological age in years",
        "random_seed": SEED,
        "top_k": args.top_k,
        "missingness_cutoff": NAN_FRAC_CUT,
        "tuning_scope": "3-fold CV in all 18 primary donors; external data never accessed",
        "best_parameters": best,
        "selected_features_before_variance": selected,
        "training_imputation_median": imp.statistics_.astype(float).tolist(),
        "variance_support": variance_mask.tolist(),
        "model_features": transformed_features,
        "scaler_mean": scale.mean_.astype(float).tolist(),
        "scaler_scale": scale.scale_.astype(float).tolist(),
        "elasticnet_coefficients": model.coef_.astype(float).tolist(),
        "elasticnet_intercept": float(model.intercept_),
        "feature_matrix_sha256": sha256(x_path),
        "feature_dictionary_sha256": sha256(d_path),
        "donor_metadata_sha256": sha256(m_path),
        "claim_boundary": (
            "Frozen development model for independent feasibility testing; "
            "not a validated clinical age clock"
        ),
    }
    out = args.output / "frozen_elasticnet.json"
    out.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    (args.output / "frozen_elasticnet.sha256").write_text(
        f"{sha256(out)}  {out.name}\n", encoding="utf-8"
    )
    print(f"Frozen model written: {out}")
    print(f"Selected features: {len(selected)}; model coefficients: {len(model.coef_)}")


if __name__ == "__main__":
    main()


"""Freeze a deployable T05 model only if it reproduces every held-out T05C prediction.

Run on the analysis server. This script never trains on uploaded website samples.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
STAGES = ROOT / "release" / "transformer_annotation_v1"
DEFAULT_FEATURES = (STAGES / "t05b_donor_age_features_v1"
                    / "X_scgpt_global_plus_composition.tsv.gz")
DEFAULT_OUTCOMES = STAGES / "t05b_donor_age_features_v1" / "locked_age_outcome.tsv"
DEFAULT_OOF = (STAGES / "t05c_nested_lodo_age_prediction_v1"
               / "primary_model_oof_predictions.tsv")
DEFAULT_ALPHA_GRID = (100.0, 1000.0)


def digest(path: Path) -> str:
    hash_value = sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            hash_value.update(block)
    return hash_value.hexdigest()


def read_table(path: Path, columns: set[str]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, sep="\t")
    if "donor_id" not in frame and str(frame.columns[0]).startswith("Unnamed:"):
        frame = frame.rename(columns={frame.columns[0]: "donor_id"})
    missing = columns.difference(frame.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")
    if frame["donor_id"].isna().any() or not frame["donor_id"].is_unique:
        raise ValueError(f"{path.name} has missing or repeated donor IDs")
    frame["donor_id"] = frame["donor_id"].astype(str)
    return frame.set_index("donor_id")


def select_alpha_by_lodo_mae(
    x: np.ndarray, y: np.ndarray, alpha_grid: tuple[float, ...]
) -> tuple[float, dict[float, float]]:
    """Tune on training donors only, refitting scaling inside every inner fold."""
    scores: dict[float, float] = {}
    for alpha in alpha_grid:
        errors = []
        for held_out in range(len(y)):
            train = np.arange(len(y)) != held_out
            scaler = StandardScaler().fit(x[train])
            regressor = Ridge(alpha=alpha).fit(scaler.transform(x[train]), y[train])
            prediction = float(regressor.predict(scaler.transform(x[held_out:held_out + 1]))[0])
            errors.append(abs(prediction - y[held_out]))
        scores[alpha] = float(np.mean(errors))
    # As in an ordered grid search, keep the first candidate on an exact tie.
    return min(alpha_grid, key=lambda alpha: scores[alpha]), scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--outcomes", type=Path, default=DEFAULT_OUTCOMES)
    parser.add_argument("--oof", type=Path, default=DEFAULT_OOF)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "release" / "age_prediction_model_v1")
    parser.add_argument("--verify-only", action="store_true",
                        help="Audit fold choices and full-data tuning without writing a model")
    parser.add_argument(
        "--alpha-grid", type=float, nargs="+", default=list(DEFAULT_ALPHA_GRID),
        help="Ordered candidate grid for reconstructed inner LODO MAE tuning; "
             "each outer-fold choice must match T05C before export",
    )
    args = parser.parse_args()
    alpha_grid = tuple(args.alpha_grid)
    if (len(set(alpha_grid)) != len(alpha_grid)
            or any(not np.isfinite(alpha) or alpha <= 0 for alpha in alpha_grid)):
        raise ValueError("--alpha-grid must contain unique positive finite values")

    x_frame = read_table(args.features, {"donor_id"})
    outcome = read_table(args.outcomes, {"donor_id", "age_years"})
    oof = read_table(args.oof, {"donor_id", "chronological_age",
                               "predicted_age", "selected_parameters"})
    donor_ids = x_frame.index.tolist()
    if len(donor_ids) != 18 or set(donor_ids) != set(outcome.index) or set(donor_ids) != set(oof.index):
        raise ValueError("T05B features, outcomes and T05C predictions must match 18 donors")
    if x_frame.columns.duplicated().any() or x_frame.shape[1] != 542:
        raise ValueError("Expected 542 unique scGPT-plus-composition feature columns")
    try:
        x = x_frame.apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
        y = pd.to_numeric(outcome.loc[donor_ids, "age_years"], errors="raise").to_numpy(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("T05 training features and ages must be numeric") from exc
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("T05 training table contains missing or non-finite values")
    if y.min() != 4 or y.max() != 69 or not np.allclose(
        y, pd.to_numeric(oof.loc[donor_ids, "chronological_age"])
    ):
        raise ValueError("Age metadata differs from the locked T05C outcomes")

    selected_alphas = []
    for item in oof.loc[donor_ids, "selected_parameters"]:
        parameters = json.loads(item)
        selected_alphas.append(float(parameters["regressor__alpha"]))
    if any(alpha not in alpha_grid for alpha in selected_alphas):
        raise ValueError(
            f"T05C selected alpha outside --alpha-grid {alpha_grid}: "
            f"{sorted(set(selected_alphas))}"
        )

    reproduced = np.empty(len(donor_ids), dtype=float)
    for held_out in range(len(donor_ids)):
        train = np.arange(len(donor_ids)) != held_out
        tuned_alpha, _ = select_alpha_by_lodo_mae(x[train], y[train], alpha_grid)
        if tuned_alpha != selected_alphas[held_out]:
            raise RuntimeError(
                "REFUSING TO EXPORT: reconstructed inner LODO MAE tuning "
                f"selected alpha={tuned_alpha:g} for {donor_ids[held_out]}, "
                f"but T05C recorded alpha={selected_alphas[held_out]:g}. "
                "Check the original candidate grid and scoring rule."
            )
        scaler = StandardScaler().fit(x[train])
        regressor = Ridge(alpha=tuned_alpha).fit(scaler.transform(x[train]), y[train])
        reproduced[held_out] = regressor.predict(
            scaler.transform(x[held_out:held_out + 1])
        )[0]
    observed = pd.to_numeric(oof.loc[donor_ids, "predicted_age"]).to_numpy(float)
    max_difference = float(np.max(np.abs(reproduced - observed)))
    if not np.isfinite(max_difference) or max_difference > 1e-3:
        raise RuntimeError(
            "REFUSING TO EXPORT: this training pipeline does not reproduce T05C "
            f"held-out predictions (maximum difference {max_difference:.6f} years). "
            "Use the original T05C preprocessing code; do not deploy a different model."
        )

    final_alpha, final_scores = select_alpha_by_lodo_mae(x, y, alpha_grid)
    print(f"All 18 T05C alpha selections and predictions reproduced; "
          f"maximum difference: {max_difference:.9f} years")
    print(f"Full-training inner LODO MAE by alpha: {final_scores}")
    print(f"Selected final alpha: {final_alpha:g}")
    if args.verify_only:
        print("Verification only: no model file was written")
        return

    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty model release: {args.output}")
    scaler = StandardScaler().fit(x)
    regressor = Ridge(alpha=final_alpha).fit(scaler.transform(x), y)
    bundle = {
        "schema_version": "1.0",
        "model_name": "scgpt_global_plus_composition__ridge",
        "preprocessing": "StandardScaler",
        "n_training_donors": 18,
        "n_features": 542,
        "ridge_alpha": final_alpha,
        "alpha_candidate_grid": list(alpha_grid),
        "alpha_selection_rule": "inner leave-one-donor-out mean absolute error",
        "full_training_inner_lodo_mae_by_alpha": {
            str(alpha): score for alpha, score in final_scores.items()
        },
        "t05c_outer_fold_alpha_selections_reproduced": True,
        "feature_names": x_frame.columns.tolist(),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "ridge_coefficients": regressor.coef_.tolist(),
        "ridge_intercept": float(regressor.intercept_),
        "training_feature_min": np.min(x, axis=0).tolist(),
        "training_feature_max": np.max(x, axis=0).tolist(),
        "training_age_min": float(y.min()),
        "training_age_max": float(y.max()),
        "verified_against_t05c_oof": True,
        "oof_reproduction_max_abs_error_years": max_difference,
        "source_sha256": {
            "features": digest(args.features),
            "outcomes": digest(args.outcomes),
            "oof_predictions": digest(args.oof),
        },
        "validation_scope": (
            "Nested leave-one-donor-out within 18 discovery donors only; "
            "no independent multi-donor external age validation"
        ),
        "intended_use": "Research prototype only; not a clinical or health-age estimate",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / "frozen_t05_model.json"
    destination.write_text(json.dumps(bundle, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Frozen T05 web model exported: {destination}")
    print(f"Maximum held-out reproduction difference: {max_difference:.9f} years")


if __name__ == "__main__":
    main()

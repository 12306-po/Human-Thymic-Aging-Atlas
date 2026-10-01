#!/usr/bin/env python3
"""Integrate external results without pooling incompatible validation targets."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


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
    values = []
    for permutation in itertools.permutations(y.tolist()):
        statistic = abs(float(stats.spearmanr(x, np.asarray(permutation)).statistic))
        values.append(statistic >= observed - 1e-12)
    return float(np.mean(values))


def exact_metrics(frame: pd.DataFrame, label: str) -> dict[str, object]:
    evaluable = frame.dropna(subset=["chronological_age", "predicted_age"])
    y = evaluable.chronological_age.to_numpy(float)
    p = evaluable.predicted_age.to_numpy(float)
    row: dict[str, object] = {
        "analysis": label,
        "n_donors": int(len(evaluable)),
        "MAE_years": float(mean_absolute_error(y, p)),
        "RMSE_years": float(np.sqrt(mean_squared_error(y, p))),
        "Spearman_rho": np.nan,
        "Spearman_p": np.nan,
        "R2": np.nan,
    }
    if len(evaluable) >= 2 and np.unique(y).size >= 2:
        row["Spearman_rho"] = float(stats.spearmanr(y, p).statistic)
        row["Spearman_p"] = exact_spearman_p(y, p)
    if len(evaluable) >= 3 and np.unique(y).size >= 2:
        row["R2"] = float(r2_score(y, p))
    return row


def load_generic(results: Path, stem: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    predictions = pd.read_csv(results / f"{stem}_frozen_predictions.tsv", sep="\t")
    coverage = pd.read_csv(results / f"{stem}_feature_coverage.tsv", sep="\t")
    metrics = json.loads((results / f"{stem}_metrics.json").read_text(encoding="utf-8"))
    return predictions, coverage, metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--hra-results", type=Path, required=True)
    parser.add_argument("--tabula-results", type=Path, required=True)
    parser.add_argument("--park-results", type=Path, required=True)
    parser.add_argument("--gse147520-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    model_hash = sha256(args.model)
    hra = pd.read_csv(args.hra_results / "hra007984_frozen_predictions.tsv", sep="\t")
    hra_cov = pd.read_csv(args.hra_results / "hra007984_feature_coverage.tsv", sep="\t")
    hra["cohort"] = "HRA007984"
    hra["analysis_type"] = "exact_age"
    hra_cov["cohort"] = "HRA007984"
    hra_cov["analysis_type"] = "exact_age"
    hra_cov["observed_fraction"] = 1.0 - hra_cov.missing_fraction

    tabula, tabula_cov, tabula_metrics = load_generic(args.tabula_results, "tabula_sapiens")
    park, park_cov, park_metrics = load_generic(args.park_results, "park_e_mtab_8581")
    gse, gse_cov, gse_metrics = load_generic(args.gse147520_results, "gse147520")
    for name, metrics in (
        ("Tabula Sapiens", tabula_metrics),
        ("Park E-MTAB-8581", park_metrics),
        ("GSE147520", gse_metrics),
    ):
        if metrics.get("model_sha256") != model_hash:
            raise RuntimeError(f"{name} was not scored with the supplied frozen model")

    exact = pd.concat([hra, tabula], ignore_index=True, sort=False)
    if len(hra) != 4 or len(tabula) != 2 or len(exact) != 6:
        raise RuntimeError(
            "Prespecified exact-age integration requires 4 HRA007984 plus 2 Tabula Sapiens donors"
        )
    if exact.donor_id.duplicated().any():
        raise RuntimeError("Donor IDs overlap across exact-age external cohorts")
    if not (exact.analysis_type == "exact_age").all():
        raise RuntimeError("Only exact-age cohorts may enter the six-donor exact-age summary")

    exact_rows = [exact_metrics(hra, "HRA007984 exact-age"),
                  exact_metrics(tabula, "Tabula Sapiens exact-age"),
                  exact_metrics(exact, "Combined exact-age external")]
    cohort_metrics = pd.DataFrame(exact_rows)
    interval_summary = pd.DataFrame([{
        "analysis": "Park E-MTAB-8581 interval-age",
        "n_donors": int(len(park)),
        "n_within_interval": int(park.within_reported_age_interval.sum()),
        "mean_absolute_distance_to_interval_years": float(park.absolute_distance_to_interval.mean()),
        "point_age_metrics_calculated": False,
    }])
    stromal_summary = pd.DataFrame([{
        "analysis": "GSE147520 stromal sensitivity",
        "n_donors": int(len(gse)),
        "predicted_age": float(gse.predicted_age.iloc[0]),
        "observed_feature_fraction": float(gse_cov.observed_fraction.iloc[0]),
        "inferential_age_metrics_calculated": False,
    }])

    coverage = pd.concat([hra_cov, tabula_cov, park_cov, gse_cov], ignore_index=True, sort=False)
    args.output.mkdir(parents=True, exist_ok=True)
    exact.to_csv(args.output / "combined_six_donor_exact_age_predictions.tsv", sep="\t", index=False)
    park.to_csv(args.output / "park_interval_age_predictions.tsv", sep="\t", index=False)
    gse.to_csv(args.output / "gse147520_stromal_sensitivity.tsv", sep="\t", index=False)
    cohort_metrics.to_csv(args.output / "exact_age_cohort_metrics.tsv", sep="\t", index=False)
    interval_summary.to_csv(args.output / "park_interval_summary.tsv", sep="\t", index=False)
    stromal_summary.to_csv(args.output / "gse147520_stromal_summary.tsv", sep="\t", index=False)
    coverage.to_csv(args.output / "multicohort_feature_coverage.tsv", sep="\t", index=False)

    combined = exact_rows[-1]
    summary = {
        "frozen_model_sha256": model_hash,
        "exact_age_external_evaluation": combined,
        "exact_age_cohorts": ["HRA007984", "Tabula Sapiens"],
        "park_interval_age_evaluation": interval_summary.iloc[0].to_dict(),
        "gse147520_stromal_sensitivity": stromal_summary.iloc[0].to_dict(),
        "external_refitting": False,
        "external_calibration": False,
        "external_age_used_during_prediction": False,
        "prohibited_pooling": (
            "Park age intervals and GSE147520 stromal projections are not included in exact-age MAE, "
            "RMSE, Spearman correlation, or R-squared"
        ),
        "claim_boundary": (
            "Six exact-age adult donors provide independent multi-cohort feasibility evidence, "
            "not clinical validation of an age clock"
        ),
    }
    (args.output / "multicohort_external_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

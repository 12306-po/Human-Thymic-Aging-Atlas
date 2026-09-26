#!/usr/bin/env python
"""Step 27: age+sex composition-aware donor-level analysis.

Primary model: per-component CLR abundance ~ standardized age + sex, with HC3
standard errors. Zeros are replaced by half a cell relative to each donor's
all-QC denominator before closure. Donor stability and prespecified cohort/
age-shape sensitivities are exported. A joint Dirichlet-regression bootstrap is
included as a sensitivity analysis, not as a replacement for donor replication.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, stats
from scipy.special import gammaln, logsumexp


SEED = 371


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project", type=Path,
        default=Path(os.environ.get("PROJ", "/data/zxy/projects/human_thymus_age_ML_DL")),
    )
    parser.add_argument("--dirichlet-bootstrap", type=int, default=500)
    return parser.parse_args()


def bh_adjust(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    result = np.full(values.shape, np.nan)
    valid_index = np.where(np.isfinite(values))[0]
    if not len(valid_index):
        return result
    order = valid_index[np.argsort(values[valid_index])]
    adjusted = values[order] * len(order) / np.arange(1, len(order) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result[order] = np.minimum(adjusted, 1.0)
    return result


def zero_replace_and_close(proportions: pd.DataFrame, totals: pd.Series) -> pd.DataFrame:
    matrix = proportions.to_numpy(float, copy=True)
    for index in range(len(matrix)):
        delta = 0.5 / max(float(totals.iloc[index]), 1.0)
        row = matrix[index]
        row[~np.isfinite(row)] = 0.0
        row[row <= 0] = delta
        matrix[index] = row / row.sum()
    return pd.DataFrame(matrix, index=proportions.index, columns=proportions.columns)


def fixed_zero_replace_and_close(proportions: pd.DataFrame, delta: float = 1e-6) -> pd.DataFrame:
    matrix = proportions.to_numpy(float, copy=True)
    matrix[~np.isfinite(matrix)] = 0.0
    matrix[matrix <= 0] = delta
    matrix /= matrix.sum(axis=1, keepdims=True)
    return pd.DataFrame(matrix, index=proportions.index, columns=proportions.columns)


def clr(proportions: pd.DataFrame) -> pd.DataFrame:
    log_values = np.log(proportions.to_numpy(float))
    transformed = log_values - log_values.mean(axis=1, keepdims=True)
    return pd.DataFrame(transformed, index=proportions.index, columns=proportions.columns)


def hc3_fit(y: np.ndarray, design: np.ndarray, terms: list[str]) -> pd.DataFrame:
    inverse = np.linalg.pinv(design.T @ design)
    beta = inverse @ design.T @ y
    residual = y - design @ beta
    leverage = np.sum((design @ inverse) * design, axis=1)
    adjusted = residual / np.clip(1.0 - leverage, 1e-8, None)
    meat = design.T @ ((adjusted ** 2)[:, None] * design)
    covariance = inverse @ meat @ inverse
    standard_error = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    statistic = beta / standard_error
    degrees = max(1, len(y) - design.shape[1])
    pvalue = 2 * stats.t.sf(np.abs(statistic), degrees)
    return pd.DataFrame({
        "term": terms, "estimate": beta, "standard_error_HC3": standard_error,
        "ci_low_95": beta - stats.t.ppf(0.975, degrees) * standard_error,
        "ci_high_95": beta + stats.t.ppf(0.975, degrees) * standard_error,
        "t_HC3": statistic, "p_value": pvalue,
    })


def design_matrix(metadata: pd.DataFrame, nonlinear: bool, female_only: bool) -> tuple[np.ndarray, list[str]]:
    age_z = (metadata["age_years"] - metadata["age_years"].mean()) / metadata["age_years"].std(ddof=0)
    columns = [np.ones(len(metadata)), age_z.to_numpy(float)]
    terms = ["intercept", "age_z"]
    if nonlinear:
        columns.append((age_z ** 2).to_numpy(float))
        terms.append("age_z_squared")
    if not female_only:
        columns.append(metadata["sex"].astype(str).str.lower().eq("male").astype(float).to_numpy())
        terms.append("sex_male")
    return np.column_stack(columns), terms


def fit_clr_models(
    composition: pd.DataFrame,
    metadata: pd.DataFrame,
    analysis: str,
    nonlinear: bool = False,
    female_only: bool = False,
) -> pd.DataFrame:
    design, terms = design_matrix(metadata, nonlinear, female_only)
    rows: list[pd.DataFrame] = []
    for cell_type in composition.columns:
        fit = hc3_fit(composition[cell_type].to_numpy(float), design, terms)
        fit.insert(0, "cell_type", cell_type)
        fit.insert(0, "analysis", analysis)
        fit["n_donors"] = len(metadata)
        rows.append(fit)
    result = pd.concat(rows, ignore_index=True)
    for term in result["term"].unique():
        mask = result["term"].eq(term)
        result.loc[mask, "q_value_BH"] = bh_adjust(result.loc[mask, "p_value"].to_numpy())
    return result


def dirichlet_negative_loglikelihood(
    parameters: np.ndarray, y: np.ndarray, design: np.ndarray, ridge: float = 1e-5
) -> float:
    n_covariates = design.shape[1]
    n_components = y.shape[1]
    coefficients = parameters[:-1].reshape(n_covariates, n_components - 1)
    logits = np.column_stack([design @ coefficients, np.zeros(len(y))])
    log_mu = logits - logsumexp(logits, axis=1, keepdims=True)
    mu = np.exp(log_mu)
    phi = np.exp(np.clip(parameters[-1], -5, 12))
    alpha = np.clip(mu * phi, 1e-9, None)
    loglik = np.sum(
        gammaln(phi) - np.sum(gammaln(alpha), axis=1)
        + np.sum((alpha - 1.0) * np.log(y), axis=1)
    )
    return float(-loglik + ridge * np.sum(coefficients ** 2))


def fit_dirichlet(y: np.ndarray, design: np.ndarray) -> tuple[np.ndarray, bool]:
    n_parameters = design.shape[1] * (y.shape[1] - 1) + 1
    start = np.zeros(n_parameters)
    start[-1] = np.log(20.0)
    fit = optimize.minimize(
        dirichlet_negative_loglikelihood, start, args=(y, design),
        method="L-BFGS-B", options={"maxiter": 3000, "ftol": 1e-10},
    )
    return fit.x, bool(fit.success)


def dirichlet_age_effect(
    parameters: np.ndarray, n_covariates: int, n_components: int, sex_male: float
) -> np.ndarray:
    coefficients = parameters[:-1].reshape(n_covariates, n_components - 1)
    baseline = np.array([1.0, 0.0, sex_male])
    older = np.array([1.0, 1.0, sex_male])
    logits0 = np.r_[baseline @ coefficients, 0.0]
    logits1 = np.r_[older @ coefficients, 0.0]
    mu0 = np.exp(logits0 - logsumexp(logits0))
    mu1 = np.exp(logits1 - logsumexp(logits1))
    return mu1 - mu0


def dirichlet_bootstrap(
    composition: pd.DataFrame, metadata: pd.DataFrame, n_bootstrap: int
) -> tuple[pd.DataFrame, dict[str, object]]:
    design, terms = design_matrix(metadata, nonlinear=False, female_only=False)
    if terms != ["intercept", "age_z", "sex_male"]:
        raise AssertionError("Unexpected Dirichlet design")
    y = composition.to_numpy(float)
    parameters, success = fit_dirichlet(y, design)
    modal_sex = float(metadata["sex"].astype(str).str.lower().eq("male").mean() >= 0.5)
    observed = dirichlet_age_effect(parameters, design.shape[1], y.shape[1], modal_sex)

    rng = np.random.default_rng(SEED)
    samples: list[np.ndarray] = []
    failures = 0
    for _ in range(n_bootstrap):
        index = rng.integers(0, len(y), len(y))
        try:
            boot_parameters, boot_success = fit_dirichlet(y[index], design[index])
            if not boot_success:
                failures += 1
                continue
            samples.append(dirichlet_age_effect(
                boot_parameters, design.shape[1], y.shape[1], modal_sex
            ))
        except Exception:
            failures += 1
    if not samples:
        result = pd.DataFrame({
            "cell_type": composition.columns,
            "delta_expected_fraction_per_1SD_age": observed,
            "ci_low_95": np.nan, "ci_high_95": np.nan,
            "bootstrap_two_sided_p": np.nan, "q_value_BH": np.nan,
        })
    else:
        sample_matrix = np.vstack(samples)
        pvalues = 2 * np.minimum(
            np.mean(sample_matrix <= 0, axis=0), np.mean(sample_matrix >= 0, axis=0)
        )
        pvalues = np.minimum(pvalues, 1.0)
        result = pd.DataFrame({
            "cell_type": composition.columns,
            "delta_expected_fraction_per_1SD_age": observed,
            "ci_low_95": np.quantile(sample_matrix, 0.025, axis=0),
            "ci_high_95": np.quantile(sample_matrix, 0.975, axis=0),
            "bootstrap_two_sided_p": pvalues,
            "q_value_BH": bh_adjust(pvalues),
        })
    result["analysis"] = "joint Dirichlet regression sensitivity"
    result["sex_for_contrast"] = "Male" if modal_sex else "Female"
    result["n_donors"] = len(metadata)
    audit = {
        "optimizer_success_full_fit": success,
        "requested_bootstrap_replicates": n_bootstrap,
        "successful_bootstrap_replicates": len(samples),
        "failed_bootstrap_replicates": failures,
        "contrast": "expected fraction at age_z=1 minus age_z=0 at modal sex",
        "interpretation": "sensitivity only; donor is the resampling unit",
    }
    return result, audit


def main() -> None:
    args = parse_args()
    project = args.project.resolve()
    pseudobulk = project / "02_pseudobulk"
    metadata_path = project / "01_raw_processing" / "metadata" / "05B_final_regression_cohort.csv"
    output = project / "03_feature_selection" / "composition_adjusted"
    output.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(metadata_path).set_index("donor_id")
    metadata.index = metadata.index.astype(str)
    official = pd.read_csv(pseudobulk / "donor_celltype_proportion_all.csv", index_col=0)
    official.index = official.index.astype(str)
    official = official.loc[metadata.index]

    qc = pd.read_csv(pseudobulk / "pseudobulk_qc_summary_all.csv", index_col=0)
    qc.index = qc.index.astype(str)
    count_columns = [column for column in qc if str(column).startswith("n_cells_")]
    if not count_columns:
        raise ValueError("No n_cells_* columns found in pseudobulk_qc_summary_all.csv")
    totals = qc.loc[metadata.index, count_columns].sum(axis=1)
    if (totals <= 0).any():
        raise ValueError("All donors must have positive all-QC cell totals")

    replaced = zero_replace_and_close(official, totals)
    clr_official = clr(replaced)
    main_results = fit_clr_models(
        clr_official, metadata, "primary_CLR_all_QC_denominator_age_plus_sex"
    )
    main_results.to_csv(output / "composition_results.tsv", sep="\t", index=False)

    sensitivity_frames = []
    fixed = clr(fixed_zero_replace_and_close(official))
    sensitivity_frames.append(fit_clr_models(
        fixed, metadata, "zero_replacement_fixed_1e-6_age_plus_sex"
    ))

    analysis_path = pseudobulk / "donor_celltype_proportion_analysis.csv"
    if analysis_path.exists():
        eligible = pd.read_csv(analysis_path, index_col=0)
        eligible.index = eligible.index.astype(str)
        eligible = eligible.loc[metadata.index]
        eligible_replaced = fixed_zero_replace_and_close(eligible)
        sensitivity_frames.append(fit_clr_models(
            clr(eligible_replaced), metadata,
            "eligible_cells_only_denominator_age_plus_sex",
        ))

    ordered = metadata["age_years"].sort_values()
    subsets = {
        "remove_youngest": metadata.index.difference([ordered.index[0]], sort=False),
        "remove_oldest": metadata.index.difference([ordered.index[-1]], sort=False),
        "remove_both_extremes": metadata.index.difference(
            [ordered.index[0], ordered.index[-1]], sort=False
        ),
        "restricted_age_18_64": metadata.index[
            metadata["age_years"].between(18, 64, inclusive="both")
        ],
        "female_only": metadata.index[
            metadata["sex"].astype(str).str.lower().eq("female")
        ],
    }
    for name, donors in subsets.items():
        if len(donors) < 8:
            continue
        female_only = name == "female_only"
        sensitivity_frames.append(fit_clr_models(
            clr_official.loc[donors], metadata.loc[donors], name,
            female_only=female_only,
        ))
    sensitivity_frames.append(fit_clr_models(
        clr_official, metadata, "nonlinear_age_quadratic_plus_sex", nonlinear=True
    ))
    pd.concat(sensitivity_frames, ignore_index=True).to_csv(
        output / "composition_sensitivity_results.tsv", sep="\t", index=False
    )

    stability_rows = []
    primary_age = main_results[
        main_results["term"].eq("age_z")
    ].set_index("cell_type")
    for excluded in metadata.index:
        donors = metadata.index.difference([excluded], sort=False)
        result = fit_clr_models(
            clr_official.loc[donors], metadata.loc[donors], f"exclude_{excluded}"
        )
        age_rows = result[result["term"].eq("age_z")]
        for _, row in age_rows.iterrows():
            stability_rows.append({
                "excluded_donor": excluded,
                "cell_type": row["cell_type"],
                "age_estimate": row["estimate"],
                "same_sign_as_full": np.sign(row["estimate"]) == np.sign(
                    primary_age.loc[row["cell_type"], "estimate"]
                ),
            })
    stability = pd.DataFrame(stability_rows)
    stability.to_csv(output / "composition_leave_one_donor_out.tsv", sep="\t", index=False)
    summary = stability.groupby("cell_type").agg(
        sign_stability_fraction=("same_sign_as_full", "mean"),
        minimum_age_estimate=("age_estimate", "min"),
        maximum_age_estimate=("age_estimate", "max"),
    ).reset_index()
    summary.to_csv(output / "composition_donor_stability.tsv", sep="\t", index=False)

    dirichlet, dirichlet_audit = dirichlet_bootstrap(
        replaced, metadata, args.dirichlet_bootstrap
    )
    dirichlet.to_csv(output / "composition_dirichlet_sensitivity.tsv", sep="\t", index=False)

    audit = {
        "analysis_unit": "donor",
        "n_donors": len(metadata),
        "primary_denominator": "all QC-passed cells, including unknown/unresolved",
        "primary_transform": "CLR after donor-specific half-cell zero replacement",
        "primary_model": "CLR component ~ standardized age + sex",
        "uncertainty": "HC3 heteroskedasticity-consistent standard errors",
        "multiple_testing": "BH within model term",
        "restricted_age_sensitivity": "18-64 years (prespecified in this script)",
        "dirichlet": dirichlet_audit,
        "note": "Fractions are relative abundances; coefficients are not absolute cell-count changes.",
    }
    (output / "composition_analysis_contract.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    print(f"Composition-aware outputs written to {output}")


if __name__ == "__main__":
    main()

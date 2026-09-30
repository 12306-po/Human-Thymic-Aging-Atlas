#!/usr/bin/env python3
"""Create a publication-style summary of the three external analysis tiers."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = {
    "HRA007984": "#3E73B8",
    "Tabula Sapiens": "#C84C4C",
    "Park E-MTAB-8581": "#4C9B8C",
    "GSE147520": "#8B6BB8",
}
GREY = "#B8B8B8"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    exact = pd.read_csv(args.results / "combined_six_donor_exact_age_predictions.tsv", sep="\t")
    park = pd.read_csv(args.results / "park_interval_age_predictions.tsv", sep="\t")
    gse = pd.read_csv(args.results / "gse147520_stromal_sensitivity.tsv", sep="\t")
    coverage = pd.read_csv(args.results / "multicohort_feature_coverage.tsv", sep="\t")
    summary = json.loads((args.results / "multicohort_external_summary.json").read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "font.size": 8,
        "axes.linewidth": 0.7,
    })
    fig = plt.figure(figsize=(7.2, 5.0), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.05, 1.0])

    ax = fig.add_subplot(grid[0, 0])
    lo = min(exact.chronological_age.min(), exact.predicted_age.min()) - 4
    hi = max(exact.chronological_age.max(), exact.predicted_age.max()) + 4
    ax.plot([lo, hi], [lo, hi], ls="--", lw=0.9, color=GREY, zorder=0)
    for cohort, group in exact.groupby("cohort", sort=False):
        ax.scatter(group.chronological_age, group.predicted_age, s=48,
                   color=COLORS[cohort], edgecolor="white", linewidth=0.7,
                   label=cohort, zorder=3)
        for row in group.itertuples():
            ax.annotate(row.donor_id, (row.chronological_age, row.predicted_age),
                        xytext=(3, 3), textcoords="offset points", fontsize=6)
    ax.set(xlim=(lo, hi), ylim=(lo, hi), xlabel="Chronological age (years)",
           ylabel="Frozen-model predicted age (years)")
    metrics = summary["exact_age_external_evaluation"]
    ax.text(0.03, 0.97,
            f"n = {metrics['n_donors']}\nMAE = {metrics['MAE_years']:.1f} years\n"
            f"Spearman ρ = {metrics['Spearman_rho']:.2f}",
            transform=ax.transAxes, va="top", fontsize=7)
    ax.legend(frameon=False, loc="lower right", fontsize=7)
    ax.set_title("Exact-age external feasibility", fontweight="bold")

    ax = fig.add_subplot(grid[0, 1])
    y = np.arange(len(park))
    ax.hlines(y, park.age_lower, park.age_upper, color=COLORS["Park E-MTAB-8581"], lw=6, alpha=0.28)
    ax.scatter(park.predicted_age, y, color=COLORS["Park E-MTAB-8581"], s=48,
               edgecolor="white", linewidth=0.7, zorder=3, label="Frozen prediction")
    for position, row in enumerate(park.itertuples()):
        ax.text((row.age_lower + row.age_upper) / 2, position + 0.16,
                f"reported {row.age_lower:g}–{row.age_upper:g} y",
                ha="center", va="bottom", fontsize=6.5)
    ax.set_yticks(y)
    ax.set_yticklabels(park.donor_id)
    ax.set_xlabel("Age (years)")
    ax.set_title("Park reported age intervals", fontweight="bold")
    ax.legend(frameon=False, fontsize=7, loc="best")

    ax = fig.add_subplot(grid[1, 0])
    coverage = coverage.copy()
    coverage["label"] = coverage.cohort.astype(str) + " · " + coverage.donor_id.astype(str)
    coverage = coverage.sort_values(["analysis_type", "cohort", "donor_id"]).reset_index(drop=True)
    colors = [COLORS.get(cohort, "#666666") for cohort in coverage.cohort]
    ax.barh(np.arange(len(coverage)), coverage.observed_fraction, color=colors, height=0.68)
    ax.set_yticks(np.arange(len(coverage)))
    ax.set_yticklabels(coverage.label, fontsize=6.5)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Observed fraction of 2,000 frozen features")
    ax.set_title("Feature compatibility", fontweight="bold")
    ax.invert_yaxis()

    ax = fig.add_subplot(grid[1, 1])
    ax.axis("off")
    gse_row = gse.iloc[0]
    gse_cov = coverage.loc[coverage.cohort.eq("GSE147520"), "observed_fraction"].iloc[0]
    lines = [
        "GSE147520 stromal sensitivity",
        "",
        f"Adult donor: {gse_row.get('chronological_age', 25):g} years",
        f"Feature-restricted projection: {gse_row.predicted_age:.1f} years",
        f"Observed frozen features: {gse_cov:.1%}",
        "",
        "Whole-thymus and composition features",
        "were not reconstructed from stromal-enriched data.",
        "No inferential performance metric was calculated.",
    ]
    ax.text(0.04, 0.94, "\n".join(lines), va="top", ha="left", linespacing=1.45,
            bbox={"boxstyle": "round,pad=0.6", "facecolor": "#F4F0FA",
                  "edgecolor": COLORS["GSE147520"], "linewidth": 0.8})
    ax.set_title("Compartment-restricted sensitivity", fontweight="bold")

    for axis, letter in zip(fig.axes, "ABCD"):
        axis.text(-0.13 if axis.axison else -0.02, 1.05, letter, transform=axis.transAxes,
                  fontsize=11, fontweight="bold", va="top")
        if axis.axison:
            axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Independent multi-cohort frozen-model evaluation", fontsize=10, fontweight="bold")
    fig.text(0.5, -0.015, summary["claim_boundary"], ha="center", fontsize=6.5)
    for extension in ("pdf", "svg", "png"):
        fig.savefig(
            args.output / f"Figure_external_multicohort_validation.{extension}",
            dpi=600 if extension == "png" else None,
            bbox_inches="tight",
        )
    plt.close(fig)


if __name__ == "__main__":
    main()

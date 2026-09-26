#!/usr/bin/env python3
"""Plot the completed HRA007984 feasibility test without adding claims."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

UP, DOWN, GREY = "#C84C4C", "#3E73B8", "#B8B8B8"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    pred = pd.read_csv(args.results / "hra007984_frozen_predictions.tsv", sep="\t")
    cov = pd.read_csv(args.results / "hra007984_feature_coverage.tsv", sep="\t")
    metrics = json.loads((args.results / "hra007984_metrics.json").read_text())
    args.output.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"],
                         "pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 8})
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.35), constrained_layout=True)
    ax = axes[0]
    evaluable = pred.dropna(subset=["chronological_age"])
    if len(evaluable):
        lo = min(evaluable.chronological_age.min(), evaluable.predicted_age.min()) - 3
        hi = max(evaluable.chronological_age.max(), evaluable.predicted_age.max()) + 3
        ax.plot([lo, hi], [lo, hi], ls="--", lw=.8, color=GREY)
        ax.scatter(evaluable.chronological_age, evaluable.predicted_age, s=42,
                   c=np.where(evaluable.stage.str.contains("ger", case=False, na=False), UP, DOWN),
                   edgecolor="white", linewidth=.6, zorder=3)
        for _, row in evaluable.iterrows():
            ax.annotate(row.donor_id, (row.chronological_age, row.predicted_age),
                        xytext=(3, 3), textcoords="offset points", fontsize=6)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("Chronological age (years)"); ax.set_ylabel("Frozen-model predicted age (years)")
    ax.set_title("External donor predictions")

    ax = axes[1]
    order = pred.sort_values("predicted_age").reset_index(drop=True)
    colors = np.where(order.stage.str.contains("ger", case=False, na=False), UP, DOWN)
    ax.hlines(np.arange(len(order)), 0, order.predicted_age, color=colors, lw=1.3)
    ax.scatter(order.predicted_age, np.arange(len(order)), c=colors, s=35, zorder=3)
    ax.set_yticks(np.arange(len(order))); ax.set_yticklabels(order.donor_id)
    ax.set_xlabel("Predicted age (years)"); ax.set_title("No-refitting transfer")

    ax = axes[2]
    cov = cov.sort_values("missing_fraction")
    ax.barh(cov.donor_id, 1 - cov.missing_fraction, color="#4C9B8C")
    ax.axvline(.5, ls="--", lw=.8, color=GREY)
    ax.set_xlim(0, 1); ax.set_xlabel("Observed fraction of frozen features")
    ax.set_title("Feature compatibility")
    for a, letter in zip(axes, "ABC"):
        a.text(-.16, 1.04, letter, transform=a.transAxes, fontsize=11, fontweight="bold")
        a.spines[["top", "right"]].set_visible(False)
    fig.suptitle("HRA007984 independent frozen-model feasibility test", fontsize=10, fontweight="bold")
    fig.text(.5, -.02, metrics["claim_boundary"], ha="center", fontsize=6.5)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(args.output / f"Figure_external_HRA007984_feasibility.{ext}",
                    dpi=600 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()


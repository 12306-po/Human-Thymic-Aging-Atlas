#!/usr/bin/env python3
"""Rebuild corrected Figure 5 and Figure 5C from frozen atlas source tables.

This script fixes the Figure 5C stage-key bug.  The source table uses
``DP_CD3min`` and ``DP_CD3plus``; earlier plotting code requested labels with
spaces and therefore created two all-NA columns during ``reindex``.

Inputs under ``<project>/10_results/human_thymic_aging_atlas``:
  atlas_composition_age_effects.tsv
  atlas_gene_age_effects.tsv.gz
  atlas_developmental_context.tsv
  atlas_donor_scores.tsv

Outputs under ``--output-dir``:
  Figure5_corrected.png / .pdf
  Figure5C_corrected.png / .pdf
  Figure5C_source_table.tsv
  Figure5C_validation.tsv
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd


STAGE_KEYS = ["DN1", "DN2", "DN3", "ISP", "DP_CD3min", "DP_CD3plus"]
STAGE_LABELS = ["DN1", "DN2", "DN3", "ISP", "DP CD3min", "DP CD3plus"]
STAGE_OBJECT = {
    "DN1": "Sort1",
    "DN2": "Sort1",
    "DN3": "Sort1",
    "ISP": "Sort2",
    "DP_CD3min": "Sort2",
    "DP_CD3plus": "Sort2",
}

RED = "#B84A4A"
BLUE = "#4575B4"
TEAL = "#2A9D8F"
GRAY = "#7A7A7A"
LIGHT = "#D9D9D9"


def parse_args() -> argparse.Namespace:
    default_project = os.environ.get("PROJECT_DIR") or os.environ.get("PROJ") or "."
    parser = argparse.ArgumentParser(
        description="Rebuild Figure 5 with correctly populated Sort2 DP columns."
    )
    parser.add_argument(
        "--project",
        type=Path,
        default=Path(default_project),
        help="Project root. Defaults to PROJECT_DIR, then PROJ, then current directory.",
    )
    parser.add_argument(
        "--atlas-dir",
        type=Path,
        default=None,
        help="Atlas input directory. Default: <project>/10_results/human_thymic_aging_atlas",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Default: <project>/10_results/figures_manuscript",
    )
    parser.add_argument("--top-per-direction", type=int, default=6)
    parser.add_argument("--dpi", type=int, default=450)
    parser.add_argument(
        "--font-dir",
        type=Path,
        default=None,
        help="Optional directory containing Times New Roman TTF files.",
    )
    return parser.parse_args()


def configure_fonts(font_dir: Path | None) -> None:
    candidates: list[Path] = []
    if font_dir is not None:
        candidates.extend(font_dir.glob("*.ttf"))
        candidates.extend(font_dir.glob("*.TTF"))
    candidates.extend(
        [
            Path.home() / "fonts" / "times_new_roman" / "TIMES.ttf",
            Path.home() / "fonts" / "times_new_roman" / "TIMESBD.ttf",
            Path.home() / "fonts" / "times_new_roman" / "TIMESI.ttf",
            Path.home() / "fonts" / "times_new_roman" / "TIMESBI.ttf",
            Path(r"C:\Windows\Fonts\times.ttf"),
            Path(r"C:\Windows\Fonts\timesbd.ttf"),
            Path(r"C:\Windows\Fonts\timesi.ttf"),
            Path(r"C:\Windows\Fonts\timesbi.ttf"),
        ]
    )
    for path in dict.fromkeys(candidates):
        if path.exists():
            mpl.font_manager.fontManager.addfont(str(path))

    available = {f.name for f in mpl.font_manager.fontManager.ttflist}
    family = "Times New Roman" if "Times New Roman" in available else "DejaVu Serif"
    mpl.rcParams.update(
        {
            "font.family": family,
            "font.size": 7.5,
            "axes.titlesize": 8.5,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 6.7,
            "ytick.labelsize": 6.7,
            "legend.fontsize": 6.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.65,
        }
    )
    print(f"Selected font: {family}")


def require_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def load_inputs(atlas_dir: Path) -> tuple[pd.DataFrame, ...]:
    paths = {
        "composition": atlas_dir / "atlas_composition_age_effects.tsv",
        "genes": atlas_dir / "atlas_gene_age_effects.tsv.gz",
        "developmental": atlas_dir / "atlas_developmental_context.tsv",
        "scores": atlas_dir / "atlas_donor_scores.tsv",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing Figure 5 input files:\n" + "\n".join(missing))

    composition = pd.read_csv(paths["composition"], sep="\t")
    genes = pd.read_csv(paths["genes"], sep="\t")
    developmental = pd.read_csv(paths["developmental"], sep="\t")
    scores = pd.read_csv(paths["scores"], sep="\t")

    require_columns(
        composition,
        {"cell_type", "estimate", "ci_low_95", "ci_high_95", "q_value_BH"},
        "composition table",
    )
    require_columns(
        genes,
        {"gene", "cell_context", "logFC_age_perSD", "P.Value", "adj.P.Val", "n_donors"},
        "gene table",
    )
    require_columns(
        developmental,
        {
            "gene",
            "source_object",
            "developmental_stage",
            "n_cells",
            "mean_expr",
            "pct_expr",
            "mean_zscore",
        },
        "developmental table",
    )
    require_columns(
        scores,
        {
            "donor_id",
            "chronological_age",
            "thymic_age_deviation_years",
            "deviation_percentile_within_18_donors",
        },
        "donor-score table",
    )
    return composition, genes, developmental, scores


def select_panel_c_genes(
    genes: pd.DataFrame, developmental: pd.DataFrame, top_per_direction: int
) -> pd.DataFrame:
    whole = genes.loc[genes["cell_context"].eq("whole_thymus")].copy()
    whole = whole.sort_values(["adj.P.Val", "P.Value"], kind="mergesort")
    present = set(developmental["gene"].dropna().astype(str))
    usable = whole["gene"].astype(str).isin(present)
    top_up = whole.loc[usable & whole["logFC_age_perSD"].gt(0)].head(top_per_direction)
    top_down = whole.loc[usable & whole["logFC_age_perSD"].lt(0)].head(top_per_direction)
    selected = pd.concat([top_up, top_down], ignore_index=True)
    expected = 2 * top_per_direction
    if len(selected) != expected:
        raise RuntimeError(f"Expected {expected} selected genes, found {len(selected)}")
    if selected["gene"].duplicated().any():
        raise RuntimeError("Selected Figure 5C genes are not unique")
    selected = selected.copy()
    selected["age_effect_direction"] = np.where(
        selected["logFC_age_perSD"].gt(0), "age-up", "age-down"
    )
    return selected


def build_panel_c_source(
    selected: pd.DataFrame, developmental: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected_genes = selected["gene"].astype(str).tolist()
    direction = selected.set_index("gene")["age_effect_direction"].to_dict()
    logfc = selected.set_index("gene")["logFC_age_perSD"].to_dict()

    dev = developmental.loc[
        developmental["gene"].astype(str).isin(selected_genes)
        & developmental["developmental_stage"].isin(STAGE_KEYS)
    ].copy()
    duplicates = dev.duplicated(["gene", "developmental_stage"], keep=False)
    if duplicates.any():
        duplicate_keys = dev.loc[duplicates, ["gene", "developmental_stage"]].drop_duplicates()
        raise RuntimeError(
            "Duplicate gene-stage rows in developmental source:\n"
            + duplicate_keys.to_string(index=False)
        )

    grid = pd.MultiIndex.from_product(
        [selected_genes, STAGE_KEYS], names=["gene", "developmental_stage"]
    ).to_frame(index=False)
    source_columns = [
        "gene",
        "developmental_stage",
        "source_object",
        "n_cells",
        "mean_expr",
        "median_expr",
        "pct_expr",
        "mean_zscore",
        "median_zscore",
    ]
    source_columns = [column for column in source_columns if column in dev.columns]
    source = grid.merge(dev[source_columns], on=["gene", "developmental_stage"], how="left")
    source["expected_source_object"] = source["developmental_stage"].map(STAGE_OBJECT)
    source["source_object"] = source["source_object"].fillna(source["expected_source_object"])
    source["age_effect_direction"] = source["gene"].map(direction)
    source["whole_thymus_logFC_age_perSD"] = source["gene"].map(logfc)
    source["availability"] = np.where(source["mean_zscore"].notna(), "observed", "missing")
    source["gene_order"] = pd.Categorical(source["gene"], selected_genes, ordered=True)
    source["stage_order"] = pd.Categorical(
        source["developmental_stage"], STAGE_KEYS, ordered=True
    )
    source = source.sort_values(["gene_order", "stage_order"]).drop(
        columns=["gene_order", "stage_order"]
    )

    heat = source.pivot(
        index="gene", columns="developmental_stage", values="mean_zscore"
    ).reindex(index=selected_genes, columns=STAGE_KEYS)

    # Fail loudly if the old space-vs-underscore error returns.
    dp_nonmissing = heat[["DP_CD3min", "DP_CD3plus"]].notna().sum()
    if dp_nonmissing.eq(0).any():
        raise RuntimeError(
            "A Sort2 DP column is entirely missing after pivoting. "
            "Check that stage keys are DP_CD3min and DP_CD3plus."
        )
    return source, heat


def draw_panel_c(
    ax: plt.Axes,
    selected: pd.DataFrame,
    heat: pd.DataFrame,
    *,
    include_letter: bool,
    include_note: bool,
) -> None:
    values = heat.to_numpy(dtype=float)
    finite = np.abs(values[np.isfinite(values)])
    vmax = max(float(np.nanpercentile(finite, 97)), 0.25) if finite.size else 0.25
    cmap = mpl.colormaps["RdBu_r"].copy()
    cmap.set_bad("#F2F2F2")
    image = ax.imshow(
        np.ma.masked_invalid(values), aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax
    )

    direction_symbol = {
        row.gene: "+" if row.age_effect_direction == "age-up" else "−"
        for row in selected.itertuples(index=False)
    }
    ax.set_yticks(
        np.arange(len(heat.index)),
        [f"{gene} ({direction_symbol[gene]})" for gene in heat.index],
    )
    ax.set_xticks(np.arange(len(STAGE_KEYS)), STAGE_LABELS, rotation=38, ha="right")
    ax.axvline(2.5, color="white", lw=1.5)
    ax.set_title("Developmental localization of top age-associated genes", pad=5)
    ax.text(1.0, -1.12, "Sort1", ha="center", va="bottom", fontsize=6.5)
    ax.text(4.0, -1.12, "Sort2", ha="center", va="bottom", fontsize=6.5)

    # Mark only genuinely absent gene-stage combinations; never mask a whole
    # stage column based on a plotting-label mismatch.
    for row_index, column_index in np.argwhere(np.isnan(values)):
        ax.add_patch(
            Rectangle(
                (column_index - 0.5, row_index - 0.5),
                1,
                1,
                facecolor="#F2F2F2",
                edgecolor="#C8C8C8",
                linewidth=0.35,
                hatch="///",
            )
        )
        ax.text(
            column_index,
            row_index,
            "NA",
            ha="center",
            va="center",
            fontsize=4.8,
            color="#777777",
        )

    colorbar = ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    colorbar.set_label("Mean within-object z-score")
    colorbar.ax.tick_params(labelsize=6.2)
    if include_letter:
        ax.text(
            -0.15,
            1.02,
            "C",
            transform=ax.transAxes,
            fontsize=11,
            fontweight="bold",
            ha="left",
            va="bottom",
        )
    if include_note:
        ax.text(
            0.0,
            -0.24,
            "+/− indicates the human whole-thymus age-effect direction;\n"
            "hatched cells are genuinely unavailable in that source object.\n"
            "GSE195812 provides developmental context, not aging replication.",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=6.2,
            color="#444444",
        )


def draw_panel_a(ax: plt.Axes, composition: pd.DataFrame) -> None:
    comp = composition.sort_values("estimate").reset_index(drop=True)
    y = np.arange(len(comp))
    significant = comp["q_value_BH"].lt(0.05)
    ax.axvline(0, color="#555555", lw=0.7)
    ax.hlines(y, comp["ci_low_95"], comp["ci_high_95"], color=LIGHT, lw=1.2)
    ax.scatter(
        comp.loc[~significant, "estimate"],
        y[~significant],
        facecolors="white",
        edgecolors=GRAY,
        s=22,
        linewidths=0.8,
        label="BH q≥0.05",
    )
    ax.scatter(
        comp.loc[significant, "estimate"],
        y[significant],
        color=TEAL,
        edgecolors="#1C6F65",
        s=25,
        linewidths=0.5,
        label="BH q<0.05",
    )
    ax.set_yticks(y, comp["cell_type"])
    ax.set_xlabel("Sex-adjusted CLR age effect per 1 SD (95% CI)")
    ax.set_title("Composition-aware donor-level effects", pad=5)
    ax.grid(axis="x", color="#E6E6E6", lw=0.5)
    ax.legend(frameon=False, loc="lower right", handletextpad=0.4)


def draw_panel_b(ax: plt.Axes, genes: pd.DataFrame) -> None:
    significant = genes.loc[genes["adj.P.Val"].lt(0.05)].copy()
    significant["direction"] = np.where(
        significant["logFC_age_perSD"].gt(0), "Age-up", "Age-down"
    )
    order = genes["cell_context"].drop_duplicates()
    counts = (
        significant.groupby(["cell_context", "direction"])
        .size()
        .unstack(fill_value=0)
        .reindex(order)
        .fillna(0)
    )
    coverage = genes.groupby("cell_context")["n_donors"].agg(["min", "max"])
    counts["total"] = counts.sum(axis=1)
    counts = counts.sort_values("total")
    y = np.arange(len(counts))
    up = counts.get("Age-up", pd.Series(0, index=counts.index)).astype(int)
    down = counts.get("Age-down", pd.Series(0, index=counts.index)).astype(int)
    x_up, x_down = np.log10(up + 1), np.log10(down + 1)
    ax.scatter(x_up, y + 0.13, color=RED, marker="o", s=25, label="Age-up")
    ax.scatter(x_down, y - 0.13, color=BLUE, marker="s", s=22, label="Age-down")
    for index, (up_count, down_count) in enumerate(zip(up, down)):
        if up_count > 0:
            ax.text(x_up.iloc[index] + 0.035, index + 0.13, str(up_count), va="center", fontsize=5.8, color=RED)
        if down_count > 0:
            ax.text(x_down.iloc[index] + 0.035, index - 0.13, str(down_count), va="center", fontsize=5.8, color=BLUE)
    labels = []
    for context in counts.index:
        low, high = coverage.loc[context, ["min", "max"]]
        n_label = f"n={int(low)}" if low == high else f"n={int(low)}–{int(high)}"
        labels.append(f"{context} ({n_label})")
    ax.set_yticks(y, labels)
    ax.set_xticks(np.arange(0, 4, 1), ["0", "9", "99", "999"])
    ax.set_xlim(-0.05, max(3.65, float(max(x_up.max(), x_down.max())) + 0.35))
    ax.set_xlabel("Number of BH q<0.05 genes (log10[n+1])")
    ax.set_title("Cell-context transcriptional atlas", pad=5)
    ax.grid(axis="x", color="#E6E6E6", lw=0.5)
    ax.legend(frameon=False, loc="lower right", ncol=2, handletextpad=0.35, columnspacing=0.8)


def draw_panel_d(ax: plt.Axes, scores: pd.DataFrame) -> None:
    score = scores.sort_values("thymic_age_deviation_years").reset_index(drop=True)
    y = np.arange(len(score))
    values = score["thymic_age_deviation_years"].to_numpy(dtype=float)
    colors = np.where(values >= 0, RED, BLUE)
    ax.axvline(0, color="#555555", lw=0.8)
    ax.hlines(y, 0, values, color=colors, lw=1.35)
    ax.scatter(values, y, color=colors, s=25, zorder=3)
    labels = [
        f"{donor} ({age:.0f} y)"
        for donor, age in zip(score["donor_id"], score["chronological_age"])
    ]
    ax.set_yticks(y, labels)
    ax.set_xlabel("Post-hoc age-conditioned residual (years)")
    ax.set_title("Descriptive post-hoc donor residual", pad=5)
    ax.grid(axis="x", color="#E6E6E6", lw=0.5)
    ax.set_xlim(min(values) - 2.2, max(values) + 2.8)
    for index, (value, percentile) in enumerate(
        zip(values, score["deviation_percentile_within_18_donors"])
    ):
        horizontal = "left" if value >= 0 else "right"
        offset = 0.18 if value >= 0 else -0.18
        ax.text(value + offset, index, f"P{int(round(percentile))}", ha=horizontal, va="center", fontsize=5.8, color=colors[index])
    ax.text(
        0.0,
        -0.20,
        "Post-hoc residual from existing OOF predictions; descriptive within these 18 donors only.\n"
        "Not a strictly held-out score and not a validated biological-age measure.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.2,
        color="#444444",
    )


def save_panel_c(
    selected: pd.DataFrame, heat: pd.DataFrame, output_dir: Path, dpi: int
) -> None:
    figure, axis = plt.subplots(figsize=(4.2, 4.8))
    figure.subplots_adjust(left=0.30, right=0.88, top=0.91, bottom=0.28)
    draw_panel_c(axis, selected, heat, include_letter=True, include_note=True)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)
    figure.savefig(output_dir / "Figure5C_corrected.png", dpi=dpi, bbox_inches="tight", facecolor="white")
    figure.savefig(output_dir / "Figure5C_corrected.pdf", bbox_inches="tight", facecolor="white")
    plt.close(figure)


def save_full_figure(
    composition: pd.DataFrame,
    genes: pd.DataFrame,
    scores: pd.DataFrame,
    selected: pd.DataFrame,
    heat: pd.DataFrame,
    output_dir: Path,
    dpi: int,
) -> None:
    figure = plt.figure(figsize=(7.15, 8.55), constrained_layout=False)
    grid = figure.add_gridspec(
        2,
        2,
        left=0.11,
        right=0.985,
        top=0.975,
        bottom=0.075,
        hspace=0.34,
        wspace=0.43,
        height_ratios=[1.0, 1.05],
    )
    axes = [figure.add_subplot(grid[row, column]) for row in range(2) for column in range(2)]
    draw_panel_a(axes[0], composition)
    draw_panel_b(axes[1], genes)
    draw_panel_c(axes[2], selected, heat, include_letter=False, include_note=True)
    draw_panel_d(axes[3], scores)
    for label, axis in zip("ABCD", axes):
        axis.text(
            -0.15,
            1.02,
            label,
            transform=axis.transAxes,
            fontsize=11,
            fontweight="bold",
            ha="left",
            va="bottom",
        )
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    figure.savefig(output_dir / "Figure5_corrected.png", dpi=dpi, bbox_inches="tight", facecolor="white")
    figure.savefig(output_dir / "Figure5_corrected.pdf", bbox_inches="tight", facecolor="white")
    plt.close(figure)


def write_validation(
    developmental: pd.DataFrame,
    selected: pd.DataFrame,
    heat: pd.DataFrame,
    output_dir: Path,
) -> None:
    rows: list[dict[str, object]] = []
    for stage in STAGE_KEYS:
        stage_data = developmental.loc[developmental["developmental_stage"].eq(stage)]
        rows.append(
            {
                "check_scope": "full_developmental_table",
                "stage": stage,
                "n_rows": len(stage_data),
                "n_genes": stage_data["gene"].nunique(),
                "n_nonmissing_mean_zscore": stage_data["mean_zscore"].notna().sum(),
                "status": "PASS" if len(stage_data) > 0 and stage_data["mean_zscore"].notna().any() else "FAIL",
            }
        )
    for stage in ("DP_CD3min", "DP_CD3plus"):
        rows.append(
            {
                "check_scope": "selected_Figure5C_genes",
                "stage": stage,
                "n_rows": len(selected),
                "n_genes": len(selected),
                "n_nonmissing_mean_zscore": int(heat[stage].notna().sum()),
                "status": "PASS" if heat[stage].notna().any() else "FAIL",
            }
        )
    validation = pd.DataFrame(rows)
    validation.to_csv(output_dir / "Figure5C_validation.tsv", sep="\t", index=False)
    if validation["status"].eq("FAIL").any():
        raise RuntimeError("Figure 5C validation failed; inspect Figure5C_validation.tsv")


def main() -> None:
    args = parse_args()
    project = args.project.expanduser().resolve()
    atlas_dir = (
        args.atlas_dir.expanduser().resolve()
        if args.atlas_dir is not None
        else project / "10_results" / "human_thymic_aging_atlas"
    )
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else project / "10_results" / "figures_manuscript"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_fonts(args.font_dir)

    composition, genes, developmental, scores = load_inputs(atlas_dir)
    selected = select_panel_c_genes(genes, developmental, args.top_per_direction)
    source, heat = build_panel_c_source(selected, developmental)

    source.to_csv(output_dir / "Figure5C_source_table.tsv", sep="\t", index=False)
    write_validation(developmental, selected, heat, output_dir)
    save_panel_c(selected, heat, output_dir, args.dpi)
    save_full_figure(composition, genes, scores, selected, heat, output_dir, args.dpi)

    print("Selected Figure 5C genes:", ", ".join(selected["gene"]))
    print("Sort2 DP observed values:")
    print(heat[["DP_CD3min", "DP_CD3plus"]].notna().sum().to_string())
    print("Outputs written to:", output_dir)


if __name__ == "__main__":
    main()

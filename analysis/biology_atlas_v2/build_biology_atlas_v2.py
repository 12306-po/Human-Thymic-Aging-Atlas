from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests


COLORS = {
    "Thymocyte/T": "#355C8C",
    "B lineage": "#9D5C9A",
    "Myeloid/DC": "#D18A39",
    "TEC": "#B84A5A",
    "Fibroblast": "#4B8F77",
    "Endothelial": "#57A6A1",
    "Unresolved": "#9A9A9A",
}

LEVEL2_COLORS = {
    "Early DN-like": "#213B67",
    "Cycling DN": "#355C8C",
    "ISP-like": "#577FB1",
    "Alpha-beta entry": "#7A9CC6",
    "Cycling DP": "#4169A1",
    "Quiescent DP": "#6C8EC0",
    "HSP-associated DP": "#9AAED0",
    "CD4 SP": "#BD5B73",
    "CD8 SP": "#8A4F8D",
    "Treg and agonist": "#C87B93",
    "Gamma-delta NKT and NK": "#6B4D8D",
    "B and plasma": "#A06AA5",
    "DC and myeloid": "#D18A39",
    "cTEC": "#C94C5D",
    "mTEC": "#E07972",
    "Specialized TEC": "#E5A1A1",
    "Fibroblast and VSMC": "#4B8F77",
    "Endothelial": "#57A6A1",
    "Unresolved": "#A0A0A0",
}

T_LEVEL2_ORDER = [
    "Early DN-like", "Cycling DN", "ISP-like", "Alpha-beta entry",
    "Cycling DP", "Quiescent DP", "HSP-associated DP", "CD4 SP",
    "CD8 SP", "Treg and agonist", "Gamma-delta NKT and NK",
]

NICHE_LEVEL2_ORDER = [
    "cTEC", "mTEC", "Specialized TEC", "Fibroblast and VSMC",
    "Endothelial", "DC and myeloid",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build biology-centered atlas v2 figures and source tables.")
    parser.add_argument("--project-root", type=Path, required=True,
                        help="human_thymus_age_ML_DL project root")
    parser.add_argument("--atlas-root", type=Path, default=None,
                        help="Human-Thymic-Aging-Atlas repository root")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output directory; relative paths resolve from atlas root")
    parser.add_argument("--seed", type=int, default=371)
    return parser.parse_args()


def configure_style() -> None:
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 8.5,
        "axes.titlesize": 10,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7.4,
        "ytick.labelsize": 7.4,
        "axes.linewidth": 0.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
    })
    sns.set_style("whitegrid", {"axes.grid": False})


def decode(values: np.ndarray) -> np.ndarray:
    return np.array([x.decode("utf-8") if isinstance(x, (bytes, np.bytes_)) else str(x)
                     for x in values], dtype=object)


def read_obs_column(obs: h5py.Group, name: str) -> np.ndarray:
    obj = obs[name]
    if isinstance(obj, h5py.Group) and obj.attrs.get("encoding-type") == "categorical":
        categories = decode(obj["categories"][:])
        codes = obj["codes"][:]
        result = np.full(codes.shape, None, dtype=object)
        valid = codes >= 0
        result[valid] = categories[codes[valid]]
        return result
    values = obj[:]
    return decode(values) if values.dtype.kind in {"O", "S", "U"} else values


def load_cell_metadata(h5ad: Path) -> tuple[pd.DataFrame, np.ndarray]:
    columns = [
        "donor_id", "broad_celltype", "author_id_lv1", "author_id_lv2",
        "author_id_lv3", "author_id_lv4", "age_years", "sex",
    ]
    with h5py.File(h5ad, "r") as handle:
        obs = pd.DataFrame({name: read_obs_column(handle["obs"], name) for name in columns})
        umap = np.asarray(handle["obsm"]["X_umap"][:], dtype=float)
    for column in ["author_id_lv1", "author_id_lv2", "author_id_lv3", "author_id_lv4"]:
        obs[column] = obs[column].replace({None: np.nan, "None": np.nan, "nan": np.nan})
    return obs, umap


def make_level1(row: pd.Series) -> str:
    broad = row["broad_celltype"]
    if broad in {"DN", "DP", "SP_CD4", "SP_CD8", "SP_unresolved", "Treg",
                 "GammaDelta_T", "NKT_like", "NK"}:
        return "Thymocyte/T"
    if broad in {"B", "Plasma"}:
        return "B lineage"
    if broad in {"DC", "Macrophage", "Monocyte", "Myeloid"}:
        return "Myeloid/DC"
    if broad == "TEC":
        return "TEC"
    if broad == "Fibroblast":
        return "Fibroblast"
    if broad == "Endothelial":
        return "Endothelial"
    return "Unresolved"


def make_level2(row: pd.Series) -> str:
    broad = row["broad_celltype"]
    fine = row["author_id_lv4"]
    if broad == "DN":
        if fine == "DN_P":
            return "Cycling DN"
        if fine == "ISP-like":
            return "ISP-like"
        if fine in {"ETP", "TP1", "TP2", "DN_Q1", "DN_Q2"}:
            return "Early DN-like"
        if fine == "γδT_P":
            return "Gamma-delta NKT and NK"
        return "Early DN-like"
    if broad == "DP":
        return {
            "αβ_Entry": "Alpha-beta entry",
            "DP_P": "Cycling DP",
            "DP_Q": "Quiescent DP",
            "HSP_DP": "HSP-associated DP",
        }.get(fine, "Quiescent DP")
    if broad == "SP_CD4":
        return "CD4 SP"
    if broad == "SP_CD8":
        return "CD8 SP"
    if broad in {"Treg", "SP_unresolved"}:
        return "Treg and agonist"
    if broad in {"GammaDelta_T", "NKT_like", "NK"}:
        return "Gamma-delta NKT and NK"
    if broad in {"B", "Plasma"}:
        return "B and plasma"
    if broad in {"DC", "Macrophage", "Monocyte", "Myeloid"}:
        return "DC and myeloid"
    if broad == "TEC":
        if fine in {"cTEC_hi", "cTEC_lo", "MKI67+cTEC"}:
            return "cTEC"
        if fine in {"Immature_TEC", "mTEC_lo", "mTEC_hi", "post_AIRE_mTEC"}:
            return "mTEC"
        return "Specialized TEC"
    if broad == "Fibroblast":
        return "Fibroblast and VSMC"
    if broad == "Endothelial":
        return "Endothelial"
    return "Unresolved"


def add_hierarchy(obs: pd.DataFrame) -> pd.DataFrame:
    result = obs.copy()
    result["atlas_level1"] = result.apply(make_level1, axis=1)
    result["atlas_level2"] = result.apply(make_level2, axis=1)
    result["atlas_level3"] = result["author_id_lv4"].fillna(result["broad_celltype"] + " unresolved")
    return result


def annotation_audit(obs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    level2 = (obs.groupby(["atlas_level1", "atlas_level2"], dropna=False)
              .agg(n_cells=("donor_id", "size"), n_donors=("donor_id", "nunique"),
                   min_age=("age_years", "min"), max_age=("age_years", "max"))
              .reset_index())
    level3 = (obs.groupby(["atlas_level1", "atlas_level2", "atlas_level3"], dropna=False)
              .agg(n_cells=("donor_id", "size"), n_donors=("donor_id", "nunique"),
                   min_age=("age_years", "min"), max_age=("age_years", "max"))
              .reset_index())
    level3["analysis_status"] = np.select(
        [
            (level3["n_donors"] >= 12) & (level3["n_cells"] >= 100),
            (level3["n_donors"] >= 8) & (level3["n_cells"] >= 20),
        ],
        ["eligible_for_donor_model", "exploratory_only"],
        default="descriptive_only",
    )
    level3["status_reason"] = np.select(
        [
            level3["analysis_status"].eq("eligible_for_donor_model"),
            level3["analysis_status"].eq("exploratory_only"),
        ],
        [
            "at least 100 cells and at least 12 donors",
            "at least 20 cells and at least 8 donors; not a primary endpoint",
        ],
        default="insufficient cell and/or donor coverage for donor-level modeling",
    )
    return level2.sort_values(["atlas_level1", "n_cells"], ascending=[True, False]), level3.sort_values("n_cells", ascending=False)


def donor_level_fractions(obs: pd.DataFrame) -> pd.DataFrame:
    metadata = obs[["donor_id", "age_years", "sex"]].drop_duplicates("donor_id")
    rows: list[dict[str, object]] = []
    families = {
        "T lineage": T_LEVEL2_ORDER,
        "TEC": ["cTEC", "mTEC", "Specialized TEC"],
        "Stromal vascular": ["Fibroblast and VSMC", "Endothelial"],
        "Myeloid": ["DC and myeloid"],
    }
    for family, labels in families.items():
        family_cells = obs.loc[obs["atlas_level2"].isin(labels)]
        counts = pd.crosstab(family_cells["donor_id"], family_cells["atlas_level2"])
        counts = counts.reindex(index=metadata["donor_id"], columns=labels, fill_value=0)
        totals = counts.sum(axis=1)
        for label in labels:
            for donor_id in counts.index:
                count = int(counts.loc[donor_id, label])
                denominator = int(totals.loc[donor_id])
                fraction = count / denominator if denominator else np.nan
                rows.append({
                    "donor_id": donor_id,
                    "family": family,
                    "atlas_level2": label,
                    "n_cells": count,
                    "lineage_denominator": denominator,
                    "within_lineage_fraction": fraction,
                })
    result = pd.DataFrame(rows).merge(metadata, on="donor_id", how="left")
    return result


def fit_fraction_models(fractions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (family, label), frame in fractions.groupby(["family", "atlas_level2"]):
        frame = frame.dropna(subset=["within_lineage_fraction", "age_years", "sex"]).copy()
        if len(frame) < 8 or frame["within_lineage_fraction"].nunique() < 2:
            continue
        frame["age_z"] = (frame["age_years"] - frame["age_years"].mean()) / frame["age_years"].std(ddof=0)
        numerator = frame["n_cells"].astype(float) + 0.5
        denominator = frame["lineage_denominator"].astype(float) + 1.0
        probability = np.clip(numerator / denominator, 1e-6, 1 - 1e-6)
        frame["logit_fraction"] = np.log(probability / (1 - probability))
        x = pd.DataFrame({
            "age_z": frame["age_z"].astype(float),
            "male": frame["sex"].eq("Male").astype(float),
        })
        x = sm.add_constant(x, has_constant="add")
        fit = sm.OLS(frame["logit_fraction"].astype(float), x).fit(cov_type="HC3")
        rho, rho_p = spearmanr(frame["age_years"], frame["within_lineage_fraction"])
        rows.append({
            "family": family,
            "atlas_level2": label,
            "n_donors": int(len(frame)),
            "n_donors_detected": int((frame["n_cells"] > 0).sum()),
            "n_cells": int(frame["n_cells"].sum()),
            "beta_age_per_sd_logit": float(fit.params["age_z"]),
            "se_hc3": float(fit.bse["age_z"]),
            "ci_low_95": float(fit.conf_int().loc["age_z", 0]),
            "ci_high_95": float(fit.conf_int().loc["age_z", 1]),
            "p_value": float(fit.pvalues["age_z"]),
            "spearman_rho_descriptive": float(rho),
            "spearman_p_descriptive": float(rho_p),
            "analysis_scope": "exploratory sex-adjusted within-lineage logit-fraction model; HC3 uncertainty",
        })
    result = pd.DataFrame(rows)
    if not result.empty:
        result["q_value_within_family"] = np.nan
        for family, idx in result.groupby("family").groups.items():
            result.loc[idx, "q_value_within_family"] = multipletests(result.loc[idx, "p_value"], method="fdr_bh")[1]
    return result.sort_values(["family", "beta_age_per_sd_logit"])


def save_figure(fig: plt.Figure, output: Path, name: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for extension, kwargs in [("png", {"dpi": 400}), ("pdf", {}), ("svg", {})]:
        fig.savefig(output / f"{name}.{extension}", **kwargs)
    plt.close(fig)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.11, 1.06, label, transform=ax.transAxes, fontsize=13, fontweight="bold", va="top")


def clean_axis(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)


def scatter_umap(ax: plt.Axes, umap: np.ndarray, labels: pd.Series,
                 palette: dict[str, str], seed: int, max_cells: int = 45000,
                 legend_columns: int = 2, point_size: float = 1.0,
                 legend_inside: bool = False) -> None:
    rng = np.random.default_rng(seed)
    n = len(labels)
    idx = np.arange(n) if n <= max_cells else np.sort(rng.choice(n, max_cells, replace=False))
    values = labels.astype(str).to_numpy()
    order = sorted(pd.unique(values[idx]))
    for label in order:
        keep = idx[values[idx] == label]
        ax.scatter(umap[keep, 0], umap[keep, 1], s=point_size, color=palette.get(label, "#A0A0A0"),
                   alpha=0.68, linewidth=0, rasterized=True, label=label)
    ax.set(xticks=[], yticks=[], xlabel="UMAP 1", ylabel="UMAP 2")
    if legend_inside:
        ax.legend(loc="upper left", bbox_to_anchor=(.01, .99), frameon=True,
                  facecolor="white", edgecolor="#D9D9D9", framealpha=.88,
                  markerscale=3.5, fontsize=6.2, ncol=legend_columns,
                  columnspacing=.7, handletextpad=.25)
    else:
        ax.legend(loc="center left", bbox_to_anchor=(1.01, .5), frameon=False,
                  markerscale=4, fontsize=6.8, ncol=legend_columns, columnspacing=.8, handletextpad=.3)
    ax.set_aspect("equal")


def draw_box(ax: plt.Axes, xy: tuple[float, float], width: float, height: float,
             text: str, face: str, size: float = 7.5) -> None:
    patch = FancyBboxPatch(xy, width, height, boxstyle="round,pad=0.015,rounding_size=0.02",
                           linewidth=.8, edgecolor="#455A64", facecolor=face)
    ax.add_patch(patch)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, text, ha="center", va="center", fontsize=size)


def figure1(obs: pd.DataFrame, umap: np.ndarray, level2_audit: pd.DataFrame,
            output: Path, seed: int) -> None:
    fig = plt.figure(figsize=(8.2, 9.2))
    gs = fig.add_gridspec(3, 2, height_ratios=[.78, 1.45, 1.05], hspace=.46, wspace=.38)
    ax_a = fig.add_subplot(gs[0, 0]); ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0]); ax_d = fig.add_subplot(gs[1, 1])
    ax_e = fig.add_subplot(gs[2, 0]); ax_f = fig.add_subplot(gs[2, 1])

    panel_label(ax_a, "A")
    ax_a.axis("off")
    steps = [
        ("18 donors\n4-69 y", "#E8F1F8"),
        ("114,957\npost-QC cells", "#E4F2EE"),
        ("Level 1\ninference", "#F6ECEE"),
        ("Levels 2-3\natlas", "#F4EFE4"),
        ("Interactive\nresource", "#EEEAF4"),
    ]
    for i, (text, color) in enumerate(steps):
        x = .01 + i * .197
        draw_box(ax_a, (x, .31), .165, .38, text, color, 6.55)
        if i < len(steps) - 1:
            ax_a.annotate("", xy=(x + .195, .50), xytext=(x + .17, .50),
                          arrowprops=dict(arrowstyle="->", color="#607D8B", lw=1))
    ax_a.set(xlim=(0, 1), ylim=(0, 1))
    ax_a.set_title("Study design", loc="left", fontweight="bold")

    panel_label(ax_b, "B")
    donor = obs[["donor_id", "age_years", "sex"]].drop_duplicates().sort_values("age_years")
    sex_palette = {"Female": "#B54E6D", "Male": "#355C8C"}
    for sex, frame in donor.groupby("sex"):
        ax_b.scatter(frame["age_years"], np.arange(len(frame)) * 0 + (0 if sex == "Female" else 1),
                     s=52, color=sex_palette[sex], edgecolor="white", linewidth=.6, label=f"{sex} (n={len(frame)})")
    ax_b.set(yticks=[0, 1], yticklabels=["Female", "Male"], xlabel="Age (years)", title="Donor age and sex")
    ax_b.set_ylim(-.55, 1.55); clean_axis(ax_b)

    panel_label(ax_c, "C")
    scatter_umap(ax_c, umap, obs["atlas_level1"], COLORS, seed, max_cells=50000, legend_columns=1, point_size=1.2)
    ax_c.set_title("Level 1 broad atlas", loc="left", fontweight="bold")

    panel_label(ax_d, "D")
    scatter_umap(ax_d, umap, obs["atlas_level2"], LEVEL2_COLORS, seed + 1, max_cells=50000, legend_columns=2, point_size=1.1)
    ax_d.set_title("Level 2 lineage-resolved atlas", loc="left", fontweight="bold")

    panel_label(ax_e, "E")
    ax_e.axis("off")
    draw_box(ax_e, (.36, .84), .28, .11, "Human thymus", "#F2F5F4", 8)
    branches = [
        ("Thymocytes", T_LEVEL2_ORDER[:5] + ["SP and regulatory"]),
        ("TEC", ["cTEC", "mTEC", "Specialized TEC"]),
        ("Stromal", ["Fibroblast/VSMC", "Endothelial"]),
        ("Immune", ["B/plasma", "DC/myeloid", "NK/NKT/gamma-delta"]),
    ]
    xs = [.03, .28, .53, .78]
    for x, (parent, children) in zip(xs, branches):
        ax_e.plot([.5, x + .085], [.84, .70], color="#8A9698", lw=.8)
        draw_box(ax_e, (x, .57), .17, .12, parent, "#E7EFED", 7.3)
        for j, child in enumerate(children[:4]):
            y = .42 - j * .105
            ax_e.plot([x + .085, x + .085], [.57, y + .06], color="#B1B8B9", lw=.6)
            ax_e.text(x + .09, y, child, ha="center", va="center", fontsize=5.9)
    ax_e.set(xlim=(0, 1), ylim=(0, 1), title="Annotation hierarchy")

    panel_label(ax_f, "F")
    plot = level2_audit.sort_values("n_cells", ascending=True)
    y = np.arange(len(plot))
    colors = [LEVEL2_COLORS.get(x, "#999999") for x in plot["atlas_level2"]]
    ax_f.barh(y, plot["n_donors"], color=colors, alpha=.9)
    ax_f.set(yticks=y, yticklabels=plot["atlas_level2"], xlim=(0, 18.5), xlabel="Donors with detected cells",
             title="Level 2 donor coverage")
    ax_f.axvline(12, color="#333333", ls="--", lw=.8)
    ax_f.text(12.1, len(plot) - .5, "audit threshold", fontsize=6.7, va="top")
    clean_axis(ax_f)
    fig.suptitle("A donor-resolved hierarchical atlas of the aging human thymus", y=.995, fontsize=14, fontweight="bold")
    fig.text(.5, .008, "Level 1 is retained for primary donor-level inference; Levels 2-3 provide an audited biological atlas layer.",
             ha="center", fontsize=7.2)
    save_figure(fig, output, "Figure1_hierarchical_atlas")


def plot_fraction_trajectories(ax: plt.Axes, fractions: pd.DataFrame, labels: list[str], palette: dict[str, str]) -> None:
    for label in labels:
        frame = fractions.loc[fractions["atlas_level2"].eq(label)].dropna(subset=["within_lineage_fraction"])
        if frame.empty:
            continue
        ax.scatter(frame["age_years"], frame["within_lineage_fraction"], s=18,
                   color=palette.get(label, "#777777"), alpha=.65, edgecolor="white", linewidth=.3)
        if frame["age_years"].nunique() >= 4:
            z = np.polyfit(frame["age_years"], frame["within_lineage_fraction"], 1)
            x = np.linspace(frame["age_years"].min(), frame["age_years"].max(), 80)
            ax.plot(x, np.polyval(z, x), color=palette.get(label, "#777777"), lw=1.4, label=label)
    ax.set(xlabel="Age (years)", ylabel="Fraction within recovered lineage")
    clean_axis(ax)


def figure2(obs: pd.DataFrame, umap: np.ndarray, fractions: pd.DataFrame, models: pd.DataFrame,
            level3_audit: pd.DataFrame, output: Path, seed: int) -> None:
    mask = obs["atlas_level2"].isin(T_LEVEL2_ORDER)
    fig = plt.figure(figsize=(8.2, 9.2))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.35, .82, 1.08], hspace=.48, wspace=.43)
    ax_a = fig.add_subplot(gs[0, 0]); ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :]); ax_d = fig.add_subplot(gs[2, 0]); ax_e = fig.add_subplot(gs[2, 1])

    panel_label(ax_a, "A")
    scatter_umap(ax_a, umap[mask.to_numpy()], obs.loc[mask, "atlas_level2"], LEVEL2_COLORS, seed,
                 max_cells=50000, legend_columns=1, point_size=1.2)
    ax_a.set_title("Thymocyte and T-lineage states", loc="left", fontweight="bold")

    panel_label(ax_b, "B")
    ax_b.axis("off")
    developmental = ["Early DN-like", "Cycling DN", "ISP-like", "Alpha-beta entry",
                     "Cycling DP", "Quiescent DP", "CD4 SP", "CD8 SP"]
    for i, label in enumerate(developmental):
        y = .92 - i * .105
        draw_box(ax_b, (.18, y - .045), .58, .07, label, LEVEL2_COLORS[label] + "35", 7.2)
        if i < len(developmental) - 1:
            ax_b.annotate("", xy=(.47, y - .085), xytext=(.47, y - .052),
                          arrowprops=dict(arrowstyle="->", lw=.8, color="#667777"))
    ax_b.text(.82, .25, "Parallel states\nTreg and agonist\nGamma-delta NKT and NK",
              fontsize=7, va="center")
    ax_b.set(xlim=(0, 1), ylim=(0, 1), title="Conservative developmental ordering")

    panel_label(ax_c, "C")
    selected = ["Early DN-like", "Cycling DN", "Cycling DP", "Quiescent DP", "CD4 SP", "CD8 SP"]
    plot_fraction_trajectories(ax_c, fractions.loc[fractions["family"].eq("T lineage")], selected, LEVEL2_COLORS)
    ax_c.legend(ncol=3, frameon=False, fontsize=6.5, loc="upper center", bbox_to_anchor=(.62, 1.19))
    ax_c.set_title("Exploratory donor trajectories within the recovered T lineage", loc="left", fontweight="bold", y=1.23)

    panel_label(ax_d, "D")
    model = models.loc[models["family"].eq("T lineage")].copy().sort_values("beta_age_per_sd_logit")
    y = np.arange(len(model))
    ax_d.errorbar(model["beta_age_per_sd_logit"], y,
                  xerr=[model["beta_age_per_sd_logit"] - model["ci_low_95"],
                        model["ci_high_95"] - model["beta_age_per_sd_logit"]],
                  fmt="o", color="#355C8C", ecolor="#8DA3BC", capsize=2, ms=4)
    ax_d.axvline(0, color="#555555", lw=.8)
    ax_d.set(yticks=y, yticklabels=model["atlas_level2"], xlabel="Age coefficient per SD\nlogit within-lineage fraction",
             title="Sex-adjusted exploratory effects")
    clean_axis(ax_d)

    panel_label(ax_e, "E")
    focus = level3_audit.loc[level3_audit["atlas_level1"].eq("Thymocyte/T")].head(18).copy()
    focus = focus.sort_values(["n_donors", "n_cells"], ascending=True)
    status_colors = {"eligible_for_donor_model": "#4B8F77", "exploratory_only": "#D3A24F", "descriptive_only": "#B5B5B5"}
    ax_e.barh(np.arange(len(focus)), focus["n_donors"], color=focus["analysis_status"].map(status_colors))
    ax_e.set(yticks=np.arange(len(focus)), yticklabels=focus["atlas_level3"], xlim=(0, 18.5),
             xlabel="Donors detected", title="Source Level 3 coverage audit")
    ax_e.axvline(12, color="#333333", ls="--", lw=.8); clean_axis(ax_e)
    ax_e.text(.99, .02, "Green: eligible by coverage audit", transform=ax_e.transAxes,
              ha="right", va="bottom", fontsize=6.2, color="#356F5B")

    fig.suptitle("Thymocyte development is resolved as an age-stratified atlas layer", y=.995, fontsize=14, fontweight="bold")
    fig.text(.5, .008, "Fine-state fraction models are exploratory and use lineage-specific recovered-cell denominators, not intact-thymus abundance.", ha="center", fontsize=7.2)
    save_figure(fig, output, "Figure2_thymocyte_development")


def gene_effect_matrix(effects: pd.DataFrame, contexts: list[str], n_genes: int = 22) -> pd.DataFrame:
    subset = effects.loc[effects["cell_context"].isin(contexts)].copy()
    subset["sig"] = subset["adj.P.Val"].lt(.05)
    score = (subset.groupby("gene")
             .agg(n_sig=("sig", "sum"), max_abs=("logFC_age_perSD", lambda x: np.nanmax(np.abs(x))))
             .sort_values(["n_sig", "max_abs"], ascending=False))
    genes = score.head(n_genes).index
    matrix = subset.loc[subset["gene"].isin(genes)].pivot_table(index="gene", columns="cell_context", values="logFC_age_perSD")
    return matrix.reindex(index=genes, columns=contexts)


def figure3(obs: pd.DataFrame, umap: np.ndarray, fractions: pd.DataFrame, models: pd.DataFrame,
            effects: pd.DataFrame, cell_pathways: pd.DataFrame, output: Path, seed: int) -> None:
    mask = obs["atlas_level2"].isin(NICHE_LEVEL2_ORDER)
    fig = plt.figure(figsize=(8.2, 9.2))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.32, .86, 1.05],
                          width_ratios=[.92, 1.08], hspace=.48, wspace=.72)
    ax_a = fig.add_subplot(gs[0, 0]); ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :]); ax_d = fig.add_subplot(gs[2, 0]); ax_e = fig.add_subplot(gs[2, 1])

    panel_label(ax_a, "A")
    scatter_umap(ax_a, umap[mask.to_numpy()], obs.loc[mask, "atlas_level2"], LEVEL2_COLORS, seed,
                 max_cells=25000, legend_columns=1, point_size=2.2, legend_inside=True)
    ax_a.set_title("Epithelial stromal and myeloid niche", loc="left", fontweight="bold")

    panel_label(ax_b, "B")
    audit = (obs.loc[mask].groupby("atlas_level2")
             .agg(n_cells=("donor_id", "size"), n_donors=("donor_id", "nunique"))
             .reindex(NICHE_LEVEL2_ORDER).dropna().sort_values("n_cells"))
    y = np.arange(len(audit))
    ax_b.barh(y, audit["n_cells"], color=[LEVEL2_COLORS[x] for x in audit.index])
    for i, (_, row) in enumerate(audit.iterrows()):
        ax_b.text(row["n_cells"] * 1.02, i, f"{int(row['n_donors'])} donors", va="center", fontsize=7)
    ax_b.set(yticks=y, yticklabels=audit.index, xlabel="Captured cells", title="Niche coverage")
    ax_b.set_xscale("log"); clean_axis(ax_b)

    panel_label(ax_c, "C")
    selected = ["cTEC", "mTEC", "Specialized TEC", "Fibroblast and VSMC", "Endothelial"]
    plot_fraction_trajectories(ax_c, fractions.loc[fractions["atlas_level2"].isin(selected)], selected, LEVEL2_COLORS)
    ax_c.legend(ncol=3, frameon=False, fontsize=6.5, loc="upper center", bbox_to_anchor=(.62, 1.19))
    ax_c.set_title("Exploratory donor trajectories within recovered epithelial or stromal compartments", loc="left", fontweight="bold", y=1.23)

    panel_label(ax_d, "D")
    matrix = gene_effect_matrix(effects, ["TEC", "Fibroblast", "DC"], n_genes=18)
    sns.heatmap(matrix, ax=ax_d, cmap="RdBu_r", center=0, linewidths=.25,
                linecolor="white", cbar=False)
    ax_d.set(xlabel="", ylabel="", title="Niche-context age programs")
    ax_d.tick_params(axis="x", rotation=30, labelsize=6.5)
    ax_d.text(.5, -.30, "Blue age-down   Red age-up", transform=ax_d.transAxes,
              ha="center", fontsize=6.6, color="#59696A")

    panel_label(ax_e, "E")
    path = cell_pathways.loc[cell_pathways["cell_type"].isin(["TEC", "Fibroblast", "DC"])].copy()
    path["label"] = path["pathway"].str.replace(r"\s+R-HSA-\d+$", "", regex=True).str.slice(0, 28)
    summary = (path.groupby("label").agg(n_contexts=("cell_type", "nunique"), n_overlap=("n_overlap_genes", "sum"))
               .sort_values(["n_contexts", "n_overlap"], ascending=False).head(12).sort_values("n_overlap"))
    ax_e.barh(np.arange(len(summary)), summary["n_overlap"], color="#4B8F77")
    ax_e.set(yticks=np.arange(len(summary)), yticklabels=summary.index,
             xlabel="Candidate-gene overlaps", title="Existing pathway-evidence overlap")
    ax_e.tick_params(axis="y", labelsize=6.6)
    clean_axis(ax_e)
    fig.suptitle("Aging signals extend across epithelial and stromal niche compartments", y=.995, fontsize=14, fontweight="bold")
    fig.text(.5, .008, "Pathway panel summarizes overlap with existing candidate-enrichment outputs; it is not de novo subtype enrichment.", ha="center", fontsize=7.2)
    save_figure(fig, output, "Figure3_epithelial_stromal_niche")


def figure4(effects: pd.DataFrame, cell_summary: pd.DataFrame, gene_summary: pd.DataFrame,
            output: Path) -> None:
    fig = plt.figure(figsize=(8.2, 9.2))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.05, 1.15, .92], hspace=.48, wspace=.45)
    ax_a = fig.add_subplot(gs[0, 0]); ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :]); ax_d = fig.add_subplot(gs[2, 0]); ax_e = fig.add_subplot(gs[2, 1])

    panel_label(ax_a, "A")
    whole = effects.loc[effects["cell_context"].eq("whole_thymus")].copy()
    whole["minus_log10_q"] = -np.log10(whole["adj.P.Val"].clip(lower=1e-300))
    whole["direction"] = np.where(whole["adj.P.Val"].lt(.05), np.where(whole["logFC_age_perSD"] > 0, "Age-up", "Age-down"), "Not significant")
    color = {"Age-up": "#B84A5A", "Age-down": "#355C8C", "Not significant": "#C9CECF"}
    for label in ["Not significant", "Age-down", "Age-up"]:
        frame = whole.loc[whole["direction"].eq(label)]
        ax_a.scatter(frame["logFC_age_perSD"], frame["minus_log10_q"], s=5 if label == "Not significant" else 8,
                     color=color[label], alpha=.55 if label == "Not significant" else .75, linewidth=0, rasterized=True, label=label)
    ax_a.axhline(-math.log10(.05), color="#666666", lw=.7, ls="--")
    ax_a.set(xlabel="Sex-adjusted age coefficient per SD", ylabel="-log10 BH q", title="Whole-library pseudobulk")
    ax_a.legend(frameon=False, fontsize=7); clean_axis(ax_a)

    panel_label(ax_b, "B")
    counts = cell_summary.set_index("cell_type")[["n_age_down_q05", "n_age_up_q05"]].sort_values("n_age_up_q05")
    y = np.arange(len(counts))
    ax_b.barh(y, -counts["n_age_down_q05"], color="#355C8C", label="Age-down")
    ax_b.barh(y, counts["n_age_up_q05"], color="#B84A5A", label="Age-up")
    ax_b.set(yticks=y, yticklabels=counts.index, xlabel="BH-significant gene-context rows", title="Context-specific associations")
    ax_b.axvline(0, color="#555555", lw=.7); ax_b.legend(frameon=False, fontsize=7); clean_axis(ax_b)

    ax_c.text(-.16, 1.06, "C", transform=ax_c.transAxes, fontsize=13, fontweight="bold", va="top")
    contexts = ["whole_thymus", "DN", "DP", "SP_CD4", "SP_CD8", "Treg", "TEC", "Fibroblast", "DC", "B"]
    matrix = gene_effect_matrix(effects, contexts, n_genes=24)
    sns.heatmap(matrix, ax=ax_c, cmap="RdBu_r", center=0, linewidths=.2, linecolor="white",
                cbar_kws={"label": "Age coefficient per SD", "shrink": .68})
    ax_c.set(xlabel="Cell context", ylabel="", title="Shared and context-restricted age-associated programs")
    ax_c.tick_params(axis="x", rotation=35)

    panel_label(ax_d, "D")
    evidence = pd.DataFrame({
        "Evidence layer": ["Whole-thymus q<0.05", "ML consensus", "Transformer top 100", "Mouse RNA concordant", "Mouse ATAC concordant"],
        "Genes": [
            int(gene_summary["whole_thymus_q_value"].lt(.05).sum()),
            int(gene_summary["in_ML_consensus"].fillna(False).astype(bool).sum()),
            int(gene_summary["in_DL_top100"].fillna(False).astype(bool).sum()),
            int(gene_summary["mouse_RNA_concordant"].fillna(False).astype(bool).sum()),
            int(gene_summary["mouse_ATAC_concordant"].fillna(False).astype(bool).sum()),
        ],
    }).sort_values("Genes")
    ax_d.barh(np.arange(len(evidence)), evidence["Genes"], color=["#5F8FB8", "#4B8F77", "#8866A5", "#D18A39", "#B84A5A"])
    ax_d.set(yticks=np.arange(len(evidence)), yticklabels=evidence["Evidence layer"], xlabel="Genes", title="Evidence-layer sizes")
    clean_axis(ax_d)

    panel_label(ax_e, "E")
    top = gene_summary.dropna(subset=["evidence_score"]).sort_values(["evidence_score", "n_evidence_layers"], ascending=False).head(15).sort_values("evidence_score")
    ax_e.barh(np.arange(len(top)), top["evidence_score"], color="#4B8F77")
    ax_e.set(yticks=np.arange(len(top)), yticklabels=top["gene"], xlabel="Integrated evidence score", title="Prioritized cross-layer candidates")
    clean_axis(ax_e)
    fig.suptitle("Donor-level aging programs are shared across and restricted to thymic contexts", y=.995, fontsize=14, fontweight="bold")
    fig.text(.5, .008, "Association and evidence-integration panels prioritize hypotheses; they do not establish causal molecular mechanisms.", ha="center", fontsize=7.2)
    save_figure(fig, output, "Figure4_cross_lineage_programs")


def figure5(data_dir: Path, gene_summary: pd.DataFrame, effects: pd.DataFrame,
            donors: pd.DataFrame, external_metrics: dict[str, object] | None,
            output: Path) -> None:
    fig = plt.figure(figsize=(8.2, 9.2))
    gs = fig.add_gridspec(3, 2, height_ratios=[.92, 1.1, 1.02], hspace=.5, wspace=.42)
    ax_a = fig.add_subplot(gs[0, :]); ax_b = fig.add_subplot(gs[1, 0]); ax_c = fig.add_subplot(gs[1, 1])
    ax_d = fig.add_subplot(gs[2, 0]); ax_e = fig.add_subplot(gs[2, 1])

    panel_label(ax_a, "A"); ax_a.axis("off")
    layers = [
        ("Cell atlas", "Levels 1-3\nUMAP and coverage", "#E8F1F8"),
        ("Donor evidence", "age sex QC\nlineage fractions", "#E4F2EE"),
        ("Molecular programs", "gene-context effects\nand pathways", "#F6ECEE"),
        ("Model diagnostics", "internal OOF and\nexternal boundary", "#F4EFE4"),
        ("Downloads", "tables figures code\nand checksums", "#EEEAF4"),
    ]
    for i, (title, subtitle, color) in enumerate(layers):
        x = .02 + i * .195
        draw_box(ax_a, (x, .3), .17, .42, f"{title}\n{subtitle}", color, 7.3)
        if i < 4:
            ax_a.annotate("", xy=(x + .195, .51), xytext=(x + .172, .51), arrowprops=dict(arrowstyle="->", lw=.9, color="#667777"))
    ax_a.set(xlim=(0, 1), ylim=(0, 1), title="Interactive resource architecture")

    panel_label(ax_b, "B")
    catalog = pd.DataFrame({
        "Object": ["Genes", "Gene-context rows", "Cell contexts", "Primary donors", "Level 2 classes"],
        "Count": [gene_summary["gene"].nunique(), len(effects), effects["cell_context"].nunique(), donors["donor_id"].nunique(), 19],
    }).sort_values("Count")
    ax_b.barh(np.arange(len(catalog)), catalog["Count"], color="#4B8F77")
    ax_b.set_xscale("log"); ax_b.set(yticks=np.arange(len(catalog)), yticklabels=catalog["Object"], xlabel="Records log scale", title="Atlas catalog")
    clean_axis(ax_b)

    ax_c.text(-.17, 1.06, "C", transform=ax_c.transAxes, fontsize=13,
              fontweight="bold", va="top")
    ax_c.axis("off")
    pages = ["Biological story", "Hierarchical cell atlas", "Thymocyte development", "TEC and stromal niche", "Aging programs", "Donor evidence", "Model diagnostics", "Methods and downloads"]
    for i, page in enumerate(pages):
        y = .91 - i * .105
        ax_c.text(.03, y, f"{i + 1:02d}", color="#0B756D", fontsize=8, fontweight="bold", va="center")
        ax_c.text(.13, y, page, fontsize=8, va="center")
        ax_c.plot([.13, .95], [y - .042, y - .042], color="#E1E7E6", lw=.7)
    ax_c.set_title("Website information architecture", loc="left", fontweight="bold")

    panel_label(ax_d, "D")
    ax_d.axis("off")
    rules = [
        ("Primary", "Level 1 donor pseudobulk", "formal age inference"),
        ("Exploratory", "Levels 2-3 fractions and states", "hypothesis generation"),
        ("Context", "developmental and mouse data", "localization or direction"),
        ("Boundary", "external frozen transfer", "technical feasibility only"),
    ]
    for i, (kind, source, claim) in enumerate(rules):
        y = .88 - i * .21
        draw_box(ax_d, (.02, y - .05), .18, .1, kind, "#E7EFED", 7.2)
        ax_d.text(.25, y + .022, source, fontsize=7.2, va="center")
        ax_d.text(.25, y - .034, claim, fontsize=6.8, color="#59696A", va="center")
    ax_d.set(xlim=(0, 1), ylim=(0, 1), title="Claim and evidence boundaries")

    panel_label(ax_e, "E")
    age = donors["chronological_age"].astype(float)
    pred = donors["elasticnet_OOF_predicted_age"].astype(float)
    ax_e.scatter(age, pred, color="#355C8C", s=35, edgecolor="white", linewidth=.6)
    limits = [min(age.min(), pred.min()) - 3, max(age.max(), pred.max()) + 3]
    ax_e.plot(limits, limits, color="#777777", ls="--", lw=.9)
    ax_e.set(xlim=limits, ylim=limits, xlabel="Chronological age", ylabel="Internal OOF prediction", title="Model retained as a diagnostic")
    clean_axis(ax_e)
    label = "Internal n=18; not a clinical clock"
    if external_metrics:
        label += f"\nExternal n={external_metrics.get('n_external_donors_predicted', 4)}; R2={external_metrics.get('R2', float('nan')):.2f}"
    ax_e.text(.03, .96, label, transform=ax_e.transAxes, va="top", fontsize=7.2)

    fig.suptitle("The atlas connects biological findings to an auditable interactive resource", y=.995, fontsize=14, fontweight="bold")
    fig.text(.5, .008, "Every plotted result is linked to a downloadable table and an explicit interpretation boundary.", ha="center", fontsize=7.2)
    save_figure(fig, output, "Figure5_interactive_resource")


def supplementary_figures(obs: pd.DataFrame, umap: np.ndarray, level3_audit: pd.DataFrame,
                          donors: pd.DataFrame, developmental: pd.DataFrame,
                          external_predictions: pd.DataFrame | None,
                          external_metrics: dict[str, object] | None,
                          output: Path, seed: int) -> None:
    # S1: donor and broad atlas diagnostics.
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 8.6))
    donor_counts = obs.groupby("donor_id").size().sort_values()
    axes[0, 0].barh(np.arange(len(donor_counts)), donor_counts, color="#5F8FB8")
    axes[0, 0].set(yticks=np.arange(len(donor_counts)), yticklabels=donor_counts.index, xlabel="QC-passed cells", title="Cells per donor")
    scatter_umap(axes[0, 1], umap, obs["broad_celltype"], {x: LEVEL2_COLORS.get(make_level2(pd.Series({"broad_celltype": x, "author_id_lv4": np.nan})), "#999999") for x in obs["broad_celltype"].unique()}, seed, 45000, 2, 1.0)
    axes[0, 1].set_title("Original broad labels")
    ctab = pd.crosstab(obs["donor_id"], obs["atlas_level1"])
    sns.heatmap(np.log10(ctab + 1), ax=axes[1, 0], cmap="Blues", cbar_kws={"label": "log10 cells + 1"})
    axes[1, 0].set(title="Level 1 donor coverage", xlabel="", ylabel="")
    missing = obs[["author_id_lv1", "author_id_lv2", "author_id_lv3", "author_id_lv4"]].isna().mean().mul(100)
    axes[1, 1].bar(missing.index, missing, color="#9A9A9A")
    axes[1, 1].set(ylabel="Missing labels percent", title="Source hierarchy completeness")
    axes[1, 1].tick_params(axis="x", rotation=30)
    for ax, label in zip(axes.ravel(), list("ABCD")): panel_label(ax, label)
    fig.suptitle("Supplementary Figure S1  Cohort and annotation diagnostics", y=.995, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, .97]); save_figure(fig, output, "Supplementary_Figure_S1_diagnostics")

    # S2: complete Level 3 audit.
    focus = level3_audit.sort_values("n_cells", ascending=False).head(55).sort_values("n_cells")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.2, 9.0), gridspec_kw={"width_ratios": [1.2, 1]})
    status_colors = {"eligible_for_donor_model": "#4B8F77", "exploratory_only": "#D3A24F", "descriptive_only": "#B5B5B5"}
    ax1.barh(np.arange(len(focus)), focus["n_cells"], color=focus["analysis_status"].map(status_colors))
    ax1.set_xscale("log"); ax1.set(yticks=np.arange(len(focus)), yticklabels=focus["atlas_level3"], xlabel="Cells log scale", title="Source Level 3 cell coverage")
    ax2.barh(np.arange(len(focus)), focus["n_donors"], color=focus["analysis_status"].map(status_colors))
    ax2.set(yticks=np.arange(len(focus)), yticklabels=[], xlim=(0, 18.5), xlabel="Donors detected", title="Donor coverage")
    ax2.axvline(12, color="#333333", ls="--", lw=.8)
    panel_label(ax1, "A"); panel_label(ax2, "B")
    fig.suptitle("Supplementary Figure S2  Fine-annotation coverage audit", y=.995, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, .97]); save_figure(fig, output, "Supplementary_Figure_S2_fine_annotation_audit")

    # S3: internal model diagnostics.
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.8))
    age = donors["chronological_age"].astype(float); pred = donors["elasticnet_OOF_predicted_age"].astype(float)
    axes[0].scatter(age, pred, c=age, cmap="viridis", s=42, edgecolor="white", linewidth=.5)
    limits = [min(age.min(), pred.min()) - 3, max(age.max(), pred.max()) + 3]
    axes[0].plot(limits, limits, ls="--", color="#666666"); axes[0].set(xlim=limits, ylim=limits, xlabel="Chronological age", ylabel="Elastic Net OOF age", title="Internal donor-level OOF")
    errors = pd.DataFrame({"Elastic Net": donors["absolute_error"].astype(float), "HumanThymusFormer": np.abs(donors["HumanThymusFormer_strict_OOF_predicted_age"].astype(float) - age)})
    plot = errors.melt(var_name="Model", value_name="Absolute error")
    sns.boxplot(data=plot, x="Model", y="Absolute error", ax=axes[1], color="#DCE7EF", width=.5)
    sns.stripplot(data=plot, x="Model", y="Absolute error", ax=axes[1], color="#355C8C", size=4, jitter=.15)
    axes[1].set(title="Prediction retained as secondary analysis", ylabel="Absolute error years", xlabel="")
    for ax, label in zip(axes, "AB"): panel_label(ax, label); clean_axis(ax)
    fig.suptitle("Supplementary Figure S3  Internal age-prediction diagnostics", y=.995, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, .94]); save_figure(fig, output, "Supplementary_Figure_S3_internal_prediction")

    # S4: external transfer and developmental context.
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 4.0))
    if external_predictions is not None and not external_predictions.empty:
        x = external_predictions["chronological_age"].astype(float)
        y = external_predictions["predicted_age"].astype(float)
        axes[0].scatter(x, y, s=52, color="#B84A5A", edgecolor="white", linewidth=.6)
        for _, row in external_predictions.iterrows():
            axes[0].text(float(row["chronological_age"]) + .7, float(row["predicted_age"]), str(row["donor_id"]), fontsize=7)
        limits = [min(x.min(), y.min()) - 4, max(x.max(), y.max()) + 4]
        axes[0].plot(limits, limits, ls="--", color="#666666"); axes[0].set(xlim=limits, ylim=limits)
        title = "Frozen external transfer"
        if external_metrics: title += f"\nn={external_metrics.get('n_external_donors_predicted', 4)}  R2={external_metrics.get('R2', float('nan')):.2f}"
        axes[0].set(xlabel="Chronological age", ylabel="Predicted age", title=title)
    else:
        axes[0].axis("off"); axes[0].text(.5, .5, "External prediction table not found", ha="center")
    dev = developmental.copy()
    genes = (dev.groupby("gene")["mean_zscore"].agg(lambda x: np.nanmax(x) - np.nanmin(x)).sort_values(ascending=False).head(16).index)
    mat = dev.loc[dev["gene"].isin(genes)].pivot_table(index="gene", columns="developmental_stage", values="mean_zscore")
    ordered_stages = [x for x in ["DN1", "DN2", "DN3", "ISP", "DP_CD3min", "DP_CD3plus"] if x in mat.columns]
    sns.heatmap(mat.reindex(index=genes, columns=ordered_stages), ax=axes[1], cmap="RdBu_r", center=0,
                cbar_kws={"label": "Within-object z score", "shrink": .75})
    axes[1].set(title="Pooled developmental localization", xlabel="", ylabel="")
    for ax, label in zip(axes, "AB"): panel_label(ax, label); clean_axis(ax)
    fig.suptitle("Supplementary Figure S4  External evidence and interpretation boundary", y=.995, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, .94]); save_figure(fig, output, "Supplementary_Figure_S4_external_context")


def manifest_path(path: Path, atlas_root: Path, project_root: Path) -> str:
    resolved = path.resolve()
    for prefix, root in (("atlas_root", atlas_root.resolve()), ("project_root", project_root.resolve())):
        try:
            return f"{prefix}/{resolved.relative_to(root).as_posix()}"
        except ValueError:
            continue
    return f"external/{resolved.name}"


def write_manifest(output: Path, source_files: dict[str, Path], generated_files: list[Path],
                   atlas_root: Path, project_root: Path) -> None:
    payload = {
        "release": "biology-atlas-v2-draft",
        "claim_boundary": "biology-centered draft; Level 2-3 fraction models are exploratory; no clinical age-clock claim",
        "source_files": {name: manifest_path(path, atlas_root, project_root) for name, path in source_files.items()},
        "generated_files": [manifest_path(path, atlas_root, project_root) for path in generated_files],
    }
    (output / "build_manifest.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    configure_style()
    project = args.project_root.resolve()
    atlas = args.atlas_root.resolve() if args.atlas_root else Path(__file__).resolve().parents[2]
    output = args.output if args.output.is_absolute() else atlas / args.output
    data_out = output / "data"; figure_out = output / "figures"
    data_out.mkdir(parents=True, exist_ok=True); figure_out.mkdir(parents=True, exist_ok=True)

    h5ad = project / "01_raw_processing" / "filtered" / "07_GSE231906_thymus_annotated.h5ad"
    release_data = atlas / "release" / "data"
    effects_path = release_data / "gene_context_effects.csv.gz"
    cell_summary_path = release_data / "cell_type_summary.csv"
    gene_summary_path = release_data / "gene_summary.csv.gz"
    cell_pathways_path = release_data / "cell_type_pathway_overlap.csv.gz"
    donor_path = release_data / "donor_summary.csv"
    developmental_path = release_data / "developmental_context.csv.gz"
    external_root = project / "09_external_human" / "HRA007984_frozen_transfer" / "results"
    external_pred_path = external_root / "hra007984_frozen_predictions.tsv"
    external_metrics_path = external_root / "hra007984_metrics.json"
    required = [h5ad, effects_path, cell_summary_path, gene_summary_path, cell_pathways_path, donor_path, developmental_path]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required inputs:\n" + "\n".join(missing))

    obs, umap = load_cell_metadata(h5ad)
    obs = add_hierarchy(obs)
    level2_audit, level3_audit = annotation_audit(obs)
    fractions = donor_level_fractions(obs)
    models = fit_fraction_models(fractions)
    effects = pd.read_csv(effects_path)
    cell_summary = pd.read_csv(cell_summary_path)
    gene_summary = pd.read_csv(gene_summary_path)
    cell_pathways = pd.read_csv(cell_pathways_path)
    donors = pd.read_csv(donor_path)
    developmental = pd.read_csv(developmental_path)
    external_predictions = pd.read_csv(external_pred_path, sep="\t") if external_pred_path.exists() else None
    external_metrics = json.loads(external_metrics_path.read_text(encoding="utf-8")) if external_metrics_path.exists() else None

    level2_audit.to_csv(data_out / "hierarchical_annotation_level2_audit.csv", index=False)
    level3_audit.to_csv(data_out / "hierarchical_annotation_level3_audit.csv", index=False)
    fractions.to_csv(data_out / "donor_level2_within_lineage_fractions.csv", index=False)
    models.to_csv(data_out / "level2_exploratory_age_models.csv", index=False)
    pd.DataFrame({"umap_1": umap[:, 0], "umap_2": umap[:, 1],
                  "donor_id": obs["donor_id"], "atlas_level1": obs["atlas_level1"],
                  "atlas_level2": obs["atlas_level2"], "atlas_level3": obs["atlas_level3"]}).to_csv(
                      data_out / "hierarchical_umap_source.csv.gz", index=False, compression="gzip")

    figure1(obs, umap, level2_audit, figure_out, args.seed)
    figure2(obs, umap, fractions, models, level3_audit, figure_out, args.seed + 10)
    figure3(obs, umap, fractions, models, effects, cell_pathways, figure_out, args.seed + 20)
    figure4(effects, cell_summary, gene_summary, figure_out)
    figure5(data_out, gene_summary, effects, donors, external_metrics, figure_out)
    supplementary_figures(obs, umap, level3_audit, donors, developmental,
                          external_predictions, external_metrics, figure_out, args.seed + 30)

    generated = sorted([*data_out.glob("*"), *figure_out.glob("*")])
    write_manifest(output, {
        "annotated_h5ad": h5ad,
        "gene_context_effects": effects_path,
        "cell_type_summary": cell_summary_path,
        "gene_summary": gene_summary_path,
        "cell_type_pathway_overlap": cell_pathways_path,
        "donor_summary": donor_path,
        "developmental_context": developmental_path,
        "external_predictions": external_pred_path,
        "external_metrics": external_metrics_path,
    }, generated, atlas, project)
    print(f"Wrote biology atlas v2 to {output}")


if __name__ == "__main__":
    main()

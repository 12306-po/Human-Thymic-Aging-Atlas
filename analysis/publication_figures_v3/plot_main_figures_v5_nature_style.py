#!/usr/bin/env python3
"""Publication-style rebuild of the five main manuscript figures.

Design goals
------------
1. Follow the visual grammar of high-impact human-thymus atlas papers:
   restrained schematic -> cohort overview -> atlas/marker panels -> one dominant
   quantitative panel per figure.
2. Keep the scientific claim boundaries of the manuscript:
   * GSE231906 is the 18-donor discovery cohort.
   * captured fractions are library fractions, not native-tissue abundance.
   * Figure 3 is internal nested OOF prediction, not an external age clock.
   * GSE195812 is a pooled developmental reference.
   * HRA007984 completed an independent frozen-model feasibility test reported separately and was not used to construct Figures 1-5.
3. Quantitative panels are source-table driven. Existing UMAP/marker panels are only
   composited as display images; no values are digitized from figures.

The command-line interface is intentionally compatible with plot_main_figures_v3.py.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib import gridspec
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Restrained publication palette
# -----------------------------------------------------------------------------
UP = "#C95858"          # age-up / positive effect
DOWN = "#3F78B5"        # age-down / negative effect
TEAL = "#2B8C82"        # focal model / primary result
PURPLE = "#7C6AA6"      # Sort2 developmental reference
INK = "#202124"
MID = "#6F7378"
LIGHT = "#D9DDE2"
PALE = "#F4F6F8"
PALE_BLUE = "#EEF3F7"
FEMALE = "#B86A9A"
MALE = "#326EA3"
MM = 1 / 25.4
WIDTH = 183 * MM  # common two-column journal width


def style() -> None:
    """Use the project-bundled Times New Roman for all generated text."""
    font_dir = Path(__file__).resolve().parents[3] / "resources" / "fonts"
    font_files = [font_dir / name for name in ("times.ttf", "timesbd.ttf", "timesi.ttf", "timesbi.ttf")]
    missing = [str(path) for path in font_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Times New Roman font files are missing: {missing}")
    for path in font_files:
        font_manager.fontManager.addfont(str(path))
    if font_manager.FontProperties(fname=str(font_files[0])).get_name() != "Times New Roman":
        raise RuntimeError("The bundled regular font is not Times New Roman")
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "mathtext.fontset": "stix",
        "font.size": 7.3,
        "axes.titlesize": 8.4,
        "axes.labelsize": 7.5,
        "xtick.labelsize": 6.8,
        "ytick.labelsize": 6.8,
        "legend.fontsize": 6.6,
        "axes.linewidth": 0.65,
        "xtick.major.width": 0.55,
        "ytick.major.width": 0.55,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    })


def clean(ax, grid: bool = False, grid_axis: str = "x") -> None:
    ax.spines[["top", "right"]].set_visible(False)
    if grid:
        ax.grid(axis=grid_axis, color="#E8EAED", lw=0.5)
        ax.set_axisbelow(True)


def letter(ax, text: str, x: float = -0.10, y: float = 1.04) -> None:
    ax.text(x, y, text, transform=ax.transAxes, fontsize=10.5,
            fontweight="bold", ha="left", va="bottom", color=INK)


def save(fig, out: Path, stem: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg", "png", "tif"):
        fig.savefig(
            out / f"{stem}.{ext}",
            dpi=600 if ext in {"png", "tif"} else None,
            bbox_inches="tight",
            pad_inches=0.025,
        )
    plt.close(fig)


def require(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Required source file is missing: {path}")
    return path


def panel_image(ax, path: Path, title: str) -> None:
    image = plt.imread(require(path))
    ax.imshow(image)
    ax.set_axis_off()
    ax.set_title(title, loc="left", pad=2, fontweight="bold")


def safe_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def pick_first_existing(columns, candidates):
    for c in candidates:
        if c in columns:
            return c
    return None


def donor_count_column(df: pd.DataFrame):
    return pick_first_existing(
        df.columns,
        ["n_donors", "donor_n", "n_donor", "n", "donors_tested", "n_tested_donors"],
    )


def draw_workflow(ax, labels, subtitle=None, future_note=None) -> None:
    """Minimal atlas-style horizontal workflow with one accent colour."""
    ax.set_axis_off()
    n = len(labels)
    xs = np.linspace(0.08, 0.92, n)
    box_w = min(0.128, 0.72 / max(n - 1, 1))
    for i, (xc, label) in enumerate(zip(xs, labels), start=1):
        left = xc - box_w / 2
        patch = FancyBboxPatch(
            (left, 0.36), box_w, 0.33,
            boxstyle="round,pad=0.010,rounding_size=0.018",
            fc=PALE_BLUE, ec="#A5B1BC", lw=0.65,
            transform=ax.transAxes,
        )
        ax.add_patch(patch)
        ax.text(xc, 0.57, str(i), ha="center", va="center", fontsize=6.7,
                color="white",
                bbox=dict(boxstyle="circle,pad=.20", fc="#58758D", ec="none"),
                transform=ax.transAxes)
        ax.text(xc, 0.445, label, ha="center", va="center", fontsize=6.6,
                linespacing=1.12, transform=ax.transAxes, color=INK)
        if i < n:
            ax.add_patch(FancyArrowPatch(
                (xc + box_w / 2 + 0.005, 0.525),
                (xs[i] - box_w / 2 - 0.005, 0.525),
                arrowstyle="-|>", mutation_scale=7, lw=0.75,
                color="#75838E", transform=ax.transAxes,
            ))
    if subtitle:
        ax.text(0.5, 0.19, subtitle, ha="center", va="center",
                color=MID, fontsize=6.25, transform=ax.transAxes)
    if future_note:
        ax.plot([0.74, 0.98], [0.08, 0.08], transform=ax.transAxes,
                color="#A8ADB2", lw=0.7, ls=(0, (2.5, 2.5)))
        ax.text(0.86, 0.015, future_note, ha="center", va="bottom",
                color=MID, fontsize=5.9, transform=ax.transAxes)


def draw_vertical_evidence_flow(ax, labels) -> None:
    """Compact vertical flow for Fig. 5 so the evidence matrix can dominate."""
    ax.set_axis_off()
    ys = np.linspace(0.91, 0.10, len(labels))
    for i, (y, label) in enumerate(zip(ys, labels)):
        ax.add_patch(FancyBboxPatch(
            (0.12, y - 0.055), 0.76, 0.105,
            boxstyle="round,pad=.012,rounding_size=.02",
            fc=PALE_BLUE if i < len(labels) - 1 else "#E8F3F1",
            ec="#AAB4BC" if i < len(labels) - 1 else "#74A69F",
            lw=0.65, transform=ax.transAxes,
        ))
        ax.text(0.50, y, label, ha="center", va="center", fontsize=6.6,
                transform=ax.transAxes)
        if i < len(labels) - 1:
            ax.add_patch(FancyArrowPatch(
                (0.50, y - 0.06), (0.50, ys[i + 1] + 0.06),
                arrowstyle="-|>", mutation_scale=7, lw=0.7,
                color="#7C8993", transform=ax.transAxes,
            ))


# -----------------------------------------------------------------------------
# Figure 1: cohort -> atlas, borrowing the clean visual hierarchy of atlas papers
# -----------------------------------------------------------------------------
def figure1(donor: pd.DataFrame, umap: Path, marker: Path, out: Path):
    donor = donor.copy()
    donor["age_years"] = safe_numeric(donor["age_years"])
    donor = donor.sort_values(["age_years", "donor_id"]).reset_index(drop=True)

    fig = plt.figure(figsize=(WIDTH, 172 * MM))
    outer = gridspec.GridSpec(
        3, 5, figure=fig,
        height_ratios=[0.66, 0.78, 1.58],
        hspace=0.42, wspace=0.34,
        left=0.065, right=0.985, top=0.985, bottom=0.065,
    )

    ax = fig.add_subplot(outer[0, :])
    letter(ax, "A", x=-0.025, y=1.01)

    draw_workflow(
        ax,
        [
            "Human thymus\nstudy inputs",
            "Donor-resolved\nanalysis unit",
            "Predeclared QC\n& feature rules",
            "Leakage-controlled\ninference",
            "Claim-bounded\nevidence synthesis",
            "Versioned Atlas\n& provenance",
        ],
        subtitle=(
            "Governance flow only; donor is the statistical unit and "
            "quantitative results are shown in Figures 2–5."
        ),
        future_note=None,
    )

    ax = fig.add_subplot(outer[1, :3]); letter(ax, "B")
    ax.axhline(0, color=LIGHT, lw=1.0, zorder=0)
    levels = (0.12, -0.12, 0.21, -0.21)
    for i, row in donor.iterrows():
        is_f = str(row.sex).lower().startswith("f")
        color = FEMALE if is_f else MALE
        shape = "o" if is_f else "s"
        ax.scatter(row.age_years, 0, s=34, marker=shape, color=color,
                   edgecolor="white", lw=0.6, zorder=3)
        ax.text(row.age_years, levels[i % len(levels)],
                str(row.donor_id).replace("donor", "d"),
                ha="center", va="center", fontsize=5.4)
    ax.set_yticks([])
    ax.set_ylim(-0.28, 0.28)
    ax.set_xlim(max(0, donor.age_years.min() - 3), donor.age_years.max() + 4)
    ax.set_xlabel("Chronological age (years)")
    ax.set_title("Donor-resolved continuous-age cohort", loc="left", fontweight="bold")
    clean(ax, grid=True)
    ax.scatter([], [], color=FEMALE, marker="o", label="Female")
    ax.scatter([], [], color=MALE, marker="s", label="Male")
    ax.legend(frameon=False, ncol=2, loc="upper right", bbox_to_anchor=(1, 1.12), handletextpad=.35)

    ax = fig.add_subplot(outer[1, 3:]); letter(ax, "C", x=-0.16)
    y = np.arange(len(donor))
    pre = safe_numeric(donor["total_cells_preQC"])
    post = safe_numeric(donor["total_cells_postQC"])
    ax.hlines(y, post, pre, color=LIGHT, lw=1.15)
    ax.scatter(pre, y, s=17, facecolor="white", edgecolor=MID, lw=.7, label="Before QC")
    ax.scatter(post, y, s=21, color=TEAL, edgecolor="white", lw=.35, label="After QC", zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(donor.donor_id.str.replace("donor", "d"), fontsize=5.6)
    ax.invert_yaxis()
    ax.set_xlabel("Cells per donor")
    ax.set_title("Cell recovery", loc="left", fontweight="bold")
    clean(ax, grid=True)
    ax.legend(frameon=False, ncol=2, loc="upper right", bbox_to_anchor=(1.0, 1.12), handletextpad=.35)

    ax = fig.add_subplot(outer[2, :3]); letter(ax, "D")
    panel_image(ax, umap, "Annotated human thymus atlas")

    ax = fig.add_subplot(outer[2, 3:]); letter(ax, "E", x=-0.15)
    panel_image(ax, marker, "Canonical-marker support")

    save(fig, out, "Figure1_nature_style")


# -----------------------------------------------------------------------------
# Figure 2: one dominant donor heatmap + supporting quantitative panels
# -----------------------------------------------------------------------------
def figure2(atlas_genes: pd.DataFrame, cell_summary: pd.DataFrame,
            composition: pd.DataFrame, heatmap: pd.DataFrame,
            donor: pd.DataFrame, hallmark: pd.DataFrame, out: Path):
    wt = atlas_genes[atlas_genes.cell_context.eq("whole_thymus")].copy()
    wt["logFC_age_perSD"] = safe_numeric(wt["logFC_age_perSD"])
    wt["adj.P.Val"] = safe_numeric(wt["adj.P.Val"])
    wt["logq"] = -np.log10(wt["adj.P.Val"].clip(lower=np.finfo(float).tiny))
    wt["sig"] = wt["adj.P.Val"] < 0.05
    wt["color"] = np.where(~wt.sig, LIGHT, np.where(wt.logFC_age_perSD > 0, UP, DOWN))

    fig = plt.figure(figsize=(WIDTH, 214 * MM))
    outer = gridspec.GridSpec(
        3, 2, figure=fig,
        height_ratios=[0.95, 1.65, 1.05],
        hspace=0.44, wspace=0.38,
        left=0.085, right=0.975, top=0.978, bottom=0.068,
    )

    # A. Volcano
    ax = fig.add_subplot(outer[0, 0]); letter(ax, "A")
    ax.scatter(wt.logFC_age_perSD, wt.logq, s=5, c=wt.color,
               alpha=0.70, linewidth=0, rasterized=True)
    ax.axvline(0, color=MID, lw=.55)
    ax.axhline(-np.log10(.05), color=MID, lw=.55, ls=(0, (3, 2)))
    wanted = ["FOXN1", "LMNB1", "IGFBP5", "AIRE", "PSMB11", "B2M", "STAT1", "JUN"]
    for _, row in wt[wt.gene.isin(wanted)].iterrows():
        if pd.notna(row.logFC_age_perSD) and pd.notna(row.logq):
            ax.annotate(row.gene, (row.logFC_age_perSD, row.logq),
                        xytext=(3, 2), textcoords="offset points", fontsize=5.8)
    n_up = int(((wt.sig) & (wt.logFC_age_perSD > 0)).sum())
    n_down = int(((wt.sig) & (wt.logFC_age_perSD < 0)).sum())
    ax.text(.02, .98, f"Age-down  {n_down:,}\nAge-up     {n_up:,}", transform=ax.transAxes,
            ha="left", va="top", fontsize=6.3, color=MID)
    ax.set_xlabel("Age coefficient (log2 CPM per SD age)")
    ax.set_ylabel("−log10(BH q)")
    ax.set_title("Mixed-library age associations", loc="left", fontweight="bold")
    clean(ax)

    # B. Diverging lollipop with donor coverage
    ax = fig.add_subplot(outer[0, 1]); letter(ax, "B")
    cs = cell_summary.copy()
    for c in ["n_age_up_q05", "n_age_down_q05"]:
        cs[c] = safe_numeric(cs[c]).fillna(0)
    cs["total_sig"] = cs.n_age_up_q05 + cs.n_age_down_q05
    cs = cs.sort_values("total_sig", ascending=True)
    y = np.arange(len(cs))
    ax.hlines(y, -cs.n_age_down_q05, cs.n_age_up_q05, color=LIGHT, lw=1)
    ax.scatter(-cs.n_age_down_q05, y, color=DOWN, s=24, label="Age-down")
    ax.scatter(cs.n_age_up_q05, y, color=UP, s=24, label="Age-up")
    ax.axvline(0, color=INK, lw=.6)
    ax.set_yticks(y); ax.set_yticklabels(cs.cell_type, fontsize=6.1)
    ax.set_xlabel("BH q<0.05 genes  (left: down; right: up)")
    ax.set_title("Cell-context transcriptional remodeling", loc="left", fontweight="bold")
    ncol = donor_count_column(cs)
    if ncol:
        xmax = max(float(cs.n_age_up_q05.max()), float(cs.n_age_down_q05.max()))
        for yi, (_, row) in enumerate(cs.iterrows()):
            if pd.notna(row[ncol]):
                ax.text(xmax * 1.04, yi, f"n={int(row[ncol])}", ha="left", va="center",
                        fontsize=5.4, color=MID, clip_on=False)
    clean(ax, grid=True)
    ax.legend(frameon=False, ncol=2, loc="lower right", handletextpad=.35)

    # C. Dominant heatmap with age/sex annotation strips
    donor_meta = donor.copy()
    donor_meta["age_years"] = safe_numeric(donor_meta["age_years"])
    donor_meta = donor_meta.set_index("donor_id")
    valid_cols = [c for c in heatmap.columns if c in donor_meta.index]
    if valid_cols:
        valid_cols = sorted(valid_cols, key=lambda d: donor_meta.loc[d, "age_years"])
        heatmap = heatmap.loc[:, valid_cols]
    nested = gridspec.GridSpecFromSubplotSpec(
        3, 1, subplot_spec=outer[1, :], height_ratios=[0.08, 0.08, 1.0], hspace=0.04
    )
    ax_age = fig.add_subplot(nested[0, 0])
    ages = donor_meta.loc[heatmap.columns, "age_years"].to_numpy(float)
    age_norm = Normalize(vmin=np.nanmin(ages), vmax=np.nanmax(ages))
    ax_age.imshow(ages.reshape(1, -1), aspect="auto", cmap="viridis", norm=age_norm)
    ax_age.set_yticks([0]); ax_age.set_yticklabels(["Age"], fontsize=6.0)
    ax_age.set_xticks([])
    for s in ax_age.spines.values(): s.set_visible(False)

    ax_sex = fig.add_subplot(nested[1, 0])
    sex_vals = donor_meta.loc[heatmap.columns, "sex"].astype(str)
    sex_rgba = np.array([
        mpl.colors.to_rgba(FEMALE if s.lower().startswith("f") else MALE) for s in sex_vals
    ]).reshape(1, -1, 4)
    ax_sex.imshow(sex_rgba, aspect="auto")
    ax_sex.set_yticks([0]); ax_sex.set_yticklabels(["Sex"], fontsize=6.0)
    ax_sex.set_xticks([])
    for s in ax_sex.spines.values(): s.set_visible(False)

    ax = fig.add_subplot(nested[2, 0]); letter(ax, "C", x=-0.065, y=1.18)
    values = heatmap.to_numpy(float)
    vmax = max(2.0, np.nanpercentile(np.abs(values), 98))
    im = ax.imshow(values, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_yticks(np.arange(len(heatmap)))
    ax.set_yticklabels(heatmap.index, fontsize=6.0)
    ax.set_xticks(np.arange(len(heatmap.columns)))
    ax.set_xticklabels([
        f"{d.replace('donor','d')}\n{donor_meta.loc[d,'age_years']:g}y"
        for d in heatmap.columns
    ], rotation=90, fontsize=5.5)
    ax.set_title("Representative age-associated genes across age-ordered donors",
                 loc="left", fontweight="bold", pad=4)
    cb = fig.colorbar(im, ax=ax, fraction=.014, pad=.008)
    cb.set_label("Row z-score")
    cb.outline.set_linewidth(.5)
    for s in ax.spines.values(): s.set_visible(False)

    # D. CLR forest
    ax = fig.add_subplot(outer[2, 0]); letter(ax, "D")
    c = composition.copy()
    for col in ["estimate", "ci_low_95", "ci_high_95", "q_value_BH"]:
        c[col] = safe_numeric(c[col])
    c = c.sort_values("estimate")
    y = np.arange(len(c))
    sig = c.q_value_BH < .05
    point_colors = np.where(sig, np.where(c.estimate > 0, UP, DOWN), "white")
    edge_colors = np.where(sig, np.where(c.estimate > 0, UP, DOWN), MID)
    ax.hlines(y, c.ci_low_95, c.ci_high_95, color=LIGHT, lw=1.15)
    ax.scatter(c.estimate, y, c=point_colors, edgecolor=edge_colors, s=27, lw=.75, zorder=3)
    ax.axvline(0, color=INK, lw=.65)
    ax.set_yticks(y); ax.set_yticklabels(c.cell_type, fontsize=6.1)
    ax.set_xlabel("Sex-adjusted CLR age effect per SD (95% CI)")
    ax.set_title("Captured-library composition", loc="left", fontweight="bold")
    clean(ax, grid=True)
    ax.text(.99, .02, "filled: BH q<0.05", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=5.6, color=MID)

    # E. Hallmark bubble
    ax = fig.add_subplot(outer[2, 1]); letter(ax, "E")
    h = hallmark.copy()
    h["GeneRatio"] = safe_numeric(h["GeneRatio"])
    h["Count"] = safe_numeric(h["Count"])
    h["q"] = safe_numeric(h["q"])
    h = h.dropna(subset=["GeneRatio", "Count", "q"]).sort_values("q").head(12)
    h = h.sort_values("GeneRatio")
    denom = max(float(h.Count.max()), 1.0)
    sizes = 18 + 105 * (h.Count / denom)
    sc = ax.scatter(h.GeneRatio, np.arange(len(h)), s=sizes,
                    c=-np.log10(h.q.clip(lower=np.finfo(float).tiny)),
                    cmap="Blues", edgecolor="white", lw=.4)
    ax.set_yticks(np.arange(len(h)))
    ax.set_yticklabels(h.term.astype(str).str.replace("HALLMARK_", "", regex=False)
                       .str.replace("_", " ").str.title(), fontsize=5.9)
    ax.set_xlabel("Gene ratio")
    ax.set_title("Hallmark enrichment", loc="left", fontweight="bold")
    cb = fig.colorbar(sc, ax=ax, fraction=.04, pad=.02)
    cb.set_label("−log10(BH q)")
    cb.outline.set_linewidth(.5)
    clean(ax, grid=True)

    save(fig, out, "Figure2_nature_style")


# -----------------------------------------------------------------------------
# Figure 3: method schematic + three compact, directly interpretable diagnostics
# -----------------------------------------------------------------------------
def figure3(donor: pd.DataFrame, out: Path):
    d = donor.copy()
    d["age_years"] = safe_numeric(d["age_years"])
    d["elasticnet_OOF_predicted_age"] = safe_numeric(d["elasticnet_OOF_predicted_age"])
    d = d.sort_values("age_years").reset_index(drop=True)
    y = d.age_years.to_numpy(float)
    p = d.elasticnet_OOF_predicted_age.to_numpy(float)
    null = np.asarray([(np.nansum(y) - v) / (np.isfinite(y).sum() - 1) for v in y])

    fig = plt.figure(figsize=(WIDTH, 140 * MM))
    outer = gridspec.GridSpec(
        2, 12, figure=fig,
        height_ratios=[0.65, 1.0],
        hspace=0.42, wspace=0.65,
        left=0.075, right=0.985, top=0.975, bottom=0.085,
    )

    ax = fig.add_subplot(outer[0, :]); letter(ax, "A")
    draw_workflow(
        ax,
        ["Outer LODO\n1 test donor", "Inner 3-fold CV\non 17 donors",
         "Filtering &\nscreening", "Imputation, scaling\n& tuning",
         "Refit on 17\ntraining donors", "Predict held-out\ndonor once"],
        "Every preprocessing, feature-selection and tuning step is repeated inside training donors only; pool 18 OOF predictions at the end.",
    )

    # B predicted vs actual, largest panel
    ax = fig.add_subplot(outer[1, 0:5]); letter(ax, "B")
    lo = np.nanmin([y.min(), p.min()]) - 3
    hi = np.nanmax([y.max(), p.max()]) + 3
    ax.plot([lo, hi], [lo, hi], color=MID, ls=(0, (3, 2)), lw=.8)
    ax.scatter(y, p, color=TEAL, s=31, edgecolor="white", lw=.55, zorder=3)
    for _, row in d.iterrows():
        ax.annotate(str(row.donor_id).replace("donor", "d"),
                    (row.age_years, row.elasticnet_OOF_predicted_age),
                    xytext=(2.5, 2.0), textcoords="offset points", fontsize=5.2)
    r2 = 1 - np.nansum((y - p) ** 2) / np.nansum((y - np.nanmean(y)) ** 2)
    mae = np.nanmean(np.abs(y - p))
    rho = pd.Series(y).corr(pd.Series(p), method="spearman")
    ax.text(.03, .96, f"R² = {r2:.3f}\nMAE = {mae:.2f} y\nρ = {rho:.3f}",
            transform=ax.transAxes, ha="left", va="top", fontsize=6.4,
            bbox=dict(boxstyle="round,pad=.25", fc="white", ec=LIGHT, lw=.5))
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("Chronological age (years)")
    ax.set_ylabel("OOF predicted age (years)")
    ax.set_title("Final held-out predictions", loc="left", fontweight="bold")
    clean(ax)

    # C paired dumbbell
    ax = fig.add_subplot(outer[1, 5:9]); letter(ax, "C", x=-0.14)
    en_err = np.abs(y - p)
    null_err = np.abs(y - null)
    improvement = null_err - en_err
    order = np.argsort(improvement)
    yy = np.arange(len(y))
    for j, idx in enumerate(order):
        ax.plot([en_err[idx], null_err[idx]], [j, j], color=LIGHT, lw=1.0)
        ax.scatter(en_err[idx], j, color=TEAL, s=19, zorder=3)
        ax.scatter(null_err[idx], j, facecolor="white", edgecolor=MID, s=19, lw=.7, zorder=3)
    ax.set_yticks(yy)
    ax.set_yticklabels(d.iloc[order].donor_id.str.replace("donor", "d"), fontsize=5.2)
    ax.set_xlabel("Absolute OOF error (years)")
    ax.set_title("Elastic Net vs training-mean null", loc="left", fontweight="bold")
    clean(ax, grid=True)
    ax.scatter([], [], color=TEAL, label="Elastic Net")
    ax.scatter([], [], facecolor="white", edgecolor=MID, label="Null")
    ax.legend(frameon=False, ncol=2, loc="lower right", handletextpad=.3)

    # D fixed-pair influence
    ax = fig.add_subplot(outer[1, 9:12]); letter(ax, "D", x=-0.18)
    influence = []
    for i in range(len(y)):
        keep = np.arange(len(y)) != i
        r = 1 - np.nansum((y[keep] - p[keep]) ** 2) / np.nansum((y[keep] - np.nanmean(y[keep])) ** 2)
        influence.append(r - r2)
    order = np.argsort(influence)
    vals = np.asarray(influence)[order]
    yy = np.arange(len(vals))
    colors = np.where(vals >= 0, UP, DOWN)
    ax.hlines(yy, 0, vals, color=colors, lw=1.1)
    ax.scatter(vals, yy, c=colors, s=18)
    ax.axvline(0, color=INK, lw=.65)
    ax.set_yticks(yy)
    ax.set_yticklabels(d.iloc[order].donor_id.str.replace("donor", "d"), fontsize=5.1)
    ax.set_xlabel("Δ pooled OOF R²")
    ax.set_title("Fixed-pair influence", loc="left", fontweight="bold")
    clean(ax, grid=True)
    ax.text(.98, .02, "models not refitted", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=5.4, color=MID)

    save(fig, out, "Figure3_nature_style")


# -----------------------------------------------------------------------------
# Figure 4: state-level bubble map rather than donor-style plots (pooled reference)
# -----------------------------------------------------------------------------
def figure4(dot: pd.DataFrame, coverage: pd.DataFrame, delta: pd.DataFrame, out: Path):
    state_col = "developmental_stage" if "developmental_stage" in dot.columns else "state"
    dot = dot.copy()
    dot["mean_zscore"] = safe_numeric(dot["mean_zscore"])
    dot["pct_expr"] = safe_numeric(dot["pct_expr"])

    gene_span = (dot.groupby("gene")["mean_zscore"]
                 .apply(lambda x: np.nanmax(x) - np.nanmin(x) if np.isfinite(x).any() else np.nan)
                 .dropna().sort_values(ascending=False))
    genes = gene_span.head(20).index.tolist()
    states = ["DN1", "DN2", "DN3", "ISP", "DP_CD3min", "DP_CD3plus"]
    sub = dot[dot.gene.isin(genes) & dot[state_col].isin(states)].copy()

    fig = plt.figure(figsize=(WIDTH, 151 * MM))
    outer = gridspec.GridSpec(
        2, 5, figure=fig,
        height_ratios=[0.48, 1.52],
        width_ratios=[1, 1, 1, .82, .82],
        hspace=0.39, wspace=0.50,
        left=0.10, right=0.975, top=0.975, bottom=0.085,
    )

    # A coverage
    ax = fig.add_subplot(outer[0, :]); letter(ax, "A")
    cov = coverage.copy()
    cov["n_cells_state"] = safe_numeric(cov["n_cells_state"])
    cov["label"] = cov.source_object.astype(str) + " · " + cov.developmental_stage.astype(str).str.replace("_", " ")
    colors = np.where(cov.source_object.eq("Sort1"), DOWN, PURPLE)
    yy = np.arange(len(cov))[::-1]
    ax.barh(yy, cov.n_cells_state, color=colors, height=.60, alpha=.90)
    xmax = max(float(cov.n_cells_state.max()), 1)
    for y0, (_, row) in zip(yy, cov.iterrows()):
        ax.text(row.n_cells_state + xmax * .015, y0,
                f"{int(row.n_cells_state):,} cells",
                va="center", fontsize=5.9)
    ax.set_yticks(yy); ax.set_yticklabels(cov.label, fontsize=6.1)
    ax.set_xlabel("Cells in pooled developmental state")
    ax.set_title("GSE195812 pooled developmental-reference coverage", loc="left", fontweight="bold")
    clean(ax, grid=True)
    ax.text(.99, .04, "Each state contains cells pooled from six healthy donors; not six donor-level observations.",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=5.6, color=MID)

    # B large bubble heatmap
    ax = fig.add_subplot(outer[1, 0:3]); letter(ax, "B")
    finite_z = sub.mean_zscore.to_numpy(float)
    zmax = np.nanpercentile(np.abs(finite_z[np.isfinite(finite_z)]), 95) if np.isfinite(finite_z).any() else 1.0
    zmax = max(float(zmax), 0.5)
    norm = TwoSlopeNorm(vcenter=0, vmin=-zmax, vmax=zmax)
    pct_max = np.nanmax(sub.pct_expr.to_numpy(float)) if len(sub) else 100
    pct_scale = 100.0 if pct_max > 1.5 else 1.0

    for i, gene in enumerate(genes[::-1]):
        for j, state in enumerate(states):
            row = sub[(sub.gene == gene) & (sub[state_col] == state)]
            if row.empty or pd.isna(row.iloc[0].mean_zscore):
                ax.text(j, i, "×", ha="center", va="center", color="#B5B8BB", fontsize=6.6)
            else:
                r = row.iloc[0]
                frac = np.clip(float(r.pct_expr) / pct_scale, 0, 1) if pd.notna(r.pct_expr) else 0
                ax.scatter(j, i, s=10 + 82 * frac, c=[float(r.mean_zscore)],
                           cmap="RdBu_r", norm=norm, edgecolor="white", lw=.45)
    ax.axvline(2.5, color="#8D9398", lw=.7, ls=(0, (3, 2)))
    ax.set_xticks(range(len(states)))
    ax.set_xticklabels([s.replace("DP_", "DP\n").replace("_", " ") for s in states], fontsize=6.0)
    ax.set_yticks(range(len(genes)))
    ax.set_yticklabels(genes[::-1], fontsize=5.9)
    ax.text(1, len(genes)+.15, "Sort1", ha="center", fontsize=6.8, fontweight="bold")
    ax.text(4, len(genes)+.15, "Sort2", ha="center", fontsize=6.8, fontweight="bold")
    ax.set_xlim(-.5, 5.5); ax.set_ylim(-.7, len(genes)+.45)
    ax.set_title("Developmental localization", loc="left", fontweight="bold")
    ax.tick_params(length=0)
    for s in ax.spines.values(): s.set_visible(False)

    # bubble size legend
    legend_fracs = [0.25, 0.50, 0.75]
    handles = [ax.scatter([], [], s=10 + 82*f, facecolor="#BFC7CE", edgecolor="white") for f in legend_fracs]
    ax.legend(handles, [f"{int(f*100)}%" for f in legend_fracs], title="Expressing",
              frameon=False, loc="lower right", bbox_to_anchor=(1.00, -0.02),
              borderaxespad=0, handletextpad=.2, labelspacing=.3, title_fontsize=6.1)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap="RdBu_r")
    cb = fig.colorbar(sm, ax=ax, fraction=.026, pad=.015)
    cb.set_label("Within-object mean z-score")
    cb.outline.set_linewidth(.5)

    # C contrasts
    ax = fig.add_subplot(outer[1, 3:5]); letter(ax, "C", x=-0.16)
    delta = delta.copy()
    delta["delta_z_late_minus_early"] = safe_numeric(delta["delta_z_late_minus_early"])
    blocks = []
    for obj in ["Sort1", "Sort2"]:
        z = delta[delta.source_object.eq(obj)].dropna(subset=["delta_z_late_minus_early"]).copy()
        z = z.reindex(z.delta_z_late_minus_early.abs().sort_values(ascending=False).index).head(7)
        blocks.append((obj, z))

    labels, vals, block_names = [], [], []
    for obj, z in blocks:
        for _, r in z.iterrows():
            labels.append(str(r.gene)); vals.append(float(r.delta_z_late_minus_early)); block_names.append(obj)
        labels.append(""); vals.append(np.nan); block_names.append(obj)
    if labels:
        labels, vals, block_names = labels[:-1], vals[:-1], block_names[:-1]
    yy = np.arange(len(labels))
    vals_arr = np.asarray(vals, dtype=float)
    mask = np.isfinite(vals_arr)
    colors = np.where(vals_arr[mask] >= 0, UP, DOWN)
    ax.barh(yy[mask], vals_arr[mask], color=colors, height=.62)
    ax.axvline(0, color=INK, lw=.65)
    ax.set_yticks(yy); ax.set_yticklabels(labels, fontsize=5.7)
    ax.invert_yaxis()
    ax.set_xlabel("Late − early mean z-score")
    ax.set_title("Within-object contrasts", loc="left", fontweight="bold")
    clean(ax, grid=True)
    # subtle object labels on the right
    for obj in ["Sort1", "Sort2"]:
        idx = [i for i, b in enumerate(block_names) if b == obj and labels[i] != ""]
        if idx:
            ax.text(.98, np.mean(idx), obj, transform=ax.get_yaxis_transform(),
                    ha="right", va="center", fontsize=6.2, fontweight="bold", color=MID)

    save(fig, out, "Figure4_nature_style")


# -----------------------------------------------------------------------------
# Figure 5: evidence matrix is the visual centre, not a website screenshot
# -----------------------------------------------------------------------------
def state_text(value):
    if pd.isna(value):
        return "NA"
    if isinstance(value, (bool, np.bool_)):
        return "●" if value else "○"
    text = str(value)
    if text.lower() in {"true", "1.0", "1"}:
        return "●"
    if text.lower() in {"false", "0.0", "0"}:
        return "○"
    return text


def q_col_for_beta(beta_col: str) -> str | None:
    mapping = {
        "whole_thymus_beta_age": "whole_thymus_q_value",
        "tec_beta_age": "tec_q_value",
        "fibroblast_beta_age": "fibroblast_q_value",
        "dp_beta_age": "dp_q_value",
        "sp_cd4_beta_age": "sp_cd4_q_value",
        "sp_cd8_beta_age": "sp_cd8_q_value",
    }
    return mapping.get(beta_col)


def figure5(gene_summary: pd.DataFrame, atlas_genes: pd.DataFrame,
            composition: pd.DataFrame, out: Path):
    genes = ["FOXN1", "LMNB1", "AIRE", "PSMB11", "IGFBP5", "B2M", "STAT1", "JUN"]
    data = gene_summary.set_index("gene").reindex(genes)

    fig = plt.figure(figsize=(WIDTH, 178 * MM))
    outer = gridspec.GridSpec(
        2, 6, figure=fig,
        height_ratios=[1.35, 1.0],
        width_ratios=[1.0, 1.0, 1.2, 1.2, 1.2, 1.2],
        hspace=0.48, wspace=0.55,
        left=0.08, right=0.975, top=0.975, bottom=0.09,
    )

    # A compact evidence architecture
    ax = fig.add_subplot(outer[0, 0:2]); letter(ax, "A", x=-0.16)
    draw_vertical_evidence_flow(ax, [
        "Human donor-level\nage effects",
        "Nested ML\nselection",
        "HumanThymusFormer\nattribution",
        "Pooled human\ndevelopment",
        "Mouse directional\ncontext",
        "Versioned Atlas\nrecord",
    ])
    ax.set_title("Evidence architecture", loc="left", fontweight="bold", pad=2)

    # B dominant evidence matrix
    ax = fig.add_subplot(outer[0, 2:6]); letter(ax, "B", x=-0.10)
    beta_cols = [
        "whole_thymus_beta_age", "tec_beta_age", "fibroblast_beta_age",
        "dp_beta_age", "sp_cd4_beta_age", "sp_cd8_beta_age",
    ]
    beta_labels = ["Mixed", "TEC", "Fibroblast", "DP", "SP CD4", "SP CD8"]
    extra_cols = ["in_ML_consensus", "in_DL_top100", "developmental_top_stage",
                  "mouse_RNA_concordant", "mouse_ATAC_concordant"]
    extra_labels = ["ML", "HTF top100", "Development", "Mouse RNA", "Mouse ATAC"]

    for c in beta_cols:
        if c not in data.columns:
            data[c] = np.nan
    matrix = data[beta_cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    finite = matrix[np.isfinite(matrix)]
    lim = max(float(np.nanmax(np.abs(finite))) if finite.size else 1.0, 0.5)
    norm = TwoSlopeNorm(vcenter=0, vmin=-lim, vmax=lim)
    ax.imshow(matrix, aspect="auto", cmap="RdBu_r", norm=norm, interpolation="nearest")

    # neutral tiles for categorical evidence
    for i, g in enumerate(genes):
        row = data.loc[g]
        for j, col in enumerate(extra_cols, start=len(beta_cols)):
            ax.add_patch(Rectangle((j-.5, i-.5), 1, 1, fc="#F2F4F6", ec="white", lw=.9))
            val = row[col] if col in row.index else np.nan
            txt = state_text(val)
            if col == "developmental_top_stage" and txt not in {"NA", "●", "○"}:
                color = INK
                fontsize = 5.5
            else:
                color = TEAL if txt == "●" else MID
                fontsize = 6.1
            ax.text(j, i, txt, ha="center", va="center", fontsize=fontsize, color=color)

    # overlay significance dots on human beta columns when q values exist
    for j, bcol in enumerate(beta_cols):
        qcol = q_col_for_beta(bcol)
        if qcol and qcol in data.columns:
            qvals = pd.to_numeric(data[qcol], errors="coerce")
            for i, q in enumerate(qvals):
                if pd.notna(q) and q < .05 and pd.notna(matrix[i, j]):
                    ax.scatter(j + .31, i - .31, s=6, color=INK, edgecolor="white", lw=.2, zorder=5)

    ax.set_yticks(range(len(genes)))
    ax.set_yticklabels(genes, fontweight="bold", fontsize=6.3)
    ax.set_xticks(range(len(beta_cols) + len(extra_cols)))
    ax.set_xticklabels(beta_labels + extra_labels, rotation=38, ha="right", fontsize=5.8)
    ax.set_xlim(-.5, len(beta_cols) + len(extra_cols) - .5)
    ax.set_title("Multi-layer evidence matrix", loc="left", fontweight="bold")
    ax.tick_params(length=0)
    for s in ax.spines.values(): s.set_visible(False)
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap="RdBu_r"), ax=ax,
                      fraction=.025, pad=.015)
    cb.set_label("Human age coefficient")
    cb.outline.set_linewidth(.5)
    ax.text(1.0, -0.23, "● selected/concordant   ○ tested-not-selected   NA unavailable   • corner dot: q<0.05",
            transform=ax.transAxes, ha="right", va="top", fontsize=5.4, color=MID)

    # C FOXN1 focused evidence strip
    ax = fig.add_subplot(outer[1, 0:2]); letter(ax, "C", x=-0.16)
    if "FOXN1" in data.index:
        fox = data.loc["FOXN1"]
        contexts = ["Mixed library", "TEC"]
        vals = [pd.to_numeric(fox.get("whole_thymus_beta_age"), errors="coerce"),
                pd.to_numeric(fox.get("tec_beta_age"), errors="coerce")]
        qs = [pd.to_numeric(fox.get("whole_thymus_q_value"), errors="coerce"),
              pd.to_numeric(fox.get("tec_q_value"), errors="coerce")]
        yy = np.arange(2)[::-1]
        for y0, v, q in zip(yy, vals, qs):
            if pd.isna(v):
                continue
            color = DOWN if v < 0 else UP
            ax.hlines(y0, 0, v, color=color, lw=1.35)
            ax.scatter(v, y0, color=color, s=31, zorder=3)
            qtxt = "NA" if pd.isna(q) else f"{q:.3g}"
            dx = 0.045 * max(1.0, abs(v))
            ax.text(v + (dx if v >= 0 else -dx), y0,
                    f"β={v:.2f}\nq={qtxt}",
                    ha="left" if v >= 0 else "right", va="center", fontsize=5.9)
        ax.axvline(0, color=INK, lw=.65)
        ax.set_yticks(yy); ax.set_yticklabels(contexts, fontsize=6.2)
        ax.set_xlabel("Age coefficient per SD age")
        ax.set_title("FOXN1: concordant human age-down signal", loc="left", fontweight="bold")
        clean(ax, grid=True)
        ax.text(.99, .03, "mixed library ≠ intact whole thymus", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=5.3, color=MID)
    else:
        ax.text(.5, .5, "FOXN1 not available", ha="center", va="center", color=MID)
        ax.set_axis_off()

    # D TEC top q-ranked effects, wide enough to be readable
    ax = fig.add_subplot(outer[1, 2:6]); letter(ax, "D", x=-0.10)
    tec = atlas_genes[atlas_genes.cell_context.eq("TEC")].copy()
    tec["adj.P.Val"] = safe_numeric(tec["adj.P.Val"])
    tec["logFC_age_perSD"] = safe_numeric(tec["logFC_age_perSD"])
    tec = tec.dropna(subset=["adj.P.Val", "logFC_age_perSD"])
    tec = tec[tec["adj.P.Val"] < .05]
    down = tec[tec.logFC_age_perSD < 0].nsmallest(5, "adj.P.Val")
    up = tec[tec.logFC_age_perSD > 0].nsmallest(5, "adj.P.Val")
    top = pd.concat([down, up], ignore_index=True).drop_duplicates("gene")
    top = top.sort_values("logFC_age_perSD")
    yy = np.arange(len(top))
    colors = np.where(top.logFC_age_perSD > 0, UP, DOWN)
    ax.hlines(yy, 0, top.logFC_age_perSD, color=colors, lw=1.15)
    ax.scatter(top.logFC_age_perSD, yy, c=colors, s=25)
    ax.axvline(0, color=INK, lw=.65)
    ax.set_yticks(yy); ax.set_yticklabels(top.gene, fontsize=6.0)
    ax.set_xlabel("TEC age coefficient per SD age")
    ax.set_title("TEC age-associated genes (q-ranked within direction)", loc="left", fontweight="bold")
    clean(ax, grid=True)
    for y0, (_, row) in zip(yy, top.iterrows()):
        ax.text(row.logFC_age_perSD, y0 + .20, f"q={row['adj.P.Val']:.2g}",
                ha="center", va="bottom", fontsize=4.9, color=MID)

    fig.text(.52, .018,
             "Atlas → frozen CSV/H5AD → source tables → scripts → SHA-256 manifest",
             ha="center", fontsize=6.4, color=MID)
    save(fig, out, "Figure5_nature_style")


# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True, help="Folder containing 10_results")
    ap.add_argument("--atlas-release", type=Path, required=True)
    ap.add_argument("--audit-dir", type=Path, required=True)
    ap.add_argument("--umap-panel", type=Path, required=True)
    ap.add_argument("--marker-panel", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    style()

    donor = pd.read_csv(require(args.atlas_release / "donor_summary.csv"))
    cell_summary = pd.read_csv(require(args.atlas_release / "cell_type_summary.csv"))
    gene_summary = pd.read_csv(require(args.atlas_release / "gene_summary.csv.gz"))
    atlas_genes = pd.read_csv(
        require(args.root / "10_results" / "human_thymic_aging_atlas" / "atlas_gene_age_effects.tsv.gz"),
        sep="\t",
    )
    composition = pd.read_csv(
        require(args.root / "10_results" / "human_thymic_aging_atlas" / "atlas_composition_age_effects.tsv"),
        sep="\t",
    )
    heatmap = pd.read_csv(require(args.audit_dir / "fig2d_heatmap_row_z_source.tsv"), sep="\t", index_col=0)
    hallmark = pd.read_csv(require(args.audit_dir / "fig2f_hallmark_enrichment_source.tsv"), sep="\t")
    dot = pd.read_csv(require(args.audit_dir / "fig4_dotplot_source.tsv"), sep="\t")
    coverage = pd.read_csv(require(args.audit_dir / "fig4_state_coverage_audit.tsv"), sep="\t")
    delta = pd.read_csv(require(args.audit_dir / "fig4_stage_delta_audit.tsv"), sep="\t")

    figure1(donor, args.umap_panel, args.marker_panel, args.output)
    figure2(atlas_genes, cell_summary, composition, heatmap, donor, hallmark, args.output)
    figure3(donor, args.output)
    figure4(dot, coverage, delta, args.output)
    figure5(gene_summary, atlas_genes, composition, args.output)

    contract = {
        "figures": [f"Figure{i}_nature_style" for i in range(1, 6)],
        "source_values_digitized_from_images": False,
        "figure1_display_panels": [str(args.umap_panel), str(args.marker_panel)],
        "claim_boundaries": {
            "primary_unit": "donor",
            "composition": "captured-library fractions, not native-tissue abundance",
            "prediction": "internal nested OOF proof of concept, not a validated biological-age clock",
            "development": "GSE195812 pooled state-level localization",
            "hra007984": "completed independent frozen-model feasibility test; reported separately and not used to construct these figures",
        },
        "note": "UMAP/marker panels may be rasterized; all quantitative panels are source-table driven.",
        "font_family": "Times New Roman",
        "raster_panel_font_note": "Figure 1D/1E PNG text is not searchable in the PDF; use the Times New Roman assets generated by plot_figure1DE_from_h5ad.py.",
    }
    (args.output / "figure_build_contract.json").write_text(
        json.dumps(contract, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()

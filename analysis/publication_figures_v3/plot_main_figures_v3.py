#!/usr/bin/env python3
"""Rebuild the five manuscript figures in a restrained atlas-paper style."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import pandas as pd
from scipy import stats

UP = "#C64E4E"
DOWN = "#3E73B8"
TEAL = "#2A8C82"
INK = "#222222"
MID = "#7A7A7A"
LIGHT = "#D9DDE2"
PALE = "#F3F5F7"
FEMALE = "#B86A9A"
MALE = "#326EA3"
MM = 1 / 25.4
WIDTH = 183 * MM


def style() -> None:
    mpl.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman"],
        "font.size": 7.5, "axes.titlesize": 8.2, "axes.labelsize": 7.7,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.8,
        "axes.linewidth": .65, "pdf.fonttype": 42, "ps.fonttype": 42,
        "svg.fonttype": "none", "savefig.facecolor": "white",
    })


def clean(ax, grid=False):
    ax.spines[["top", "right"]].set_visible(False)
    if grid:
        ax.grid(axis="x", color="#E7E9EC", lw=.55)
        ax.set_axisbelow(True)


def letter(ax, text):
    ax.text(-.12, 1.04, text, transform=ax.transAxes, fontsize=11,
            fontweight="bold", ha="left", va="bottom")


def save(fig, out: Path, stem: str):
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg", "png", "tif"):
        fig.savefig(out / f"{stem}.{ext}", dpi=600 if ext in {"png", "tif"} else None,
                    bbox_inches="tight", pad_inches=.03)
    plt.close(fig)


def require(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Required source file is missing: {path}")
    return path


def panel_image(ax, path: Path, title: str):
    image = plt.imread(require(path))
    ax.imshow(image)
    ax.set_axis_off()
    ax.set_title(title, loc="left", pad=2)


def draw_workflow(ax, labels, subtitle=None):
    ax.set_axis_off()
    xs = np.linspace(.02, .84, len(labels))
    for i, (x, label) in enumerate(zip(xs, labels), start=1):
        patch = FancyBboxPatch((x, .32), .135, .36,
                               boxstyle="round,pad=0.012,rounding_size=.025",
                               fc="#EDF2F6", ec="#758A9B", lw=.75,
                               transform=ax.transAxes)
        ax.add_patch(patch)
        ax.text(x + .0675, .52, f"{i}", ha="center", va="center", fontsize=7,
                color="white", bbox=dict(boxstyle="circle,pad=.22", fc="#4D6E87", ec="none"),
                transform=ax.transAxes)
        ax.text(x + .0675, .40, label, ha="center", va="center", fontsize=7.1,
                transform=ax.transAxes)
        if i < len(labels):
            ax.add_patch(FancyArrowPatch((x + .138, .50), (xs[i] - .005, .50),
                                         arrowstyle="-|>", mutation_scale=8,
                                         lw=.8, color="#687A88", transform=ax.transAxes))
    if subtitle:
        ax.text(.5, .10, subtitle, ha="center", va="center", color=MID,
                fontsize=6.6, transform=ax.transAxes)


def figure1(donor: pd.DataFrame, umap: Path, marker: Path, out: Path):
    donor = donor.sort_values(["age_years", "donor_id"]).reset_index(drop=True)
    fig = plt.figure(figsize=(WIDTH, 178 * MM))
    gs = gridspec.GridSpec(3, 2, figure=fig, height_ratios=[.62, .8, 1.45],
                           hspace=.48, wspace=.28, left=.07, right=.98, top=.98, bottom=.07)
    ax = fig.add_subplot(gs[0, :]); letter(ax, "A")
    draw_workflow(ax, ["GSE231906\n18 donors", "scRNA QC\nand annotation",
                       "Donor-level\naggregation", "Continuous-age\nassociation",
                       "Nested age\nprediction", "Evidence integration\nand Atlas"],
                  "Primary inference remains donor-level; contextual datasets are not model validation")

    ax = fig.add_subplot(gs[1, 0]); letter(ax, "B")
    for _, row in donor.iterrows():
        color = FEMALE if str(row.sex).lower().startswith("f") else MALE
        marker_shape = "o" if color == FEMALE else "s"
        ax.scatter(row.age_years, 0, s=42, marker=marker_shape, color=color,
                   edgecolor="white", lw=.6, zorder=3)
        levels = (.10, -.10, .18, -.18)
        ax.text(row.age_years, levels[int(row.name) % len(levels)],
                str(row.donor_id).replace("donor", "d"), ha="center", va="center", fontsize=5.7)
    ax.set_yticks([]); ax.set_ylim(-.25, .25); ax.set_xlim(1, 72)
    ax.set_xlabel("Chronological age (years)"); ax.set_title("Donor timeline", loc="left")
    clean(ax, grid=True)
    ax.scatter([], [], c=FEMALE, marker="o", label="Female")
    ax.scatter([], [], c=MALE, marker="s", label="Male")
    ax.legend(frameon=False, ncol=2, loc="upper right", bbox_to_anchor=(1, 1.12))

    ax = fig.add_subplot(gs[1, 1]); letter(ax, "C")
    y = np.arange(len(donor))
    ax.hlines(y, donor.total_cells_postQC, donor.total_cells_preQC, color=LIGHT, lw=1.2)
    ax.scatter(donor.total_cells_preQC, y, s=20, facecolor="white", edgecolor=MID, label="Before QC")
    ax.scatter(donor.total_cells_postQC, y, s=24, color=TEAL, label="After QC", zorder=3)
    ax.set_yticks(y); ax.set_yticklabels(donor.donor_id.str.replace("donor", "d"))
    ax.invert_yaxis(); ax.set_xlabel("Cells per donor"); ax.set_title("Cell recovery", loc="left")
    clean(ax, grid=True); ax.legend(frameon=False, ncol=2, loc="upper right", bbox_to_anchor=(1, 1.12))

    ax = fig.add_subplot(gs[2, 0]); letter(ax, "D")
    panel_image(ax, umap, "Annotated human thymus atlas")
    ax = fig.add_subplot(gs[2, 1]); letter(ax, "E")
    panel_image(ax, marker, "Canonical marker support")
    save(fig, out, "Figure1_restyled")


def figure2(atlas_genes: pd.DataFrame, cell_summary: pd.DataFrame,
            composition: pd.DataFrame, heatmap: pd.DataFrame,
            donor: pd.DataFrame, hallmark: pd.DataFrame, out: Path):
    wt = atlas_genes[atlas_genes.cell_context.eq("whole_thymus")].copy()
    wt["logq"] = -np.log10(wt["adj.P.Val"].clip(lower=np.finfo(float).tiny))
    wt["sig"] = wt["adj.P.Val"] < .05
    wt["color"] = np.where(~wt.sig, LIGHT, np.where(wt.logFC_age_perSD > 0, UP, DOWN))
    fig = plt.figure(figsize=(WIDTH, 210 * MM))
    gs = gridspec.GridSpec(3, 2, figure=fig, height_ratios=[1.0, 1.0, 1.18],
                           hspace=.48, wspace=.38, left=.085, right=.97, top=.97, bottom=.07)

    ax = fig.add_subplot(gs[0, 0]); letter(ax, "A")
    ax.scatter(wt.logFC_age_perSD, wt.logq, s=5, c=wt.color, alpha=.72, linewidth=0, rasterized=True)
    ax.axvline(0, color=MID, lw=.55); ax.axhline(-np.log10(.05), color=MID, lw=.55, ls="--")
    wanted = ["FOXN1", "LMNB1", "IGFBP5", "AIRE", "PSMB11", "B2M", "STAT1", "JUN"]
    for _, row in wt[wt.gene.isin(wanted)].iterrows():
        ax.annotate(row.gene, (row.logFC_age_perSD, row.logq), xytext=(3, 2),
                    textcoords="offset points", fontsize=6)
    ax.set_xlabel("Age coefficient (log2 CPM per SD age)"); ax.set_ylabel("−log10(BH q)")
    ax.set_title("Mixed-library age effects", loc="left"); clean(ax)

    ax = fig.add_subplot(gs[0, 1]); letter(ax, "B")
    cs = cell_summary.copy().sort_values("n_age_up_q05", ascending=True)
    y = np.arange(len(cs))
    ax.hlines(y, -cs.n_age_down_q05, cs.n_age_up_q05, color=LIGHT, lw=1)
    ax.scatter(-cs.n_age_down_q05, y, color=DOWN, s=27, label="Age-down")
    ax.scatter(cs.n_age_up_q05, y, color=UP, s=27, label="Age-up")
    ax.axvline(0, color=INK, lw=.6)
    ax.set_yticks(y); ax.set_yticklabels(cs.cell_type)
    ax.set_xlabel("BH q<0.05 genes (left: down; right: up)")
    ax.set_title("Cell-context transcriptional remodeling", loc="left")
    clean(ax, grid=True); ax.legend(frameon=False, ncol=2, loc="lower right")

    ax = fig.add_subplot(gs[1, :]); letter(ax, "C")
    ages = donor.set_index("donor_id").loc[heatmap.columns, "age_years"]
    im = ax.imshow(heatmap.to_numpy(float), aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2)
    ax.set_yticks(np.arange(len(heatmap))); ax.set_yticklabels(heatmap.index, fontsize=6.3)
    ax.set_xticks(np.arange(len(heatmap.columns)))
    ax.set_xticklabels([f"{d.replace('donor','d')}\n{ages[d]:g}y" for d in heatmap.columns],
                       rotation=90, fontsize=5.8)
    ax.set_title("Representative age-associated genes across age-ordered donors", loc="left")
    cb = fig.colorbar(im, ax=ax, fraction=.018, pad=.01); cb.set_label("Row z-score")

    ax = fig.add_subplot(gs[2, 0]); letter(ax, "D")
    c = composition.sort_values("estimate")
    y = np.arange(len(c))
    sig = c.q_value_BH < .05
    colors = np.where(sig, np.where(c.estimate > 0, UP, DOWN), MID)
    ax.hlines(y, c.ci_low_95, c.ci_high_95, color=LIGHT, lw=1.2)
    ax.scatter(c.estimate, y, c=colors, s=28, edgecolor="white", lw=.4, zorder=3)
    ax.axvline(0, color=INK, lw=.65)
    ax.set_yticks(y); ax.set_yticklabels(c.cell_type)
    ax.set_xlabel("Sex-adjusted CLR age effect per SD (95% CI)")
    ax.set_title("Captured-library composition", loc="left"); clean(ax, grid=True)

    ax = fig.add_subplot(gs[2, 1]); letter(ax, "E")
    h = hallmark.sort_values("GeneRatio").tail(12)
    sizes = 18 + 100 * (h.Count / h.Count.max())
    sc = ax.scatter(h.GeneRatio, np.arange(len(h)), s=sizes,
                    c=-np.log10(h.q), cmap="Blues", edgecolor=MID, lw=.4)
    ax.set_yticks(np.arange(len(h))); ax.set_yticklabels(h.term.str.title(), fontsize=6.2)
    ax.set_xlabel("Gene ratio"); ax.set_title("Hallmark enrichment", loc="left")
    cb = fig.colorbar(sc, ax=ax, fraction=.04, pad=.02); cb.set_label("−log10(BH q)")
    clean(ax, grid=True)
    save(fig, out, "Figure2_restyled")


def figure3(donor: pd.DataFrame, out: Path):
    d = donor.sort_values("age_years").copy()
    y = d.age_years.to_numpy(float); p = d.elasticnet_OOF_predicted_age.to_numpy(float)
    null = np.asarray([(y.sum() - v) / (len(y) - 1) for v in y])
    fig = plt.figure(figsize=(WIDTH, 142 * MM))
    gs = gridspec.GridSpec(2, 3, figure=fig, height_ratios=[.62, 1.0],
                           hspace=.40, wspace=.38, left=.08, right=.98, top=.97, bottom=.09)
    ax = fig.add_subplot(gs[0, :]); letter(ax, "A")
    draw_workflow(ax, ["Outer LODO\n1 test donor", "Inner 3-fold CV\non 17 donors",
                       "Filtering, screening,\nimputation, scaling, tuning",
                       "Refit on 17\ntraining donors", "Predict held-out\ndonor once",
                       "Pool 18 OOF\npredictions"],
                  "No held-out donor enters feature selection, preprocessing, or tuning")

    ax = fig.add_subplot(gs[1, 0]); letter(ax, "B")
    lo, hi = min(y.min(), p.min()) - 3, max(y.max(), p.max()) + 3
    ax.plot([lo, hi], [lo, hi], color=MID, ls="--", lw=.8)
    ax.scatter(y, p, color=TEAL, s=30, edgecolor="white", lw=.5)
    for _, row in d.iterrows():
        ax.annotate(row.donor_id.replace("donor", "d"),
                    (row.age_years, row.elasticnet_OOF_predicted_age),
                    xytext=(2, 2), textcoords="offset points", fontsize=5.5)
    r2 = 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)
    mae = np.mean(np.abs(y - p))
    ax.text(.03, .96, f"R²={r2:.3f}\nMAE={mae:.2f} y", transform=ax.transAxes,
            ha="left", va="top", fontsize=7)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("Chronological age (years)"); ax.set_ylabel("OOF predicted age (years)")
    ax.set_title("Final 18 held-out predictions", loc="left"); clean(ax)

    ax = fig.add_subplot(gs[1, 1]); letter(ax, "C")
    en_err = np.abs(y - p); null_err = np.abs(y - null)
    order = np.argsort(null_err - en_err)
    yy = np.arange(len(y))
    for j, idx in enumerate(order):
        ax.plot([en_err[idx], null_err[idx]], [j, j], color=LIGHT, lw=1)
        ax.scatter(en_err[idx], j, color=TEAL, s=22, zorder=3)
        ax.scatter(null_err[idx], j, facecolor="white", edgecolor=MID, s=22, zorder=3)
    ax.set_yticks(yy); ax.set_yticklabels(d.iloc[order].donor_id.str.replace("donor", "d"), fontsize=5.8)
    ax.set_xlabel("Absolute OOF error (years)")
    ax.set_title("Elastic Net vs training-mean null", loc="left"); clean(ax, grid=True)
    ax.scatter([], [], color=TEAL, label="Elastic Net"); ax.scatter([], [], facecolor="white", edgecolor=MID, label="Null")
    ax.legend(frameon=False, ncol=2, loc="lower right")

    ax = fig.add_subplot(gs[1, 2]); letter(ax, "D")
    full_r2 = r2
    influence = []
    for i in range(len(y)):
        keep = np.arange(len(y)) != i
        r = 1 - np.sum((y[keep] - p[keep]) ** 2) / np.sum((y[keep] - y[keep].mean()) ** 2)
        influence.append(r - full_r2)
    order = np.argsort(influence)
    vals = np.asarray(influence)[order]
    yy = np.arange(len(vals))
    ax.hlines(yy, 0, vals, color=np.where(vals >= 0, UP, DOWN), lw=1.2)
    ax.scatter(vals, yy, c=np.where(vals >= 0, UP, DOWN), s=22)
    ax.axvline(0, color=INK, lw=.65)
    ax.set_yticks(yy); ax.set_yticklabels(d.iloc[order].donor_id.str.replace("donor", "d"), fontsize=5.8)
    ax.set_xlabel("Change in pooled OOF R² after exclusion")
    ax.set_title("Fixed-pair influence diagnostic", loc="left"); clean(ax, grid=True)
    save(fig, out, "Figure3_restyled")


def figure4(dot: pd.DataFrame, coverage: pd.DataFrame, delta: pd.DataFrame, out: Path):
    state_col = "developmental_stage" if "developmental_stage" in dot.columns else "state"
    genes = (dot.groupby("gene")["mean_zscore"].apply(lambda x: np.nanmax(x)-np.nanmin(x))
             .sort_values(ascending=False).head(20).index.tolist())
    states = ["DN1", "DN2", "DN3", "ISP", "DP_CD3min", "DP_CD3plus"]
    sub = dot[dot.gene.isin(genes) & dot[state_col].isin(states)].copy()
    fig = plt.figure(figsize=(WIDTH, 150 * MM))
    gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[.45, 1.35],
                           hspace=.42, wspace=.34, left=.11, right=.96, top=.97, bottom=.09)
    ax = fig.add_subplot(gs[0, :]); letter(ax, "A")
    cov = coverage.copy(); cov["label"] = cov.source_object + " · " + cov.developmental_stage.str.replace("_", " ")
    colors = np.where(cov.source_object.eq("Sort1"), DOWN, "#8A6FA8")
    yy = np.arange(len(cov))[::-1]
    ax.barh(yy, cov.n_cells_state, color=colors, height=.62)
    for y0, (_, row) in zip(yy, cov.iterrows()):
        ax.text(row.n_cells_state + 70, y0, f"{int(row.n_cells_state):,} cells; pooled 6 donors", va="center", fontsize=6.5)
    ax.set_yticks(yy); ax.set_yticklabels(cov.label)
    ax.set_xlabel("Cells in developmental state")
    ax.set_title("Pooled human developmental-reference coverage", loc="left"); clean(ax, grid=True)

    ax = fig.add_subplot(gs[1, 0]); letter(ax, "B")
    for i, gene in enumerate(genes[::-1]):
        for j, state in enumerate(states):
            row = sub[(sub.gene == gene) & (sub[state_col] == state)]
            if row.empty or pd.isna(row.iloc[0].mean_zscore):
                ax.text(j, i, "×", ha="center", va="center", color="#B0B0B0", fontsize=7)
            else:
                r = row.iloc[0]
                ax.scatter(j, i, s=8 + 58 * np.clip(r.pct_expr / 100, 0, 1),
                           c=[r.mean_zscore], cmap="RdBu_r", norm=TwoSlopeNorm(0, vmin=-.6, vmax=.6),
                           edgecolor=MID, lw=.35)
    ax.axvline(2.5, color=INK, lw=.7)
    ax.set_xticks(range(len(states))); ax.set_xticklabels([s.replace("_", "\n") for s in states])
    ax.set_yticks(range(len(genes))); ax.set_yticklabels(genes[::-1], fontsize=6.2)
    ax.text(1, len(genes)+.25, "Sort1", ha="center", fontsize=7, fontweight="bold")
    ax.text(4, len(genes)+.25, "Sort2", ha="center", fontsize=7, fontweight="bold")
    ax.set_xlim(-.5, 5.5); ax.set_ylim(-.7, len(genes)+.6)
    ax.set_title("Developmental localization (colour: within-object z; size: % expressing)", loc="left")

    ax = fig.add_subplot(gs[1, 1]); letter(ax, "C")
    for k, obj in enumerate(["Sort1", "Sort2"]):
        z = delta[delta.source_object.eq(obj)].copy()
        z = z.reindex(z.delta_z_late_minus_early.abs().sort_values(ascending=False).index).head(8)
        offset = (0 if k == 0 else 10)
        yy = np.arange(len(z)) + offset
        ax.barh(yy, z.delta_z_late_minus_early,
                color=np.where(z.delta_z_late_minus_early >= 0, UP, DOWN), height=.62)
        ax.set_yticks(list(ax.get_yticks()) + yy.tolist())
        ax.text(.98, offset + 7.6, obj, transform=ax.get_yaxis_transform(),
                ha="right", va="top", fontsize=7, fontweight="bold")
        for y0, gene in zip(yy, z.gene):
            ax.text(-.02, y0, gene, transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=6.2)
    ax.axvline(0, color=INK, lw=.7); ax.set_yticks([])
    ax.set_xlabel("Late minus early mean z-score")
    ax.set_title("Within-object developmental contrasts", loc="left"); clean(ax, grid=True)
    save(fig, out, "Figure4_restyled")


def state_text(value):
    if pd.isna(value): return "NA"
    if isinstance(value, (bool, np.bool_)): return "●" if value else "○"
    text = str(value)
    if text.lower() in {"true", "1.0"}: return "●"
    if text.lower() in {"false", "0.0"}: return "○"
    return text


def figure5(gene_summary: pd.DataFrame, atlas_genes: pd.DataFrame,
            composition: pd.DataFrame, out: Path):
    genes = ["FOXN1", "LMNB1", "AIRE", "PSMB11", "IGFBP5", "B2M", "STAT1", "JUN"]
    data = gene_summary.set_index("gene").reindex(genes)
    fig = plt.figure(figsize=(WIDTH, 185 * MM))
    gs = gridspec.GridSpec(3, 2, figure=fig, height_ratios=[.55, 1.35, 1.0],
                           hspace=.50, wspace=.38, left=.09, right=.97, top=.97, bottom=.07)
    ax = fig.add_subplot(gs[0, :]); letter(ax, "A")
    draw_workflow(ax, ["Human mixed-library\nand cell-context βage", "Nested ML\nselection",
                       "HumanThymusFormer\nattribution", "Pooled human\ndevelopment",
                       "Mouse cross-dataset\ndirectional support", "Versioned Atlas\nrecord"],
                  "Unavailable evidence remains NA; absence is not converted to negative support")

    ax = fig.add_subplot(gs[1, :]); letter(ax, "B")
    cols = ["whole_thymus_beta_age", "tec_beta_age", "fibroblast_beta_age", "dp_beta_age",
            "sp_cd4_beta_age", "sp_cd8_beta_age"]
    labels = ["Mixed", "TEC", "Fibroblast", "DP", "SP CD4", "SP CD8"]
    matrix = data[cols].to_numpy(float)
    norm = TwoSlopeNorm(0, vmin=np.nanmin(matrix), vmax=np.nanmax(matrix))
    ax.imshow(matrix, aspect="auto", cmap="RdBu_r", norm=norm)
    ax.set_yticks(range(len(genes))); ax.set_yticklabels(genes, fontweight="bold")
    ax.set_xticks(range(len(cols)+5));
    extra = ["ML", "HTF top100", "Development", "Mouse RNA", "Mouse ATAC"]
    ax.set_xticklabels(labels + extra, rotation=35, ha="right")
    ax.set_xlim(-.5, len(cols)+4.5)
    for i, g in enumerate(genes):
        row = data.loc[g]
        states = [row.in_ML_consensus, row.in_DL_top100, row.developmental_top_stage,
                  row.mouse_RNA_concordant, row.mouse_ATAC_concordant]
        for j, val in enumerate(states, start=len(cols)):
            ax.add_patch(plt.Rectangle((j-.5, i-.5), 1, 1, fc="#F2F3F5", ec="white", lw=.8))
            txt = state_text(val)
            ax.text(j, i, txt, ha="center", va="center", fontsize=6.3,
                    color=TEAL if txt == "●" else MID)
    ax.set_title("Multi-layer evidence matrix", loc="left")
    ax.tick_params(length=0)

    ax = fig.add_subplot(gs[2, 0]); letter(ax, "C")
    fox = data.loc["FOXN1"]
    contexts = ["Mixed library", "TEC"]
    vals = [fox.whole_thymus_beta_age, fox.tec_beta_age]
    qs = [fox.whole_thymus_q_value, fox.tec_q_value]
    yy = np.arange(2)[::-1]
    ax.hlines(yy, 0, vals, color=DOWN, lw=1.4)
    ax.scatter(vals, yy, color=DOWN, s=35, zorder=3)
    for y0, v, q in zip(yy, vals, qs):
        dy = -.11 if y0 > .5 else .11
        va = "top" if y0 > .5 else "bottom"
        ax.text(v+.05, y0+dy, f"β={v:.2f}; q={q:.3g}", ha="left", va=va, fontsize=6.5)
    ax.axvline(0, color=INK, lw=.7); ax.set_yticks(yy); ax.set_yticklabels(contexts)
    ax.set_xlabel("Age coefficient per SD age"); ax.set_title("FOXN1 evidence strip", loc="left")
    clean(ax, grid=True)

    ax = fig.add_subplot(gs[2, 1]); letter(ax, "D")
    tec = atlas_genes[(atlas_genes.cell_context == "TEC") & (atlas_genes["adj.P.Val"] < .05)].copy()
    tec = pd.concat([tec.nsmallest(6, "logFC_age_perSD"), tec.nlargest(6, "logFC_age_perSD")]).drop_duplicates("gene")
    tec = tec.sort_values("logFC_age_perSD")
    yy = np.arange(len(tec))
    ax.hlines(yy, 0, tec.logFC_age_perSD, color=np.where(tec.logFC_age_perSD > 0, UP, DOWN), lw=1.2)
    ax.scatter(tec.logFC_age_perSD, yy, c=np.where(tec.logFC_age_perSD > 0, UP, DOWN), s=27)
    ax.axvline(0, color=INK, lw=.7); ax.set_yticks(yy); ax.set_yticklabels(tec.gene, fontsize=6.2)
    ax.set_xlabel("TEC age coefficient per SD age"); ax.set_title("TEC top age-effect genes", loc="left")
    clean(ax, grid=True)
    fig.text(.5, .015, "Atlas → frozen CSV/H5AD → source tables → scripts → SHA-256 manifest",
             ha="center", fontsize=7, color=MID)
    save(fig, out, "Figure5_restyled")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True, help="Folder containing 10_results")
    ap.add_argument("--atlas-release", type=Path, required=True)
    ap.add_argument("--audit-dir", type=Path, required=True)
    ap.add_argument("--umap-panel", type=Path, required=True)
    ap.add_argument("--marker-panel", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args(); style()

    donor = pd.read_csv(require(args.atlas_release / "donor_summary.csv"))
    cell_summary = pd.read_csv(require(args.atlas_release / "cell_type_summary.csv"))
    gene_summary = pd.read_csv(require(args.atlas_release / "gene_summary.csv.gz"))
    atlas_genes = pd.read_csv(require(args.root / "10_results" / "human_thymic_aging_atlas" / "atlas_gene_age_effects.tsv.gz"), sep="\t")
    composition = pd.read_csv(require(args.root / "10_results" / "human_thymic_aging_atlas" / "atlas_composition_age_effects.tsv"), sep="\t")
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
    contract = {"figures": [f"Figure{i}_restyled" for i in range(1, 6)],
                "source_values_digitized_from_images": False,
                "figure1_display_panels": [str(args.umap_panel), str(args.marker_panel)],
                "note": "UMAP/dotplot may be rasterized; quantitative panels are source-table driven."}
    (args.output / "figure_build_contract.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

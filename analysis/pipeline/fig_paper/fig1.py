"""Figure 1: cohort, QC, annotation, composition (publication rebuild v2)."""
from __future__ import annotations

import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, MaxNLocator

from figure_common import *


def _design(ax, n_donors, n_cells):
    ax.axis("off")
    boxes = [
        (0.01, f"Cohort\n{n_donors} donors"),
        (0.26, f"QC + annotation\n{n_cells:,} cells"),
        (0.51, "Donor-level analysis\ncontinuous age"),
        (0.76, "External context\nhuman + mouse"),
    ]
    for x, t in boxes:
        ax.add_patch(mpatches.FancyBboxPatch(
            (x, 0.22), 0.21, 0.56, boxstyle="round,pad=0.02",
            fc="#EEF3FB", ec="#3B5B8A", lw=0.8))
        ax.text(x + 0.105, 0.50, t, ha="center", va="center", fontsize=7.0)
    for x in (0.22, 0.47, 0.72):
        ax.annotate("", xy=(x + 0.04, 0.50), xytext=(x, 0.50),
                    arrowprops=dict(arrowstyle="->", lw=1.0, color="#555"))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)


def _donor_strip(ax, fz):
    age_col = {"Young": AGEGROUP["Young"], "Mid": AGEGROUP["Mid"],
               "Old": AGEGROUP["Old"]}
    y = np.arange(len(fz))[::-1]
    for yi, (_, r) in zip(y, fz.iterrows()):
        male = str(r.sex).lower().startswith("m")
        ax.scatter(r.age_years, yi, s=23, color=age_col[str(r.age_group)],
                   marker="s" if male else "o", edgecolor="k", lw=0.3, zorder=3)
    for cut in (18, 40):
        ax.axvline(cut, ls="--", lw=0.7, color=GREY)
    ax.set_yticks(y); ax.set_yticklabels([short(d) for d in fz.donor_id], fontsize=5.5)
    ax.set_xlabel("Age (years)")
    ax.set_xlim(0, 73)
    sex_handles = [Line2D([0], [0], marker="o", ls="", color="k", label="Female"),
                   Line2D([0], [0], marker="s", ls="", color="k", label="Male")]
    ax.legend(handles=sex_handles, fontsize=6, frameon=False,
              loc="upper left", bbox_to_anchor=(1.02, 0.98))
    ax.text(1.02, 0.46, f"n={len(fz)}", transform=ax.transAxes, fontsize=6)
    ax.tick_params(axis="x", labelsize=6)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)


def _donor_qc(axs, obs):
    """Three independent QC axes; donors share age order, not a y-scale."""
    d = (obs.groupby("donor_id")
         .agg(genes=("n_genes_by_counts", "median"),
              umis=("total_counts", "median"),
              mt=("pct_counts_mt", "median"))
         .reset_index())
    fz = load_freeze()[["donor_id", "age_years"]]
    d = d.merge(fz, on="donor_id").sort_values("age_years")
    x = np.arange(len(d))
    for ax, col, title, c in zip(axs,
            ["genes", "umis", "mt"],
            ["Median genes/cell", "Median UMIs/cell", "Median % mito"],
            ["#0072B2", "#009E73", "#D55E00"]):
        ax.scatter(x, d[col], s=13, color=c, zorder=3)
        ax.set_title(title, fontsize=6.5, pad=3)
        ax.set_xlim(-0.7, len(d) - 0.3)
        ax.set_xticks([0, 5, 11, 17])
        ax.set_xticklabels(["1", "6", "12", "18"], fontsize=5.5)
        ax.set_xlabel("Donors (age order)", fontsize=5.8)
        ax.tick_params(axis="y", labelsize=5.5)
    # y-axes: genes and UMIs on log, mito on linear (percentage 0–100)
    for ax in axs[:2]:
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(base=10, numticks=3))
    axs[2].yaxis.set_major_locator(MaxNLocator(nbins=3))
    axs[2].set_ylim(0, max(17, d.mt.max() * 1.15))
    axs[2].axhline(15, ls="--", color=GREY, lw=0.6)
    axs[2].text(0.98, 15, "15%", transform=axs[2].get_yaxis_transform(),
                fontsize=5.2, va="bottom", ha="right")


def _retention(ax, fz, cc):
    """Fig1D: per-donor before→after dumbbell (review: slope/dumbbell, no
    overlapping labels on the line). Retention % placed right of each pair.
    """
    d = cc.merge(fz[["donor_id", "age_years"]], on="donor_id").sort_values("age_years")
    x = np.arange(len(d))
    b, a = d.before_QC.values, d.after_QC.values
    ax.hlines(x, b, a, color=LGREY, lw=1.1, zorder=1)
    ax.scatter(b, x, s=9, color="#BBBBBB", label="before QC", zorder=3)
    ax.scatter(a, x, s=9, color="#0072B2", label="after QC", zorder=3)
    for xi, (bb, aa) in enumerate(zip(b, a)):
        ax.text(max(bb, aa) * 1.04, xi, f"{aa/bb*100:.0f}%",
                fontsize=4.8, va="center", ha="left", color="#333333")
    ax.set_yticks(x); ax.set_yticklabels([short(v) for v in d.donor_id],
                                         fontsize=5)
    ax.set_ylim(-0.6, len(d) - 0.4)
    ax.set_xlabel("Cells before / after QC (log scale)", fontsize=6)
    ax.set_xscale("log")
    ax.set_xlim(min(np.r_[b, a]) * 0.8, max(np.r_[b, a]) * 2.5)
    ax.xaxis.set_major_locator(LogLocator(base=10, numticks=3))
    ax.legend(fontsize=5.3, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, -0.20), ncol=2)


def _umap(ax):
    import anndata as ad
    a = ad.read_h5ad(FILT / "07_GSE231906_thymus_annotated.h5ad", backed="r")
    idx = a.obs.sample(min(50000, a.n_obs), random_state=249).index
    pos = a.obs.index.get_indexer(idx)
    xy = a.obsm["X_umap"][pos]
    ct = a.obs.loc[idx, "broad_celltype"].values
    a.file.close()
    # Keep the cell cloud free of overlapping in-plot names.  All displayed
    # colours are decoded by an external, complete legend.
    handles = []
    for t in CT_ORDER:
        m = ct == t
        if not m.any():
            continue
        ax.scatter(xy[m, 0], xy[m, 1], s=1, alpha=0.4,
                   color=CELLTYPE_COLORS.get(t, "#999"), linewidths=0,
                   rasterized=True)
        handles.append(mpatches.Patch(color=CELLTYPE_COLORS.get(t, GREY),
                                      label=t.replace("_", " ")))
    # unresolved types in light grey, no labels
    for t in sorted(set(ct) - set(CT_ORDER)):
        m = ct == t
        ax.scatter(xy[m, 0], xy[m, 1], s=1, alpha=0.3, color="#D9D9D9",
                   linewidths=0, rasterized=True)
    if set(ct) - set(CT_ORDER):
        handles.append(mpatches.Patch(color="#D9D9D9", label="Other / unmapped"))
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect("equal", adjustable="box")
    ax.legend(handles=handles, fontsize=5.1, frameon=False, ncol=2,
              loc="upper left", bbox_to_anchor=(1.03, 1.0),
              columnspacing=0.55, handletextpad=0.25)


def _composition(ax, fz):
    prop = pd.read_csv(PSEUDO / "donor_celltype_proportion_all.csv")
    prop = prop.merge(fz[["donor_id", "age_years"]], on="donor_id").sort_values("age_years")
    keep = ["DN", "DP", "SP_CD4", "SP_CD8", "Treg", "B", "NK", "TEC", "GammaDelta_T"]
    keep = [c for c in keep if c in prop.columns]
    other = 1 - prop[keep].sum(axis=1)
    bottom = np.zeros(len(prop))
    x = np.arange(len(prop))
    for c in keep:
        ax.bar(x, prop[c], bottom=bottom, width=0.9, color=CELLTYPE_COLORS[c],
               label=c.replace("_", " "), lw=0)
        bottom += prop[c].values
    ax.bar(x, other.clip(lower=0), bottom=bottom, width=0.9, color=LGREY,
           label="Other", lw=0)
    ax.set_xticks(x); ax.set_xticklabels([short(v) for v in prop.donor_id],
                                         rotation=90, fontsize=5)
    ax.set_ylim(0, 1); ax.set_ylabel("Fraction of QC cells", fontsize=6)
    ax.legend(fontsize=4.7, ncol=5, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, -0.22), handletextpad=0.2,
              columnspacing=0.35)


def _fractions(axs, fz):
    prop = pd.read_csv(PSEUDO / "donor_celltype_proportion_all.csv").merge(
        fz[["donor_id", "age_years"]], on="donor_id")
    comp = pd.read_csv(FEAT / "composition" / "composition_age_Spearman.csv")
    types = ["DN", "DP", "SP_CD4", "SP_CD8", "TEC", "Treg"]
    for k, (ax, ct) in enumerate(zip(axs.flat, types)):
        d = prop[["age_years", ct]].dropna()
        r, ci = rho_ci(d.age_years, d[ct])
        q = comp.loc[comp.celltype == ct, "padj"]
        q = float(q.iloc[0]) if len(q) else np.nan
        ax.scatter(d.age_years, d[ct], s=11, color="#0072B2", alpha=0.8)
        z = np.polyfit(d.age_years, d[ct], 1)
        xx = np.linspace(d.age_years.min(), d.age_years.max(), 50)
        ax.plot(xx, np.polyval(z, xx), ls="--", lw=0.8, color=GREY)
        ax.set_title(ct.replace("_", " "), fontsize=6.5)
        ax.text(0.04, 0.93, f"ρ={r:.2f}; q={q:.2g}",
                transform=ax.transAxes, fontsize=5.4, va="top")
        ax.set_xlim(0, 73)
        ax.tick_params(labelsize=5.5)
        if k < 3:
            ax.tick_params(axis="x", labelbottom=False)
    for ax in axs.flat[len(types):]:
        ax.axis("off")


def build():
    import anndata as ad
    fz = load_freeze()
    if len(fz) != 18 or not fz.donor_id.is_unique:
        raise ValueError("Figure 1: expected 18 unique donors in frozen cohort")
    cc = pd.read_csv(META / "06_QC_cell_counts_by_donor.csv")
    a6 = ad.read_h5ad(FILT / "06_GSE231906_thymus_QC.h5ad")
    obs6 = a6.obs.copy()
    a6.file.close()

    fig = plt.figure(figsize=(W_DOUBLE, 285 / 25.4))
    gs = gridspec.GridSpec(6, 6, figure=fig, left=0.13, right=0.96,
                           bottom=0.045, top=0.975, hspace=0.50, wspace=0.58,
                           height_ratios=[0.45, 1.2, 1.3, 1.6, 1.7, 2.6])
    axA = fig.add_subplot(gs[0, :]); letter(axA, "A"); _design(axA, len(fz), len(obs6))
    axB = fig.add_subplot(gs[1, :5]); letter(axB, "B"); _donor_strip(axB, fz)
    # Independent axes sit side by side: no compressed log tick labels.
    innerC = gs[2, :].subgridspec(1, 3, wspace=0.55)
    qc_axs = [fig.add_subplot(innerC[0, i]) for i in range(3)]
    letter(qc_axs[0], "C", dx=-0.20, dy=1.08)
    _donor_qc(qc_axs, obs6)
    axD = fig.add_subplot(gs[3, 0:3]); letter(axD, "D"); _retention(axD, fz, cc)
    axE = fig.add_subplot(gs[3, 3:6]); letter(axE, "E"); _composition(axE, fz)
    axF = fig.add_subplot(gs[4, :4]); letter(axF, "F"); _umap(axF)
    inner = gs[5, :].subgridspec(2, 3, hspace=0.72, wspace=0.55)
    axs = np.array([[fig.add_subplot(inner[i, j]) for j in range(3)]
                    for i in range(2)])
    letter(axs[0, 0], "G", dx=-0.25, dy=1.25)
    _fractions(axs, fz)
    save_composite(fig, "Figure1")

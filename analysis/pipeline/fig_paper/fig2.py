"""Figure 2: donor-level age-associated molecular & composition changes."""
from __future__ import annotations

import glob
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy import stats
from matplotlib.patches import Patch
from matplotlib.colors import Normalize
import textwrap

from figure_common import *

LIMMA = FEAT / "limma_adjusted"


def _volcano_adjusted(ax):
    d = pd.read_csv(LIMMA / "whole_thymus_limma_age_sex.csv")
    d["logq"] = -np.log10(d["adj.P.Val"].clip(lower=1e-300))
    sig = d["adj.P.Val"] < 0.05
    ax.scatter(d.loc[~sig, "logFC_age_perSD"], d.loc[~sig, "logq"],
               s=2, color=LGREY, linewidths=0, rasterized=True)
    up = sig & (d.logFC_age_perSD > 0)
    dn = sig & (d.logFC_age_perSD < 0)
    ax.scatter(d.loc[up, "logFC_age_perSD"], d.loc[up, "logq"], s=3,
               color=UP, linewidths=0, label=f"age-up ({up.sum()})")
    ax.scatter(d.loc[dn, "logFC_age_perSD"], d.loc[dn, "logq"], s=3,
               color=DOWN, linewidths=0, label=f"age-down ({dn.sum()})")
    # Export the exact labels instead of drawing six leader lines across the
    # panel title and neighbouring axes at final print size.
    top = d.sort_values("adj.P.Val").head(6)
    top.to_csv(FIGF / "fig2a_top_six_genes_audit.tsv", sep="\t", index=False)
    ax.axhline(-np.log10(0.05), ls="--", lw=0.6, color=GREY)
    ax.set_xlabel("Age coefficient (log2 CPM / SD age)", fontsize=6)
    ax.set_ylabel(r"$-\log_{10}(q)$")
    ax.legend(fontsize=5.5, frameon=False, loc="upper left")
    ax.set_title("Sex-adjusted limma-voom, 18 donors", fontsize=6.8, pad=6)


def _spearman_descriptive(ax):
    d = pd.read_csv(FEAT / "age_correlation" / "whole_thymus_Spearman_by_age.csv")
    d["logq"] = -np.log10(d["padj"].clip(lower=1e-300))
    sig = d.padj < 0.05
    ax.scatter(d.loc[~sig, "rho"], d.loc[~sig, "logq"], s=2, color=LGREY,
               linewidths=0, rasterized=True)
    ax.scatter(d.loc[sig, "rho"], d.loc[sig, "logq"], s=3, color="#009E73",
               linewidths=0, label=f"q<0.05 ({sig.sum()})")
    ax.axhline(-np.log10(0.05), ls="--", lw=0.6, color=GREY)
    ax.set_xlabel("Spearman ρ(expression, age); unadjusted", fontsize=6)
    ax.set_ylabel(r"$-\log_{10}(q)$")
    ax.legend(fontsize=5.5, frameon=False)
    ax.set_title("Mixture-level correlation (descriptive; not DE)", fontsize=6.8)


def _bubble(ax, count_ax, legend_ax):
    cov = pd.read_csv(LIMMA / "limma_celltype_coverage.csv")
    rows = []
    for _, r in cov.iterrows():
        t = pd.read_csv(LIMMA / f"{r['set']}_limma_age_sex.csv")
        sig = t[t["adj.P.Val"] < 0.05]
        up_sig = sig[sig.logFC_age_perSD > 0]
        dn_sig = sig[sig.logFC_age_perSD < 0]
        rows.append({"celltype": r["set"].replace("_", " "),
                     "n_tested": r.n_genes_tested, "n_donors": r.n_donors,
                     "up": len(up_sig), "down": len(dn_sig),
                     "med_t_up": float(up_sig.t.median()) if len(up_sig) else 0.0,
                     "med_t_down": float(dn_sig.t.median()) if len(dn_sig) else 0.0})
    b = pd.DataFrame(rows)
    b["celltype"] = b.celltype.str.replace("whole thymus", "whole thymus (all cells)")
    b = pd.concat([b[b.celltype.str.startswith("whole")],
                   b[~b.celltype.str.startswith("whole")]
                   .sort_values("up", ascending=False)], ignore_index=True)
    yy = np.arange(len(b))[::-1]
    sc = None
    for k, direction in enumerate(["up", "down"]):
        frac = b[direction] / b.n_tested
        # Area is deliberately bounded by row pitch in the full-width panel.
        sizes = 9 + 90 * frac.clip(0, 1)
        cvals = b.med_t_up if direction == "up" else b.med_t_down
        sc = ax.scatter(np.full(len(b), k), yy, s=sizes, c=cvals,
                        cmap="RdBu_r", vmin=-6, vmax=6, edgecolor="k",
                        linewidth=0.3)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["age-up", "age-down"])
    ax.set_yticks(np.arange(len(b))[::-1]); ax.set_yticklabels(b.celltype, fontsize=5.6)
    ax.set_xlim(-0.5, 1.5)
    ax.set_title("Age-associated genes by cell type", fontsize=6.8, pad=5)
    cb = plt.colorbar(sc, ax=ax, fraction=0.025, pad=0.02)
    cb.set_label("median t (sig genes)", fontsize=6)
    cb.ax.tick_params(labelsize=5.5)
    # Dedicated legend column: no legend artist can intrude into panel D.
    legend_ax.set_xlim(0, 1); legend_ax.set_ylim(0, 1); legend_ax.axis("off")
    legend_ax.text(0.05, 0.98, "Dot area:\nsig. / tested", fontsize=5.8,
                   va="top")
    for ypos, f in zip((0.70, 0.51, 0.29), (0.05, 0.15, 0.30)):
        legend_ax.scatter(0.22, ypos, s=9 + 90 * f, color="#BBBBBB",
                          edgecolor="k", linewidth=0.3)
        legend_ax.text(0.44, ypos, f"{f:.0%}", fontsize=5.5, va="center")
    # counts placed in a dedicated right-hand column (review Fig2C)
    count_ax.axis("off")
    for y, nt, nd in zip(yy, b.n_tested, b["up"] + b["down"]):
        count_ax.text(0.0, y, f"{nd}/{nt}", fontsize=5.2, va="center",
                      ha="left")
    count_ax.set_title("sig./tested", fontsize=5.6, pad=5)
    count_ax.set_xlim(0, 1); count_ax.set_ylim(-0.5, len(b) - 0.5)


def _heatmap(ax, ax_age, ax_sex):
    """Fig2D heatmap with age (continuous) and sex (discrete) annotation strips
    above the expression matrix (review Fig2D: add strips, top margin).
    """
    lm = pd.read_csv(LIMMA / "whole_thymus_limma_age_sex.csv")
    top = lm.sort_values("adj.P.Val").gene.drop_duplicates().tolist()
    expr = pd.read_csv(PSEUDO / "donor_matrix" / "log_normalized_combined.csv.gz",
                       index_col=0)
    fz = load_freeze().set_index("donor_id")
    expr = expr.loc[fz.index]
    top = [g for g in top if g in expr.columns][:16]
    if len(top) < 12:
        raise ValueError("Figure 2D: fewer than 12 top genes present in expression matrix")
    # age strip (continuous, RdBu_r: blue=young, red=old)
    age_min, age_max = fz.age_years.min(), fz.age_years.max()
    ages = fz.age_years.values
    norm_age = plt.Normalize(age_min, age_max)
    ax_age.imshow(ages.reshape(1, -1), aspect="auto",
                  cmap="RdBu_r", norm=norm_age)
    ax_age.set_xticks([]); ax_age.set_yticks([])
    ax_age.set_ylabel("age", fontsize=5.2, rotation=0, ha="right", va="center")
    ax_age.set_title(f"Top age genes; donor ages {age_min:.0f}–{age_max:.0f} yr",
                     fontsize=6.8, pad=6)
    # sex strip (categorical, FEMALE/MALE); a separate axes, not a twinx.
    ax_sex.imshow(np.zeros((1, len(fz))), aspect="auto", cmap="Greys", vmin=0, vmax=1)
    for i, s in enumerate(fz.sex):
        ax_sex.add_patch(plt.Rectangle((i - 0.5, -0.5), 1, 1,
                                       color=FEMALE if s == "Female" else MALE,
                                       ec="none"))
    ax_sex.set_xticks([]); ax_sex.set_yticks([])
    ax_sex.set_ylabel("sex", fontsize=5.2, rotation=0, ha="right", va="center")
    # expression matrix
    M = expr[top].to_numpy(dtype=float)
    sigma = np.nanstd(M, axis=0)
    sigma[sigma == 0] = np.nan
    Z = (M - np.nanmean(M, axis=0)) / sigma
    Z = Z.T  # one complete row per selected gene
    if Z.shape != (len(top), len(expr)):
        raise AssertionError("Figure 2D: gene/donor heatmap axes reversed")
    cmap = plt.get_cmap("RdBu_r").copy(); cmap.set_bad("#B7B7B7")
    im = ax.imshow(np.ma.masked_invalid(np.clip(Z, -2, 2)), aspect="auto",
                   cmap=cmap, vmin=-2, vmax=2)
    ax.set_xticks(range(len(expr)))
    ax.set_xticklabels([short(d) for d in expr.index], fontsize=5.2, rotation=90)
    ax.set_yticks(range(len(top))); ax.set_yticklabels(top, fontsize=5.6)
    cb = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cb.set_label("row z (logCPM)", fontsize=6)
    ax.set_title("")
    pd.DataFrame(Z, index=top, columns=expr.index).to_csv(
        FIGF / "fig2d_heatmap_row_z_source.tsv", sep="\t", na_rep="NA")


def _composition_forest(ax):
    c = pd.read_csv(FEAT / "composition" / "composition_age_Spearman.csv")
    prop = pd.read_csv(PSEUDO / "donor_celltype_proportion_all.csv")
    fz = load_freeze()[["donor_id", "age_years"]]
    prop = prop.merge(fz, on="donor_id")
    rows = []
    for _, r in c.iterrows():
        r2, ci = rho_ci(prop.age_years, prop[r.celltype])
        rows.append({"ct": r.celltype.replace("_", " "), "rho": r.rho,
                     "lo": ci[0], "hi": ci[1], "q": r.padj})
    f = pd.DataFrame(rows).sort_values("rho")
    y = np.arange(len(f))
    colors = [UP if (r > 0 and q < 0.05) else DOWN if (r < 0 and q < 0.05)
              else GREY for r, q in zip(f.rho, f.q)]
    ax.hlines(y, f.lo, f.hi, color=GREY, lw=0.7, alpha=0.7)
    ax.scatter(f.rho, y, s=16, color=colors, zorder=3)
    ax.axvline(0, color="k", lw=0.6)
    ax.set_yticks(y); ax.set_yticklabels(f.ct, fontsize=5.6)
    ax.set_xlabel("Spearman ρ(fraction, age); bootstrap 95% CI; n=18", fontsize=6)
    ax.tick_params(labelsize=5.6)
    ax.set_title("Composition–age (all-QC denominator)", fontsize=6.8)
    ax.legend(handles=[Patch(color=UP, label="age-up, BH q<0.05"),
                       Patch(color=DOWN, label="age-down, q<0.05"),
                       Patch(color=GREY, label="not significant")],
              fontsize=5.2, frameon=False, loc="upper center", ncol=3,
              bbox_to_anchor=(0.5, -0.17))
    ax.axvline(0, color="k", lw=0.6)


def _hallmark_dotplot(ax, legend_ax=None):
    """Offline hypergeometric enrichment of limma age genes vs MSigDB Hallmark."""
    gmt = "/data/home/CMML/毕业课题及图片/10-13/2.2450hmark-代谢ssgsea/h.all.v2023.2.Hs.symbols.gmt"
    if not os.path.exists(gmt):
        gmt = "/data/home/CMML/毕业论文及图表/2.2450hmark-代谢ssgsea/h.all.v2023.2.Hs.symbols.gmt"
    gmt = os.environ.get("HALLMARK_GMT", gmt)
    if not os.path.isfile(gmt):
        raise FileNotFoundError("Set HALLMARK_GMT to the exact Hallmark GMT used "
                                f"by the analysis; not found: {gmt}")
    sets = {}
    with open(gmt) as fh:
        for line in fh:
            parts = line.strip().split("\t")
            sets[parts[0].replace("HALLMARK_", "").replace("_", " ")] = set(parts[2:])
    lm = pd.read_csv(LIMMA / "whole_thymus_limma_age_sex.csv")
    universe = set(lm.gene)
    sig = set(lm[lm["adj.P.Val"] < 0.05].gene)
    M, N = len(sig), len(universe)
    rows = []
    for name, gs in sets.items():
        gs = gs & universe
        k = len(sig & gs)
        if not gs:
            continue
        p = stats.hypergeom.sf(k - 1, N, len(gs), M)
        rows.append({"term": name, "Count": k, "set_size": len(gs),
                     "GeneRatio": k / M, "p": p})
    tested = pd.DataFrame(rows)
    if tested.empty or M == 0:
        raise ValueError("Figure 2F: no Hallmark tests or age-associated genes")
    tested["q"] = bh(tested.p.values)  # BH over all tested sets, not only shown sets
    tested.assign(universe_N=N, age_gene_M=M, GMT=gmt,
                  selected_direction="both age-up and age-down").to_csv(
        FIGF / "fig2f_all_hallmark_tests.tsv", sep="\t", index=False)
    e = tested[tested.Count.ge(3)].sort_values("q").head(10).iloc[::-1]
    if e.empty:
        raise ValueError("Figure 2F: no Hallmark set has >=3 overlapping age genes")
    if not (e.Count.ge(1).all() and np.all(e.Count == e.Count.astype(int))):
        raise AssertionError("Figure 2F: Count must be a positive integer")
    e.assign(universe_N=N, age_gene_M=M, GMT=gmt,
             selected_direction="both age-up and age-down").to_csv(
        FIGF / "fig2f_hallmark_enrichment_source.tsv", sep="\t", index=False)
    y = np.arange(len(e))
    count_min, count_max = int(e.Count.min()), int(e.Count.max())
    def area(count):
        return 28 + 120 * (count - count_min) / max(1, count_max - count_min)
    sizes = e.Count.map(area)
    points = ax.scatter(e.GeneRatio, y, s=sizes,
                        c=-np.log10(e.q.clip(lower=1e-30)), cmap="Blues",
                        edgecolor="k", linewidth=0.3)
    ax.set_yticks(y)
    ax.set_yticklabels(["\n".join(textwrap.wrap(t, 25)) for t in e.term], fontsize=5.4)
    ax.set_xlabel("GeneRatio = Count / all age-associated genes", fontsize=6)
    ax.set_title("Hallmark enrichment (hypergeometric, offline GMT)", fontsize=6.8)
    count_ticks = sorted(set(np.rint(np.quantile(e.Count, [0, .5, 1])).astype(int)))
    if legend_ax is None:  # standalone / contract-test use
        for count in count_ticks:
            ax.scatter([], [], s=area(count), color="#BBBBBB", edgecolor="k",
                       linewidth=0.3, label=f"{count}")
        ax.legend(title="Count (genes)", fontsize=5.2, title_fontsize=5.5,
                  frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    else:
        legend_ax.set_xlim(0, 1); legend_ax.set_ylim(0, 1); legend_ax.axis("off")
        legend_ax.text(0.03, 0.99, "Count\n(genes)", fontsize=5.5, va="top")
        for ypos, count in zip(np.linspace(0.75, 0.49, len(count_ticks)), count_ticks):
            legend_ax.scatter(0.22, ypos, s=area(count), color="#BBBBBB",
                              edgecolor="k", linewidth=0.3)
            legend_ax.text(0.55, ypos, str(count), fontsize=5.2, va="center")
        legend_ax.text(0.03, 0.29, "−log10(BH q)", fontsize=5.2)
        cax = legend_ax.inset_axes([0.03, 0.12, 0.94, 0.08])
        cb = ax.figure.colorbar(points, cax=cax, orientation="horizontal")
        cb.ax.tick_params(labelsize=4.8, pad=1)


def build():
    fig = plt.figure(figsize=(W_DOUBLE, 290 / 25.4))
    gs = gridspec.GridSpec(5, 6, figure=fig, left=0.18, right=0.95,
                           bottom=0.045, top=0.97, hspace=0.56, wspace=0.45,
                           height_ratios=[1.35, 1.5, 1.7, 1.7, 1.45])
    axA = fig.add_subplot(gs[0, :3]); letter(axA, "A", dx=-0.14); _volcano_adjusted(axA)
    axB = fig.add_subplot(gs[0, 3:]); letter(axB, "B", dx=-0.08); _spearman_descriptive(axB)
    # Full-width bubble panel with separate count and area-legend columns.
    innerC = gs[1, :].subgridspec(1, 3, width_ratios=[6.5, 1.3, 1.2], wspace=0.10)
    axC = fig.add_subplot(innerC[0, 0]); letter(axC, "C", dx=-0.14, dy=1.06)
    axCcount = fig.add_subplot(innerC[0, 1], sharey=axC)
    axClegend = fig.add_subplot(innerC[0, 2])
    _bubble(axC, axCcount, axClegend)
    # Full-width heatmap: 16 genes and 18 donors require their own row.
    innerD = gs[2, :].subgridspec(3, 1, height_ratios=[0.07, 0.07, 0.86], hspace=0.03)
    axD_age = fig.add_subplot(innerD[0, 0])
    axD_sex = fig.add_subplot(innerD[1, 0], sharex=axD_age)
    axD_exp = fig.add_subplot(innerD[2, 0], sharex=axD_age)
    letter(axD_age, "D", dx=-0.14, dy=1.15)
    _heatmap(axD_exp, axD_age, axD_sex)
    axE = fig.add_subplot(gs[3, :]); letter(axE, "E", dx=-0.14); _composition_forest(axE)
    axF = fig.add_subplot(gs[4, :5]); letter(axF, "F", dx=-0.14)
    axFlegend = fig.add_subplot(gs[4, 5])
    _hallmark_dotplot(axF, axFlegend)
    save_composite(fig, "Figure2")

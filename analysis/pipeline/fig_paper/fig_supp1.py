"""Supplementary figures set 1 (S01-S06)."""
from __future__ import annotations

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from figure_common import *


def s01_age_task():
    """Age/sex distribution strip (review S01): donors in ascending age order,
    labels placed next to points with alternating offsets, sex legend, short
    canvas — no floating labels over empty space."""
    fz = load_freeze().sort_values("age_years").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(W_DOUBLE, 40 / 25.4))
    for i, (_, r) in enumerate(fz.iterrows()):
        sex_m = (str(r.get("sex", "")).strip().lower().startswith("m"))
        c = MALE if sex_m else FEMALE
        ax.scatter(r.age_years, 0, s=16, color=c, zorder=3,
                   marker="s" if sex_m else "o")
        # alternating above/below offsets keep adjacent donors readable
        dy = 0.09 if i % 2 == 0 else -0.09
        ax.annotate(short(r.donor_id), xy=(r.age_years, 0),
                    xytext=(r.age_years, dy), ha="center", va="center",
                    fontsize=5.2, color="0.25",
                    arrowprops=dict(arrowstyle="-", lw=0.35, color=GREY))
    for cut in (18, 40):
        ax.axvline(cut, ls="--", lw=0.7, color=GREY)
    ax.text(18, -0.02, "Young<18 / binary", fontsize=5, color=GREY,
            ha="center", va="top")
    ax.text(40, -0.02, "Old>=40 / binary", fontsize=5, color=GREY,
            ha="center", va="top")
    ax.set_yticks([])
    ax.set_ylim(-0.17, 0.17)
    ax.set_xlabel("Age (years)")
    ax.set_title("Age task design: continuous regression n=18; exploratory "
                 "binary n=11 (4 Young / 7 Old)", fontsize=7)
    handles = [Line2D([0], [0], marker="o", ls="", color=FEMALE, label="Female"),
               Line2D([0], [0], marker="s", ls="", color=MALE, label="Male")]
    ax.legend(handles=handles, loc="upper right", fontsize=5.5,
              frameon=False, ncol=2)
    save_supp(fig, "Supplementary_Figure_01_age_task_design")


def s02_batch_umap():
    import anndata as ad
    a = ad.read_h5ad(FILT / "07_GSE231906_thymus_annotated.h5ad", backed="r")
    idx = a.obs.sample(min(50000, a.n_obs), random_state=249).index
    pos = a.obs.index.get_indexer(idx)
    xy = a.obsm["X_umap"][pos]
    obs = a.obs.loc[idx]
    a.file.close()
    fig, axs = plt.subplots(1, 3, figsize=(W_DOUBLE, 65 / 25.4))
    fig.subplots_adjust(left=0.06, right=0.76, bottom=0.12, top=0.72, wspace=0.45)
    for ax, col, cmap in [
            (axs[0], "age_years", "viridis"),
            (axs[1], "sex", None),
            (axs[2], "gsm_id", None)]:
        if col == "age_years":
            sc = ax.scatter(xy[:, 0], xy[:, 1], c=obs[col].astype(float),
                            s=1, alpha=0.4, cmap=cmap, rasterized=True)
            cb = fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.03)
            cb.set_label("age (years)", fontsize=6)
        else:
            cats = obs[col].astype(str)
            for i, v in enumerate(sorted(cats.unique())):
                m = cats == v
                ax.scatter(xy[m, 0], xy[m, 1], s=1, alpha=0.4, label=v,
                           rasterized=True)
            # legend OUTSIDE the axes so it never covers the UMAP
            ax.legend(fontsize=4.2, markerscale=3, frameon=False,
                      loc="upper left", bbox_to_anchor=(1.02, 1.0),
                      title=col, title_fontsize=5, ncol=1)
        ax.set_title(col, fontsize=7); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Exploratory batch diagnostic (not a formal mixing test)",
                 fontsize=8, y=0.96)
    save_supp(fig, "Supplementary_Figure_02_batch_diagnostic_UMAP")


def s03_marker_dotplot():
    import anndata as ad
    a = ad.read_h5ad(FILT / "07_GSE231906_thymus_annotated.h5ad", backed="r")
    all_ct = a.obs["broad_celltype"].astype(str)
    cts = [c for c in ["DN", "DP", "SP_CD4", "SP_CD8", "Treg", "GammaDelta_T",
                       "NKT_like", "NK", "B", "Plasma", "DC", "Macrophage",
                       "TEC", "Endothelial", "Fibroblast"] if c in set(all_ct)]
    genes = []
    for ct in cts:
        for g in MARKERS.get(ct, [])[:3]:
            genes.append((g, ct))
    sym_arr = a.var["gene_symbol"].astype(str).values
    g2col = {}
    for j, g in enumerate(sym_arr):
        g2col.setdefault(g, []).append(j)
    rows_exp, rows_frac = [], []
    genes_kept = []
    for g, ct in genes:
        cols = g2col.get(g, [])
        ii = np.where((all_ct == ct).values)[0]
        if len(cols) == 0 or len(ii) == 0:
            rows_exp.append(0.0); rows_frac.append(0.0); genes_kept.append((g, ct)); continue
        sub = np.asarray(a.layers["counts"][ii][:, cols].sum(1)).ravel()
        libs = np.asarray(a.layers["counts"][ii].sum(1)).ravel()
        lx = np.log1p(sub / np.maximum(libs, 1) * 1e6)
        rows_exp.append(float(lx.mean())); rows_frac.append(float((sub > 0).mean()))
        genes_kept.append((g, ct))
    genes = genes_kept
    a.file.close()
    ug = list(dict.fromkeys([g for g, _ in genes]))
    M = np.full((len(cts), len(ug)), np.nan)
    F = np.full_like(M, np.nan)
    for (g, ct), e, f in zip(genes, rows_exp, rows_frac):
        M[cts.index(ct), ug.index(g)] = e; F[cts.index(ct), ug.index(g)] = f
    fig, ax = plt.subplots(figsize=(W_DOUBLE, 90 / 25.4))
    im = ax.scatter(np.tile(range(len(ug)), len(cts)),
                    np.repeat(range(len(cts)), len(ug)),
                    s=20 + 160 * np.nan_to_num(F).ravel(),
                    c=np.nan_to_num(M).ravel(), cmap="Purples",
                    edgecolor="k", linewidth=0.2)
    ax.set_xticks(range(len(ug))); ax.set_xticklabels(ug, rotation=90, fontsize=5.4)
    ax.set_yticks(range(len(cts))); ax.set_yticklabels([c.replace("_", " ") for c in cts], fontsize=5.6)
    ax.set_ylim(len(cts) - 0.5, -0.5)
    cb = plt.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    cb.set_label("mean logCPM", fontsize=6)
    # size legend for % expressing (review S03)
    fracs = [0, 0.25, 0.50, 0.75, 1.0]
    sizes = [20 + 160 * f for f in fracs]
    for sz, lab in zip(sizes, [f"{int(f*100)}%" for f in fracs]):
        ax.scatter([], [], s=sz, c="gray", alpha=0.6, edgecolor="k",
                   linewidth=0.2, label=lab)
    ax.legend(title="% expressing", title_fontsize=5, fontsize=5,
              frameon=True, loc="upper left", bbox_to_anchor=(1.02, 0.98), borderpad=0.3,
              handlelength=1.2, labelspacing=0.6)
    fig.subplots_adjust(left=0.12, right=0.80, bottom=0.20, top=0.88)
    ax.set_title("Canonical marker dotplot (gene symbols; eligible cells only)", fontsize=7)
    save_supp(fig, "Supplementary_Figure_03_marker_dotplot")

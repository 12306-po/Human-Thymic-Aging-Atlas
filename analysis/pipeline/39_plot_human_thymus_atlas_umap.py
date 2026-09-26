#!/usr/bin/env python3
"""Plot the real 18-donor human thymus single-cell atlas from annotated h5ad.

The script never simulates coordinates.  It uses ``obsm['X_umap']`` when that
coordinate matrix is present.  With ``--recompute-umap`` it rebuilds PCA,
neighbors and UMAP from the annotated object before plotting.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


CELLTYPE_ORDER = [
    "DN", "DP", "SP_CD4", "SP_CD8", "Treg", "GammaDelta_T", "NKT_like",
    "NK", "B", "Plasma", "DC", "Monocyte", "Macrophage", "Myeloid",
    "TEC", "Endothelial", "Fibroblast", "SP_unresolved", "Unknown",
]

CELLTYPE_COLORS = {
    "DN": "#123F8C", "DP": "#3574B8", "SP_CD4": "#5B9BD5",
    "SP_CD8": "#9CC3E6", "Treg": "#7030A0", "GammaDelta_T": "#8B1E5A",
    "NKT_like": "#AF7AA1", "NK": "#4AAAD8", "B": "#EFA500",
    "Plasma": "#F5E72E", "DC": "#009E73", "Monocyte": "#D95F02",
    "Macrophage": "#A65628", "Myeloid": "#E5AE7A", "TEC": "#147A36",
    "Endothelial": "#79C4E3", "Fibroblast": "#725B3F",
    "SP_unresolved": "#E2E2E2", "Unknown": "#CFCFCF",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--h5ad",
        default=(
            "/data/zxy/projects/human_thymus_age_ML_DL/"
            "01_raw_processing/filtered/07_GSE231906_thymus_annotated.h5ad"
        ),
    )
    p.add_argument("--outdir", default="10_results/human_thymic_aging_atlas/umap")
    p.add_argument("--celltype-key", default="broad_celltype")
    p.add_argument("--donor-key", default="donor_id")
    p.add_argument("--age-key", default="age_years")
    p.add_argument("--sex-key", default="sex")
    p.add_argument("--expected-donors", type=int, default=18)
    p.add_argument("--expected-cells", type=int, default=114957)
    p.add_argument("--max-display-cells", type=int, default=50000,
                   help="0 plots every cell; otherwise deterministic subsample")
    p.add_argument("--seed", type=int, default=249)
    p.add_argument("--recompute-umap", action="store_true")
    return p.parse_args()


def configure_style():
    for f in [
        "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf",
        str(Path.home() / "fonts" / "times_new_roman" / "TIMES.ttf"),
    ]:
        if Path(f).exists():
            mpl.font_manager.fontManager.addfont(f)
    mpl.rcParams.update({
        "font.family": "Times New Roman", "font.size": 8,
        "axes.titlesize": 10, "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def recompute_umap(adata, seed):
    import scanpy as sc

    adata = adata.copy()
    if "counts" in adata.layers:
        adata.X = adata.layers["counts"].copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=3000, flavor="seurat_v3")
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, n_comps=50, use_highly_variable=True, random_state=seed)
    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=40, random_state=seed)
    sc.tl.umap(adata, min_dist=0.3, spread=1.0, random_state=seed)
    return adata


def main():
    args = parse_args()
    configure_style()
    import anndata as ad

    h5ad = Path(args.h5ad)
    if not h5ad.exists():
        raise FileNotFoundError(f"Annotated h5ad not found: {h5ad}")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    adata = ad.read_h5ad(h5ad)
    needed = [args.celltype_key, args.donor_key]
    missing = [c for c in needed if c not in adata.obs]
    if missing:
        raise KeyError(f"Missing required obs columns: {missing}")
    if args.recompute_umap:
        adata = recompute_umap(adata, args.seed)
    if "X_umap" not in adata.obsm:
        raise KeyError("X_umap is absent; rerun with --recompute-umap")

    n_cells = int(adata.n_obs)
    n_donors = int(adata.obs[args.donor_key].nunique())
    if args.expected_cells and n_cells != args.expected_cells:
        raise ValueError(f"Expected {args.expected_cells} cells, found {n_cells}")
    if args.expected_donors and n_donors != args.expected_donors:
        raise ValueError(f"Expected {args.expected_donors} donors, found {n_donors}")

    rng = np.random.default_rng(args.seed)
    if args.max_display_cells and n_cells > args.max_display_cells:
        pos = np.sort(rng.choice(n_cells, args.max_display_cells, replace=False))
    else:
        pos = np.arange(n_cells)
    xy = np.asarray(adata.obsm["X_umap"])[pos]
    obs = adata.obs.iloc[pos].copy()

    fig, ax = plt.subplots(figsize=(7.1, 4.9))
    handles = []
    observed = set(obs[args.celltype_key].astype(str))
    for celltype in CELLTYPE_ORDER:
        if celltype not in observed:
            continue
        mask = obs[args.celltype_key].astype(str).eq(celltype).to_numpy()
        ax.scatter(xy[mask, 0], xy[mask, 1], s=1.3, alpha=0.48,
                   color=CELLTYPE_COLORS[celltype], linewidths=0, rasterized=True)
        handles.append(Patch(color=CELLTYPE_COLORS[celltype], label=celltype.replace("_", " ")))
    unexpected = sorted(observed - set(CELLTYPE_ORDER))
    for celltype in unexpected:
        mask = obs[args.celltype_key].astype(str).eq(celltype).to_numpy()
        ax.scatter(xy[mask, 0], xy[mask, 1], s=1.3, alpha=0.35,
                   color="#AAAAAA", linewidths=0, rasterized=True)
        handles.append(Patch(color="#AAAAAA", label=celltype.replace("_", " ")))

    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_title("Integrated human thymic single-cell transcriptome atlas")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect("equal", adjustable="box")
    ax.legend(handles=handles, frameon=False, ncol=2, loc="upper left",
              bbox_to_anchor=(1.02, 1.0), columnspacing=0.8, handletextpad=0.35)
    fig.text(0.12, 0.015,
             f"GSE231906; {n_donors} donors; {n_cells:,} QC-passed cells; "
             f"{len(pos):,} displayed; seed {args.seed}.", fontsize=7)
    fig.subplots_adjust(left=0.10, right=0.72, top=0.92, bottom=0.09)
    png = outdir / "human_thymus_integrated_UMAP.png"
    pdf = outdir / "human_thymus_integrated_UMAP.pdf"
    fig.savefig(png, dpi=600, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    plt.close(fig)

    source = pd.DataFrame({
        "cell_id": obs.index.astype(str),
        "UMAP1": xy[:, 0], "UMAP2": xy[:, 1],
        "broad_celltype": obs[args.celltype_key].astype(str).to_numpy(),
        "donor_id": obs[args.donor_key].astype(str).to_numpy(),
    })
    for key, name in [(args.age_key, "age_years"), (args.sex_key, "sex")]:
        if key in obs:
            source[name] = obs[key].to_numpy()
    source.to_csv(outdir / "human_thymus_integrated_UMAP_source.tsv.gz",
                  sep="\t", index=False, compression="gzip")
    print(f"PNG: {png}")
    print(f"PDF: {pdf}")
    print(f"Source: {outdir / 'human_thymus_integrated_UMAP_source.tsv.gz'}")


if __name__ == "__main__":
    main()

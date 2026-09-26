#!/usr/bin/env python
"""Step 07: Broad cell-type annotation using author hierarchical labels (rev. 3).

P0 fixes per Codex review 2026-09-14:
  1. Author label source is explicit and verified: the 'thymocyte_metadata'
     sheet of GSE231906_cell-level_metadata.xlsx covers ALL lineages
     (id_lv1 in {Hema, Mes, Epi, Endo}; id_lv2 in {Bcell, DP, ETP_DN,
     Innate_T, Myeloid, 'NK & ILC', SP, Epi, Mes, Endo}). Non-T cells are
     therefore annotated from REAL author labels, NOT propagated from a
     thymocyte-majority vote. ETP_TP / stromal sheets are probed as a
     defensive fallback (verified: their sample_id keys do not match our
     barcodes, so they add 0 cells).
  2. SP without CD4/CD8 information maps to 'SP_unresolved' (NOT SP_CD4).
  3. RTE: author labels contain no explicit RTE -> code path exists but
     produces 0 RTE cells; this is logged explicitly (RTE-like/mature-SP
     is NOT silently forced).
  4. Myeloid keeps Monocyte / Macrophage / DC distinct when the author
     label supports it.
  5. Propagation of 'Unknown' cells is only allowed when cluster purity
     >= 0.80 AND the majority celltype's marker score is concordant.
  6. Per-celltype marker-program scores (score_<CT>) are computed and
     compared with the assigned label (marker-score validation).
  7. P0-1: matching key is fixed to GSM + barcode (never (donor_id,
     barcode)). Author 'geo_sample_id' (e.g. 'donor7-1') is parsed to a
     library_key, mapped through the Step04 frozen library metadata
     (04A/04B library_prefix) to the canonical GSM id; obs-side key is
     gsm_id + barcode_original (Step06). Duplicate (GSM, barcode) rows in
     the author table raise RuntimeError (audit written first).
  8. P0-1b: library_key that misses the Step04 frozen list is logged in
     07_annotation_key_metadata_audit.csv and reported per-library; such
     author rows are NOT silently matched.
  9. P0-2: Innate_T without gamma-delta/NKT evidence maps to
     'Innate_T_unresolved' (NOT forced GammaDelta_T).
 10. P0-3: new obs column annotation_analysis_eligible, with
     EXCLUDED_MAIN = {Unknown, Innate_T_unresolved, SP_unresolved};
     07_analysis_eligibility_counts.csv written.

Outputs:
  01_raw_processing/filtered/07_GSE231906_thymus_annotated.h5ad
  01_raw_processing/metadata/07_author_label_mapping.csv
  01_raw_processing/metadata/07_annotation_counts.csv
  01_raw_processing/metadata/07_annotation_source_counts.csv
  01_raw_processing/metadata/07_cluster_label_purity.csv
  01_raw_processing/metadata/07_marker_score_validation.csv
  01_raw_processing/metadata/07_annotation_match_rate_by_donor.csv
  01_raw_processing/metadata/07_annotation_match_rate_by_library.csv
  01_raw_processing/metadata/07_annotation_key_duplicate_audit.csv
  01_raw_processing/metadata/07_annotation_key_metadata_audit.csv
  01_raw_processing/metadata/07_analysis_eligibility_counts.csv
  09_figures/07_annotation_umap_by_broad.pdf
  09_figures/07_annotation_umap_donor.pdf
  09_figures/07_annotation_umap_annotation_source.pdf
  09_figures/07_marker_dotplot_broad.pdf
  09_figures/07_marker_score_heatmap.pdf
  09_figures/07_cluster_label_purity.pdf
  10_results/logs/07_annotation.log
"""
from __future__ import annotations

import logging
import os
import re
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

PROJ = Path(os.environ["PROJ"])
META = PROJ / "01_raw_processing" / "metadata"
FILTERED = PROJ / "01_raw_processing" / "filtered"
FIGURES = PROJ / "09_figures"
LOGS = PROJ / "10_results" / "logs"

FILTERED.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)
LOGS.mkdir(parents=True, exist_ok=True)

QC_H5AD = FILTERED / "06_GSE231906_thymus_QC.h5ad"
META_XLSX = META / "GSE231906_cell-level_metadata.xlsx"

logger = logging.getLogger("step07")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler(sys.stdout)
fh = logging.FileHandler(LOGS / "07_annotation.log", mode="w")
fmt = logging.Formatter("%(asctime)s  %(levelname)s  %(message)s")
ch.setFormatter(fmt)
fh.setFormatter(fmt)
logger.addHandler(ch)
logger.addHandler(fh)

# ---------------------------------------------------------------------------
# Broad-celltype schema (review-mandated; includes RTE + SP_unresolved)
# ---------------------------------------------------------------------------
# P1 (2026-09-16): Innate_T_unresolved is a real broad_celltype value (it is in
# EXCLUDED_MAIN) and must appear in BROAD_ORDER, or the crosstab silently drops
# it and under-counts cells.
BROAD_ORDER = [
    "DN", "DP", "SP_CD4", "SP_CD8", "SP_unresolved", "RTE", "Treg",
    "GammaDelta_T", "NKT_like", "Innate_T_unresolved", "NK", "B", "Plasma",
    "DC", "Monocyte", "Macrophage", "Myeloid", "TEC", "Fibroblast",
    "Endothelial", "Unknown",
]

PURITY_THRESHOLD = 0.80  # propagation purity threshold (review-mandated)

# ---------------------------------------------------------------------------
# Analysis-eligibility schema (review-mandated P0-3)
# ---------------------------------------------------------------------------
EXCLUDED_MAIN = {"Unknown", "Innate_T_unresolved", "SP_unresolved"}

# Fallback sheets that keep their own (sample_id) keys: they cannot be
# resolved through the Step04 GSM map and are treated as unresolved rows.
_FALLBACK_SHEETS = ("ETP_TP_metadata", "stromal_cell_metadata")


def _parse_library_key(value) -> str | None:
    """Extract the library key (e.g. 'donor7-1') from an author sample id."""
    m = re.search(r"(donor\d+(?:-\d+)?)", str(value))
    return m.group(1) if m else None


def _symbol_to_gene_id(adata) -> dict:
    """Map unique gene_symbol -> Ensembl gene_id (review P0, 2026-09-15).

    Step 06 made `var_names` the canonical Ensembl gene_id and stored the
    biological symbol in `var["gene_symbol"]`. Marker programs are defined by
    gene symbol, so they must be translated to Ensembl IDs before being passed
    to `sc.tl.score_genes()`.

    Only symbols that map to EXACTLY ONE Ensembl ID are used (audited unique
    symbol mapping). A symbol mapping to >1 Ensembl ID is ambiguous and is
    excluded from marker lookup (never silently summed).
    """
    vm = pd.DataFrame({
        "gene_id": adata.var_names.astype(str),
        "gene_symbol": adata.var["gene_symbol"].astype(str),
    })
    counts = vm.groupby("gene_symbol")["gene_id"].nunique()
    unique_symbols = set(counts[counts == 1].index)
    return (
        vm[vm["gene_symbol"].isin(unique_symbols)]
        .drop_duplicates("gene_symbol")
        .set_index("gene_symbol")["gene_id"]
        .to_dict()
    )


def _read_step04_library_meta(meta_dir: Path) -> pd.DataFrame:
    """Frozen donor x library metadata from Step 04 (library_prefix -> gsm_id)."""
    frames = []
    for fname in ("04A_GSE231906_thymus_library_metadata.csv",
                  "04B_GSE231906_non_thymus_libraries_removed.csv"):
        f = meta_dir / fname
        if not f.exists():
            raise FileNotFoundError(f"Step04 frozen library metadata missing: {f}")
        d = pd.read_csv(f, low_memory=False)
        frames.append(d[["library_prefix", "gsm_id", "donor_id"]])
    lib = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["library_prefix"])
    lib["library_key"] = lib["library_prefix"].str.split("_", n=1).str[1]
    return lib


def map_broad(row: pd.Series) -> str:
    l1 = row.get("id_lv1")
    l2 = row.get("id_lv2")
    l3 = row.get("id_lv3")
    l4 = row.get("id_lv4")
    l2s = "" if pd.isna(l2) else str(l2)
    l3s = "" if pd.isna(l3) else str(l3)
    l4s = "" if pd.isna(l4) else str(l4)
    joined = f"{l2s}|{l3s}|{l4s}"

    # RTE (recent thymic emigrant) — author labels have no explicit RTE;
    # keep the rule so future label versions are handled.
    if "rte" in joined.lower() or "recent thymic emigrant" in joined.lower():
        return "RTE"

    if l2s == "DP":
        return "DP"
    if l2s == "ETP_DN":
        return "DN"  # ETP/TP/ISP are DN-stage progenitors
    if l2s == "SP":
        if "Treg" in l3s or "Treg" in l4s:
            return "Treg"
        if "CD4" in l3s or "CD4" in l4s or "pre-CD4" in l3s:
            return "SP_CD4"
        if "CD8" in l3s or "CD8" in l4s or "pre-CD8" in l3s:
            return "SP_CD8"
        # No CD4/CD8 information: do NOT default to SP_CD4
        return "SP_unresolved"
    if l2s == "Innate_T":
        if "γδ" in joined or "GammaDelta" in joined:
            return "GammaDelta_T"
        if "NKT" in joined or "CD8αα" in joined:
            return "NKT_like"
        return "Innate_T_unresolved"
    if l2s == "Bcell":
        if "Plasma" in joined:
            return "Plasma"
        return "B"
    if l2s == "NK & ILC":
        return "NK"
    if l2s == "Myeloid":
        if any(d in (l3s + l4s) for d in ["DC1", "DC2", "aDC", "pDC", "DC"]):
            return "DC"
        if "Macrophage" in (l3s + l4s):
            return "Macrophage"
        if "Monocyte" in (l3s + l4s):
            return "Monocyte"
        return "Myeloid"  # residual: Mast / Neutrophil etc.
    if l1 == "Epi":
        return "TEC"
    if l1 == "Mes":
        return "Fibroblast"
    if l1 == "Endo":
        return "Endothelial"
    return "Unknown"


# ---------------------------------------------------------------------------
# Marker programs (per-celltype, review-mandated)
# ---------------------------------------------------------------------------
MARKERS = {
    "DN": ["CD34", "CD7", "PTCRA"],
    "DP": ["CD4", "CD8A", "CD8B", "PTCRA"],
    "SP_CD4": ["CD4", "IL7R", "CD8A"],
    "SP_CD8": ["CD8A", "CD8B", "GZMK", "NKG7"],
    "RTE": ["PTK7", "PECAM1", "CD38", "CCR7"],
    "Treg": ["FOXP3", "CD4", "IL2RA"],
    "GammaDelta_T": ["TRDC", "TRGC1", "TRGC2"],
    "NKT_like": ["NKG7", "KLRD1", "CD3D"],
    "B": ["MS4A1", "CD79A", "CD19"],
    "Plasma": ["MZB1", "JCHAIN", "XBP1"],
    "NK": ["NKG7", "NCAM1", "KLRD1"],
    "DC": ["CLEC9A", "CD1C", "LILRA4", "IRF8"],
    "Monocyte": ["LYZ", "CD68", "FCGR3A", "CSF1R", "CD14"],
    "Macrophage": ["CD68", "CSF1R", "C1QA", "C1QB"],
    "TEC": ["EPCAM", "KRT8", "KRT18", "AIRE"],
    "Fibroblast": ["COL1A1", "COL1A2", "PDGFRA", "DCN"],
    "Endothelial": ["PECAM1", "VWF", "CDH5"],
}


def _normalize_author_table(df: pd.DataFrame, key_col: str,
                            donor_strip: bool = True,
                            library_meta: pd.DataFrame | None = None,
                            meta_dir: Path | None = None) -> pd.DataFrame:
    """Normalize an author annotation table to GSM + barcode_raw.

    Matching key (review P0-1) is the canonical GSM id + raw barcode:
      - 'geo_sample_id' (e.g. 'donor7-1') is regex-parsed to a library_key,
        mapped through the Step04 frozen library metadata to gsm_id;
      - unresolved keys are kept with gsm_id=NaN (reported, not silently
        matched);
      - barcode_original is stripped of any trailing '_donorN' suffix.
    Returns a df with columns: gsm_id, barcode_raw, donor_key, library_key,
    library_key_resolved, source_sheet, plus all id_* label columns.
    """
    df = df.copy()
    df["source_sheet"] = getattr(df, "attrs", {}).get("sheet", "unknown")
    if donor_strip:
        df["donor_key"] = df[key_col].astype(str).str.replace(r"-\d+$", "", regex=True)
    else:
        df["donor_key"] = df[key_col].astype(str)
    df["library_key"] = df[key_col].astype(str).map(_parse_library_key)
    if library_meta is None:
        library_meta = _read_step04_library_meta(meta_dir or META)
    key2gsm = dict(zip(library_meta["library_key"], library_meta["gsm_id"]))
    df["gsm_id"] = df["library_key"].map(key2gsm)
    df["library_key_resolved"] = df["gsm_id"].notna()
    df["barcode_raw"] = df["barcode"].astype(str).str.replace(r"_donor\S*$", "", regex=True)
    label_cols = [c for c in df.columns if c.startswith("id_lv") or c in ("identity", "phenotypical_id")]
    keep = ["gsm_id", "barcode_raw", "donor_key", "library_key",
            "library_key_resolved", "source_sheet"] + label_cols
    return df[keep]


def main() -> int:
    adata = sc.read_h5ad(QC_H5AD)
    logger.info(f"Loaded QC h5ad: {adata.n_obs} cells x {adata.n_vars} genes")
    logger.info(f"obs columns: {list(adata.obs.columns)}")

    # ------------------------------------------------------------------
    # 1. Read ALL author annotation sheets; normalize to GSM + barcode
    # ------------------------------------------------------------------
    library_meta = _read_step04_library_meta(META)
    xl = pd.ExcelFile(META_XLSX)
    logger.info(f"Author metadata sheets: {xl.sheet_names}")

    # Primary: thymocyte_metadata (verified: covers all lineages Hema/Mes/Epi/Endo)
    th_raw = xl.parse("thymocyte_metadata")
    th_raw.attrs["sheet"] = "thymocyte_metadata"
    th = _normalize_author_table(th_raw, key_col="geo_sample_id",
                                 library_meta=library_meta)
    logger.info(f"thymocyte_metadata rows: {len(th)} "
                f"(resolved keys: {int(th['library_key_resolved'].sum())})")

    # P0-1: duplicate (gsm_id, barcode_raw) in author table -> audit + fail
    th_dup = th[th["gsm_id"].notna() &
                th.duplicated(subset=["gsm_id", "barcode_raw"], keep=False)].copy()
    if len(th_dup) > 0:
        th_dup.to_csv(META / "07_annotation_key_duplicate_audit.csv", index=False)
        logger.error(f"Duplicate (gsm_id, barcode_raw) in author table: "
                     f"{len(th_dup)} rows -> {META / '07_annotation_key_duplicate_audit.csv'}")
        raise RuntimeError(
            f"Duplicate (gsm_id, barcode_raw) rows found in thymocyte_metadata "
            f"({len(th_dup)} rows); audit saved; resolve before continuing."
        )

    # P0-1b: library_key -> Step04 frozen metadata coverage audit
    key_audit_rows = []
    for lk, grp in th[~th["library_key_resolved"]].groupby("library_key"):
        key_audit_rows.append({
            "library_key": lk, "n_cells": len(grp),
            "source_sheet": "thymocyte_metadata",
            "status": "unresolved_in_metadata",
        })

    # Defensive fallbacks (verified to add 0 cells; kept for robustness):
    # ETP_TP / stromal sheets use sample_id formats that do not match our
    # Step04 library prefixes for the 18 eligible donors; still probed so a
    # future label version is handled. Their unresolved keys are audited.
    fallback_tables = []
    for sheet, key_col in [("ETP_TP_metadata", "sample_id"),
                           ("stromal_cell_metadata", "sample_id")]:
        try:
            fb_raw = xl.parse(sheet)
            fb_raw.attrs["sheet"] = sheet
            t = _normalize_author_table(fb_raw, key_col=key_col,
                                        library_meta=library_meta)
            fallback_tables.append(t)
            n_unres = int((~t["library_key_resolved"]).sum())
            logger.info(f"{sheet}: {len(t)} rows, {n_unres} unresolved keys")
            for lk, grp in t[~t["library_key_resolved"]].groupby("library_key"):
                key_audit_rows.append({
                    "library_key": lk, "n_cells": len(grp),
                    "source_sheet": sheet, "status": "unresolved_in_metadata",
                })
        except Exception as e:  # noqa: BLE001
            logger.warning(f"{sheet} read failed (skipped): {e}")

    if key_audit_rows:
        pd.DataFrame(key_audit_rows).to_csv(
            META / "07_annotation_key_metadata_audit.csv", index=False)
        logger.warning(f"Unresolved library keys logged -> "
                       f"{META / '07_annotation_key_metadata_audit.csv'}")

    # Build the combined author label index.
    # Priority: thymocyte first, then fallbacks (combine fills only missing
    # keys). Matching key (review P0-1) = (gsm_id, barcode_raw).
    label_cols = ["id_lv1", "id_lv2", "id_lv3", "id_lv4"]
    th_resolved = th[th["gsm_id"].notna()].copy()
    lab_idx = (th_resolved
               .set_index(["gsm_id", "barcode_raw"])[label_cols]
               .sort_index())
    for t2 in fallback_tables:
        if "identity" in t2.columns:
            t2["id_lv3"] = t2["id_lv3"].fillna(t2["identity"]) if "id_lv3" in t2.columns \
                else t2["identity"]
        if "phenotypical_id" in t2.columns:
            t2["id_lv4"] = t2["id_lv4"].fillna(t2["phenotypical_id"]) if "id_lv4" in t2.columns \
                else t2["phenotypical_id"]
        keep = [c for c in label_cols if c in t2.columns]
        t2_r = t2[t2["gsm_id"].notna()].copy()
        t2_idx = t2_r.set_index(["gsm_id", "barcode_raw"])[keep].sort_index()
        lab_idx = lab_idx.combine_first(t2_idx)
    lab_idx = lab_idx[~lab_idx.index.duplicated(keep="first")].sort_index()
    logger.info(f"Combined author label index (resolved): "
                f"{len(lab_idx)} unique (gsm, barcode)")

    n_unres_total = int((~th["library_key_resolved"]).sum()) + sum(
        int((~t["library_key_resolved"]).sum()) for t in fallback_tables)
    if n_unres_total > 0:
        logger.warning(f"Author rows with no GSM mapping (never used as labels): "
                       f"{n_unres_total}")

    # ------------------------------------------------------------------
    # 2. Match onto our cells — canonical key = (gsm_id, barcode_original)
    # ------------------------------------------------------------------
    obs = adata.obs
    cell_keys = list(zip(obs["gsm_id"].astype(str),
                         obs["barcode_original"].astype(str)))
    matched = lab_idx.reindex(cell_keys)
    for c in label_cols:
        adata.obs[f"author_{c}"] = matched[c].values

    has_label = adata.obs["author_id_lv2"].notna()
    logger.info(f"Matched cells with author labels: {has_label.sum()} / {adata.n_obs} "
                f"({100 * has_label.mean():.2f}%)")

    # Per-donor match rate
    mr_rows = []
    for d, g in adata.obs.groupby("donor_id"):
        mr_rows.append({
            "donor_id": d,
            "n_cells": len(g),
            "n_matched": int(g["author_id_lv2"].notna().sum()),
            "match_rate": round(float(g["author_id_lv2"].notna().mean()), 4),
        })
    mr = pd.DataFrame(mr_rows)
    mr.to_csv(META / "07_annotation_match_rate_by_donor.csv", index=False)
    logger.info("\nPer-donor author match rate:\n" + mr.to_string(index=False))

    # Per-library match rate (GSM level)
    mr_lib_rows = []
    for (d, gsm), g in adata.obs.groupby(["donor_id", "gsm_id"]):
        mr_lib_rows.append({
            "donor_id": d,
            "gsm_id": gsm,
            "n_cells": len(g),
            "n_matched": int(g["author_id_lv2"].notna().sum()),
            "match_rate": round(float(g["author_id_lv2"].notna().mean()), 4),
        })
    mr_lib = pd.DataFrame(mr_lib_rows)
    mr_lib.to_csv(META / "07_annotation_match_rate_by_library.csv", index=False)
    logger.info("\nPer-library author match rate:\n" + mr_lib.to_string(index=False))

    # ------------------------------------------------------------------
    # 3. original_label + broad_celltype
    # ------------------------------------------------------------------
    adata.obs["original_label"] = adata.obs["author_id_lv2"].astype(str).where(
        has_label, "Unmatched")

    author_cols = [f"author_{c}" for c in label_cols]
    broad = []
    for r in adata.obs[author_cols].itertuples(index=False):
        row = pd.Series({c: getattr(r, f"author_{c}") for c in label_cols})
        broad.append(map_broad(row))
    adata.obs["broad_celltype"] = pd.Series(broad, index=adata.obs_names)

    adata.obs.loc[~has_label, "broad_celltype"] = "Unknown"
    logger.info("\nBroad celltype counts (before propagation):\n" +
                adata.obs["broad_celltype"].value_counts().to_string())

    # ------------------------------------------------------------------
    # 4. Shared graph: normalize -> HVG -> PCA -> neighbors -> leiden -> UMAP.
    #    Computed ONCE and reused for propagation + all UMAP figures.
    # ------------------------------------------------------------------
    work = adata.copy()
    if "counts" in work.layers:
        work.X = work.layers["counts"].copy()
    sc.pp.normalize_total(work, target_sum=1e4)
    sc.pp.log1p(work)
    sc.pp.highly_variable_genes(work, n_top_genes=3000, flavor="seurat")
    sc.pp.pca(work, n_comps=30, use_highly_variable=True)
    sc.pp.neighbors(work, n_neighbors=15, n_pcs=30)
    sc.tl.leiden(work, resolution=1.0, key_added="leiden_anno", random_state=1)
    adata.obs["leiden_anno"] = work.obs["leiden_anno"].astype(str).values
    sc.tl.umap(work, random_state=1)
    adata.obsm["X_umap"] = work.obsm["X_umap"].copy()
    del work
    logger.info("Graph (neighbors/leiden) + UMAP computed once")

    # ------------------------------------------------------------------
    # 5. Per-celltype marker scores (computed on author-labeled cells so that
    #    propagation's marker-support check is not self-referential)
    # ------------------------------------------------------------------
    # P0 (2026-09-15): var_names are Ensembl IDs; translate marker symbols to
    # Ensembl IDs via the audited unique-symbol mapping before scoring.
    symbol_to_gene_id = _symbol_to_gene_id(adata)
    n_unique_symbols = len(symbol_to_gene_id)
    logger.info(f"Audited unique symbol -> Ensembl mapping: {n_unique_symbols} symbols "
                f"(ambiguous multi-Ensembl symbols excluded from marker lookup)")
    ct_scores = {}
    for ct, marker_list in MARKERS.items():
        avail_ids = [symbol_to_gene_id[g] for g in marker_list if g in symbol_to_gene_id]
        if len(avail_ids) < 2:
            logger.warning(f"marker set '{ct}': only {len(avail_ids)} mapped markers "
                           f"(symbols {marker_list}) -> marker score skipped")
            ct_scores[ct] = None
            continue
        try:
            sc.tl.score_genes(adata, gene_list=avail_ids,
                              score_name=f"score_{ct}", use_raw=False)
            ct_scores[ct] = f"score_{ct}"
        except Exception as e:  # noqa: BLE001
            logger.warning(f"score_genes failed for {ct}: {e}")
            ct_scores[ct] = None
    score_cols = [f"score_{ct}" for ct, s in ct_scores.items() if s]
    logger.info(f"Per-celltype marker scores computed: {len(score_cols)}")

    # ------------------------------------------------------------------
    # 6. Propagation of Unknown cells with purity threshold + marker check
    # ------------------------------------------------------------------
    unknown_mask = adata.obs["broad_celltype"] == "Unknown"
    n_unknown = int(unknown_mask.sum())
    logger.info(f"Unknown cells before propagation: {n_unknown} "
                f"({100 * unknown_mask.mean():.2f}%)")

    purity_rows = []
    if n_unknown > 0:
        known = adata.obs.loc[~unknown_mask]
        ct_cross = pd.crosstab(known["leiden_anno"], known["broad_celltype"])
        if len(ct_cross) > 0:
            majority = ct_cross.idxmax(axis=1)
            n_cluster_labeled = ct_cross.sum(axis=1)
            purity = ct_cross.max(axis=1) / n_cluster_labeled.clip(lower=1)
            cluster_majority = majority.to_dict()
            cluster_purity = purity.to_dict()

            # cluster x celltype mean marker score (author-labeled cells only)
            if score_cols:
                labeled = adata.obs.loc[~unknown_mask, ["leiden_anno"] + score_cols].copy()
                score_means = labeled.groupby("leiden_anno")[score_cols].mean()
            else:
                score_means = pd.DataFrame()

            marker_support = {}
            for cluster in ct_cross.index:
                maj_ct = cluster_majority.get(cluster)
                pur = cluster_purity.get(cluster, 0.0)
                sc_col = ct_scores.get(maj_ct)
                if sc_col is not None and sc_col in score_means.columns:
                    ranks = score_means.loc[cluster].rank(ascending=False)
                    mk_support = bool(ranks.get(sc_col, 99) <= 2)
                else:
                    mk_support = False  # cannot verify -> no propagation
                marker_support[cluster] = mk_support
                purity_rows.append({
                    "cluster": cluster,
                    "majority_celltype": maj_ct,
                    "n_author_labeled": int(n_cluster_labeled.get(cluster, 0)),
                    "n_cells_total": int(ct_cross.sum(axis=1).get(cluster, 0)),
                    "purity": round(float(pur), 4),
                    "marker_support": mk_support,
                    "n_propagated": 0,
                })

            # Propagate only when purity >= threshold AND marker support
            prop_mask = (
                unknown_mask
                & adata.obs["leiden_anno"].map(cluster_majority).notna()
                & adata.obs["leiden_anno"].map(cluster_purity).ge(PURITY_THRESHOLD)
                & adata.obs["leiden_anno"].map(marker_support)
            )
            adata.obs.loc[prop_mask, "broad_celltype"] = (
                adata.obs.loc[prop_mask, "leiden_anno"].map(cluster_majority).values)
            n_filled = int(prop_mask.sum())
            logger.info(f"Propagated {n_filled} unknown cells "
                        f"(purity>={PURITY_THRESHOLD} AND marker support)")

            # update purity rows with n_propagated
            prop_counts = adata.obs.loc[prop_mask, "leiden_anno"].value_counts().to_dict()
            for r in purity_rows:
                r["n_propagated"] = int(prop_counts.get(r["cluster"], 0))

            adata.obs.loc[prop_mask & ~has_label, "original_label"] = "Propagated"
            adata.obs["annotation_source"] = np.where(
                has_label, "author",
                np.where(adata.obs["broad_celltype"] != "Unknown",
                         "propagated", "unmatched"))
        else:
            adata.obs["annotation_source"] = np.where(has_label, "author", "unmatched")
    else:
        adata.obs["annotation_source"] = np.where(has_label, "author", "unmatched")

    purity_df = pd.DataFrame(purity_rows)
    purity_df.to_csv(META / "07_cluster_label_purity.csv", index=False)
    logger.info(f"Cluster purity table saved: {META / '07_cluster_label_purity.csv'}")

    # ------------------------------------------------------------------
    # 6b. Analysis eligibility (review P0-3)
    # ------------------------------------------------------------------
    adata.obs["annotation_analysis_eligible"] = (
        ~adata.obs["broad_celltype"].isin(EXCLUDED_MAIN))
    elig_counts = (
        adata.obs.groupby(["broad_celltype", "annotation_analysis_eligible"])
        .size().reset_index(name="n_cells"))
    elig_counts.to_csv(META / "07_analysis_eligibility_counts.csv", index=False)
    logger.info("\nAnnotation analysis eligibility:\n" + elig_counts.to_string(index=False))

    logger.info("\nBroad celltype counts after propagation:\n" +
                adata.obs["broad_celltype"].value_counts().to_string())

    # ------------------------------------------------------------------
    # 7. Marker-score validation (assigned label vs argmax program)
    # ------------------------------------------------------------------
    valid_cts = [ct for ct, s in ct_scores.items() if s is not None]
    score_cols_valid = [f"score_{ct}" for ct in valid_cts]
    validation_rows = []
    if score_cols_valid:
        score_mat = adata.obs[score_cols_valid].copy()
        argmax_ct = score_mat.idxmax(axis=1).str.replace("score_", "", regex=False)
        assigned = adata.obs["broad_celltype"]
        for ct in valid_cts:
            cells = assigned == ct
            if int(cells.sum()) == 0:
                continue
            own_score = float(adata.obs.loc[cells, f"score_{ct}"].mean())
            argmax_match = float((argmax_ct[cells] == ct).mean())
            validation_rows.append({
                "celltype": ct,
                "n_cells": int(cells.sum()),
                "mean_own_marker_score": round(own_score, 4),
                "frac_argmax_match": round(argmax_match, 4),
            })
    val_df = pd.DataFrame(validation_rows)
    val_df.to_csv(META / "07_marker_score_validation.csv", index=False)
    logger.info(f"Marker-score validation saved: {META / '07_marker_score_validation.csv'}")
    if len(val_df) > 0:
        logger.info("\nMarker-score validation:\n" + val_df.to_string(index=False))

    # annotation source counts
    src_counts = (adata.obs["annotation_source"].value_counts()
                  .rename_axis("source").reset_index(name="n_cells"))
    src_counts.to_csv(META / "07_annotation_source_counts.csv", index=False)
    logger.info(f"Annotation sources:\n{src_counts.to_string(index=False)}")

    # ------------------------------------------------------------------
    # 8. Figures
    # ------------------------------------------------------------------
    try:
        fig, ax = plt.subplots(figsize=(10, 8))
        sc.pl.umap(adata, color="broad_celltype", ax=ax, show=False,
                   title="Broad celltype (author + conservative propagation)")
        fig.savefig(FIGURES / "07_annotation_umap_by_broad.pdf", dpi=150,
                    bbox_inches="tight")
        plt.close(fig)
        logger.info(f"UMAP broad saved: {FIGURES / '07_annotation_umap_by_broad.pdf'}")
    except Exception as e:  # noqa: BLE001
        logger.warning(f"UMAP broad skipped: {e}")

    try:
        fig, ax = plt.subplots(figsize=(10, 8))
        sc.pl.umap(adata, color="donor_id", ax=ax, show=False, title="Donor")
        fig.savefig(FIGURES / "07_annotation_umap_donor.pdf", dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"UMAP donor saved: {FIGURES / '07_annotation_umap_donor.pdf'}")
    except Exception as e:  # noqa: BLE001
        logger.warning(f"UMAP donor skipped: {e}")

    try:
        fig, ax = plt.subplots(figsize=(10, 8))
        sc.pl.umap(adata, color="annotation_source", ax=ax, show=False,
                   title="Annotation source")
        fig.savefig(FIGURES / "07_annotation_umap_annotation_source.pdf", dpi=150,
                    bbox_inches="tight")
        plt.close(fig)
        logger.info(f"UMAP source saved: {FIGURES / '07_annotation_umap_annotation_source.pdf'}")
    except Exception as e:  # noqa: BLE001
        logger.warning(f"UMAP source skipped: {e}")

    try:
        # P0 (2026-09-15): dotplot also uses the symbol -> Ensembl mapping.
        avail_list = [symbol_to_gene_id[g]
                      for gs in MARKERS.values() for g in gs
                      if g in symbol_to_gene_id]
        if len(avail_list) >= 5:
            sc.pl.dotplot(adata, avail_list, groupby="broad_celltype", show=False,
                          save="_broad_dotplot.pdf", standard_scale="var")
            logger.info("Dotplot saved")
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Dotplot skipped: {e}")

    try:
        if score_cols_valid:
            mean_scores = adata.obs.groupby("broad_celltype")[score_cols_valid].mean()
            fig, ax = plt.subplots(figsize=(max(8, len(valid_cts) * 0.5),
                                            max(6, len(mean_scores) * 0.5)))
            im = ax.imshow(mean_scores.values, aspect="auto", cmap="RdBu_r")
            ax.set_xticks(range(len(valid_cts)))
            ax.set_xticklabels(valid_cts, rotation=90, fontsize=7)
            ax.set_yticks(range(len(mean_scores)))
            ax.set_yticklabels(mean_scores.index, fontsize=7)
            ax.set_title("Mean marker-program score by broad celltype")
            fig.colorbar(im, ax=ax, fraction=0.03)
            fig.tight_layout()
            fig.savefig(FIGURES / "07_marker_score_heatmap.pdf", dpi=150,
                        bbox_inches="tight")
            plt.close(fig)
            logger.info(f"Marker-score heatmap saved: {FIGURES / '07_marker_score_heatmap.pdf'}")
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Marker-score heatmap skipped: {e}")

    try:
        if len(purity_df) > 0:
            p = purity_df.sort_values("purity")
            fig, ax = plt.subplots(figsize=(10, max(4, len(p) * 0.3)))
            ax.barh(p["cluster"].astype(str), p["purity"], color="steelblue")
            ax.axvline(PURITY_THRESHOLD, color="red", ls="--", lw=1,
                       label=f"purity threshold {PURITY_THRESHOLD}")
            ax.set_xlabel("Cluster purity (author-labeled majority)")
            ax.set_xlim(0, 1)
            ax.legend()
            fig.tight_layout()
            fig.savefig(FIGURES / "07_cluster_label_purity.pdf", dpi=150,
                        bbox_inches="tight")
            plt.close(fig)
            logger.info(f"Cluster purity figure saved: {FIGURES / '07_cluster_label_purity.pdf'}")
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Cluster purity figure skipped: {e}")

    # ------------------------------------------------------------------
    # 9. Outputs
    # ------------------------------------------------------------------
    mapping_rows = []
    for _, row in adata.obs[author_cols].drop_duplicates().iterrows():
        row_dict = {c: row[f"author_{c}"] for c in label_cols}
        row_dict["broad_celltype"] = map_broad(pd.Series(row_dict))
        mapping_rows.append(row_dict)
    mapping_out = pd.DataFrame(mapping_rows)
    mapping_out = mapping_out.rename(columns={f"author_{c}": c for c in label_cols})
    mapping_out = mapping_out.drop_duplicates(subset=label_cols + ["broad_celltype"])
    mapping_out.to_csv(META / "07_author_label_mapping.csv", index=False)
    logger.info(f"Author->broad mapping saved: {META / '07_author_label_mapping.csv'}")

    # P1 (2026-09-16): fail fast if a broad_celltype value is not listed in
    # BROAD_ORDER — reindex() would otherwise drop it silently from the counts.
    _unlisted = sorted(set(adata.obs["broad_celltype"].dropna().unique())
                       - set(BROAD_ORDER))
    if _unlisted:
        raise RuntimeError(
            f"Unlisted broad celltypes missing from BROAD_ORDER: {_unlisted}"
        )

    counts = (
        pd.crosstab(adata.obs["donor_id"], adata.obs["broad_celltype"])
        .reindex(index=sorted(adata.obs["donor_id"].unique()),
                 columns=[c for c in BROAD_ORDER], fill_value=0)
    )
    counts.insert(0, "total", counts.sum(axis=1))
    counts.to_csv(META / "07_annotation_counts.csv")
    logger.info(f"Annotation counts saved: {META / '07_annotation_counts.csv'}")

    adata.write_h5ad(FILTERED / "07_GSE231906_thymus_annotated.h5ad")
    logger.info(f"Annotated h5ad saved: {FILTERED / '07_GSE231906_thymus_annotated.h5ad'}")

    logger.info("\n=== Step 07 COMPLETE ===")
    logger.info(f"Cells: {adata.n_obs}; annotation_source: "
                + adata.obs["annotation_source"].value_counts().to_dict().__str__())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
#!/usr/bin/env python
"""Step 06: Read 10x libraries, QC, per-library Scrublet audit, merge.

Inputs:
  04A_GSE231906_thymus_library_metadata.csv  (has matrix_path, features_path,
   barcodes_path, donor_id, gsm_id, age_years, sex, etc.)
  05B_final_regression_cohort.csv

Predeclared QC thresholds (frozen, saved to 06_QC_thresholds.json):
  min_genes_by_counts     = 500
  max_genes_by_counts     = 8000
  min_total_counts        = 1000
  max_total_counts        = 60000
  max_pct_counts_mt       = 15
  scrublet_doublet_score  = 0.25  (per-library threshold, reported but kept as
                                   information only – scrublet is an independent
                                   audit, not a mandatory filter in this pipeline)
  doublet_removal         = False (DOUBLE-T POLICY: doublets are NEVER removed)

Doublet policy (frozen, also written to 06_doublet_policy.txt):
  "Scrublet is independent audit only; doublets NOT removed"

Key design decisions (review fixes):
  * Scrublet uses the PUBLIC API and is SPARSE-FIRST: scr.Scrublet(X_csr,
    n_neighbors=15) on the CSR count matrix, then scores, predicted =
    scrub.scrub_doublets(). The matrix is densified ONLY if the installed
    scrublet rejects sparse input (dense fallback). The private
    `scr.Scrublet._LogTransformer()` no longer exists in the installed
    scrublet and is NOT used.
  * Per-library `filter_genes(min_cells=3)` is REMOVED.  Each library keeps its
    complete gene universe; gene symbols are aligned at concat time with
    ad.concat(join="outer", merge="unique"); gene filtering is done ONLY after
    concat on the merged object (sc.pp.filter_genes(min_cells=3)).
  * obs_names are unique across libraries: donor_id__gsm_id__barcode_original,
    and the same value is stored in obs['cell_uid'].
  * The data matrix is kept SPARSE (CSR) throughout, including the Scrublet
    call; a per-library dense copy is created only as a fallback when the
    installed scrublet cannot accept sparse input, and its peak memory is logged.
  * Predicted doublets are recorded in obs (scrublet_score, predicted_doublet)
    and reported in audit/sensitivity tables but are NOT removed.

Outputs:
  01_raw_processing/filtered/06_GSE231906_thymus_QC.h5ad
  01_raw_processing/metadata/06_QC_thresholds.json
  01_raw_processing/metadata/06_doublet_policy.txt
  01_raw_processing/metadata/06_QC_cell_counts_by_library.csv
  01_raw_processing/metadata/06_QC_cell_counts_by_donor.csv
  01_raw_processing/metadata/06_QC_retention_rate_by_library.csv
  01_raw_processing/metadata/06_QC_retention_rate_by_donor.csv
  01_raw_processing/metadata/06_scrublet_audit.csv
  01_raw_processing/metadata/06_doublet_sensitivity_summary.csv
  09_figures/06_QC_violin.pdf
  09_figures/06_QC_scatter.pdf
  09_figures/06_QC_retention_by_donor.pdf
  09_figures/06_scrublet_score_by_library.pdf
  10_results/logs/06_QC.log
"""

from __future__ import annotations

import json
import logging
import os
import resource
import sys
import warnings
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scrublet as scr
from scipy import sparse

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Paths — read from environment (set by source 00_project_config.sh)
# ---------------------------------------------------------------------------
def _env(key):
    val = os.environ.get(key)
    if not val:
        raise RuntimeError(f"Required env var {key} is not set. Run: source scripts/00_project_config.sh")
    return Path(val)

PROJ = _env("PROJ")
META = PROJ / "01_raw_processing" / "metadata"
FILTERED = PROJ / "01_raw_processing" / "filtered"
FIGURES = PROJ / "09_figures"
LOGS = PROJ / "10_results" / "logs"

FILTERED.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)
LOGS.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Memory helper (Linux: ru_maxrss is in KB)
# ---------------------------------------------------------------------------
def peak_mem_gb() -> float:
    """Current process peak RSS in GB (Linux: ru_maxrss is in KB)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


# ---------------------------------------------------------------------------
# Predeclared QC thresholds (frozen)
# ---------------------------------------------------------------------------
QC_THRESHOLDS = {
    "min_genes_by_counts": 500,
    "max_genes_by_counts": 8000,
    "min_total_counts": 1000,
    "max_total_counts": 60000,
    "max_pct_counts_mt": 15.0,
    "scrublet_doublet_score_threshold": 0.25,
    # Doublet policy is frozen: Scrublet is an independent audit ONLY.
    "doublet_removal": False,
    "min_cells_per_donor_celltype": 20,  # from plan v2.1 (for Step 08)
}
DOUBLET_POLICY = "record_only"
DOUBLET_POLICY_LINE = "Scrublet is independent audit only; doublets NOT removed"

# JSON thresholds file carries both the frozen thresholds and the explicit
# doublet policy marker.
qc_json = dict(QC_THRESHOLDS)
qc_json["doublet_policy"] = DOUBLET_POLICY
with open(META / "06_QC_thresholds.json", "w") as fh:
    json.dump(qc_json, fh, indent=2)

# One-line doublet policy file (auditable).
with open(META / "06_doublet_policy.txt", "w") as fh:
    fh.write(DOUBLET_POLICY_LINE + "\n")

# Logging
logger = logging.getLogger("step06")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler(sys.stdout)
fh = logging.FileHandler(LOGS / "06_QC.log", mode="w")
fmt = logging.Formatter("%(asctime)s  %(levelname)s  %(message)s")
ch.setFormatter(fmt)
fh.setFormatter(fmt)
logger.addHandler(ch)
logger.addHandler(fh)

logger.info("=== Step 06: Read, QC, Scrublet audit, merge ===")
for k, v in QC_THRESHOLDS.items():
    logger.info(f"  {k} = {v}")
logger.info(f"  doublet_policy = {DOUBLET_POLICY}: {DOUBLET_POLICY_LINE}")
logger.info(f"  peak RSS at start: {peak_mem_gb():.2f} GB")

# ---------------------------------------------------------------------------
# Load cohort / metadata
# ---------------------------------------------------------------------------
lib_meta = pd.read_csv(META / "04A_GSE231906_thymus_library_metadata.csv")
reg_cohort = pd.read_csv(META / "05B_final_regression_cohort.csv")

# Only keep donors that are in the regression cohort
reg_donors = set(reg_cohort["donor_id"])
lib_meta = lib_meta[lib_meta["donor_id"].isin(reg_donors)].copy()
logger.info(f"Libraries to read: {len(lib_meta)} across {lib_meta['donor_id'].nunique()} donors")


# ---------------------------------------------------------------------------
# Frozen task labels come from Step 05 (never re-derived from age here)
# ---------------------------------------------------------------------------
bin_lookup_05 = {}
fem_lookup_05 = {}
bin05_file = META / "05C_final_binary_classification_cohort.csv"
if bin05_file.exists():
    _bc = pd.read_csv(bin05_file).astype({"donor_id": str})
    bin_lookup_05 = dict(zip(_bc["donor_id"], _bc["label_binary"]))
fem05_file = META / "05G_female_only_sensitivity_cohort.csv"
if fem05_file.exists():
    _fc = pd.read_csv(fem05_file).astype({"donor_id": str})
    fem_lookup_05 = dict(zip(_fc["donor_id"], _fc["label_binary"]))


# ---------------------------------------------------------------------------
# Per-library read + QC + Scrublet (NO per-library gene filtering)
# ---------------------------------------------------------------------------
all_adatas = []
scrublet_rows = []
sensitivity_rows = []
count_rows_by_lib = []

for idx, row in lib_meta.iterrows():
    donor_id = str(row["donor_id"])
    gsm_id = str(row["gsm_id"])
    lib_name = str(row["library_prefix"])
    mtx = Path(row["matrix_path"])
    feat = Path(row["features_path"])
    barc = Path(row["barcodes_path"])

    logger.info(f"Reading library {lib_name} ({donor_id}, {gsm_id}) ...")

    # --- Read 10x MTX triplet ---
    adata = sc.read_mtx(mtx).T          # transpose: cells x genes
    features_raw = pd.read_csv(feat, sep="\t", header=None,
                               names=["gene_id", "gene_name", "gene_type"])
    barcodes = pd.read_csv(barc, sep="\t", header=None, names=["barcode_raw"])

    # P0-1: keep only "Gene Expression" features (drop intronic / ambiguous)
    gex_mask = features_raw["gene_type"] == "Gene Expression"
    features = features_raw[gex_mask].reset_index(drop=True)
    gex_idx = np.where(gex_mask.values)[0]
    if len(features) == 0:
        raise RuntimeError(f"No 'Gene Expression' features found in {feat}")
    adata = adata[:, gex_idx]          # filter matrix to GEX genes only
    assert adata.n_vars == len(features), (
        f"Matrix/feature count mismatch: adata.n_vars={adata.n_vars} != len(features)={len(features)}"
    )
    logger.info(f"  Gene Expression features: {len(features)} / {features_raw.shape[0]}")

    # P0-2: canonical var_names = Ensembl gene_id (stable molecular identifier);
    # gene_name stored as var["gene_symbol"] (display/mapping only).
    # var_names_make_unique() is NOT used: it appends -1/-2/-N and creates
    # fake symbols downstream.
    # P1 (2026-09-15): enforce, rather than assume, Ensembl ID uniqueness.
    dup_ids = features.loc[features["gene_id"].duplicated(keep=False), "gene_id"].unique()
    if len(dup_ids) > 0:
        raise RuntimeError(
            f"Duplicate Ensembl gene_id in {feat}: {dup_ids[:10].tolist()} "
            f"({len(dup_ids)} total); resolve feature annotation before merging."
        )
    adata.var_names = features["gene_id"].values
    adata.var["gene_id"] = features["gene_id"].values
    adata.var["gene_symbol"] = features["gene_name"].values

    # Keep the matrix sparse (CSR) for everything except Scrublet.
    if not sparse.issparse(adata.X):
        adata.X = sparse.csr_matrix(adata.X)
    else:
        adata.X = adata.X.tocsr()

    # --- Unique cell barcodes: donor_id__gsm_id__barcode_original ---
    bc_raw = barcodes["barcode_raw"].astype(str).values
    cell_uid = np.array([f"{donor_id}__{gsm_id}__{bc}" for bc in bc_raw], dtype=object)
    adata.obs_names = cell_uid
    adata.obs["cell_uid"] = cell_uid
    adata.obs["barcode_original"] = bc_raw

    # --- Annotate obs from 04A metadata ---
    adata.obs["donor_id"] = donor_id
    adata.obs["gsm_id"] = gsm_id
    adata.obs["library_name"] = lib_name
    adata.obs["age_years"] = float(row.get("age_years", np.nan))
    adata.obs["sex"] = str(row.get("sex_clean", row.get("sex", "unknown")))
    adata.obs["platform"] = str(row.get("platform_id", "GPL24676"))
    adata.obs["preparation"] = str(row.get("cell_type_original", row.get("preparation_group", "")))
    adata.obs["cohort"] = "GSE231906"

    age = adata.obs["age_years"].iloc[0]
    # P0-3: label_binary from frozen 05C/05G cohort (mid-age → NaN)
    adata.obs["label_binary"] = bin_lookup_05.get(donor_id, np.nan)
    adata.obs["in_binary_cohort"] = donor_id in bin_lookup_05
    adata.obs["in_female_sensitivity"] = donor_id in fem_lookup_05

    # --- Basic QC metrics on the COMPLETE gene universe (no gene filter) ---
    # var_names are Ensembl IDs; MT- detection must use gene_symbol.
    adata.var["mt"] = adata.var["gene_symbol"].str.startswith("MT-")
    if adata.var["mt"].sum() == 0:
        # Fallback: gene_name column may hold symbols; try both
        adata.var["mt"] = features["gene_name"].str.startswith("MT-").values
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)

    if "pct_counts_mt" not in adata.obs.columns:
        adata.obs["pct_counts_mt"] = 0.0

    adata.obs["total_counts"] = np.asarray(adata.X.sum(axis=1)).flatten()
    adata.obs["n_genes"] = np.asarray((adata.X > 0).sum(axis=1)).flatten()

    n_before = adata.n_obs
    logger.info(f"  {lib_name}: {n_before} cells x {adata.n_vars} genes (raw, sparse CSR), "
                f"peak RSS {peak_mem_gb():.2f} GB")

    # --- Per-library Scrublet (independent audit) ---
    # P1 (2026-09-15): sparse-first. Modern Scrublet accepts a CSR count
    # matrix; densify only if the sparse path raises (avoids a large dense
    # copy when not required). Peak memory around densification is logged.
    def _fit_scrublet(X_scrub):
        # PUBLIC API only (the private _LogTransformer no longer exists).
        scrublet = scr.Scrublet(X_scrub, n_neighbors=15)
        scores, preds = scrublet.scrub_doublets()
        return scrublet, scores, preds

    try:
        mem_before = peak_mem_gb()
        try:
            X_csr = adata.X.tocsr() if sparse.issparse(adata.X) else sparse.csr_matrix(adata.X)
            scrub, doublet_scores, predicted_doublets = _fit_scrublet(X_csr)
            logger.info(
                f"  Scrublet sparse-CSR path {adata.X.shape}; peak RSS "
                f"{peak_mem_gb():.2f} GB (before {mem_before:.2f} GB)"
            )
        except Exception as sparse_err:  # noqa: BLE001
            logger.info(
                f"  Scrublet sparse path failed ({sparse_err}); falling back to dense"
            )
            X_dense = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
            mem_after_dense = peak_mem_gb()
            logger.info(
                f"  Scrublet dense matrix {X_dense.shape} "
                f"({X_dense.nbytes / 1e9:.2f} GB); peak RSS {mem_after_dense:.2f} GB "
                f"(before densify {mem_before:.2f} GB)"
            )
            scrub, doublet_scores, predicted_doublets = _fit_scrublet(X_dense)
            del X_dense

        adata.obs["scrublet_score"] = doublet_scores
        adata.obs["predicted_doublet"] = predicted_doublets.astype(int)

        # Find threshold from scrublet's auto detection (if available)
        auto_thresh = float(getattr(scrub, "threshold_", float("nan")))
        if not np.isfinite(auto_thresh):
            auto_thresh = QC_THRESHOLDS["scrublet_doublet_score_threshold"]

        scrub_pct = 100.0 * predicted_doublets.mean()
        logger.info(
            f"  Scrublet [{lib_name}]: {predicted_doublets.sum()} doublets / {n_before} cells "
            f"({scrub_pct:.2f}%), auto_threshold={auto_thresh:.4f}, "
            f"peak RSS {peak_mem_gb():.2f} GB"
        )

        scrublet_rows.append({
            "library_prefix": lib_name,
            "donor_id": donor_id,
            "gsm_id": gsm_id,
            "n_cells": n_before,
            "n_predicted_doublets": int(predicted_doublets.sum()),
            "pct_doublets": round(scrub_pct, 2),
            "auto_threshold": round(auto_thresh, 4),
            "scrublet_status": "SUCCESS",
            "scrublet_error": "",
        })

        adata.obs["scrublet_status"] = "SUCCESS"
        adata.obs["scrublet_error"] = ""

        # Sensitivity of the doublet call across alternative thresholds
        # (auto threshold + a spread of fixed thresholds).  record-only audit.
        for label, thr in [
            ("auto", auto_thresh),
            (0.20, 0.20),
            (0.25, 0.25),
            (0.30, 0.30),
            (0.50, 0.50),
        ]:
            if label == "auto":
                n_dbl = int(predicted_doublets.sum())
            else:
                n_dbl = int((doublet_scores >= thr).sum())
            sensitivity_rows.append({
                "library_prefix": lib_name,
                "donor_id": donor_id,
                "gsm_id": gsm_id,
                "n_cells": n_before,
                "threshold": str(label),
                "n_doublets": n_dbl,
                "pct_doublets": round(100.0 * n_dbl / n_before, 3),
            })
    except Exception as e:
        # P0-4: Scrublet failure → float NaN (not 0) to distinguish from
        # zero-score passed QC.  pd.NA is NOT used (dtype/serialization issues).
        adata.obs["scrublet_score"] = np.float64(np.nan)
        adata.obs["predicted_doublet"] = np.float64(np.nan)
        adata.obs["scrublet_status"] = "FAILED"
        adata.obs["scrublet_error"] = str(e)[:500]
        scrublet_rows.append({
            "library_prefix": lib_name,
            "donor_id": donor_id,
            "gsm_id": gsm_id,
            "n_cells": n_before,
            "n_predicted_doublets": np.nan,
            "pct_doublets": np.nan,
            "auto_threshold": np.nan,
            "scrublet_status": "FAILED",
            "scrublet_error": str(e)[:500],
        })
        for label in ["auto", 0.20, 0.25, 0.30, 0.50]:
            sensitivity_rows.append({
                "library_prefix": lib_name,
                "donor_id": donor_id,
                "gsm_id": gsm_id,
                "n_cells": n_before,
                "threshold": str(label),
                "n_doublets": np.nan,
                "pct_doublets": np.nan,
            })

    count_rows_by_lib.append({
        "library_prefix": lib_name,
        "donor_id": donor_id,
        "gsm_id": gsm_id,
        "before_basic_QC": n_before,
    })

    all_adatas.append(adata)

logger.info(f"All libraries read; peak RSS {peak_mem_gb():.2f} GB")

# ---------------------------------------------------------------------------
# Concatenate all libraries (align gene symbols; keep complete gene universe)
# ---------------------------------------------------------------------------
logger.info(f"Concatenating {len(all_adatas)} libraries with join='outer' "
            f"(gene-symbol alignment) ...")
adata_all = ad.concat(all_adatas, join="outer", merge="unique")
adata_all.obs_names_make_unique()
logger.info(f"Total cells after concat: {adata_all.n_obs}; genes: {adata_all.n_vars}; "
            f"peak RSS {peak_mem_gb():.2f} GB")

# Raw counts layer BEFORE any further filtering (downstream Steps 07/08 use it).
adata_all.layers["counts"] = adata_all.X.copy()

# Free memory: per-library adatas are no longer needed.
del all_adatas

# Recompute the mt flag on the merged gene universe (union of features).
adata_all.var["mt"] = adata_all.var["gene_symbol"].str.startswith("MT-")
sc.pp.calculate_qc_metrics(adata_all, qc_vars=["mt"], inplace=True)
adata_all.obs["n_genes"] = adata_all.obs["n_genes_by_counts"].astype(int)
adata_all.obs["total_counts"] = adata_all.obs["total_counts"].astype(float)

# P0-3: task labels are FROZEN in Step 05 — do NOT re-derive from age here.
# label_binary / in_binary_cohort / in_female_sensitivity were set per-library
# from 05C/05G during the read loop and persist through concat.

# Freeze policy + thresholds into .uns for traceability.
adata_all.uns["doublet_policy"] = DOUBLET_POLICY
adata_all.uns["doublet_policy_line"] = DOUBLET_POLICY_LINE
adata_all.uns["qc_thresholds"] = QC_THRESHOLDS

# ---------------------------------------------------------------------------
# Global QC thresholds (basic QC – applied to all cells)
# ---------------------------------------------------------------------------
min_genes = QC_THRESHOLDS["min_genes_by_counts"]
max_genes = QC_THRESHOLDS["max_genes_by_counts"]
min_total = QC_THRESHOLDS["min_total_counts"]
max_total = QC_THRESHOLDS["max_total_counts"]
max_mt = QC_THRESHOLDS["max_pct_counts_mt"]

n_before_qc = adata_all.n_obs
logger.info(f"Before basic QC: {n_before_qc} cells, {adata_all.n_vars} genes, "
            f"peak RSS {peak_mem_gb():.2f} GB")

# --- Gene filtering NOW (after concat, on the merged object) ---
sc.pp.filter_genes(adata_all, min_cells=3)
adata_all.var["gene_symbol"] = adata_all.var["gene_symbol"].astype(str)
logger.info(f"After filter_genes(min_cells=3) on merged object: {adata_all.n_vars} genes")

# P0-2b: gene_id / symbol audit — do NOT merge symbols here; audit only.
#   * Are Ensembl gene_ids unique?  (canonical identifier — must be)
#   * Is each gene_symbol unique?     (may have 1→many Ensembl IDs; report it)
# NOTE: var already carries a real "gene_id" COLUMN (dropped here to avoid
# duplicate-column collisions after reset_index().rename({"index": "gene_id"})).
var_id_sym = pd.DataFrame({
    "gene_id": adata_all.var_names.values,
    "gene_symbol": adata_all.var["gene_symbol"].astype(str).values,
})
n_ids = adata_all.n_vars
n_unique_ids = var_id_sym["gene_id"].nunique()
n_unique_symbols = var_id_sym["gene_symbol"].nunique()
sym_to_ids = var_id_sym.groupby("gene_symbol")["gene_id"].nunique()
multi_symbol = sym_to_ids[sym_to_ids > 1]
logger.info(
    f"  Gene identity audit: {n_unique_ids}/{n_ids} unique Ensembl IDs; "
    f"{n_unique_symbols} unique symbols; {len(multi_symbol)} symbols → >1 Ensembl ID"
)
mapping_df = (
    var_id_sym
    .groupby("gene_symbol")["gene_id"]
    .agg(lambda s: ";".join(sorted(s)))
    .reset_index()
    .rename(columns={"gene_id": "ensembl_gene_ids"})
)
mapping_df = mapping_df[["gene_symbol", "ensembl_gene_ids"]].copy()
mapping_df["n_ensembl_ids"] = mapping_df["ensembl_gene_ids"].str.count(";") + 1
mapping_df.to_csv(META / "06_gene_id_symbol_mapping.csv", index=False)
if len(multi_symbol) > 0:
    multi_df = (
        var_id_sym
        .merge(multi_symbol.rename("n_ensembl_ids"), on="gene_symbol")
    )
    multi_df.to_csv(META / "06_symbol_multi_ensembl_audit.csv", index=False)
    logger.info(
        f"  WARNING: {len(multi_symbol)} gene symbols map to >1 Ensembl ID "
        f"(saved 06_symbol_multi_ensembl_audit.csv); NOT merged silently"
    )

# Basic QC cell filter (descriptive; doublets are NOT part of this filter)
sc.pp.filter_cells(adata_all, min_genes=min_genes)
sc.pp.filter_cells(adata_all, max_genes=max_genes)
sc.pp.filter_cells(adata_all, min_counts=min_total)
sc.pp.filter_cells(adata_all, max_counts=max_total)
adata_all = adata_all[adata_all.obs["pct_counts_mt"] <= max_mt].copy()

n_after_qc = adata_all.n_obs
logger.info(f"After basic QC: {n_after_qc} cells (dropped {n_before_qc - n_after_qc}); "
            f"peak RSS {peak_mem_gb():.2f} GB")

# ---------------------------------------------------------------------------
# Per-library / per-donor count summaries (before & after QC)
# ---------------------------------------------------------------------------
count_by_lib = pd.DataFrame(count_rows_by_lib)
before_qc = count_by_lib.groupby(["library_prefix", "donor_id", "gsm_id"])["before_basic_QC"].sum().reset_index()
before_qc.rename(columns={"before_basic_QC": "before_QC"}, inplace=True)

after_qc = (
    adata_all.obs.groupby(["library_name", "donor_id", "gsm_id"])
    .size()
    .reset_index(name="after_basic_QC")
)
after_qc.rename(columns={"library_name": "library_prefix"}, inplace=True)

scrub_df = pd.DataFrame(scrublet_rows)

qc_lib = before_qc.merge(after_qc, on=["library_prefix", "donor_id", "gsm_id"], how="outer")
qc_lib = qc_lib.merge(
    scrub_df[["library_prefix", "n_predicted_doublets", "pct_doublets"]],
    on="library_prefix", how="left",
)
qc_lib["after_final_QC"] = qc_lib["after_basic_QC"]  # doublets are record-only
qc_lib = qc_lib.sort_values("library_prefix").reset_index(drop=True)
qc_lib.to_csv(META / "06_QC_cell_counts_by_library.csv", index=False)

# --- Retention rate by library (new) ---
qc_lib_ret = qc_lib[["library_prefix", "donor_id", "gsm_id", "before_QC", "after_basic_QC"]].copy()
qc_lib_ret["after_QC"] = qc_lib_ret["after_basic_QC"].fillna(0).astype(int)
qc_lib_ret["before_QC"] = qc_lib_ret["before_QC"].fillna(0).astype(int)
qc_lib_ret["retention_rate_pct"] = (
    100.0 * qc_lib_ret["after_QC"] / qc_lib_ret["before_QC"].replace(0, np.nan)
).round(2)
qc_lib_ret = qc_lib_ret.drop(columns=["after_basic_QC"])
qc_lib_ret.to_csv(META / "06_QC_retention_rate_by_library.csv", index=False)
logger.info(f"Retention rate by library: saved to {META / '06_QC_retention_rate_by_library.csv'}")

qc_donor = (
    adata_all.obs.groupby("donor_id")
    .agg(
        after_QC=("library_name", "size"),
        pct_counts_mt_mean=("pct_counts_mt", "mean"),
        n_genes_mean=("n_genes", "mean"),
        total_counts_mean=("total_counts", "mean"),
    )
    .round(2)
    .reset_index()
)
before_donor = lib_meta.groupby("donor_id").size().reset_index(name="n_libraries")
qc_donor = qc_donor.merge(before_donor, on="donor_id", how="left")
qc_donor["before_QC"] = (
    count_by_lib.groupby("donor_id")["before_basic_QC"].sum().reindex(qc_donor["donor_id"]).values
)
qc_donor = qc_donor.sort_values("donor_id").reset_index(drop=True)
qc_donor.to_csv(META / "06_QC_cell_counts_by_donor.csv", index=False)

# --- Retention rate by donor (new) ---
qc_donor_ret = qc_donor[["donor_id", "n_libraries", "before_QC", "after_QC"]].copy()
qc_donor_ret["retention_rate_pct"] = (
    100.0 * qc_donor_ret["after_QC"] / qc_donor_ret["before_QC"].replace(0, np.nan)
).round(2)
qc_donor_ret.to_csv(META / "06_QC_retention_rate_by_donor.csv", index=False)
logger.info(f"Retention rate by donor: saved to {META / '06_QC_retention_rate_by_donor.csv'}")

scrub_df.sort_values("library_prefix").to_csv(META / "06_scrublet_audit.csv", index=False)
pd.DataFrame(sensitivity_rows).sort_values(["library_prefix", "threshold"]).to_csv(
    META / "06_doublet_sensitivity_summary.csv", index=False
)

logger.info(f"Cell counts by library: saved to {META / '06_QC_cell_counts_by_library.csv'}")
logger.info(f"Cell counts by donor: saved to {META / '06_QC_cell_counts_by_donor.csv'}")
logger.info(f"Scrublet audit: saved to {META / '06_scrublet_audit.csv'}")
logger.info(f"Doublet sensitivity: saved to {META / '06_doublet_sensitivity_summary.csv'}")
logger.info(f"Doublet policy: saved to {META / '06_doublet_policy.txt'}")

# ---------------------------------------------------------------------------
# Save QC h5ad
# ---------------------------------------------------------------------------
adata_all.write_h5ad(FILTERED / "06_GSE231906_thymus_QC.h5ad")
logger.info(f"QC h5ad saved: {FILTERED / '06_GSE231906_thymus_QC.h5ad'}")
logger.info(f"Peak RSS at save: {peak_mem_gb():.2f} GB")

# ---------------------------------------------------------------------------
# QC figures
# ---------------------------------------------------------------------------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # --- 06_QC_violin.pdf (existing): QC metrics by donor ---
    fig, axes = plt.subplots(1, 5, figsize=(22, 4))
    for i, col in enumerate(["n_genes", "total_counts", "pct_counts_mt", "scrublet_score"]):
        ax = axes[i]
        sc.pl.violin(adata_all, col, groupby="donor_id", rotation=45, ax=ax, show=False)
        ax.set_title(col)
    axes[-1].axis("off")
    fig.tight_layout()
    fig.savefig(FIGURES / "06_QC_violin.pdf", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"QC violin saved: {FIGURES / '06_QC_violin.pdf'}")

    # --- 06_QC_scatter.pdf (existing): metric scatter pairs ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].scatter(adata_all.obs["n_genes"], adata_all.obs["total_counts"], s=1, alpha=0.3)
    axes[0].set_xlabel("n_genes"); axes[0].set_ylabel("total_counts")
    axes[1].scatter(adata_all.obs["total_counts"], adata_all.obs["pct_counts_mt"], s=1, alpha=0.3)
    axes[1].set_xlabel("total_counts"); axes[1].set_ylabel("pct_counts_mt")
    axes[2].scatter(adata_all.obs["n_genes"], adata_all.obs["pct_counts_mt"], s=1, alpha=0.3)
    axes[2].set_xlabel("n_genes"); axes[2].set_ylabel("pct_counts_mt")
    fig.tight_layout()
    fig.savefig(FIGURES / "06_QC_scatter.pdf", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"QC scatter saved: {FIGURES / '06_QC_scatter.pdf'}")

    # --- 06_QC_retention_by_donor.pdf (new): bar chart of retention rate ---
    ret_donor = pd.read_csv(META / "06_QC_retention_rate_by_donor.csv").sort_values(
        "retention_rate_pct", ascending=False
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    xpos = np.arange(len(ret_donor))
    bars = ax.bar(xpos, ret_donor["retention_rate_pct"], color="#4C72B0")
    ax.set_xticks(xpos)
    ax.set_xticklabels(ret_donor["donor_id"], rotation=45, ha="right")
    ax.set_ylabel("Cell retention rate (%)")
    ax.set_xlabel("Donor")
    ax.set_title("Per-donor cell retention rate after QC")
    ax.axhline(ret_donor["retention_rate_pct"].median(), color="grey", ls="--", lw=0.8,
               label=f"median = {ret_donor['retention_rate_pct'].median():.1f}%")
    ax.legend()
    for b, v in zip(bars, ret_donor["retention_rate_pct"]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.4, f"{v:.1f}",
                ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES / "06_QC_retention_by_donor.pdf", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Retention by donor saved: {FIGURES / '06_QC_retention_by_donor.pdf'}")

    # --- 06_scrublet_score_by_library.pdf (new): violin of scrublet score ---
    fig, ax = plt.subplots(figsize=(16, 5))
    sc.pl.violin(adata_all, "scrublet_score", groupby="library_name", rotation=90,
                 ax=ax, show=False)
    ax.set_title("Per-library Scrublet doublet scores (audit only, NOT filtered)")
    ax.set_xlabel("Library")
    ax.set_ylabel("Scrublet doublet score")
    fig.tight_layout()
    fig.savefig(FIGURES / "06_scrublet_score_by_library.pdf", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Scrublet score by library saved: {FIGURES / '06_scrublet_score_by_library.pdf'}")
except Exception as e:
    logger.warning(f"QC figure saving skipped: {e}")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
n_pred_total = int(adata_all.obs["predicted_doublet"].sum())
pct_pred_total = 100.0 * n_pred_total / adata_all.n_obs if adata_all.n_obs else 0.0
logger.info(f"\n=== Step 06 COMPLETE ===")
logger.info(f"Total donors: {adata_all.obs['donor_id'].nunique()}")
logger.info(f"Total cells after QC: {adata_all.n_obs}")
logger.info(f"Predicted doublets (record-only, NOT removed): {n_pred_total} ({pct_pred_total:.2f}%)")
logger.info(f"Peak RSS: {peak_mem_gb():.2f} GB")
logger.info(f"Anndata: {adata_all.shape[0]} cells x {adata_all.shape[1]} genes")
logger.info(f"Obs columns: {adata_all.obs.columns.tolist()}")
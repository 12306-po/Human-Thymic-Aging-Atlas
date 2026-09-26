#!/usr/bin/env python
"""Step 10: Prepare donor-level ML matrices (rev. 2).

P0 fixes per Codex review 2026-09-14:
  - P0-1: label_binary / in_binary_cohort / binary_task_group are read from the
    Step 05 frozen cohort files (05B/05C). The age_group re-derivation
    (`np.where(age < 18, ...)`) is removed; mid-age donors keep label_binary=NA.
  - P0-2: Gene x CellType features are built from the Step 08 *analysis*
    pseudobulk (eligible cells only, <donor>_<celltype> files under
    celltype_matrix/log_normalized/); composition features come from
    donor_celltype_proportion_all.csv (official denominator = all QC-passed
    cells) and only eligible celltype columns enter the matrix (Unknown/
    Innate_T_unresolved/SP_unresolved are NOT features, though already counted
    in the denominator).
  - Whole-thymus features still come from donor_matrix/log_normalized_combined
    (all QC cells; sample-level reference).

Outputs:
  04_machine_learning/dataset/
    feature_matrix_gene_celltype.csv.gz
    feature_matrix_whole_thymus.csv.gz
    feature_matrix_proportion.csv.gz
    feature_matrix_all_combined.csv.gz
    feature_dictionary.csv
    donor_metadata.csv
    feature_missingness_summary.csv, feature_type_summary.csv, feature_metadata.json
  10_results/logs/10_prepare_ML_matrix.log
"""
from __future__ import annotations

import json
import logging
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

PROJ = Path(os.environ["PROJ"])
META = PROJ / "01_raw_processing" / "metadata"
PSEUDO = PROJ / "02_pseudobulk"
OUT = PROJ / "04_machine_learning" / "dataset"
OUT.mkdir(parents=True, exist_ok=True)
LOGS = PROJ / "10_results" / "logs"
LOGS.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("step10")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler(sys.stdout)
fh = logging.FileHandler(LOGS / "10_prepare_ML_matrix.log", mode="w")
fmt = logging.Formatter("%(asctime)s  %(levelname)s  %(message)s")
ch.setFormatter(fmt)
fh.setFormatter(fmt)
logger.addHandler(ch)
logger.addHandler(fh)

logger.info("=== Step 10 (rev.2): Prepare donor-level ML matrices ===")

# ------------------------------------------------------------------
# Cohort / frozen labels (P0-1)
# ------------------------------------------------------------------
reg = pd.read_csv(META / "05B_final_regression_cohort.csv")
reg["donor_id"] = reg["donor_id"].astype(str)
donors = sorted(reg["donor_id"].tolist())
logger.info(f"Donors: {len(donors)}")

bin_cohort = pd.read_csv(META / "05C_final_binary_classification_cohort.csv")
bin_cohort["donor_id"] = bin_cohort["donor_id"].astype(str)
bin_lookup = bin_cohort.set_index("donor_id")["label_binary"].to_dict()
bin_group_lookup = bin_cohort.set_index("donor_id")["binary_task_group"].to_dict()
for d in donors:
    if d not in bin_lookup:
        logger.info(f"  donor {d}: not in binary cohort (label_binary=NA)")

# ------------------------------------------------------------------
# Whole-thymus log-normalized matrix (all QC cells)
# ------------------------------------------------------------------
wt = pd.read_csv(PSEUDO / "donor_matrix" / "log_normalized_combined.csv.gz", index_col=0)
wt.index = wt.index.astype(str)
wt = wt.loc[donors]
wt.columns = [f"WT_{g}" for g in wt.columns]
logger.info(f"Whole-thymus features: {wt.shape}")

# ------------------------------------------------------------------
# Composition features: official denominator (proportion_all), eligible cols only
# ------------------------------------------------------------------
prop_all = pd.read_csv(PSEUDO / "donor_celltype_proportion_all.csv", index_col=0)
prop_all.index = prop_all.index.astype(str)
# Eligible celltypes (those present in the analysis pseudobulk) only
analysis_ct = sorted({f.name.replace(".csv.gz", "").split("_", 1)[1]
                      for f in (PSEUDO / "celltype_matrix" / "log_normalized").glob("*.csv.gz")})
prop_cols = [c for c in prop_all.columns if c in analysis_ct]
logger.info(f"Composition features (eligible celltypes): {len(prop_cols)} / "
            f"{len(prop_all.columns)} columns of proportion_all")
prop = prop_all.loc[donors, prop_cols].copy()
prop.columns = [f"PROP_{c}" for c in prop.columns]
logger.info(f"Proportion features: {prop.shape}")

# ------------------------------------------------------------------
# Gene x CellType features from *analysis* pseudobulk (eligible cells)
# ------------------------------------------------------------------
gc_files = list((PSEUDO / "celltype_matrix" / "log_normalized").glob("*.csv.gz"))

def _parse_gc_stem(p: Path):
    s = p.name.replace(".csv.gz", "")  # "donor1_B" / "donor1_SP_CD4"
    donor, ct = s.split("_", 1)
    return donor, ct

celltypes = sorted({_parse_gc_stem(f)[1] for f in gc_files})
logger.info(f"Distinct celltypes with pseudobulk (analysis): {len(celltypes)} "
            f"-> {celltypes}")
logger.info(f"Celltype pseudobulk files: {len(gc_files)}")

gene_ct = {}
all_genes = set()
for f in gc_files:
    donor, ct = _parse_gc_stem(f)
    df = pd.read_csv(f, index_col=0)
    ser = df.iloc[:, 0]
    gene_ct[(donor, ct)] = ser
    all_genes.update(ser.index)
logger.info(f"Gene x CellType pairs loaded: {len(gene_ct)} (donor, celltype) combos")

all_genes = sorted(all_genes)
logger.info(f"Genes across celltype matrices: {len(all_genes)}")

blocks = []
feature_dict_rows = []
for ct in sorted(celltypes):
    mat = pd.DataFrame(index=donors, columns=all_genes, dtype=float)
    for donor in donors:
        if (donor, ct) in gene_ct:
            mat.loc[donor] = gene_ct[(donor, ct)].reindex(all_genes).values
    mat.columns = [f"{g}_{ct}" for g in mat.columns]
    blocks.append(mat)
    for g in all_genes:
        feature_dict_rows.append({
            "feature": f"{g}_{ct}",
            "feature_type": "gene_celltype",
            "gene": g,
            "celltype": ct,
            "source_matrix": "celltype_matrix",
            "is_whole_thymus": False,
            "is_composition": False,
        })
    del mat

gc = pd.concat(blocks, axis=1)
gc.index = gc.index.astype(str)
n_na = int(gc.isna().sum().sum())
logger.info(f"Gene x CellType matrix: {gc.shape} (NaN entries: {n_na}, "
            f"{100 * n_na / (gc.shape[0] * gc.shape[1]):.2f}%)")

# ------------------------------------------------------------------
# Outcome / metadata columns (frozen labels only)
# ------------------------------------------------------------------
meta_cols = pd.DataFrame(index=donors)
meta_cols["age_years"] = reg.set_index("donor_id")["age_years"].reindex(donors).astype(float)
meta_cols["sex"] = reg.set_index("donor_id")["sex"].reindex(donors).values
meta_cols["binary_task_group"] = (
    pd.Series(bin_group_lookup).reindex(donors).values)
meta_cols["label_binary"] = (
    pd.Series(bin_lookup).reindex(donors).values)   # NaN for mid-age donors
meta_cols["in_binary_cohort"] = meta_cols["label_binary"].notna()
logger.info("Frozen binary labels mapped from 05C "
            "(mid-age donors: label_binary=NA, in_binary_cohort=False)")

# ------------------------------------------------------------------
# Feature dictionary: whole-thymus + proportion entries too
# ------------------------------------------------------------------
for g in wt.columns:
    feature_dict_rows.append({
        "feature": g,
        "feature_type": "whole_thymus",
        "gene": g[3:],
        "celltype": "WT",
        "source_matrix": "donor_matrix",
        "is_whole_thymus": True,
        "is_composition": False,
    })
for c in prop.columns:
    feature_dict_rows.append({
        "feature": c,
        "feature_type": "proportion",
        "gene": None,
        "celltype": c[5:],
        "source_matrix": "composition",
        "is_whole_thymus": False,
        "is_composition": True,
    })

feature_dict = pd.DataFrame(feature_dict_rows)

# P1 (2026-09-16): fail fast on feature-key collisions. Downstream Steps
# 12/13/16/20 treat `feature` as the authoritative key; a duplicate (e.g. a
# gene symbol that collides with a proportion key) would silently mis-map rows.
if feature_dict["feature"].duplicated().any():
    dup = feature_dict.loc[
        feature_dict["feature"].duplicated(keep=False)
    ].sort_values("feature")
    dup.to_csv(OUT / "feature_dictionary_duplicate_keys.csv", index=False)
    raise RuntimeError(
        "Feature key collision detected in feature_dictionary.csv; see "
        "feature_dictionary_duplicate_keys.csv. Use an unambiguous delimiter "
        "(e.g. GENE@@CellType, WT@@GENE, PROP@@CellType) or fix the generator."
    )
feature_dict.to_csv(OUT / "feature_dictionary.csv", index=False)
logger.info(f"Feature dictionary saved: {OUT / 'feature_dictionary.csv'} "
            f"({len(feature_dict)} rows)")

# Combined feature matrix (without outcome)
all_features = pd.concat([wt, gc, prop], axis=1)
assert all_features.columns.is_unique, (
    "Combined feature matrix has duplicated columns "
    "(whole-thymus / gene-celltype / proportion key collision)")
all_features.to_csv(OUT / "feature_matrix_all_combined.csv.gz", compression="gzip")

wt.to_csv(OUT / "feature_matrix_whole_thymus.csv.gz", compression="gzip")
prop.to_csv(OUT / "feature_matrix_proportion.csv.gz", compression="gzip")
gc.to_csv(OUT / "feature_matrix_gene_celltype.csv.gz", compression="gzip")

# Descriptive missingness / feature-type summaries (NOT used as selection)
miss_summary = pd.DataFrame({
    "feature": all_features.columns,
    "overall_missing_fraction": all_features.isna().mean().values,
    "overall_variance": all_features.var(axis=0, skipna=True).values,
})
miss_summary = miss_summary.merge(
    feature_dict[["feature", "feature_type"]], on="feature", how="left")
miss_summary.to_csv(OUT / "feature_missingness_summary.csv", index=False)
type_summary = (feature_dict.groupby("feature_type").size()
                .rename("n_features").reset_index())
type_summary.to_csv(OUT / "feature_type_summary.csv", index=False)
logger.info("Descriptive missingness / type summaries saved")

meta_cols.to_csv(OUT / "donor_metadata.csv", index=True)

feature_meta = {
    "n_donors": len(donors),
    "n_genes": len(all_genes),
    "n_celltypes": len(celltypes),
    "n_features_gene_celltype": gc.shape[1],
    "n_features_whole_thymus": wt.shape[1],
    "n_features_proportion": prop.shape[1],
    "n_features_total": all_features.shape[1],
    "min_cells_per_donor_celltype": 20,
    "normalization": "log1p(CPM) per donor-celltype pseudobulk",
    "analysis_cells": "annotation_analysis_eligible only",
    "composition_denominator": "all QC-passed cells (proportion_all)",
    "labels_frozen_from": "05B/05C",
}
with open(OUT / "feature_metadata.json", "w") as fh:
    json.dump(feature_meta, fh, indent=2)
logger.info(f"Feature metadata: {feature_meta}")

logger.info(f"\n=== Step 10 (rev.2) COMPLETE ===")
logger.info(f"Whole-thymus: {wt.shape}")
logger.info(f"Proportion: {prop.shape}")
logger.info(f"Gene x CellType: {gc.shape}")
logger.info(f"Combined: {all_features.shape}")
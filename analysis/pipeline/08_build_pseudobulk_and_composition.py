#!/usr/bin/env python
"""Step 08: Build donor-level pseudobulk and cell-type composition (rev. 2).

P0/P1 fixes per Codex review 2026-09-14:
  - Main Gene x CellType pseudobulk (raw/CPM/log1pCPM) is built from
    annotation_analysis_eligible cells only (Unknown / Innate_T_unresolved /
    SP_unresolved excluded from the feature matrix).
  - Whole-thymus donor matrices keep ALL QC-passed cells (sample-level
    reference; not restricted to eligible cells).
  - Composition denominator is formally defined as:

        prop(celltype, donor) = n_cells(celltype, donor)
                                / n_all_QC_passed_cells(donor)

    Unknown/unresolved cells stay in the denominator (they do not inflate
    eligible celltype proportions).
  - Three outputs:
      donor_celltype_proportion_all.csv      (denominator = all QC cells, official)
      donor_celltype_proportion_analysis.csv (eligible-only renormalized, sensitivity)
      pseudobulk_qc_summary_all.csv          (all QC cells x all broad celltypes)
      pseudobulk_qc_summary_analysis.csv     (eligible cells x eligible celltypes)
  - P0 (2026-09-15): pseudobulk gene universe is the audited unique gene
    SYMBOL matrix, not raw Ensembl IDs. Ensembl gene_id is the canonical raw
    molecular identifier (kept in the h5ad); gene symbol is the biological
    feature identity used from here onward. Only symbols mapping to exactly
    ONE Ensembl ID enter the official symbol-level universe:
      * 08_symbol_level_gene_universe.csv  (unique symbol -> single Ensembl ID)
      * 08_excluded_ambiguous_symbols.csv  (symbol -> >1 Ensembl ID, EXCLUDED)
    Ambiguous symbols are excluded, never summed, and never renamed with
    var_names_make_unique().

Inputs:
  01_raw_processing/filtered/07_GSE231906_thymus_annotated.h5ad

Pre-declared parameter (v2.1):
  min_cells_per_donor_celltype = 20  (fixed; does not change post-hoc)

Outputs:
  02_pseudobulk/celltype_matrix/{raw_counts,normalized,log_normalized}/
    <donor>_<celltype>.csv.gz
  02_pseudobulk/donor_matrix/{raw,normalized,log_normalized}.csv.gz
  02_pseudobulk/donor_celltype_proportion_all.csv
  02_pseudobulk/donor_celltype_proportion_analysis.csv
  02_pseudobulk/pseudobulk_qc_summary_all.csv
  02_pseudobulk/pseudobulk_qc_summary_analysis.csv
  09_figures/08_composition_barplot.pdf
  10_results/logs/08_pseudobulk.log
"""
from __future__ import annotations

import logging
import os
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

PROJ = Path(os.environ["PROJ"])
META = PROJ / "01_raw_processing" / "metadata"
FILTERED = PROJ / "01_raw_processing" / "filtered"
PSEUDO = PROJ / "02_pseudobulk"
FIGURES = PROJ / "09_figures"
LOGS = PROJ / "10_results" / "logs"

for d in [PSEUDO / "celltype_matrix" / "raw_counts",
          PSEUDO / "celltype_matrix" / "normalized",
          PSEUDO / "celltype_matrix" / "log_normalized",
          PSEUDO / "donor_matrix", FIGURES, LOGS]:
    d.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("step08")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler(sys.stdout)
fh = logging.FileHandler(LOGS / "08_pseudobulk.log", mode="w")
fmt = logging.Formatter("%(asctime)s  %(levelname)s  %(message)s")
ch.setFormatter(fmt)
fh.setFormatter(fmt)
logger.addHandler(ch)
logger.addHandler(fh)

MIN_CELLS = 20
logger.info(f"=== Step 08: Build pseudobulk + composition (rev. 2) ===")
logger.info(f"min_cells_per_donor_celltype = {MIN_CELLS}")

adata = sc.read_h5ad(FILTERED / "07_GSE231906_thymus_annotated.h5ad")
logger.info(f"Annotated h5ad: {adata.n_obs} cells x {adata.n_vars} genes")

if "annotation_analysis_eligible" not in adata.obs.columns:
    raise KeyError(
        "obs['annotation_analysis_eligible'] missing — Step 07 rev.3 must run "
        "before Step 08 rev.2.")

# Ensure raw counts layer is available
if "counts" in adata.layers:
    raw_X = adata.layers["counts"]
elif sparse.issparse(adata.X):
    raw_X = adata.X.copy()
else:
    raw_X = adata.X.copy()

obs = adata.obs
donors = obs["donor_id"].astype(str)
celltypes = obs["broad_celltype"].astype(str)
eligible = obs["annotation_analysis_eligible"].astype(bool)

# ------------------------------------------------------------------
# P0 (2026-09-15): audited symbol-level gene universe.
# var_names = Ensembl gene_id (canonical molecular ID); var["gene_symbol"] =
# biological display ID. Build the audited unique-symbol mapping:
#   1 symbol -> 1 Ensembl ID   : retained (official symbol-level feature)
#   1 symbol -> >1 Ensembl ID  : EXCLUDED (never summed / never make-unique)
# ------------------------------------------------------------------
gene_id_arr = adata.var_names.astype(str).to_numpy()
# P1 (2026-09-16): astype(str) would silently turn missing symbols into "nan".
# Detect and exclude invalid/placeholder symbols BEFORE building the universe.
_raw_symbol = adata.var["gene_symbol"]
gene_symbol_arr = _raw_symbol.astype(str).to_numpy()
vm = pd.DataFrame({"ensembl_gene_id": gene_id_arr, "gene_symbol": gene_symbol_arr})

_BAD_SYMBOLS = {"", "nan", "none", "na", "n/a", "unknown", "?"}
bad_mask = (
    vm["gene_symbol"].astype(str).str.strip().str.lower().isin(_BAD_SYMBOLS)
    | _raw_symbol.isna().to_numpy()
)
invalid_symbol_df = vm.loc[bad_mask].copy()
invalid_symbol_df.to_csv(PSEUDO / "08_excluded_invalid_gene_symbols.csv", index=False)
if len(invalid_symbol_df):
    logger.warning(
        "Excluded %d features with missing/placeholder gene symbols -> %s",
        len(invalid_symbol_df), PSEUDO / "08_excluded_invalid_gene_symbols.csv")
vm = vm.loc[~bad_mask].copy()

sym_counts = vm.groupby("gene_symbol")["ensembl_gene_id"].nunique()
unique_symbols = set(sym_counts[sym_counts == 1].index)
ambiguous_symbols = sorted(sym_counts[sym_counts > 1].index)

symbol_universe = (vm[vm["gene_symbol"].isin(unique_symbols)]
                   .sort_values("gene_symbol")
                   .reset_index(drop=True))
symbol_universe.to_csv(PSEUDO / "08_symbol_level_gene_universe.csv", index=False)
logger.info(f"Symbol-level gene universe: {len(symbol_universe)} unique symbols "
            f"(1 symbol -> 1 Ensembl ID) -> "
            f"{PSEUDO / '08_symbol_level_gene_universe.csv'}")

if ambiguous_symbols:
    amb_rows = []
    for sym in ambiguous_symbols:
        ids = vm.loc[vm["gene_symbol"] == sym, "ensembl_gene_id"].tolist()
        amb_rows.append({
            "gene_symbol": sym,
            "ensembl_gene_ids": ";".join(ids),
            "n_ensembl_ids": len(ids),
        })
    pd.DataFrame(amb_rows).to_csv(PSEUDO / "08_excluded_ambiguous_symbols.csv",
                                  index=False)
    logger.warning(f"Excluded {len(ambiguous_symbols)} ambiguous symbols "
                   f"(1 symbol -> >1 Ensembl ID) from official ML universe -> "
                   f"{PSEUDO / '08_excluded_ambiguous_symbols.csv'}")
else:
    pd.DataFrame(columns=["gene_symbol", "ensembl_gene_ids", "n_ensembl_ids"]).to_csv(
        PSEUDO / "08_excluded_ambiguous_symbols.csv", index=False)

# column mask + symbol names for the symbol-level matrices
keep_mask = np.isin(gene_symbol_arr, list(unique_symbols))
gene_names = gene_symbol_arr[keep_mask].tolist()
assert len(gene_names) == len(symbol_universe), (
    "symbol universe size mismatch after unique-symbol masking")

donors_list = sorted(obs["donor_id"].unique())
cts_all = sorted(obs["broad_celltype"].unique())
cts_eligible = sorted(obs.loc[eligible, "broad_celltype"].unique())
logger.info(f"All broad celltypes: {cts_all}")
logger.info(f"Eligible celltypes: {cts_eligible}")
logger.info(f"Eligible cells: {int(eligible.sum())} / {adata.n_obs} "
            f"({100 * eligible.mean():.2f}%)")

# ------------------------------------------------------------------
# 1. Composition with formally defined denominators
# ------------------------------------------------------------------
cell_df = pd.DataFrame({
    "donor_id": obs["donor_id"].astype(str).values,
    "broad_celltype": celltypes.values,
    "eligible": eligible.values,
})
n_total_per_donor = cell_df.groupby("donor_id").size().rename("n_all_qc_cells")
n_eligible_per_donor = (
    cell_df[cell_df["eligible"]].groupby("donor_id").size()
    .rename("n_eligible_cells"))

# --- ALL QC cells: official denominator = all QC-passed cells ---
ct_all = (cell_df.groupby(["donor_id", "broad_celltype"]).size()
          .reset_index(name="n_cells"))
ct_all = ct_all.merge(n_total_per_donor, on="donor_id", how="left")
ct_all["fraction"] = ct_all["n_cells"] / ct_all["n_all_qc_cells"]
ct_all["pass_min_cells"] = ct_all["n_cells"] >= MIN_CELLS

# --- ANALYSIS: eligible cells only, renormalized within eligible set ---
ct_an = (cell_df[cell_df["eligible"]]
         .groupby(["donor_id", "broad_celltype"]).size()
         .reset_index(name="n_cells"))
ct_an = ct_an.merge(n_eligible_per_donor, on="donor_id", how="left")
ct_an["fraction"] = ct_an["n_cells"] / ct_an["n_eligible_cells"]
ct_an["pass_min_cells"] = ct_an["n_cells"] >= MIN_CELLS

# Wide QC summaries
def _qc_wide(ct, celltype_list):
    qc = ct.pivot_table(index="donor_id", columns="broad_celltype",
                        values=["n_cells", "pass_min_cells", "fraction"],
                        fill_value=0)
    qc = qc.reindex(columns=pd.MultiIndex.from_product(
        [["n_cells", "pass_min_cells", "fraction"], celltype_list]))
    qc.columns = [f"{v}_{c}" for v, c in qc.columns]
    return qc

qc_all = _qc_wide(ct_all, cts_all)
qc_all.to_csv(PSEUDO / "pseudobulk_qc_summary_all.csv")
logger.info(f"QC summary (all QC cells) saved: "
            f"{PSEUDO / 'pseudobulk_qc_summary_all.csv'}")

qc_an = _qc_wide(ct_an, cts_eligible)
qc_an.to_csv(PSEUDO / "pseudobulk_qc_summary_analysis.csv")
logger.info(f"QC summary (eligible cells) saved: "
            f"{PSEUDO / 'pseudobulk_qc_summary_analysis.csv'}")

# Proportion tables (wide)
prop_all = (ct_all.pivot(index="donor_id", columns="broad_celltype",
                         values="fraction")
            .reindex(index=donors_list, columns=cts_all).fillna(0.0))
prop_all.to_csv(PSEUDO / "donor_celltype_proportion_all.csv")
logger.info(f"Composition (denominator = all QC cells, official) saved: "
            f"{PSEUDO / 'donor_celltype_proportion_all.csv'}")

prop_an = (ct_an.pivot(index="donor_id", columns="broad_celltype",
                       values="fraction")
           .reindex(index=donors_list, columns=cts_eligible).fillna(0.0))
prop_an.to_csv(PSEUDO / "donor_celltype_proportion_analysis.csv")
logger.info(f"Composition (eligible-only renormalized, sensitivity) saved: "
            f"{PSEUDO / 'donor_celltype_proportion_analysis.csv'}")

# ------------------------------------------------------------------
# 2. Pseudobulk: sum raw counts per (donor, eligible celltype)
#    passing min_cells. Only eligible cells enter the feature matrix.
# ------------------------------------------------------------------
pass_pairs = ct_an[ct_an["pass_min_cells"]].copy()
logger.info(f"Passing (donor, eligible celltype) pairs: "
            f"{len(pass_pairs)} / {len(ct_an)}")

eligible_mask_full = eligible.values

# 2a. Per-celltype pseudobulk matrices (eligible cells only)
for _, row in pass_pairs.iterrows():
    donor = str(row["donor_id"])
    ct = str(row["broad_celltype"])
    mask = (donors == donor) & (celltypes == ct) & eligible_mask_full
    sub = raw_X[mask.values]
    sum_vec = np.asarray(sub.sum(axis=0)).flatten()[keep_mask]

    prefix = f"{donor}_{ct}"

    # --- raw_counts ---
    pd.DataFrame({"gene": gene_names, "count": sum_vec}).to_csv(
        PSEUDO / "celltype_matrix" / "raw_counts" / f"{prefix}.csv.gz",
        index=False, compression="gzip")

    # --- normalised (CPM) ---
    total = sum_vec.sum()
    cpm = (sum_vec / total) * 1e6 if total > 0 else sum_vec
    pd.DataFrame({"gene": gene_names, "cpm": cpm}).to_csv(
        PSEUDO / "celltype_matrix" / "normalized" / f"{prefix}.csv.gz",
        index=False, compression="gzip")

    # --- log_normalised (log1p(CPM)) ---
    pd.DataFrame({"gene": gene_names, "log_cpm": np.log1p(cpm)}).to_csv(
        PSEUDO / "celltype_matrix" / "log_normalized" / f"{prefix}.csv.gz",
        index=False, compression="gzip")

logger.info(f"Wrote {len(pass_pairs)} celltype-level pseudobulk files "
            f"(eligible cells only)")

# 2b. Whole-thymus donor matrices — ALL QC-passed cells (sample-level reference)
for donor in donors_list:
    mask = donors == donor
    sub = raw_X[mask.values]
    sum_vec = np.asarray(sub.sum(axis=0)).flatten()[keep_mask]
    total = sum_vec.sum()

    pd.DataFrame({"gene": gene_names, "raw_count": sum_vec}).to_csv(
        PSEUDO / "donor_matrix" / f"{donor}_raw.csv.gz",
        index=False, compression="gzip")

    cpm = (sum_vec / total) * 1e6 if total > 0 else sum_vec
    pd.DataFrame({"gene": gene_names, "cpm": cpm}).to_csv(
        PSEUDO / "donor_matrix" / f"{donor}_normalized.csv.gz",
        index=False, compression="gzip")

    pd.DataFrame({"gene": gene_names, "log_cpm": np.log1p(cpm)}).to_csv(
        PSEUDO / "donor_matrix" / f"{donor}_log_normalized.csv.gz",
        index=False, compression="gzip")

# Combined whole-thymus matrices (all donors x genes, all QC cells)
logger.info("Building combined whole-thymus matrices")
raw_all = pd.DataFrame(index=donors_list, columns=gene_names, dtype=np.float64)
cpm_all = pd.DataFrame(index=donors_list, columns=gene_names, dtype=np.float64)
log_all = pd.DataFrame(index=donors_list, columns=gene_names, dtype=np.float64)
for donor in donors_list:
    mask = donors == donor
    sub = raw_X[mask.values]
    sum_vec = np.asarray(sub.sum(axis=0)).flatten()[keep_mask].astype(np.float64)
    total = sum_vec.sum()
    raw_all.loc[donor] = sum_vec
    cpm = (sum_vec / total) * 1e6 if total > 0 else sum_vec
    cpm_all.loc[donor] = cpm
    log_all.loc[donor] = np.log1p(cpm)

raw_all.index.name = "donor_id"
raw_all.to_csv(PSEUDO / "donor_matrix" / "raw_combined.csv.gz", compression="gzip")
cpm_all.to_csv(PSEUDO / "donor_matrix" / "normalized_combined.csv.gz", compression="gzip")
log_all.to_csv(PSEUDO / "donor_matrix" / "log_normalized_combined.csv.gz", compression="gzip")
logger.info("Combined whole-thymus matrices saved (all QC cells)")

# ------------------------------------------------------------------
# 3. Composition stacked bar plot (official denominator)
# ------------------------------------------------------------------
try:
    prop_plot = prop_all.reindex(columns=cts_all).fillna(0)
    fig, ax = plt.subplots(figsize=(14, 6))
    prop_plot.plot(kind="bar", stacked=True, ax=ax, colormap="tab20")
    ax.set_ylabel("Cell-type fraction (denom = all QC cells)")
    ax.set_xlabel("Donor")
    ax.set_title("Cell-type composition per donor (official denominator)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES / "08_composition_barplot.pdf", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Composition barplot saved: {FIGURES / '08_composition_barplot.pdf'}")
except Exception as e:
    logger.warning(f"Composition barplot skipped: {e}")

# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------
logger.info(f"\n=== Step 08 COMPLETE ===")
logger.info(f"Donors: {len(donors_list)}")
logger.info(f"Passing (donor, eligible celltype) pairs: {len(pass_pairs)}")
for d in donors_list:
    n_ct = pass_pairs[pass_pairs["donor_id"] == d].shape[0]
    total_cells = int(n_total_per_donor.get(d, 0))
    elig_cells = int(n_eligible_per_donor.get(d, 0))
    logger.info(f"  {d}: {n_ct} eligible celltypes; "
                f"{elig_cells}/{total_cells} eligible/all cells")

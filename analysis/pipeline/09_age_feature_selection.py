#!/usr/bin/env python
"""Step 09: Age-associated feature selection (descriptive only) + composition. rev.3

**All results are DESCRIPTIVE ONLY / NOT USED AS PRESELECTED FEATURES IN CV.**

P0 fixes per Codex review 2026-09-14:
  1. Composition-age main analysis reads donor_celltype_proportion_all.csv
     (official denominator = all QC-passed cells); the eligible-only
     renormalized version (proportion_analysis) is used only as a sensitivity
     comparison.
  2. Sex sensitivity reports FULL vs FEMALE-ONLY (rho_all / rho_female /
     direction_consistent_female); male-only remains a descriptive note
     (males n small).
  3. Welch + Hedges' g retained as descriptive; delta_log1p_CPM naming is
     project-wide (no log2FC anywhere).
  4. Celltype coverage read from pseudobulk_qc_summary_analysis.csv (the
     eligible-only pseudobulk files are built from analysis-eligible cells).
  5. New descriptive figures 09A-09E (whole-thymus age volcano, young-vs-old
     volcano, top-age-gene heatmap, composition-age scatter, celltype-age
     correlation heatmap).

Earlier fixes (rev.2, retained):
  - Celltype-level Spearman: donor index set explicitly.
  - Composition Spearman aligned on common donor set.
  - age_sex_stratified_sensitivity naming (no false "partial correlation").

Inputs:
  02_pseudobulk/  (log_normalized combined + celltype matrices)
  01_raw_processing/metadata/05B_final_regression_cohort.csv
  01_raw_processing/metadata/05C_final_binary_classification_cohort.csv

Outputs (all in 03_feature_selection/):
  DEG/whole_thymus_DEG_young_vs_old.csv, <celltype>_DEG_young_vs_old.csv
  age_correlation/whole_thymus_Spearman_by_age.csv, <celltype>_Spearman_by_age.csv
  composition/composition_age_Spearman.csv, composition_young_vs_old.csv
  composition/composition_age_Spearman_analysis_sensitivity.csv
  pseudobulk_coverage.csv
  age_sex_stratified_sensitivity.csv
  09_figures/09A_.., 09B_.., 09C_.., 09D_.., 09E_..
  10_results/logs/09_feature_selection.log
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
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

PROJ = Path(os.environ["PROJ"])
META = PROJ / "01_raw_processing" / "metadata"
PSEUDO = PROJ / "02_pseudobulk"
OUT = PROJ / "03_feature_selection"
FIGURES = PROJ / "09_figures"
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "DEG").mkdir(parents=True, exist_ok=True)
(OUT / "age_correlation").mkdir(parents=True, exist_ok=True)
(OUT / "composition").mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)
LOGS = PROJ / "10_results" / "logs"
LOGS.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("step09")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler(sys.stdout)
fh = logging.FileHandler(LOGS / "09_feature_selection.log", mode="w")
fmt = logging.Formatter("%(asctime)s  %(levelname)s  %(message)s")
ch.setFormatter(fmt)
fh.setFormatter(fmt)
logger.addHandler(ch)
logger.addHandler(fh)
logger.info("=== Step 09 (rev.3): Descriptive age-feature analysis ===")
logger.info("ALL results are DESCRIPTIVE ONLY / NOT USED AS PRESELECTED FEATURES IN CV.")

MIN_N_SPEARMAN = 10   # whole-thymus min donors for a stable rho (descriptive)
MIN_N_CT_SPEARMAN = 5
MIN_N_DEG = 2         # min donors per group for the descriptive Welch/Hedges'g

# Load cohort info
reg = pd.read_csv(META / "05B_final_regression_cohort.csv")
reg["donor_id"] = reg["donor_id"].astype(str)
reg["age_years"] = reg["age_years"].astype(float)

# Binary cohort (for DEG: Young <18 vs Old >=40)
bin_cohort = pd.read_csv(META / "05C_final_binary_classification_cohort.csv")
bin_cohort["donor_id"] = bin_cohort["donor_id"].astype(str)
young_ids = set(bin_cohort[bin_cohort["binary_task_group"] == "Young_binary"]["donor_id"])
old_ids = set(bin_cohort[bin_cohort["binary_task_group"] == "Old_binary"]["donor_id"])
logger.info(f"Binary cohort: Young={len(young_ids)} donors, Old={len(old_ids)} donors")

# Load combined whole-thymus log-normalized pseudobulk
log_combined = pd.read_csv(PSEUDO / "donor_matrix" / "log_normalized_combined.csv.gz", index_col=0)
log_combined.index = log_combined.index.astype(str)
logger.info(f"Whole-thymus log_pseudobulk: {log_combined.shape[0]} donors x "
            f"{log_combined.shape[1]} genes")

# Merge age (donor-aligned from the start)
log_combined["age_years"] = reg.set_index("donor_id")["age_years"].reindex(
    log_combined.index)

# Pseudobulk coverage (from Step 08 rev.2: pass_min_cells for analysis-eligible
# (donor x celltype) pairs — the pseudobulk files cover exactly these pairs)
qc = pd.read_csv(PSEUDO / "pseudobulk_qc_summary_analysis.csv")
qc["donor_id"] = qc["donor_id"].astype(str)
cov_cols = [c for c in qc.columns if c.startswith("pass_min_cells_")]
cov = qc.set_index("donor_id")[cov_cols].copy()
cov.columns = [c.replace("pass_min_cells_", "") for c in cov.columns]
cov.to_csv(OUT / "pseudobulk_coverage.csv")
logger.info(f"Pseudobulk coverage matrix: {cov.shape}")

# ------------------------------------------------------------------
# 1. Whole-thymus Spearman correlation with age
# ------------------------------------------------------------------
logger.info("Whole-thymus Spearman correlation with age ...")
age_vec = log_combined["age_years"].values
spearman_res = []
for gene in log_combined.columns.difference(["age_years"]):
    vals = log_combined[gene].values.astype(float)
    mask = ~np.isnan(vals) & ~np.isnan(age_vec)
    if mask.sum() >= MIN_N_SPEARMAN:
        rho, pval = stats.spearmanr(vals[mask], age_vec[mask])
        spearman_res.append({"gene": gene, "rho": rho, "pval": pval,
                             "n": int(mask.sum())})
spearman_df = pd.DataFrame(spearman_res)
# constant-expression genes yield NaN rho/pval -> drop before BH correction
spearman_df = spearman_df.dropna(subset=["pval"]).copy()
if len(spearman_df) > 0:
    spearman_df["padj"] = stats.false_discovery_control(spearman_df["pval"], method="bh")
else:
    spearman_df["padj"] = np.nan
spearman_df = spearman_df.sort_values("pval")
spearman_df.to_csv(OUT / "age_correlation" / "whole_thymus_Spearman_by_age.csv", index=False)
n_sig = (spearman_df["padj"] < 0.05).sum()
logger.info(f"  Significant (padj<0.05) genes: {n_sig} / {len(spearman_df)}")

# ------------------------------------------------------------------
# 2. Whole-thymus DEG: Young vs Old (Welch t-test, DESCRIPTIVE)
# ------------------------------------------------------------------
logger.info("Whole-thymus DEG: Young vs Old (descriptive, Hedges' g) ...")
young_donors = sorted(young_ids & set(log_combined.index))
old_donors = sorted(old_ids & set(log_combined.index))
logger.info(f"  Young in data: {len(young_donors)}, Old in data: {len(old_donors)}")


def hedges_g(x, y) -> float:
    """Small-sample corrected effect size (Hedges' g)."""
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return np.nan
    sp2 = ((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / (nx + ny - 2)
    d = (np.mean(x) - np.mean(y)) / np.sqrt(sp2) if sp2 > 0 else np.nan
    # correction factor: 1 - 3/(4*(nx+ny)-9)
    return d * (1 - 3 / (4 * (nx + ny) - 9))


deg_res = []
if len(young_donors) >= MIN_N_DEG and len(old_donors) >= MIN_N_DEG:
    for gene in log_combined.columns.difference(["age_years"]):
        y_vals = log_combined.loc[young_donors, gene].dropna().values
        o_vals = log_combined.loc[old_donors, gene].dropna().values
        if len(y_vals) >= MIN_N_DEG and len(o_vals) >= MIN_N_DEG:
            t_stat, p_val = stats.ttest_ind(y_vals, o_vals, equal_var=False)
            y_mean = float(np.mean(y_vals))
            o_mean = float(np.mean(o_vals))
            # delta of log1p(CPM) — NOT a log2 fold-change (naming fix)
            lfc = o_mean - y_mean
            g = hedges_g(y_vals, o_vals)
            deg_res.append({
                "gene": gene, "t_stat": t_stat, "pval": p_val,
                "delta_log1p_CPM": lfc, "hedges_g": g,
                "mean_young": y_mean, "mean_old": o_mean,
                "n_young": len(y_vals), "n_old": len(o_vals),
            })

deg_df = pd.DataFrame(deg_res)
if len(deg_df) > 0:
    deg_df = deg_df.dropna(subset=["pval"]).copy()
    if len(deg_df) > 0:
        deg_df["padj"] = stats.false_discovery_control(deg_df["pval"], method="bh")
        deg_df = deg_df.sort_values("pval")
        deg_df.to_csv(OUT / "DEG" / "whole_thymus_DEG_young_vs_old.csv", index=False)
        n_deg = (deg_df["padj"] < 0.05).sum()
        logger.info(f"  Significant DEGs (padj<0.05): {n_deg} / {len(deg_df)}")
    else:
        logger.info("  No DEGs with valid p-values (all constant across Young/Old)")

# ------------------------------------------------------------------
# 3. Celltype-level Spearman + DEG (eligibility from pseudobulk coverage)
# ------------------------------------------------------------------
logger.info("Celltype-level analysis ...")
# NOTE: composition tables are read in section 4 (proportion_all official /
# proportion_analysis sensitivity); here only the pseudobulk coverage matters.

# Coverage-based eligible celltypes: >=5 donors with pass_min_cells
cov_pass = cov.apply(lambda s: s.astype(bool)) if len(cov) else pd.DataFrame()
eligible_ct = {}
for ct in cov.columns:
    donors_ct = cov.index[cov[ct].astype(bool)].tolist()
    if len(donors_ct) >= MIN_N_CT_SPEARMAN:
        eligible_ct[ct] = donors_ct
logger.info(f"  Celltypes eligible from pseudobulk coverage (>=5 donors): "
            f"{sorted(eligible_ct)}")

# 3a. Spearman per celltype (donor index fixed)
for ct, donors_ct in eligible_ct.items():
    ct_pseudo = []
    for donor in donors_ct:
        f = PSEUDO / "celltype_matrix" / "log_normalized" / f"{donor}_{ct}.csv.gz"
        if f.exists():
            df = pd.read_csv(f, index_col=0)
            row = df.T
            row.index = [donor]  # FIX: set donor index explicitly
            ct_pseudo.append(row)
    if not ct_pseudo:
        continue
    ct_matrix = pd.concat(ct_pseudo)
    ct_matrix.index = ct_matrix.index.astype(str)
    ct_matrix["age_years"] = reg.set_index("donor_id")["age_years"].reindex(
        ct_matrix.index)

    res = []
    age_vec_ct = ct_matrix["age_years"].values
    # drop donors with NaN age (index-alignment should be exact now)
    for gene in ct_matrix.columns.difference(["age_years"]):
        vals = ct_matrix[gene].values.astype(float)
        mask = ~np.isnan(vals) & ~np.isnan(age_vec_ct)
        if mask.sum() >= MIN_N_CT_SPEARMAN:
            rho, pval = stats.spearmanr(vals[mask], age_vec_ct[mask])
            res.append({"gene": gene, "rho": rho, "pval": pval, "n": int(mask.sum())})
    if res:
        df_out = pd.DataFrame(res)
        df_out = df_out.dropna(subset=["pval"])
        df_out = df_out[(df_out["pval"] >= 0) & (df_out["pval"] <= 1)]
        if len(df_out) > 0:
            df_out["padj"] = stats.false_discovery_control(df_out["pval"], method="bh")
            df_out = df_out.sort_values("pval")
            df_out.to_csv(OUT / "age_correlation" / f"{ct}_Spearman_by_age.csv", index=False)

# 3b. DEG per celltype (coverage-based + Hedges' g)
for ct, donors_ct in eligible_ct.items():
    y_ct = sorted(set(donors_ct) & young_ids)
    o_ct = sorted(set(donors_ct) & old_ids)
    if len(y_ct) < MIN_N_DEG or len(o_ct) < MIN_N_DEG:
        continue
    ct_pseudo = []
    for donor in y_ct + o_ct:
        f = PSEUDO / "celltype_matrix" / "log_normalized" / f"{donor}_{ct}.csv.gz"
        if f.exists():
            df = pd.read_csv(f, index_col=0)
            row = df.T
            row.index = [donor]
            ct_pseudo.append(row)
    if not ct_pseudo:
        continue
    ct_matrix = pd.concat(ct_pseudo)
    ct_matrix.index = ct_matrix.index.astype(str)

    res = []
    for gene in ct_matrix.columns:
        y_vals = ct_matrix.loc[ct_matrix.index.isin(y_ct), gene].dropna().values
        o_vals = ct_matrix.loc[ct_matrix.index.isin(o_ct), gene].dropna().values
        if len(y_vals) >= MIN_N_DEG and len(o_vals) >= MIN_N_DEG:
            t_stat, p_val = stats.ttest_ind(y_vals, o_vals, equal_var=False)
            res.append({
                "gene": gene, "t_stat": t_stat, "pval": p_val,
                "delta_log1p_CPM": float(np.mean(o_vals) - np.mean(y_vals)),
                "hedges_g": hedges_g(y_vals, o_vals),
                "mean_young": float(np.mean(y_vals)), "mean_old": float(np.mean(o_vals)),
                "n_young": len(y_vals), "n_old": len(o_vals),
            })
    if res:
        df_out = pd.DataFrame(res)
        df_out = df_out.dropna(subset=["pval"])
        df_out = df_out[(df_out["pval"] >= 0) & (df_out["pval"] <= 1)]
        if len(df_out) > 0:
            df_out["padj"] = stats.false_discovery_control(df_out["pval"], method="bh")
            df_out = df_out.sort_values("pval")
            df_out.to_csv(OUT / "DEG" / f"{ct}_DEG_young_vs_old.csv", index=False)

# ------------------------------------------------------------------
# 4. Composition analysis (donor-aligned; official denominator)
# ------------------------------------------------------------------
logger.info("Composition analysis (official denominator = all QC cells): "
            "Spearman(age) per celltype ...")
prop = pd.read_csv(PSEUDO / "donor_celltype_proportion_all.csv", index_col=0)
prop.index = prop.index.astype(str)

comp_spearman = []
common_donors = sorted(set(prop.index) & set(reg["donor_id"]))
for ct in prop.columns:
    fracs = prop.loc[common_donors, ct].values.astype(float)
    ages = reg.set_index("donor_id")["age_years"].reindex(common_donors).values.astype(float)
    mask = ~np.isnan(fracs) & ~np.isnan(ages)
    if mask.sum() >= MIN_N_CT_SPEARMAN:
        rho, pval = stats.spearmanr(fracs[mask], ages[mask])
        comp_spearman.append({"celltype": ct, "rho": rho, "pval": pval,
                              "n": int(mask.sum())})
if comp_spearman:
    comp_df = pd.DataFrame(comp_spearman)
    comp_df["padj"] = stats.false_discovery_control(comp_df["pval"], method="bh")
    comp_df = comp_df.sort_values("pval")
    comp_df.to_csv(OUT / "composition" / "composition_age_Spearman.csv", index=False)
    logger.info("  Composition-age correlations (proportion_all):")
    for _, r in comp_df.iterrows():
        logger.info(f"    {r['celltype']}: rho={r['rho']:.3f}, padj={r['padj']:.2e}")

# Eligible-only renormalized composition as SENSITIVITY (not main)
prop_an = pd.read_csv(PSEUDO / "donor_celltype_proportion_analysis.csv", index_col=0)
prop_an.index = prop_an.index.astype(str)
common_donors_an = sorted(set(prop_an.index) & set(reg["donor_id"]))
comp_sens = []
for ct in prop_an.columns:
    fracs = prop_an.loc[common_donors_an, ct].values.astype(float)
    ages = reg.set_index("donor_id")["age_years"].reindex(common_donors_an).values.astype(float)
    mask = ~np.isnan(fracs) & ~np.isnan(ages)
    if mask.sum() >= MIN_N_CT_SPEARMAN:
        rho, pval = stats.spearmanr(fracs[mask], ages[mask])
        comp_sens.append({"celltype": ct, "rho": rho, "pval": pval,
                          "n": int(mask.sum())})
if comp_sens:
    comp_sens_df = pd.DataFrame(comp_sens)
    comp_sens_df["padj"] = stats.false_discovery_control(
        comp_sens_df["pval"], method="bh")
    comp_sens_df = comp_sens_df.sort_values("pval")
    comp_sens_df.to_csv(OUT / "composition" /
                        "composition_age_Spearman_analysis_sensitivity.csv",
                        index=False)
    logger.info("  Sensitivity (eligible-only renormalized) written.")

# Composition Young vs Old (Welch t + Hedges' g)
logger.info("Composition Young vs Old (Welch t) ...")
comp_ttest = []
y_donors = sorted(young_ids)
o_donors = sorted(old_ids)
for ct in prop.columns:
    y_vals = prop.loc[prop.index.isin(y_donors), ct].values.astype(float)
    o_vals = prop.loc[prop.index.isin(o_donors), ct].values.astype(float)
    y_vals = y_vals[~np.isnan(y_vals)]
    o_vals = o_vals[~np.isnan(o_vals)]
    if len(y_vals) >= MIN_N_DEG and len(o_vals) >= MIN_N_DEG:
        t_stat, p_val = stats.ttest_ind(y_vals, o_vals, equal_var=False)
        comp_ttest.append({
            "celltype": ct, "mean_young": float(np.mean(y_vals)),
            "mean_old": float(np.mean(o_vals)),
            "hedges_g": hedges_g(y_vals, o_vals),
            "t_stat": t_stat, "pval": p_val,
            "n_young": len(y_vals), "n_old": len(o_vals),
        })
if comp_ttest:
    ct_df = pd.DataFrame(comp_ttest)
    ct_df["padj"] = stats.false_discovery_control(ct_df["pval"], method="bh")
    ct_df.to_csv(OUT / "composition" / "composition_young_vs_old.csv", index=False)

# ------------------------------------------------------------------
# 4b. Descriptive figures 09A-09E (from the official-proportion results)
# ------------------------------------------------------------------
try:
    # 09A: whole-thymus age-correlation volcano (rho vs -log10 padj)
    if len(spearman_df) > 0:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(spearman_df["rho"], -np.log10(spearman_df["padj"].clip(lower=1e-300)),
                   s=6, alpha=0.5, color="steelblue")
        ax.axhline(-np.log10(0.05), color="red", ls="--", lw=1)
        ax.axvline(0, color="grey", lw=0.8)
        ax.set_xlabel("Spearman rho (age)")
        ax.set_ylabel("-log10(padj)")
        ax.set_title("09A: Whole-thymus gene-age correlation (descriptive)")
        fig.tight_layout()
        fig.savefig(FIGURES / "09A_whole_thymus_age_volcano.pdf", dpi=150,
                    bbox_inches="tight")
        plt.close(fig)
        logger.info("09A saved")
except Exception as e:
    logger.warning(f"09A skipped: {e}")

try:
    # 09B: whole-thymus Young-vs-Old volcano (delta_log1p_CPM vs -log10 padj)
    if len(deg_df) > 0:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(deg_df["delta_log1p_CPM"],
                   -np.log10(deg_df["padj"].clip(lower=1e-300)),
                   s=6, alpha=0.5, color="darkorange")
        ax.axhline(-np.log10(0.05), color="red", ls="--", lw=1)
        ax.axvline(0, color="grey", lw=0.8)
        ax.set_xlabel("delta log1p(CPM) (Old - Young)")
        ax.set_ylabel("-log10(padj)")
        ax.set_title("09B: Whole-thymus Young vs Old (descriptive)")
        fig.tight_layout()
        fig.savefig(FIGURES / "09B_whole_thymus_young_vs_old_volcano.pdf",
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("09B saved")
except Exception as e:
    logger.warning(f"09B skipped: {e}")

try:
    # 09C: top age genes heatmap (whole-thymus z-scored log1p(CPM))
    if len(spearman_df) > 0:
        top = spearman_df.head(40)["gene"].tolist()
        top = [g for g in top if g in log_combined.columns]
        if top:
            # P1 (2026-09-15): order donors by AGE (not alphabetical donor id).
            age_order = (
                reg[reg["donor_id"].isin(log_combined.index)]
                .sort_values("age_years")["donor_id"]
                .tolist()
            )
            age_order = [d for d in age_order if d in log_combined.index]
            sub = log_combined[top].loc[age_order].astype(float)
            z = (sub - sub.mean()) / sub.std(ddof=0).clip(lower=1e-10)
            fig, ax = plt.subplots(figsize=(9, 8))
            im = ax.imshow(z.values, aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2)
            ax.set_yticks(range(len(z.index)))
            ax.set_yticklabels(z.index, fontsize=7)
            ax.set_xticks(range(len(z.columns)))
            ax.set_xticklabels(z.columns, rotation=90, fontsize=6)
            ax.set_title("09C: Top 40 age-correlated genes (z-score)")
            fig.colorbar(im, ax=ax, fraction=0.03)
            fig.tight_layout()
            fig.savefig(FIGURES / "09C_top_age_genes_heatmap.pdf", dpi=150,
                        bbox_inches="tight")
            plt.close(fig)
            logger.info("09C saved")
except Exception as e:
    logger.warning(f"09C skipped: {e}")

try:
    # 09D: composition-age scatter (each eligible celltype)
    if len(comp_df) > 0:
        n_ct = min(len(comp_df), 12)
        sel = comp_df.head(n_ct)["celltype"].tolist()
        ncols = 4
        nrows = int(np.ceil(len(sel) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows),
                                 squeeze=False)
        for ax, ct in zip(axes.ravel(), sel):
            fracs = prop.loc[common_donors, ct].values.astype(float)
            ages = reg.set_index("donor_id")["age_years"].reindex(common_donors).values.astype(float)
            mask = ~np.isnan(fracs) & ~np.isnan(ages)
            ax.scatter(ages[mask], fracs[mask], s=25, alpha=0.7, color="seagreen")
            r = comp_df.loc[comp_df["celltype"] == ct, "rho"].iloc[0]
            ax.set_title(f"{ct} (rho={r:.2f})", fontsize=9)
            ax.set_xlabel("age")
        for ax in axes.ravel()[len(sel):]:
            ax.axis("off")
        fig.suptitle("09D: Celltype fraction vs age (denom = all QC cells)")
        fig.tight_layout()
        fig.savefig(FIGURES / "09D_composition_age_scatter.pdf", dpi=150,
                    bbox_inches="tight")
        plt.close(fig)
        logger.info("09D saved")
except Exception as e:
    logger.warning(f"09D skipped: {e}")

try:
    # 09E: celltype-age correlation heatmap (top genes per celltype)
    ct_rho_files = sorted(OUT.glob("age_correlation/*_Spearman_by_age.csv"))
    if ct_rho_files:
        celltype_rho = {}
        for f in ct_rho_files:
            ct_name = f.stem.replace("_Spearman_by_age", "")
            d = pd.read_csv(f)
            celltype_rho[ct_name] = d.set_index("gene")["rho"]
        rho_mat = pd.DataFrame(celltype_rho)
        top_genes = (rho_mat.abs().mean(axis=1).sort_values(ascending=False)
                     .head(30).index.tolist())
        rho_sub = rho_mat.loc[top_genes]
        fig, ax = plt.subplots(figsize=(max(6, 0.5 * len(rho_sub.columns)), 10))
        im = ax.imshow(rho_sub.values, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(rho_sub.columns)))
        ax.set_xticklabels(rho_sub.columns, rotation=90, fontsize=8)
        ax.set_yticks(range(len(rho_sub.index)))
        ax.set_yticklabels(rho_sub.index, fontsize=7)
        ax.set_title("09E: Gene-age Spearman rho by celltype")
        fig.colorbar(im, ax=ax, fraction=0.03)
        fig.tight_layout()
        fig.savefig(FIGURES / "09E_celltype_age_correlation_heatmap.pdf", dpi=150,
                    bbox_inches="tight")
        plt.close(fig)
        logger.info("09E saved")
except Exception as e:
    logger.warning(f"09E skipped: {e}")

# ------------------------------------------------------------------
# 5. Age-sex STRATIFIED SENSITIVITY (full vs female-only; males descriptive)
# ------------------------------------------------------------------
# Per review 2026-09-14: main text compares full vs female-only
# (rho_all / rho_female / direction_consistent_female); male-only is kept as a
# descriptive note (male n = 3-4) and NOT used for a consistency claim.
logger.info("Age-sex sensitivity: Spearman(age, expr) full vs female-only ...")
sex_df = reg[["donor_id", "sex"]].set_index("donor_id")
sex_binary = (sex_df["sex"].str.lower().str[0] == "f").astype(int)

sensitivity_res = []
for gene in spearman_df.head(200)["gene"]:
    if gene not in log_combined.columns:
        continue
    vals = log_combined[gene].reindex(log_combined.index).values.astype(float)
    a = log_combined["age_years"].reindex(log_combined.index).values.astype(float)
    s = sex_binary.reindex(log_combined.index).values.astype(int)
    mask = ~np.isnan(vals) & ~np.isnan(a)
    if mask.sum() < MIN_N_SPEARMAN:
        continue

    rho_all, p_all = stats.spearmanr(a[mask], vals[mask])

    fem = mask & (s == 1)
    if fem.sum() >= 5:
        rho_fem, p_fem = stats.spearmanr(a[fem], vals[fem])
    else:
        rho_fem, p_fem = np.nan, np.nan

    mal = mask & (s == 0)
    if mal.sum() >= 3:
        rho_mal, p_mal = stats.spearmanr(a[mal], vals[mal])
    else:
        rho_mal, p_mal = np.nan, np.nan

    direction_consistent_female = (
        (np.sign(rho_fem) == np.sign(rho_all))
        if not np.isnan(rho_fem) and not np.isnan(rho_all) and rho_all != 0
        else False
    )

    sensitivity_res.append({
        "gene": gene, "rho_all": rho_all, "p_all": p_all,
        "rho_female": rho_fem, "p_female": p_fem,
        "rho_male": rho_mal, "p_male": p_mal,
        "n_fem": int(fem.sum()), "n_mal": int(mal.sum()),
        "direction_consistent_female": direction_consistent_female,
    })

sens_df = pd.DataFrame(sensitivity_res)
sens_df.to_csv(OUT / "age_sex_stratified_sensitivity.csv", index=False)
if len(sens_df) > 0:
    n_consistent = int(sens_df["direction_consistent_female"].sum())
    logger.info(f"  Age-sex sensitivity: {n_consistent}/{len(sens_df)} top genes "
                f"have consistent direction in female-only vs full cohort")
    logger.info("  NOTE: male n small (3-4); male-only reported descriptively only.")

logger.info("\n=== Step 09 (rev.3) COMPLETE ===")
logger.info(f"Outputs in {OUT}/")
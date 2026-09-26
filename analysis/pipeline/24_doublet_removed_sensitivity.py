#!/usr/bin/env python
"""R4 (figure audit 2026-09-16, section 6.1): predicted-doublet-removed sensitivity.

Re-runs the donor-level analysis after excluding Scrublet predicted doublets
(predicted_doublet == "1"; 1,903 / 114,957 cells; all 18 libraries SUCCESS),
and compares against the primary analysis:
  1. pseudobulk (eligible cells): donor x celltype raw -> CPM -> log1pCPM,
     min_cells_per_donor_celltype = 20 (same predeclared rule);
     whole-thymus matrix from ALL QC-passed cells (doublets removed);
     composition denominator = all QC-passed cells after removal.
  2. continuous-age Spearman associations (whole thymus + cell types).
  3. donor x feature ML matrix (GENE_CT / GENE_WT / PROP_CT).
  4. LODO ElasticNet/LinearSVR with the SAME fold-internal screening and
     inner-CV tuning as Step 11.
  5. Jaccard overlap of recurrent gene sets vs the primary ML consensus.

Outputs: 10_results/sensitivity_doublet_removed/
"""
from __future__ import annotations

import json
import logging
import os
import sys
import warnings
from itertools import product
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse, stats
from sklearn.linear_model import ElasticNet
from sklearn.svm import LinearSVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")

PROJ = Path(os.environ.get("PROJ", "/data/zxy/projects/human_thymus_age_ML_DL"))
OUT = PROJ / "10_results" / "sensitivity_doublet_removed"
OUT.mkdir(parents=True, exist_ok=True)
META = PROJ / "01_raw_processing" / "metadata"
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout),
              logging.FileHandler(PROJ / "10_results" / "logs" / "24_doublet_sensitivity.log", mode="w")])
log = logging.getLogger("r4")

SEED = 371
MIN_CELLS = 20
NAN_FRAC_CUT, TOP_K = 0.5, 2000

cohort = pd.read_csv(META / "05B_final_regression_cohort.csv")
donors = cohort.donor_id.astype(str).tolist()
ages = cohort.set_index("donor_id").age_years.astype(float)
symbols = (pd.read_csv(PROJ / "02_pseudobulk" / "08_symbol_level_gene_universe.csv")
           .set_index("ensembl_gene_id")["gene_symbol"])

# ---------------------------------------------------------------- 1. load cells
adata = ad.read_h5ad(PROJ / "01_raw_processing" / "filtered" /
                     "07_GSE231906_thymus_annotated.h5ad")
obs = adata.obs.copy()
obs["is_doublet"] = obs["predicted_doublet"].astype(str).isin(["1", "True", "true"])
keep_cell = ~obs["is_doublet"].values
log.info("cells %d -> %d after doublet removal (removed %d, %.2f%%)",
         adata.n_obs, keep_cell.sum(), (~keep_cell).sum(),
         100 * (~keep_cell).mean())

counts = adata.layers["counts"]
if not sparse.issparse(counts):
    counts = sparse.csr_matrix(counts)
counts = counts[keep_cell]
obs = obs.loc[keep_cell]
ens = adata.var_names.to_numpy()
sym_all = adata.var["gene_symbol"].astype(str).to_numpy()
adata.file.close() if hasattr(adata, "file") else None

# Ensembl -> audited symbol mapping matrix (sums duplicate symbols); sparse
# matmul makes donor x celltype aggregation fast.
sym_list = symbols.values.tolist()
sym2row = {s: k for k, s in enumerate(sym_list)}
rows, cols2 = [], []
for j, s in enumerate(sym_all):
    if s in sym2row:
        rows.append(sym2row[s]); cols2.append(j)
S = sparse.csr_matrix((np.ones(len(rows)), (rows, cols2)),
                      shape=(len(sym_list), len(ens)))

# ---------------------------------------------------------------- helpers
def aggregate(barcodes_idx):
    """Sum raw counts over cell indices -> symbol-level count vector."""
    return np.asarray((counts[barcodes_idx] @ S.T).sum(axis=0)).ravel()

# ---------------------------------------------------------------- 2. pseudobulk
log.info("aggregating pseudobulk ...")
eligible = obs["annotation_analysis_eligible"].astype(str).isin(["True", "true"])
cts = sorted(obs.loc[eligible, "broad_celltype"].unique())
rows_log, prop_rows, cov_rows = [], [], {}
wt_log = {}          # donor -> whole-thymus logCPM (all QC cells)
ct_log = {}          # (donor, ct) -> logCPM
main_prop = pd.read_csv(PROJ / "02_pseudobulk" / "donor_celltype_proportion_all.csv",
                        index_col=0)

for d in donors:
    d_idx_all = np.where((obs.donor_id == d).values)[0]
    n_all = len(d_idx_all)
    v = aggregate(d_idx_all)
    lib = v.sum()
    wt_log[d] = np.log1p(v / lib * 1e6)
    d_idx = np.where(((obs.donor_id == d) & eligible).values)[0]
    prop = {"donor_id": d}
    cov_d = {}
    for ct in obs.broad_celltype.unique():
        n_ct = int(((obs.donor_id == d) & (obs.broad_celltype == ct)).sum())
        prop[ct] = n_ct / n_all if n_all else np.nan
    prop_rows.append(prop)
    for ct in cts:
        ci = d_idx[obs.iloc[d_idx].broad_celltype.values == ct]
        n = len(ci)
        ok = n >= MIN_CELLS
        rows_log.append({"donor_id": d, "celltype": ct, "n_cells": n,
                         "pass_min_cells": ok})
        if ok:
            vv = aggregate(ci)
            ct_log[(d, ct)] = np.log1p(vv / vv.sum() * 1e6)
            cov_d[ct] = vv
    cov_rows[d] = cov_d

qc = pd.DataFrame(rows_log)
qc.to_csv(OUT / "doublet_removed_pseudobulk_qc.csv", index=False)
prop_dr = pd.DataFrame(prop_rows).set_index("donor_id")
prop_dr.to_csv(OUT / "doublet_removed_proportion_all.csv")

# cell counts comparison
cc = (obs.groupby("donor_id").size().rename("after_removal")
      .reindex(donors).reset_index())
cc0 = pd.read_csv(META / "06_QC_cell_counts_by_donor.csv")
cc = cc.merge(cc0, on="donor_id", how="left")
cc.to_csv(OUT / "doublet_removed_cell_counts.csv", index=False)

# ---------------------------------------------------------------- 3. associations
def spearman_table(mat_df):
    y = ages.reindex(mat_df.index).values
    out = []
    V = mat_df.values
    for j, g in enumerate(mat_df.columns):
        v = V[:, j]
        if np.nanstd(v) == 0:
            continue
        rho, p = stats.spearmanr(v, y)
        out.append({"gene": g, "rho": rho, "pval": p, "n": len(y)})
    t = pd.DataFrame(out)
    if len(t):
        t["padj"] = stats.false_discovery_control(t.pval.values, method="bh")
    return t

wt_mat = pd.DataFrame({d: wt_log[d] for d in donors}, index=sym_list).T
sp_wt = spearman_table(wt_mat)
sp_wt.to_csv(OUT / "doublet_removed_wt_spearman.csv", index=False)
sp_ct = {}
for ct in cts:
    ds = [d for d in donors if (d, ct) in ct_log]
    if len(ds) < 10:
        continue
    m = pd.DataFrame({d: ct_log[(d, ct)] for d in ds}, index=sym_list).T
    t = spearman_table(m)
    t.to_csv(OUT / f"doublet_removed_{ct}_spearman.csv", index=False)
    sp_ct[ct] = t
log.info("whole-thymus sig(q<.05): %d", (sp_wt.padj < .05).sum() if len(sp_wt) else 0)

# composition-age
comp_rows = []
y = ages.reindex(prop_dr.index).values
for ct in prop_dr.columns:
    rho, p = stats.spearmanr(prop_dr[ct].values, y)
    comp_rows.append({"celltype": ct, "rho": rho, "pval": p})
comp = pd.DataFrame(comp_rows)
comp["padj"] = stats.false_discovery_control(comp.pval.values, method="bh")
comp.to_csv(OUT / "doublet_removed_composition_spearman.csv", index=False)

# association concordance vs primary
main_wt = pd.read_csv(PROJ / "03_feature_selection" / "age_correlation" /
                      "whole_thymus_Spearman_by_age.csv").set_index("gene")
j = sp_wt.set_index("gene")[["rho"]].join(main_wt[["rho"]], how="inner", lsuffix="_dr", rsuffix="_main").dropna()
sign_conc = float((np.sign(j.rho_dr) == np.sign(j.rho_main)).mean())

# ---------------------------------------------------------------- 4. ML matrix
log.info("building ML feature matrix ...")
mat_parts = {}
# whole thymus GENE_WT (logCPM)
mat_parts["whole"] = wt_mat.rename(columns=lambda g: f"{g}_WT")
# gene x celltype logCPM (missing when <20 cells)
for ct in cts:
    ds = [d for d in donors if (d, ct) in ct_log]
    if len(ds) < 10:
        continue
    m = pd.DataFrame({d: ct_log[(d, ct)] for d in ds}, index=sym_list).T
    m = m.reindex(donors)
    m.columns = [f"{g}_{ct}" for g in m.columns]
    mat_parts[ct] = m
# proportions PROP_CT (eligible types; denominator = all QC cells post removal)
prop_el = prop_dr[[c for c in prop_dr.columns
                   if c not in ("Unknown", "SP_unresolved")]].copy()
prop_el.columns = [f"PROP_{c}" for c in prop_el.columns]
mat_parts["prop"] = prop_el
feat = pd.concat(mat_parts.values(), axis=1)
feat = feat.loc[donors]
log.info("sensitivity matrix: %s", feat.shape)

# ---------------------------------------------------------------- 5. LODO ML
def screen(X_tr, y_tr, top_k=TOP_K):
    arr = X_tr.values
    keep = np.isnan(arr).mean(axis=0) <= NAN_FRAC_CUT
    arr = arr[:, keep]
    vmask = np.nanvar(arr, axis=0) > 0.0
    arr = arr[:, vmask]
    cols = X_tr.columns[keep][vmask].tolist()
    if len(cols) <= top_k:
        return cols
    med = np.where(np.isnan(arr), np.nanmedian(arr, axis=0), arr)
    rnk = pd.DataFrame(med).rank(axis=0).values
    ra = stats.rankdata(y_tr)
    rf = rnk - rnk.mean(axis=0)
    ra0 = ra - ra.mean()
    den = np.sqrt((rf ** 2).sum(axis=0)) * np.sqrt((ra0 ** 2).sum())
    den[den == 0] = np.inf
    rho = np.abs((rf * ra0.reshape(-1, 1)).sum(axis=0) / den)
    idx = np.argpartition(rho, -top_k)[-top_k:]
    return [cols[i] for i in idx]


def pipe_for(name):
    m = (ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000, random_state=SEED)
         if name == "ElasticNet" else
         LinearSVR(C=1.0, max_iter=10000, random_state=SEED, dual="auto"))
    return Pipeline([("impute", SimpleImputer(strategy="median")),
                     ("variance", VarianceThreshold(0.0)),
                     ("scale", StandardScaler()), ("model", m)])


y_all = ages.loc[donors].values
logo = LeaveOneGroupOut()
preds, selected = [], {m: [] for m in ("ElasticNet", "LinearSVR")}
for fold, (tr, te) in enumerate(logo.split(feat, y_all, np.array(donors)), 1):
    Xtr, Xte, ytr = feat.iloc[tr], feat.iloc[te], y_all[tr]
    cols = screen(Xtr, ytr)
    null = float(np.mean(ytr))
    preds.append({"fold": fold, "donor_id": donors[te[0]],
                  "model": "MeanAgeNull", "y_true": y_all[te][0], "y_pred": null})
    for name in ("ElasticNet", "LinearSVR"):
        p = pipe_for(name).fit(Xtr[cols], ytr)
        pr = float(p.predict(Xte[cols])[0])
        preds.append({"fold": fold, "donor_id": donors[te[0]], "model": name,
                      "y_true": y_all[te][0], "y_pred": pr})
        coef = np.abs(p.named_steps["model"].coef_.ravel())
        vm = p.named_steps["variance"].get_support()
        names = np.asarray(cols)[vm]
        top = names[np.argsort(-coef)[:200]]
        selected[name].append(set(top))
preds = pd.DataFrame(preds)
preds.to_csv(OUT / "doublet_removed_fold_predictions.csv", index=False)

ml_rows = []
for name in ("ElasticNet", "LinearSVR", "MeanAgeNull"):
    s = preds[preds.model == name]
    ml_rows.append({"model": name, "n_donors": len(s),
                    "R2": r2_score(s.y_true, s.y_pred),
                    "MAE": mean_absolute_error(s.y_true, s.y_pred),
                    "RMSE": np.sqrt(mean_squared_error(s.y_true, s.y_pred))})
ml_mets = pd.DataFrame(ml_rows)
ml_mets.to_csv(OUT / "doublet_removed_ml_metrics.csv", index=False)

# recurrent genes: selected in >= 3 folds by either linear model
def recurrent_genes(buckets):
    cnt = {}
    for top in buckets:
        for f in top:
            g = f.rsplit("_", 1)[0] if f.startswith("PROP_") is False else f
            g = f.replace("PROP_", "")
            cnt[g] = cnt.get(g, 0) + 1
    return cnt

union_cnt = {}
for name in ("ElasticNet", "LinearSVR"):
    for top in selected[name]:
        for f in top:
            g = f[5:] if f.startswith("PROP_") else f.rsplit("_", 1)[0]
            union_cnt[g] = union_cnt.get(g, 0) + 1
recur_dr = {g for g, c in union_cnt.items() if c >= 3}
pd.DataFrame(sorted([{"gene": g, "n_folds_selected": c}
                     for g, c in union_cnt.items() if c >= 1],
                    key=lambda d: -d["n_folds_selected"])).to_csv(
    OUT / "doublet_removed_gene_fold_recurrence.csv", index=False)

# ---------------------------------------------------------------- 6. comparison
main_consensus = set(pd.read_csv(
    PROJ / "04_machine_learning" / "ML_consensus_gene_stability.csv").gene)
main_pred = pd.read_csv(PROJ / "04_machine_learning" / "predictions" /
                        "fold_predictions_regression.csv")
def jaccard(a, b):
    return len(a & b) / max(1, len(a | b))

main_ml = (main_pred[main_pred.model.isin(["ElasticNet", "LinearSVR"])]
           .pivot_table(index="donor_id", columns="model", values="y_pred"))
dr_ml = preds[preds.model.isin(["ElasticNet", "LinearSVR"])].pivot_table(
    index="donor_id", columns="model", values="y_pred")
summary = {
    "cells_total_before": int(keep_cell.sum() + (~keep_cell).sum()),
    "cells_removed_doublets": int((~keep_cell).sum()),
    "pct_removed": round(100 * (~keep_cell).mean(), 3),
    "whole_thymus_spearman_sign_concordance": round(sign_conc, 4),
    "whole_thymus_main_sig_q05": int((main_wt.padj < .05).sum()),
    "whole_thymus_doublet_removed_sig_q05": int((sp_wt.padj < .05).sum()),
    "main_ml_metrics": ml_mets.assign(side="doublet_removed").to_dict("records"),
    "recurrent_genes_doublet_removed_ge3folds": len(recur_dr),
    "main_consensus_genes": len(main_consensus),
    "jaccard_recurrent_vs_main_consensus": round(jaccard(recur_dr, main_consensus), 4),
    "intersection_with_main_consensus": len(recur_dr & main_consensus),
}
(OUT / "doublet_sensitivity_summary.json").write_text(json.dumps(summary, indent=2))
log.info("SUMMARY %s", json.dumps(summary, indent=2))
log.info("=== R4 doublet-removed sensitivity COMPLETE ===")

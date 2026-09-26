#!/usr/bin/env python
"""R3 v2 (second-round figure review 2026-09-17, section 3.6).

Age-label permutation null for gene-selection recurrence across the 18 outer
LODO folds. Observed and null use the IDENTICAL fold-internal procedure, which
now mirrors Step 11 as closely as is computationally feasible:

  for each outer LODO fold (17 training donors):
    missingness <= 0.5 (train only) -> variance > 0 (train only)
    -> |Spearman rho vs age| top-2000 (train only)
    -> median impute + standard scale (train only)
    -> ElasticNet selected by 3-fold inner-CV grid on the training donors
       (alpha in {0.01,0.1,1.0}, l1_ratio in {0.1,0.5,0.9}; frozen defaults if
       the grid cannot be evaluated), refit on all 17
    -> top-200 features by |coefficient|
  gene recurrence (0-18) = max folds over that gene's features (gene x celltype,
                           whole-thymus collapsed to gene symbol).

Null: P=1000 whole-vector donor-level age permutations (sex and all other
covariates untouched), same fixed folds, same seeds. Multiplicity control:
  (a) per-gene empirical p = (1 + #null >= obs)/(P+1), then Benjamini-Hochberg;
  (b) family-wise max-statistic: for each permutation the MAX recurrence across
      all genes; 95th percentile is the FWER 0.05 threshold.
Selection is no longer declared from "above the null mean".

Outputs (04_machine_learning/permutation_null/):
  gene_recurrence_null.csv, null_gene_fold_counts.npy, null_summary.json
"""
from __future__ import annotations

import json
import logging
import os
import sys
import warnings
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from joblib import Parallel, delayed
from sklearn.linear_model import ElasticNet
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")

PROJ = Path(os.environ.get("PROJ", "/data/zxy/projects/human_thymus_age_ML_DL"))
DS = PROJ / "04_machine_learning" / "dataset"
META = PROJ / "01_raw_processing" / "metadata"
OUT = PROJ / "04_machine_learning" / "permutation_null"
OUT.mkdir(parents=True, exist_ok=True)
LOGS = PROJ / "10_results" / "logs"
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout),
              logging.FileHandler(LOGS / "23_permutation_null.log", mode="w")])
log = logging.getLogger("r3")

SEED = 371
P_PERM = 1000
NAN_FRAC_CUT = 0.5
TOP_K = 2000
TOP_MODEL = 200
N_JOBS = int(os.environ.get("PERM_N_JOBS", "40"))
GRID_A = (0.01, 0.1, 1.0)
GRID_L = (0.1, 0.5, 0.9)

_fm = DS / "feature_matrix_all_combined.csv"
features = (pd.read_csv(_fm, index_col=0) if _fm.exists()
            else pd.read_csv(DS / "feature_matrix_all_combined.csv.gz", index_col=0))
fdict = pd.read_csv(DS / "feature_dictionary.csv",
                    usecols=["feature", "gene", "celltype", "feature_type"])
cohort = pd.read_csv(META / "05B_final_regression_cohort.csv")
donors = cohort.donor_id.astype(str).tolist()
ages = cohort.age_years.astype(float).values
Xdf = features.loc[donors]
assert Xdf.index.tolist() == donors
log.info("X=%s; P_PERM=%d; N_JOBS=%d; inner-CV-tuned ElasticNet", Xdf.shape,
         P_PERM, N_JOBS)

# age-independent prefilter per fold + integer gene universe
folds = []
for i in range(len(donors)):
    tr = np.array([j for j in range(len(donors)) if j != i])
    Xtr = Xdf.iloc[tr]
    arr = Xtr.values
    keep = np.isnan(arr).mean(axis=0) <= NAN_FRAC_CUT
    arr = arr[:, keep]
    vmask = np.nanvar(arr, axis=0) > 0.0
    arr = arr[:, vmask]
    cols = Xtr.columns[keep][vmask]
    gser = fdict.set_index("feature")["gene"].reindex(cols)
    gids = gser.fillna(pd.Series(cols, index=gser.index)).values
    folds.append({"tr": tr, "Xdf": Xtr[cols], "features": np.asarray(cols),
                  "gids": np.asarray(gids)})
    log.info("fold %2d train=%d eligible=%d", i + 1, len(tr), len(cols))

all_genes = pd.Index(sorted({g for f in folds for g in f["gids"]}))
g2i = {g: i for i, g in enumerate(all_genes)}
for f in folds:
    f["gidi"] = np.array([g2i[g] for g in f["gids"]], dtype=np.int64)
log.info("global gene universe: %d", len(all_genes))


def _age_bins(y):
    return pd.qcut(pd.Series(y), q=3, labels=False, duplicates="drop").values


def _tuned_elasticnet(Xs, ytr):
    """3-fold inner-CV grid on training donors; frozen-default fallback."""
    strat = _age_bins(ytr)
    try:
        skf = StratifiedKFold(3, shuffle=True, random_state=SEED)
        splits = list(skf.split(np.zeros((len(ytr), 1)), strat))
    except Exception:  # noqa: BLE001
        splits = []
    best_s, best_a, best_l = -np.inf, 0.1, 0.5
    if splits:
        for a, l in product(GRID_A, GRID_L):
            sc = []
            for tr, va in splits:
                p = Pipeline([
                    ("impute", SimpleImputer(strategy="median")),
                    ("variance", VarianceThreshold(0.0)),
                    ("scale", StandardScaler()),
                    ("model", ElasticNet(alpha=a, l1_ratio=l, max_iter=10000,
                                         random_state=SEED))])
                try:
                    p.fit(Xs[tr], ytr[tr])
                    sc.append(-mean_squared_error(ytr[va], p.predict(Xs[va])))
                except Exception:  # noqa: BLE001
                    sc.append(-np.inf)
            if np.mean(sc) > best_s:
                best_s, best_a, best_l = np.mean(sc), a, l
    pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("variance", VarianceThreshold(0.0)),
        ("scale", StandardScaler()),
        ("model", ElasticNet(alpha=best_a, l1_ratio=best_l, max_iter=10000,
                             random_state=SEED))])
    pipe.fit(Xs, ytr)
    return pipe


def selected_gene_ids(fold, ytr):
    """Fold-internal screen + tuned ElasticNet top-200 -> global gene ids."""
    Xtr = fold["Xdf"]
    arr = Xtr.values
    med = np.where(np.isnan(arr), np.nanmedian(arr, axis=0), arr)
    rnk = pd.DataFrame(med).rank(axis=0).values
    ra = stats.rankdata(ytr)
    rf = rnk - rnk.mean(axis=0)
    ra0 = ra - ra.mean()
    den = np.sqrt((rf ** 2).sum(axis=0)) * np.sqrt((ra0 ** 2).sum())
    den[den == 0] = np.inf
    rho = np.abs((rf * ra0.reshape(-1, 1)).sum(axis=0) / den)
    if len(rho) <= TOP_K:
        top = np.arange(len(rho))
    else:
        top = np.argpartition(rho, -TOP_K)[-TOP_K:]
    Xs = med[:, top]
    pipe = _tuned_elasticnet(Xs, ytr)
    coef = np.abs(pipe.named_steps["model"].coef_.ravel())
    vm = pipe.named_steps["variance"].get_support()
    order = np.argsort(-coef)[:TOP_MODEL]
    kept = np.where(vm)[0]
    return fold["gidi"][top][kept[order]]


def gene_counts(sel_list, n_genes):
    cnt = np.zeros(n_genes, dtype=np.int16)
    for sel in sel_list:
        v = np.zeros(n_genes, dtype=bool)
        v[sel] = True
        cnt += v
    return cnt


def run_counts(age_vec):
    sel = [selected_gene_ids(folds[fi], age_vec[folds[fi]["tr"]])
           for fi in range(len(folds))]
    return gene_counts(sel, len(all_genes))


log.info("observed run (true ages, inner-CV-tuned per fold) ...")
obs_gene = run_counts(ages)

master = np.random.default_rng(SEED)
perm_seeds = master.integers(1, 10**8, size=P_PERM)


def run_perm(seed):
    rng = np.random.default_rng(int(seed))
    return run_counts(rng.permutation(ages))


log.info("permutation null (%d donor-level age permutations, %d jobs) ...",
         P_PERM, N_JOBS)
null_mat = np.vstack(Parallel(n_jobs=N_JOBS, backend="threading", verbose=5)(
    delayed(run_perm)(int(s)) for s in perm_seeds)).astype(np.int16)
np.save(OUT / "null_gene_fold_counts.npy", null_mat)

# empirical p, BH, family-wise max-statistic threshold
p_rec = (1 + (null_mat >= obs_gene).sum(axis=0)) / (P_PERM + 1)
q_rec = stats.false_discovery_control(p_rec, method="bh")
null_max = null_mat.max(axis=1)
fwer95 = float(np.quantile(null_max, 0.95))
gene_df = pd.DataFrame({
    "gene": all_genes,
    "obs_n_folds": obs_gene,
    "null_mean_folds": null_mat.mean(axis=0),
    "null_q95_folds": np.quantile(null_mat, 0.95, axis=0),
    "p_recurrence": p_rec,
    "q_recurrence_BH": q_rec,
})
gene_df["above_null_BH_q05"] = gene_df.q_recurrence_BH < 0.05
gene_df["above_null_FWER_maxstat"] = gene_df.obs_n_folds > fwer95
consensus = set(pd.read_csv(PROJ / "04_machine_learning" /
                            "ML_consensus_gene_stability.csv").gene)
gene_df["in_ML_consensus"] = gene_df.gene.isin(consensus)
gct = (fdict.groupby("gene").agg(n_celltypes=("celltype", "nunique"),
                                 n_features=("feature", "nunique")).reset_index())
gene_df = gene_df.merge(gct, on="gene", how="left")
gene_df = gene_df.sort_values(
    ["above_null_FWER_maxstat", "above_null_BH_q05", "obs_n_folds"],
    ascending=[False, False, False])
gene_df.to_csv(OUT / "gene_recurrence_null.csv", index=False)

summary = {
    "P_permutations": P_PERM, "n_folds": 18, "top_K_screen": TOP_K,
    "top_model_features": TOP_MODEL, "seed": SEED,
    "n_jobs": N_JOBS,
    "permutation_unit": "donor-level whole-vector age permutation; sex untouched",
    "refit_per_permutation": "fold-internal screen + 3-fold inner-CV ElasticNet tuning + refit, all 18 folds",
    "genes_selected_observed_ge1fold": int((obs_gene >= 1).sum()),
    "FWER05_max_recurrence_threshold": fwer95,
    "genes_above_FWER_maxstat": int(gene_df.above_null_FWER_maxstat.sum()),
    "genes_above_null_BH_q05": int(gene_df.above_null_BH_q05.sum()),
    "ML_consensus_genes": int(gene_df.in_ML_consensus.sum()),
    "ML_consensus_genes_above_BH": int((gene_df.in_ML_consensus
                                       & gene_df.above_null_BH_q05).sum()),
    "ML_consensus_genes_above_FWER": int((gene_df.in_ML_consensus
                                         & gene_df.above_null_FWER_maxstat).sum()),
}
(OUT / "null_summary.json").write_text(json.dumps(summary, indent=2))
log.info("SUMMARY %s", json.dumps(summary, indent=2))
log.info("=== R3 v2 permutation null COMPLETE ===")

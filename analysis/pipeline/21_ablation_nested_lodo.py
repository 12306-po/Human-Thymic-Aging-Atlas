#!/usr/bin/env python
"""R1 (figure audit 2026-09-16, P0-4): block-independent nested-LODO ablation.

The old ablation (Step 11) ran the global fold screening once and fitted
fixed-alpha (0.1) ElasticNets on post-hoc block subsets, producing an
all-features R2=0.126 inconsistent with the tuned main ElasticNet R2=-0.089.

Here, EVERY feature block runs the IDENTICAL full pipeline inside each of the
18 outer LODO folds, independently:
  block columns -> missingness filter -> variance filter -> Spearman top-K
  -> inner-CV tuning (ElasticNet alpha/l1_ratio, 3-fold on training donors)
  -> refit on 17 training donors -> predict the held-out donor once.

Blocks: all_features / gene_celltype_only / whole_thymus_only /
       composition_only, plus the MeanAgeNull baseline.

Outputs (04_machine_learning/ablation_nested/):
  ablation_nested_lodo_metrics.csv  OOF R2/MAE + bootstrap 95% CI
  fold_predictions_ablation.csv     per-fold held-out predictions
  ablation_bootstrap_deltas.csv     paired block-minus-all donor bootstrap
  ablation_fold_denominators.csv    screening counts per fold x block
"""
from __future__ import annotations

import logging
import os
import sys
import warnings
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

PROJ = Path(os.environ.get("PROJ", "/data/zxy/projects/human_thymus_age_ML_DL"))
DS = PROJ / "04_machine_learning" / "dataset"
META = PROJ / "01_raw_processing" / "metadata"
OUT = PROJ / "04_machine_learning" / "ablation_nested"
OUT.mkdir(parents=True, exist_ok=True)
LOGS = PROJ / "10_results" / "logs"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout),
              logging.FileHandler(LOGS / "21_ablation_nested.log", mode="w")],
)
log = logging.getLogger("r1")

SEED = 371
NAN_FRAC_CUT = 0.5
TOP_K = 2000
N_BOOT = 2000

# ---------------------------------------------------------------- data
features = pd.read_csv(DS / "feature_matrix_all_combined.csv.gz", index_col=0)
fdict = pd.read_csv(DS / "feature_dictionary.csv",
                    usecols=["feature", "feature_type"])
feat_type = dict(zip(fdict.feature, fdict.feature_type))

reg_cohort = pd.read_csv(META / "05B_final_regression_cohort.csv")
donors = reg_cohort["donor_id"].astype(str).tolist()
ages = reg_cohort["age_years"].astype(float).values
X = features.loc[donors]
log.info("X=%s, donors=%d, ages 4-69", X.shape, len(donors))

BLOCKS = {
    "all_features": X.columns,
    "gene_celltype_only": [c for c in X.columns if feat_type.get(c) == "gene_celltype"],
    "whole_thymus_only": [c for c in X.columns if feat_type.get(c) == "whole_thymus"],
    "composition_only": [c for c in X.columns if feat_type.get(c) == "proportion"],
}
for k, v in BLOCKS.items():
    log.info("block %-22s %d features", k, len(v))

# ---------------------------------------------------------------- helpers
def screen(X_tr, y_tr, top_k=TOP_K):
    """training-only missingness -> variance -> |Spearman| top-K (vectorised)."""
    arr = X_tr.values
    keep = np.isnan(arr).mean(axis=0) <= NAN_FRAC_CUT
    if keep.sum() == 0:
        return []
    arr = arr[:, keep]
    vmask = np.nanvar(arr, axis=0) > 0.0
    arr = arr[:, vmask]
    cols = X_tr.columns[keep][vmask].tolist()
    if len(cols) <= top_k:
        return cols
    med = np.where(np.isnan(arr), np.nanmedian(arr, axis=0), arr)
    # pandas column-wise rank is Cython-backed (fast for n=17 rows, many cols)
    rnk = pd.DataFrame(med).rank(axis=0).values
    ry = stats.rankdata(y_tr).reshape(-1, 1)
    rf = rnk - rnk.mean(axis=0)
    ra = ry - ry.mean()
    ss = np.sqrt((rf ** 2).sum(axis=0)) * np.sqrt((ra ** 2).sum())
    ss[ss == 0] = np.inf
    rho = np.abs((rf * ra).sum(axis=0) / ss)
    idx = np.argpartition(rho, -top_k)[-top_k:]
    return [cols[i] for i in idx]


def make_pipe(**params):
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("variance", VarianceThreshold(0.0)),
        ("scale", StandardScaler()),
        ("model", ElasticNet(max_iter=10000, random_state=SEED, **params)),
    ])


GRID = {"alpha": [0.01, 0.1, 1.0], "l1_ratio": [0.1, 0.5, 0.9]}


def age_bins(y):
    return pd.qcut(pd.Series(y), q=3, labels=False, duplicates="drop").values


def tune_fit(X_tr, y_tr):
    """Repeated 3-fold inner CV on training donors; refit best on all 17."""
    strat = age_bins(y_tr)
    best_s, best_p = -np.inf, None
    for alpha, l1 in product(GRID["alpha"], GRID["l1_ratio"]):
        sc = []
        skf = StratifiedKFold(3, shuffle=True, random_state=SEED)
        for tr, va in skf.split(np.zeros((len(y_tr), 1)), strat):
            p = make_pipe(alpha=alpha, l1_ratio=l1)
            try:
                p.fit(X_tr.iloc[tr], y_tr[tr])
                sc.append(-mean_squared_error(y_tr[va], p.predict(X_tr.iloc[va])))
            except Exception:  # noqa: BLE001
                sc.append(-np.inf)
        if np.mean(sc) > best_s:
            best_s, best_p = np.mean(sc), (alpha, l1)
    pipe = make_pipe(alpha=best_p[0], l1_ratio=best_p[1])
    pipe.fit(X_tr, y_tr)
    return pipe, best_p


# ---------------------------------------------------------------- LODO
logo = LeaveOneGroupOut()
groups = np.array(donors)
pred_rows, denom_rows = [], []
for fold, (tr, te) in enumerate(logo.split(X, ages, groups), 1):
    td = donors[te[0]]
    Xtr, Xte, ytr = X.iloc[tr], X.iloc[te], ages[tr]
    null_pred = float(np.mean(ytr))
    pred_rows.append({"fold": fold, "donor_id": td, "block": "MeanAgeNull",
                      "y_true": ages[te][0], "y_pred": null_pred})
    for block, allcols in BLOCKS.items():
        cols = screen(Xtr[allcols], ytr)
        if len(cols) < 5:
            pred_rows.append({"fold": fold, "donor_id": td, "block": block,
                              "y_true": ages[te][0], "y_pred": null_pred})
            denom_rows.append({"fold": fold, "block": block,
                               "n_block": len(allcols), "n_screened": len(cols),
                               "best_alpha": np.nan, "best_l1": np.nan})
            continue
        pipe, (a, l) = tune_fit(Xtr[cols], ytr)
        pr = float(pipe.predict(Xte[cols])[0])
        pred_rows.append({"fold": fold, "donor_id": td, "block": block,
                          "y_true": ages[te][0], "y_pred": pr})
        denom_rows.append({"fold": fold, "block": block,
                           "n_block": len(allcols), "n_screened": len(cols),
                           "best_alpha": a, "best_l1": l})

preds = pd.DataFrame(pred_rows)
# proper fold log line
for fold in range(1, 19):
    sub = preds[preds.fold == fold].set_index("block").y_pred
    log.info("fold %2d: %s", fold,
             " ".join(f"{b}={sub.get(b, np.nan):.1f}" for b in
                      ["all_features", "gene_celltype_only",
                       "whole_thymus_only", "composition_only", "MeanAgeNull"]))

preds.to_csv(OUT / "fold_predictions_ablation.csv", index=False)
pd.DataFrame(denom_rows).to_csv(OUT / "ablation_fold_denominators.csv", index=False)

# ---------------------------------------------------------------- OOF metrics
rng = np.random.default_rng(SEED)

def boot_metric(yt, yp, B=N_BOOT):
    r2, mae = [], []
    for _ in range(B):
        i = rng.integers(0, len(yt), len(yt))
        r2.append(r2_score(yt[i], yp[i]))
        mae.append(mean_absolute_error(yt[i], yp[i]))
    return np.quantile(r2, [.025, .975]), np.quantile(mae, [.025, .975])

rows = []
for block in list(BLOCKS) + ["MeanAgeNull"]:
    s = preds[preds.block == block]
    yt, yp = s.y_true.values, s.y_pred.values
    ci_r2, ci_mae = boot_metric(yt, yp)
    rows.append({"block": block, "n_donors": len(s),
                 "R2": r2_score(yt, yp), "MAE": mean_absolute_error(yt, yp),
                 "RMSE": np.sqrt(mean_squared_error(yt, yp)),
                 "R2_CI95_low": ci_r2[0], "R2_CI95_high": ci_r2[1],
                 "MAE_CI95_low": ci_mae[0], "MAE_CI95_high": ci_mae[1]})
mets = pd.DataFrame(rows)
mets.to_csv(OUT / "ablation_nested_lodo_metrics.csv", index=False)

# paired donor bootstrap: block vs all_features (R2 and MAE difference)
allp = preds[preds.block == "all_features"].set_index("donor_id")
drows = []
for block in list(BLOCKS) + ["MeanAgeNull"]:
    if block == "all_features":
        continue
    s = preds[preds.block == block].set_index("donor_id").loc[allp.index]
    dr2, dmae = [], []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(allp), len(allp))
        yt = allp.y_true.values[i]
        dr2.append(r2_score(yt, s.y_pred.values[i]) - r2_score(yt, allp.y_pred.values[i]))
        dmae.append(mean_absolute_error(yt, s.y_pred.values[i])
                    - mean_absolute_error(yt, allp.y_pred.values[i]))
    drows.append({"block": block,
                  "dR2_CI95_low": np.quantile(dr2, .025),
                  "dR2_CI95_high": np.quantile(dr2, .975),
                  "dMAE_CI95_low": np.quantile(dmae, .025),
                  "dMAE_CI95_high": np.quantile(dmae, .975)})
pd.DataFrame(drows).to_csv(OUT / "ablation_bootstrap_deltas.csv", index=False)

log.info("\nOOF metrics:\n%s", mets.to_string(index=False))
log.info("=== R1 ablation COMPLETE ===")

#!/usr/bin/env python
"""Step 11 (P0+P1 fixed): Outer LODO evaluation with outer-training-only feature screening and inner-CV hyperparameter tuning.

Design (from statistical design v2.1):
  1. PRIMARY: continuous-age regression (18 donors); SECONDARY: binary Young/Old (11 donors, exploratory).
  2. Outer CV = Leave-One-Donor-Out; inner CV = stratified KFold on training donors only.
  3. All feature preprocessing (missingness, variance, selection, scaling, imputation) fit on training donors.
  4. Test donor only transform + predict (no leakage).
  5. n_splits = min(3, n_minority_in_training); if n_splits < 2, skip hyperparameter search.
  6. rank_pct denominator = number of features AFTER pipeline selection (VarianceThreshold).
  7. Female-only sensitivity: re-run entire pipeline from scratch (no reuse of main features/scaler/hyperparams).
  8. LABEL_MAP = {"Young": 0, "Old": 1} (positive = Old = 1).

Inputs:
  04_machine_learning/dataset/feature_matrix_all_combined.csv.gz
  04_machine_learning/dataset/feature_dictionary.csv
  04_machine_learning/dataset/donor_metadata.csv

Outputs:
  04_machine_learning/predictions/fold_predictions_regression.csv
  04_machine_learning/predictions/fold_predictions_classification.csv
  04_machine_learning/predictions/fold_predictions_regression_female_only.csv  (NEW)
  04_machine_learning/predictions/fold_predictions_classification_female_only.csv  (NEW)
  04_machine_learning/metrics/regression_metrics_oof.csv
  04_machine_learning/metrics/classification_metrics_oof.csv
  04_machine_learning/metrics/fold_level_importance.csv
  04_machine_learning/metrics/fold_feature_denominators.csv
  10_results/logs/11_nested_ML.log
"""
from __future__ import annotations

import logging
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    roc_auc_score, average_precision_score, balanced_accuracy_score, f1_score,
    confusion_matrix,
)
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold
from sklearn.svm import LinearSVR, LinearSVC
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from xgboost import XGBRegressor, XGBClassifier

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def _env(key):
    val = os.environ.get(key)
    if not val:
        raise RuntimeError(f"Required env var {key} is not set. Run: source scripts/00_project_config.sh")
    return Path(val)

PROJ = _env("PROJ")
DS = PROJ / "04_machine_learning" / "dataset"
OUT = PROJ / "04_machine_learning"
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "predictions").mkdir(parents=True, exist_ok=True)
(OUT / "metrics").mkdir(parents=True, exist_ok=True)
LOGS = PROJ / "10_results" / "logs"
LOGS.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("step11")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler(sys.stdout)
fh = logging.FileHandler(LOGS / "11_nested_ML.log", mode="w")
fmt = logging.Formatter("%(asctime)s  %(levelname)s  %(message)s")
ch.setFormatter(fmt)
fh.setFormatter(fmt)
logger.addHandler(ch)
logger.addHandler(fh)
logger.info("=== Step 11: Outer LODO evaluation with outer-training-only feature screening and inner-CV hyperparameter tuning ===")

SEED = 371
rng = np.random.default_rng(SEED)

NAN_FRAC_CUT = 0.5        # training-donor missingness cut (inside each fold)
TOP_K = 2000              # fold-internal univariate selection cap
INNER_SPLITS_DEFAULT = 3   # max inner CV splits; dynamically reduced per n_minority
N_BOOT = 500
N_PERM = 500

LABEL_MAP = {"Young": 0, "Old": 1}   # P0-2: positive=Old=1 everywhere

# ------------------------------------------------------------------
# 1. Load data  (labels come from Step 05 frozen cohort files)
# ------------------------------------------------------------------
features = pd.read_csv(DS / "feature_matrix_all_combined.csv.gz", index_col=0)
meta = pd.read_csv(DS / "donor_metadata.csv", index_col=0)
features = features.loc[meta.index]
logger.info(f"Features: {features.shape}")

meta["age_years"] = meta["age_years"].astype(float)

# PRIMARY: continuous-age regression cohort is FROZEN in Step 05. Read the
# frozen cohort file; do NOT re-derive Young/Old cut-offs here.
META = PROJ / "01_raw_processing" / "metadata"
reg_file = META / "05B_final_regression_cohort.csv"
if not reg_file.exists():
    raise FileNotFoundError(
        "Step 05 frozen regression cohort file not found; run Step 05 first: "
        f"{reg_file}")
reg_cohort = pd.read_csv(reg_file)
reg_donors = reg_cohort["donor_id"].astype(str).tolist()
reg_ages = reg_cohort["age_years"].astype(float).values
reg_groups = np.array(reg_donors)
# CRITICAL alignment fix (2026-09-17): `features` rows are in lexicographic donor
# order (donor1, donor10, ...), while reg_donors/reg_ages come from the Step 05
# cohort file sorted by age. LODO position indices must index an X whose rows
# follow reg_donors order, or labels get permuted across donors.
X_reg = features.loc[reg_donors]
# P0 donor-alignment audit (second-round figure review 2026-09-17): explicit,
# exhaustive checks BEFORE any fold split is built.
features.index = features.index.astype(str)
meta.index = meta.index.astype(str)
assert len(reg_donors) == 18
assert len(set(reg_donors)) == 18
assert set(reg_donors) == set(features.index)
assert X_reg.index.tolist() == reg_donors
_expected_age = meta.loc[reg_donors, "age_years"].astype(float).to_numpy()
assert np.allclose(_expected_age, np.asarray(reg_ages, dtype=float)), \
    "reg_ages do not match donor_metadata age_years in cohort order"
logger.info(f"Regression cohort: {len(reg_donors)} donors (frozen Step 05); "
            f"X aligned to cohort order; donor/age alignment asserts passed")

# SECONDARY (EXPLORATORY ONLY): binary Young (<18) vs Old (>=40) labels are
# FROZEN in Step 05. Read them; never re-derive from age here.
bin_file = META / "05C_final_binary_classification_cohort.csv"
if bin_file.exists():
    bin_cohort = pd.read_csv(bin_file)
else:
    raise FileNotFoundError(
        f"Step 05 frozen binary cohort file not found; run Step 05 first: {bin_file}")
bin_cohort = bin_cohort.astype({"donor_id": str})
label_col = "label_binary" if "label_binary" in bin_cohort.columns else \
            "binary_label" if "binary_label" in bin_cohort.columns else "label"
if label_col not in bin_cohort.columns:
    raise KeyError(
        f"Step 05 frozen binary cohort lacks label column '{label_col}'; "
        "columns: " + ", ".join(bin_cohort.columns))
bin_lookup = dict(zip(bin_cohort["donor_id"], bin_cohort[label_col]))
bin_donors = [d for d in reg_donors if d in bin_lookup
              and bin_lookup[d] == bin_lookup[d]]  # drop NaN labels
# P0-2: map string labels to integers via LABEL_MAP
bin_labels = np.array([LABEL_MAP.get(bin_lookup[d], -1) for d in bin_donors], dtype=int)
if (bin_labels == -1).any():
    bad = [d for d, l in zip(bin_donors, bin_labels) if l == -1]
    raise ValueError(f"Unmapped binary labels for donors: {bad}. LABEL_MAP: {LABEL_MAP}")
assert len(bin_donors) == len(set(bin_donors)) == 11
_bin_age = meta.loc[bin_donors, "age_years"].astype(float).to_numpy()
assert np.all((_bin_age < 18) == (bin_labels == LABEL_MAP["Young"]))
assert np.all((_bin_age >= 40) == (bin_labels == LABEL_MAP["Old"]))
X_bin = features.loc[bin_donors]
assert X_bin.index.tolist() == bin_donors
logger.info(f"Classification cohort: {len(bin_donors)} donors "
            f"(Young={int((bin_labels==LABEL_MAP['Young']).sum())}, "
            f"Old={int((bin_labels==LABEL_MAP['Old']).sum())}) "
            f"(frozen Step 05, exploratory, Old=1/Young=0); alignment asserts passed")

# Authoritative feature dictionary (Step 10) -> feature type / gene / celltype
fdict = pd.read_csv(DS / "feature_dictionary.csv")
feat_type = dict(zip(fdict["feature"], fdict["feature_type"]))
feat_gene = dict(zip(fdict["feature"], fdict["gene"]))
feat_ct = dict(zip(fdict["feature"], fdict["celltype"]))
logger.info(f"Feature dictionary: {len(fdict)} features")

# Feature-type blocks for ablations
def feature_subsets(feats):
    return {
        "composition": [f for f in feats if feat_type.get(f) == "proportion"],
        "whole_thymus": [f for f in feats if feat_type.get(f) == "whole_thymus"],
        "gene_celltype": [f for f in feats if feat_type.get(f) == "gene_celltype"],
    }

# ------------------------------------------------------------------
# 2. Model factories with small inner-CV grids (n=17 training donors)
# ------------------------------------------------------------------
def make_pipe(model):
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("variance", VarianceThreshold(threshold=0.0)),
        ("scale", StandardScaler()),
        ("model", model),
    ])

def reg_models():
    return {
        "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000, random_state=SEED),
        "LinearSVR":  LinearSVR(C=1.0, max_iter=10000, random_state=SEED),
        "RF":         RandomForestRegressor(n_estimators=200, max_depth=5,
                                            min_samples_leaf=3, random_state=SEED, n_jobs=-1),
        "XGB":        XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.1,
                                   subsample=0.8, random_state=SEED, verbosity=0),
    }

REG_GRIDS = {
    "ElasticNet": {"model__alpha": [0.01, 0.1, 1.0],
                   "model__l1_ratio": [0.1, 0.5, 0.9]},
    "LinearSVR":  {"model__C": [0.01, 0.1, 1.0, 10.0],
                   "model__epsilon": [0.0, 1.0, 3.0]},
    "RF":         {"model__max_depth": [3, 5], "model__min_samples_leaf": [3, 5]},
    "XGB":        {"model__max_depth": [3, 4], "model__learning_rate": [0.05, 0.1]},
}

def clf_models():
    return {
        "LogisticReg": LogisticRegression(C=1.0, max_iter=3000, random_state=SEED),
        "LinearSVC":   LinearSVC(C=1.0, max_iter=10000, random_state=SEED),
        "RF":          RandomForestClassifier(n_estimators=200, max_depth=5,
                                              min_samples_leaf=3, random_state=SEED, n_jobs=-1),
        "XGB":         XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                                     subsample=0.8, objective="binary:logistic",
                                     random_state=SEED, verbosity=0, eval_metric="logloss"),
    }

CLF_GRIDS = {
    "LogisticReg": {"model__C": [0.1, 1.0, 10.0]},
    "LinearSVC":   {"model__C": [0.1, 1.0, 10.0]},
    "RF":          {"model__max_depth": [3, 5], "model__min_samples_leaf": [3, 5]},
    "XGB":         {"model__max_depth": [3, 4], "model__learning_rate": [0.05, 0.1]},
}


def inner_cv_select(X_tr, y_tr, model, grid, task="reg"):
    """Stratified inner CV on training donors for a small hyperparameter grid.

    P1: n_splits = min(INNER_SPLITS_DEFAULT, n_minority_in_training); if
    n_splits < 2, skip hyperparameter search and refit with frozen defaults.

    Returns (best_pipe_refit, best_params, inner_score).
    best_pipe_refit is a NEW estimator with best_params, fit on ALL X_tr/y_tr
    (the full outer-training donors), NOT the last inner-fold estimator.
    """
    from itertools import product

    # P1: dynamic inner-CV splits from minority class size (classification)
    # or number of distinct training outcomes (regression)
    if task == "clf":
        uniq, counts = np.unique(y_tr, return_counts=True)
        n_minority = int(counts.min())
        n_splits = min(INNER_SPLITS_DEFAULT, n_minority)
    else:
        n_splits = min(INNER_SPLITS_DEFAULT, len(np.unique(y_tr)))

    if n_splits < 2:
        best_pipe = make_pipe(model)
        best_pipe.fit(X_tr, y_tr)
        logger.info("    inner CV n_splits=%d < 2 -> frozen defaults, no tuning", n_splits)
        return best_pipe, {}, -np.inf

    if task == "reg":
        scorer = lambda est, X, y: -mean_squared_error(y, est.predict(X))
        fold_splitter = lambda X, y: _stratified_donor_splits(X, y, n_splits)
    else:
        scorer = lambda est, X, y: roc_auc_score(y, _predict_score(est, X))
        fold_splitter = lambda X, y: _stratified_donor_splits(X, y, n_splits)

    best_score, best_params = -np.inf, None
    keys = list(grid.keys())
    vals = [grid[k] for k in keys]
    for combo in product(*vals):
        # Strip "model__" prefix — we're setting params on the bare estimator
        params = {k.replace("model__", ""): v for k, v in zip(keys, combo)}
        scores = []
        try:
            est = clone(model)
            est.set_params(**params)
            pipe = make_pipe(est)
            for tr, va in fold_splitter(X_tr, y_tr):
                Xt, Xv = X_tr.iloc[tr], X_tr.iloc[va]
                yt, yv = y_tr[tr], y_tr[va]
                pipe.fit(Xt, yt)
                scores.append(scorer(pipe, Xv, yv))
        except Exception as e:  # noqa: BLE001
            logger.debug(f"  inner CV combo {params} failed: {e}")
            continue
        s = float(np.mean(scores)) if scores else -np.inf
        if s > best_score:
            best_score = s
            best_params = params

    # Refit on ALL outer-training donors with best params
    if best_params is not None:
        est = clone(model)
        est.set_params(**best_params)
        best_pipe = make_pipe(est)
        best_pipe.fit(X_tr, y_tr)
    else:
        best_pipe = make_pipe(model)
        best_pipe.fit(X_tr, y_tr)
        best_score = -np.inf
    return best_pipe, best_params or {}, best_score


def _stratified_donor_splits(X, y, n_splits):
    """Stratified KFold on donor indices (works for both reg/class labels)."""
    idx = np.arange(len(y))
    # stratify by binned age for regression (3 bins) or by class label
    if np.all(np.isin(y, [0, 1])):
        strat = y
    else:
        strat = pd.qcut(pd.Series(y), q=min(3, len(np.unique(y))), labels=False,
                        duplicates="drop").values
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    return skf.split(np.zeros((len(y), 1)), strat)


def _predict_score(pipe, X):
    m = pipe.named_steps["model"]
    if hasattr(m, "predict_proba"):
        return pipe.predict_proba(X)[:, 1]
    return pipe.decision_function(X)


# ------------------------------------------------------------------
# 3. Fold-internal feature selection (training donors only)
# ------------------------------------------------------------------
def select_features_fold(X_tr, y_tr, top_k=TOP_K):
    """Missingness -> variance -> univariate age association (Spearman |rho|), train-only.

    All steps use vectorised numpy operations for speed on high-dimensional data.

    Returns (keep_cols, selected_cols, n_before, n_after_missing, n_after_sel).
    """
    n_before = X_tr.shape[1]
    arr = X_tr.values  # (n_donors, n_features) — single copy

    # 3a. missingness (training donors only)
    nan_frac = np.isnan(arr).mean(axis=0)
    miss_mask = nan_frac <= NAN_FRAC_CUT
    n_after_missing = int(miss_mask.sum())
    if n_after_missing == 0:
        return [], [], n_before, 0, 0

    # 3b. variance (training donors only) on non-missing columns
    arr_m = arr[:, miss_mask]
    col_var = np.nanvar(arr_m, axis=0)
    var_mask = col_var > 0.0
    arr_mv = arr_m[:, var_mask]
    keep_idx = np.where(miss_mask)[0][var_mask]
    keep_cols = X_tr.columns[keep_idx].tolist()

    # 3c. univariate Spearman |rho| via vectorised numpy ranks (no scipy loop)
    #     Median-impute remaining NaNs once for the kept columns.
    col_med = np.nanmedian(arr_mv, axis=0)
    col_med = np.where(np.isnan(col_med), 0.0, col_med)
    nan_mask_mv = np.isnan(arr_mv)
    arr_mv = np.where(nan_mask_mv, col_med, arr_mv)

    n, p = arr_mv.shape
    # Ranking along donor axis (axis=0) — fully vectorised
    rnk_feat = np.empty_like(arr_mv)
    for j in range(p):
        rnk_feat[:, j] = stats.rankdata(arr_mv[:, j])
    rnk_age = stats.rankdata(y_tr).reshape(-1, 1)

    # Spearman rho = Pearson of ranks
    rf = rnk_feat - rnk_feat.mean(axis=0)
    ra = rnk_age - rnk_age.mean()
    rf_ss = np.sqrt((rf ** 2).sum(axis=0))  # (p,)
    ra_ss = np.sqrt((ra ** 2).sum())
    denom = rf_ss * ra_ss
    denom[denom == 0.0] = 1.0
    rho = np.abs((rf * ra).sum(axis=0) / denom)

    # Top-K
    if len(rho) <= top_k:
        sel_idx = np.arange(len(rho))
    else:
        sel_idx = np.argpartition(rho, -top_k)[-top_k:]
    selected = [keep_cols[i] for i in sel_idx]
    return keep_cols, selected, n_before, n_after_missing, len(selected)


# ------------------------------------------------------------------
# 4. Bootstrap CI + permutation test helpers
# ------------------------------------------------------------------
def bootstrap_ci_reg(y_true, y_pred, n_boot=N_BOOT, seed=SEED):
    rng2 = np.random.default_rng(seed)
    r2s, rhos = [], []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng2.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        r2s.append(r2_score(y_true[idx], y_pred[idx]))
        rho, _ = stats.spearmanr(y_true[idx], y_pred[idx])
        rhos.append(rho)
    return (np.quantile(r2s, [0.025, 0.975]),
            np.quantile(rhos, [0.025, 0.975]))


def pred_association_permutation_p_reg(y_true, y_pred, n_perm=N_PERM, seed=SEED):
    """OOF PREDICTION-ASSOCIATION permutation p-value (NOT a full-model
    permutation test).

    The fixed OOF predictions are permuted against the observed ages and the
    R2 of the shuffled association is compared to the observed OOF R2. This
    asks whether the held-out predictions are associated with age beyond chance,
    but it does NOT refit the pipeline (screening + tuning + training) under
    label permutation, so it must not be reported as a full model-permutation
    test. A full-pipeline permutation would be far more expensive."""
    rng2 = np.random.default_rng(seed)
    obs_r2 = r2_score(y_true, y_pred)
    cnt = 0
    for _ in range(n_perm):
        perm = rng2.permutation(y_pred)
        cnt += (r2_score(y_true, perm) >= obs_r2)
    return (cnt + 1) / (n_perm + 1)


def bootstrap_ci_clf(y_true, y_prob, n_boot=N_BOOT, seed=SEED):
    rng2 = np.random.default_rng(seed)
    aucs = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng2.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        try:
            aucs.append(roc_auc_score(y_true[idx], y_prob[idx]))
        except Exception:  # noqa: BLE001
            continue
    return np.quantile(aucs, [0.025, 0.975]) if aucs else (np.nan, np.nan)


def pred_association_permutation_p_clf(y_true, y_prob, n_perm=N_PERM, seed=SEED):
    """OOF PREDICTION-ASSOCIATION permutation p-value (NOT a full-model
    permutation test): fixed OOF probabilities are permuted against labels.
    See pred_association_permutation_p_reg for the interpretation caveat."""
    rng2 = np.random.default_rng(seed)
    obs = roc_auc_score(y_true, y_prob)
    cnt = 0
    for _ in range(n_perm):
        perm = rng2.permutation(y_prob)
        try:
            cnt += (roc_auc_score(y_true, perm) >= obs)
        except Exception:  # noqa: BLE001
            continue
    return (cnt + 1) / (n_perm + 1)


# ------------------------------------------------------------------
# 5. Regression: Outer LODO (18 folds) + inner-CV hyperparameter tuning
# ------------------------------------------------------------------
logger.info("\n--- REGRESSION (Outer LODO + inner-CV hyperparameter tuning, 18 folds) ---")
logo = LeaveOneGroupOut()
reg_fold_preds, reg_fold_importance = [], []
fold_denom_rows = []
fold_audit_rows = []
fold_selected_features = {}
all_feats = features.columns.tolist()
subsets_all = feature_subsets(all_feats)

for fold_idx, (train_idx, test_idx) in enumerate(logo.split(X_reg, reg_ages, reg_groups)):
    test_donor = reg_donors[test_idx[0]]
    X_tr_full, X_te_full = X_reg.iloc[train_idx], X_reg.iloc[test_idx]
    y_tr, y_te = reg_ages[train_idx], reg_ages[test_idx]
    # per-fold alignment asserts: row order, donor identity, age identity
    assert X_tr_full.index.tolist() == [reg_donors[i] for i in train_idx]
    assert X_te_full.index.tolist() == [test_donor]
    assert float(y_te[0]) == float(meta.loc[test_donor, "age_years"])
    train_donors_fold = [reg_donors[i] for i in train_idx]
    logger.info(f"  Fold {fold_idx+1}: train {len(train_idx)}, test={test_donor}")

    # ---- fold-internal feature selection (train only) ----
    keep_cols, sel_cols, n_bef, n_miss, n_sel = select_features_fold(X_tr_full, y_tr)
    fold_selected_features[fold_idx + 1] = sel_cols
    logger.info(f"    feature selection: {n_bef} -> {n_miss} (missing) -> {n_sel} (selected)")

    # ---- ablations on the SAME selected features (subset by feature type) ----
    subsets = {
        "all_features": sel_cols,
        "composition_only": [f for f in sel_cols if feat_type.get(f) == "proportion"],
        "whole_thymus_only": [f for f in sel_cols if feat_type.get(f) == "whole_thymus"],
        "gene_celltype_only": [f for f in sel_cols if feat_type.get(f) == "gene_celltype"],
    }
    if fold_idx == 0:
        logger.info("    ablation subsets sizes: " +
                    ", ".join(f"{k}={len(v)}" for k, v in subsets.items()))

    reg_models_this = reg_models()
    for name, model_factory in reg_models_this.items():
        # ---- inner CV hyperparameter selection on training donors ----
        # Returns a pipeline REFIT on ALL outer-training donors with best params
        best_pipe, best_params, inner_score = inner_cv_select(
            X_tr_full[sel_cols], y_tr, reg_models_this[name], REG_GRIDS[name], task="reg")
        pred = best_pipe.predict(X_te_full[sel_cols])[0]
        reg_fold_preds.append({
            "fold": fold_idx + 1, "donor_id": test_donor, "model": name,
            "y_true": y_te[0], "y_pred": pred, "abs_error": abs(y_te[0] - pred),
        })
        # importance with TRUE fold denominator
        model = best_pipe.named_steps["model"]
        var_mask = best_pipe.named_steps["variance"].get_support()
        feat_names = X_tr_full[sel_cols].columns[var_mask]
        if hasattr(model, "feature_importances_"):
            imp = model.feature_importances_
            sign = np.full(len(feat_names), np.nan)
        elif hasattr(model, "coef_"):
            imp = np.abs(model.coef_.ravel())
            sign = np.sign(model.coef_.ravel())
        else:
            imp = np.zeros(len(feat_names))
            sign = np.full(len(feat_names), np.nan)
        imp_df = pd.DataFrame({"model": name, "fold": fold_idx + 1,
                               "feature": feat_names, "importance": imp,
                               "sign": sign})
        imp_df["rank"] = imp_df["importance"].rank(ascending=False, method="first").astype(int)
        # P0 (2026-09-15): column name must match Step 12's denominator column.
        # This is the TRUE post-pipeline denominator (features retained after
        # VarianceThreshold), NOT the top-200 truncated set and NOT max(rank).
        imp_df["n_features_after_selection"] = len(feat_names)
        imp_df["rank_pct"] = imp_df["rank"] / max(1, len(feat_names))
        imp_df = imp_df.sort_values("rank").head(200)
        reg_fold_importance.append(imp_df)
        # P1 (2026-09-16): distinguish the two post-selection stages explicitly.
        # n_sel = features after univariate screening (BEFORE the pipeline);
        # len(feat_names) = features actually used after pipeline VarianceThreshold.
        fold_denom_rows.append({"model": name, "fold": fold_idx + 1,
                                "n_features_before_filter": n_bef,
                                "n_features_after_missing_filter": n_miss,
                                "n_features_after_univariate_selection": n_sel,
                                "n_features_after_pipeline_variance": len(feat_names),
                                "best_params": str(best_params) if best_params else "",
                                "inner_cv_score": round(float(inner_score), 4) if np.isfinite(inner_score) else None})
        fold_audit_rows.append({
            "fold_id": fold_idx + 1,
            "test_donor": test_donor,
            "test_age": float(y_te[0]),
            "train_donors": ";".join(train_donors_fold),
            "n_train_donors": len(train_donors_fold),
            "model": name,
            "n_selected_features": len(sel_cols),
            "n_features_used": int(len(feat_names)),
            "inner_best_params": str(best_params) if best_params else "frozen_defaults",
            "y_pred": float(pred),
            "abs_error": float(abs(y_te[0] - pred)),
            "random_seed": SEED,
        })
        logger.info(f"    {name}: pred={pred:.1f} (true={y_te[0]}), innerCV={inner_score:.3f}")

    # ---- mean-age null baseline (no features) ----
    reg_fold_preds.append({
        "fold": fold_idx + 1, "donor_id": test_donor, "model": "MeanAgeNull",
        "y_true": y_te[0], "y_pred": float(np.mean(y_tr)), "abs_error": abs(y_te[0] - np.mean(y_tr)),
    })

    # ---- ablation models (ElasticNet only, on fold-selected subsets) ----
    if fold_idx == 0:
        logger.info("  Ablations (ElasticNet on same fold-selected features):")
    for abl, cols in subsets.items():
        if len(cols) < 5:
            reg_fold_preds.append({
                "fold": fold_idx + 1, "donor_id": test_donor,
                "model": f"Ablation_ElasticNet_{abl}",
                "y_true": y_te[0], "y_pred": float(np.mean(y_tr)),
                "abs_error": abs(y_te[0] - np.mean(y_tr))})
            continue
        apipe = make_pipe(ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000, random_state=SEED))
        apipe.fit(X_tr_full[cols], y_tr)
        apred = apipe.predict(X_te_full[cols])[0]
        reg_fold_preds.append({
            "fold": fold_idx + 1, "donor_id": test_donor,
            "model": f"Ablation_ElasticNet_{abl}",
            "y_true": y_te[0], "y_pred": float(apred),
            "abs_error": abs(y_te[0] - apred)})
    logger.info(f"    mean-age null: pred={np.mean(y_tr):.1f}")

reg_preds_df = pd.DataFrame(reg_fold_preds)
reg_preds_df.to_csv(OUT / "predictions" / "fold_predictions_regression.csv", index=False)

pd.DataFrame(fold_denom_rows).to_csv(OUT / "metrics" / "fold_feature_denominators.csv", index=False)

# ------------------------------------------------------------------
# 5b. Donor-alignment audit table (second-round review P0, section 3.1).
#     One row per fold x model, plus the frozen fold-selected feature lists.
# ------------------------------------------------------------------
AUDIT = OUT / "audit"
AUDIT.mkdir(parents=True, exist_ok=True)
# pooled-OOF post-checks for every model: exactly 18 unique donors, one row each,
# and y_true identical to the frozen metadata age.
for _name in list(reg_models().keys()) + ["MeanAgeNull"]:
    _sub = reg_preds_df[reg_preds_df["model"] == _name]
    assert _sub["donor_id"].nunique() == 18, f"{_name}: not 18 unique OOF donors"
    assert len(_sub) == 18, f"{_name}: expected 18 OOF rows, got {len(_sub)}"
    assert _sub["donor_id"].value_counts().eq(1).all(), f"{_name}: duplicated donor rows"
    _check_age = meta.loc[_sub["donor_id"].astype(str), "age_years"].astype(float).to_numpy()
    assert np.allclose(_check_age, _sub["y_true"].to_numpy()), \
        f"{_name}: OOF y_true misaligned to metadata ages"
logger.info("Pooled-OOF alignment asserts passed (18 unique donors, one prediction each, "
            "y_true == frozen metadata age for every model)")

# persist the fold-selected feature lists (identical for all models within a fold)
for fold_id, cols in fold_selected_features.items():
    test_d = reg_donors[fold_id - 1]
    pd.DataFrame({"feature": cols}).to_csv(
        AUDIT / f"fold_{fold_id:02d}_selected_features.csv", index=False)
    for r in fold_audit_rows:
        if int(r["fold_id"]) == fold_id and "selected_features_file" not in r:
            r["selected_features_file"] = f"fold_{fold_id:02d}_selected_features.csv"
pd.DataFrame(fold_audit_rows).to_csv(AUDIT / "donor_alignment_audit.tsv",
                                     sep="\t", index=False)
logger.info("Wrote per-fold audit: %s (%d rows) + %d fold feature lists",
            AUDIT / "donor_alignment_audit.tsv", len(fold_audit_rows),
            len(fold_selected_features))

# OOF regression metrics + CI + permutation
oof_reg, oof_reg_ci, oof_reg_perm = [], [], []
models_reg = list(reg_models().keys()) + ["MeanAgeNull"]
for name in models_reg:
    sub = reg_preds_df[reg_preds_df["model"] == name]
    y_true, y_pred = sub["y_true"].values, sub["y_pred"].values
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    rho, _ = stats.spearmanr(y_true, y_pred)
    oof_reg.append({"model": name, "n_donors": len(sub),
                    "RMSE": round(rmse, 4), "MAE": round(mae, 4),
                    "R2": round(r2, 4), "Spearman_rho": round(rho, 4)})
    ci_r2, ci_rho = bootstrap_ci_reg(y_true, y_pred)
    p_perm = pred_association_permutation_p_reg(y_true, y_pred)
    oof_reg_ci.append({"model": name,
                       "R2_CI95_low": round(ci_r2[0], 4), "R2_CI95_high": round(ci_r2[1], 4),
                       "Spearman_CI95_low": round(ci_rho[0], 4), "Spearman_CI95_high": round(ci_rho[1], 4)})
    oof_reg_perm.append({
        "model": name,
        "R2_fixedOOF_prediction_association_perm_p": round(p_perm, 4),
        "test_type": ("FIXED-OOF prediction-age association permutation: the 18 "
                      "held-out predictions are held fixed and permuted against age. "
                      "This is NOT a model-training permutation test (the fold-internal "
                      "screening / tuning / refit are NOT repeated under label "
                      "permutation); it must not be reported as model-level significance."),
    })
    logger.info(f"  OOF {name}: R2={r2:.3f}, RMSE={rmse:.3f}, MAE={mae:.3f}, ρ={rho:.3f}, "
                f"pred-association perm_p={p_perm:.3f}")

oof_reg_df = pd.DataFrame(oof_reg)
oof_reg_df.to_csv(OUT / "metrics" / "regression_metrics_oof.csv", index=False)
pd.DataFrame(oof_reg_ci).to_csv(OUT / "metrics" / "regression_metrics_oof_ci.csv", index=False)
pd.DataFrame(oof_reg_perm).to_csv(OUT / "metrics" / "regression_permutation_test.csv", index=False)

# Fold-level metrics
fold_reg = []
for (fold, model), grp in reg_preds_df.groupby(["fold", "model"]):
    fold_reg.append({"fold": fold, "model": model,
                     "MAE": round(mean_absolute_error(grp["y_true"], grp["y_pred"]), 4),
                     "abs_error": round(grp["abs_error"].values[0], 4)})
pd.DataFrame(fold_reg).to_csv(OUT / "metrics" / "regression_metrics_by_fold.csv", index=False)

# Fold-level importance CSV (for Step 12 rank-based aggregation)
importance_df = pd.concat(reg_fold_importance, ignore_index=True)
importance_df.to_csv(OUT / "metrics" / "fold_level_importance.csv", index=False)
logger.info(f"Fold-level importance saved: {importance_df.shape}")

# Ablation comparison (OOF, mean over folds for each ablation model)
abl_rows = []
for model in sorted(reg_preds_df["model"].unique()):
    sub = reg_preds_df[reg_preds_df["model"] == model]
    if len(sub) == 0:
        continue
    abl_rows.append({"model": model,
                     "R2": round(r2_score(sub["y_true"], sub["y_pred"]), 4),
                     "MAE": round(mean_absolute_error(sub["y_true"], sub["y_pred"]), 4)})
pd.DataFrame(abl_rows).to_csv(OUT / "metrics" / "ablation_comparison.csv", index=False)

# ------------------------------------------------------------------
# 6. Classification: Outer LODO (11 donors) + inner-CV tuning (exploratory)
# ------------------------------------------------------------------
logger.info("\n--- CLASSIFICATION (Outer LODO + inner-CV tuning, 11 folds, exploratory) ---")
clf_fold_preds = []
for fold_idx, (train_idx, test_idx) in enumerate(logo.split(X_bin, bin_labels, np.array(bin_donors))):
    test_donor = bin_donors[test_idx[0]]
    X_tr_full, X_te_full = X_bin.iloc[train_idx], X_bin.iloc[test_idx]
    y_tr, y_te = bin_labels[train_idx], bin_labels[test_idx]
    logger.info(f"  Fold {fold_idx+1}: train {len(train_idx)}, test={test_donor}")

    # fold-internal feature selection (train only)
    _, sel_cols, _, _, _ = select_features_fold(X_tr_full, y_tr)

    clf_models_this = clf_models()
    for name, model_factory in clf_models_this.items():
        best_pipe, _, inner_score = inner_cv_select(
            X_tr_full[sel_cols], y_tr, clf_models_this[name], CLF_GRIDS[name], task="clf")
        y_score = _predict_score(best_pipe, X_te_full[sel_cols])
        # y_score from decision_function for SVC / prob for LR/RF/XGB; do NOT
        # call it calibrated probability for SVC.
        prob = float(y_score[0])
        if hasattr(best_pipe.named_steps["model"], "predict_proba"):
            pred_label = int((prob >= 0.5))
        else:
            pred_label = int((best_pipe.decision_function(X_te_full[sel_cols]) >= 0)[0])
        clf_fold_preds.append({
            "fold": fold_idx + 1, "donor_id": test_donor, "model": name,
            "y_true": int(y_te[0]), "y_score": prob, "y_pred": pred_label,
        })
        logger.info(f"    {name}: score={prob:.3f} (true={int(y_te[0])}), innerCV={inner_score:.3f}")

clf_preds_df = pd.DataFrame(clf_fold_preds)
clf_preds_df.to_csv(OUT / "predictions" / "fold_predictions_classification.csv", index=False)

oof_clf, oof_clf_ci, oof_clf_perm = [], [], []
models_clf = list(clf_models().keys())
for name in models_clf:
    sub = clf_preds_df[clf_preds_df["model"] == name]
    y_true, y_score, y_pred = sub["y_true"].values, sub["y_score"].values, sub["y_pred"].values
    if len(set(y_true)) < 2:
        logger.warning(f"  {name}: single class in OOF")
        continue
    try:
        auc = roc_auc_score(y_true, y_score)
    except Exception:
        auc = np.nan
    try:
        pr_auc = average_precision_score(y_true, y_score)
    except Exception:
        pr_auc = np.nan
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    oof_clf.append({"model": name, "n_donors": len(sub),
                    "AUC": round(auc, 4), "PR_AUC": round(pr_auc, 4),
                    "Balanced_Accuracy": round(bal_acc, 4), "F1": round(f1, 4),
                    "Sensitivity": round(sens, 4) if not np.isnan(sens) else np.nan,
                    "Specificity": round(spec, 4) if not np.isnan(spec) else np.nan})
    ci_auc = bootstrap_ci_clf(y_true, y_score)
    p_perm = pred_association_permutation_p_clf(y_true, y_score)
    oof_clf_ci.append({"model": name, "AUC_CI95_low": round(ci_auc[0], 4),
                       "AUC_CI95_high": round(ci_auc[1], 4)})
    oof_clf_perm.append({
        "model": name,
        "AUC_fixedOOF_prediction_association_perm_p": round(p_perm, 4),
        "test_type": ("FIXED-OOF prediction-label association permutation: OOF scores "
                      "are held fixed and permuted against labels. NOT a model-training "
                      "permutation test (no fold-internal screening/tuning/refit under "
                      "permutation); exploratory binary task only, must not be reported "
                      "as model-level significance."),
    })
    logger.info(f"  OOF {name}: AUC={auc:.3f}, BalAcc={bal_acc:.3f}, F1={f1:.3f}, "
                f"pred-association perm_p={p_perm:.3f}")

oof_clf_df = pd.DataFrame(oof_clf)
oof_clf_df.to_csv(OUT / "metrics" / "classification_metrics_oof.csv", index=False)
pd.DataFrame(oof_clf_ci).to_csv(OUT / "metrics" / "classification_metrics_oof_ci.csv", index=False)
pd.DataFrame(oof_clf_perm).to_csv(OUT / "metrics" / "classification_permutation_test.csv", index=False)

# ------------------------------------------------------------------
# 7. Female-only sensitivity ML (P1): full independent rerun on the frozen
#    05G female-only cohort. NEVER reuses main-model features, scaler,
#    imputer, or hyperparameters — each outer fold re-runs missingness /
#    variance / Spearman selection -> inner-CV tuning (or frozen defaults)
#    -> refit -> predict held-out female donor.
# ------------------------------------------------------------------
fem_file = META / "05G_female_only_sensitivity_cohort.csv"
SENS_OUT = OUT / "sensitivity_female"
(SENS_OUT / "predictions").mkdir(parents=True, exist_ok=True)
(SENS_OUT / "metrics").mkdir(parents=True, exist_ok=True)

if fem_file.exists():
    fem_cohort = pd.read_csv(fem_file)
    fem_cohort = fem_cohort.astype({"donor_id": str})
    fem_reg = fem_cohort["donor_id"].tolist()
    fem_ages = fem_cohort["age_years"].astype(float).values
    fem_groups = np.array(fem_reg)
    X_fem = features.loc[fem_reg]
    assert X_fem.index.tolist() == fem_reg
    assert np.allclose(meta.loc[fem_reg, "age_years"].astype(float).to_numpy(), fem_ages)
    assert (fem_cohort["sex"] == "Female").all()
    logger.info(f"\n--- FEMALE-ONLY SENSITIVITY: {len(fem_reg)} female donors "
                f"(frozen 05G; alignment asserts passed) ---")

    # 7a. continuous regression (always run when sample size meets minimum fit)
    if len(fem_reg) >= 5:
        fem_reg_preds, fem_fold_denom = [], []
        for fold_idx, (tr_i, te_i) in enumerate(logo.split(X_fem, fem_ages, fem_groups)):
            test_donor = fem_reg[te_i[0]]
            Xf_tr, Xf_te = X_fem.iloc[tr_i], X_fem.iloc[te_i]
            yf_tr, yf_te = fem_ages[tr_i], fem_ages[te_i]
            _, sel_f, _, _, _ = select_features_fold(Xf_tr, yf_tr)
            if len(sel_f) < 5:
                continue
            logger.info(f"  [female] Fold {fold_idx+1}: train {len(tr_i)}, test={test_donor}")
            for name, mf in reg_models().items():
                bp, bp_params, bp_score = inner_cv_select(Xf_tr[sel_f], yf_tr, mf, REG_GRIDS[name], task="reg")
                pred_f = bp.predict(Xf_te[sel_f])[0]
                fem_reg_preds.append({"fold": fold_idx + 1, "donor_id": test_donor,
                                      "model": name, "y_true": yf_te[0], "y_pred": pred_f,
                                      "abs_error": abs(yf_te[0] - pred_f)})
                fem_fold_denom.append({"model": name, "fold": fold_idx + 1,
                                       "n_features_after_selection": len(sel_f),
                                       "best_params": str(bp_params)})
        fem_reg_df = pd.DataFrame(fem_reg_preds)
        fem_reg_df.to_csv(SENS_OUT / "predictions" / "fold_predictions_regression.csv", index=False)
        pd.DataFrame(fem_fold_denom).to_csv(SENS_OUT / "metrics" / "fold_feature_denominators.csv", index=False)
        fem_oof = []
        for name in reg_models().keys():
            sub = fem_reg_df[fem_reg_df["model"] == name]
            if len(sub) < 2:
                continue
            fem_oof.append({"model": name, "n_donors": len(sub),
                            "R2": round(r2_score(sub["y_true"], sub["y_pred"]), 4),
                            "RMSE": round(float(np.sqrt(mean_squared_error(sub["y_true"], sub["y_pred"]))), 4),
                            "MAE": round(mean_absolute_error(sub["y_true"], sub["y_pred"]), 4),
                            "Spearman_rho": round(stats.spearmanr(sub["y_true"], sub["y_pred"]).statistic, 4)})
        pd.DataFrame(fem_oof).to_csv(SENS_OUT / "metrics" / "regression_metrics_oof.csv", index=False)
        logger.info("  [female] regression OOF: " +
                    ", ".join(f"{r['model']}: R2={r['R2']}" for r in fem_oof))
    else:
        logger.warning(f"Female-only regression skipped: n={len(fem_reg)} < 5")

    # 7b. binary Young/Old (exploratory): skip with warning if any class < 2 donors
    # P0 (2026-09-15): use the FROZEN 05G label_binary / in_binary_cohort
    # columns. Mid-age female donors have label_binary=NA and must NOT be
    # relabeled Old; they are excluded from the female-only binary task.
    fem_bin_df = fem_cohort.loc[fem_cohort["label_binary"].notna()].copy()
    fem_bin = fem_bin_df["donor_id"].tolist()
    fem_labels = np.array([LABEL_MAP[lbl] for lbl in fem_bin_df["label_binary"]], dtype=int)
    counts = pd.Series(fem_labels).value_counts()
    if len(fem_bin) < 5 or counts.min() < 2:
        logger.warning("Female-only binary classification skipped: too few donors in one class "
                       f"(n={len(fem_bin)}, counts={counts.to_dict()}). No AUC computed.")
    else:
        fem_clf_preds = []
        X_fem_bin = features.loc[fem_bin]
        assert X_fem_bin.index.tolist() == fem_bin
        for fold_idx, (tr_i, te_i) in enumerate(logo.split(X_fem_bin, fem_labels, np.array(fem_bin))):
            test_donor = fem_bin[te_i[0]]
            Xc_tr, Xc_te = X_fem_bin.iloc[tr_i], X_fem_bin.iloc[te_i]
            yc_tr, yc_te = fem_labels[tr_i], fem_labels[te_i]
            _, sel_c, _, _, _ = select_features_fold(Xc_tr, yc_tr)
            if len(sel_c) < 5:
                continue
            for name, mf in clf_models().items():
                bp, _, _ = inner_cv_select(Xc_tr[sel_c], yc_tr, mf, CLF_GRIDS[name], task="clf")
                ysc = _predict_score(bp, Xc_te[sel_c])
                fem_clf_preds.append({"fold": fold_idx + 1, "donor_id": test_donor,
                                      "model": name, "y_true": int(yc_te[0]), "y_score": float(ysc[0])})
        fem_clf_df = pd.DataFrame(fem_clf_preds)
        fem_clf_df.to_csv(SENS_OUT / "predictions" / "fold_predictions_classification.csv", index=False)
        fem_clf_oof = []
        for name in clf_models().keys():
            sub = fem_clf_df[fem_clf_df["model"] == name]
            if len(sub) < 2 or len(set(sub["y_true"])) < 2:
                continue
            try:
                auc = roc_auc_score(sub["y_true"], sub["y_score"])
            except Exception:
                auc = np.nan
            fem_clf_oof.append({"model": name, "n_donors": len(sub), "AUC": round(auc, 4)})
        pd.DataFrame(fem_clf_oof).to_csv(SENS_OUT / "metrics" / "classification_metrics_oof.csv", index=False)
        logger.info("  [female] classification OOF: " +
                    ", ".join(f"{r['model']}: AUC={r['AUC']}" for r in fem_clf_oof))
else:
    logger.warning("Female-only sensitivity skipped: 05G cohort not found at "
                   f"{fem_file} (run Step 05 first)")

# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------
logger.info("\n=== Step 11 COMPLETE ===")
logger.info("Regression (PRIMARY, 18 donors OOF, Outer LODO):")
for _, r in oof_reg_df.iterrows():
    logger.info(f"  {r['model']}: R2={r['R2']}, RMSE={r['RMSE']}, MAE={r['MAE']}, ρ={r['Spearman_rho']}")
logger.info("Classification (exploratory, 11 donors OOF, Outer LODO):")
for _, r in oof_clf_df.iterrows():
    logger.info(f"  {r['model']}: AUC={r['AUC']}, BalAcc={r['Balanced_Accuracy']}")

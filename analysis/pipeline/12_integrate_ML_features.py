#!/usr/bin/env python
"""Step 12 (P0-fixed): integrate ML fold-level feature importance -> consensus.

Review fixes (2026-09-11/12, checklist P0):
  1. CIRCULAR DEPENDENCY REMOVED: feature -> (gene, celltype) mapping now comes
     from the authoritative `04_machine_learning/dataset/feature_dictionary.csv`
     (Step 10). It does NOT read Step 13's token_metadata anymore, so the
     dependency chain is 10 -> 11 -> 12 -> 13.
  2. Consensus threshold TIGHTENED (previously too wide -> 675 features):
       A. fold recurrence: n_outer_folds_selected / 18 >= 0.25 in >=1 model
          AND median rank_pct <= 0.10 in that model, OR
       B. multi-model recurrence: >= 2 models, each with n_folds/18 >= 0.10,
          AND median rank_pct <= 0.25 in each supporting model, AND
       C. direction/coefficient-sign consistency for non-tree models: the sign
          of each model's mean coefficient must agree (where available).
     Rank percentiles use the TRUE fold denominator saved by Step 11
     (rank / n_features_after_selection in each (model, fold)).

Inputs:
  04_machine_learning/metrics/fold_level_importance.csv   (Step 11)
  04_machine_learning/dataset/feature_dictionary.csv      (Step 10)

Outputs:
  04_machine_learning/ML_consensus_features.csv
  04_machine_learning/ML_consensus_genes.csv / .txt
  10_results/logs/12_integrate_ML_features.log
"""
from __future__ import annotations

import logging
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

def _env(key):
    val = os.environ.get(key)
    if not val:
        raise RuntimeError(f"Required env var {key} is not set. Run: source scripts/00_project_config.sh")
    return Path(val)

PROJ = _env("PROJ")
ML = PROJ / "04_machine_learning"
LOGS = PROJ / "10_results" / "logs"
LOGS.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("step12")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler(sys.stdout)
fh = logging.FileHandler(LOGS / "12_integrate_ML_features.log", mode="w")
fmt = logging.Formatter("%(asctime)s  %(levelname)s  %(message)s")
ch.setFormatter(fmt)
fh.setFormatter(fmt)
logger.addHandler(ch)
logger.addHandler(fh)
logger.info("=== Step 12 (P0-fixed): Rank-based ML feature integration ===")

N_FOLDS = 18
RANK_PCT_CUT_SINGLE = 0.10   # rank-based single-model consensus cut
RANK_PCT_CUT_MULTI = 0.25    # rank cut for multi-model consensus
FOLD_RECUR_SINGLE = 0.25     # min fold recurrence (single-model path): >= ceil(18*0.25) = 5 folds
FOLD_RECUR_MULTI = 0.10      # min fold recurrence per model (multi-model path)

fli = pd.read_csv(ML / "metrics" / "fold_level_importance.csv")
logger.info(f"Fold-level importance: {fli.shape} (columns {list(fli.columns)})")

# P0 (2026-09-15): Step 11 now writes `n_features_after_selection` as the TRUE
# post-pipeline denominator in fold_level_importance.csv. Require it rather
# than silently falling back to rank/max(rank) (which was wrong for top-200
# truncated sets). Accept the legacy `n_features_after_pipeline` name too.
if "n_features_after_selection" in fli.columns:
    denom_col = "n_features_after_selection"
elif "n_features_after_pipeline" in fli.columns:
    denom_col = "n_features_after_pipeline"
else:
    raise RuntimeError(
        "fold_level_importance.csv lacks a post-pipeline denominator column "
        "(n_features_after_selection). Re-run Step 11 (2026-09-15 fix).")
fli["rank_pct"] = fli["rank"] / fli[denom_col].clip(lower=1)

# -------------------------------------------------------------------
# Load authoritative feature dictionary (Step 10) for feature parsing
# -------------------------------------------------------------------
fdict = pd.read_csv(ML / "dataset" / "feature_dictionary.csv")
feat_map = {}
for _, r in fdict.iterrows():
    feat_map[r["feature"]] = {
        "gene": r["gene"], "celltype": r["celltype"],
        "feature_type": r["feature_type"],
        "is_whole_thymus": bool(r["is_whole_thymus"]),
        "is_composition": bool(r["is_composition"]),
    }
logger.info(f"Feature dictionary loaded: {len(feat_map)} features")

# -------------------------------------------------------------------
# Per model x feature aggregation
# -------------------------------------------------------------------
rows = []
for (model, feat), grp in fli.groupby(["model", "feature"]):
    # coefficient sign (when the model reports coef_); sign consistency requires
    # every recorded fold to agree
    signs = grp["sign"].dropna() if "sign" in grp.columns else pd.Series(dtype=float)
    n_signed = len(signs)
    sign_consistent = bool(n_signed > 0 and signs.nunique() == 1)
    rows.append({
        "model": model,
        "feature": feat,
        "n_folds_selected": grp["fold"].nunique(),
        "stability": grp["fold"].nunique() / N_FOLDS,
        "median_rank_pct": grp["rank_pct"].median(),
        "min_rank_pct": grp["rank_pct"].min(),
        "mean_importance": grp["importance"].mean(),
        "max_importance": grp["importance"].max(),
        "n_sign_consistent_folds": int(n_signed),
        "sign_consistent": sign_consistent,
    })
agg = pd.DataFrame(rows)
logger.info(f"Aggregated feature-model observations: {len(agg)}")
agg.to_csv(ML / "metrics" / "model_importance_features.csv", index=False)
logger.info(f"Saved: {ML / 'metrics' / 'model_importance_features.csv'}")

# -------------------------------------------------------------------
# Consensus across features (tightened)
# -------------------------------------------------------------------
feat_rows = []
for feat, fg in agg.groupby("feature"):
    n_models = fg["model"].nunique()
    fg_sorted = fg.sort_values("median_rank_pct")

    # path A: single-model strong recurrence
    top = fg_sorted.iloc[0]
    path_A = (
        (top["stability"] >= FOLD_RECUR_SINGLE)
        and (top["median_rank_pct"] <= RANK_PCT_CUT_SINGLE)
    )

    # path B: multi-model recurrence (each supporting model meets min recurrence)
    sup = fg[(fg["stability"] >= FOLD_RECUR_MULTI) & (fg["median_rank_pct"] <= RANK_PCT_CUT_MULTI)]
    path_B = len(sup) >= 2

    # P0-1 (sign veto scoped to SUPPORTING models only): direction/coefficient-
    # sign consistency applies ONLY to the linear models that actually meet the
    # supporting criteria (stability >= FOLD_RECUR_MULTI & median_rank_pct <=
    # RANK_PCT_CUT_MULTI).  Non-supporting linear models do not veto.
    lin_names = ("ElasticNet", "LinearSVR", "LogisticReg", "LinearSVC")
    supporting = fg[(fg["stability"] >= FOLD_RECUR_MULTI) &
                    (fg["median_rank_pct"] <= RANK_PCT_CUT_MULTI)]
    sup_lin = supporting[supporting["model"].isin(lin_names)]
    if len(sup_lin) >= 1:
        dir_ok = bool(sup_lin["sign_consistent"].all())
    else:
        dir_ok = True

    # path A: single-model strong recurrence — a linear best model must also be
    # sign-consistent across its own folds
    best_is_linear = top["model"] in lin_names
    if best_is_linear and top["sign_consistent"] == False:  # noqa: E712
        path_A = False

    is_consensus = bool((path_A or path_B) and dir_ok)
    feat_rows.append({
        "feature": feat,
        "n_models_selected": n_models,
        "n_folds_total_selections": int(fg["n_folds_selected"].sum()),
        "best_median_rank_pct": fg_sorted["median_rank_pct"].iloc[0],
        "best_model": fg_sorted["model"].iloc[0],
        "best_stability": fg_sorted["stability"].iloc[0],
        "best_mean_importance": fg_sorted["mean_importance"].iloc[0],
        "is_single_model_strong": bool(path_A),
        "is_multimodel_recurrent": bool(path_B),
        "direction_sign_ok": dir_ok,
        "is_consensus": is_consensus,
        "models": "|".join(sorted(fg["model"].tolist())),
        "gene": feat_map.get(feat, {}).get("gene"),
        "celltype": feat_map.get(feat, {}).get("celltype"),
        "feature_type": feat_map.get(feat, {}).get("feature_type"),
        "is_whole_thymus": bool(feat_map.get(feat, {}).get("is_whole_thymus", False)),
        "is_composition": bool(feat_map.get(feat, {}).get("is_composition", False)),
    })

cons_df = pd.DataFrame(feat_rows)
cons_df = cons_df.sort_values(
    ["is_consensus", "best_median_rank_pct", "n_models_selected", "n_folds_total_selections"],
    ascending=[False, True, False, False],
)
cons_df.to_csv(ML / "ML_consensus_features.csv", index=False)
logger.info(f"Saved: {ML / 'ML_consensus_features.csv'}")

n_cons = int(cons_df["is_consensus"].sum())
logger.info(f"Consensus features: {n_cons} / {len(cons_df)} tracked")

# -------------------------------------------------------------------
# P0-2: split consensus into expression (gene_celltype + whole_thymus)
# and composition (proportion) files.  Step 13 reads ONLY the expression
# file for DL tokens; proportions never enter tokens.
# -------------------------------------------------------------------
cons_expr = cons_df[cons_df["feature_type"].isin(["gene_celltype", "whole_thymus"])].copy()
cons_comp = cons_df[cons_df["feature_type"] == "proportion"].copy()
cons_expr.to_csv(ML / "ML_consensus_expression_features.csv", index=False)
cons_comp.to_csv(ML / "ML_consensus_composition_features.csv", index=False)
logger.info(f"Saved expression consensus ({len(cons_expr)} features) and "
            f"composition consensus ({len(cons_comp)} features) files")

# -------------------------------------------------------------------
# P1: gene-level stability summary
# -------------------------------------------------------------------
gene_rows = []
for gene, gg in cons_df[cons_df["is_consensus"]].dropna(subset=["gene"]).groupby("gene"):
    model_sets = [set(str(m).split("|")) for m in gg["models"]]
    union_models = set().union(*model_sets) if model_sets else set()
    gene_rows.append({
        "gene": gene,
        "n_consensus_features": int((gg["is_consensus"]).sum()),
        "best_stability": float(gg["best_stability"].max()),
        "best_rank_pct": float(gg["best_median_rank_pct"].min()),
        "n_celltypes_supported": int(gg["celltype"].nunique()),
        "n_models_supported": len(union_models),
        "direction_sign_ok": bool(gg["direction_sign_ok"].all()),
    })
pd.DataFrame(gene_rows).sort_values(["n_consensus_features", "best_rank_pct"],
                                    ascending=[False, True]).to_csv(
    ML / "ML_consensus_gene_stability.csv", index=False)
logger.info(f"Saved gene stability summary ({len(gene_rows)} genes)")

logger.info("Top consensus (rank-based):")
for _, r in cons_df[cons_df["is_consensus"]].head(40).iterrows():
    logger.info(f"  {r['feature']}: pct={r['best_median_rank_pct']:.3f}, "
                f"models={r['n_models_selected']}, folds={r['n_folds_total_selections']}, "
                f"best_model={r['best_model']}, gene={r['gene']}, ct={r['celltype']}")

# -------------------------------------------------------------------
# Consensus genes (authoritative gene column from feature_dictionary)
# -------------------------------------------------------------------
genes = cons_df[cons_df["is_consensus"]][["gene", "celltype", "feature_type",
                                          "is_whole_thymus", "is_composition",
                                          "feature", "n_models_selected",
                                          "best_median_rank_pct",
                                          "is_single_model_strong",
                                          "is_multimodel_recurrent"]].copy()
genes = genes.dropna(subset=["gene"])
genes.to_csv(ML / "ML_consensus_genes.csv", index=False)

distinct_genes = sorted(genes["gene"].unique())
with open(ML / "ML_consensus_genes.txt", "w") as fh:
    fh.write("\n".join(distinct_genes))
logger.info(f"Consensus distinct genes: {len(distinct_genes)} -> {ML / 'ML_consensus_genes.txt'}")

logger.info("\n=== Step 12 (P0-fixed) COMPLETE ===")
logger.info(f"Consensus features: {n_cons}; distinct genes: {len(distinct_genes)}")
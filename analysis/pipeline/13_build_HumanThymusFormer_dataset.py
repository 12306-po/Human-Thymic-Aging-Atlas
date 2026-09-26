#!/usr/bin/env python
"""Step 13 (P0-fixed): Build HumanThymusFormer dataset (donor-level CV-ready).

Design (plan v2.1, Step 13):
  * Tokens come from the Step 12 ML consensus features (Gene x CellType + WholeThymus).
  * Because the global ML consensus was computed on ALL 18 donors, its membership
    leaks into each DL outer fold.  Step 13 explicitly logs this and marks the
    DL LODO as POST-HOC INTERPRETATION-ONLY, not strict generalization
    (Option B from the checklist).  The strict evaluation case would require
    per-fold re-deriving the consensus inside each training set, which is
    beyond the 18-donor budget. DL performance is therefore descriptive/post-hoc
    and must NOT be compared against the ML OOF estimate as an equivalent
    generalization number (no hardcoded performance values are kept here; they
    are read from the rerun outputs).
  * Token parsing uses the authoritative Step 10 feature_dictionary.csv instead
    of regex / string-suffix heuristics; a feature missing from the dictionary
    raises KeyError (no guess/fallback).
  * Tokens come from the Step 12 ML consensus EXPRESSION features file
    (ML_consensus_expression_features.csv; feature_type in {gene_celltype,
    whole_thymus}).  Composition/proportion features NEVER enter DL tokens.
  * Token list is capped at MAX_DL_TOKENS = 500, sorted by
    (n_models_selected, best_median_rank_pct) then model support.
  * Fold-specific manifests are saved for traceability.

Outputs:
  05_deep_learning/dataset/
    tokens.csv.gz                 donor x token expression (log-CPM)
    token_mask.csv.gz             donor x token observed mask
    token_metadata.csv            token -> gene, celltype, feature
    vocab.json                    gene -> gene_idx
    celltype_vocab.json           celltype -> ct_idx
    donor_metadata.csv            donor -> age, sex, binary label
    cv_folds.json                 LODO folds (train/test donor lists)
    dl_scope_statement.txt        documents the global-consensus leakage caveat
    fold_manifests/fold_*.json    per-fold train/test donors + token list
  10_results/logs/13_build_DL_dataset.log
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

def _env(key):
    val = os.environ.get(key)
    if not val:
        raise RuntimeError(f"Required env var {key} is not set. Run: source scripts/00_project_config.sh")
    return Path(val)

PROJ = _env("PROJ")
DS = PROJ / "04_machine_learning" / "dataset"
ML = PROJ / "04_machine_learning"
PSEUDO = PROJ / "02_pseudobulk"
OUT = PROJ / "05_deep_learning" / "dataset"
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "fold_manifests").mkdir(parents=True, exist_ok=True)
LOGS = PROJ / "10_results" / "logs"
LOGS.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("step13")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler(sys.stdout)
fh = logging.FileHandler(LOGS / "13_build_DL_dataset.log", mode="w")
fmt = logging.Formatter("%(asctime)s  %(levelname)s  %(message)s")
ch.setFormatter(fmt)
fh.setFormatter(fmt)
logger.addHandler(ch)
logger.addHandler(fh)
logger.info("=== Step 13 (P0-fixed): Build HumanThymusFormer dataset ===")

# ------------------------------------------------------------------
# 0. Token parsing via Step 10 authoritative feature dictionary
# ------------------------------------------------------------------
fdict = pd.read_csv(DS / "feature_dictionary.csv")
fdict_map = {}
for _, r in fdict.iterrows():
    fdict_map[r["feature"]] = {
        "gene": r["gene"],
        "celltype": r["celltype"],
        "feature_type": r["feature_type"],
        "is_whole_thymus": bool(r["is_whole_thymus"]),
    }

def parse_feature(feat):
    """Authoritative feature -> (gene, celltype|WT) via feature_dictionary.

    P0-2: a feature that is NOT in the authoritative dictionary raises
    KeyError — the old regex/suffix fallback could silently mis-label
    proportion features as genes, or produce wrong (gene, celltype) pairs.
    """
    if feat not in fdict_map:
        raise KeyError(
            f"Feature '{feat}' not found in Step 10 feature_dictionary.csv; "
            "refusing to guess gene/celltype (P0-2). Check the consensus "
            "feature source before rebuilding the DL dataset.")
    info = fdict_map[feat]
    return info["gene"], info["celltype"] if not info["is_whole_thymus"] else "WT"

# ------------------------------------------------------------------
# 1. Consensus expression features -> tokens (P0-1: expression ONLY;
#    proportions never enter DL tokens)
# ------------------------------------------------------------------
MAX_DL_TOKENS = 500   # P1: token cap (small model; 200-1000 guidance)
cons = pd.read_csv(ML / "ML_consensus_expression_features.csv")
cons = cons[cons["is_consensus"]].copy()
if "feature_type" in cons.columns:
    cons = cons[cons["feature_type"].isin(["gene_celltype", "whole_thymus"])].copy()
if cons.empty:
    raise RuntimeError(
        "No consensus expression features found in "
        "ML_consensus_expression_features.csv; run Step 12 first")
feats = (
    cons.sort_values(["n_models_selected", "best_median_rank_pct"],
                     ascending=[False, True])["feature"]
    .head(MAX_DL_TOKENS)
    .tolist()
)
logger.info(f"Consensus expression features -> tokens (capped at {MAX_DL_TOKENS}): "
            f"{len(feats)}")

tokens = pd.DataFrame([parse_feature(f) + (f,) for f in feats],
                      columns=["gene", "celltype", "feature"])
# WT token celltype label: special
tokens["token_type"] = np.where(tokens["celltype"] == "WT", "whole_thymus", "gene_celltype")
logger.info(f"  gene_celltype tokens: {(tokens['token_type']=='gene_celltype').sum()}, "
            f"whole_thymus tokens: {(tokens['token_type']=='whole_thymus').sum()}")

# ------------------------------------------------------------------
# 2. Load donor pseudobulk values for each token
# ------------------------------------------------------------------
# Whole-thymus: donor_matrix/log_normalized_combined.csv.gz
wt = pd.read_csv(PSEUDO / "donor_matrix" / "log_normalized_combined.csv.gz", index_col=0)
wt.index = wt.index.astype(str)

# Celltype pseudobulk: celltype_matrix/log_normalized/<donor>_<CT>.csv.gz
def load_celltype_value(donor, ct, gene):
    f = PSEUDO / "celltype_matrix" / "log_normalized" / f"{donor}_{ct}.csv.gz"
    if not f.exists():
        return np.nan
    df = pd.read_csv(f, index_col=0)
    if gene not in df.index:
        return np.nan
    return float(df.loc[gene].iloc[0])

donors = sorted(wt.index.tolist())
logger.info(f"Donors: {len(donors)}")

expr = pd.DataFrame(index=donors, columns=feats, dtype=float)
for feat in feats:
    gene, ct = tokens.loc[tokens["feature"] == feat, ["gene", "celltype"]].iloc[0]
    if ct == "WT":
        expr.loc[donors, feat] = wt.loc[donors, gene].values
    else:
        for donor in donors:
            expr.loc[donor, feat] = load_celltype_value(donor, ct, gene)

mask = (~expr.isna()).astype(int)
expr = expr.fillna(0.0)  # value 0 where missing; mask distinguishes

logger.info(f"Token expression matrix: {expr.shape}")
logger.info(f"Observed fraction: {mask.mean().mean():.3f}")

# ------------------------------------------------------------------
# 3. Vocabularies
# ------------------------------------------------------------------
genes = sorted(tokens["gene"].unique())
cts = sorted(tokens["celltype"].unique())
vocab = {g: i for i, g in enumerate(genes)}
ct_vocab = {c: i for i, c in enumerate(cts)}
logger.info(f"Gene vocab: {len(vocab)}, Celltype vocab: {len(ct_vocab)}")

# Token ids
tokens["gene_id"] = tokens["gene"].map(vocab)
tokens["celltype_id"] = tokens["celltype"].map(ct_vocab)

# ------------------------------------------------------------------
# 4. Save
# ------------------------------------------------------------------
expr.to_csv(OUT / "tokens.csv.gz", compression="gzip")
mask.to_csv(OUT / "token_mask.csv.gz", compression="gzip")
tokens.to_csv(OUT / "token_metadata.csv", index=False)

with open(OUT / "vocab.json", "w") as fh:
    json.dump(vocab, fh, indent=2)
with open(OUT / "celltype_vocab.json", "w") as fh:
    json.dump(ct_vocab, fh, indent=2)

donor_meta = pd.read_csv(DS / "donor_metadata.csv", index_col=0)
donor_meta = donor_meta.loc[donors]
donor_meta.to_csv(OUT / "donor_metadata.csv")

# LODO folds
folds = {}
for i, donor in enumerate(donors):
    folds[f"fold{i+1}"] = {
        "train": [d for d in donors if d != donor],
        "test": [donor],
    }
with open(OUT / "cv_folds.json", "w") as fh:
    json.dump(folds, fh, indent=2)

# Fold-specific manifests (for traceability)
for fkey, fv in folds.items():
    with open(OUT / "fold_manifests" / f"{fkey}.json", "w") as fh:
        json.dump({
            "train_donors": fv["train"],
            "test_donors": fv["test"],
            "token_list": feats,
            "n_tokens": len(feats),
        }, fh, indent=2)

# DL scope statement
with open(OUT / "dl_scope_statement.txt", "w") as fh:
    fh.write("DL SCOPE: POST-HOC INTERPRETATION-ONLY\n")
    fh.write("=" * 50 + "\n")
    fh.write("The ML consensus features (Step 12) were computed on ALL 18 donors.\n")
    fh.write("Each DL outer test donor therefore indirectly influenced token/feature\n")
    fh.write("selection via the global ML consensus. DL LODO R2/RMSE/Spearman rho\n")
    fh.write("are NOT valid generalization estimates and should NOT be compared to\n")
    fh.write("ML LODO R2 as if they measure the same quantity.\n")
    fh.write("DL's primary role is non-linear representation and attribution\n")
    fh.write("(attention / Integrated Gradients) for the candidate gene panel.\n")
logger.info("DL scope statement written")

logger.info("Saved tokens / mask / metadata / vocab / cv_folds / fold_manifests")
logger.info(f"\n=== Step 13 (P0-fixed) COMPLETE ===")
logger.info(f"Tokens: {len(feats)} ({expr.shape[1]}); Donors: {len(donors)}")
logger.info(f"Observed values: {mask.mean().mean():.3f}")

#!/usr/bin/env python
"""14b (figure audit 2026-09-17): STRICT out-of-fold HumanThymusFormer.

Unlike Step 13/14 (whose single global token universe came from an all-18-donor
ML consensus and so leaks into every fold), here the token universe for each
outer LODO fold is built ONLY from that fold's outer-TRAINING donors:

  token set_f = top expression features (ElasticNet + LinearSVR union, cap 500)
                saved by Step 11 for fold f (importance computed on the 17
                training donors; held-out donor never seen)
  -> per-fold scaler (fit donors only), per-fold gene/celltype vocab
  -> fit/val disjoint split within the 17 training donors, early stopping on val
  -> ONE prediction for the held-out donor; 18 held-out predictions pooled.

Fold-specific Integrated Gradients are computed on each held-out donor with
that fold's model/scaler/vocab and aggregated to gene level across folds.

Outputs (05_deep_learning/strict_oof/):
  dl_strict_oof_predictions.csv, dl_strict_oof_metrics.csv (+bootstrap CI)
  oof_IG_gene_consistency.csv, oof_IG_token_long.csv
  fold_token_counts.csv
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
import torch
import torch.nn as nn
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

PROJ = Path(os.environ.get("PROJ", "/data/zxy/projects/human_thymus_age_ML_DL"))
DS = PROJ / "04_machine_learning" / "dataset"
ML = PROJ / "04_machine_learning"
OUT = PROJ / "05_deep_learning" / "strict_oof"
OUT.mkdir(parents=True, exist_ok=True)
LOGS = PROJ / "10_results" / "logs"
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout),
              logging.FileHandler(LOGS / "14b_strict_oof_DL.log", mode="w")])
log = logging.getLogger("14b")

SEED = 371
MAX_TOKENS = 500
torch.manual_seed(SEED)
np.random.seed(SEED)

# ---------------------------------------------------------------- data
Xall = pd.read_csv(DS / "feature_matrix_all_combined.csv.gz", index_col=0)
fdict = pd.read_csv(DS / "feature_dictionary.csv",
                    usecols=["feature", "gene", "celltype", "feature_type"])
fmeta = fdict.set_index("feature")
cohort = pd.read_csv(PROJ / "01_raw_processing" / "metadata" /
                     "05B_final_regression_cohort.csv")
donors = cohort.donor_id.astype(str).tolist()
ages = cohort.set_index("donor_id").age_years.astype(float).loc[donors].values
Xall = Xall.loc[donors]
log.info("X=%s donors=%d", Xall.shape, len(donors))

# Step11 fold-level importance. CRITICAL fold mapping (second-round review P0,
# 2026-09-17): sklearn LeaveOneGroupOut iterates groups via np.unique, i.e. in
# LEXICOGRAPHIC donor order, so Step11 fold f held out sorted(donors)[f-1], NOT
# the age-sorted cohort position donors[f-1]. The previous version of this
# script mismatched every fold's token set (importance from a different held-out
# donor's training models). We iterate folds in the SAME lexicographic order and
# resolve the held-out row by cohort position.
imp = pd.read_csv(ML / "metrics" / "fold_level_importance.csv")
fold_test_donors = sorted(donors)
assert len(fold_test_donors) == 18
# sanity: Step11 importance fold counts per fold must be non-zero for all 18
_folds_present = set(imp.fold.unique())
assert _folds_present == set(range(1, 19)), f"Step11 folds missing: {_folds_present}"
log.info("fold importance: %s; models=%s", imp.shape,
         sorted(imp.model.unique()))

try:
    from captum.attr import IntegratedGradients
    HAVE_CAPTUM = True
except Exception:  # noqa: BLE001
    HAVE_CAPTUM = False


class HumanThymusFormer(nn.Module):
    def __init__(self, n_genes, n_cts, n_tokens, dim=64, n_blocks=2, n_heads=2,
                 ffn_dim=128, dropout=0.2):
        super().__init__()
        self.gene_emb = nn.Embedding(n_genes, dim)
        self.ct_emb = nn.Embedding(n_cts, dim)
        self.pos = nn.Parameter(torch.randn(1, n_tokens, dim) * 0.02)
        self.expr_proj = nn.Linear(1, dim)
        self.mask_proj = nn.Linear(1, dim)
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=n_heads, dim_feedforward=ffn_dim,
            dropout=dropout, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_blocks)
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))

    def forward(self, expr, maskv, gid, cid, key_padding_mask=None):
        B, T = expr.shape
        tok = (self.gene_emb(gid) + self.ct_emb(cid)
               + self.pos.expand(B, -1, -1)
               + self.expr_proj(expr.unsqueeze(-1))
               + self.mask_proj(maskv.unsqueeze(-1)))
        h = self.encoder(tok, src_key_padding_mask=key_padding_mask)
        m3 = maskv.unsqueeze(-1)
        pooled = (h * m3).sum(1) / (m3.sum(1) + 1e-6)
        return self.head(pooled).squeeze(-1)


class AttrModel(nn.Module):
    def __init__(self, base, m, g, c):
        super().__init__()
        self.base, self.m, self.g, self.c = base, m, g, c
        self.kpm = (m == 0).bool()

    def forward(self, expr):
        B = expr.shape[0]  # captum evaluates n_steps interpolated rows at once
        return self.base(expr,
                         self.m[None].expand(B, -1),
                         self.g[None].expand(B, -1),
                         self.c[None].expand(B, -1),
                         key_padding_mask=self.kpm[None].expand(B, -1))


def fold_tokens(fold):
    """Top expression features from fold's TRAINING-donor models (Step11)."""
    sub = imp[(imp.fold == fold) & (imp.model.isin(["ElasticNet", "LinearSVR"]))]
    sub = sub[sub.feature.map(lambda f: fmeta["feature_type"].get(f))
              .isin(["gene_celltype", "whole_thymus"])]
    # rank by best (min) rank across the two linear models, then union
    g = (sub.groupby("feature").agg(rk=("rank_pct", "min"),
                                    n=("model", "nunique"))
         .sort_values(["n", "rk"], ascending=[False, True]))
    return g.head(MAX_TOKENS).index.tolist()


def standardize_fit(Xfit, Mfit):
    mu = np.zeros(Xfit.shape[1], np.float32)
    sd = np.ones(Xfit.shape[1], np.float32)
    for j in range(Xfit.shape[1]):
        obs = Mfit[:, j] == 1
        if obs.sum() > 0:
            mu[j] = Xfit[obs, j].mean()
            sd[j] = Xfit[obs, j].std() or 1.0
    return mu, sd


preds, ig_long, tok_counts = [], [], []
for f in range(1, 19):
    test_d = fold_test_donors[f - 1]
    ti = donors.index(test_d)   # row position of held-out donor in cohort-order X
    assert donors[ti] == test_d
    train_idx = [i for i in range(18) if i != ti]
    feats = fold_tokens(f)
    sub = fmeta.reindex(feats)
    gene = sub["gene"].fillna(pd.Series([x.rsplit("_", 1)[0] for x in feats],
                                        index=feats)).values
    ct = np.where(sub["feature_type"].eq("whole_thymus").values, "WT",
                  sub["celltype"].values)
    gmap = {g: i for i, g in enumerate(sorted(set(gene)))}
    cmap = {c: i for i, c in enumerate(sorted(set(ct)))}
    gid = np.array([gmap[g] for g in gene], dtype=np.int64)
    cid = np.array([cmap[c] for c in ct], dtype=np.int64)

    Xr = Xall[feats]
    M = (~Xr.isna()).astype(np.float32).values
    Xv = Xr.fillna(0.0).values.astype(np.float32)
    # deterministic age-stratified fit/val split (3 val) among the 17 train
    order = np.argsort(ages[train_idx])
    val_pos = order[:: max(1, len(train_idx) // 3)][:3]
    val_idx = [train_idx[i] for i in val_pos]
    fit_idx = [i for i in train_idx if i not in set(val_idx)]
    mu, sd = standardize_fit(Xv[fit_idx], M[fit_idx])
    Xs = (Xv - mu) / sd
    Xs[M == 0] = 0.0
    ymu, ysd = ages[fit_idx].mean(), ages[fit_idx].std() or 1.0
    Ys = (ages - ymu) / ysd

    def tt(ii):
        return (torch.from_numpy(Xs[ii].astype(np.float32)),
                torch.from_numpy(M[ii].astype(np.float32)),
                torch.from_numpy(Ys[ii].astype(np.float32)))
    Xf, Mf, Yf = tt(fit_idx)
    Xv_, Mv_, Yv_ = tt(val_idx)
    Gt, Ct = torch.from_numpy(gid), torch.from_numpy(cid)

    torch.manual_seed(SEED)
    model = HumanThymusFormer(len(gmap), len(cmap), len(feats))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=300)
    lossf = nn.MSELoss()
    best, best_state, wait, best_ep = 1e9, None, 0, 0
    ds = DataLoader(TensorDataset(Xf, Mf, Yf), batch_size=len(fit_idx), shuffle=True)
    for ep in range(300):
        model.train()
        for xb, mb, yb in ds:
            opt.zero_grad()
            lossf(model(xb, mb, Gt.expand(xb.shape[0], -1),
                        Ct.expand(xb.shape[0], -1),
                        key_padding_mask=(mb == 0).bool()), yb).backward()
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            vp = model(Xv_, Mv_, Gt.expand(Xv_.shape[0], -1),
                       Ct.expand(Xv_.shape[0], -1),
                       key_padding_mask=(Mv_ == 0).bool())
            vr = torch.sqrt(((vp - Yv_) ** 2).mean()).item()
        if vr < best - 1e-4:
            best, best_state, wait, best_ep = vr, \
                {k: v.clone() for k, v in model.state_dict().items()}, 0, ep
        else:
            wait += 1
            if wait >= 40:
                break
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        xt = torch.from_numpy(Xs[[ti]].astype(np.float32))
        mt = torch.from_numpy(M[[ti]].astype(np.float32))
        yp = float(model(xt, mt, Gt[None], Ct[None],
                         key_padding_mask=(mt == 0).bool())[0]) * ysd + ymu
    preds.append({"fold": f, "donor_id": test_d, "y_true": ages[ti],
                  "y_pred": yp, "n_tokens": len(feats), "best_epoch": best_ep})
    tok_counts.append({"fold": f, "test_donor": test_d, "n_tokens": len(feats),
                       "n_genes": len(gmap), "best_val_rmse_yr": best * ysd})
    log.info("fold %2d test=%s T=%d ep=%d pred=%.1f true=%.0f",
             f, test_d, len(feats), best_ep, yp, ages[ti])

    # fold-specific IG on the held-out donor
    if HAVE_CAPTUM:
        am = AttrModel(model, mt[0], Gt, Ct)
        ig = IntegratedGradients(am)
        attr = ig.attribute(xt, torch.zeros_like(xt), n_steps=20)[0].detach().numpy()
        rk = pd.Series(np.abs(attr)).rank(ascending=False, method="first").values
        for j, feat in enumerate(feats):
            ig_long.append({"fold": f, "test_donor": test_d, "feature": feat,
                            "gene": gene[j], "celltype": ct[j],
                            # Signed expression-channel attribution is retained
                            # for donor-level biological interpretation.  The
                            # model predicts standardized age, hence multiplying
                            # by the fold-specific ysd expresses IG in years.
                            "IG_signed": float(attr[j]),
                            "IG_signed_years": float(attr[j] * ysd),
                            "IG_abs": float(abs(attr[j])),
                            "IG_abs_years": float(abs(attr[j]) * ysd),
                            "rank": int(rk[j]), "rank_pct": float(rk[j] / len(feats))})

pred = pd.DataFrame(preds)
pred.to_csv(OUT / "dl_strict_oof_predictions.csv", index=False)
pd.DataFrame(tok_counts).to_csv(OUT / "fold_token_counts.csv", index=False)

yt, yp2 = pred.y_true.values, pred.y_pred.values
r2, mae = r2_score(yt, yp2), mean_absolute_error(yt, yp2)
rmse = np.sqrt(mean_squared_error(yt, yp2))
rho, _ = stats.spearmanr(yt, yp2)
rng = np.random.default_rng(SEED)
br2, bmae = [], []
for _ in range(2000):
    i = rng.integers(0, 18, 18)
    br2.append(r2_score(yt[i], yp2[i])); bmae.append(mean_absolute_error(yt[i], yp2[i]))
mets = pd.DataFrame([{
    "config": "cfgB_strict_fold_internal_tokens", "n_donors": 18,
    "R2": r2, "MAE": mae, "RMSE": rmse, "Spearman_rho": rho,
    "R2_CI95_low": np.quantile(br2, .025), "R2_CI95_high": np.quantile(br2, .975),
    "MAE_CI95_low": np.quantile(bmae, .025), "MAE_CI95_high": np.quantile(bmae, .975),
    "scope": "strict LODO: fold-internal token universe, scaler, early stopping"}])
mets.to_csv(OUT / "dl_strict_oof_metrics.csv", index=False)
log.info("STRICT OOF: R2=%.3f (%.2f,%.2f) MAE=%.2f RMSE=%.2f rho=%.3f",
         r2, mets.R2_CI95_low[0], mets.R2_CI95_high[0], mae, rmse, rho)

if ig_long:
    lg = pd.DataFrame(ig_long)
    lg.to_csv(OUT / "oof_IG_token_long.csv", index=False)
    gc = (lg.groupby("gene")
          .agg(median_rank_pct=("rank_pct", "median"),
               min_rank_pct=("rank_pct", "min"),
               n_folds_token=("fold", "nunique"),
               n_folds_top50=("rank", lambda r: int((r <= 50).sum())),
               mean_IG_abs=("IG_abs", "mean"),
               mean_IG_abs_years=("IG_abs_years", "mean"),
               mean_IG_signed_years=("IG_signed_years", "mean"))
          .reset_index().sort_values("median_rank_pct"))
    gc.to_csv(OUT / "oof_IG_gene_consistency.csv", index=False)
    log.info("OOF-IG gene consistency rows: %d; top: %s",
             len(gc), gc.head(10).gene.tolist())
log.info("=== 14b strict OOF DL COMPLETE ===")

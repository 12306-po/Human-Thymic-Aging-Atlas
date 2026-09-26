#!/usr/bin/env python
"""Step 14 (P0-fixed): Train HumanThymusFormer (minimal, donor-level LODO).

Review fixes (2026-09-11/12, checklist P0):
  1. VALIDATION LEAKAGE FIXED: within each outer LODO fold, the outer-training
     donors are split into fit_train and validation (disjoint). Early stopping
     uses ONLY the validation donors; the held-out donor never enters model
     selection or early stopping.
  2. Fold-specific scalers are saved with each checkpoint
     (token_mean, token_std, y_mean, y_std) so Step 15 can apply the SAME
     per-fold normalization for attribution (no all-18 standardization leak).
  3. Missing tokens are masked in the Transformer self-attention via
     src_key_padding_mask, in addition to the masked mean pooling.
  4. Training curve: real per-epoch history per fold (not only the last row),
     saved as fold x epoch x train_loss x val_loss and plotted as
     mean/median curves.
  5. cfgB is PRE-DECLARED and frozen (no claim that cfgB was selected by outer
     test performance).
  6. DL is explicitly documented as POST-HOC INTERPRETATION-ONLY (global ML
     consensus token leakage caveat; see Step 13 dl_scope_statement.txt). DL
     R2/RMSE/Spearman rho are reported but NOT compared to ML as if
     generalization estimates.

Outputs:
  05_deep_learning/checkpoints/best_model_<config>_fold*.pt  (per fold; incl. scalers)
  05_deep_learning/checkpoints/final_interp_model.pt
  05_deep_learning/checkpoints/fold_metadata.csv
  05_deep_learning/training_history_per_fold.csv
  05_deep_learning/training_history.csv            (final model per-epoch)
  05_deep_learning/dl_posthoc_summary.csv        (renamed from dl_vs_ml_summary.csv; used by Steps 15/16/20)
  05_deep_learning/predictions/fold_predictions_dl.csv
  09_figures/14_training_curves_<config>.pdf
  10_results/logs/14_train_HumanThymusFormer.log
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def _env(key):
    val = os.environ.get(key)
    if not val:
        raise RuntimeError(f"Required env var {key} is not set. Run: source scripts/00_project_config.sh")
    return Path(val)

PROJ = _env("PROJ")
DLDS = PROJ / "05_deep_learning" / "dataset"
OUT = PROJ / "05_deep_learning"
CKPT = OUT / "checkpoints"
CKPT.mkdir(parents=True, exist_ok=True)
(OUT / "predictions").mkdir(parents=True, exist_ok=True)
FIGURES = PROJ / "09_figures"
FIGURES.mkdir(parents=True, exist_ok=True)
LOGS = PROJ / "10_results" / "logs"
LOGS.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("step14")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler(sys.stdout)
fh = logging.FileHandler(LOGS / "14_train_HumanThymusFormer.log", mode="w")
fmt = logging.Formatter("%(asctime)s  %(levelname)s  %(message)s")
ch.setFormatter(fmt)
fh.setFormatter(fmt)
logger.addHandler(ch)
logger.addHandler(fh)
logger.info("=== Step 14 (P0-fixed): Train HumanThymusFormer (minimal, LODO) ===")

torch.manual_seed(371)
np.random.seed(371)

# ------------------------------------------------------------------
# 1. Load dataset
# ------------------------------------------------------------------
expr = pd.read_csv(DLDS / "tokens.csv.gz", index_col=0)
mask = pd.read_csv(DLDS / "token_mask.csv.gz", index_col=0)
tok_meta = pd.read_csv(DLDS / "token_metadata.csv")
donor_meta = pd.read_csv(DLDS / "donor_metadata.csv", index_col=0)
with open(DLDS / "vocab.json") as fh:
    vocab = json.load(fh)
with open(DLDS / "celltype_vocab.json") as fh:
    ct_vocab = json.load(fh)
with open(DLDS / "cv_folds.json") as fh:
    folds = json.load(fh)

donors = expr.index.tolist()
ages = donor_meta.loc[donors, "age_years"].astype(float).values
gene_ids = tok_meta["gene_id"].values.astype(np.int64)
ct_ids = tok_meta["celltype_id"].values.astype(np.int64)
expr_vals = expr.values.astype(np.float32)
mask_vals = mask.values.astype(np.float32)

logger.info(f"Donors: {len(donors)}, tokens: {expr.shape[1]}, "
            f"genes: {len(vocab)}, celltypes: {len(ct_vocab)}")
logger.info(f"Age range: {ages.min()}-{ages.max()}")

# ------------------------------------------------------------------
# 2. Model
# ------------------------------------------------------------------
class HumanThymusFormer(nn.Module):
    def __init__(self, n_genes, n_cts, n_tokens, dim=32, n_blocks=1, n_heads=1,
                 ffn_dim=64, dropout=0.2):
        super().__init__()
        self.dim = dim
        self.gene_emb = nn.Embedding(n_genes, dim)
        self.ct_emb = nn.Embedding(n_cts, dim)
        self.pos = nn.Parameter(torch.randn(1, n_tokens, dim) * 0.02)
        self.expr_proj = nn.Linear(1, dim)
        self.mask_proj = nn.Linear(1, dim)
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=n_heads, dim_feedforward=ffn_dim,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_blocks)
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))

    def forward(self, expr, maskv, gene_id, ct_id, key_padding_mask=None):
        # expr: (B, T); maskv: (B, T); gene_id/ct_id: (B, T)
        # key_padding_mask: (B, T) True = ignore (missing token)
        B, T = expr.shape
        tok = (self.gene_emb(gene_id) + self.ct_emb(ct_id)
               + self.pos.expand(B, -1, -1)
               + self.expr_proj(expr.unsqueeze(-1))
               + self.mask_proj(maskv.unsqueeze(-1)))
        out = self.encoder(tok, src_key_padding_mask=key_padding_mask)  # (B, T, dim)
        # Masked mean pooling over tokens
        mask3 = maskv.unsqueeze(-1)                   # (B, T, 1)
        pooled = (out * mask3).sum(dim=1) / (mask3.sum(dim=1) + 1e-6)  # (B, dim)
        return self.head(pooled).squeeze(-1)


def make_model(dim, n_blocks, n_heads, ffn_dim, dropout):
    return HumanThymusFormer(len(vocab), len(ct_vocab), expr.shape[1],
                             dim=dim, n_blocks=n_blocks, n_heads=n_heads,
                             ffn_dim=ffn_dim, dropout=dropout)


# ------------------------------------------------------------------
# 3. Training helper (fit/val DISJOINT; fold scalers returned)
# ------------------------------------------------------------------
def train_eval(model, fit_idx, val_idx, test_idx, n_epochs=300, patience=40,
               lr=1e-3, batch_size=None, tag=""):
    """Train with early stopping on val (val NOT in fit set).

    Returns (test_pred, history, best_state, scalers, best_epoch).
    """
    # --- fit normalization on FIT donors only (not val, not test) ---
    # P0 fix: mean/std must be computed ONLY on observed tokens (mask==1);
    # structural missing values (mask==0, expr=0) must NOT enter the scaler.
    fit_expr = expr_vals[fit_idx]
    fit_mask = mask_vals[fit_idx]  # (n_fit, n_tokens); 1=observed, 0=missing
    token_mean = np.zeros(fit_expr.shape[1], dtype=np.float32)
    token_std = np.ones(fit_expr.shape[1], dtype=np.float32)
    for j in range(fit_expr.shape[1]):
        obs = fit_mask[:, j] == 1.0
        if obs.sum() > 0:
            vals = fit_expr[obs, j]
            token_mean[j] = float(np.nanmean(vals))
            token_std[j] = float(np.nanstd(vals))
            if token_std[j] < 1e-6:
                token_std[j] = 1.0
        # else: keep default mean=0, std=1 for tokens unseen in fit donors
    y_all = ages.astype(np.float32)
    y_mean_fit = float(y_all[fit_idx].mean())
    y_std_fit = float(y_all[fit_idx].std()) or 1.0

    X_all = (expr_vals - token_mean) / token_std
    # P1: structural missing tokens (mask==0) have standardized value
    # -mean/std; zero them so they carry no expression signal (mask + padding
    # mask still handle their exclusion from pooling/attention).
    X_all[mask_vals == 0] = 0.0
    Y_all = (y_all - y_mean_fit) / y_std_fit

    X_fit = torch.from_numpy(X_all[fit_idx].astype(np.float32))
    M_fit = torch.from_numpy(mask_vals[fit_idx].astype(np.float32))
    Y_fit = torch.from_numpy(Y_all[fit_idx].astype(np.float32))
    # padding mask: True where token is MISSING (ignore in attention)
    Pad_fit = (M_fit == 0).bool()
    X_va = torch.from_numpy(X_all[val_idx].astype(np.float32))
    M_va = torch.from_numpy(mask_vals[val_idx].astype(np.float32))
    Y_va = torch.from_numpy(Y_all[val_idx].astype(np.float32))
    Pad_va = (M_va == 0).bool()

    if batch_size is None:
        batch_size = max(1, len(fit_idx))
    ds = TensorDataset(X_fit, M_fit, Y_fit, Pad_fit)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    lossf = nn.MSELoss()

    G = torch.from_numpy(gene_ids)
    C = torch.from_numpy(ct_ids)

    best_val = float("inf")
    best_state = None
    best_epoch = 0
    patience_cnt = 0
    history = []
    for ep in range(n_epochs):
        model.train()
        ep_loss = 0.0
        for xb, mb, yb, pb in dl:
            opt.zero_grad()
            pred = model(xb, mb, G.expand(xb.shape[0], -1), C.expand(xb.shape[0], -1),
                         key_padding_mask=pb)
            loss = lossf(pred, yb)
            loss.backward()
            opt.step()
            ep_loss += loss.item() * len(xb)
        sched.step()

        model.eval()
        with torch.no_grad():
            vp = model(X_va, M_va, G.expand(X_va.shape[0], -1), C.expand(X_va.shape[0], -1),
                       key_padding_mask=Pad_va)
            val_mae = (vp - Y_va).abs().mean().item() * y_std_fit
            val_rmse = ((vp - Y_va) ** 2).mean().sqrt().item() * y_std_fit
        history.append({"epoch": ep, "train_loss": ep_loss / len(fit_idx),
                        "val_mae": val_mae, "val_rmse": val_rmse})

        if val_rmse < best_val - 1e-4:
            best_val = val_rmse
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch = ep
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        X_te = torch.from_numpy(X_all[test_idx].astype(np.float32))
        M_te = torch.from_numpy(mask_vals[test_idx].astype(np.float32))
        Pad_te = (M_te == 0).bool()
        tp = model(X_te, M_te, G.expand(X_te.shape[0], -1), C.expand(X_te.shape[0], -1),
                   key_padding_mask=Pad_te)
        test_pred = (tp * y_std_fit + y_mean_fit).numpy()

    scalers = {
        "token_mean": token_mean.tolist(),
        "token_std": token_std.tolist(),
        "y_mean": float(y_mean_fit),
        "y_std": float(y_std_fit),
    }
    return test_pred, history, best_state, scalers, best_epoch


# ------------------------------------------------------------------
# 4. LODO evaluation (PRE-DECLARED cfgB; fit/val disjoint)
# ------------------------------------------------------------------
# cfgB is PRE-DECLARED and frozen (plan v2.1). It is NOT claimed to be the
# result of outer-fold model selection.
RUN_CONFIGS = {
    "cfgB": dict(dim=64, n_blocks=2, n_heads=2, ffn_dim=128, dropout=0.2),
}

all_rows = []
histories = {c: [] for c in RUN_CONFIGS}
fold_meta_rows = []
test_preds_all = {}

for cfg_name, cfg in RUN_CONFIGS.items():
    logger.info(f"\n--- Config {cfg_name} (PRE-DECLARED): {cfg} ---")
    test_preds = {}
    t0 = time.time()
    for fi, (fkey, fv) in enumerate(folds.items()):
        train_d = fv["train"]; test_d = fv["test"][0]
        outer_train_idx = [donors.index(d) for d in train_d]
        test_idx = [donors.index(test_d)]

        # DISJOINT fit/val split of the outer-training donors (age-stratified):
        # val = 3 age-stratified donors; fit = remaining outer-training donors.
        ages_ot = ages[outer_train_idx]
        order = np.argsort(ages_ot)
        val_idx_ot = [outer_train_idx[i] for i in order[:: max(1, len(outer_train_idx) // 3)]][:3]
        val_idx_ot = val_idx_ot[:3]
        fit_idx = [i for i in outer_train_idx if i not in set(val_idx_ot)]
        logger.info(f"  {fkey}: fit={len(fit_idx)}, val={len(val_idx_ot)}, "
                    f"test={test_d} (DISJOINT)")

        model = make_model(**cfg)
        pred, hist, best_state, scalers, best_ep = train_eval(
            model, fit_idx, val_idx_ot, test_idx, tag=f"{cfg_name}_f{fi}")
        test_preds[test_d] = pred[0]
        # store full per-epoch history per fold
        for h in hist:
            histories[cfg_name].append({"fold": fi + 1, **h})
        # checkpoint + scalers (use TRUE best_epoch, not last epoch)
        ckpt_path = CKPT / f"best_model_{cfg_name}_fold{fi+1}.pt"
        torch.save({"state_dict": best_state, "config": cfg,
                    "token_mean": scalers["token_mean"], "token_std": scalers["token_std"],
                    "y_mean": scalers["y_mean"], "y_std": scalers["y_std"],
                    "train_donors": train_d,
                    "fit_donors": [donors[i] for i in fit_idx],
                    "val_donors": [donors[i] for i in val_idx_ot],
                    "test_donor": test_d, "best_epoch": best_ep},
                   ckpt_path)
        fold_meta_rows.append({
            "fold": fi + 1, "test_donor": test_d,
            "n_fit": len(fit_idx), "n_val": len(val_idx_ot),
            "fit_donors": "|".join(donors[i] for i in fit_idx),
            "val_donors": "|".join(donors[i] for i in val_idx_ot),
            "best_epoch": best_ep,
        })
        test_preds_all[test_d] = pred[0]
        logger.info(f"  {fkey}: pred={pred[0]:.1f} (true={ages[test_idx[0]]:.0f}), "
                    f"epochs={len(hist)}")

    # OOF metrics (REPORT ONLY; not generalization estimates — see scope statement)
    y_true = np.array([ages[donors.index(d)] for d in donors])
    y_pred = np.array([test_preds[d] for d in donors])
    from scipy import stats
    r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - y_true.mean()) ** 2)
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rho, p = stats.spearmanr(y_true, y_pred)
    all_rows.append({"config": cfg_name, "n_donors": 18, "R2": round(r2, 4),
                     "RMSE": round(rmse, 4), "MAE": round(mae, 4),
                     "Spearman_rho": round(rho, 4),
                     "scope": "posthoc_interpretation_only"})
    logger.info(f"  {cfg_name} OOF (post-hoc): R2={r2:.3f}, RMSE={rmse:.3f}, "
                f"MAE={mae:.3f}, ρ={rho:.3f} ({time.time()-t0:.0f}s)")

    # Save fold predictions
    pd.DataFrame([{"donor_id": d, "y_true": float(ages[donors.index(d)]),
                   "y_pred": float(test_preds[d])} for d in donors]
                 ).to_csv(OUT / "predictions" / "fold_predictions_dl.csv", index=False)

    # Plot training curve: per-epoch mean/median across folds
    hist_df = pd.DataFrame(histories[cfg_name])
    if len(hist_df) > 0:
        agg = hist_df.groupby("epoch").agg(
            val_mae_mean=("val_mae", "mean"),
            val_mae_med=("val_mae", "median"),
            val_rmse_mean=("val_rmse", "mean"),
            train_loss_mean=("train_loss", "mean"),
        ).reset_index()
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(agg["epoch"], agg["train_loss_mean"], label="train loss (mean)")
        axes[0].set_xlabel("epoch"); axes[0].set_ylabel("train loss (MSE, z-scale)")
        axes[0].set_title(f"{cfg_name}: train loss (mean across folds)")
        axes[0].legend()
        axes[1].plot(agg["epoch"], agg["val_mae_mean"], label="val MAE mean")
        axes[1].plot(agg["epoch"], agg["val_mae_med"], label="val MAE median", ls="--")
        axes[1].set_xlabel("epoch"); axes[1].set_ylabel("val MAE (years)")
        axes[1].set_title(f"{cfg_name}: val MAE (mean/median across folds)")
        axes[1].legend()
        fig.tight_layout()
        fig.savefig(FIGURES / f"14_training_curves_{cfg_name}.pdf", dpi=150)
        plt.close(fig)

dl_summary = pd.DataFrame(all_rows)
# P1-3: renamed dl_vs_ml_summary.csv -> dl_posthoc_summary.csv (Steps 15/16/20 read this)
dl_summary.to_csv(OUT / "dl_posthoc_summary.csv", index=False)
pd.DataFrame(fold_meta_rows).to_csv(CKPT / "fold_metadata.csv", index=False)
pd.concat([pd.DataFrame(histories[c]) for c in RUN_CONFIGS],
          ignore_index=True).to_csv(OUT / "training_history_per_fold.csv", index=False)
logger.info(f"\nDL summary (post-hoc):\n{dl_summary.to_string()}")

# ------------------------------------------------------------------
# 5. Final interpretation model (all 18 donors; two-stage)
# ------------------------------------------------------------------
best_cfg = RUN_CONFIGS["cfgB"]
best_cfg_name = "cfgB"
logger.info(f"\nFinal interpretation model: {best_cfg_name} on all 18 donors")

# Stage 1: determine best_epoch via fit/val disjoint (15 fit, 3 val)
model_final = make_model(**best_cfg)
train_idx_all = list(range(len(donors)))
val_idx_all = [train_idx_all[i] for i in np.argsort(ages)[::6]][:3]
fit_idx_all = [i for i in train_idx_all if i not in set(val_idx_all)]
pred_final, hist_final, best_state_final, scalers_final, determined_best_epoch = train_eval(
    model_final, fit_idx_all, val_idx_all, train_idx_all,
    tag="final", n_epochs=400, patience=60)
logger.info(f"Stage 1: best_epoch = {determined_best_epoch} "
            f"(from {len(fit_idx_all)} fit + {len(val_idx_all)} val donors)")

# Stage 2: retrain from scratch with ALL 18 donors, fixed best_epoch, no early stopping
logger.info(f"Stage 2: retrain all 18 donors for {determined_best_epoch + 1} epochs")
# Compute scaler on all 18 donors
all_expr = expr_vals
all_mask = mask_vals
token_mean_all = np.zeros(all_expr.shape[1], dtype=np.float32)
token_std_all = np.ones(all_expr.shape[1], dtype=np.float32)
for j in range(all_expr.shape[1]):
    obs = all_mask[:, j] == 1.0
    if obs.sum() > 0:
        vals = all_expr[obs, j]
        token_mean_all[j] = float(np.nanmean(vals))
        token_std_all[j] = float(np.nanstd(vals))
        if token_std_all[j] < 1e-6:
            token_std_all[j] = 1.0
y_mean_all = float(ages.mean())
y_std_all = float(ages.std()) or 1.0

X_std_all = (all_expr - token_mean_all) / token_std_all
X_std_all[all_mask == 0] = 0.0   # P1: same zero-fill for missing tokens
Y_std_all = (ages - y_mean_all) / y_std_all

X_all_t = torch.from_numpy(X_std_all.astype(np.float32))
M_all_t = torch.from_numpy(all_mask.astype(np.float32))
Y_all_t = torch.from_numpy(Y_std_all.astype(np.float32))
Pad_all = (M_all_t == 0).bool()

model_final2 = make_model(**best_cfg)
ds_all = TensorDataset(X_all_t, M_all_t, Y_all_t, Pad_all)
dl_all = DataLoader(ds_all, batch_size=len(donors), shuffle=True)
opt2 = torch.optim.Adam(model_final2.parameters(), lr=1e-3, weight_decay=1e-4)
G = torch.from_numpy(gene_ids)
C = torch.from_numpy(ct_ids)
for ep in range(determined_best_epoch + 1):
    model_final2.train()
    for xb, mb, yb, pb in dl_all:
        opt2.zero_grad()
        pred = model_final2(xb, mb, G.expand(xb.shape[0], -1),
                            C.expand(xb.shape[0], -1), key_padding_mask=pb)
        loss = nn.MSELoss()(pred, yb)
        loss.backward()
        opt2.step()

model_final2.eval()
best_state_final2 = {k: v.clone() for k, v in model_final2.state_dict().items()}
scalers_final2 = {
    "token_mean": token_mean_all.tolist(),
    "token_std": token_std_all.tolist(),
    "y_mean": float(y_mean_all),
    "y_std": float(y_std_all),
}
torch.save({"state_dict": best_state_final2, "config": best_cfg,
            "token_mean": scalers_final2["token_mean"], "token_std": scalers_final2["token_std"],
            "y_mean": scalers_final2["y_mean"], "y_std": scalers_final2["y_std"],
            "train_donors": [donors[i] for i in train_idx_all],
            "fit_donors": [donors[i] for i in fit_idx_all],
            "val_donors": [donors[i] for i in val_idx_all],
            # P1-2: explicit stage provenance for the final interp model
            "stage1_fit_donors": [donors[i] for i in fit_idx_all],
            "stage1_val_donors": [donors[i] for i in val_idx_all],
            "stage2_train_donors": [donors[i] for i in train_idx_all],
            "best_epoch_source": "stage1_validation (15 fit / 3 val, disjoint)",
            "test_donor": None,
            "best_epoch": determined_best_epoch,
            "scope": "all_18_donors_fixed_epoch_interpretation_only"},
           CKPT / "final_interp_model.pt")
logger.info(f"Final interp model saved (all 18 donors, {determined_best_epoch + 1} epochs): "
            f"{CKPT / 'final_interp_model.pt'}")

# Save config + training history
with open(OUT / "final_config.json", "w") as fh:
    json.dump({"config_name": best_cfg_name, **best_cfg}, fh, indent=2)
pd.DataFrame(hist_final).to_csv(OUT / "training_history.csv", index=False)
logger.info("Training history saved")

logger.info(f"\n=== Step 14 (P0-fixed) COMPLETE ===")
logger.info(f"Config: {best_cfg_name} (pre-declared)")
logger.info(f"DL LODO metrics (post-hoc only): {dl_summary.to_dict('records')}")

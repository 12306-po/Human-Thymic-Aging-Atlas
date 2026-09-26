#!/usr/bin/env python
"""Step 15: Interpret HumanThymusFormer (attention + Integrated Gradients).

P0 review fixes (2026-09-14):
  1. Model forward is IDENTICAL to Step 14 (cfgB): the same
     HumanThymusFormer class with `src_key_padding_mask` in the encoder and
     masked mean pooling. Missing tokens never participate in self-attention
     during interpretation, exactly as during training.
  2. Attribution input normalization uses the scaler saved IN EACH CHECKPOINT
     (token_mean/token_std), NOT an all-18-donors re-computed scaler.
  3. Across-fold IG: each LODO fold checkpoint applies ITS OWN fold scaler,
     and attribution is computed on that fold's HELD-OUT donor only
     (OOF-style attribution stability).
  4. Attention extraction replicates the PyTorch post-norm
     TransformerEncoderLayer path (norm1 -> self-attn with key_padding_mask),
     so the attended matrix matches the trained forward.

Scope: DL is post-hoc interpretation ONLY (see Step 13 scope statement). No
generalization claim is made here.

Outputs:
  05_deep_learning/interpretation/
    final_attention_token_importance.csv
    final_IG_token_importance.csv
    IG_across_fold_consistency.csv
    gene_importance.csv
    celltype_importance.csv
    gene_celltype_importance.csv
  10_results/logs/15_interpret_HumanThymusFormer.log
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

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def _env(key):
    val = os.environ.get(key)
    if not val:
        raise RuntimeError(f"Required env var {key} is not set. Run: source scripts/00_project_config.sh")
    return Path(val)

PROJ = _env("PROJ")
DLDS = PROJ / "05_deep_learning" / "dataset"
CKPT = PROJ / "05_deep_learning" / "checkpoints"
OUT = PROJ / "05_deep_learning" / "interpretation"
OUT.mkdir(parents=True, exist_ok=True)
LOGS = PROJ / "10_results" / "logs"
LOGS.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("step15")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler(sys.stdout)
fh = logging.FileHandler(LOGS / "15_interpret_HumanThymusFormer.log", mode="w")
fmt = logging.Formatter("%(asctime)s  %(levelname)s  %(message)s")
ch.setFormatter(fmt)
fh.setFormatter(fmt)
logger.addHandler(ch)
logger.addHandler(fh)
logger.info("=== Step 15: Interpret HumanThymusFormer (P0-fixed) ===")

torch.manual_seed(371)
np.random.seed(371)


# ------------------------------------------------------------------
# Model architecture — MUST mirror Step 14 exactly (cfgB)
# ------------------------------------------------------------------
with open(DLDS / "vocab.json") as fh:
    vocab = json.load(fh)
with open(DLDS / "celltype_vocab.json") as fh:
    ct_vocab = json.load(fh)
with open(PROJ / "05_deep_learning" / "final_config.json") as fh:
    final_cfg = json.load(fh)


class HumanThymusFormer(nn.Module):
    """Identical definition to Step 14 (src_key_padding_mask in forward)."""

    def __init__(self, n_genes, n_cts, n_tokens, dim=64, n_blocks=2, n_heads=2,
                 ffn_dim=128, dropout=0.2):
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
        mask3 = maskv.unsqueeze(-1)                   # (B, T, 1)
        pooled = (out * mask3).sum(dim=1) / (mask3.sum(dim=1) + 1e-6)  # (B, dim)
        return self.head(pooled).squeeze(-1)


expr = pd.read_csv(DLDS / "tokens.csv.gz", index_col=0)
mask = pd.read_csv(DLDS / "token_mask.csv.gz", index_col=0)
tok_meta = pd.read_csv(DLDS / "token_metadata.csv")
donor_meta = pd.read_csv(DLDS / "donor_metadata.csv", index_col=0)

donors = expr.index.tolist()
expr_vals = expr.values.astype(np.float32)
mask_vals = mask.values.astype(np.float32)
gene_ids = tok_meta["gene_id"].values.astype(np.int64)
ct_ids = tok_meta["celltype_id"].values.astype(np.int64)
ages = donor_meta.loc[donors, "age_years"].astype(float).values

G = torch.from_numpy(gene_ids)
C = torch.from_numpy(ct_ids)

cfg = {k: final_cfg[k] for k in ["dim", "n_blocks", "n_heads", "ffn_dim", "dropout"]}
logger.info(f"Donors: {len(donors)}, tokens: {expr.shape[1]}, config: {cfg}")


def load_model(ckpt_dict):
    """Build model and load state from a checkpoint dict (or bare state_dict)."""
    m = HumanThymusFormer(len(vocab), len(ct_vocab), expr.shape[1], **cfg)
    if isinstance(ckpt_dict, dict) and "state_dict" in ckpt_dict:
        m.load_state_dict(ckpt_dict["state_dict"])
    else:
        m.load_state_dict(ckpt_dict)
    m.eval()
    return m


def standardize(expr_rows, mask_rows, token_mean, token_std):
    """Apply a checkpoint's scaler.

    P1 (2026-09-16): after standardization, MISSING tokens must be reset to 0.0,
    exactly as Step 14 does during training (X_std[mask == 0] = 0). Otherwise
    a missing raw value (often 0) becomes a nonzero standardized value and the
    attribution inputs differ from training inputs.
    """
    x = (expr_rows - np.asarray(token_mean, dtype=np.float32)) / np.asarray(
        token_std, dtype=np.float32)
    x = np.asarray(x, dtype=np.float32)
    x[np.asarray(mask_rows) == 0] = 0.0
    return x


def attention_importance(model, X, M):
    """Token importance = mean attention received (over queries x heads), masked.

    Replicates the PyTorch POST-NORM TransformerEncoderLayer path used by
    Step 14 (norm_first=False, the default): inside each layer the self-attn
    operates on the RAW layer input (not norm1(input)), and `src_key_padding_mask`
    is passed to EVERY layer so missing tokens never attend during
    interpretation, exactly as during training.

    P1: attention weights are masked on BOTH query and key sides
    (valid_q * valid_k) to prevent missing tokens from contributing to
    the attention denominator.
    """
    B, T = X.shape
    kpm = (M == 0).bool()           # True = missing -> ignore
    valid = M == 1                   # (B, T) True = observed token
    valid_q = valid[:, None, None, :]  # (B,1,1,T) query side
    valid_k = valid[:, None, :, None]  # (B,1,T,1) key side
    valid_pair = valid_q & valid_k     # (B,1,T,T) True for both valid
    with torch.no_grad():
        tok = (model.gene_emb(G.expand(B, -1)) + model.ct_emb(C.expand(B, -1))
               + model.pos.expand(B, -1, -1)
               + model.expr_proj(X.unsqueeze(-1))
               + model.mask_proj(M.unsqueeze(-1)))
        h = tok
        for layer in model.encoder.layers[:-1]:
            h = layer(h, src_key_padding_mask=kpm)
        last = model.encoder.layers[-1]
        # post-norm: self-attn input is the raw layer input h (not norm1(h))
        _, attn_w = last.self_attn(
            h, h, h, key_padding_mask=kpm,
            need_weights=True, average_attn_weights=False)
        # attn_w: (B, n_heads, T, T)
        attn_w = attn_w.detach().numpy()
    # P1: mask both query and key positions in the attention matrix
    attn_w_masked = np.where(valid_pair.numpy(), attn_w, 0.0)
    attn_tok = attn_w_masked.mean(axis=(1, 2))  # (B, T): mean over heads and queries
    # Renormalize per-query after masking
    row_sums = attn_tok.sum(axis=1, keepdims=True) + 1e-9
    attn_tok = attn_tok / row_sums
    return attn_tok.mean(axis=0)


# ------------------------------------------------------------------
# 1. Final interpretation model: checkpoint scaler + attention
# ------------------------------------------------------------------
final_ckpt = torch.load(CKPT / "final_interp_model.pt", map_location="cpu")
final_model = load_model(final_ckpt)
final_token_mean = final_ckpt["token_mean"]
final_token_std = final_ckpt["token_std"]
logger.info(f"Final model loaded; scaler from checkpoint (n_tokens={len(final_token_mean)})")

X_final = torch.from_numpy(
    standardize(expr_vals, mask_vals, final_token_mean, final_token_std).astype(np.float32))
M_final = torch.from_numpy(mask_vals.astype(np.float32))

attn_imp = attention_importance(final_model, X_final, M_final)
attn_df = pd.DataFrame({
    "feature": tok_meta["feature"].values,
    "gene": tok_meta["gene"].values,
    "celltype": tok_meta["celltype"].values,
    "attention_importance": attn_imp,
})
attn_df["attention_rank"] = attn_df["attention_importance"].rank(ascending=False).astype(int)
attn_df = attn_df.sort_values("attention_rank")
attn_df.to_csv(OUT / "final_attention_token_importance.csv", index=False)
logger.info(f"Attention importance saved: {OUT / 'final_attention_token_importance.csv'}")


# ------------------------------------------------------------------
# 2. Integrated Gradients on the final model (checkpoint scaler)
# ------------------------------------------------------------------
try:
    from captum.attr import IntegratedGradients
    captum_available = True
except Exception as e:  # noqa: BLE001
    logger.warning(f"captum not available: {e}; IG skipped, attention-only output")
    captum_available = False


class AttrModel(nn.Module):
    """Wraps HumanThymusFormer for captum: only `expr` is interpolated; all
    auxiliary inputs (mask, ids, key_padding_mask) are fixed attributes."""

    def __init__(self, base, maskv, gene_id, ct_id):
        super().__init__()
        self.base = base
        self.maskv = maskv          # (T,)
        self.gene_id = gene_id      # (T,)
        self.ct_id = ct_id          # (T,)
        self.kpm = (maskv == 0).bool()  # (T,) True = missing -> ignore

    def forward(self, expr):
        B = expr.shape[0]
        return self.base(
            expr,
            self.maskv.unsqueeze(0).expand(B, -1),
            self.gene_id.unsqueeze(0).expand(B, -1),
            self.ct_id.unsqueeze(0).expand(B, -1),
            key_padding_mask=self.kpm.unsqueeze(0).expand(B, -1),
        )


def run_ig(model, X, M, n_steps=50, internal_batch=None):
    """Integrated Gradients computed donor-by-donor (review P0).

    X: (n, T) standardized; M: (n, T) mask. Returns (n, T) attributions.

    For each donor i, builds a separate AttrModel with that donor's per-token
    mask (T,) and attributes that single donor's standardized vector (1, T),
    so the missing-token mask inside AttrModel is always donor-specific.
    (AttrModel holds maskv/gene_id/ct_id as (T,) vectors and expands them to
    the batch — passing the full (n, T) mask would be a dimension error.)

    P1: the zero baseline corresponds to zero in standardized space
    (i.e. training-mean expression baseline for each token).
    """
    X = np.asarray(X, dtype=np.float32)
    M = np.asarray(M, dtype=np.float32)
    n, T = X.shape
    attribs = np.zeros((n, T), dtype=np.float32)
    for i in range(n):
        mask_i = torch.from_numpy(M[i].astype(np.float32))          # (T,)
        attr_model = AttrModel(model, mask_i, G, C)
        ig = IntegratedGradients(attr_model)
        x_i = torch.from_numpy(X[i : i + 1].astype(np.float32))     # (1, T)
        # P1: baseline = zero in standardized space (training-mean baseline)
        baseline = torch.zeros_like(x_i)
        out, delta = ig.attribute(
            x_i, baseline, return_convergence_delta=True, n_steps=n_steps,
            internal_batch_size=internal_batch)
        attribs[i] = out.detach().numpy()
        logger.info(f"  IG donor {i + 1}/{n}: shape={tuple(out.shape)}, "
                    f"|delta|={np.abs(delta.numpy()).mean():.4f}")
    return attribs


if captum_available:
    ig_attr = run_ig(final_model, X_final.numpy(), M_final.numpy(), n_steps=50)
    ig_imp = np.abs(ig_attr).mean(axis=0)
    ig_df = pd.DataFrame({
        "feature": tok_meta["feature"].values,
        "gene": tok_meta["gene"].values,
        "celltype": tok_meta["celltype"].values,
        "IG_importance": ig_imp,
    })
    ig_df["IG_rank"] = ig_df["IG_importance"].rank(ascending=False).astype(int)
    ig_df = ig_df.sort_values("IG_rank")
    ig_df.to_csv(OUT / "final_IG_token_importance.csv", index=False)
    logger.info(f"IG importance saved: {OUT / 'final_IG_token_importance.csv'}")


# ------------------------------------------------------------------
# 3. Across-fold IG consistency (18 LODO cfgB checkpoints)
#    Each fold: load fold scaler from checkpoint; attribute ONLY the held-out
#    donor (OOF-style). n_steps=20 for speed.
# ------------------------------------------------------------------
fold_cons = []
if captum_available:
    for fi in range(1, 19):
        ckpt_f = CKPT / f"best_model_cfgB_fold{fi}.pt"
        if not ckpt_f.exists():
            logger.warning(f"  checkpoint {ckpt_f} missing, skipping")
            continue
        _c = torch.load(ckpt_f, map_location="cpu")
        fold_model = load_model(_c)
        test_donor = _c.get("test_donor")
        if test_donor is None or test_donor not in donors:
            logger.warning(f"  fold {fi}: test_donor unknown ({test_donor}); skipping")
            continue
        ho_idx = donors.index(test_donor)
        ho_mean = np.asarray(_c["token_mean"], dtype=np.float32)
        ho_std = np.asarray(_c["token_std"], dtype=np.float32)
        M_ho = mask_vals[[ho_idx]]                                # (1, T)
        X_ho = standardize(expr_vals[[ho_idx]], M_ho, ho_mean, ho_std)  # (1, T)
        attr = run_ig(fold_model, X_ho, M_ho, n_steps=20)
        imp = np.abs(attr).mean(axis=0)  # (T,)
        fold_cons.append(pd.DataFrame({
            "fold": fi, "test_donor": test_donor,
            "feature": tok_meta["feature"].values,
            "IG_importance": imp,
        }))
        logger.info(f"  fold {fi}: IG computed on held-out donor {test_donor}")

    if fold_cons:
        fold_df = pd.concat(fold_cons, ignore_index=True)
        fold_df["rank"] = fold_df.groupby("fold")["IG_importance"].rank(
            ascending=False, method="first")
        fold_df["rank_pct"] = fold_df.groupby("fold")["rank"].transform(
            lambda r: r / r.max())
        agg_cons = fold_df.groupby("feature").agg(
            median_rank_pct=("rank_pct", "median"),
            min_rank_pct=("rank_pct", "min"),
            n_folds_in_top50=("rank", lambda r: (r <= 50).sum()),
            mean_importance=("IG_importance", "mean"),
        ).reset_index()
        agg_cons["n_folds_in_top50"] = agg_cons["n_folds_in_top50"].astype(int)
        agg_cons = agg_cons.sort_values("median_rank_pct")
        agg_cons.to_csv(OUT / "IG_across_fold_consistency.csv", index=False)
        logger.info(f"Across-fold IG consistency saved: {OUT / 'IG_across_fold_consistency.csv'}")
        logger.info("Top stable tokens (across-fold IG):")
        for _, r in agg_cons.head(30).iterrows():
            logger.info(f"  {r['feature']}: median_pct={r['median_rank_pct']:.3f}, "
                        f"n_top50={r['n_folds_in_top50']}/18")


# ------------------------------------------------------------------
# 4. Gene / celltype / gene-celltype aggregates
# ------------------------------------------------------------------
merged = attn_df.copy()
if captum_available and (OUT / "final_IG_token_importance.csv").exists():
    ig_df2 = pd.read_csv(OUT / "final_IG_token_importance.csv")
    merged = merged.merge(ig_df2[["feature", "IG_importance", "IG_rank"]], on="feature")

merged.to_csv(OUT / "gene_celltype_importance.csv", index=False)

gene_agg = merged.groupby("gene").agg(
    mean_attention=("attention_importance", "mean"),
    max_attention=("attention_importance", "max"),
    n_tokens=("feature", "count"),
)
if "IG_importance" in merged.columns:
    gene_agg["mean_IG"] = merged.groupby("gene")["IG_importance"].mean().values
    gene_agg["max_IG"] = merged.groupby("gene")["IG_importance"].max().values
gene_agg = gene_agg.sort_values("mean_attention", ascending=False)
gene_agg.to_csv(OUT / "gene_importance.csv")

ct_agg = merged.groupby("celltype").agg(
    mean_attention=("attention_importance", "mean"),
    max_attention=("attention_importance", "max"),
    n_tokens=("feature", "count"),
)
if "IG_importance" in merged.columns:
    ct_agg["mean_IG"] = merged.groupby("celltype")["IG_importance"].mean().values
    ct_agg["max_IG"] = merged.groupby("celltype")["IG_importance"].max().values
ct_agg = ct_agg.sort_values("mean_attention", ascending=False)
ct_agg.to_csv(OUT / "celltype_importance.csv")

logger.info(f"\n=== Step 15 COMPLETE ===")
logger.info(f"Top genes (attention): {list(gene_agg.head(10).index)}")
logger.info(f"Top celltypes (attention): {list(ct_agg.index)}")
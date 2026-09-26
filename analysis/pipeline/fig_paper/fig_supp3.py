"""Supplementary figures set 3 (S07-S09): binary, DL, mouse programs."""
from __future__ import annotations

import json
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

from figure_common import *


def s07_classification():
    m = pd.read_csv(ML / "metrics" / "classification_metrics_oof.csv")
    ci = pd.read_csv(ML / "metrics" / "classification_metrics_oof_ci.csv").merge(
        m, on="model")
    pred = pd.read_csv(ML / "predictions" / "fold_predictions_classification.csv")
    fig, axs = plt.subplots(1, 2, figsize=(W_DOUBLE, 60 / 25.4))
    ax = axs[0]
    y = np.arange(len(ci))[::-1]
    lo = ci.AUC_CI95_low.clip(lower=0)
    hi = ci.AUC_CI95_high.clip(upper=1)
    if (lo > ci.AUC).any() or (hi < ci.AUC).any():
        raise ValueError("S07: AUC estimate lies outside clipped bootstrap interval")
    ax.errorbar(ci.AUC, y, xerr=[ci.AUC - lo, hi - ci.AUC],
                fmt="o", ms=5, color="#0072B2", capsize=3)
    ax.set_yticks(y); ax.set_yticklabels(ci.model, fontsize=6)
    ax.set_xlim(0, 1); ax.set_xlabel("OOF ROC-AUC (bootstrap CI clipped to [0,1])")
    ax.set_title("Exploratory binary Young(<18)/Old(>=40), n=11", fontsize=6.8)
    ax2 = axs[1]
    x = np.arange(len(ci)); w = 0.38
    ax2.bar(x - w/2, ci.Sensitivity, w, label="sensitivity", color=UP)
    ax2.bar(x + w/2, ci.Specificity, w, label="specificity", color=DOWN)
    for i, name in enumerate(ci.model):
        sub = pred[pred.model.eq(name)]
        if len(sub) != 11 or not sub.donor_id.is_unique:
            raise ValueError(f"S07: {name} needs 11 unique held-out predictions")
        tn = int(((sub.y_true == 0) & (sub.y_pred == 0)).sum())
        fp = int(((sub.y_true == 0) & (sub.y_pred == 1)).sum())
        if ci.iloc[i].Specificity == 0:
            ax2.text(i + w/2, 0.03, f"{tn}/{tn+fp}", ha="center",
                     va="bottom", rotation=90, fontsize=5.2)
    ax2.set_xticks(x); ax2.set_xticklabels(ci.model, rotation=30, fontsize=5.4)
    ax2.set_ylim(0, 1.05); ax2.legend(fontsize=6, frameon=False)
    ax2.set_title("Sensitivity/specificity (Old=positive; 4 Young/7 Old)", fontsize=6.8)
    save_supp(fig, "Supplementary_Figure_07_exploratory_binary")


def s08_deep_learning():
    strict_m = DL / "strict_oof" / "dl_strict_oof_metrics.csv"
    strict_p = DL / "strict_oof" / "dl_strict_oof_predictions.csv"
    post_f = DL / "dl_posthoc_summary.csv"
    fig = plt.figure(figsize=(W_DOUBLE, 145 / 25.4))
    gs = gridspec.GridSpec(2, 3, figure=fig, left=0.10, right=0.93,
                           bottom=0.13, top=0.82, hspace=0.70, wspace=0.75)
    # strict OOF performance
    ax = fig.add_subplot(gs[0, 0])
    if strict_p.exists():
        p = pd.read_csv(strict_p)
        ax.scatter(p.y_true, p.y_pred, s=14, color="#009E73")
        lo, hi = 0, 73
        ax.plot([lo, hi], [lo, hi], ls=":", color=GREY)
        r = pd.read_csv(strict_m).iloc[0]
        ax.set_title(f"Strict OOF (fold-internal tokens)\nR²={r.R2:.2f} "
                     f"({r.R2_CI95_low:.2f},{r.R2_CI95_high:.2f}), MAE={r.MAE:.1f}",
                     fontsize=6)
    else:
        ax.text(0.1, 0.5, "strict OOF pending", transform=ax.transAxes)
    ax.set_xlabel("observed"); ax.set_ylabel("predicted")
    # A blank note occupies this cell; the leakage plot is a separate S08b
    # artifact, never adjacent to the strict OOF performance panel.
    ax = fig.add_subplot(gs[0, 1])
    ax.axis("off")
    ax.text(0, 0.85, "Strict OOF is the only DL\nperformance estimate here.\n\n"
            "Global-token fit is a\nseparate leakage diagnostic\n(S08b).",
            transform=ax.transAxes, fontsize=6, va="top")
    # training histories (per-epoch median + IQR, with fold coverage)
    ax = fig.add_subplot(gs[0, 2])
    h = pd.read_csv(DL / "training_history_per_fold.csv")
    cov = h.groupby("epoch").size().reset_index(name="n_folds")
    a = h.groupby("epoch").agg(med=("val_mae", "median"),
                               q1=("val_mae", lambda s: s.quantile(.25)),
                               q3=("val_mae", lambda s: s.quantile(.75)),
                               tl=("train_loss", "median")).reset_index()
    a = a.merge(cov, on="epoch")
    # restrict to epochs covered by at least half the folds so the IQR band
    # is not distorted by a few long-running folds (review S08)
    a = a[a.n_folds >= 9]
    ax.plot(a.epoch, a.med, color="#0072B2", label="val MAE median")
    ax.fill_between(a.epoch, a.q1, a.q3, color="#0072B2", alpha=0.2)
    ax.set_xlabel("epoch"); ax.set_ylabel("val MAE (yr)")
    ax.set_title("DL validation across folds (IQR band;\n"
                 "≥9 folds per epoch shown)", fontsize=6)
    ax.legend(fontsize=5.5, frameon=False)
    # attention vs IG (QC only)
    ax = fig.add_subplot(gs[1, 0])
    gi = pd.read_csv(DL / "interpretation" / "gene_importance.csv").dropna(
        subset=["mean_attention", "mean_IG"])
    ax.scatter(gi.mean_attention, gi.mean_IG.abs(), s=8, alpha=0.7)
    rho = gi[["mean_attention", "mean_IG"]].corr(method="spearman").iloc[0, 1]
    ax.set_xlabel("mean attention"); ax.set_ylabel("|mean IG|")
    ax.set_title(f"Attention vs IG (method QC; ρ={rho:.2f})", fontsize=6)
    # cellstate percentile attribution (separate normalizations)
    ax = fig.add_subplot(gs[1, 1])
    ci = pd.read_csv(DL / "interpretation" / "celltype_importance.csv")
    ci["att_pct"] = ci.mean_attention.rank(pct=True)
    ci["ig_pct"] = ci.mean_IG.abs().rank(pct=True)
    ci = ci.sort_values("ig_pct")
    y = np.arange(len(ci))
    ax.scatter(ci.att_pct, y, color="#0072B2", s=14, label="attention %ile")
    ax.scatter(ci.ig_pct, y + 0.15, color="#D55E00", s=14, label="|IG| %ile")
    ax.set_yticks(y); ax.set_yticklabels(ci.celltype, fontsize=5.4)
    ax.set_xlabel("within-metric percentile")
    ax.legend(fontsize=5, frameon=False)
    ax.set_title("Cell-state attribution (whole-thymus = WT)", fontsize=6)
    # strict OOF IG frequency (sorted by fold count, NOT by rank; review S08:
    # never call 1-2/18 folds "stable")
    ax = fig.add_subplot(gs[1, 2])
    igf = DL / "strict_oof" / "oof_IG_gene_consistency.csv"
    if igf.exists():
        g = (pd.read_csv(igf).sort_values("n_folds_top50", ascending=False)
             .head(15).iloc[::-1])
        cols = ["#009E73" if f >= IG_STABLE_FOLDS else "#B08A00"
                for f in g.n_folds_top50]
        ax.barh(range(len(g)), g.n_folds_top50, color=cols)
        ax.set_yticks(range(len(g))); ax.set_yticklabels(g.gene, fontsize=5.2)
        ax.set_xlim(0, 18); ax.set_xlabel("folds (of 18) in OOF-IG top-50")
        ax.set_title(f"OOF-IG frequency (descriptive; green ≥ "
                     f"{IG_STABLE_FOLDS}/18)", fontsize=6)
        ax.axvline(IG_STABLE_FOLDS, ls="--", lw=0.7, color=GREY)
    else:
        ax.text(0.1, 0.5, "pending", transform=ax.transAxes)
    fig.suptitle("HumanThymusFormer: strict OOF and separate method diagnostics",
                 fontsize=8, y=0.96)
    save_supp(fig, "Supplementary_Figure_08_deep_learning")
    dlp = pd.read_csv(DL / "predictions" / "fold_predictions_dl.csv")
    post = pd.read_csv(post_f).iloc[0]
    fig2, ax2 = plt.subplots(figsize=(W_SINGLE, 70 / 25.4))
    ax2.scatter(dlp.y_true, dlp.y_pred, s=12, color=DISC)
    ax2.plot([0, 73], [0, 73], ls=":", color=GREY)
    ax2.set_xlim(0, 73); ax2.set_ylim(0, 73)
    ax2.set_xlabel("observed age"); ax2.set_ylabel("post-hoc predicted age")
    ax2.set_title(f"S08b: global-token leakage diagnostic\n"
                  f"R²={post.R2:.2f}; not an OOF performance estimate", fontsize=7)
    save_supp(fig2, "Supplementary_Figure_08b_leakage_diagnostic")


def s09_mouse_programs():
    """Descriptive direction split (review S09): one 100% stacked bar,
    consistent direction colours (orange=age-up, blue=age-down, grey=
    discordant), n kept on the figure, no inferential p-values."""
    o = pd.read_csv(MOUSE / "18_mouse_human_orthologs.csv")
    a = o[o.assessable == True].dropna(subset=["direction"])  # noqa: E712
    n = len(a)
    spec = [("Age_up_concordant", UP), ("Young_up_concordant", DOWN),
            ("Discordant", GREY)]
    counts = {k: int((a.direction == k).sum()) for k, _ in spec}
    # any direction not in the fixed spec (e.g. "Other") still gets a grey bin
    extra = set(a.direction.unique()) - {k for k, _ in spec}
    for k in sorted(extra):
        counts[k] = int((a.direction == k).sum()); spec.append((k, GREY))
    spec = [(k, c) for k, c in spec if counts[k] > 0]
    fig, ax = plt.subplots(figsize=(W_DOUBLE, 30 / 25.4))
    fracs = [counts[k] / n for k, _ in spec]
    left = 0.0
    for (k, c), f in zip(spec, fracs):
        ax.barh(0, f, left=left, color=c, height=0.45,
                edgecolor="white", linewidth=0.5)
        if f > 0.04:
            ax.text(left + f / 2, 0, f"{counts[k]} ({f:.0%})",
                    ha="center", va="center", fontsize=6, color="white")
        left += f
    ax.set_xlim(0, 1); ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=6)
    ax.set_yticks([])
    ax.set_xlabel(f"assessable ortholog genes (n={n}; 1 vs 1 mouse, descriptive)")
    handles = [Patch(color=c, label=k.replace("_", " ")) for k, c in spec]
    ax.legend(handles=handles, fontsize=5.5, frameon=False, ncol=3,
              loc="center left", bbox_to_anchor=(1.01, 0.5))
    ax.set_title("Mouse-internal stage-program directions (no gene-level "
                 "inference; counts only)", fontsize=6.8)
    save_supp(fig, "Supplementary_Figure_09_mouse_stage_programs")

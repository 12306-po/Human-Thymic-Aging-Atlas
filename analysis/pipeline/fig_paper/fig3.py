"""Figure 3: honest donor-level LODO prediction, ablation, permutation null."""
from __future__ import annotations

import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import json
import numpy as np
import pandas as pd
from scipy import stats

from figure_common import *

MODELS = ["ElasticNet", "LinearSVR", "RF", "XGB"]


def _schema(ax):
    ax.axis("off")
    steps = [
        ("1. Hold out 1 donor", "17 train / 1 test"),
        ("2. Train-only pipeline", "screen + inner-CV tune"),
        ("3. Refit on 17", "same selected pipeline"),
        ("4. Predict held-out", "once per donor; repeat ×18"),
    ]
    n = len(steps)
    bw = 0.20
    gap = (1.0 - n * bw) / (n + 1)
    y0, bh = 0.33, 0.52
    for i, (head, body) in enumerate(steps):
        x = gap + i * (bw + gap)
        ax.add_patch(mpatches.FancyBboxPatch(
            (x, y0), bw, bh, boxstyle="round,pad=0.015",
            fc="#EEF3FB", ec="#3B5B8A", lw=0.8))
        ax.text(x + bw / 2, y0 + bh - 0.14, head, ha="center", va="center",
                fontsize=7.2, fontweight="bold")
        ax.text(x + bw / 2, y0 + bh * 0.34, body, ha="center", va="center",
                fontsize=6.2)
        if i < n - 1:
            ax.annotate("", xy=(x + bw + gap * 0.75, y0 + bh / 2),
                        xytext=(x + bw + gap * 0.25, y0 + bh / 2),
                        arrowprops=dict(arrowstyle="->", lw=1.0))
    ax.text(0.5, 0.12, "Test donor never enters screening or tuning; pooled OOF metrics, n=18",
            ha="center", fontsize=6.4, style="italic")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)


def _true_pred(ax):
    pred = pd.read_csv(ML / "predictions" / "fold_predictions_regression.csv")
    mets = pd.read_csv(ML / "metrics" / "regression_metrics_oof.csv").set_index("model")
    ci = pd.read_csv(ML / "metrics" / "regression_metrics_oof_ci.csv").set_index("model")
    inner = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=ax.get_subplotspec(),
                                             wspace=0.34)
    ax.axis("off")
    sub_axes = []
    for k, name in enumerate(["ElasticNet", "MeanAgeNull"]):
        a = ax.figure.add_subplot(inner[k])
        sub_axes.append(a)
        s = pred[pred.model == name]
        a.scatter(s.y_true, s.y_pred, s=18, color="#0072B2", zorder=3)
        lo, hi = 0, 73
        a.plot([lo, hi], [lo, hi], ls=":", color=GREY, lw=0.9)
        r2 = mets.loc[name, "R2"]; mae = mets.loc[name, "MAE"]
        c = ci.loc[name]
        a.set_title(f"{name}\nR²={r2:.2f} [{c.R2_CI95_low:.2f}, "
                    f"{c.R2_CI95_high:.2f}]; MAE={mae:.1f} yr",
                    fontsize=5.8, pad=3)
        a.set_xlabel("observed age"); a.set_ylabel("predicted age")
        a.tick_params(labelsize=5.5)
        a.set_xlim(lo, hi); a.set_ylim(lo, hi)
        a.set_aspect("equal", adjustable="box")
    return sub_axes


def _model_compare(ax):
    m = pd.read_csv(ML / "metrics" / "regression_metrics_oof.csv")
    ci = pd.read_csv(ML / "metrics" / "regression_metrics_oof_ci.csv").merge(
        m[["model", "MAE", "R2"]], on="model")
    order = MODELS + ["MeanAgeNull"]
    ci = ci.set_index("model").loc[order].reset_index()
    x = np.arange(len(ci))
    lo = np.clip(ci.R2 - ci.R2_CI95_low, 0, None)
    hi = np.clip(ci.R2_CI95_high - ci.R2, 0, None)
    ax.errorbar(ci.R2, x, xerr=[lo, hi], fmt="o", ms=5, color="#0072B2",
                capsize=3, lw=0.9)
    ax.axvline(0, color="k", lw=0.6)
    ax.set_yticks(x); ax.set_yticklabels(order, fontsize=6)
    ax.invert_yaxis()
    ax.set_xlabel("OOF R² (donor bootstrap 95% CI; n=18)")
    ax.set_title("Continuous-age regression models", fontsize=6.8)
    ax.tick_params(labelsize=5.8)


def _ablation(ax):
    d = pd.read_csv(ML / "ablation_nested" / "ablation_nested_lodo_metrics.csv")
    order = ["all_features", "gene_celltype_only", "whole_thymus_only",
             "composition_only", "MeanAgeNull"]
    d = d.set_index("block").loc[order].reset_index()
    y = np.arange(len(d))[::-1]
    lo = np.clip(d.R2 - d.R2_CI95_low, 0, None)
    hi = np.clip(d.R2_CI95_high - d.R2, 0, None)
    ax.errorbar(d.R2, y, xerr=[lo, hi], fmt="o", ms=5, color="#D55E00",
                capsize=3, lw=0.9)
    ax.axvline(0, color="k", lw=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels([b.replace("_", " ") for b in d.block], fontsize=6)
    ax.set_xlim(min(-0.65, float(d.R2_CI95_low.min()) - 0.06),
                max(1.1, float(d.R2_CI95_high.max()) + 0.28))
    right = ax.get_xlim()[1] - 0.03
    for yi, r in zip(y, d.itertuples()):
        ax.text(right, yi, f"MAE {r.MAE:.1f}", fontsize=5.2,
                va="center", ha="right")
    ax.set_xlabel("OOF R² (paired donor-bootstrap 95% CI)", fontsize=6)
    ax.set_title("Feature-block ablation (ElasticNet, nested LODO)", fontsize=6.8)
    ax.tick_params(labelsize=5.8)


def _null_recurrence(ax):
    g = read_gene_recurrence_null()
    null = np.load(ML / "permutation_null" / "null_gene_fold_counts.npy")
    summary = json.loads((ML / "permutation_null" / "null_summary.json").read_text())
    if null.ndim != 2 or null.shape != (summary["P_permutations"], len(g)) or \
       summary["P_permutations"] < 1000 or summary["n_folds"] != 18:
        raise ValueError("Figure 3E: permutation matrix, gene table and run summary disagree")
    (FIGF / "fig3e_null_summary_audit.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    ks = np.arange(0, 19)
    obs_cnt = np.array([(g.obs_n_folds == k).sum() for k in ks], float)
    null_cnt = np.vstack([(null == k).sum(1) for k in ks]).T  # P x 19
    q05, q50, q95 = np.quantile(null_cnt, [.05, .5, .95], axis=0)
    ax.fill_between(ks, q05, q95, color=LGREY, alpha=0.8,
                    label="permutation 5-95% band")
    ax.plot(ks, q50, color=GREY, lw=0.8)
    ax.plot(ks, obs_cnt, "-o", ms=3, color="#0072B2", label="observed")
    ax.set_yscale("symlog", linthresh=1)
    ax.set_xlabel("folds (of 18) in which gene is selected")
    ax.set_ylabel("Number of genes (symlog)", fontsize=6)
    ax.set_yticks([0, 1, 10, 100, 1000, 10000])
    ax.legend(fontsize=5.5, frameon=False)
    n_bh = int(g[NULL_COL_CORRECTED].fillna(False).sum())
    n_fwer = int(g[NULL_COL_FWER].fillna(False).sum())
    ax.set_title(f"Selection recurrence vs age-label null\n"
                 f"(BH q<0.05: {n_bh} genes; FWER max-stat: {n_fwer})",
                 fontsize=6.8)
    ax.tick_params(labelsize=5.8)


def _top_recurrent(ax):
    g = read_gene_recurrence_null()
    n_bh = int(g[NULL_COL_CORRECTED].fillna(False).sum())
    n_fwer = int(g[NULL_COL_FWER].fillna(False).sum())
    # merge direction-sign consistency + median rank from the consensus table
    stab = pd.read_csv(ML / "ML_consensus_gene_stability.csv")
    g = g.merge(stab[["gene", "direction_sign_ok", "best_rank_pct"]],
                on="gene", how="left")
    g = g.sort_values(["obs_n_folds", "p_recurrence"],
                      ascending=[False, True]).head(8).iloc[::-1]
    y = np.arange(len(g))
    ax.barh(y, g.obs_n_folds, color="#0072B2", height=0.7,
            label="observed folds")
    ax.scatter(g.null_q95_folds, y, color="#D55E00", s=12, zorder=3,
               label="null 95% upper bound")
    for yi, (_, r) in zip(y, g.iterrows()):
        ok = r.direction_sign_ok if pd.notna(r.direction_sign_ok) else False
        ax.text(r.obs_n_folds + 0.3, yi, "✓" if ok else "·",
                fontsize=5.5, va="center",
                color=CONC if ok else "#7A2E00")
    ax.set_yticks(y); ax.set_yticklabels(g.gene, fontsize=5.2)
    ax.set_xlim(0, 19); ax.set_xlabel("folds selected (max 18)")
    ax.legend(fontsize=5.2, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, -0.30), ncol=2)
    if n_bh == 0 and n_fwer == 0:
        ax.set_title("Exploratory observed recurrence\n"
                     "(0 genes pass BH/FWER correction)",
                     fontsize=6.8)
    else:
        ax.set_title(f"Permutation-corrected genes\n"
                     f"(BH={n_bh}, FWER={n_fwer}; ✓ sign-consistent)",
                     fontsize=6.8)
    ax.tick_params(labelsize=5.6)


def _residuals(ax):
    pred = pd.read_csv(ML / "predictions" / "fold_predictions_regression.csv")
    s = pred[pred.model == "ElasticNet"].copy()
    fz = load_freeze()[["donor_id", "age_years", "sex", "age_group"]]
    s = s.merge(fz, on="donor_id").sort_values("age_years")
    s["res"] = s.y_pred - s.y_true
    x = np.arange(len(s))
    colors = [FEMALE if sx == "Female" else MALE for sx in s.sex]
    ax.scatter(x, s.res, s=22, c=colors, edgecolor="k", linewidth=0.3, zorder=3)
    ax.axhline(0, color="k", lw=0.7)
    ax.set_xticks(x); ax.set_xticklabels([short(d) for d in s.donor_id],
                                         fontsize=4.6, rotation=90)
    ax.set_ylabel("Predicted − observed (years)", fontsize=6)
    ax.set_title("Per-donor ElasticNet residuals (age-ordered)", fontsize=6.8)
    ax.tick_params(labelsize=5.6)
    ax.legend(handles=[mpatches.Patch(color=FEMALE, label="Female"),
                       mpatches.Patch(color=MALE, label="Male")],
              fontsize=5.2, frameon=False, loc="upper right")
    s[["donor_id", "y_true", "y_pred", "res", "sex"]].to_csv(
        FIGF / "fig3g_oof_donor_residual_audit.tsv", sep="\t", index=False)


def _audit_predictions():
    p = pd.read_csv(ML / "predictions" / "fold_predictions_regression.csv")
    m = pd.read_csv(ML / "metrics" / "regression_metrics_oof.csv").set_index("model")
    required = {"model", "donor_id", "y_true", "y_pred"}
    if required - set(p):
        raise ValueError(f"Figure 3 OOF table lacks {sorted(required - set(p))}")
    for model in MODELS + ["MeanAgeNull"]:
        sub = p[p.model.eq(model)]
        if len(sub) != 18 or not sub.donor_id.is_unique:
            raise ValueError(f"Figure 3: {model} must have 18 unique held-out donors")
        yt = sub.y_true.to_numpy(dtype=float)
        yp = sub.y_pred.to_numpy(dtype=float)
        r2 = 1 - np.sum((yt - yp) ** 2) / np.sum((yt - yt.mean()) ** 2)
        mae = np.mean(np.abs(yt - yp))
        if abs(r2 - float(m.loc[model, "R2"])) > 0.02 or \
           abs(mae - float(m.loc[model, "MAE"])) > 0.05:
            raise ValueError(f"Figure 3: {model} OOF predictions disagree with metrics")
    ref = set(p.loc[p.model.eq("ElasticNet"), "donor_id"])
    if any(set(p.loc[p.model.eq(model), "donor_id"]) != ref
           for model in MODELS + ["MeanAgeNull"]):
        raise ValueError("Figure 3 models are not evaluated on identical donors")
    p.to_csv(FIGF / "fig3_oof_predictions_audit.tsv", sep="\t", index=False)
    ab = pd.read_csv(ML / "ablation_nested" / "fold_predictions_ablation.csv")
    blocks = ["all_features", "gene_celltype_only", "whole_thymus_only",
              "composition_only", "MeanAgeNull"]
    for block in blocks:
        sub = ab[ab.block.eq(block)]
        if len(sub) != 18 or not sub.donor_id.is_unique or set(sub.donor_id) != ref:
            raise ValueError(f"Figure 3D: {block} lacks 18 matched unique donors")
    ab.to_csv(FIGF / "fig3d_ablation_oof_audit.tsv", sep="\t", index=False)


def build():
    _audit_predictions()
    fig = plt.figure(figsize=(W_DOUBLE, 265 / 25.4))
    gs = gridspec.GridSpec(5, 2, figure=fig, left=0.15, right=0.96,
                           bottom=0.07, top=0.97, hspace=0.64, wspace=0.48,
                           height_ratios=[0.62, 1.65, 1.2, 1.45, 1.2])
    axA = fig.add_subplot(gs[0, :]); letter(axA, "A", dx=-0.08); _schema(axA)
    axB = fig.add_subplot(gs[1, :]); sub_axes = _true_pred(axB)
    letter(sub_axes[0], "B", dx=-0.17, dy=1.10)
    axC = fig.add_subplot(gs[2, 0]); letter(axC, "C"); _model_compare(axC)
    axD = fig.add_subplot(gs[2, 1]); letter(axD, "D", dx=-0.08); _ablation(axD)
    axE = fig.add_subplot(gs[3, 0]); letter(axE, "E"); _null_recurrence(axE)
    axF = fig.add_subplot(gs[3, 1]); letter(axF, "F", dx=-0.08); _top_recurrent(axF)
    axG = fig.add_subplot(gs[4, :]); letter(axG, "G", dx=-0.08); _residuals(axG)
    save_composite(fig, "Figure3")

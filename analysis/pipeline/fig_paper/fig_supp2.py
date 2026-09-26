"""Supplementary figures set 2 (S04-S06): QC, annotation, full ML results."""
from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

from figure_common import *


def s04_qc_scrublet_doublet():
    sc = pd.read_csv(META / "06_scrublet_audit.csv").sort_values(
        "pct_doublets", ascending=False)
    fig = plt.figure(figsize=(W_DOUBLE, 95 / 25.4))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.55, wspace=0.3)
    ax = fig.add_subplot(gs[0, 0])
    y = np.arange(len(sc))
    ax.barh(y, sc.pct_doublets, color="#C98A8A", height=0.75)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r.gsm_id}" for r in sc.itertuples()],
                       fontsize=4.6)
    ax.set_xlabel("predicted doublets (%)")
    ax.set_title(f"Scrublet per library ({(sc.scrublet_status=='SUCCESS').sum()}"
                 "/18 success; record-only)", fontsize=6.8)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.scatter(sc.auto_threshold, sc.pct_doublets, s=12, color="#0072B2")
    ax2.set_xlabel("per-library auto threshold"); ax2.set_ylabel("doublet (%)")
    ax2.set_title("Doublet rate vs per-library threshold", fontsize=6.8)

    # doublet-removed sensitivity summary
    summ_f = RES / "sensitivity_doublet_removed" / "doublet_sensitivity_summary.json"
    import json
    s = json.loads(summ_f.read_text())
    ax3 = fig.add_subplot(gs[1, :])
    ax3.axis("off")
    mm = {r["model"]: r for r in s["main_ml_metrics"]}
    primary = pd.read_csv(ML / "metrics" / "regression_metrics_oof.csv").set_index("model")
    dr_rec = pd.read_csv(RES / "sensitivity_doublet_removed" /
                         "doublet_removed_gene_fold_recurrence.csv").query(
        "n_folds_selected>=3").gene
    null_g = read_gene_recurrence_null()
    corrected = set(null_g.loc[null_g[NULL_COL_CORRECTED].fillna(False), "gene"])
    inter = len(set(dr_rec) & corrected)
    jac = inter / max(1, len(set(dr_rec) | corrected))
    # Jaccard vs the frozen main consensus (review P1: honest stability claim)
    cons = set(pd.read_csv(ML / "ML_consensus_genes.txt", header=None)[0])
    jac_cons = len(set(dr_rec) & cons) / max(
        1, len(set(dr_rec) | cons))
    rows = [["Metric", "Primary", "Doublet removed"],
            ["ElasticNet OOF R²", f"{primary.loc['ElasticNet','R2']:.2f}",
             f"{mm['ElasticNet']['R2']:.2f}"],
            ["LinearSVR OOF R²", f"{primary.loc['LinearSVR','R2']:.2f}",
             f"{mm['LinearSVR']['R2']:.2f}"],
            ["Age-ρ sign concordance", "—",
             f"{s['whole_thymus_spearman_sign_concordance']:.1%}"],
            ["Age genes q<0.05", str(s['whole_thymus_main_sig_q05']),
             str(s['whole_thymus_doublet_removed_sig_q05'])],
            ["Recurrent-gene Jaccard (main consensus)", "—", f"{jac_cons:.2f}"]]
    table = ax3.table(cellText=rows[1:], colLabels=rows[0], loc="center",
                      cellLoc="center", colWidths=[0.52, 0.20, 0.28])
    table.auto_set_font_size(False); table.set_fontsize(6); table.scale(1, 1.35)
    ax3.set_title(f"Doublet sensitivity: {s['cells_removed_doublets']} cells removed "
                  f"({s['pct_removed']}%); gene-level agreement is limited",
                  fontsize=7, pad=8)
    pd.DataFrame(rows[1:], columns=rows[0]).to_csv(
        FIGF / "s04_doublet_sensitivity_summary.tsv", sep="\t", index=False)
    save_supp(fig, "Supplementary_Figure_04_QC_scrublet_doublet_sensitivity")


def s05_annotation():
    mr = pd.read_csv(META / "07_annotation_match_rate_by_library.csv")
    el = pd.read_csv(META / "07_analysis_eligibility_counts.csv")
    fz = load_freeze()
    fig = plt.figure(figsize=(W_DOUBLE, 70 / 25.4))
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.5)
    ax = fig.add_subplot(gs[0, 0])
    m = mr.merge(fz[["donor_id", "age_years"]], on="donor_id", how="left").sort_values("age_years")
    ax.bar(range(len(m)), m.match_rate, color="#5B8DB8")
    ax.set_xticks(range(len(m))); ax.set_xticklabels(
        [short(d) for d in m.donor_id], rotation=90, fontsize=5)
    ax.set_ylim(0, 1.05); ax.set_ylabel("author-label match rate")
    ax.set_title("GSM+barcode annotation match", fontsize=6.8)
    ax2 = fig.add_subplot(gs[0, 1])
    e = el.groupby("annotation_analysis_eligible").n_cells.sum()
    ax2.bar(["eligible", "excluded"], [e.get(True, 0), e.get(False, 0)],
            color=["#0072B2", "#C98A8A"])
    ax2.set_ylabel("cells"); ax2.set_title("Analysis eligibility", fontsize=6.8)
    ax3 = fig.add_subplot(gs[0, 2])
    qc = pd.read_csv(PSEUDO / "pseudobulk_qc_summary_all.csv")
    ncols = [c for c in qc.columns if c.startswith("n_cells_")]
    tot = qc[ncols].sum().sort_values(ascending=False)
    tot.index = [c.replace("n_cells_", "").replace("_", " ") for c in tot.index]
    ax3.barh(range(len(tot))[::-1], tot.values, color="#5B8DB8")
    ax3.set_yticks(range(len(tot))[::-1]); ax3.set_yticklabels(tot.index, fontsize=4.6)
    ax3.set_xlabel("cells"); ax3.set_title("Cells per broad type (all donors)", fontsize=6.8)
    save_supp(fig, "Supplementary_Figure_05_annotation_audit")


def s06_full_ml():
    pred = pd.read_csv(ML / "predictions" / "fold_predictions_regression.csv")
    mets = pd.read_csv(ML / "metrics" / "regression_metrics_oof.csv")
    perm = pd.read_csv(ML / "metrics" / "regression_permutation_test.csv")
    fig = plt.figure(figsize=(W_DOUBLE, 145 / 25.4))
    gs = gridspec.GridSpec(2, 3, figure=fig, left=0.10, right=0.95,
                           bottom=0.15, top=0.80, hspace=0.65, wspace=0.65)
    names = ["ElasticNet", "LinearSVR", "RF", "XGB", "MeanAgeNull"]
    for k, name in enumerate(names):
        ax = fig.add_subplot(gs[k // 3, k % 3])
        s = pred[pred.model == name]
        ax.scatter(s.y_true, s.y_pred, s=12, color="#0072B2")
        lo = min(s.y_true.min(), s.y_pred.min()) - 2
        hi = max(s.y_true.max(), s.y_pred.max()) + 2
        ax.plot([lo, hi], [lo, hi], ls=":", color=GREY, lw=0.8)
        r = mets[mets.model == name].iloc[0]
        ax.set_title(name, fontsize=7)
        ax.text(0.02, 1.02, f"R²={r.R2:.2f}; MAE={r.MAE:.1f} yr",
                transform=ax.transAxes, fontsize=5.5, va="bottom")
        ax.set_xlabel("observed"); ax.set_ylabel("predicted")
        ax.tick_params(labelsize=5.4)
        ax.set_xlim(0, 73); ax.set_ylim(0, 73)
    # female-only
    fr = pd.read_csv(ML / "sensitivity_female" / "metrics" /
                     "regression_metrics_oof.csv")
    fp = pd.read_csv(ML / "sensitivity_female" / "predictions" /
                     "fold_predictions_regression.csv")
    ax = fig.add_subplot(gs[1, 2])
    fsub = fp[fp.model == "ElasticNet"]
    ax.scatter(fsub.y_true, fsub.y_pred, s=14, color=FEMALE)
    lo = min(fsub.y_true.min(), fsub.y_pred.min()) - 2
    hi = max(fsub.y_true.max(), fsub.y_pred.max()) + 2
    ax.plot([lo, hi], [lo, hi], ls=":", color=GREY, lw=0.8)
    r = fr[fr.model == "ElasticNet"].iloc[0]
    ax.set_title("Female-only (n=14)", fontsize=7)
    ax.text(0.02, 1.02, f"R²={r.R2:.2f}; MAE={r.MAE:.1f} yr",
            transform=ax.transAxes, fontsize=5.5, va="bottom")
    ax.set_xlabel("observed"); ax.set_ylabel("predicted")
    ax.tick_params(labelsize=5.4)
    ax.set_xlim(0, 73); ax.set_ylim(0, 73)
    fig.suptitle("Continuous-age OOF models and female-only sensitivity",
                 fontsize=8, y=0.96)
    fig.text(0.10, 0.04, "Fixed-OOF prediction–age association permutation is not "
             "a model-training permutation test; P values omitted from panels.",
             fontsize=5.8)
    save_supp(fig, "Supplementary_Figure_06_full_ML_female_sensitivity")

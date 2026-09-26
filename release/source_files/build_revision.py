from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
try:
    import docx  # noqa: F401
except ModuleNotFoundError:
    sys.path.append(
        r"C:\Users\DELL\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\Lib\site-packages"
    )
from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor
from docx.text.paragraph import Paragraph


ROOT = Path(r"F:\论文材料")
SOURCE = Path(
    r"D:\xwechat_files\wxid_i36un3grasw822_084b\msg\file\2026-09\胸腺衰老初稿C-000.docx"
)
WORK = ROOT / "manuscript_C000_atlas_revision"
ATLAS = ROOT / "10_results" / "human_thymic_aging_atlas"
SCORE_DIR = ROOT / "05_deep_learning" / "thymic_aging_score"
FIGURE_PNG = WORK / "Figure6_human_thymic_aging_atlas_and_score.png"
FIGURE_PDF = WORK / "Figure6_human_thymic_aging_atlas_and_score.pdf"
FIGURE_SOURCE = WORK / "Figure6_source_data.xlsx"
FIGURE3_PNG = WORK / "Figure3_fully_nested_final_OOF.png"
FIGURE3_PDF = WORK / "Figure3_fully_nested_final_OOF.pdf"
FIGURE3_SOURCE = WORK / "Figure3_final_OOF_source_data.xlsx"
SUPP_ML_PNG = WORK / "Supplementary_Figure_S2_final_ML_and_confounding.png"
SUPP_ML_PDF = WORK / "Supplementary_Figure_S2_final_ML_and_confounding.pdf"
REAL_UMAP_PANEL = WORK / "human_thymus_real_umap_panel.png"
AGE_UMAP_PANEL = WORK / "human_thymus_age_umap_panel.png"
SEX_UMAP_PANEL = WORK / "human_thymus_sex_umap_panel.png"
DONOR_UMAP_PANEL = WORK / "human_thymus_donor_umap_panel.png"
EXT_FIGURE = ROOT / "10_results" / "figures_external_evidence" / "Figure_external_cross_dataset_support.png"
EXT_SOURCE = ROOT / "10_results" / "figures_external_evidence" / "source_data"
SENSITIVITY_FILE = ROOT / "10_results" / "submission_completion" / "final_result_summary.md"
OUTPUT = ROOT / "胸腺衰老初稿C-000_加入人类胸腺图谱和评分表_20260924.docx"


def set_font(run, name="Times New Roman", size=None, bold=None, italic=None, color=None):
    run.font.name = name
    if run._element.get_or_add_rPr().rFonts is None:
        run._element.get_or_add_rPr().append(OxmlElement("w:rFonts"))
    rfonts = run._element.get_or_add_rPr().rFonts
    for attr in ("ascii", "hAnsi", "eastAsia", "cs"):
        rfonts.set(qn(f"w:{attr}"), name)
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if italic is not None:
        run.italic = italic
    if color is not None:
        run.font.color.rgb = RGBColor(*color)


def shade_cell(cell, fill):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_cell_width(cell, inches):
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_w = tc_pr.find(qn("w:tcW"))
    if tc_w is None:
        tc_w = OxmlElement("w:tcW")
        tc_pr.append(tc_w)
    tc_w.set(qn("w:w"), str(int(inches * 1440)))
    tc_w.set(qn("w:type"), "dxa")


def repeat_table_header(row):
    tr_pr = row._tr.get_or_add_trPr()
    tbl_header = OxmlElement("w:tblHeader")
    tbl_header.set(qn("w:val"), "true")
    tr_pr.append(tbl_header)


def prevent_row_split(row):
    tr_pr = row._tr.get_or_add_trPr()
    cant_split = OxmlElement("w:cantSplit")
    tr_pr.append(cant_split)


def add_alt_text(inline_shape, title, description):
    doc_pr = inline_shape._inline.docPr
    doc_pr.set("title", title)
    doc_pr.set("descr", description)


def load_data():
    composition = pd.read_csv(ATLAS / "atlas_composition_age_effects.tsv", sep="\t")
    genes = pd.read_csv(ATLAS / "atlas_gene_age_effects.tsv.gz", sep="\t")
    developmental = pd.read_csv(ATLAS / "atlas_developmental_context.tsv", sep="\t")
    pathways = pd.read_csv(ATLAS / "atlas_pathways.tsv", sep="\t")
    scores = pd.read_csv(ATLAS / "atlas_donor_scores.tsv", sep="\t")
    donor_manifest = pd.read_csv(ATLAS / "atlas_donor_manifest.tsv", sep="\t")
    return composition, genes, developmental, pathways, scores, donor_manifest


def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residual = y_true - y_pred
    ss_res = float(np.sum(residual ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - ss_res / ss_tot
    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    rho = float(pd.Series(y_true).rank().corr(pd.Series(y_pred).rank()))
    return {"R2": r2, "MAE": mae, "RMSE": rmse, "Spearman_rho": rho}


def final_oof_audits(scores):
    frame = scores[[
        "donor_id", "chronological_age", "elasticnet_OOF_predicted_age",
        "HumanThymusFormer_strict_OOF_predicted_age", "sex",
        "raw_age_gap_years", "thymic_age_deviation_years",
    ]].copy()
    frame = frame.rename(columns={
        "chronological_age": "y_true",
        "elasticnet_OOF_predicted_age": "elasticnet_pred",
        "HumanThymusFormer_strict_OOF_predicted_age": "htf_pred",
    })
    ages = frame["y_true"].to_numpy(float)
    frame["null_pred"] = [(ages.sum() - a) / (len(ages) - 1) for a in ages]
    frame["elasticnet_residual"] = frame["elasticnet_pred"] - frame["y_true"]
    frame["elasticnet_abs_error"] = frame["elasticnet_residual"].abs()
    frame["null_abs_error"] = (frame["null_pred"] - frame["y_true"]).abs()
    frame["htf_abs_error"] = (frame["htf_pred"] - frame["y_true"]).abs()

    metrics = []
    for model, col in [
        ("ElasticNet", "elasticnet_pred"),
        ("MeanAgeNull", "null_pred"),
        ("HumanThymusFormer_strict_OOF", "htf_pred"),
    ]:
        row = {"model": model, "n_donors": len(frame)}
        row.update(regression_metrics(frame["y_true"], frame[col]))
        metrics.append(row)
    metrics = pd.DataFrame(metrics)

    rng = np.random.default_rng(371)
    boot_rows = []
    y = frame["y_true"].to_numpy(float)
    pred = frame["elasticnet_pred"].to_numpy(float)
    for b in range(2000):
        idx = rng.integers(0, len(frame), len(frame))
        if np.unique(y[idx]).size < 2:
            continue
        row = {"bootstrap": b}
        row.update(regression_metrics(y[idx], pred[idx]))
        boot_rows.append(row)
    bootstrap = pd.DataFrame(boot_rows)

    paired = frame["elasticnet_abs_error"].to_numpy() - frame["null_abs_error"].to_numpy()
    rng = np.random.default_rng(371 + 17)
    delta_samples = np.asarray([
        paired[rng.integers(0, len(paired), len(paired))].mean() for _ in range(2000)
    ])
    paired_summary = pd.DataFrame([{
        "n_donors": len(frame),
        "delta_MAE_model_minus_null": float(paired.mean()),
        "ci_low": float(np.quantile(delta_samples, 0.025)),
        "ci_high": float(np.quantile(delta_samples, 0.975)),
    }])

    full = regression_metrics(frame["y_true"], frame["elasticnet_pred"])
    influence_rows = []
    for donor in frame["donor_id"]:
        sub = frame.loc[~frame["donor_id"].eq(donor)]
        met = regression_metrics(sub["y_true"], sub["elasticnet_pred"])
        influence_rows.append({
            "excluded_donor": donor,
            "R2_without_donor": met["R2"],
            "delta_R2_vs_full": met["R2"] - full["R2"],
            "MAE_without_donor": met["MAE"],
            "delta_MAE_vs_full": met["MAE"] - full["MAE"],
            "interpretation": "jackknife influence on fixed pooled OOF predictions; model not refit",
        })
    influence = pd.DataFrame(influence_rows)
    return frame, metrics, bootstrap, paired_summary, influence


def panel_letter(ax, letter):
    ax.text(-0.16, 1.06, letter, transform=ax.transAxes, fontsize=11,
            fontweight="bold", va="top", ha="left")


def build_final_ml_figures(scores, donor_manifest):
    frame, metrics, bootstrap, paired_summary, influence = final_oof_audits(scores)
    en = metrics.loc[metrics["model"].eq("ElasticNet")].iloc[0]
    null = metrics.loc[metrics["model"].eq("MeanAgeNull")].iloc[0]
    htf = metrics.loc[metrics["model"].eq("HumanThymusFormer_strict_OOF")].iloc[0]
    ci = bootstrap[["R2", "MAE", "RMSE"]].quantile([0.025, 0.975])

    red = "#B84A4A"
    blue = "#4575B4"
    teal = "#2A9D8F"
    orange = "#E07B39"
    gray = "#6F6F6F"
    light = "#E4E4E4"

    # Main Figure 3: only final fully nested OOF outputs.
    fig = plt.figure(figsize=(7.15, 7.2))
    gs = fig.add_gridspec(2, 2, left=0.10, right=0.98, top=0.96, bottom=0.08,
                          hspace=0.36, wspace=0.34)

    ax = fig.add_subplot(gs[0, 0])
    ax.set_axis_off()
    steps = [
        (0.03, 0.70, "Outer LODO\nhold out 1 donor"),
        (0.38, 0.70, "Inner 3-fold donor CV\nrepeat screening, imputation,\nscaling and tuning"),
        (0.38, 0.31, "Refit selected pipeline\non 17 training donors"),
        (0.03, 0.31, "Predict held-out donor\nonce; pool 18 OOF values"),
    ]
    for x, y, text in steps:
        box = FancyBboxPatch((x, y), 0.28, 0.18, boxstyle="round,pad=0.018",
                             transform=ax.transAxes, facecolor="#F3F6F8",
                             edgecolor="#5C6B73", linewidth=0.8)
        ax.add_patch(box)
        ax.text(x + 0.14, y + 0.09, text, transform=ax.transAxes,
                ha="center", va="center", fontsize=7)
    arrows = [((0.31, 0.79), (0.38, 0.79)), ((0.52, 0.70), (0.52, 0.49)),
              ((0.38, 0.40), (0.31, 0.40)), ((0.17, 0.31), (0.17, 0.18))]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, xycoords=ax.transAxes,
                    arrowprops=dict(arrowstyle="->", color="#5C6B73", lw=0.9))
    ax.text(0.17, 0.10, "No held-out donor enters feature selection or tuning",
            transform=ax.transAxes, ha="center", va="center", fontsize=6.6,
            color="#444444")
    ax.set_title("Final fully nested donor-level workflow", pad=5)
    panel_letter(ax, "A")

    ax = fig.add_subplot(gs[0, 1])
    ax.scatter(frame["y_true"], frame["elasticnet_pred"], c=teal, s=35,
               edgecolor="white", linewidth=0.5, zorder=3)
    lo, hi = 0, 75
    ax.plot([lo, hi], [lo, hi], color="#555555", lw=0.8, ls="--")
    for _, r in frame.iterrows():
        ax.text(r["y_true"] + 0.8, r["elasticnet_pred"] + 0.3, r["donor_id"], fontsize=5.4)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Chronological age (years)")
    ax.set_ylabel("Elastic Net OOF predicted age (years)")
    ax.set_title("Final 18 held-out predictions", pad=5)
    ax.text(0.04, 0.96,
            f"R²={en.R2:.3f}\nMAE={en.MAE:.2f} y\nRMSE={en.RMSE:.2f} y\nSpearman ρ={en.Spearman_rho:.3f}",
            transform=ax.transAxes, ha="left", va="top", fontsize=6.8,
            bbox=dict(facecolor="white", edgecolor="#BBBBBB", linewidth=0.5, pad=2))
    ax.grid(color="#ECECEC", lw=0.5)
    panel_letter(ax, "B")

    ax = fig.add_subplot(gs[1, 0])
    order = frame.sort_values("y_true").reset_index(drop=True)
    for _, r in order.iterrows():
        ax.plot([0, 1], [r["elasticnet_abs_error"], r["null_abs_error"]],
                color=light, lw=0.8, zorder=1)
    ax.scatter(np.zeros(len(order)), order["elasticnet_abs_error"], color=teal, s=25, zorder=2)
    ax.scatter(np.ones(len(order)), order["null_abs_error"], color=gray, marker="s", s=22, zorder=2)
    ax.set_xticks([0, 1], ["Elastic Net", "Training-mean null"])
    ax.set_ylabel("Absolute OOF error (years)")
    ax.set_title("Paired donor-level error comparison", pad=5)
    ps = paired_summary.iloc[0]
    ax.text(0.5, 0.97,
            f"ΔMAE={ps.delta_MAE_model_minus_null:.2f} y\n95% CI {ps.ci_low:.2f} to {ps.ci_high:.2f}",
            transform=ax.transAxes, ha="center", va="top", fontsize=6.8,
            bbox=dict(facecolor="white", edgecolor="#BBBBBB", linewidth=0.5, pad=2))
    ax.grid(axis="y", color="#ECECEC", lw=0.5)
    panel_letter(ax, "C")

    ax = fig.add_subplot(gs[1, 1])
    inf = influence.sort_values("delta_R2_vs_full").reset_index(drop=True)
    yy = np.arange(len(inf))
    col = np.where(inf["delta_R2_vs_full"] >= 0, red, blue)
    ax.axvline(0, color="#555555", lw=0.7)
    ax.hlines(yy, 0, inf["delta_R2_vs_full"], color=col, lw=1.1)
    ax.scatter(inf["delta_R2_vs_full"], yy, color=col, s=22)
    ax.set_yticks(yy, inf["excluded_donor"])
    ax.set_xlabel("Change in pooled OOF R² after exclusion")
    ax.set_title("Donor-influence audit without model refitting", pad=5)
    ax.grid(axis="x", color="#ECECEC", lw=0.5)
    ax.text(0.0, -0.18, "This jackknife summarizes influence on fixed OOF pairs; it is not a refitted sensitivity model.",
            transform=ax.transAxes, ha="left", va="top", fontsize=6.2, color="#444444")
    panel_letter(ax, "D")

    for axis in fig.axes:
        if hasattr(axis, "spines"):
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
    fig.savefig(FIGURE3_PNG, dpi=450, bbox_inches="tight", facecolor="white")
    fig.savefig(FIGURE3_PDF, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # Replacement Supplementary Figure S2: diagnostics tied to the same 18-row table.
    fig = plt.figure(figsize=(7.15, 8.4))
    gs = fig.add_gridspec(3, 2, left=0.10, right=0.98, top=0.97, bottom=0.07,
                          hspace=0.45, wspace=0.36)

    ax = fig.add_subplot(gs[0, 0])
    ax.scatter(frame["y_true"], frame["elasticnet_pred"], color=teal, s=30, label="Elastic Net")
    ax.scatter(frame["y_true"], frame["htf_pred"], facecolors="none", edgecolors=orange,
               marker="s", s=28, label="HumanThymusFormer strict OOF")
    ax.plot([0, 75], [0, 75], color="#555555", lw=0.7, ls="--")
    ax.set_xlim(0, 75); ax.set_ylim(0, 85)
    ax.set_xlabel("Chronological age (years)"); ax.set_ylabel("OOF predicted age (years)")
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("Model-specific held-out predictions")
    panel_letter(ax, "A")

    ax = fig.add_subplot(gs[0, 1])
    metric_names = ["R2", "MAE", "RMSE"]
    vals = [en[m] for m in metric_names]
    lows = [ci.loc[0.025, m] for m in metric_names]
    highs = [ci.loc[0.975, m] for m in metric_names]
    xx = np.arange(3)
    ax.errorbar(xx, vals, yerr=[np.asarray(vals) - lows, highs - np.asarray(vals)], fmt="o",
                color=teal, ecolor="#6EAFA7", capsize=3)
    ax.set_xticks(xx, ["R²", "MAE, y", "RMSE, y"])
    ax.set_title("Donor-bootstrap uncertainty (2,000 resamples)")
    for x, v in zip(xx, vals):
        ax.text(x, v, f" {v:.2f}", va="bottom", ha="left", fontsize=6.2)
    ax.grid(axis="y", color="#ECECEC", lw=0.5)
    panel_letter(ax, "B")

    ax = fig.add_subplot(gs[1, 0])
    sc = scores.sort_values("chronological_age")
    ax.scatter(sc["chronological_age"], sc["raw_age_gap_years"], color=gray, s=24, label="Raw age gap")
    ax.scatter(sc["chronological_age"], sc["thymic_age_deviation_years"], color=blue, marker="s",
               s=22, label="Age-adjusted deviation")
    ax.axhline(0, color="#555555", lw=0.7)
    ax.set_xlabel("Chronological age (years)"); ax.set_ylabel("Difference (years)")
    ax.set_title("Age-bias calibration of the exploratory score")
    ax.legend(frameon=False)
    ax.grid(color="#ECECEC", lw=0.5)
    panel_letter(ax, "C")

    ax = fig.add_subplot(gs[1, 1])
    melt = frame[["donor_id", "elasticnet_abs_error", "htf_abs_error", "null_abs_error"]].melt(
        id_vars="donor_id", var_name="model", value_name="absolute_error")
    names = ["elasticnet_abs_error", "htf_abs_error", "null_abs_error"]
    xlabels = ["Elastic Net", "HTF strict OOF", "Mean-age null"]
    for i, name in enumerate(names):
        vals_i = melt.loc[melt["model"].eq(name), "absolute_error"].to_numpy()
        jitter = np.linspace(-0.07, 0.07, len(vals_i))
        ax.scatter(np.full(len(vals_i), i) + jitter, vals_i, s=18,
                   color=[teal, orange, gray][i], alpha=0.85)
        ax.plot([i - 0.18, i + 0.18], [np.median(vals_i)] * 2, color="#222222", lw=1.2)
    ax.set_xticks(range(3), xlabels, rotation=15, ha="right")
    ax.set_ylabel("Absolute OOF error (years)")
    ax.set_title("Per-donor error distributions")
    ax.grid(axis="y", color="#ECECEC", lw=0.5)
    panel_letter(ax, "D")

    ax = fig.add_subplot(gs[2, 0])
    inf = influence.sort_values("delta_MAE_vs_full")
    yy = np.arange(len(inf))
    col = np.where(inf["delta_MAE_vs_full"] <= 0, blue, red)
    ax.axvline(0, color="#555555", lw=0.7)
    ax.hlines(yy, 0, inf["delta_MAE_vs_full"], color=col, lw=1.0)
    ax.scatter(inf["delta_MAE_vs_full"], yy, color=col, s=20)
    ax.set_yticks(yy, inf["excluded_donor"])
    ax.set_xlabel("Change in pooled OOF MAE after exclusion (years)")
    ax.set_title("Donor influence on fixed OOF errors")
    ax.grid(axis="x", color="#ECECEC", lw=0.5)
    panel_letter(ax, "E")

    ax = fig.add_subplot(gs[2, 1])
    dm = donor_manifest.sort_values("age_years")
    positions = {"Female": 0, "Male": 1}
    for sex, group in dm.groupby("sex"):
        x = positions[sex]
        jitter = np.linspace(-0.08, 0.08, len(group))
        ax.scatter(np.full(len(group), x) + jitter, group["age_years"],
                   color=red if sex == "Female" else blue,
                   marker="o" if sex == "Female" else "s", s=28)
        for j, (_, r) in enumerate(group.iterrows()):
            ax.text(x + jitter[j] + 0.03, r["age_years"], r["donor_id"], fontsize=5.2, va="center")
    ax.set_xticks([0, 1], ["Female (n=14)", "Male (n=4)"])
    ax.set_ylabel("Chronological age (years)")
    ax.set_title("Age–sex structure of the primary cohort")
    ax.set_xlim(-0.35, 1.35)
    ax.grid(axis="y", color="#ECECEC", lw=0.5)
    panel_letter(ax, "F")

    for axis in fig.axes:
        if hasattr(axis, "spines"):
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
    fig.savefig(SUPP_ML_PNG, dpi=450, bbox_inches="tight", facecolor="white")
    fig.savefig(SUPP_ML_PDF, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    with pd.ExcelWriter(FIGURE3_SOURCE, engine="openpyxl") as writer:
        frame.to_excel(writer, sheet_name="final_18_OOF", index=False)
        metrics.to_excel(writer, sheet_name="recomputed_metrics", index=False)
        bootstrap.to_excel(writer, sheet_name="bootstrap_2000", index=False)
        paired_summary.to_excel(writer, sheet_name="paired_vs_null", index=False)
        influence.to_excel(writer, sheet_name="donor_influence", index=False)
        donor_manifest.to_excel(writer, sheet_name="donor_manifest", index=False)
    return frame, metrics, bootstrap, paired_summary, influence


def build_figure(composition, genes, developmental, scores):
    font_files = [
        r"C:\Windows\Fonts\times.ttf",
        r"C:\Windows\Fonts\timesbd.ttf",
        r"C:\Windows\Fonts\timesi.ttf",
        r"C:\Windows\Fonts\timesbi.ttf",
    ]
    for f in font_files:
        if Path(f).exists():
            mpl.font_manager.fontManager.addfont(f)
    mpl.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 7.5,
            "axes.titlesize": 8.5,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 6.7,
            "ytick.labelsize": 6.7,
            "legend.fontsize": 6.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.65,
        }
    )
    red = "#B84A4A"
    blue = "#4575B4"
    teal = "#2A9D8F"
    gray = "#7A7A7A"
    light = "#D9D9D9"

    fig = plt.figure(figsize=(7.15, 8.55), constrained_layout=False)
    gs = fig.add_gridspec(
        2,
        2,
        left=0.11,
        right=0.985,
        top=0.975,
        bottom=0.075,
        hspace=0.34,
        wspace=0.43,
        height_ratios=[1.0, 1.05],
    )

    # A: composition-aware age effects.
    ax = fig.add_subplot(gs[0, 0])
    ax_a = ax
    comp = composition.sort_values("estimate").reset_index(drop=True)
    y = np.arange(len(comp))
    sig = comp["q_value_BH"] < 0.05
    ax.axvline(0, color="#555555", lw=0.7)
    ax.hlines(y, comp["ci_low_95"], comp["ci_high_95"], color=light, lw=1.2, zorder=1)
    ax.scatter(
        comp.loc[~sig, "estimate"], y[~sig], facecolors="white", edgecolors=gray,
        s=22, linewidths=0.8, zorder=2, label="BH q≥0.05"
    )
    ax.scatter(
        comp.loc[sig, "estimate"], y[sig], color=teal, edgecolors="#1C6F65",
        s=25, linewidths=0.5, zorder=3, label="BH q<0.05"
    )
    ax.set_yticks(y, comp["cell_type"])
    ax.set_xlabel("Sex-adjusted CLR age effect per 1 SD (95% CI)")
    ax.set_title("Composition-aware donor-level effects", pad=5)
    ax.grid(axis="x", color="#E6E6E6", lw=0.5)
    ax.legend(frameon=False, loc="lower right", handletextpad=0.4)

    # B: counts of significant expression effects by context.
    ax = fig.add_subplot(gs[0, 1])
    ax_b = ax
    sig_genes = genes.loc[genes["adj.P.Val"] < 0.05].copy()
    sig_genes["direction"] = np.where(sig_genes["logFC_age_perSD"] > 0, "Age-up", "Age-down")
    counts = (
        sig_genes.groupby(["cell_context", "direction"]).size().unstack(fill_value=0)
        .reindex(genes["cell_context"].drop_duplicates())
        .fillna(0)
    )
    coverage = genes.groupby("cell_context")["n_donors"].agg(["min", "max"])
    counts["total"] = counts.sum(axis=1)
    counts = counts.sort_values("total")
    yy = np.arange(len(counts))
    up = counts.get("Age-up", pd.Series(0, index=counts.index)).astype(int)
    down = counts.get("Age-down", pd.Series(0, index=counts.index)).astype(int)
    x_up = np.log10(up + 1)
    x_down = np.log10(down + 1)
    ax.scatter(x_up, yy + 0.13, color=red, marker="o", s=25, label="Age-up")
    ax.scatter(x_down, yy - 0.13, color=blue, marker="s", s=22, label="Age-down")
    for i, (u, d) in enumerate(zip(up, down)):
        if u > 0:
            ax.text(x_up.iloc[i] + 0.035, i + 0.13, f"{u}", va="center", fontsize=5.8, color=red)
        if d > 0:
            ax.text(x_down.iloc[i] + 0.035, i - 0.13, f"{d}", va="center", fontsize=5.8, color=blue)
    labels = []
    for ctx in counts.index:
        lo, hi = coverage.loc[ctx, ["min", "max"]]
        nlab = f"n={int(lo)}" if lo == hi else f"n={int(lo)}–{int(hi)}"
        labels.append(f"{ctx} ({nlab})")
    ax.set_yticks(yy, labels)
    ticks = np.arange(0, 4, 1)
    ax.set_xticks(ticks, ["0", "9", "99", "999"])
    ax.set_xlim(-0.05, max(3.65, float(max(x_up.max(), x_down.max())) + 0.35))
    ax.set_xlabel("Number of BH q<0.05 genes (log10[n+1])")
    ax.set_title("Cell-context transcriptional atlas", pad=5)
    ax.grid(axis="x", color="#E6E6E6", lw=0.5)
    ax.legend(frameon=False, loc="lower right", ncol=2, handletextpad=0.35, columnspacing=0.8)

    # C: developmental localization for top whole-thymus age genes.
    ax = fig.add_subplot(gs[1, 0])
    ax_c = ax
    whole = genes.loc[genes["cell_context"].eq("whole_thymus")].sort_values(["adj.P.Val", "P.Value"])
    present = set(developmental["gene"].dropna())
    top_up = whole.loc[(whole["logFC_age_perSD"] > 0) & whole["gene"].isin(present)].head(6)
    top_down = whole.loc[(whole["logFC_age_perSD"] < 0) & whole["gene"].isin(present)].head(6)
    selected = pd.concat([top_up, top_down], ignore_index=True)
    selected_genes = selected["gene"].tolist()
    stages = ["DN1", "DN2", "DN3", "ISP", "DP CD3min", "DP CD3plus"]
    heat = (
        developmental.loc[developmental["gene"].isin(selected_genes)]
        .pivot_table(index="gene", columns="developmental_stage", values="mean_zscore", aggfunc="mean")
        .reindex(index=selected_genes, columns=stages)
    )
    vmax = np.nanpercentile(np.abs(heat.to_numpy()), 97)
    vmax = max(vmax, 0.25)
    im = ax.imshow(heat.to_numpy(), aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    directions = dict(zip(selected["gene"], np.where(selected["logFC_age_perSD"] > 0, "+", "−")))
    ax.set_yticks(np.arange(len(heat)), [f"{g} ({directions[g]})" for g in heat.index])
    ax.set_xticks(np.arange(len(stages)), stages, rotation=38, ha="right")
    ax.axvline(2.5, color="white", lw=1.5)
    ax.set_title("Developmental localization of top age-associated genes", pad=5)
    ax.text(1.0, -1.1, "Sort1", ha="center", va="bottom", fontsize=6.5)
    ax.text(4.0, -1.1, "Sort2", ha="center", va="bottom", fontsize=6.5)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Mean within-object z-score")
    cbar.ax.tick_params(labelsize=6.2)
    ax.text(
        0.0, -0.23,
        "+/− indicates the human whole-thymus age-effect direction;\nGSE195812 provides developmental context, not aging replication.",
        transform=ax.transAxes, ha="left", va="top", fontsize=6.2, color="#444444"
    )

    # D: age-adjusted internal score by donor.
    ax = fig.add_subplot(gs[1, 1])
    ax_d = ax
    sc = scores.sort_values("thymic_age_deviation_years").reset_index(drop=True)
    yy = np.arange(len(sc))
    vals = sc["thymic_age_deviation_years"].to_numpy()
    colors = np.where(vals >= 0, red, blue)
    ax.axvline(0, color="#555555", lw=0.8)
    ax.hlines(yy, 0, vals, color=colors, lw=1.35)
    ax.scatter(vals, yy, color=colors, s=25, zorder=3)
    labels = [f"{d} ({a:.0f} y)" for d, a in zip(sc["donor_id"], sc["chronological_age"])]
    ax.set_yticks(yy, labels)
    ax.set_xlabel("Age-adjusted thymic age deviation (years)")
    ax.set_title("Internal exploratory donor score", pad=5)
    ax.grid(axis="x", color="#E6E6E6", lw=0.5)
    ax.set_xlim(min(vals) - 2.2, max(vals) + 2.8)
    for i, (v, pct) in enumerate(zip(vals, sc["deviation_percentile_within_18_donors"])):
        ha = "left" if v >= 0 else "right"
        offset = 0.35 if v >= 0 else -0.35
        ax.text(v + offset, i, f"P{pct:.0f}", va="center", ha=ha, fontsize=5.7, color=colors[i])
    ax.text(
        0.0, -0.18,
        "Fully nested Elastic Net OOF prediction with leave-one-donor-out age-bias calibration;\nwithin-cohort percentile only; not a clinical biological-age clock.",
        transform=ax.transAxes, ha="left", va="top", fontsize=6.2, color="#444444"
    )

    for label, axis in zip("ABCD", [ax_a, ax_b, ax_c, ax_d]):
        axis.text(-0.16, 1.05, label, transform=axis.transAxes, fontsize=11, fontweight="bold", va="top")
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    fig.savefig(FIGURE_PNG, dpi=450, bbox_inches="tight", facecolor="white")
    fig.savefig(FIGURE_PDF, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    with pd.ExcelWriter(FIGURE_SOURCE, engine="openpyxl") as writer:
        comp.to_excel(writer, sheet_name="A_composition", index=False)
        counts.reset_index().merge(
            coverage.reset_index(), on="cell_context", how="left"
        ).to_excel(writer, sheet_name="B_gene_counts", index=False)
        heat.reset_index().to_excel(writer, sheet_name="C_development", index=False)
        sc.to_excel(writer, sheet_name="D_scores", index=False)


def build_atlas_figure_v2(composition, genes, developmental, scores):
    """Build the dedicated atlas figure with the real GSE231906 UMAP as panel A.

    The UMAP panel is a clean crop of the already frozen Figure 1F, whose source
    code reads X_umap and broad_celltype from the annotated 18-donor h5ad.  The
    underlying h5ad is not present in the local delivery, so the frozen panel is
    reused rather than inventing coordinates.
    """
    if not REAL_UMAP_PANEL.exists():
        raise FileNotFoundError(f"Frozen real UMAP panel missing: {REAL_UMAP_PANEL}")

    for f in [r"C:\Windows\Fonts\times.ttf", r"C:\Windows\Fonts\timesbd.ttf",
              r"C:\Windows\Fonts\timesi.ttf", r"C:\Windows\Fonts\timesbi.ttf"]:
        if Path(f).exists():
            mpl.font_manager.fontManager.addfont(f)
    mpl.rcParams.update({
        "font.family": "Times New Roman", "font.size": 7.5, "axes.titlesize": 8.5,
        "axes.labelsize": 7.5, "xtick.labelsize": 6.7, "ytick.labelsize": 6.7,
        "legend.fontsize": 6.7, "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.linewidth": 0.65,
    })

    red, blue, teal, gray, light = "#B84A4A", "#4575B4", "#2A9D8F", "#7A7A7A", "#D9D9D9"
    fig = plt.figure(figsize=(7.15, 7.75), constrained_layout=False)
    gs = fig.add_gridspec(2, 2, left=0.10, right=0.985, top=0.97, bottom=0.075,
                          hspace=0.38, wspace=0.40, height_ratios=[1.02, 1.0])

    # A: authentic integrated UMAP from the frozen primary atlas figure.
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.imshow(plt.imread(REAL_UMAP_PANEL))
    ax_a.set_axis_off()
    ax_a.set_title("Integrated human thymic single-cell atlas", pad=5)
    ax_a.text(
        0.0, -0.055,
        "18 donors; 114,957 QC-passed cells. Frozen Figure 1F UMAP\n"
        "uses a deterministic ≤50,000-cell display subsample (seed 249).",
        transform=ax_a.transAxes, ha="left", va="top", fontsize=6.2, color="#444444"
    )

    # B: composition-aware donor-level effects.
    ax_b = fig.add_subplot(gs[0, 1])
    comp = composition.sort_values("estimate").reset_index(drop=True)
    y = np.arange(len(comp))
    sig = comp["q_value_BH"] < 0.05
    ax_b.axvline(0, color="#555555", lw=0.7)
    ax_b.hlines(y, comp["ci_low_95"], comp["ci_high_95"], color=light, lw=1.2, zorder=1)
    ax_b.scatter(comp.loc[~sig, "estimate"], y[~sig], facecolors="white", edgecolors=gray,
                 s=21, linewidths=0.8, zorder=2, label="BH q≥0.05")
    ax_b.scatter(comp.loc[sig, "estimate"], y[sig], color=teal, edgecolors="#1C6F65",
                 s=24, linewidths=0.5, zorder=3, label="BH q<0.05")
    ax_b.set_yticks(y, comp["cell_type"])
    ax_b.set_xlabel("Sex-adjusted CLR age effect per 1 SD (95% CI)")
    ax_b.set_title("Composition-aware donor-level effects", pad=5)
    ax_b.grid(axis="x", color="#E6E6E6", lw=0.5)
    ax_b.legend(frameon=False, loc="lower right", handletextpad=0.4)

    # C: cell-context significant effect counts and donor coverage.
    ax_c = fig.add_subplot(gs[1, 0])
    sig_genes = genes.loc[genes["adj.P.Val"] < 0.05].copy()
    sig_genes["direction"] = np.where(sig_genes["logFC_age_perSD"] > 0, "Age-up", "Age-down")
    counts = (sig_genes.groupby(["cell_context", "direction"]).size().unstack(fill_value=0)
              .reindex(genes["cell_context"].drop_duplicates()).fillna(0))
    coverage = genes.groupby("cell_context")["n_donors"].agg(["min", "max"])
    counts["total"] = counts.sum(axis=1)
    counts = counts.sort_values("total")
    yy = np.arange(len(counts))
    up = counts.get("Age-up", pd.Series(0, index=counts.index)).astype(int)
    down = counts.get("Age-down", pd.Series(0, index=counts.index)).astype(int)
    x_up, x_down = np.log10(up + 1), np.log10(down + 1)
    ax_c.scatter(x_up, yy + 0.13, color=red, marker="o", s=24, label="Age-up")
    ax_c.scatter(x_down, yy - 0.13, color=blue, marker="s", s=21, label="Age-down")
    for i, (u, d) in enumerate(zip(up, down)):
        if u > 0:
            ax_c.text(x_up.iloc[i] + 0.035, i + 0.13, f"{u}", va="center", fontsize=5.7, color=red)
        if d > 0:
            ax_c.text(x_down.iloc[i] + 0.035, i - 0.13, f"{d}", va="center", fontsize=5.7, color=blue)
    labels = []
    for ctx in counts.index:
        lo, hi = coverage.loc[ctx, ["min", "max"]]
        labels.append(f"{ctx} (n={int(lo)})" if lo == hi else f"{ctx} (n={int(lo)}–{int(hi)})")
    ax_c.set_yticks(yy, labels)
    ax_c.set_xticks(np.arange(0, 4, 1), ["0", "9", "99", "999"])
    ax_c.set_xlim(-0.05, max(3.65, float(max(x_up.max(), x_down.max())) + 0.35))
    ax_c.set_xlabel("Number of BH q<0.05 genes (log10[n+1])")
    ax_c.set_title("Cell-context transcriptional aging atlas", pad=5)
    ax_c.grid(axis="x", color="#E6E6E6", lw=0.5)
    ax_c.legend(frameon=False, loc="lower right", ncol=2, handletextpad=0.35, columnspacing=0.8)

    # D: age-adjusted internal score by donor.
    ax_d = fig.add_subplot(gs[1, 1])
    sc = scores.sort_values("thymic_age_deviation_years").reset_index(drop=True)
    yy = np.arange(len(sc))
    vals = sc["thymic_age_deviation_years"].to_numpy()
    colors = np.where(vals >= 0, red, blue)
    ax_d.axvline(0, color="#555555", lw=0.8)
    ax_d.hlines(yy, 0, vals, color=colors, lw=1.35)
    ax_d.scatter(vals, yy, color=colors, s=24, zorder=3)
    ax_d.set_yticks(yy, [f"{d} ({a:.0f} y)" for d, a in zip(sc["donor_id"], sc["chronological_age"])])
    ax_d.set_xlabel("Age-adjusted thymic age deviation (years)")
    ax_d.set_title("Internal exploratory donor score", pad=5)
    ax_d.grid(axis="x", color="#E6E6E6", lw=0.5)
    ax_d.set_xlim(min(vals) - 2.2, max(vals) + 2.8)
    for i, (v, pct) in enumerate(zip(vals, sc["deviation_percentile_within_18_donors"])):
        ha, offset = ("left", 0.35) if v >= 0 else ("right", -0.35)
        ax_d.text(v + offset, i, f"P{pct:.0f}", va="center", ha=ha, fontsize=5.7, color=colors[i])
    ax_d.text(0.0, -0.17,
              "Fully nested Elastic Net OOF prediction with leave-one-donor-out age-bias calibration;\n"
              "within-cohort percentile only; not a clinical biological-age clock.",
              transform=ax_d.transAxes, ha="left", va="top", fontsize=6.1, color="#444444")

    for label, axis in zip("ABCD", [ax_a, ax_b, ax_c, ax_d]):
        axis.text(-0.16, 1.05, label, transform=axis.transAxes, fontsize=11,
                  fontweight="bold", va="top")
        if axis is not ax_a:
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)

    fig.savefig(FIGURE_PNG, dpi=450, bbox_inches="tight", facecolor="white")
    fig.savefig(FIGURE_PDF, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    umap_provenance = pd.DataFrame([{
        "display_source": "Frozen Figure 1F derived from annotated GSE231906 h5ad",
        "expected_h5ad": "/data/zxy/projects/human_thymus_age_ML_DL/01_raw_processing/filtered/07_GSE231906_thymus_annotated.h5ad",
        "coordinate_key": "X_umap",
        "celltype_key": "broad_celltype",
        "n_donors": 18,
        "n_QC_cells": 114957,
        "display_subsample_max_cells": 50000,
        "display_seed": 249,
        "local_limitation": "annotated h5ad absent; authentic frozen panel reused without inventing coordinates",
    }])
    with pd.ExcelWriter(FIGURE_SOURCE, engine="openpyxl") as writer:
        umap_provenance.to_excel(writer, sheet_name="A_UMAP_provenance", index=False)
        comp.to_excel(writer, sheet_name="B_composition", index=False)
        counts.reset_index().merge(coverage.reset_index(), on="cell_context", how="left").to_excel(
            writer, sheet_name="C_gene_counts", index=False)
        sc.to_excel(writer, sheet_name="D_scores", index=False)
        developmental.to_excel(writer, sheet_name="developmental_not_plotted", index=False)


def build_atlas_figure_v3(composition, genes, developmental, scores):
    """Dedicated UMAP atlas: cell type, age, sex and donor/library overlays."""
    panels = [REAL_UMAP_PANEL, AGE_UMAP_PANEL, SEX_UMAP_PANEL, DONOR_UMAP_PANEL]
    missing = [str(p) for p in panels if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Frozen UMAP panels missing: {missing}")

    for f in [r"C:\Windows\Fonts\times.ttf", r"C:\Windows\Fonts\timesbd.ttf",
              r"C:\Windows\Fonts\timesi.ttf", r"C:\Windows\Fonts\timesbi.ttf"]:
        if Path(f).exists():
            mpl.font_manager.fontManager.addfont(f)
    mpl.rcParams.update({
        "font.family": "Times New Roman", "font.size": 7.5,
        "axes.titlesize": 9.2, "pdf.fonttype": 42, "ps.fonttype": 42,
    })

    fig = plt.figure(figsize=(7.15, 7.1))
    gs = fig.add_gridspec(2, 3, left=0.035, right=0.985, top=0.95, bottom=0.08,
                          height_ratios=[1.02, 1.0], hspace=0.23, wspace=0.10)
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])
    ax_d = fig.add_subplot(gs[1, 2])

    # Panel A is wide because it includes the full broad-cell-type legend.
    ax_a.imshow(plt.imread(REAL_UMAP_PANEL))
    ax_a.set_title("Integrated human thymic single-cell transcriptome atlas", pad=4)
    ax_a.set_axis_off()

    overlay_data = [
        (AGE_UMAP_PANEL, "Chronological age"),
        (SEX_UMAP_PANEL, "Sex"),
        (DONOR_UMAP_PANEL, "Donor / library identity"),
    ]
    for ax, (path, title) in zip([ax_b, ax_c, ax_d], overlay_data):
        image = plt.imread(path)
        # Remove the old raw-field title while retaining the real UMAP and legend.
        image = image[70:, :, :]
        ax.imshow(image)
        ax.set_title(title, pad=3)
        ax.set_axis_off()

    for label, ax in zip("ABCD", [ax_a, ax_b, ax_c, ax_d]):
        ax.text(0.0, 1.02, label, transform=ax.transAxes, fontsize=11,
                fontweight="bold", ha="left", va="bottom")
    fig.text(
        0.035, 0.025,
        "All panels reuse the same frozen UMAP coordinates from the annotated GSE231906 object. "
        "The analytical object contains 114,957 QC-passed cells from 18 donors; a deterministic "
        "50,000-cell display subsample was drawn with seed 249. Panels B–D are visual diagnostics, "
        "not formal tests of age, sex or batch mixing.",
        fontsize=6.3, ha="left", va="bottom", wrap=True,
    )
    fig.savefig(FIGURE_PNG, dpi=450, bbox_inches="tight", facecolor="white")
    fig.savefig(FIGURE_PDF, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    provenance = pd.DataFrame([
        {
            "panel": "A", "color": "broad_celltype",
            "frozen_source": "Figure1F",
            "expected_h5ad": "/data/zxy/projects/human_thymus_age_ML_DL/01_raw_processing/filtered/07_GSE231906_thymus_annotated.h5ad",
            "coordinate_key": "X_umap", "n_donors": 18, "n_QC_cells": 114957,
            "display_cells": 50000, "display_seed": 249,
        },
        {
            "panel": "B", "color": "age_years",
            "frozen_source": "Supplementary_Figure_02_batch_diagnostic_UMAP",
            "expected_h5ad": "/data/zxy/projects/human_thymus_age_ML_DL/01_raw_processing/filtered/07_GSE231906_thymus_annotated.h5ad",
            "coordinate_key": "X_umap", "n_donors": 18, "n_QC_cells": 114957,
            "display_cells": 50000, "display_seed": 249,
        },
        {
            "panel": "C", "color": "sex",
            "frozen_source": "Supplementary_Figure_02_batch_diagnostic_UMAP",
            "expected_h5ad": "/data/zxy/projects/human_thymus_age_ML_DL/01_raw_processing/filtered/07_GSE231906_thymus_annotated.h5ad",
            "coordinate_key": "X_umap", "n_donors": 18, "n_QC_cells": 114957,
            "display_cells": 50000, "display_seed": 249,
        },
        {
            "panel": "D", "color": "gsm_id",
            "frozen_source": "Supplementary_Figure_02_batch_diagnostic_UMAP",
            "expected_h5ad": "/data/zxy/projects/human_thymus_age_ML_DL/01_raw_processing/filtered/07_GSE231906_thymus_annotated.h5ad",
            "coordinate_key": "X_umap", "n_donors": 18, "n_QC_cells": 114957,
            "display_cells": 50000, "display_seed": 249,
        },
    ])
    with pd.ExcelWriter(FIGURE_SOURCE, engine="openpyxl") as writer:
        provenance.to_excel(writer, sheet_name="UMAP_panel_provenance", index=False)
        composition.to_excel(writer, sheet_name="composition_not_plotted", index=False)
        genes.groupby("cell_context")["n_donors"].agg(["min", "max"]).reset_index().to_excel(
            writer, sheet_name="cell_context_coverage", index=False)
        scores.to_excel(writer, sheet_name="donor_scores_Table1", index=False)
        developmental.to_excel(writer, sheet_name="developmental_not_plotted", index=False)


def replace_text(paragraph, old, new):
    if old not in paragraph.text:
        return False
    full = paragraph.text.replace(old, new)
    for run in paragraph.runs:
        run.text = ""
    run = paragraph.runs[0] if paragraph.runs else paragraph.add_run()
    run.text = full
    set_font(run, size=10.5)
    return True


def set_paragraph_text(paragraph, text, size=10.5, bold=False, italic=False):
    paragraph._p.clear_content()
    run = paragraph.add_run(text)
    set_font(run, size=size, bold=bold, italic=italic)
    return paragraph


def replace_picture_before_caption(doc, caption_prefix, image_path, width=6.55, title="", description=""):
    caption = next(p for p in doc.paragraphs if p.text.strip().startswith(caption_prefix))
    previous = caption._p.getprevious()
    while previous is not None and previous.tag != qn("w:p"):
        previous = previous.getprevious()
    if previous is None:
        raise RuntimeError(f"No picture paragraph found before {caption_prefix}")
    picture_paragraph = Paragraph(previous, caption._parent)
    picture_paragraph._p.clear_content()
    picture_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    picture_paragraph.paragraph_format.keep_with_next = True
    shape = picture_paragraph.add_run().add_picture(str(image_path), width=Inches(width))
    add_alt_text(shape, title, description)
    return caption


def add_compact_table_before(anchor, title_text, columns, rows, note_text=None,
                             header_fill="D9E7F3", font_size=7.0, widths=None,
                             page_break=True):
    if page_break:
        pb = anchor.insert_paragraph_before()
        pb.add_run().add_break(WD_BREAK.PAGE)
    title = anchor.insert_paragraph_before()
    title.style = "Caption"
    title.paragraph_format.keep_with_next = True
    set_font(title.add_run(title_text), size=9, bold=True)
    table_width = Inches(sum(widths) if widths else 6.5)
    table = anchor._parent.add_table(rows=1, cols=len(columns), width=table_width)
    anchor._p.addprevious(table._tbl)
    table.style = "Table Grid"
    table.alignment = 1
    table.autofit = False
    repeat_table_header(table.rows[0])
    for j, label in enumerate(columns):
        cell = table.rows[0].cells[j]
        if widths:
            set_cell_width(cell, widths[j])
        shade_cell(cell, header_fill)
        cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
        cell.text = str(label)
        for p in cell.paragraphs:
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(0)
            for r in p.runs:
                set_font(r, size=font_size, bold=True)
    for row_values in rows:
        cells = table.add_row().cells
        for j, value in enumerate(row_values):
            if widths:
                set_cell_width(cells[j], widths[j])
            cells[j].vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            cells[j].text = str(value)
            for p in cells[j].paragraphs:
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT if j in (0, 1) else WD_ALIGN_PARAGRAPH.CENTER
                p.paragraph_format.space_before = Pt(0)
                p.paragraph_format.space_after = Pt(0)
                for r in p.runs:
                    set_font(r, size=font_size)
        prevent_row_split(table.rows[-1])
    if note_text:
        note = anchor.insert_paragraph_before()
        note.paragraph_format.space_before = Pt(3)
        note.paragraph_format.space_after = Pt(6)
        note.paragraph_format.keep_together = True
        set_font(note.add_run(note_text), size=max(font_size, 7.2), italic=True)
    return table


def add_results_section(doc, composition, genes, developmental, pathways, scores):
    discussion = next(p for p in doc.paragraphs if p.text.strip() == "DISCUSSION")

    heading = discussion.insert_paragraph_before(
        "A donor-resolved human thymic aging atlas and exploratory score integrate multi-layer evidence"
    )
    heading.style = doc.styles["Heading 2"]

    p = discussion.insert_paragraph_before()
    p.style = doc.styles["Normal"]
    text = (
        "We assembled the frozen cell- and donor-level outputs into a continuous-age human thymic aging atlas. "
        f"The atlas contains {len(composition):,} composition-effect records, {len(genes):,} gene-by-context "
        f"age-effect records, {len(pathways):,} pathway-enrichment records, {len(developmental):,} developmental-"
        f"localization records, and scores for {len(scores)} primary donors. The integrated UMAP displays the broad "
        "cell-type architecture of the 114,957 QC-passed cells from all 18 donors; the plotted points are the deterministic "
        "up-to-50,000-cell display subsample used in the frozen primary Figure 1 (seed 249; Fig. 6A). These outputs organize "
        "existing analyses and do not constitute a new cohort or external validation experiment. In the sex-adjusted "
        "centered log-ratio analysis, "
        "B-cell and fibroblast components increased with age, whereas TEC, SP-unresolved, and unknown components "
        "decreased at BH q<0.05. Because these are log-ratio effects, each coefficient describes a relative shift within "
        "the measured composition rather than an absolute gain or loss of cells."
    )
    r = p.add_run(text)
    set_font(r, size=10.5)

    p = discussion.insert_paragraph_before()
    p.style = doc.styles["Normal"]
    text = (
        "The transcriptional layer retained the complete sex-adjusted pseudobulk effect tables for whole thymus and "
        "ten annotated cell contexts. Donor coverage varied by context (10–18 donors), so counts of significant genes "
        "were used as an atlas index rather than a direct measure of biological effect size. The developmental layer "
        "projects selected whole-thymus age-associated genes onto DN1–DN3 and ISP–DP states using within-object "
        "z-scores; it remains a localization analysis because GSE195812 is not an adult aging cohort (Fig. 4). "
        "Figure 6B–C links the UMAP atlas to composition-aware effects and cell-context transcriptional effects."
    )
    r = p.add_run(text)
    set_font(r, size=10.5)

    p = discussion.insert_paragraph_before()
    p.style = doc.styles["Normal"]
    text = (
        "For each donor, the fully nested Elastic Net out-of-fold prediction was compared with the expected out-of-fold "
        "prediction at the same chronological age, estimated from the other 17 donors. The resulting age-adjusted thymic "
        f"age deviation ranged from {scores['thymic_age_deviation_years'].min():.1f} to "
        f"{scores['thymic_age_deviation_years'].max():.1f} years. Positive values indicate a relatively older position and "
        "negative values a relatively younger position within this 18-donor dataset after internal age-bias correction "
        "(Fig. 6D; Table 1). The associated percentile is a within-cohort rank and must not be interpreted as a clinical "
        "reference range, diagnosis, prognosis, or externally validated biological-age clock."
    )
    r = p.add_run(text)
    set_font(r, size=10.5)

    img_p = discussion.insert_paragraph_before()
    img_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    img_p.paragraph_format.keep_with_next = True
    shape = img_p.add_run().add_picture(str(FIGURE_PNG), width=Inches(6.55))
    add_alt_text(
        shape,
        "Human thymic aging atlas and internal score",
        "Four-panel figure showing an authentic integrated UMAP of human thymic cells colored by broad cell type, sex-adjusted composition effects, counts of age-associated genes by cell context, and age-adjusted internal donor score deviations.",
    )

    caption = discussion.insert_paragraph_before()
    caption.style = doc.styles["Figure Caption"]
    caption.paragraph_format.keep_together = True
    caption.paragraph_format.space_after = Pt(6)
    cap_text = (
        "Figure 6. Integrated human thymic single-cell atlas, cell-context aging effects and exploratory internal score. "
        "(A) Authentic integrated UMAP derived from the annotated GSE231906 object for all 18 donors and colored by broad "
        "cell type. The analytical object contains 114,957 QC-passed cells; for legibility, the frozen display panel plots "
        "a deterministic subsample of up to 50,000 cells (seed 249). The local delivery does not contain the annotated "
        "h5ad, so the already frozen Figure 1F panel was reused without recreating or inventing coordinates. "
        "(B) Sex-adjusted centered log-ratio composition effects per 1 standard deviation of age with HC3 95% confidence "
        "intervals; filled symbols denote BH q<0.05. Effects are relative to the full measured composition. "
        "(C) Counts of age-up and age-down genes with BH q<0.05 in whole-thymus and cell-type-specific sex-adjusted "
        "pseudobulk analyses. Donor coverage is shown beside each context and differs across contexts. "
        "(D) Age-adjusted donor deviations derived from fully nested Elastic Net out-of-fold predictions "
        "and leave-one-donor-out calibration of expected prediction at chronological age. P labels are percentiles within "
        "these 18 donors only. The score is an internally cross-validated proof of concept and is not clinically validated."
    )
    r = caption.add_run(cap_text)
    set_font(r, size=9)

    page_break = discussion.insert_paragraph_before()
    page_break.add_run().add_break(WD_BREAK.PAGE)

    title = discussion.insert_paragraph_before()
    title.style = doc.styles["Caption"]
    title.paragraph_format.keep_with_next = True
    title.alignment = WD_ALIGN_PARAGRAPH.LEFT
    r = title.add_run("Table 1. Internally cross-validated exploratory thymic aging scores for the 18 primary donors")
    set_font(r, size=9, bold=True)

    cols = [
        ("Donor", 0.65),
        ("Sex", 0.62),
        ("Chronological\nage, y", 0.90),
        ("Elastic Net OOF\npredicted age, y", 1.05),
        ("Age-adjusted\ndeviation, y", 0.90),
        ("Internal thymic\nbiological age, y", 1.05),
        ("Within-cohort\npercentile", 0.92),
    ]
    table = doc.add_table(rows=1, cols=len(cols))
    table.style = "Table Grid"
    table.alignment = 1
    table.autofit = False
    discussion._p.addprevious(table._tbl)
    header = table.rows[0]
    repeat_table_header(header)
    for j, (label, width) in enumerate(cols):
        cell = header.cells[j]
        set_cell_width(cell, width)
        cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
        shade_cell(cell, "D9E7F3")
        cell.text = label
        for pp in cell.paragraphs:
            pp.alignment = WD_ALIGN_PARAGRAPH.CENTER
            pp.paragraph_format.space_after = Pt(0)
            for rr in pp.runs:
                set_font(rr, size=7.4, bold=True)

    display = scores.sort_values(["chronological_age", "donor_id"]).reset_index(drop=True)
    for _, row in display.iterrows():
        cells = table.add_row().cells
        values = [
            row["donor_id"],
            row["sex"],
            f"{row['chronological_age']:.0f}",
            f"{row['elasticnet_OOF_predicted_age']:.1f}",
            f"{row['thymic_age_deviation_years']:+.1f}",
            f"{row['internal_thymic_biological_age']:.1f}",
            f"{row['deviation_percentile_within_18_donors']:.1f}",
        ]
        for j, (cell, val) in enumerate(zip(cells, values)):
            set_cell_width(cell, cols[j][1])
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            cell.text = str(val)
            for pp in cell.paragraphs:
                pp.alignment = WD_ALIGN_PARAGRAPH.LEFT if j in (0, 1) else WD_ALIGN_PARAGRAPH.CENTER
                pp.paragraph_format.space_after = Pt(0)
                pp.paragraph_format.space_before = Pt(0)
                for rr in pp.runs:
                    set_font(rr, size=7.3)
        prevent_row_split(table.rows[-1])

    note = discussion.insert_paragraph_before()
    note.style = doc.styles["Normal"]
    note.paragraph_format.space_before = Pt(4)
    note.paragraph_format.space_after = Pt(6)
    note.paragraph_format.keep_together = True
    note_text = (
        "Note: OOF, out-of-fold. The primary prediction is from the fully nested Elastic Net model. For donor i, the "
        "age-adjusted deviation is the OOF predicted age minus the expected OOF prediction at that chronological age, "
        "where the calibration line is fitted using the other 17 donors. Internal thymic biological age equals "
        "chronological age plus the age-adjusted deviation. Percentiles are empirical ranks within these 18 donors. "
        "These scores are exploratory and internally cross-validated; they are not clinical measurements or externally "
        "validated biological-age estimates."
    )
    r = note.add_run(note_text)
    set_font(r, size=8.2, italic=True)


def add_methods(doc):
    anchor = next(p for p in doc.paragraphs if p.text.strip() == "Exploratory extreme-age classification")

    h = anchor.insert_paragraph_before("Construction of the donor-resolved human thymic aging atlas")
    h.style = doc.styles["Heading 2"]
    p = anchor.insert_paragraph_before()
    p.style = doc.styles["Normal"]
    r = p.add_run(
        "The atlas was constructed as a traceable source-data packet rather than as a new statistical test. It combines "
        "the sex-adjusted donor-level composition model, whole-thymus and cell-type-specific sex-adjusted pseudobulk age "
        "effects, pathway-enrichment outputs, GSE195812 developmental-state localization, the frozen donor manifest, and "
        "the internal donor score. Each module retains its source-file path and SHA-256 digest in the atlas source manifest. "
        "The independent unit remains the human donor. Developmental localization is explicitly separated from aging "
        "replication, and modules with different donor coverage are not ranked solely by their number of significant features."
    )
    set_font(r, size=10.5)

    h = anchor.insert_paragraph_before("Exploratory internal thymic aging score")
    h.style = doc.styles["Heading 2"]
    p = anchor.insert_paragraph_before()
    p.style = doc.styles["Normal"]
    r = p.add_run(
        "The score uses only the fully nested Elastic Net out-of-fold prediction. For each donor i, a calibration line "
        "relating OOF predicted age to chronological age was fitted on the other 17 donors, yielding the expected prediction "
        "at that donor's chronological age. The age-adjusted thymic age deviation was defined as:"
    )
    set_font(r, size=10.5)
    eq = anchor.insert_paragraph_before("D_i = Ahat_i^(OOF) - mhat_(-i)(A_i)    (7)")
    eq.style = doc.styles["Normal"]
    eq.alignment = WD_ALIGN_PARAGRAPH.CENTER
    eq.paragraph_format.keep_together = True
    for r in eq.runs:
        set_font(r, name="Cambria Math", size=10.5)
    p = anchor.insert_paragraph_before()
    p.style = doc.styles["Normal"]
    r = p.add_run(
        "where A_i is chronological age, Ahat_i^(OOF) is the held-out Elastic Net prediction, and mhat_(-i)(A_i) is the "
        "expected OOF prediction at age A_i from the calibration fitted without donor i. The internal thymic biological "
        "age and empirical within-cohort percentile were defined as:"
    )
    set_font(r, size=10.5)
    eq = anchor.insert_paragraph_before("B_i = A_i + D_i    (8)")
    eq.style = doc.styles["Normal"]
    eq.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for r in eq.runs:
        set_font(r, name="Cambria Math", size=10.5)
    eq = anchor.insert_paragraph_before("S_i = 100 x rank(D_i) / 18    (9)")
    eq.style = doc.styles["Normal"]
    eq.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for r in eq.runs:
        set_font(r, name="Cambria Math", size=10.5)
    p = anchor.insert_paragraph_before()
    p.style = doc.styles["Normal"]
    r = p.add_run(
        "Here B_i is an internally derived age coordinate and S_i is the percentile rank of D_i among the 18 donors. "
        "The score was not trained or calibrated against thymic function, health outcomes, morbidity, mortality, or an "
        "independent human cohort. It is therefore reported as an internally cross-validated exploratory score rather than "
        "a clinical biological-age clock. Strict-OOF HumanThymusFormer predictions and signed Integrated Gradients were "
        "retained only as secondary model-comparison and attribution fields; they do not define the primary score."
    )
    set_font(r, size=10.5)


def revise_final_prediction_results(doc, scores, donor_manifest, metrics, paired_summary, influence):
    en = metrics.loc[metrics["model"].eq("ElasticNet")].iloc[0]
    null = metrics.loc[metrics["model"].eq("MeanAgeNull")].iloc[0]
    htf = metrics.loc[metrics["model"].eq("HumanThymusFormer_strict_OOF")].iloc[0]
    ps = paired_summary.iloc[0]
    adult = scores.loc[scores["chronological_age"].ge(18)].copy()
    adult_m = regression_metrics(adult["chronological_age"], adult["elasticnet_OOF_predicted_age"])
    female = scores.loc[scores["sex"].eq("Female")].copy()
    female_m = regression_metrics(female["chronological_age"], female["elasticnet_OOF_predicted_age"])
    most_negative = influence.sort_values("delta_R2_vs_full").iloc[0]
    most_positive = influence.sort_values("delta_R2_vs_full").iloc[-1]

    p30 = next(p for p in doc.paragraphs if p.text.startswith("Elastic Net produced the highest final"))
    set_paragraph_text(
        p30,
        f"The final 18-row held-out table gave Elastic Net R²={en.R2:.3f}, MAE={en.MAE:.2f} years, "
        f"RMSE={en.RMSE:.2f} years and Spearman ρ={en.Spearman_rho:.3f}. The donor-bootstrap 95% intervals "
        "were 0.399–0.792 for R², 6.17–12.59 years for MAE and 8.34–14.31 years for RMSE. The outer-fold "
        f"training-mean null gave R²={null.R2:.3f} and MAE={null.MAE:.2f} years; paired ΔMAE was "
        f"{ps.delta_MAE_model_minus_null:.2f} years (95% CI, {ps.ci_low:.2f} to {ps.ci_high:.2f}). "
        "All numbers in Figure 3 were recomputed directly from the frozen 18-row fully nested out-of-fold table. "
        "The source table available locally does not contain the final held-out predictions for every alternative model "
        "family, so obsolete cross-model and ablation panels were removed rather than reconstructed from incompatible "
        "outputs. These intervals condition on the fixed OOF pairs and do not estimate full pipeline-refitting uncertainty."
    )

    p31 = next(p for p in doc.paragraphs if p.text.startswith("The final pipeline recorded"))
    set_paragraph_text(
        p31,
        f"A fixed-prediction donor-influence audit showed that the pooled OOF R² changed most after excluding "
        f"{most_negative.excluded_donor} (ΔR²={most_negative.delta_R2_vs_full:+.3f}) and "
        f"{most_positive.excluded_donor} (ΔR²={most_positive.delta_R2_vs_full:+.3f}); this jackknife does not refit "
        f"feature selection or model parameters. Restricting evaluation of the existing OOF predictions to adults aged "
        f"≥18 years (n={len(adult)}) gave R²={adult_m['R2']:.3f}, MAE={adult_m['MAE']:.2f} years and "
        f"Spearman ρ={adult_m['Spearman_rho']:.3f}; this is not an independently trained adult-only model. Among women "
        f"(n={len(female)}), the corresponding fixed-prediction values were R²={female_m['R2']:.3f} and "
        f"MAE={female_m['MAE']:.2f} years. The strict fold-internal HumanThymusFormer table gave "
        f"R²={htf.R2:.3f} and MAE={htf.MAE:.2f} years. These sensitivity summaries narrow the claims but cannot "
        "resolve unmeasured clinical or batch confounding or replace independent adult validation."
    )

    caption = replace_picture_before_caption(
        doc, "Figure 3.", FIGURE3_PNG, 6.55,
        "Final fully nested age-prediction analysis",
        "Four-panel figure showing the fully nested leave-one-donor-out workflow, final 18 held-out Elastic Net predictions, paired donor errors versus a training-mean null, and donor influence on fixed out-of-fold predictions.",
    )
    set_paragraph_text(
        caption,
        "Figure 3. Final fully nested donor-level age-prediction analysis reconstructed from the frozen 18-row out-of-fold table. "
        "(A) In each outer leave-one-donor-out fold, all screening, preprocessing and hyperparameter selection occurred "
        "within the 17 training donors before the held-out donor was predicted once. (B) Chronological age versus final "
        "Elastic Net OOF prediction for all 18 donors. (C) Paired absolute errors for Elastic Net and the outer-fold "
        "training-mean null; the interval is a donor bootstrap over fixed OOF pairs. (D) Change in pooled OOF R² after "
        "removing each donor from the already generated prediction pairs. Panel D is an influence diagnostic, not a "
        "pipeline-refitted leave-one-donor sensitivity analysis. Obsolete panels from the preceding diagnostic pipeline "
        "were withdrawn. No independent adult cohort was used; these data do not establish a validated age clock.",
        size=9,
    )
    caption.style = doc.styles["Figure Caption"]

    supp_caption = replace_picture_before_caption(
        doc, "Supplementary Figure S2.", SUPP_ML_PNG, 6.55,
        "Final prediction and confounding diagnostics",
        "Six-panel supplement comparing final Elastic Net and strict OOF Transformer predictions, bootstrap uncertainty, age-bias calibration, per-donor errors, donor influence, and the cohort age-sex structure.",
    )
    set_paragraph_text(
        supp_caption,
        "Supplementary Figure S2. Final prediction, score-calibration and confounding diagnostics based on the frozen "
        "18-donor tables. (A) Final Elastic Net and strict fold-internal HumanThymusFormer OOF predictions. "
        "(B) Donor-bootstrap intervals for Elastic Net R², MAE and RMSE (2,000 resamples of fixed OOF pairs). "
        "(C) Raw prediction gap and leave-one-donor age-calibrated deviation. (D) Per-donor absolute errors for Elastic "
        "Net, strict OOF HumanThymusFormer and the training-mean null. (E) Change in pooled MAE after omitting each fixed "
        "OOF pair; models were not refitted. (F) Age–sex structure of the cohort. Adult-restricted and female-only values "
        "reported in the text evaluate subsets of the existing held-out predictions and are not independently trained "
        "sensitivity models. Clinical indication and technical batch could not be tested because those fields were "
        "unavailable in the public metadata.",
        size=9,
    )
    supp_caption.style = doc.styles["Figure Caption"]


def revise_confounding_and_cell_intrinsic_text(doc, donor_manifest, genes, composition):
    female = donor_manifest.loc[donor_manifest["sex"].eq("Female"), "age_years"]
    male = donor_manifest.loc[donor_manifest["sex"].eq("Male"), "age_years"]
    p17 = next(p for p in doc.paragraphs if p.text.startswith("The study was organized around"))
    set_paragraph_text(
        p17,
        "The study was organized around a donor-level aging framework. GSE231906 served as the primary human discovery "
        "cohort, and all 18 eligible thymus donors aged 4–69 years were retained. The frozen donor audit contains one "
        "combined CD45-positive/CD45-negative library per donor, 114,957 post-QC cells, no technical replicate library, "
        "and no donor excluded from the primary continuous-age analysis. Fourteen donors were female and four were male. "
        "Female ages ranged from 4 to 64 years (median 35; mean 33.2), and male ages from 9 to 69 years (median 37.5; "
        "mean 38.3). Thus sex was not perfectly confounded with age, but four males are insufficient for a reassuring "
        "sex-stratified test. Diagnosis, surgical indication, procurement site and method, processing time, library batch "
        "and sequencing batch were unavailable for every donor in the current public metadata and are marked explicitly "
        "as missing in Supplementary Table S1; batch–age relations therefore cannot be tested rather than being assumed absent."
    )

    p24 = next(p for p in doc.paragraphs if p.text.startswith("The cell-type overview suggested"))
    set_paragraph_text(
        p24,
        "Whole-thymus pseudobulk is a mixture-level analysis: its coefficients can reflect both cell-intrinsic expression "
        "change and altered cellular composition. We therefore froze separate sex-adjusted donor-pseudobulk effects for "
        "ten annotated cell contexts and retained the 95% intervals, raw P values, BH q values and donor coverage for each "
        "gene. Coverage ranged from 10 donors for NKT-like cells to 18 donors for DP, SP CD4, SP CD8 and Treg contexts "
        "(Figure 6B; full source tables). The strongest counts of context-specific BH-significant effects occurred in "
        "SP CD4 cells, whereas TEC analyses used 12 donors and fibroblast analyses 13 donors. These results permit "
        "cell-type-resolved association statements but not a blanket claim of cell-intrinsic aging: dissociation, "
        "state abundance within broad labels and residual composition can still contribute. The separate centered "
        "log-ratio model reports relative composition effects with HC3 intervals, and the joint Dirichlet bootstrap is "
        "retained as a composition sensitivity analysis; neither estimates absolute cell numbers."
    )

    p63 = next(p for p in doc.paragraphs if p.text.startswith("Donor ID, GSM/library identity"))
    set_paragraph_text(
        p63,
        "The frozen donor manifest records donor ID, GSM/library identity, age, sex, platform, sample preparation, "
        "pre- and post-QC cell counts, replicate status, primary-analysis inclusion and exclusion reason. Every donor had "
        "one GPL24676 library generated from combined CD45-positive and CD45-negative material in a stated 6:4 design, "
        "and all 18 entered the primary analysis. Diagnosis, surgical collection indication, health/comorbidity, "
        "procurement site and method, processing time, library batch and sequencing batch were absent from the current "
        "public records and are encoded as unavailable rather than inferred. We summarized age by sex, evaluated adult "
        "(≥18 years) and female-only subsets of the fixed OOF predictions, and calculated per-donor influence on pooled "
        "OOF metrics. These diagnostics do not substitute for pipeline-refitted subgroup models. No age–batch analysis "
        "was possible because no batch variable was available."
    )

    p80 = next(p for p in doc.paragraphs if p.text.startswith("where Y(g,d) is the modeled"))
    extra = (
        " Cell-type-specific pseudobulk models used the same age-and-sex design when a context had adequate donor coverage; "
        "each released row retains n_donors, coefficient, 95% interval, P value and BH q value. Whole-thymus and cell-type "
        "effects are reported as distinct estimands. A restricted 18–64-year expression sensitivity (n=13) gave effect "
        "correlation 0.881 and 86.8% sign agreement with the primary whole-thymus analysis; female-only analysis (n=14) "
        "gave correlation 0.955 and 91.7% sign agreement. These are stability summaries, not proof that sex or unmeasured "
        "batch/clinical factors are controlled."
    )
    if "restricted 18–64-year expression sensitivity" not in p80.text:
        set_paragraph_text(p80, p80.text + extra)

    p53 = next(p for p in doc.paragraphs if p.text.startswith("The observed decrease in DP and TEC fractions"))
    set_paragraph_text(
        p53,
        "The observed composition shifts are relative within the measured cell mixture and cannot be equated with absolute "
        "cell-number changes. The frozen donor manifest resolves age, sex, GSM identity, platform, preparation, library "
        "count, pre/post-QC cells and inclusion status, but it also establishes that diagnosis, surgical indication, "
        "procurement, processing and batch are unavailable for all 18 donors. This is an identified limitation, not a "
        "completed confounding adjustment. Sex adjustment, female-only stability and fixed-prediction donor-influence "
        "analyses cannot exclude confounding in a cohort with 14 women and four men. Experimental work implicates epithelial "
        "lamin B1 and FOXN1 in thymic maintenance and regeneration [40–42], and other work links involution with negative "
        "selection and central tolerance [43,44]; those mechanisms remain hypotheses rather than direct measurements here."
    )

    p57 = next(p for p in doc.paragraphs if p.text.startswith("Several limitations are central"))
    set_paragraph_text(
        p57,
        "Several limitations are central. First, only 18 independent donors were available, with 14 women and four men. "
        "Second, diagnosis, operative indication, procurement, processing and batch were unavailable for every donor, so "
        "age–batch and clinical-confounding analyses could not be performed. Third, adult-restricted and female-only "
        "prediction summaries use subsets of already generated OOF predictions rather than independently trained models. "
        "Fourth, whole-thymus effects are mixture-level; cell-context pseudobulk and composition models reduce ambiguity but "
        "do not prove cell-intrinsic mechanisms or absolute count changes. Fifth, the internal model lacks an independent "
        "adult human validation cohort. Sixth, Figure 5 has frozen endpoint source tables but the complete raw-to-figure "
        "code chain is not present locally. Finally, mouse libraries are pooled or cumulative and the legacy comparison is "
        "a single pair, precluding animal-level replication or conserved-causality claims."
    )

    p58 = next(p for p in doc.paragraphs if p.text.startswith("Overall, the study is best interpreted"))
    set_paragraph_text(
        p58,
        "Overall, this is an 18-donor public-data reanalysis centered on continuous-age human associations, internally "
        "cross-validated prediction and explicitly bounded external context. Figure 3 now reports the final frozen OOF "
        "table, and the donor manifest, cell-context effects, composition model outputs, Figure 5 evidence units and "
        "exploratory score are provided as auditable source tables. The remaining requirements for a stronger translational "
        "claim are recovery of missing clinical/batch metadata, release of the complete Figure 5 code chain, and validation "
        "of the frozen model in an independent adult human thymus cohort. Until then, 'age clock' and clinical biomarker "
        "language is not warranted."
    )


def revise_external_evidence(doc):
    p47 = next(p for p in doc.paragraphs if p.text.startswith("The current integrated summary highlights"))
    set_paragraph_text(
        p47,
        "The endpoint Figure 5 source packet contains the context-level concordance table, effect-direction heatmap and "
        "gene-level RNA–ATAC panel tables. The evidence comes from only two independent study families: GSE132136 and "
        "GSE223050. GSE223049 RNA-seq and GSE221034 ATAC-seq are companion modalities within GSE223050 and therefore are "
        "not counted as two independent replications. GSE132136 contributes 5 untreated young and 4 untreated aged "
        "profiles per compartment. In GSE223050, the RNA contrasts contain 5 young and 5 aged TEC pooled libraries and "
        "4 young and 4 aged fibroblast pooled libraries; the ATAC contrasts contain 4 young and 3 aged TEC libraries and "
        "4 young and 4 aged fibroblast libraries. Each RNA library pools five mice, and the published study describes "
        "cumulative signatures from 15–25 animals per cell type/condition; the analytical unit is nevertheless the pool "
        "or library. Candidate labels are now descriptive evidence classes and never convert support-record counts into "
        "numbers of independent studies. Public mouse data support direction only and do not validate the human score."
    )
    caption = replace_picture_before_caption(
        doc, "Figure 5.", EXT_FIGURE, 6.55,
        "Cross-dataset support for human thymic aging programs",
        "Four-panel figure showing context-level human–mouse direction concordance, a direction heatmap, and RNA–ATAC agreement for thymic epithelial cells and fibroblasts.",
    )
    set_paragraph_text(
        caption,
        "Figure 5. Cross-dataset directional support for human thymic aging programs. (A) Directional concordance between "
        "sex-adjusted human whole-thymus βage and mouse effects among genes with q<0.05 in both datasets; denominators are "
        "genes, not animals or studies. The 100% cortical-tissue value is 4/4. (B) Direction of selected effects across "
        "human and mouse contexts; black points denote q<0.05. (C, D) RNA Aged−Young logFC versus nearest-TSS ATAC logFC "
        "for thymic epithelial cells and fibroblasts within GSE223050. GSE132136 and GSE223050 are the only two independent "
        "study families; GSE223049 and GSE221034 are companion modalities, and their pool/library observations are not "
        "independent animal replicates. Endpoint panel source tables are frozen, but the complete raw-download-to-figure "
        "reconstruction script is not present in the delivered local code bundle and must be released before submission. "
        "Accordingly, the figure is cross-dataset directional support, not independent replication or model validation.",
        size=9,
    )
    caption.style = doc.styles["Figure Caption"]

    p99 = next(p for p in doc.paragraphs if p.text.startswith("For GSE221034"))
    set_paragraph_text(
        p99,
        p99.text + " The released endpoint source tables identify every displayed denominator. Because the local delivery "
        "lacks the complete raw-data download, preprocessing, effect-estimation, ortholog-mapping and plotting chain, "
        "Figure 5 is retained only with an explicit reproducibility limitation; any stronger conserved-program or "
        "independent-replication claim has been removed."
    )

    p101 = next(p for p in doc.paragraphs if p.text.startswith("Candidate summaries integrate"))
    set_paragraph_text(
        p101,
        "Candidate summaries integrate sex-adjusted human βage, fully nested machine-learning membership, post-hoc deep-"
        "learning attribution, developmental localization and public mouse directional evidence. Evidence classes are "
        "assigned after collapsing records by study family, modality and cell context. Multiple rows from the same GEO "
        "study, multiple contexts, multiple genes or the RNA and ATAC components of GSE223050 never increase the number "
        "of independent studies. The maximum independent external-study count in the current package is two. Human "
        "donor-level continuous-age analyses remain primary; no candidate is a clinically validated biomarker or conserved "
        "causal regulator."
    )


def add_novelty_comparison(doc):
    discussion = next(p for p in doc.paragraphs if p.text.strip() == "DISCUSSION")
    h = discussion.insert_paragraph_before("What this reanalysis adds beyond the source atlas and prior human thymus studies")
    h.style = doc.styles["Heading 2"]
    p = discussion.insert_paragraph_before()
    set_font(p.add_run(
        "Deng et al. generated the GSE231906 atlas and already reported age-associated thymic and peripheral T-cell "
        "changes [26]. Yang et al. integrated 350,678 cells from 36 human thymus samples, described age-associated "
        "cellular programs and experimentally examined IGFBP5 [66]. The present manuscript does not claim discovery of "
        "thymic aging or construction of the first human thymus atlas. Its narrower addition is a donor-as-unit "
        "continuous-age reanalysis of GSE231906 with sex-adjusted pseudobulk effects, explicit composition models, "
        "cell-context effect tables, a fully nested 18-donor prediction audit and a traceable internal score table. "
        "These are analytical extensions of public data, not a new cohort or experimental validation."
    ), size=10.5)
    p = discussion.insert_paragraph_before()
    set_font(p.add_run(
        "The age-prediction result is therefore described as internal proof of concept. Validation of selected genes, "
        "pathways or mouse directions does not validate the frozen predictive mapping. A defensible 'age clock' claim "
        "would require the prespecified frozen preprocessing and Elastic Net model to be applied without refitting to an "
        "independent adult human thymus cohort with known clinical and technical metadata. No such cohort is analyzed here."
    ), size=10.5)

    rows = [
        ["Cohort/data", "Original GSE231906 discovery resource; human thymus and peripheral blood", "Integrated 36 human thymus samples (350,678 cells)", "Reanalysis of the same 18 eligible GSE231906 thymus donors; no new cohort"],
        ["Primary contribution", "Single-cell atlas of T-cell development and aging; immune-age analyses", "Human thymus transcriptional atlas; age-related programs; IGFBP5 focus", "Donor-unit continuous-age statistics, composition models, cell-context effect atlas and audit trail"],
        ["Experimental validation", "Source-study analyses and experiments", "Gene/cell findings supported experimentally", "None newly performed; public-data computational reanalysis only"],
        ["Prediction", "Naive T-cell immune-age model reported in source study", "No independent validation of this manuscript's score", "Fully nested 18-donor internal OOF evaluation; exploratory within-cohort score"],
        ["Independent adult validation", "Not a validation cohort for this reanalysis", "Not a validation cohort for this reanalysis", "Absent; therefore no validated thymic age-clock claim"],
        ["Claim boundary", "Original biological discovery", "Independent prior biological and experimental evidence", "Incremental analytical framework and frozen donor-level source tables"],
    ]
    add_compact_table_before(
        discussion,
        "Table 2. Itemized positioning of the present reanalysis relative to Deng et al. and Yang et al.",
        ["Dimension", "Deng et al. 2025 [26]", "Yang et al. 2024 [66]", "Present manuscript"],
        rows,
        note_text=(
            "The table distinguishes dataset-generating and experimental novelty from analytical reanalysis. The present "
            "study's potentially useful addition is donor-level continuous-age inference and auditability; it does not "
            "supersede the original atlas or validate a clinical clock."
        ),
        font_size=6.6,
        widths=[1.0, 1.75, 1.75, 2.1],
        page_break=True,
    )


def add_supplementary_tables(doc, donor_manifest):
    supp_heading = next(p for p in doc.paragraphs if p.text.startswith("Supplementary interpretation note."))
    rows = []
    unavailable = "NA—public metadata unavailable"
    for _, r in donor_manifest.sort_values(["age_years", "donor_id"]).iterrows():
        rows.append([
            r["donor_id"], r["gsm_ids"], f"{r['age_years']:.0f}", r["sex"],
            f"{int(r['total_cells_postQC']):,}", unavailable, unavailable, unavailable,
            "Included; no replicate library",
        ])
    add_compact_table_before(
        supp_heading,
        "Supplementary Table S1. Frozen donor-level metadata, missingness and primary-analysis inclusion",
        ["Donor", "GSM", "Age, y", "Sex", "Post-QC cells", "Sample source / procurement", "Surgical indication / diagnosis", "Library / sequencing batch", "Primary status"],
        rows,
        note_text=(
            "All donors were profiled on GPL24676 from combined CD45-positive/CD45-negative material (6:4 design), with "
            "one thymus library per donor. Procurement site/method, processing time, diagnosis, health/comorbidity, surgical "
            "indication, library batch and sequencing batch were unavailable in the current public metadata. No batch-age "
            "test can therefore be performed. 'Included' refers to the primary continuous-age analysis."
        ),
        font_size=5.9,
        widths=[0.48, 0.70, 0.45, 0.45, 0.62, 1.05, 1.05, 1.00, 0.90],
        page_break=True,
    )

    external_rows = [
        ["GSE132136", "Microarray RNA", "Cortical/medullary lymphocyte and tissue", "5", "4", "Profile/sample", "Untreated baseline only; 120 post-castration profiles excluded", "1"],
        ["GSE223050: GSE223049", "RNA-seq", "Thymic epithelial", "5", "5", "Pooled library", "Five mice pooled per library", "1"],
        ["GSE223050: GSE223049", "RNA-seq", "Thymic fibroblast", "4", "4", "Pooled library", "Five mice pooled per library", "1"],
        ["GSE223050: GSE221034", "Omni-ATAC", "Thymic epithelial", "4", "3", "Library/pool", "Companion modality; cumulative signatures from 15–25 mice/condition", "0 (same study family)"],
        ["GSE223050: GSE221034", "Omni-ATAC", "Thymic fibroblast", "4", "4", "Library/pool", "Companion modality; cumulative signatures from 15–25 mice/condition", "0 (same study family)"],
    ]
    add_compact_table_before(
        supp_heading,
        "Supplementary Table S2. Independent units and study-family counting for Figure 5",
        ["Study family / accession", "Assay", "Context", "Young units", "Aged units", "Analytical unit", "Pooling/exclusion note", "Independent study contribution"],
        external_rows,
        note_text=(
            "Figure 5 uses two independent study families in total: GSE132136 and GSE223050. GSE223049 and GSE221034 "
            "are companion RNA and ATAC datasets within GSE223050 and are not independent replications. Gene denominators "
            "and support records must not be reinterpreted as animal counts or study counts."
        ),
        font_size=6.2,
        widths=[1.05, 0.62, 0.92, 0.52, 0.52, 0.72, 1.43, 0.92],
        page_break=True,
    )


def add_references_and_transparency(doc):
    refs = next(p for p in doc.paragraphs if p.text.strip() == "SUPPLEMENTARY MATERIALS")
    for text in [
        "66. Xiaoqing Yang, Xiaoyang Chen, Wei Wang, et al. Transcriptional profile of human thymus reveals IGFBP5 is correlated with age-related thymic involution. Frontiers in Immunology. 2024;15:1322214. doi:10.3389/fimmu.2024.1322214.",
        "67. Tarek Kassis, Vishal Agarwal, Yifan He, Dhruv Patel, Anna M. Brueckner. Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents. arXiv. 2026;arXiv:2609.00065. doi:10.48550/arXiv.2609.00065.",
    ]:
        p = refs.insert_paragraph_before()
        set_font(p.add_run(text), size=10.5)

    code_h = next(p for p in doc.paragraphs if p.text.strip() == "CODE AVAILABILITY")
    code_p = Paragraph(code_h._p.getnext(), code_h._parent)
    set_paragraph_text(
        code_p,
        "The supplied workspace includes frozen figure-level source tables for the final 18-row prediction audit, the "
        "human thymic aging atlas and the displayed Figure 5 panels. The delivered local code bundle still lacks the "
        "complete raw-public-mouse-download-to-Figure-5 reconstruction chain; this limitation is stated in the Results, "
        "Methods and legend, and claims that depend on untraceable intermediate steps were narrowed. A submission release "
        "should include exact commits, environments, random seeds, raw input manifests, ortholog mapping tables and every "
        "figure-generation script. Scientific Agent Skills informed the evidence-provenance, reproducibility and claim-"
        "boundary audit during revision [67]; the authors retain responsibility for all analyses and final wording."
    )

def revise_existing_text(doc):
    # Abstract: add atlas and score without changing the evidence hierarchy.
    p = doc.paragraphs[7]
    prefix = (
        "The frozen outputs were additionally assembled into a donor-resolved continuous-age atlas and an exploratory "
        "age-adjusted donor score based on fully nested Elastic Net out-of-fold predictions. The score is a within-cohort "
        "coordinate and is not clinically validated. "
    )
    if not p.text.startswith(prefix):
        old = p.text
        for run in p.runs:
            run.text = ""
        r = p.runs[0] if p.runs else p.add_run()
        r.text = prefix + old
        set_font(r, size=10.5)

    # Introduction objective.
    intro = doc.paragraphs[14]
    replace_text(
        intro,
        "how continuous age relates to pseudobulk expression and cell composition, and what age information remains under fully nested internal prediction.",
        "how continuous age relates to pseudobulk expression and cell composition, what age information remains under fully nested internal prediction, and how these outputs can be organized into a donor-resolved atlas and an explicitly exploratory internal score.",
    )
    if "Yang et al." not in intro.text:
        set_paragraph_text(
            intro,
            intro.text + " Deng et al. generated and analyzed the source aging atlas [26], while Yang et al. independently "
            "reported human thymus single-cell aging programs and experimental IGFBP5 evidence [66]. Our intended novelty "
            "is therefore donor-unit continuous-age inference and auditability, not first description of thymic aging."
        )

    # Discussion: insert a dedicated interpretation paragraph after the ML paragraph.
    anchor = next(p for p in doc.paragraphs if p.text.startswith("HumanThymusFormer provides a complementary"))
    p = anchor.insert_paragraph_before()
    p.style = doc.styles["Normal"]
    r = p.add_run(
        "The donor-resolved atlas provides a common coordinate system for composition, expression, pathway, developmental-"
        "context, and prediction outputs. Its principal value is organizational and hypothesis-generating: it preserves "
        "the donor as the independent unit and makes module-specific evidence limits visible. The age-adjusted score removes "
        "the cohort-level tendency of prediction error to vary with chronological age using leave-one-donor-out calibration, "
        "but it does not remove unmeasured clinical or technical confounding. The score table should therefore be used to "
        "select donors or molecular programs for follow-up, not to classify individual health or prognosis."
    )
    set_font(r, size=10.5)

    # Data availability: report the generated packet without claiming public deposition.
    data_heading = next(p for p in doc.paragraphs if p.text.strip() == "DATA AVAILABILITY")
    data_p = data_heading._p.getnext()
    from docx.text.paragraph import Paragraph
    data_para = Paragraph(data_p, data_heading._parent)
    addition = (
        " The analysis workspace also contains a frozen human thymic aging atlas packet comprising gene-age effects, "
        "composition effects, pathway records, developmental-context records, the donor manifest, the 18-donor score "
        "table, and a SHA-256 source manifest. These derived files should be deposited with the final versioned code release; "
        "their presence in the local analysis workspace does not by itself constitute public availability."
    )
    if "frozen human thymic aging atlas packet" not in data_para.text:
        old = data_para.text
        for run in data_para.runs:
            run.text = ""
        r = data_para.runs[0] if data_para.runs else data_para.add_run()
        r.text = old + addition
        set_font(r, size=10.5)


def normalize_equation_placeholders(doc):
    replacements = {
        "C(d,g) = Σ(i∈d) c(i,g) (1)": "C_(d,g)=∑_(i∈d)c_(i,g)    (1)",
        "C(d,c,g) = Σ[i∈(d,c)] c(i,g) (2)": "C_(d,c,g)=∑_(i∈(d,c))c_(i,g)    (2)",
        "P(d,c) = N(d,c) / N(d,all) (3)": "P_(d,c)=N_(d,c)/N_(d,all)    (3)",
        "Y(g,d) = β0(g) + βage(g)·AgeZ(d) + βsex(g)·Sex(d) + ε(g,d) (4)": "Y_(g,d)=β_0(g)+β_age(g)·Age_Z(d)+β_sex(g)·Sex(d)+ε_(g,d)    (4)",
        "Attention(Q,K,V) = softmax(QKᵀ / √d_k)V (5)": "Attention(Q,K,V)=softmax(QK^T/√(d_k))V    (5)",
        "IG_i(x) = (x_i−x'_i) ∫₀¹ [∂F(x'+α(x−x'))/∂x_i] dα (6)": "IG_i(x)=(x_i−x'_i)∫_0^1 (∂F(x'+α(x−x')))/(∂x_i) dα    (6)",
    }
    for p in doc.paragraphs:
        key = " ".join(p.text.split())
        if key in replacements:
            for run in p.runs:
                run.text = ""
            r = p.runs[0] if p.runs else p.add_run()
            r.text = replacements[key]
            set_font(r, name="Cambria Math", size=10.5)
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.keep_together = True


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    composition, genes, developmental, pathways, scores, donor_manifest = load_data()
    assert len(scores) == 18 and scores["donor_id"].nunique() == 18
    assert set(scores["score_scope"]) == {"internal proof-of-concept; not clinically validated"}
    assert len(composition) == 19
    assert {"whole_thymus", "TEC", "Fibroblast"}.issubset(set(genes["cell_context"]))
    assert EXT_FIGURE.exists()
    build_atlas_figure_v2(composition, genes, developmental, scores)
    frame, metrics, bootstrap, paired_summary, influence = build_final_ml_figures(scores, donor_manifest)

    doc = Document(SOURCE)
    revise_existing_text(doc)
    revise_final_prediction_results(doc, scores, donor_manifest, metrics, paired_summary, influence)
    revise_confounding_and_cell_intrinsic_text(doc, donor_manifest, genes, composition)
    revise_external_evidence(doc)
    add_results_section(doc, composition, genes, developmental, pathways, scores)
    add_novelty_comparison(doc)
    add_methods(doc)
    add_supplementary_tables(doc, donor_manifest)
    add_references_and_transparency(doc)
    normalize_equation_placeholders(doc)

    # Preserve source typography while ensuring newly created styles use Times New Roman.
    for style_name, size in [("Normal", 10.5), ("Heading 1", 12.5), ("Heading 2", 11.5), ("Figure Caption", 9.0), ("Caption", 9.0)]:
        if style_name in doc.styles:
            style = doc.styles[style_name]
            style.font.name = "Times New Roman"
            style.font.size = Pt(size)
            style._element.get_or_add_rPr().rFonts.set(qn("w:ascii"), "Times New Roman")
            style._element.get_or_add_rPr().rFonts.set(qn("w:hAnsi"), "Times New Roman")
            style._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), "Times New Roman")

    # Record a compact provenance ledger alongside the working files.
    provenance = {
        "source_document": str(SOURCE),
        "output_document": str(OUTPUT),
        "figure": str(FIGURE_PNG),
        "score_table": str(ATLAS / "atlas_donor_scores.tsv"),
        "atlas_manifest": str(ATLAS / "atlas_source_manifest.tsv"),
        "atlas_contract": str(ATLAS / "atlas_contract.json"),
        "score_contract": str(SCORE_DIR / "thymic_aging_score_contract.json"),
        "figure3_final_source": str(FIGURE3_SOURCE),
        "figure5_endpoint_source": str(EXT_SOURCE),
        "figure5_reproducibility_limit": "endpoint source tables present; complete raw-to-figure script absent from local delivery",
        "n_donors": int(len(scores)),
        "scope": "internal proof-of-concept; not clinically validated",
    }
    (WORK / "revision_provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    doc.save(OUTPUT)
    print(OUTPUT)
    print(FIGURE_PNG)


if __name__ == "__main__":
    main()

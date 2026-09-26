"""Figure 5: descriptive cross-species direction agreement + evidence matrix.

All mouse summaries use raw counts only: the mouse design is 1 young / 1 aged
animal, so genes/peaks are NOT independent replicates — no binomial CI or
p-values are drawn. Evidence cells use four explicit states.

Panels C and D read ONE frozen assessable table
(08_mouse_validation/cross_species_gene_assessment.tsv) so their denominators
and numerators always agree (review 3.4).
"""
from __future__ import annotations

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from functools import lru_cache
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

from figure_common import *

EXPECTED_RNA = (160, 257)   # (concordant, assessable), frozen round-3 audit
EXPECTED_ATAC = (122, 256)


@lru_cache(maxsize=1)
def _cross_species_table():
    """Frozen single mother table (review 3.4): one row per human gene with
    RNA + ATAC direction layers, exclusion reasons, and OOF-IG fold counts.

    Rebuilt from the Step 18/19 frozen source tables once per run and exported
    to the *revision output*; the old project-side table is never overwritten.
    """
    rna = pd.read_csv(MOUSE / "18_mouse_human_orthologs.csv")
    atac = pd.read_csv(MOUSE / "19_scATAC_gene_level_join.csv")
    for col in ("assessable", "concordant"):
        rna[col] = rna[col].map(_bool)
    for col in ("atac_detectable", "atac_direction_ok"):
        atac[col] = atac[col].map(_bool)
    if rna.human_gene.duplicated().any() or atac.human_gene.duplicated().any():
        raise ValueError("Figure 5 mother-table join is not one row per human gene")
    # the ATAC join stores the mouse symbol in `feature`; align to `mouse_gene`
    atac = atac.rename(columns={"feature": "mouse_gene"})
    igf = DL / "strict_oof" / "oof_IG_gene_consistency.csv"
    ig = pd.read_csv(igf)[["gene", "n_folds_top50"]] \
        .rename(columns={"n_folds_top50": "oof_ig_fold_count"}) \
        if igf.exists() else pd.DataFrame(columns=["gene", "oof_ig_fold_count"])
    tab = (rna.drop(columns=["ATAC_adjusted_effect"], errors="ignore")
               .merge(atac[["human_gene", "mouse_gene", "ATAC_adjusted_effect",
                            "atac_detectable", "atac_direction_ok"]],
                      on=["human_gene", "mouse_gene"], how="left",
                      validate="one_to_one"))
    tab = tab.merge(ig, left_on="human_gene", right_on="gene", how="left")
    tab = tab.drop(columns=["gene"])
    tab["oof_ig_fold_count"] = pd.to_numeric(
        tab["oof_ig_fold_count"], errors="coerce")
    # RNA exclusion reasons (per row)
    tab["exclusion_reason_RNA"] = np.where(
        tab["assessable"].fillna(False), "",
        np.where(tab["WT_age_rho"].isna(), "no human age rho",
                 np.where(tab["RNA_adjusted_effect"].isna(),
                          "no mouse RNA effect", "zero/undirected RNA effect")))
    # ATAC layer: assessable iff detectable + directed + effect + human rho
    tab["atac_assessable"] = (
        tab["atac_detectable"].fillna(False)
        & tab["atac_direction_ok"].notna()
        & tab["ATAC_adjusted_effect"].notna()
        & tab["WT_age_rho"].notna())
    tab["atac_concordant"] = (
        tab["atac_assessable"] & tab["atac_direction_ok"].eq(True))
    tab["exclusion_reason_ATAC"] = np.where(
        tab["atac_assessable"], "",
        np.where(tab["atac_detectable"].fillna(False),
                 "ATAC effect present but undirected",
                 "ATAC not detectable"))
    tab["exclusion_reason"] = (
        tab["exclusion_reason_RNA"]
        + np.where(tab["exclusion_reason_ATAC"].eq(""), "",
                   "; " + tab["exclusion_reason_ATAC"]))
    # descriptive mouse cell counts (1 young vs 1 aged specimen; counts only)
    tab["mouse_total_count"] = (
        tab["RNA_young_cells"] + tab["RNA_age_cells"])
    tab["mouse_positive_count"] = np.round(
        tab["RNA_detected_Young"] * tab["RNA_young_cells"]
        + tab["RNA_detected_Age"] * tab["RNA_age_cells"]).astype("Int64")
    keep = ["human_gene", "mouse_gene", "WT_age_rho", "WT_rho_padj",
            "assessable", "concordant", "direction", "RNA_adjusted_effect",
            "RNA_effect_z", "exclusion_reason", "exclusion_reason_RNA",
            "exclusion_reason_ATAC", "mouse_total_count",
            "mouse_positive_count", "atac_assessable", "atac_concordant",
            "atac_direction_ok", "ATAC_adjusted_effect", "oof_ig_fold_count"]
    tab = tab[keep]
    _validate_mother_table(tab)
    tab.to_csv(FIGF / "fig5_cross_species_mother_audit.tsv", sep="\t", index=False)
    return tab


def _validate_mother_table(tab):
    required = {"human_gene", "mouse_gene", "WT_age_rho", "assessable",
                "concordant", "RNA_adjusted_effect", "atac_assessable",
                "atac_concordant", "ATAC_adjusted_effect"}
    missing = required - set(tab.columns)
    if missing:
        raise ValueError(f"Figure 5 mother table missing: {sorted(missing)}")
    if tab.human_gene.duplicated().any():
        raise ValueError("Figure 5 mother table has duplicate human_gene IDs")
    for modality, assessed_col, concord_col, expected in (
            ("RNA", "assessable", "concordant", EXPECTED_RNA),
            ("ATAC", "atac_assessable", "atac_concordant", EXPECTED_ATAC)):
        assessed = tab[assessed_col].eq(True)
        if tab.loc[assessed, concord_col].isna().any():
            raise ValueError(f"{modality}: assessable genes with missing direction")
        actual = (int(tab.loc[assessed, concord_col].eq(True).sum()),
                  int(assessed.sum()))
        if actual != expected:
            raise ValueError(f"Figure 5 {modality} counts {actual} differ from "
                             f"round-3 frozen audit {expected}; re-audit before plotting")
        print(f"[fig5] {modality}: {actual[1]} assessed, {actual[0]} concordant")


def _atac_assessable():
    """Single frozen assessable-gene table for BOTH scatter and counts panels.

    A gene is assessable iff it is atac_assessable in the frozen mother table
    (ATAC-detectable AND non-NA direction AND non-NA ATAC effect AND non-NA
    human rho). The scatter and the counts bars therefore always share the
    exact same denominator and numerator, and the 1 gene that is detectable
    but undirected is excluded identically in both panels.
    """
    t = _cross_species_table()
    a = t[t.atac_assessable == True].copy()  # noqa: E712
    a["dir_ok"] = a.atac_concordant.eq(True)
    return a


def _rna_scatter(ax):
    o = _cross_species_table()
    a = o[o.assessable == True]  # noqa: E712
    col = np.where(a.concordant.eq(True), CONC, "#E5C1C1")
    ax.scatter(a.WT_age_rho, a.RNA_adjusted_effect, s=6, c=col,
               alpha=0.7, linewidths=0, rasterized=True)
    ax.axhline(0, color=GREY, lw=0.6); ax.axvline(0, color=GREY, lw=0.6)
    k = int(a.concordant.eq(True).sum())
    n = len(a)
    ax.set_title(f"RNA effect directions: {k}/{n} concordant\n"
                 "descriptive; 1 young vs 1 aged mouse", fontsize=6.6)
    ax.set_xlabel("Human thymus age ρ", fontsize=6)
    ax.set_ylabel("Mouse RNA Δ (aged − young)", fontsize=6)
    ax.tick_params(labelsize=5.8)


def _rna_counts(ax):
    s = pd.read_csv(MOUSE / "18_cross_species_validation.csv").set_index("level")
    a = _cross_species_table()
    a = a[a.assessable.eq(True)]
    k, n = int(a.concordant.eq(True).sum()), len(a)
    if (k, n) != (int(s.loc["gene_main", "k"]), int(s.loc["gene_main", "n"])):
        raise ValueError("Figure 5B: RNA summary and mother-table counts disagree")
    kh, nh = int(s.loc["gene_highconf", "k"]), int(s.loc["gene_highconf", "n"])
    rows = [("all assessable", k, n),
            ("mouse-internal subset", kh, nh)]
    y = np.arange(len(rows))[::-1]
    for yi, (lab, kk, nn) in zip(y, rows):
        ax.barh(yi, nn, color=LGREY, height=0.55)
        ax.barh(yi, kk, color=CONC, height=0.55)
        ax.text(nn + 4, yi, f"{kk}/{nn} ({kk/nn:.0%})", va="center", fontsize=6)
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=6)
    ax.set_xlim(0, max(n, nh) * 1.35)
    ax.set_xlabel("Ortholog genes (counts only)", fontsize=6)
    ax.set_title("Cross-species RNA direction counts", fontsize=6.6)
    ax.tick_params(labelsize=5.8)


def _atac_scatter(ax):
    a = _atac_assessable()
    col = np.where(a.dir_ok, CONC, "#E5C1C1")
    ax.scatter(a.WT_age_rho, a.ATAC_adjusted_effect, s=6, c=col, alpha=0.7,
               linewidths=0, rasterized=True)
    ax.axhline(0, color=GREY, lw=0.6); ax.axvline(0, color=GREY, lw=0.6)
    k = int(a.dir_ok.sum())
    n = len(a)
    assert n == len(a.drop_duplicates(subset=["human_gene", "mouse_gene"]))
    ax.set_title(f"ATAC gene activity: {k}/{n} concordant\n"
                 "descriptive; 1 vs 1 mouse", fontsize=6.6)
    ax.set_xlabel("Human thymus age ρ", fontsize=6)
    ax.set_ylabel("Mouse ATAC Δ (aged − young)", fontsize=6)
    ax.tick_params(labelsize=5.8)


def _atac_counts(ax):
    a = _atac_assessable()
    kg, ng = int(a.dir_ok.sum()), len(a)
    pk = pd.read_csv(MOUSE / "19_scATAC_peak_validation.csv")
    pkm = pk[pk.peak_direction_ok.notna()]
    kp, npk = int((pkm.peak_direction_ok == True).sum()), len(pkm)
    kh = pkm[pkm.direction_concordant == True]
    kkh, nh = int((kh.peak_direction_ok == True).sum()), len(kh)
    rows = [("gene activity", kg, ng, ""),
            ("linked peaks", kp, npk, ""),
            ("mouse-subset peaks", kkh, nh, "subset")]
    y = np.arange(len(rows))[::-1]
    for yi, (lab, k, n, note) in zip(y, rows):
        ax.barh(yi, n, color=LGREY, height=0.55)
        ax.barh(yi, k, color=CONC, height=0.55)
        ax.text(n + 4, yi, f"{k}/{n} ({k/n:.0%}) {note}", va="center", fontsize=5.4)
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=5.8)
    ax.set_xlim(0, max(ng, npk, nh) * 1.45)
    ax.set_xlabel("Genes / peaks (counts only)", fontsize=6)
    ax.set_title("Mouse ATAC direction counts", fontsize=6.6)
    ax.tick_params(labelsize=5.8)


def _tf_attrition(ax):
    """Independent row-level summaries; never a mixed-unit attrition funnel."""
    net = pd.read_csv(MOUSE / "19_TF_motif_peak_target_validation.csv")
    for col in ("human_target_gene", "motif_direction_support", "full_chain_supported"):
        if col not in net:
            raise ValueError(f"Figure 5E network table lacks {col}")
    mapped = net.human_target_gene.notna()
    motif = net.motif_direction_support.map(_bool).eq(True)
    full = net.full_chain_supported.map(_bool).eq(True)
    summaries = [("all rows", len(net)),
                 ("human target", int(mapped.sum())),
                 ("motif support", int(motif.sum())),
                 ("full-chain support", int(full.sum()))]
    net.assign(has_human_target=mapped, has_motif_support=motif,
               has_full_chain_support=full).to_csv(
        FIGF / "fig5e_regulatory_row_audit.tsv", sep="\t", index=False)
    y = np.arange(len(summaries))[::-1]
    vals = [s[1] for s in summaries]
    colors = ["#9FB8D4", "#6F97C4", "#D55E00", "#999999"]
    ax.barh(y, vals, color=colors, height=0.6)
    for yi, (lab, v) in zip(y, summaries):
        ax.text(v + max(vals) * 0.02 + 0.3, yi, str(v), va="center", fontsize=6)
    ax.set_yticks(y); ax.set_yticklabels([s[0] for s in summaries], fontsize=5.6)
    ax.set_xlim(0, max(vals) * 1.18)
    ax.set_xlabel("Network rows (not a funnel)", fontsize=6)
    ax.set_title("Independent regulatory summaries", fontsize=6.6)
    ax.tick_params(labelsize=5.6)


def _bool(value):
    if pd.isna(value):
        return None
    if isinstance(value, str):
        value = value.strip().lower()
        if value in ("true", "1"):
            return True
        if value in ("false", "0"):
            return False
        raise ValueError(f"Ambiguous boolean in candidate table: {value!r}")
    return bool(value)


def _direction_state(assessed, concordant):
    if _bool(assessed) is not True:
        return "not_assessed"
    direction = _bool(concordant)
    if direction is None:
        raise ValueError("Assessed direction has no concordance flag")
    return "support" if direction else "discordant"


def _evidence_audit(genes):
    panel = pd.read_csv(RES / "final_candidate_panel.csv")
    if panel.gene.duplicated().any():
        raise ValueError("Figure 5F candidate panel contains duplicate gene IDs")
    p = panel.set_index("gene")
    mother = _cross_species_table().set_index("human_gene")
    igf = DL / "strict_oof" / "oof_IG_gene_consistency.csv"
    ig = pd.read_csv(igf).set_index("gene") if igf.exists() else pd.DataFrame()
    if len(ig) and not ig.index.is_unique:
        raise ValueError("Figure 5F OOF-IG source contains duplicate genes")
    rows = []
    for g in genes:
        if g not in p.index:
            raise ValueError(f"Figure 5F gene missing from candidate panel: {g}")
        r = p.loc[g]
        m = mother.loc[g] if g in mother.index else None
        rna = (_direction_state(m.assessable, m.concordant) if m is not None
               else "not_assessed")
        atac = (_direction_state(m.atac_assessable, m.atac_concordant)
                if m is not None else "not_assessed")
        peak = ("not_assessed" if _bool(r.peak_assessed) is not True else
                "support" if _bool(r.peak_support) is True else "no_support")
        dev = ("not_assessed" if _bool(r.human_external_detected) is not True else
               "support" if _bool(r.human_developmental_localized) is True
               else "no_support")
        rows.append({"gene": g, "human_age_rho": r.WT_age_rho,
                     "ML_consensus": "support" if _bool(r.in_ML_consensus) else "no_support",
                     "OOF_IG_fold_count": (ig.loc[g, "n_folds_top50"] if g in ig.index
                                           else np.nan),
                     "mouse_gene": m.mouse_gene if m is not None else np.nan,
                     "mouse_RNA": rna, "mouse_ATAC": atac,
                     "linked_peak": peak, "developmental_localization": dev,
                     "RNA_exclusion_reason": (m.get("exclusion_reason_RNA", "")
                                              if m is not None else "no ortholog row"),
                     "ATAC_exclusion_reason": (m.get("exclusion_reason_ATAC", "")
                                               if m is not None else "no ortholog row")})
    audit = pd.DataFrame(rows)
    audit.to_csv(FIGF / "fig5f_candidate_evidence_audit.tsv", sep="\t", index=False)
    counts = (audit[["mouse_RNA", "mouse_ATAC", "linked_peak",
                     "developmental_localization"]]
              .melt(var_name="modality", value_name="state")
              .groupby(["modality", "state"]).size().rename("n").reset_index())
    counts.to_csv(FIGF / "fig5f_state_counts.tsv", sep="\t", index=False)
    return audit


def _evidence_matrix(ax, genes):
    audit = _evidence_audit(genes)
    cols = ["human\nage ρ", "ML\nconsensus", "OOF-IG\nfolds / 18",
            "mouse\nRNA", "mouse\nATAC", "linked\npeak", "dev.\nlocalized"]
    colors = {"support": CONC, "discordant": DISC,
              "no_support": "#FFFFFF", "not_assessed": NA_GREY}
    for i, r in audit.iterrows():
        rho = pd.to_numeric(r.human_age_rho, errors="coerce")
        color = plt.cm.RdBu_r((np.clip(rho, -1, 1) + 1) / 2) if pd.notna(rho) else NA_GREY
        ax.add_patch(plt.Rectangle((-0.5, i - 0.5), 1, 1, fc=color,
                                   ec="white", lw=0.5))
        for j, key in enumerate(("ML_consensus", "mouse_RNA", "mouse_ATAC",
                                  "linked_peak", "developmental_localization"),
                                start=1):
            col = j if j == 1 else j + 1
            ax.add_patch(plt.Rectangle((col - 0.5, i - 0.5), 1, 1,
                                       fc=colors[r[key]], ec="#BBBBBB", lw=0.35))
        fold = r.OOF_IG_fold_count
        ax.add_patch(plt.Rectangle((1.5, i - 0.5), 1, 1, fc="#F5F5F5",
                                   ec="#BBBBBB", lw=0.35))
        ax.text(2, i, "NA" if pd.isna(fold) else f"{int(fold)}/18",
                ha="center", va="center", fontsize=5.2)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, fontsize=5.6)
    ax.set_yticks(range(len(genes))); ax.set_yticklabels(genes, fontsize=5.4)
    ax.set_xlim(-0.5, 6.5); ax.set_ylim(len(genes) - 0.5, -0.5)
    ax.set_title("Candidate evidence availability (descriptive)", fontsize=7)
    ax.tick_params(length=0)
    sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=plt.Normalize(-1, 1))
    sm.set_array([])
    cb = ax.figure.colorbar(sm, ax=ax, fraction=0.025, pad=0.02)
    cb.set_label("human age ρ", fontsize=5.5)
    cb.ax.tick_params(labelsize=5)
    handles = [Patch(color=CONC, label="concordant / support"),
               Patch(color=DISC, label="discordant"),
               Patch(color="#FFFFFF", edgecolor="#BBBBBB", label="assessed, no support"),
               Patch(color=NA_GREY, edgecolor="#BBBBBB", label="not assessed")]
    ax.figure.legend(handles=handles, fontsize=5.4, frameon=False,
                     ncol=4, loc="lower left", bbox_to_anchor=(0.17, 0.095))


def _group_counts(ax):
    panel = pd.read_csv(RES / "final_candidate_panel.csv")
    if panel.gene.duplicated().any() or panel.evidence_tier.isna().any():
        raise ValueError("Figure 5G: groups require unique, assigned genes")
    tc = panel.evidence_tier.value_counts().reindex([1, 2, 3], fill_value=0)
    if int(tc.sum()) != len(panel):
        raise ValueError("Figure 5G: groups are not exhaustive and mutually exclusive")
    labels = {1: "Group A", 2: "Group B", 3: "Group C"}
    y = np.arange(len(tc))[::-1]
    ax.barh(y, tc.values, color=["#2F5B8F", "#5B8DB8", "#C3D2E3"], height=0.6)
    for yi, v in zip(y, tc.values):
        ax.text(v + 8, yi, str(v), va="center", fontsize=6.5)
    ax.set_yticks(y); ax.set_yticklabels([labels[i] for i in tc.index], fontsize=5.8)
    ax.set_xlabel(f"Genes (exploratory; n={len(panel)})", fontsize=6)
    ax.set_xlim(0, max(tc.values) * 1.25)
    ax.set_title("Evidence-group sizes", fontsize=6.6)
    ax.tick_params(labelsize=5.8)


def build():
    # freeze the cross-species mother table once (review 3.4) before any panel
    _cross_species_table()
    panel = pd.read_csv(RES / "final_candidate_panel.csv")
    genes = (panel.sort_values(["evidence_tier", "evidence_score"],
                               ascending=[True, False]).head(20).gene.tolist())
    fig = plt.figure(figsize=(W_DOUBLE, 285 / 25.4))
    gs = gridspec.GridSpec(5, 2, figure=fig, left=0.17, right=0.96,
                           bottom=0.17, top=0.97, hspace=0.44, wspace=0.62,
                           height_ratios=[1.1, 1.1, 1.0, 2.3, 1.0])
    axA = fig.add_subplot(gs[0, 0]); letter(axA, "A"); _rna_scatter(axA)
    axB = fig.add_subplot(gs[0, 1]); letter(axB, "B", dx=-0.08); _rna_counts(axB)
    axC = fig.add_subplot(gs[1, 0]); letter(axC, "C"); _atac_scatter(axC)
    axD = fig.add_subplot(gs[1, 1]); letter(axD, "D", dx=-0.08); _atac_counts(axD)
    axE = fig.add_subplot(gs[2, :]); letter(axE, "E", dx=-0.08); _tf_attrition(axE)
    axF = fig.add_subplot(gs[3, :]); letter(axF, "F", dx=-0.08)
    _evidence_matrix(axF, genes)
    axG = fig.add_subplot(gs[4, :]); letter(axG, "G", dx=-0.08); _group_counts(axG)
    fig.text(0.17, 0.045,
             "Mouse: one young and one aged animal; genes/peaks are observations, not replicates.\n"
             "Groups A/B/C are exploratory priorities, not validated tiers.",
             fontsize=5.5, ha="left", va="center")
    save_composite(fig, "Figure5")

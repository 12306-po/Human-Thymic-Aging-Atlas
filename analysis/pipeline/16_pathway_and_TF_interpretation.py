#!/usr/bin/env python
"""Step 16: Pathway enrichment + TF interpretation of ML/DL candidates (rev. 2).

P0 fixes per Codex review 2026-09-14:
  1. Enrichment background: the gene universe is extracted from the Step 10
     feature_dictionary (gene_celltype + whole_thymus genes); it is saved to
     16_enrichment_background_genes.txt and passed to gseapy as `background`
     when supported.
  2. IG features are NEVER parsed with regex: candidate direction table merges
     token_metadata (feature -> gene, celltype) with validate='many_to_one' and
     aggregates by gene.
  3. WT_log2fc_old_vs_young is renamed WT_delta_log1p_CPM_old_vs_young
     (project-wide naming; the matrix is log1p(CPM), not log2).
  4. Frozen DL top-100 gene list is written once to 16_DL_top100_genes.txt;
     Steps 17/18a/20 read this frozen file instead of recomputing head(100).

P1 (important, replaces the old "keep old results on failure" behavior):
  - If Enrichr/gseapy fails (network unavailable), old enrichment outputs are
    moved to 06_interpretation/archive/ as reference ONLY; the official new
    outputs are marked NOT_RERUN with a reason; old enrichment files are never
    copied into the new result chain. Re-run (or offline GMT) is required once
    the network is back.

Inputs:
  04_machine_learning/ML_consensus_genes.txt, ML_consensus_features.csv,
    ML_consensus_genes.csv
  05_deep_learning/dataset/token_metadata.csv (feature -> gene, celltype)
  05_deep_learning/interpretation/gene_importance.csv,
    IG_across_fold_consistency.csv
  04_machine_learning/dataset/feature_dictionary.csv (background universe)
  03_feature_selection/DEG/whole_thymus_DEG_young_vs_old.csv
  03_feature_selection/age_correlation/whole_thymus_Spearman_by_age.csv

Outputs (06_interpretation/):
  16_pathway_enrichment_ML.csv / 16_pathway_enrichment_DL.csv
  16_TF_candidates.csv
  16_candidate_direction_table.csv
  16_theme_summary.csv
  16_DL_top100_genes.txt
  16_enrichment_background_genes.txt
  10_results/logs/16_pathway_TF.log
"""
from __future__ import annotations

import logging
import os
import shutil
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

PROJ = Path(os.environ["PROJ"])
ML = PROJ / "04_machine_learning"
DLDS = PROJ / "05_deep_learning" / "dataset"
DLINT = PROJ / "05_deep_learning" / "interpretation"
FS = PROJ / "03_feature_selection"
OUT = PROJ / "06_interpretation"
OUT.mkdir(parents=True, exist_ok=True)
ARCHIVE = OUT / "archive"
LOGS = PROJ / "10_results" / "logs"
LOGS.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("step16")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler(sys.stdout)
fh = logging.FileHandler(LOGS / "16_pathway_TF.log", mode="w")
fmt = logging.Formatter("%(asctime)s  %(levelname)s  %(message)s")
ch.setFormatter(fmt)
fh.setFormatter(fmt)
logger.addHandler(ch)
logger.addHandler(fh)
logger.info("=== Step 16 (rev.2): Pathway + TF interpretation ===")

# ------------------------------------------------------------------
# 1. Load candidates (frozen lists from Steps 12/15)
# ------------------------------------------------------------------
with open(ML / "ML_consensus_genes.txt") as f:
    ml_genes = [ln.strip() for ln in f if ln.strip()]
ml_feat = pd.read_csv(ML / "ML_consensus_features.csv")
dl_gene = pd.read_csv(DLINT / "gene_importance.csv")
logger.info(f"ML consensus genes: {len(ml_genes)}; DL genes: {len(dl_gene)}")

# DL top-100 by mean of attention-rank and IG-rank; frozen to file
dl_gene = dl_gene.copy()
dl_gene["dl_mean_rank"] = (
    dl_gene["mean_attention"].rank(ascending=False)
    + dl_gene["mean_IG"].rank(ascending=False)
) / 2
dl_gene = dl_gene.sort_values("dl_mean_rank")
dl_top100 = dl_gene.head(100)["gene"].tolist()
dl_top100_file = OUT / "16_DL_top100_genes.txt"
dl_top100_file.write_text("\n".join(dl_top100) + "\n")
logger.info(f"DL top-100 genes frozen to {dl_top100_file.name}")

# ------------------------------------------------------------------
# 2. Enrichment background (P0-1): gene universe from feature_dictionary
# ------------------------------------------------------------------
feature_dict = pd.read_csv(ML / "dataset" / "feature_dictionary.csv")
universe_genes = sorted(set(
    feature_dict.loc[feature_dict["feature_type"].isin(
        ["gene_celltype", "whole_thymus"]), "gene"].dropna()))
bg_file = OUT / "16_enrichment_background_genes.txt"
bg_file.write_text("\n".join(universe_genes) + "\n")
logger.info(f"Enrichment background: {len(universe_genes)} genes -> "
            f"{bg_file.name}")

# ------------------------------------------------------------------
# 3. Enrichr enrichment (gseapy; needs internet; failure handled per P1)
# ------------------------------------------------------------------
def clean(symbols):
    return sorted({s for s in symbols if s and not s.startswith("AC")
                   and not s.startswith("AL") and "." not in s})

ml_clean = clean(ml_genes)
dl_clean = clean(dl_top100)
logger.info(f"Mappable ML genes: {len(ml_clean)}/{len(ml_genes)}; "
            f"DL: {len(dl_clean)}/{len(dl_top100)}")

LIBS = ["GO_Biological_Process_2023", "KEGG_2021_Human",
        "Reactome_2022", "TRRUST_Transcription_Factors_2019"]


def run_enrichr(genes, tag):
    import inspect
    import gseapy as gp
    # P1 (2026-09-16): an explicit background universe is a frozen design
    # requirement. Verify the INSTALLED gseapy.enrichr signature up front — a
    # dict assignment cannot raise TypeError, so the previous try/except never
    # worked. Do NOT silently downgrade to a no-background enrichment.
    sig_params = inspect.signature(gp.enrichr).parameters
    if "background" not in sig_params:
        raise RuntimeError(
            "Installed gseapy.enrichr does not support an explicit `background` "
            "universe. Upgrade gseapy or use the offline-GMT enrichment path; "
            "do not run enrichment without the frozen background.")
    enr = gp.enrichr(
        gene_list=genes,
        gene_sets=LIBS,
        organism="human",
        background=universe_genes,
        outdir=None,
        cutoff=1.0,
        verbose=False,
    )
    df = enr.results.copy()
    df["candidate_set"] = tag
    return df


def overlap_str(r) -> str:
    """Report term overlap robustly across gseapy versions.

    With a background gene set, gseapy >= 1.1 returns `Odds Ratio` but drops the
    `Overlap` ("k/K") column. Fall back to the number of hit genes parsed from
    the `Genes` field (the numerator k) when `Overlap` is absent.
    """
    if "Overlap" in r and pd.notna(r.get("Overlap")):
        return str(r["Overlap"])
    genes = r.get("Genes", "")
    k = len([g for g in str(genes).split(";") if g]) if pd.notna(genes) else 0
    return f"{k} hits" if k else "n/a"


enr_ml, enr_dl = None, None
try:
    enr_ml = run_enrichr(ml_clean, "ML_consensus")
    enr_ml.to_csv(OUT / "16_pathway_enrichment_ML.csv", index=False)
    logger.info(f"ML enrichment rows: {len(enr_ml)}")
    enr_dl = run_enrichr(dl_clean, "DL_top100")
    enr_dl.to_csv(OUT / "16_pathway_enrichment_DL.csv", index=False)
    logger.info(f"DL enrichment rows: {len(enr_dl)}")
except Exception as e:  # noqa: BLE001
    logger.error(f"Enrichr/gseapy failed: {e}")
    logger.error("P1 strategy: old enrichment results go to archive/ ONLY; "
                 "official outputs marked NOT_RERUN (network required); "
                 "re-run or offline GMT needed.")
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    for f in OUT.glob("16_pathway_enrichment_*.csv"):
        shutil.move(str(f), str(ARCHIVE / f.name))
    for f in ("16_pathway_enrichment_ML.csv", "16_pathway_enrichment_DL.csv"):
        (OUT / f).write_text(
            "status:NOT_RERUN\nreason:Enrichr/gseapy network unavailable\n"
            "note:old results archived in 06_interpretation/archive/; "
            "re-run after network restore or use offline GMT.\n")
    logger.warning("Official enrichment files marked NOT_RERUN")


def sig(df, lib):
    d = df[(df["Gene_set"] == lib) & (df["Adjusted P-value"] < 0.05)]
    return d.sort_values("Adjusted P-value")


if enr_ml is not None and enr_dl is not None:
    for lib in LIBS:
        m, d = sig(enr_ml, lib), sig(enr_dl, lib)
        logger.info(f"{lib}: ML sig={len(m)}, DL sig={len(d)}")
        for _, r in m.head(5).iterrows():
            logger.info(f"  ML: {r['Term']} (p.adj={r['Adjusted P-value']:.2e}, "
                        f"overlap={overlap_str(r)})")

# ------------------------------------------------------------------
# 4. TF candidates (only if enrichment ran; else empty + marked)
# ------------------------------------------------------------------
if enr_ml is not None and enr_dl is not None:
    trrust_ml = sig(enr_ml, "TRRUST_Transcription_Factors_2019")
    trrust_dl = sig(enr_dl, "TRRUST_Transcription_Factors_2019")
    tf_rows = []
    for _, r in trrust_ml.iterrows():
        tf_rows.append({"TF": r["Term"], "source": "TRRUST_enriched_ML",
                        "adj_p": r["Adjusted P-value"],
                        "overlap_genes": r["Genes"],
                        "in_consensus": r["Term"] in set(ml_genes)})
    for _, r in trrust_dl.iterrows():
        tf_rows.append({"TF": r["Term"], "source": "TRRUST_enriched_DL",
                        "adj_p": r["Adjusted P-value"],
                        "overlap_genes": r["Genes"],
                        "in_consensus": r["Term"] in set(ml_genes)})
    tf_df = (pd.DataFrame(tf_rows).sort_values("adj_p") if tf_rows
             else pd.DataFrame(columns=["TF", "source", "adj_p",
                                        "overlap_genes", "in_consensus"]))
    in_list_tf = sorted({t for t in (list(trrust_ml["Term"]) +
                                     list(trrust_dl["Term"]))
                         if t in set(ml_genes)})
    logger.info(f"TRRUST TF rows: {len(tf_df)}; TFs also in consensus: {in_list_tf}")
else:
    tf_df = pd.DataFrame(columns=["TF", "source", "adj_p",
                                  "overlap_genes", "in_consensus"])
    logger.warning("TF candidates empty (enrichment NOT_RERUN)")
tf_df.to_csv(OUT / "16_TF_candidates.csv", index=False)

# ------------------------------------------------------------------
# 5. Candidate direction table (P0-2: merge token_metadata, no regex)
# ------------------------------------------------------------------
deg_wt = pd.read_csv(FS / "DEG" / "whole_thymus_DEG_young_vs_old.csv")
rho_wt = pd.read_csv(FS / "age_correlation" / "whole_thymus_Spearman_by_age.csv")

ml_best = (pd.read_csv(ML / "ML_consensus_genes.csv")
           .groupby("gene")
           .agg(ml_best_median_pct=("best_median_rank_pct", "min"),
                ml_n_features=("feature", "count"),
                ml_multimodel_any=("is_multimodel_recurrent", "max")))

# P0-2: merge token_metadata (feature -> gene/celltype) many-to-one, then
# aggregate IG across-fold consistency by gene — no regex feature parsing.
tok_meta = pd.read_csv(DLDS / "token_metadata.csv")
ac = pd.read_csv(DLINT / "IG_across_fold_consistency.csv")
ac = ac.merge(tok_meta[["feature", "gene"]], on="feature",
              validate="many_to_one")
ac_best = ac.groupby("gene")["n_folds_in_top50"].max()

cand = pd.DataFrame({"gene": sorted(set(ml_genes) | set(dl_gene["gene"]))})
cand = cand.merge(ml_best, on="gene", how="left")
cand = cand.merge(dl_gene.set_index("gene")[["dl_mean_rank", "mean_attention",
                                             "mean_IG", "n_tokens"]],
                  on="gene", how="left")
cand = cand.merge(deg_wt.set_index("gene")[["delta_log1p_CPM", "padj"]].rename(
    columns={"delta_log1p_CPM": "WT_delta_log1p_CPM_old_vs_young",
             "padj": "WT_DEG_padj"}),
    on="gene", how="left")
cand = cand.merge(rho_wt.set_index("gene")[["rho", "padj"]].rename(
    columns={"rho": "WT_age_rho", "padj": "WT_rho_padj"}),
    on="gene", how="left")
cand["DL_across_fold_max_top50"] = cand["gene"].map(ac_best)
cand["in_ML_consensus"] = cand["gene"].isin(ml_genes)
cand["in_DL_top100"] = cand["gene"].isin(dl_top100)
cand = cand.sort_values(["in_ML_consensus", "ml_best_median_pct"],
                        ascending=[False, True])
cand.to_csv(OUT / "16_candidate_direction_table.csv", index=False)
logger.info(f"Candidate table: {len(cand)} genes "
            f"({int(cand['in_ML_consensus'].sum())} ML, "
            f"{int(cand['in_DL_top100'].sum())} DL-top100)")

# ------------------------------------------------------------------
# 6. Theme summary (only if enrichment ran)
# ------------------------------------------------------------------
THEMES = {
    "T_cell_development": ["t cell", "thymic", "thymus", "lymphocyte differenti",
                           "antigen receptor", "vdj", "recombination"],
    "cell_cycle_proliferation": ["cell cycle", "mitotic", "proliferation",
                                 "dna replic", "chromosome segreg"],
    "inflammation_senescence": ["inflammat", "senescen", "aging", "interferon",
                                "cytokine", "chemokine", "nf-kappa"],
    "stroma_epithelium": ["epitheli", "stroma", "fibroblast", "extracellular matrix",
                          "collagen", "adhesion"],
    "metabolism": ["metabol", "oxidative phosphorylation", "mitochondri",
                   "glycolysis", "lipid"],
    "apoptosis_survival": ["apopto", "cell death", "survival", "autophagy"],
    "signaling": ["signaling pathway", "receptor signaling", "kinase",
                  "transcription"],
}


def theme_of(term):
    t = term.lower()
    return [th for th, kws in THEMES.items() if any(k in t for k in kws)] or ["other"]


if enr_ml is not None and enr_dl is not None:
    sig_terms = pd.concat([sig(enr_ml, L) for L in LIBS[:3]] +
                          [sig(enr_dl, L) for L in LIBS[:3]],
                          ignore_index=True)
    theme_rows = []
    for _, r in sig_terms.iterrows():
        for th in theme_of(r["Term"]):
            theme_rows.append({"theme": th, "candidate_set": r["candidate_set"],
                               "library": r["Gene_set"], "term": r["Term"],
                               "adj_p": r["Adjusted P-value"],
                               "overlap": overlap_str(r)})
    theme_df = pd.DataFrame(theme_rows)
    theme_sum = (theme_df.groupby(["theme", "candidate_set"])
                 .agg(n_sig_terms=("term", "count"), min_adj_p=("adj_p", "min"))
                 .reset_index().sort_values(["candidate_set", "n_sig_terms"],
                                            ascending=[True, False]))
    theme_sum.to_csv(OUT / "16_theme_summary.csv", index=False)
    logger.info(f"Theme summary:\n{theme_sum.to_string()}")
else:
    (OUT / "16_theme_summary.csv").write_text(
        "status:NOT_RERUN\nreason:Enrichr/gseapy network unavailable\n")
    logger.warning("Theme summary marked NOT_RERUN")

logger.info("\n=== Step 16 (rev.2) COMPLETE ===")

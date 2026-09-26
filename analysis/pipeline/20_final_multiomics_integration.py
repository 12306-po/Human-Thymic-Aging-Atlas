#!/usr/bin/env python
"""Step 20: Final multi-omics integration and evidence scoring.

Aggregates every evidence layer produced in Steps 05-19 into a gene-level
evidence matrix and final candidate panel, then assigns evidence tiers and a
mechanism chain per gene.

Evidence scoring (weights reflect the frozen statistical design v2.1: the
continuous-age ML regression is PRIMARY; all other layers are supportive):
  * ML regression (in_ML_consensus, ml_best_median_pct, ml_n_features,
    ml_multimodel_any)                                weight = 3
  * DL attribution (in_DL_top100, dl_mean_rank, DL_across_fold_max_top50)  w = 1
  * External human developmental localization (Step 17) weight = 0 (context only)
  * Mouse scRNA direction concordance (Step 18)       weight = 1
  * Mouse scATAC direction concordance (Step 19)      weight = 1
Direction consistency: all directional layers must agree in sign for the
"chain_consistent" flag (human WT_age_rho vs mouse RNA vs mouse ATAC).
External human state localization is CONTEXT ONLY (weight 0): it is NOT an
aging replication layer and NOT part of the formal direction chain.

Inputs (10_results/../ step outputs):
  * 06_interpretation/16_candidate_direction_table.csv   (661-gene universe)
  * 06_interpretation/16_TF_candidates.csv
  * 07_external_validation/17_candidate_gene_top_states.csv
  * 08_mouse_validation/18_mouse_human_orthologs.csv
  * 08_mouse_validation/19_scATAC_gene_level_join.csv
  * 04_machine_learning/ML_consensus_features.csv

Outputs (10_results/):
  * final_evidence_matrix.csv
  * final_candidate_panel.csv
  * final_project_manifest.csv
  * 09_figures/20_evidence_heatmap.pdf
  * 10_results/logs/20_final_integration.log
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

def _env(key):
    val = os.environ.get(key)
    if not val:
        raise RuntimeError(f"Required env var {key} is not set. Run: source scripts/00_project_config.sh")
    return Path(val)

PROJ = _env("PROJ")
LOGS = PROJ / "10_results" / "logs"
LOGS.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("step20")
logger.setLevel(logging.INFO)
_ch = logging.StreamHandler(sys.stdout)
_fh = logging.FileHandler(LOGS / "20_final_integration.log", mode="w")
_fmt = logging.Formatter("%(asctime)s  %(levelname)s  %(message)s")
_ch.setFormatter(_fmt)
_fh.setFormatter(_fmt)
logger.addHandler(_ch)
logger.addHandler(_fh)
logger.info("=== Step 20: Final multi-omics integration ===")

W_ML = 3.0
W_DL = 1.0
# P1-7 (2026-09-16): external human localization is context only (score_HUMAN
# is forced to 0 below); keep the weight constant at 0 rather than a misleading 1.
W_HUMAN = 0.0
W_RNA = 1.0
W_ATAC = 1.0

# ----------------------------------------------------------------------
# 1. Base universe: 661-gene candidate direction table (Step 16)
# ----------------------------------------------------------------------
base = pd.read_csv(PROJ / "06_interpretation" / "16_candidate_direction_table.csv")
for c in ("in_ML_consensus", "in_DL_top100"):
    base[c] = base[c].astype(bool)
base["ml_best_median_pct"] = pd.to_numeric(base["ml_best_median_pct"], errors="coerce")
base["ml_n_features"] = pd.to_numeric(base["ml_n_features"], errors="coerce").fillna(0)
base["dl_mean_rank"] = pd.to_numeric(base["dl_mean_rank"], errors="coerce")
base["WT_age_rho"] = pd.to_numeric(base["WT_age_rho"], errors="coerce")
base["WT_rho_padj"] = pd.to_numeric(base["WT_rho_padj"], errors="coerce")
logger.info("Base universe: %d genes (%d ML consensus, %d DL top100)",
            len(base), int(base["in_ML_consensus"].sum()), int(base["in_DL_top100"].sum()))

# ----------------------------------------------------------------------
# 2. External human (Step 17): context flags only (NOT counted as aging +1)
#    P0-1: split into three boolean columns:
#      human_external_detected — gene present in external data
#      human_developmental_localized — gene passed developmental localization
#      human_aging_replication = unavailable (external dataset has no age info)
#    These are reported as context but do NOT contribute to evidence_score.
# ----------------------------------------------------------------------
# P0-1 (2026-09-15): Step 17's unique gene-level wide table
# (17_candidate_gene_developmental_context.csv) has ONE row per gene. Read that
# instead of the long 17_candidate_gene_top_states.csv (which has 2 rows per
# gene — one per Sort1/Sort2) so the merge cannot duplicate gene rows.
ctx17 = PROJ / "07_external_validation" / "17_candidate_gene_developmental_context.csv"
top_states17 = PROJ / "07_external_validation" / "17_candidate_gene_top_states.csv"
if ctx17.exists():
    ext = pd.read_csv(ctx17)
    ext = ext[["gene", "top_state_Sort1", "top_state_Sort2"]].copy()
    # collapse the two object top-states into a single display column
    ext["external_top_state"] = ext.apply(
        lambda r: "; ".join(
            [x for x in [str(r.get("top_state_Sort1")), str(r.get("top_state_Sort2"))]
             if x not in ("nan", "None", "") and x is not None]),
        axis=1,
    )
    ext["human_external_detected"] = ext["external_top_state"].ne("")
    ext["human_developmental_localized"] = ext["human_external_detected"]
    ext["external_stage_class"] = ""  # Sort1/Sort2 discrete states; no single maturity class
    ext = ext[["gene", "external_top_state", "external_stage_class",
               "human_external_detected", "human_developmental_localized"]]
else:
    # fallback (legacy): long table — de-duplicate to one row per gene
    top17 = pd.read_csv(top_states17)
    ext = top17[["gene", "developmental_stage"]].drop_duplicates("gene").copy()
    ext = ext.rename(columns={"developmental_stage": "external_top_state"})
    ext["external_stage_class"] = ""
    ext["human_external_detected"] = True
    ext["human_developmental_localized"] = ext["external_top_state"].notna()
ext["human_aging_replication"] = False   # P0-1: unavailable — no age data in external set
# P0-1: assert one row per gene before merging
assert not ext["gene"].duplicated().any(), (
    "Step 17 external context table has duplicated gene rows; use "
    "17_candidate_gene_developmental_context.csv")
base = base.merge(ext, on="gene", how="left")

# ----------------------------------------------------------------------
# 3. Mouse scRNA (Step 18): direction concordance per human gene
# ----------------------------------------------------------------------
# P0-3 (2026-09-15): Step 18's `concordant` is sign(WT_age_rho) ==
# sign(RNA_adjusted_effect), TRUE/FALSE for all assessable genes (never NA for
# assessable pairs). Only genes without an assessable effect stay NA. Do NOT
# drop Discordant genes (they are concordant=FALSE, not missing).
rna18 = pd.read_csv(PROJ / "08_mouse_validation" / "18_mouse_human_orthologs.csv")
rna18 = rna18[rna18["assessable"].fillna(False).astype(bool) &
              rna18["human_gene"].notna()]
# one row per human gene: a human gene must map to exactly one best ortholog
assert not rna18["human_gene"].duplicated().any(), (
    "Step 18 output has duplicated human_gene rows; check best-ortholog "
    "uniqueness before integration")
rna = rna18[["human_gene", "mouse_gene", "direction", "concordant",
             "RNA_adjusted_effect", "RNA_effect_z"]].rename(
    columns={
        "human_gene": "gene",
        "mouse_gene": "mouse_RNA_gene",
        "direction": "mouse_RNA_direction",
        "concordant": "mouse_RNA_concordant",
        "RNA_adjusted_effect": "mouse_RNA_effect",
        "RNA_effect_z": "mouse_RNA_effect_z",
    }
)
base = base.merge(rna, on="gene", how="left")

# ----------------------------------------------------------------------
# 4. Mouse scATAC (Step 19): gene-level ATAC direction concordance
# ----------------------------------------------------------------------
atac19 = pd.read_csv(PROJ / "08_mouse_validation" / "19_scATAC_gene_level_join.csv")
# P1-6 (2026-09-16): never silently drop_duplicates("human_gene"). Step 19's
# join is expected to be one row per human gene; if it is not, write the dups
# for inspection and fail rather than arbitrarily keeping the first row.
_dup_atac = atac19[atac19["human_gene"].notna()]
if _dup_atac["human_gene"].duplicated().any():
    dup = _dup_atac[_dup_atac["human_gene"].duplicated(keep=False)].sort_values("human_gene")
    (PROJ / "10_results").mkdir(exist_ok=True)
    dup.to_csv(PROJ / "10_results" / "20_duplicate_ATAC_gene_rows.csv", index=False)
    raise RuntimeError(
        "Step19 ATAC output is not one-row-per-human-gene; resolve upstream. "
        "See 10_results/20_duplicate_ATAC_gene_rows.csv")
del _dup_atac
atac = atac19[["human_gene", "ATAC_adjusted_effect", "ATAC_effect_z",
               "atac_detectable", "atac_direction_ok"]].rename(
    columns={
        "human_gene": "gene",
        "ATAC_adjusted_effect": "mouse_ATAC_effect",
        "ATAC_effect_z": "mouse_ATAC_effect_z",
        "atac_detectable": "mouse_ATAC_detectable",
        "atac_direction_ok": "mouse_ATAC_concordant",
    }
)
base = base.merge(atac, on="gene", how="left")

# ----------------------------------------------------------------------
# 5. TF links (Step 16)
# ----------------------------------------------------------------------
tf16 = pd.read_csv(PROJ / "06_interpretation" / "16_TF_candidates.csv")
tf_links = {}
for _, row in tf16.iterrows():
    for g in str(row["overlap_genes"]).split(";"):
        g = g.strip()
        if not g:
            continue
        tf_links.setdefault(g, []).append(f'{row["TF"]} ({row["source"]})')
base["TF_links"] = base["gene"].map(lambda g: "; ".join(tf_links.get(g, []))).fillna("")

# ----------------------------------------------------------------------
# 6. Directional support per layer (for chain consistency / tiers)
#
# Separation of responsibilities (2026-09-16):
#   Step 18 adjudicates RNA concordance -> mouse_RNA_concordant
#   Step 19 adjudicates ATAC concordance -> mouse_ATAC_concordant
# Step 20 is integration ONLY and must NOT re-derive any direction from raw
# effects (no sign_ok / sign(effect) recomputation here). Both columns are
# tri-state: TRUE = concordant, FALSE = discordant, NA = non-assessable
# (e.g. layer not detectable, or zero effect / zero human rho upstream).
# ----------------------------------------------------------------------
base["mouse_RNA_supports"] = base["mouse_RNA_concordant"].astype("boolean")
base["mouse_ATAC_supports"] = base["mouse_ATAC_concordant"].astype("boolean")

# ----------------------------------------------------------------------
# 7. Evidence scores (P0-2: all components capped; score_ML clipped 0-3)
# P0-2: ML score components:
#   in_ML_consensus = 1.5
#   multimodel (best in ≥2 models) = +0.75
#   best_pct <= 0.10 = +0.75
#   (log1p(n_features)*0.1 still adds a small count bonus)
#   Total cap: clip 0-3.
# DL: 0-1; mouse_RNA: 0-1; mouse_ATAC: 0-1.
# ----------------------------------------------------------------------
score_ml_base = np.where(base["in_ML_consensus"], 1.5, 0.0)
score_ml_base += np.where(base["ml_multimodel_any"].fillna(False).astype(bool), 0.75, 0.0)
score_ml_base += np.where(base["ml_best_median_pct"].fillna(1.0) <= 0.10, 0.75, 0.0)
score_ml_base += np.log1p(base["ml_n_features"].fillna(0)) * 0.05  # small count bonus
base["score_ML"] = np.clip(score_ml_base, 0.0, 3.0)
base["score_DL"] = np.clip(
    np.where(base["in_DL_top100"],
             (1 - base["dl_mean_rank"].fillna(1) / 100), 0.0),
    0.0, 1.0)
# P0-1: external is context only — score_HUMAN = 0 (NOT counted as aging +1)
base["score_HUMAN"] = 0.0
base["score_RNA"] = np.clip(np.where(base["mouse_RNA_supports"].eq(True).fillna(False), W_RNA, 0.0), 0.0, 1.0)
base["score_ATAC"] = np.clip(np.where(base["mouse_ATAC_supports"].eq(True).fillna(False), W_ATAC, 0.0), 0.0, 1.0)
score_cols = ["score_ML", "score_DL", "score_RNA", "score_ATAC"]
base["evidence_score"] = base[score_cols].sum(axis=1)

# binary evidence flags for the matrix
# P0-1: ev_HUMAN is now a context flag only (not counted as aging evidence)
base["ev_ML"] = base["in_ML_consensus"].astype(int)
base["ev_DL"] = base["in_DL_top100"].astype(int)
base["ev_HUMAN"] = 0   # context only — does not count toward n_evidence_layers
base["ev_RNA"] = base["mouse_RNA_supports"].eq(True).fillna(False).astype(int)
# P0-4: ev_ATAC is split into layered sub-evidence (see section 7a below)
base["ev_ATAC_gene"] = base["mouse_ATAC_supports"].eq(True).fillna(False).astype(int)
base["ev_ATAC"] = base["ev_ATAC_gene"]   # default; refined below if peak data available
ev_cols = ["ev_ML", "ev_DL", "ev_HUMAN", "ev_RNA", "ev_ATAC"]
base["n_evidence_layers"] = base[["ev_ML", "ev_DL", "ev_RNA", "ev_ATAC"]].sum(axis=1)

# NOTE: chain_consistent is computed AFTER linked-peak aggregation (section 8),
# so the strong-peak-conflict veto (P0-5) is included. Do not set it here.

# ----------------------------------------------------------------------
# 7a. ATAC layered evidence (P0-4): gene / linked-peak / TF-motif /
#     TF-peak-target. mouse_regulatory_support uses the FROZEN combination
#     rule (see docstring at section 8). Linked-peak rule (frozen):
#       * 1 assessable peak: direction agrees with human
#       * >=2 assessable peaks: concordant/assessable >= 0.5 AND median peak
#         effect sign agrees with human direction
#     The full TF->motif->peak->target chain is ENHANCED mechanistic support
#     (never a requirement for every candidate gene).
# ----------------------------------------------------------------------
base["ev_ATAC_peak"] = 0
base["ev_ATAC_TF_motif"] = 0
base["ev_ATAC_TF_peak_target"] = 0
# P1-8: TF-chain semantics split (regulator vs target-of-supported-TF)
base["ev_TF_as_regulator"] = 0
base["ev_target_of_supported_TF"] = 0
base["peak_support"] = False
base["peak_strong_conflict"] = False
_tf_net = PROJ / "08_mouse_validation" / "19_TF_motif_peak_target_validation.csv"
if _tf_net.exists():
    try:
        tfn = pd.read_csv(_tf_net)
        # P0-2 (2026-09-15): the network is keyed by human_TARGET_gene and
        # human_TF. Steps 19/20 do NOT have a `human_gene` column:
        #   * linked-peak support aggregates by human_target_gene;
        #   * TF motif / TF-peak-target support are recorded against human_TF.
        peak_support_rows = []
        # P0-1/P0-2 (2026-09-16): de-duplicate the TF x motif x peak x target
        # network to ONE ROW PER (target gene, peak) BEFORE aggregating — the
        # same peak appears on multiple network rows (one per TF/motif), and
        # counting rows would inflate n_peaks and bias the concordance fraction.
        # Linked-peak support compares sign(peak effect) against the human
        # TARGET gene's age-rho direction (target_peak_direction_support from
        # Step 19), NOT against the TF/motif reference_direction.
        peak_df = tfn.dropna(
            subset=["human_target_gene", "peak", "peak_age_effect", "target_human_age_rho"]
        ).copy()
        # Zero-direction rule (2026-09-16): 0 carries no direction. A zero
        # peak effect or a zero human target rho is NON-ASSESSABLE — it is
        # neither concordant nor discordant and must not enter the
        # n_unique_peaks / concordant / support / conflict denominators.
        #   NA      -> non-assessable (dropped above)
        #   0       -> non-assessable (dropped here)
        #   nonzero -> assessable directional peak
        peak_df = peak_df[
            peak_df["peak_age_effect"].ne(0)
            & peak_df["target_human_age_rho"].ne(0)
        ].copy()
        if "target_peak_direction_support" not in peak_df.columns:
            # recompute defensively if an older Step 19 output is read.
            # P1 (2026-09-16): zero effects have no direction; require both
            # nonzero so (rho=0, effect=0) is not counted as concordant.
            peak_df["target_peak_direction_support"] = (
                peak_df["peak_age_effect"].ne(0)
                & peak_df["target_human_age_rho"].ne(0)
                & (
                    np.sign(peak_df["peak_age_effect"])
                    == np.sign(peak_df["target_human_age_rho"])
                )
            )
        peak_df = (
            peak_df.sort_values(["human_target_gene", "peak", "motif_p_adjust"])
            .drop_duplicates(["human_target_gene", "peak"], keep="first")
        )
        # linked-peak support, aggregated at the TARGET gene level
        for g, grp in peak_df.groupby("human_target_gene"):
            n_peak = len(grp)
            n_conc = int(grp["target_peak_direction_support"].eq(True).sum())
            frac = n_conc / n_peak
            median_eff = grp["peak_age_effect"].median()
            human_sign = int(np.sign(grp["target_human_age_rho"].iloc[0]))
            if n_peak == 1:
                ok = bool(grp["target_peak_direction_support"].iloc[0])
                strong_conflict = not ok
            else:
                ok = (frac >= 0.5) and (np.sign(median_eff) == human_sign)
                # P0-5: majority opposite AND median effect opposite = strong conflict
                strong_conflict = (frac < 0.5) and (np.sign(median_eff) == -human_sign)
            peak_support_rows.append({
                "human_target_gene": g, "peak_support": ok,
                "peak_strong_conflict": strong_conflict,
                "n_unique_peaks": n_peak, "concordant_unique_peaks": n_conc,
                "median_peak_effect": float(median_eff)})
        # Drop any pre-initialized peak columns BEFORE the merge, otherwise the
        # incoming ps_df columns collide and pandas suffixes them to
        # peak_support_x/_y -> KeyError below. peak_support is initialized to
        # False earlier (default); the merge is authoritative when rows exist.
        for col in ["n_peaks", "concordant_peaks", "n_unique_peaks",
                    "concordant_unique_peaks", "median_peak_effect",
                    "peak_support", "peak_strong_conflict"]:
            if col in base.columns:
                base = base.drop(columns=col)
        if peak_support_rows:
            ps_df = pd.DataFrame(peak_support_rows).rename(
                columns={"human_target_gene": "gene"})
            base = base.merge(ps_df, on="gene", how="left", validate="one_to_one")
            base["peak_support"] = base["peak_support"].fillna(False).astype(bool)
            base["peak_strong_conflict"] = base["peak_strong_conflict"].fillna(False).astype(bool)
            base["n_unique_peaks"] = base["n_unique_peaks"].astype("Int64")
            base["concordant_unique_peaks"] = base["concordant_unique_peaks"].astype("Int64")
            base["ev_ATAC_peak"] = base["peak_support"].astype(int)
        else:
            base["peak_support"] = False
            base["peak_strong_conflict"] = False
            for col in ["n_unique_peaks", "concordant_unique_peaks", "median_peak_effect"]:
                base[col] = pd.NA
        # P1-8: split TF-chain semantics — a gene can be a supported TF
        # REGULATOR and/or a TARGET of a supported TF. Both are mechanistic
        # audit flags; neither is required for candidate tier assignment.
        base["ev_TF_as_regulator"] = base["gene"].isin(
            tfn.loc[tfn["motif_direction_support"].eq(True), "human_TF"].dropna()
        ).astype(int)
        full_targets = tfn.loc[tfn["full_chain_supported"].eq(True),
                               "human_target_gene"].dropna()
        base["ev_target_of_supported_TF"] = base["gene"].isin(full_targets).astype(int)
        base["ev_ATAC_TF_motif"] = base["ev_TF_as_regulator"]
        base["ev_ATAC_TF_peak_target"] = base["ev_target_of_supported_TF"]
        logger.info("ATAC peak layer: %d genes with unique assessable peaks, %d with peak_support",
                    int(base["n_unique_peaks"].notna().sum()), int(base["peak_support"].sum()))
        # P0-1: after major merge, gene rows must still be unique
        assert base["gene"].is_unique, "Gene duplication introduced by ATAC peak merge"
    except Exception as e:  # noqa: BLE001
        # P0 (2026-09-16): the network file EXISTS but could not be parsed/
        # integrated. Silently downgrading to gene-level ATAC would let the
        # pipeline "succeed" while linked-peak and TF-network evidence were
        # silently dropped. Write an error audit and FAIL FAST.
        error_file = PROJ / "10_results" / "20_ATAC_peak_layer_ERROR.txt"
        error_file.write_text(
            "Step20 ATAC peak/network integration failed.\n"
            f"network file: {_tf_net}\n\n"
            f"{repr(e)}\n"
        )
        logger.exception(
            "ATAC peak/network layer exists but integration failed: %s", e
        )
        raise RuntimeError(
            "Step19 ATAC peak/network file exists, but Step20 could not "
            "integrate it. Final integration aborted. "
            f"See {error_file}"
        ) from e
else:
    logger.warning("Step 19 network file %s not found — "
                   "ATAC network layer unavailable; gene-level ATAC only.",
                   _tf_net.name)
    base["peak_support"] = False
    base["peak_strong_conflict"] = False
    base["ev_TF_as_regulator"] = 0
    base["ev_target_of_supported_TF"] = 0
    for col in ["n_unique_peaks", "concordant_unique_peaks", "median_peak_effect"]:
        base[col] = pd.NA

# P0-4 frozen combination for mouse_regulatory_support (2026-09-16 update):
#   TRUE iff (ATAC gene-level direction_ok) OR (linked-peak TARGET-direction
#   evidence), and no strong opposite conflict from any measured directional
#   layer (RNA, ATAC gene, or linked peaks — P0-5 adds peak_strong_conflict).
atac_gene_ok = base["mouse_ATAC_supports"].eq(True).fillna(False)
atac_peak_ok = base["peak_support"].eq(True).fillna(False)
reg_support = (atac_gene_ok | atac_peak_ok)
# strong conflict = an assessable directional layer points opposite to human
strong_conflict = pd.Series(False, index=base.index)
atac_meas = base["mouse_ATAC_supports"].notna()
strong_conflict |= (atac_meas & ~base["mouse_ATAC_supports"].fillna(True))
# P0-5: a majority-opposite, median-opposite linked-peak set is a strong conflict
strong_conflict |= base["peak_strong_conflict"].fillna(False).astype(bool)
base["mouse_regulatory_support"] = (reg_support & ~strong_conflict).astype(bool)

# P0-3 (2026-09-16): peak evidence may have flipped ev_ATAC/mouse_regulatory_
# support on, so RECOMPUTE the ATAC layer, total evidence_score and
# n_evidence_layers HERE (after peak aggregation), keeping every column
# internally consistent (previously these were frozen gene-level-only).
base["ev_ATAC_peak"] = base["peak_support"].fillna(False).astype(int)
base["ev_ATAC"] = base["mouse_regulatory_support"].fillna(False).astype(int)
base["score_ATAC"] = (base["ev_ATAC"] * W_ATAC).clip(0.0, 1.0)
base["score_RNA"] = (
    np.where(base["mouse_RNA_supports"].eq(True).fillna(False), W_RNA, 0.0)
).clip(0.0, 1.0)
base["evidence_score"] = base[
    ["score_ML", "score_DL", "score_RNA", "score_ATAC"]
].sum(axis=1)
base["ev_RNA"] = base["mouse_RNA_supports"].eq(True).fillna(False).astype(int)
base["n_evidence_layers"] = base[
    ["ev_ML", "ev_DL", "ev_RNA", "ev_ATAC"]
].sum(axis=1)

# ----------------------------------------------------------------------
# 8. Evidence tiers (P0-3 redefinition; RULE-BASED — score never upgrades tier)
#   Tier 1: ML consensus AND RNA support AND regulatory support (ATAC
#           gene/linked-peak agrees) AND no explicit direction conflict.
#   Tier 2: ML consensus + 1 mouse modality (RNA or ATAC) without full conflict,
#           OR ML consensus + strong DL attribution + no conflict.
#   Tier 3: everything else in the candidate universe.
#   Regulatory support is defined by P0-4 (ATAC gene-level direction agreement
#   OR linked-peak evidence satisfied, without strong opposite-conflict).
#   external localization is CONTEXT ONLY (not part of tier assignment).
#
#   P0-4b (frozen): tiers are RULE-BASED; evidence_score is used only for
#   ranking WITHIN a tier. Score NEVER upgrades a Tier-2 gene to Tier-1.
# ----------------------------------------------------------------------
# mouse_regulatory_support was computed above with the frozen P0-4 combination
# (ATAC gene-level direction_ok OR linked-peak TARGET-direction evidence, and no
# strong opposite conflict incl. peak_strong_conflict), and ev_ATAC/score_ATAC/
# evidence_score/n_evidence_layers were recomputed from it. Tiers below consume
# those consistent columns.
#   mouse_regulatory_support = (atac_gene_ok | atac_peak_ok) & ~strong_conflict

# peak_assessed (P1, 2026-09-16): at least one unique assessable linked peak.
# Linked peaks are a genuine directional layer in their own right — even when
# both RNA and ATAC gene-level effects are NA, an assessable peak set must be
# counted when deciding whether all EVALUATED directional layers agree.
base["peak_assessed"] = (
    base["n_unique_peaks"].notna() & (base["n_unique_peaks"] > 0)
).astype(bool)

# chain_consistent: TRUE iff every ASSESSED directional layer (RNA, ATAC gene,
# linked peaks) agrees with the human aging direction; NA = no layer assessed.
# peak_chain_ok treats a non-assessed peak layer as no constraint; an assessed
# peak layer is OK when peak_support holds (which already encodes no strong
# conflict under the frozen linked-peak rule).
sup = (
    base["mouse_RNA_supports"].notna().astype(int)
    + base["mouse_ATAC_supports"].notna().astype(int)
    + base["peak_assessed"].astype(int)
)
peak_chain_ok = ~base["peak_assessed"] | base["peak_support"].fillna(False).astype(bool)
chain = (
    base["mouse_RNA_supports"].fillna(True)
    & base["mouse_ATAC_supports"].fillna(True)
    & peak_chain_ok
    & ~base["peak_strong_conflict"].fillna(False).astype(bool)
)
base["chain_consistent"] = np.where(sup > 0, chain, np.nan)

# P0-5 (2026-09-16): peak strong conflict joins the regulatory no-conflict gate
reg_no_conflict = (
    base["mouse_RNA_supports"].fillna(True)
    & base["mouse_ATAC_supports"].fillna(True)
    & ~base["peak_strong_conflict"].fillna(False).astype(bool)
).astype(bool)

# strong DL attribution: in_DL_top100 with meaningful mean rank
strong_dl = base["in_DL_top100"] & (base["dl_mean_rank"] <= 50)

# P0-3 (frozen): Tier1 = ML consensus AND RNA support AND regulatory support
ml_and_rna = base["in_ML_consensus"] & base["ev_RNA"].eq(1) & reg_no_conflict
ml_and_reg = base["in_ML_consensus"] & base["mouse_regulatory_support"] & reg_no_conflict

tier1 = ml_and_reg & ml_and_rna
# P0-4 (2026-09-16): Tier2 may use ANY accepted regulatory evidence —
# mouse_regulatory_support already means (ATAC gene-level OR linked-peak
# target-direction evidence) without strong conflict.
tier2 = (
    base["in_ML_consensus"]
    & (base["ev_RNA"].eq(1) | base["mouse_regulatory_support"])
    & reg_no_conflict
) | (base["in_ML_consensus"] & strong_dl & reg_no_conflict)
# remove anything already claimed by Tier1 (Tier1 is a stricter subset)
tier2 = tier2 & ~tier1
tier3 = ~tier1 & ~tier2
base["evidence_tier"] = np.select([tier1, tier2], [1, 2], default=3)

# ----------------------------------------------------------------------
# 9. Mechanism chain (celltype context from the authoritative feature
#    dictionary — P0-5: merge feature_dictionary, no suffix parsing)
# ----------------------------------------------------------------------
# The authoritative dictionary is the Step 10/12 CSV
# (04_machine_learning/dataset/feature_dictionary.csv: feature / feature_type /
#  gene / celltype / source_matrix / is_whole_thymus / is_composition).
fd_df = pd.read_csv(PROJ / "04_machine_learning" / "dataset" / "feature_dictionary.csv")
ct_by_gene = {}
for feat, g, ct in zip(fd_df["feature"], fd_df["gene"].astype(str), fd_df["celltype"].astype(str)):
    if g and ct and str(ct) != "WT" and str(ct).lower() != "nan":
        ct_by_gene.setdefault(g, []).append(ct)
base["mechanism_celltypes"] = base["gene"].map(
    lambda g: "; ".join(sorted(set(ct_by_gene.get(g, []))))
).fillna("")
# Three-state human direction (2026-09-16), aligned with Step 19: rho == 0
# carries no direction and is "unknown", NOT "age_down". NA rho -> unknown.
rho_num = pd.to_numeric(base["WT_age_rho"], errors="coerce")
base["direction_summary"] = np.select(
    [rho_num > 0, rho_num < 0],
    ["age_up", "age_down"],
    default="unknown",
)

# ----------------------------------------------------------------------
# 10. Write outputs
# ----------------------------------------------------------------------
panel_cols = [
    "gene", "evidence_tier", "evidence_score", "n_evidence_layers",
    "direction_summary", "WT_age_rho", "WT_rho_padj",
    "in_ML_consensus", "ml_best_median_pct", "ml_n_features",
    "in_DL_top100", "dl_mean_rank",
    "human_external_detected", "human_developmental_localized",
    "human_aging_replication", "external_top_state", "external_stage_class",
    "mouse_RNA_direction", "mouse_RNA_concordant",
    "mouse_ATAC_detectable", "mouse_ATAC_concordant",
    "mouse_regulatory_support", "peak_assessed", "peak_support",
    "peak_strong_conflict",
    "n_unique_peaks", "concordant_unique_peaks", "median_peak_effect",
    "ev_TF_as_regulator", "ev_target_of_supported_TF",
    "chain_consistent", "mechanism_celltypes", "TF_links",
]
panel = base[panel_cols].sort_values(
    ["evidence_tier", "evidence_score"], ascending=[True, False]
)
panel.to_csv(PROJ / "10_results" / "final_candidate_panel.csv", index=False)

evmat_cols = ["gene", "evidence_tier", "evidence_score"] + ev_cols + \
             ["peak_assessed", "ev_ATAC_peak", "peak_strong_conflict",
              "ev_ATAC_TF_motif", "ev_ATAC_TF_peak_target",
              "ev_TF_as_regulator", "ev_target_of_supported_TF"] + \
             ["chain_consistent", "direction_summary", "mouse_regulatory_support"]
evmat = base[evmat_cols].sort_values(
    ["evidence_tier", "evidence_score"], ascending=[True, False]
)
evmat.to_csv(PROJ / "10_results" / "final_evidence_matrix.csv", index=False)

logger.info("Tier counts: %s", base["evidence_tier"].value_counts().sort_index().to_dict())
logger.info("Wrote final_candidate_panel.csv (%d rows), final_evidence_matrix.csv (%d rows)",
            len(panel), len(evmat))
t1 = base[base["evidence_tier"] == 1]
if len(t1):
    logger.info("Top Tier-1 candidates:\n%s",
                t1[["gene", "evidence_score", "mechanism_celltypes", "chain_consistent"]]
                  .head(20).to_string(index=False))

# ----------------------------------------------------------------------
# 11. Project manifest (all key artifacts across steps)
# ----------------------------------------------------------------------
manifest_rows = []
key_glob = {
    "step01_04": ["01_raw_processing/metadata/*", "01_raw_processing/raw/*"],
    # P1 (2026-09-15): Steps 05-07 write under 01_raw_processing/{metadata,filtered}
    # and 09_figures/, not 10_results/. Globs corrected to real output paths.
    "step05": ["01_raw_processing/metadata/05*"],
    "step06": ["01_raw_processing/filtered/06_*",
               "01_raw_processing/metadata/06*",
               "09_figures/06_*"],
    "step07": ["01_raw_processing/filtered/07_*",
               "01_raw_processing/metadata/07*",
               "09_figures/07_*"],
    "step08": ["02_pseudobulk/**/*"],
    "step09": ["03_feature_selection/*"],
    "step10": ["04_machine_learning/dataset/*"],
    "step11": ["04_machine_learning/models/*", "04_machine_learning/metrics/*",
               "04_machine_learning/predictions/*"],
    "step12": ["04_machine_learning/ML_consensus_*"],
    "step13": ["05_deep_learning/dataset/*"],
    "step14": ["05_deep_learning/checkpoints/*"],
    "step15": ["05_deep_learning/interpretation/*"],
    "step16": ["06_interpretation/*", "06_interpretation/16_DL_top100_genes.txt",
               "06_interpretation/16_enrichment_background_genes.txt"],
    "step17": ["07_external_validation/*"],
    "step18": ["08_mouse_validation/18_*"],
    "step18a": ["08_mouse_validation/orthologs/*"],
    "step19": ["08_mouse_validation/19_*"],
    "step20": ["10_results/final_*"],
    "figures": ["09_figures/*"],
    "logs": ["10_results/logs/*"],
}
for step, pats in key_glob.items():
    for pat in pats:
        for p in sorted(PROJ.glob(pat)):
            if p.is_file():
                manifest_rows.append({"step": step, "artifact": str(p.relative_to(PROJ)),
                                      "size_bytes": p.stat().st_size})
manifest = pd.DataFrame(manifest_rows)
manifest.to_csv(PROJ / "10_results" / "final_project_manifest.csv", index=False)
logger.info("Manifest: %d artifacts across %d steps", len(manifest), manifest["step"].nunique())

# ----------------------------------------------------------------------
# 12. Figure: evidence heatmap (top-60 by score)
# ----------------------------------------------------------------------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    topn = evmat.head(60)
    mat = topn[ev_cols].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(10, 14))
    im = ax.imshow(mat, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    ax.set_yticks(range(len(topn)))
    ax.set_yticklabels(topn["gene"].tolist(), fontsize=6)
    ax.set_xticks(range(len(ev_cols)))
    ax.set_xticklabels(["ML", "DL", "Human(context)", "MouseRNA", "MouseATAC"], fontsize=8, rotation=20)
    ax.set_title("Top-60 candidates: evidence layers (Step 20)")
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout()
    fig.savefig(PROJ / "09_figures" / "20_evidence_heatmap.pdf")
    plt.close(fig)
    logger.info("Wrote 09_figures/20_evidence_heatmap.pdf (%d genes)", len(topn))
except Exception as e:  # noqa: BLE001
    logger.warning("Heatmap failed: %s", e)

logger.info("=== Step 20 COMPLETE ===")

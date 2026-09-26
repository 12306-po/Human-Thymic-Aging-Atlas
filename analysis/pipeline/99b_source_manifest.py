#!/usr/bin/env python
"""Emit a revision-source manifest with real code-bundle locations.

The frozen project data must be supplied separately through PROJ. The
manifest documents provenance; it never certifies unseen source tables.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figure_common import FIGF, PROJ  # noqa: E402

N18, N11, N14, NMOUSE = 18, 11, 14, "1 young / 1 aged mouse (descriptive)"

# Absolute paths in this delivered revision bundle; no nonexistent scripts/
# prefix and no reliance on the user's current working directory.
R = str(Path(__file__).resolve().parent) + "/"
F1 = R + "fig_paper/fig1.py"
F2 = R + "fig_paper/fig2.py"
F3 = R + "fig_paper/fig3.py"
F4 = R + "fig_paper/fig4.py"
F5 = R + "fig_paper/fig5.py"
FS1 = R + "fig_paper/fig_supp1.py"
FS2 = R + "fig_paper/fig_supp2.py"
FS3 = R + "fig_paper/fig_supp3.py"
S11 = R + "11_nested_donor_ML.py"
S21 = R + "21_ablation_nested_lodo.py"
S22 = R + "22_age_adjusted_pseudobulk_limma.R"
S23 = R + "23_permutation_null_stability.py"
S24 = R + "24_doublet_removed_sensitivity.py"
S14B = R + "14b_strict_oof_HumanThymusFormer.py"
S20 = R + "20_final_multiomics_integration.py"
S09 = R + "09_age_feature_selection.py"
S07 = R + "07_broad_cell_annotation.py"

ROWS = [
    # Figure 1
    ("Figure1", "A", "manual schematic (05B cohort)",
     F1, "primary", N18, "-", "-", "Figure1.pdf"),
    ("Figure1", "B", "01_raw_processing/metadata/04G_final_cohort_freeze.csv",
     F1, "primary", N18, "age/sex; 14F/4M", "-", "Figure1.pdf"),
    ("Figure1", "C", "01_raw_processing/filtered/06_*QC.h5ad (obs QC)",
     F1, "primary", N18, "donor median genes/UMI/mito%", "-", "Figure1.pdf"),
    ("Figure1", "D", "01_raw_processing/metadata/06_QC_cell_counts_by_donor.csv",
     F1, "primary", N18, "retention % (dumbbell)", "-", "Figure1.pdf"),
    ("Figure1", "E", "02_pseudobulk/donor_celltype_proportion_all.csv",
     F1, "primary", N18, "composition fraction", "-", "Figure1.pdf"),
    ("Figure1", "F", "01_raw_processing/filtered/07_*annotated.h5ad (X_umap)",
     F1, "primary", "114,957 cells / 18 donors", "-", "-", "Figure1.pdf"),
    ("Figure1", "G", "02_pseudobulk/donor_celltype_proportion_all.csv; 03_feature_selection/composition/composition_age_Spearman.csv",
     F1, "primary", N18, "Spearman rho + bootstrap CI", "BH q", "Figure1.pdf"),
    # Figure 2
    ("Figure2", "A", "03_feature_selection/limma_adjusted/whole_thymus_limma_age_sex.csv; output fig2a_top_six_genes_audit.tsv",
     S22, "primary (donor pseudobulk, adjusted for sex)", N18,
     "limma-voom age coefficient/t", "BH", "Figure2.pdf"),
    ("Figure2", "B", "03_feature_selection/age_correlation/whole_thymus_Spearman_by_age.csv",
     S09, "descriptive mixture-level", N18,
     "Spearman rho", "BH", "Figure2.pdf"),
    ("Figure2", "C", "03_feature_selection/limma_adjusted/*_limma_age_sex.csv + limma_celltype_coverage.csv",
     S22, "primary", "10-18 donors/type", "median signed t; sig/tested fraction", "BH", "Figure2.pdf"),
    ("Figure2", "D", "03_feature_selection/limma_adjusted/whole_thymus_limma_age_sex.csv; 02_pseudobulk/donor_matrix/log_normalized_combined.csv.gz",
     S22 + "; " + F2, "descriptive (selection and display same donors)", N18,
     "row z of logCPM (clipped +-2)", "-", "Figure2.pdf"),
    ("Figure2", "E", "03_feature_selection/composition/composition_age_Spearman.csv; 02_pseudobulk/donor_celltype_proportion_all.csv",
     S09 + "; " + F2, "primary", N18,
     "Spearman rho + bootstrap CI", "BH q", "Figure2.pdf"),
    ("Figure2", "F", "03_feature_selection/limma_adjusted/whole_thymus_limma_age_sex.csv; MSigDB Hallmark GMT (offline)",
     F2, "exploratory enrichment", "17,107-gene universe",
     "hypergeometric Count and Count/all age genes; BH over all Hallmark tests", "BH", "Figure2.pdf"),
    # Figure 3
    ("Figure3", "A", "schematic of scripts/11_nested_donor_ML.py",
     S11, "primary", N18, "-", "-", "Figure3.pdf"),
    ("Figure3", "B", "04_machine_learning/predictions/fold_predictions_regression.csv; metrics/regression_metrics_oof*.csv; audit/donor_alignment_audit.tsv",
     S11 + " (donor-alignment fixed 2026-09-17); " + F3,
     "primary LODO OOF", N18, "R2/MAE/rho + donor bootstrap CI", "-", "Figure3.pdf"),
    ("Figure3", "C", "04_machine_learning/metrics/regression_metrics_oof{,_ci}.csv",
     S11, "primary", N18, "OOF R2 + bootstrap CI", "-", "Figure3.pdf"),
    ("Figure3", "D", "04_machine_learning/ablation_nested/ablation_nested_lodo_metrics.csv",
     S21, "primary; identical fold-internal pipeline/block", N18,
     "OOF R2 + paired bootstrap CI", "-", "Figure3.pdf"),
    ("Figure3", "E", "04_machine_learning/permutation_null/gene_recurrence_null.csv (1000 donor-level permutations; full pipeline per perm)",
     S23, "primary + permutation null", N18,
     "observed recurrence vs null-95% upper bound",
     "empirical p + BH; FWER max-stat (threshold 18/18)", "Figure3.pdf"),
    ("Figure3", "F", "04_machine_learning/permutation_null/gene_recurrence_null.csv",
     S23, "exploratory observed recurrence, not corrected feature significance", N18,
     "observed folds (x/18) + direction consistency",
     "0 genes pass BH (min q=0.247) / FWER", "Figure3.pdf"),
    ("Figure3", "G", "04_machine_learning/predictions/fold_predictions_regression.csv; 01_raw_processing/metadata/04G_final_cohort_freeze.csv",
     S11 + "; " + F3, "primary", N18, "signed residual", "-", "Figure3.pdf"),
    # Figure 4
    ("Figure4", "A", "07_external_validation/17_state_donor_counts.csv; output fig4_state_coverage_audit.tsv",
     F4, "independent developmental context (not aging replication)",
     "GSE195812 Sort1+Sort2 cells", "cells and human donors per developmental state", "none", "Figure4.pdf"),
    ("Figure4", "B", "07_external_validation/17_candidate_gene_localization.csv; output fig4_candidate_selection_audit.tsv and fig4_dotplot_source.tsv",
     F4, "exploratory Sort1 localization", "20 displayed genes; Sort1 states",
     "within-Sort1 gene z-score and % expressing", "none", "Figure4.pdf"),
    ("Figure4", "C", "07_external_validation/17_candidate_gene_localization.csv; output fig4_candidate_selection_audit.tsv and fig4_dotplot_source.tsv",
     F4, "exploratory Sort2 localization", "20 displayed genes; Sort2 states",
     "within-Sort2 gene z-score and % expressing", "none", "Figure4.pdf"),
    ("Figure4", "D", "07_external_validation/17_candidate_gene_localization.csv; output fig4_stage_delta_audit.tsv",
     F4, "descriptive within-object contrast", "Sort1 DN1 versus DN3; top 8 plotted genes",
     "DN3 minus DN1 mean within-object z-score; no test", "none", "Figure4.pdf"),
    ("Figure4", "E", "07_external_validation/17_candidate_gene_localization.csv; output fig4_stage_delta_audit.tsv",
     F4, "descriptive within-object contrast", "Sort2 ISP versus DP_CD3plus; top 8 plotted genes",
     "DP_CD3plus minus ISP mean within-object z-score; no test", "none", "Figure4.pdf"),
    # Figure 5
    ("Figure5", "A", "08_mouse_validation/18_mouse_human_orthologs.csv",
     R + "18_mouse_scRNA_cross_species_validation.R", f"descriptive cross-species; {NMOUSE}",
     NMOUSE, "direction concordance (counts only)", "none (no gene-level inference)", "Figure5.pdf"),
    ("Figure5", "B", "08_mouse_validation/18_cross_species_validation.csv",
     R + "18_mouse_scRNA_cross_species_validation.R", f"descriptive; {NMOUSE}",
     NMOUSE, "k/n counts; CI/p not shown", "none", "Figure5.pdf"),
    ("Figure5", "C", "08_mouse_validation/18_mouse_human_orthologs.csv; 19_scATAC_gene_level_join.csv; output fig5_cross_species_mother_audit.tsv",
     F5, f"descriptive; {NMOUSE}", NMOUSE,
     "ATAC direction concordance counts (denominator shared with D)", "none", "Figure5.pdf"),
    ("Figure5", "D", "output fig5_cross_species_mother_audit.tsv; 08_mouse_validation/19_scATAC_peak_validation.csv",
     F5, f"descriptive; {NMOUSE}", NMOUSE,
     "k/n counts; high-confidence = mouse-internal subset", "none", "Figure5.pdf"),
    ("Figure5", "E", "08_mouse_validation/19_TF_motif_peak_target_validation.csv",
     R + "19_mouse_scATAC_validation.R", f"descriptive; {NMOUSE}", NMOUSE,
     "independent network-row summaries; not an attrition funnel", "none", "Figure5.pdf"),
    ("Figure5", "F", "10_results/final_candidate_panel.csv; 05_deep_learning/strict_oof/oof_IG_gene_consistency.csv; output fig5_cross_species_mother_audit.tsv and fig5f_candidate_evidence_audit.tsv",
     S20 + "; " + S14B + "; " + F5, "exploratory candidate evidence availability", N18,
     "rho colourbar; support/discordant/no-support/not-assessed; OOF-IG count x/18", "-", "Figure5.pdf"),
    ("Figure5", "G", "10_results/final_candidate_panel.csv (evidence_tier shown as groups A/B/C)",
     S20, "rule-based exploratory groups", N18, "counts recalculated from candidate mother set", "-", "Figure5.pdf"),
    # Supplementary
    ("Supplementary", "S01", "01_raw_processing/metadata/04G_final_cohort_freeze.csv",
     FS1, "design", N18, "age order; sex markers", "-", "Supplementary_Figure_01_age_task_design.pdf"),
    ("Supplementary", "S02", "01_raw_processing/filtered/07_*annotated.h5ad (X_umap; age/sex/gsm)",
     FS1, "visual batch diagnostic (not a formal mixing test)", N18, "-", "-",
     "Supplementary_Figure_02_batch_diagnostic_UMAP.pdf"),
    ("Supplementary", "S03", "01_raw_processing/filtered/07_*annotated.h5ad (counts layer; gene_symbol)",
     FS1, "marker QC", "114,957 cells", "mean logCPM; fraction expressing (size legend)", "-",
     "Supplementary_Figure_03_marker_dotplot.pdf"),
    ("Supplementary", "S04", "06_scrublet_audit.csv; 10_results/sensitivity_doublet_removed/*",
     R + "06_read_merge_primary_10x_and_QC.py; " + S24, "QC + sensitivity", N18,
     "primary vs doublet-removed model metrics and gene Jaccard recalculated from tables",
     "-", "Supplementary_Figure_04_QC_scrublet_doublet_sensitivity.pdf"),
    ("Supplementary", "S05", "07_annotation_match_rate_by_library.csv; 07_analysis_eligibility_counts.csv; 02_pseudobulk/pseudobulk_qc_summary_all.csv",
     S07, "annotation audit", N18,
     "match rate; counts (excluded counts stated)", "-",
     "Supplementary_Figure_05_annotation_audit.pdf"),
    ("Supplementary", "S06", "04_machine_learning/predictions/fold_predictions_regression.csv; 04_machine_learning/sensitivity_female/**; 04_machine_learning/metrics/regression_permutation_test.csv",
     S11 + "; " + FS2, "primary + female-only sensitivity (independently refit)", f"{N18} / female {N14}",
     "OOF R2/MAE; fixed-OOF association P values omitted from plotted panels",
     "-", "Supplementary_Figure_06_full_ML_female_sensitivity.pdf"),
    ("Supplementary", "S07", "04_machine_learning/metrics/classification_metrics_oof{,_ci}.csv; classification_permutation_test.csv",
     S11, "exploratory binary", N11,
     "AUC CI clipped to [0,1]; sensitivity/specificity and raw zero-specificity counts", "-",
     "Supplementary_Figure_07_exploratory_binary.pdf"),
    ("Supplementary", "S08", "05_deep_learning/strict_oof/dl_strict_oof_metrics.csv; dl_strict_oof_predictions.csv; oof_IG_gene_consistency.csv; dl_posthoc_summary.csv; training_history_per_fold.csv; interpretation/*",
     R + "14_train_HumanThymusFormer.py; " + S14B + "; " + FS3,
     "exploratory DL; strict OOF separate from post-hoc", N18,
     "strict OOF metrics read from source; IG frequency x/18 is descriptive; training curve epochs with >=9 folds",
     "-", "Supplementary_Figure_08_deep_learning.pdf"),
    ("Supplementary", "S08b", "05_deep_learning/predictions/fold_predictions_dl.csv; 05_deep_learning/dl_posthoc_summary.csv",
     FS3, "leakage diagnostic only; not model performance", N18,
     "global-token post-hoc observed/predicted diagnostic", "-",
     "Supplementary_Figure_08b_leakage_diagnostic.pdf"),
    ("Supplementary", "S09", "08_mouse_validation/18_mouse_human_orthologs.csv",
     R + "18_mouse_scRNA_cross_species_validation.R", f"descriptive; {NMOUSE}", NMOUSE,
     "100% stacked direction split; n=257 assessable; counts only", "none",
     "Supplementary_Figure_09_mouse_stage_programs.pdf"),
]

cols = ["figure", "panel", "source_table", "script", "analysis_scope",
        "n_biological_replicates", "statistic", "multiple_testing", "output_pdf"]
df = pd.DataFrame(ROWS, columns=cols)
plot_scripts = {f"Figure{i}": R + f"fig_paper/fig{i}.py" for i in range(1, 6)}
plot_scripts.update({"S01": FS1, "S02": FS1, "S03": FS1,
                     "S04": FS2, "S05": FS2, "S06": FS2,
                     "S07": FS3, "S08": FS3, "S08b": FS3, "S09": FS3})
df["plot_script"] = [plot_scripts[r.figure] if r.figure != "Supplementary"
                     else plot_scripts[r.panel] for r in df.itertuples()]
df["plot_script_exists"] = df.plot_script.map(lambda p: Path(p).is_file())
df["code_bundle_root"] = str(Path(__file__).resolve().parent)
df["frozen_project_root"] = str(PROJ)
df["frozen_project_root_exists"] = PROJ.is_dir()
FIGF.mkdir(parents=True, exist_ok=True)
df.to_csv(FIGF / "figure_source_manifest.tsv", sep="\t", index=False)
print(f"Wrote {FIGF / 'figure_source_manifest.tsv'} ({len(df)} rows)")

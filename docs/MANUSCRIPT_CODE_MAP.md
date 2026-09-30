# Manuscript to code crosswalk

This crosswalk links the manuscript analyses, figures and release objects to the repository code. It distinguishes source-data generation from plotting and website packaging.

| Manuscript component | Primary analysis code | Plot or assembly code | Frozen source or audit output |
|---|---|---|---|
| Cohort audit and donor metadata | `01_inventory_human_data.py`; `02_pair_GSE231906_10x_libraries.py`; `03_parse_GSE231906_GEO_metadata.py`; `04_build_final_thymus_donor_metadata.py`; `25_build_submission_donor_manifest.py` | `fig_paper/fig1.py` | `release/data/donor_summary.csv` and Step 25 outputs |
| QC and broad cell annotation | `06_read_merge_primary_10x_and_QC.py`; `07_broad_cell_annotation.py`; `24_doublet_removed_sensitivity.py` | `fig_paper/fig1.py`; `fig_supp1.py`; `fig_supp2.py` | server-side QC and annotation audits |
| Pseudobulk and captured-library composition | `08_build_pseudobulk_and_composition.py`; `22_age_adjusted_pseudobulk_limma.R`; `27_composition_aware_age.py`; `29_expression_age_shape_sensitivity.R` | `fig_paper/fig2.py` | `gene_context_effects.csv.gz`; `cell_type_summary.csv`; `donor_composition.csv` |
| Feature screening and ML matrix | `09_age_feature_selection.py`; `10_prepare_ML_matrix.py`; `12_integrate_ML_features.py` | — | frozen server matrices and feature audits |
| Fully nested donor-level Elastic Net | `26_fully_nested_donor_ML.py`; `21_ablation_nested_lodo.py`; `23_permutation_null_stability.py`; `31_build_thymic_aging_score.py` | `fig_paper/fig3.py` | `Figure3_final_OOF_source_data.xlsx`; `Figure3_workflow_and_results.png` |
| Frozen HRA007984 external feasibility transfer | `analysis/external_validation/41_freeze_primary_elasticnet.py`; `analysis/external_validation/42_prepare_hra007984_seurat.R`; `analysis/external_validation/43_score_hra007984_frozen.py` | `analysis/external_validation/44_plot_hra007984_validation.py` | run-time donor manifest, feature-coverage audit, frozen model JSON, predictions, metrics and SHA-256 hashes; no results are present until the external object is run |
| Multi-cohort external frozen transfer | `analysis/external_validation/45_prepare_external_h5ad.py`; `analysis/external_validation/46_score_external_frozen.py`; `analysis/external_validation/47_integrate_multicohort_results.py` | `analysis/external_validation/48_plot_multicohort_external.py` | Tabula Sapiens exact-age predictions are combined only with HRA007984; Park interval-age and GSE147520 stromal sensitivity results remain separate |
| HumanThymusFormer attribution | `13_build_HumanThymusFormer_dataset.py`; `14_train_HumanThymusFormer.py`; `14b_strict_oof_HumanThymusFormer.py`; `15_interpret_HumanThymusFormer.py` | `fig_paper/fig_supp3.py` | strict-OOF prediction and attribution tables |
| Pathway and regulatory interpretation | `16_pathway_and_TF_interpretation.py`; `35_human_pathway_age_model.R` | `fig_paper/fig2.py`; `fig_paper/fig5.py` | `gene_pathway_links.csv.gz`; `cell_type_pathway_overlap.csv.gz` |
| Developmental localization using GSE195812 | `17_external_human_validation.R`; `17b_state_donor_counts.R`; `33_build_human_reference_effects.py` | `fig_paper/fig4.py`; `40_plot_figure5_corrected.py` | `developmental_context.csv.gz`; `Figure6_source_data.xlsx` |
| Corrected Sort2 DP CD3min and DP CD3plus panel | `40_plot_figure5_corrected.py` | same script | run-time `Figure5C_source_table.tsv` and `Figure5C_validation.tsv` |
| Mouse RNA and ATAC directional support | `18_mouse_scRNA_cross_species_validation.R`; `18a_ortholog_mapping.py`; `19_mouse_scATAC_validation.R`; `28_cross_species_sex_adjusted.R` | `fig_paper/fig5.py`; `fig_supp3.py` | frozen integrated directional-support tables |
| Multi-omics evidence integration | `20_final_multiomics_integration.py`; `34_cross_dataset_evidence_validation.py` | `fig_paper/fig5.py` | `release/data/gene_summary.csv.gz` and related tables |
| Main and supplementary figures | — | `99_make_paper_figures.py`; `fig_paper/fig1.py`–`fig5.py`; `fig_supp1.py`–`fig_supp3.py`; `100_check_paper_figures.py` | figure-specific audit tables |
| Restrained publication Figures 1–5 | — | `analysis/publication_figures_v3/plot_main_figures_v3.py` | `release/figures_restyled/` and `figure_build_contract.json` |
| Atlas release and website | `32_build_human_thymic_aging_atlas.py`; top-level `build_atlas.py`; `export_server_inputs.py` | top-level `app.py` | all files under `release/` and `SHA256SUMS.txt` |

All unprefixed scripts in the table are located in `analysis/pipeline/`.

## Figure numbering note

Figure numbering changed during manuscript revision. The authoritative link is the scientific content and source workbook, not an older numeric prefix embedded in a legacy filename. The corrected developmental panel is produced by `40_plot_figure5_corrected.py`, which explicitly validates the Sort2 `DP_CD3min` and `DP_CD3plus` columns.

## Reproducibility boundaries

- Scripts require public raw data and/or frozen server-side intermediate files documented in each script.
- Pooled libraries are not counted as independent donors.
- Captured-library fractions must not be interpreted as intact-tissue abundance.
- The 18-donor prediction is internal out-of-fold evaluation, not independent external validation.
- HRA007984 is not called validated until the frozen model is actually applied to verified, eligible independent donors; the supplied transfer code fails closed and currently has no external prediction output.
- The exact-age external summary is prespecified as HRA007984 (four adults) plus Tabula Sapiens v1 (TSP2 and TSP14). Park age-band predictions and the GSE147520 stromal projection are excluded from exact-age performance metrics.
- Mouse results are cross-dataset directional support, not validation of the human prediction model.

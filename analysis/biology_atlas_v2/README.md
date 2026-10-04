# Biology atlas v2

This directory rebuilds the manuscript around donor-resolved thymic biology and the interactive atlas. It does not overwrite the frozen v1 release.

## Outputs

- `release/biology_atlas_v2/data/`: hierarchical annotation audits and exploratory donor-level fraction models.
- `release/biology_atlas_v2/figures/`: five main figures and four supplementary figures in PNG, PDF and SVG.
- `release/biology_atlas_v2/manuscript/`: the regenerated Word manuscript.

## Build figures and source tables

```powershell
D:\anaconda\python.exe analysis\biology_atlas_v2\build_biology_atlas_v2.py `
  --project-root .. `
  --output release\biology_atlas_v2
```

The script reads the annotated primary-cohort h5ad directly with `h5py`, restores the source `id_lv1` to `id_lv4` hierarchy, creates a conservative Level 2 consolidation, and exports all plotted values. Fine-subtype fraction models are exploratory. Their denominator is the recovered lineage-specific cell pool, not intact thymus tissue.

## Evidence boundary

- Level 1 remains the primary donor-level analysis label.
- Level 2 and Level 3 are an atlas and hypothesis-generation layer.
- A Level 3 label is classified as `eligible_for_donor_model`, `exploratory_only`, or `descriptive_only` from predeclared cell and donor-coverage thresholds.
- The nominal 6:4 CD45-positive/CD45-negative recombination prevents native-tissue abundance claims.
- HumanThymusFormer and age-prediction results are supplementary diagnostics, not a clinical clock claim.


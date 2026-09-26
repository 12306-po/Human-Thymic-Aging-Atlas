# Human Thymic Aging Atlas

An interactive, donor-resolved results atlas for continuous-age analyses of the human thymus. The repository contains the Streamlit application, analysis and figure-generation code, frozen release tables, source workbooks, schemas, provenance manifests and checksums used for the manuscript.

## Explore the resource

- **Gene explorer** — age effects, q values, pathways, ML selection, Transformer attribution, developmental localization and mouse directional support.
- **Cell-type explorer** — composition associations, donor coverage, age-up/down genes, pathway overlap and top predictors.
- **Donor explorer** — age, sex, QC, captured-library composition, out-of-fold prediction and age-adjusted deviation.
- **Downloads and provenance** — release tables, source workbooks, manifests and SHA-256 checksums.

## Scope and evidence boundaries

- Primary cohort: **GSE231906**, 18 independent donors, ages 4–69 years.
- Gene-context effects are sex-adjusted donor-level pseudobulk coefficients.
- Composition values are captured-library CLR associations after nominal 6:4 CD45-positive/CD45-negative recombination, not intact-thymus proportions.
- Donor deviation is internally evaluated and descriptive, not a clinically validated biological-age clock.
- GSE195812 provides developmental localization, not independent aging validation.
- Mouse results provide cross-dataset directional support, not external validation of the human model.

## Repository layout

```text
.
├── app.py
├── build_atlas.py
├── export_server_inputs.py
├── analysis/pipeline/
├── docs/MANUSCRIPT_CODE_MAP.md
├── assets/
├── release/data/
├── release/source_files/
├── release/manifest/
├── schemas/
└── requirements.txt
```

## Run the atlas

```powershell
python -m pip install -r requirements.txt
streamlit run app.py
```

On Windows, `run_atlas.ps1` also locates a configured Python installation and launches the app.

## Rebuild the release

```powershell
python build_atlas.py --source-root "F:\论文材料"
```

If the server-only donor composition table has been exported:

```powershell
python build_atlas.py --source-root "F:\论文材料" --donor-composition "path\to\donor_celltype_proportion_all.csv"
```

On the analysis server:

```bash
python export_server_inputs.py \
  --project /data/zxy/projects/human_thymus_age_ML_DL \
  --output /data/zxy/projects/human_thymus_age_ML_DL/10_results/atlas_release_inputs \
  --include-cell-h5ad
```

## Reproduce analyses and figures

The complete numbered pipeline is under `analysis/pipeline/`. It expects the public raw data and frozen server project described in each script. Start with `analysis/pipeline/README_第三轮修订.md` and `docs/MANUSCRIPT_CODE_MAP.md`.

```bash
python analysis/pipeline/99_make_paper_figures.py \
  --project /data/zxy/projects/human_thymus_age_ML_DL \
  --output /data/zxy/projects/human_thymus_age_ML_DL/figures_final_revised \
  --hallmark-gmt /absolute/path/h.all.v2023.2.Hs.symbols.gmt
```

## Main release files

- `gene_summary.csv.gz`: one row per gene.
- `gene_context_effects.csv.gz`: complete gene-by-context age-effect table.
- `cell_type_summary.csv`: composition effect, significant-gene counts and top predictors.
- `cell_type_genes.csv.gz`: age-up and age-down genes by cell type.
- `donor_summary.csv`: donor metadata, QC and out-of-fold predictions.
- `human_thymic_aging_atlas_results.h5ad`: results-level gene × context object; `X` stores beta-age, not expression.
- `source_manifest.csv` and `SHA256SUMS.txt`: provenance and integrity records.

Website image credits and reuse terms are documented in `assets/ATTRIBUTIONS.md`. The images provide visual context only and are not study results.

The manuscript citation and software license should be added by the authors when finalized. No clinical use is intended.

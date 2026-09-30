# Human Thymic Aging Atlas

An interactive, donor-resolved results atlas for continuous-age analyses of the human thymus. The repository contains the Streamlit application, analysis and figure-generation code, frozen release tables, source workbooks, schemas, provenance manifests and checksums used for the manuscript.

## Explore the resource

- **Gene explorer** — age effects, q values, pathways, ML selection, Transformer attribution, developmental localization and mouse directional support.
- **Cell-type explorer** — composition associations, donor coverage, age-up/down genes, pathway overlap and top predictors.
- **Donor explorer** — age, sex, QC, captured-library composition, out-of-fold prediction and age-adjusted deviation.
- **Downloads and provenance** — release tables, source workbooks, manifests and SHA-256 checksums.
- **Contribute data** — session-only CSV validation, cell-type mapping preview and a downloadable review package; no automatic publication.

## Scope and evidence boundaries

- Primary cohort: **GSE231906**, 18 independent donors, ages 4–69 years.
- Gene-context effects are sex-adjusted donor-level pseudobulk coefficients.
- Composition values are captured-library CLR associations after nominal 6:4 CD45-positive/CD45-negative recombination, not intact-thymus proportions.
- Donor deviation is internally evaluated and descriptive, not a clinically validated biological-age clock.
- GSE195812 provides developmental localization, not independent aging validation.
- Mouse results provide cross-dataset directional support, not external validation of the human model.

## Three evidence layers

- **Official Atlas:** frozen `v1.0-18donor` manuscript release. Community files cannot modify `release/` or D-010 main results.
- **External evidence:** separately audited transfer or support. HRA007984 remains supplementary feasibility work, with no external prediction result in this release.
- **Community datasets:** contributor-supplied standardized results. The v1 page only validates and exports a package in the browser session; inclusion requires later human review and a new curated version.

See [Community Submission v1](docs/COMMUNITY_SUBMISSIONS.md) and the [CSV field guide](schemas/README.md). Do not upload controlled-access human data or direct identifiers. There is no server-side submission inbox yet.

## Repository layout

```text
.
├── app.py
├── community_submission.py
├── submission_store.py
├── build_atlas.py
├── export_server_inputs.py
├── analysis/pipeline/
├── analysis/external_validation/
├── analysis/publication_figures_v3/
├── docs/MANUSCRIPT_CODE_MAP.md
├── docs/COMMUNITY_SUBMISSIONS.md
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

Community Submission v1 needs no additional environment variables. It has no
database or object-storage connection. Uploaded bytes and validation output
exist only in the Streamlit session; download the package before leaving the
page. Configure a separate, reviewed persistence service before advertising
this as an actual server intake channel.

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

The restrained publication versions of Figures 1–5 are generated from the
frozen release tables and figure-specific audit tables by
`analysis/publication_figures_v3/plot_main_figures_v3.py`. The program writes
PDF and SVG vector figures plus 600-dpi PNG/TIFF derivatives. UMAP and marker
dot-plot display panels may be rasterized inside the otherwise-vector output;
quantitative values are never digitized from an image.

## Prespecified HRA007984 transfer test

`analysis/external_validation/` contains a fail-closed workflow for the
HRA007984/Zenodo 13207776 processed Seurat object. It freezes the GSE231906
Elastic Net feature order, imputation values, scaling parameters, coefficients
and intercept before external donor ages are used; audits donor metadata and an
author-approved cell-type crosswalk; constructs donor-resolved compatible
features; and applies the frozen model without refitting or calibration.

The code package is **not itself an external-validation result**. The current
release contains no HRA007984 predictions because the 4.4-GB Seurat object has
not yet been executed through the workflow. Follow
`analysis/external_validation/README.md` on the analysis server. The workflow
stops if donor identity, exact age, health status, cohort overlap or frozen-
feature coverage cannot be verified.

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

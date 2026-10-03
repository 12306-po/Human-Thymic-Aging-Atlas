# Human Thymic Aging Atlas

An interactive, donor-resolved results atlas for continuous-age analyses of the human thymus. The repository contains the Streamlit application, analysis and figure-generation code, frozen release tables, source workbooks, schemas, provenance manifests and checksums used for the manuscript:

**Human Thymic Aging Atlas: donor-level continuous-age analysis with nested prediction and interpretable ThymusFormer attribution.**

All paths in this README are repository-relative. Server-side runs use the environment variables exported by `analysis/pipeline/00_project_config.sh` (see [Path conventions](#path-conventions)); no hard-coded absolute paths are used.

## Study summary and key results

- **Cohort**: GSE231906, 18 independent donors aged 4–69 years (14 female / 4 male), 114,957 quality-controlled cells. Each donor library was prepared by sorting CD45-positive and CD45-negative cells and recombining them at a nominal 6:4 ratio; the donor is the independent unit of analysis.
- **Age associations**: sex-adjusted mixed-library pseudobulk analysis identified **2,851 positive and 1,639 negative age coefficients** (BH q < 0.05), evaluated in the sequenced cell mixture and in annotated cell contexts.
- **Nested prediction (Elastic Net, leave-one-donor-out)**: out-of-fold **R² = 0.662, MAE = 9.39 y, RMSE = 11.60 y, Spearman ρ = 0.851**, versus the outer-training-mean null (R² = −0.121, MAE = 18.08 y; ΔMAE = −8.68 y, 95% CI −12.94 to −4.30). Adults (≥18 y, n = 14): R² = 0.559 / MAE = 7.87 y; females (n = 14): R² = 0.593 / MAE = 9.41 y.
- **HumanThymusFormer (interpretable Transformer)**: used for exploratory attribution; strict fold-internal prediction reached R² = 0.13, MAE = 15.0 y (no generalization claim — see [Reproducibility](#humanthymusformer-reproducibility)).
- **External feasibility test (HRA007984)**: prespecified no-refitting transfer to four independent adults (29/39/42/67 y) retained 94.65–96.05% of the frozen feature space; MAE = 11.90 y, RMSE = 16.18 y, ρ = 0.60 (P = 0.4167), R² = −0.337. Reported as a supplementary test of technical transferability, not definitive external validation.
- **Atlas content**: 17,181 genes × 12 contexts, 103,094 records; donor-level deviation is internally evaluated and descriptive (not a clinically validated biological-age clock).

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

## HumanThymusFormer reproducibility

HumanThymusFormer (gene × cell-type token Transformer for donor-level age regression) is fully specified in [HumanThymusFormer_SPEC.md](analysis/pipeline/HumanThymusFormer_SPEC.md), which fixes every hyperparameter in a submission-ready table, mapped to the code lines that implement it:

| Item | Value |
|---|---|
| Architecture | dim=64, 2 layers, 2 heads, FFN=128, dropout=0.2, GELU, post-norm, masked-mean pooling |
| Optimizer / schedule | Adam (lr=1e-3, weight_decay=1e-4, PyTorch defaults for betas/eps) + CosineAnnealingLR |
| Seed | 371 (torch + numpy, all of Steps 14/14b/15) |
| Token construction | two separate logics — global ML-consensus (Step 13, post-hoc only) vs fold-internal (Step 14b, strict OOF) — both capped at 500 tokens |
| Normalization | per-fold fit-donor token/age mean & std; missing tokens reset to 0 |
| Integrated Gradients | captum; zero baseline in standardized space; n_steps=50 (final model) / 20 (fold-wise) |
| Aggregation | token → gene / cell type / gene×cell-type; across-fold median-rank consistency |

Deep-learning dependencies are isolated in `requirements-dl.txt` (torch, captum, scikit-learn, …) so they do not pollute the Streamlit deployment environment in `requirements.txt`.

## Donor metadata audit

Public GEO metadata for GSE231906 provide only age, sex, tissue, preparation and platform. The eight fields needed to test clinical/technical confounding — diagnosis, surgical collection indication, comorbidity, procurement site, procurement method, processing time, library batch and sequencing batch — are **not available in the public metadata** for all 18 donors (verified by pipeline Step 04 and the release `donor_summary.csv`). The donor-level audit table and a ready-to-use Discussion paragraph are provided in:

- [DONOR_METADATA_AUDIT.md](docs/DONOR_METADATA_AUDIT.md) — audit summary, 18-donor table, and English/Chinese Discussion text addressing the age–batch confounding caveat.
- [donor_metadata_audit.csv](docs/donor_metadata_audit.csv) — machine-readable audit table (24 columns × 18 donors).
- [make_donor_metadata_audit.py](docs/make_donor_metadata_audit.py) — regenerates the audit CSV from `release/data/donor_summary.csv`.

## Repository layout

```text
.
├── app.py
├── community_submission.py
├── submission_store.py
├── build_atlas.py
├── export_server_inputs.py
├── requirements.txt              # Streamlit deployment dependencies
├── requirements-dl.txt           # deep-learning reproducibility dependencies
├── analysis/pipeline/            # numbered analysis pipeline (Steps 00–99)
│   └── HumanThymusFormer_SPEC.md # fixed model specification
├── analysis/external_validation/
├── analysis/publication_figures_v3/
├── docs/
│   ├── MANUSCRIPT_CODE_MAP.md
│   ├── COMMUNITY_SUBMISSIONS.md
│   ├── DONOR_METADATA_AUDIT.md   # donor metadata audit + Discussion text
│   └── donor_metadata_audit.csv
├── assets/
├── release/data/
├── release/source_files/
├── release/manifest/
├── schemas/
└── tests/
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

`build_atlas.py` reads the frozen server project outputs (`10_results/…`,
`05_deep_learning/…` and the pseudobulk donor-composition table) from
`--source-root` and writes the publishable release into `--output` (default:
`release/`). On the analysis server, set the project root once via the config
script and pass it as a variable — no absolute path is hard-coded:

```bash
source analysis/pipeline/00_project_config.sh   # exports $PROJ and raw-data paths
python build_atlas.py --source-root "$PROJ" --output release
```

If the server-only donor composition table has been exported separately:

```bash
python build_atlas.py \
  --source-root "$PROJ" \
  --donor-composition "$PROJ/02_pseudobulk/donor_celltype_proportion_all.csv" \
  --output release
```

To export the server-side atlas inputs (including the cell-level h5ad):

```bash
python export_server_inputs.py \
  --project "$PROJ" \
  --output "$PROJ/10_results/atlas_release_inputs" \
  --include-cell-h5ad
```

## Reproduce analyses and figures

The complete numbered pipeline is under `analysis/pipeline/`. It expects the public raw data and the frozen server project whose root is exported by `analysis/pipeline/00_project_config.sh` (`$PROJ`); every script fails fast if the required environment variable is unset. Start with `analysis/pipeline/README_第三轮修订.md` and `docs/MANUSCRIPT_CODE_MAP.md`.

```bash
source analysis/pipeline/00_project_config.sh
python analysis/pipeline/99_make_paper_figures.py \
  --project "$PROJ" \
  --output "$PROJ/figures_final_revised" \
  --hallmark-gmt hallmarks/h.all.v2023.2.Hs.symbols.gmt
```

Place the hallmark GMT file inside the repository (e.g. `hallmarks/`) or point
`--hallmark-gmt` at a local copy; the option accepts a repository-relative path.

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
`analysis/external_validation/README.md` on the analysis server (it uses the
environment variables from `00_project_config.sh` for all server paths). The
workflow stops if donor identity, exact age, health status, cohort overlap or
frozen-feature coverage cannot be verified.

## Path conventions

- **Repository paths are relative** — all links and file references in this
  README and in the repository documentation resolve from the repository root.
- **Server paths come from environment variables**, not from hard-coded
  absolutes. Run `source analysis/pipeline/00_project_config.sh` once per shell;
  it exports `PROJ` (project root) and the raw-data variables
  (`HUMAN_PRIMARY_RAW`, `HUMAN_EXTERNAL_RAW`, `MOUSE_SCRNA_RAW`,
  `MOUSE_SCATAC_RAW`, `MOUSE_DERIVED`). Scripts fail fast when these are unset.
- Never commit machine-specific absolute paths (for example `F:\…` or
  `/data/…`) into this repository; use `$PROJ`-relative or
  repository-relative paths instead.

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

# Multi-cohort import contract

Each donor must have one stable `cohort_id + donor_id` identity. Multiple libraries or sorted fractions from the same donor must not be counted as independent donors.

Before a cohort enters the atlas, verify:

1. exact age and units;
2. sex;
3. healthy, disease and surgical indication fields;
4. whole thymus versus enriched or sorted preparation;
5. platform and chemistry;
6. duplicated donors within and across accessions;
7. post-QC cell count and cell-type coverage;
8. discovery versus frozen-model external-validation role.

Missing or unresolved fields remain explicit. They must not be replaced with assumed values.

## Community Submission v1 CSV contract

The `Contribute data` page accepts exactly three UTF-8 CSV files (10 MiB and
200,000 rows maximum per file). Rename copies of the following templates to
`metadata.csv`, `cell_annotation.csv`, and `gene_age_effects.csv`. No other
columns or file formats are accepted.

| File | Required columns | Meaning |
| --- | --- | --- |
| `metadata.csv` | `donor_id,age_years,sex,study_id,health_status,platform,tissue` | One row per de-identified donor; one study per package; age is years, 0–120. |
| `cell_annotation.csv` | `cell_id,donor_id,cell_type` | One row per de-identified cell; cell IDs unique and donors present in metadata. |
| `gene_age_effects.csv` | `gene,cell_type,beta_age,q_value,n_donors` | One row per gene × cell context; finite beta, q in [0,1], integer n ≥ 3 and no more than annotated donors. |

At least three donors and two distinct ages are required. The cell-type labels
map to the official display vocabulary: `B`, `DC`, `DN`, `DP`, `Fibroblast`,
`GammaDelta_T`, `NKT_like`, `SP_CD4`, `SP_CD8`, `TEC`, `Treg`. The
`whole_thymus` label is allowed only in the gene-effect table. Familiar aliases
are mapped automatically; other labels require explicit selection in the page.
The mapping is a comparison aid, not an assertion that different preparations
or analysis models are equivalent.

`beta_age` must already be the study's age-effect coefficient. State its units,
covariates, test method, and multiple-testing scope when requesting human
review; the site does not recompute models or validate the statistical method.
Only share data you may disclose publicly. Do not enter names, dates of birth,
clinical records, controlled-access human data, raw FASTQ, or single-cell
objects. The site cannot verify de-identification.

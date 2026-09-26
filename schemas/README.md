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

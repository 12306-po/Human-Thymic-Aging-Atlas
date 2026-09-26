#!/usr/bin/env python
"""Step 05: Define and freeze age task cohorts.

Reads Step 04 outputs (04D eligible donors, 04F age distribution) and freezes:
  - PRIMARY task  : continuous age regression cohort (all eligible donors, N=18)
  - SECONDARY task: Young (<18) vs Old (>=40) binary classification cohort (exploratory)
  - Mid-age donors excluded from binary task (used only for regression / sensitivity)
  - Female-only sensitivity cohort (unconditionally frozen, no p-value gate)

Also writes age-by-sex design-level confounding tables (05E/05F/05G).

Outputs (all under 01_raw_processing/metadata/):
  05A_age_distribution_review.csv
  05B_final_regression_cohort.csv
  05C_final_binary_classification_cohort.csv
  05D_excluded_mid_age_for_binary.csv
  05E_age_sex_contingency.csv
  05F_age_by_sex_summary.csv
  05G_female_only_sensitivity_cohort.csv  (unconditional)
  05G_age_sex_sensitivity_plan.txt
  05_age_task_definition.txt

Parameters (predeclared, do not change after running):
  BINARY_YOUNG_MAX_AGE = 18   (strictly < 18)
  BINARY_OLD_MIN_AGE   = 40   (>= 40)
"""

import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Paths & parameters
# ---------------------------------------------------------------------------
def _env(key):
    val = os.environ.get(key)
    if not val:
        raise RuntimeError(f"Required env var {key} is not set. Run: source scripts/00_project_config.sh")
    return Path(val)

PROJ = _env("PROJ")
META = PROJ / "01_raw_processing" / "metadata"
LOGS = PROJ / "10_results" / "logs"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

BINARY_YOUNG_MAX_AGE = 18  # strictly younger than 18
BINARY_OLD_MIN_AGE = 40    # 40 or older

# ---------------------------------------------------------------------------
# Load Step 04 inputs
# ---------------------------------------------------------------------------
donors = pd.read_csv(META / "04D_GSE231906_primary_eligible_donors.csv")
age_dist = pd.read_csv(META / "04F_GSE231906_age_distribution.csv")

assert not donors.empty, "04D eligible donors file is empty"
assert set(["donor_id", "age_years", "sex"]).issubset(donors.columns), donors.columns.tolist()

# Work from the donor-level eligible table; keep descriptive bin for reference.
donors = donors.sort_values("age_years").reset_index(drop=True)
donors["age_years"] = pd.to_numeric(donors["age_years"], errors="coerce")
donors["sex"] = donors["sex"].astype(str).str.strip()

# ---------------------------------------------------------------------------
# Assign task roles
# ---------------------------------------------------------------------------
def assign_task(row):
    age = row["age_years"]
    if age < BINARY_YOUNG_MAX_AGE:
        return "Young_binary"
    if age >= BINARY_OLD_MIN_AGE:
        return "Old_binary"
    return "Mid_age_excluded_from_binary"

donors["binary_task_group"] = donors.apply(assign_task, axis=1)
donors["in_regression_cohort"] = True  # all eligible donors
donors["in_binary_cohort"] = donors["binary_task_group"].isin(["Young_binary", "Old_binary"])

# ---------------------------------------------------------------------------
# 05A: age distribution review (one row per donor, all 18)
# ---------------------------------------------------------------------------
review_cols = [
    "donor_id", "age_years", "sex", "binary_task_group",
    "in_regression_cohort", "in_binary_cohort",
    "age_bin_descriptive", "total_cells_preQC",
]
review = donors[review_cols].copy()
review.to_csv(META / "05A_age_distribution_review.csv", index=False)

# ---------------------------------------------------------------------------
# 05B: regression cohort (all 18)
# ---------------------------------------------------------------------------
reg_cohort = donors[["donor_id", "age_years", "sex", "binary_task_group", "gsm_ids"]].copy()
reg_cohort.to_csv(META / "05B_final_regression_cohort.csv", index=False)

# ---------------------------------------------------------------------------
# 05C: binary classification cohort (Young <18 vs Old >=40)
# ---------------------------------------------------------------------------
bin_cohort = donors[donors["in_binary_cohort"]].copy()
bin_cohort["label_binary"] = np.where(
    bin_cohort["binary_task_group"] == "Young_binary", "Young", "Old"
)
bin_cohort = bin_cohort.sort_values("age_years")
bin_cohort.to_csv(META / "05C_final_binary_classification_cohort.csv", index=False)

# ---------------------------------------------------------------------------
# 05D: excluded mid-age donors (regression only)
# ---------------------------------------------------------------------------
mid_cohort = donors[~donors["in_binary_cohort"]].copy()
mid_cohort.to_csv(META / "05D_excluded_mid_age_for_binary.csv", index=False)

# ---------------------------------------------------------------------------
# 05G (new): female-only sensitivity cohort — UNCONDITIONALLY frozen
# (not gated by any p-value; ML runs later in Step 11 using this cohort)
# ---------------------------------------------------------------------------
female_cohort = donors[donors["sex"].str.upper().str.startswith("F")].copy()
# P0 (2026-09-15): Mid-age female donors must NOT be relabeled Old. Only
# Young_binary -> "Young" and Old_binary -> "Old"; Mid-age -> NA (regression only).
female_cohort["label_binary"] = female_cohort["binary_task_group"].map(
    {"Young_binary": "Young", "Old_binary": "Old"}
)
female_cohort["in_binary_cohort"] = female_cohort["label_binary"].notna()
female_cohort = female_cohort.sort_values("age_years")
female_cohort.to_csv(META / "05G_female_only_sensitivity_cohort.csv", index=False)
log.info(
    "Female-only sensitivity cohort frozen: %d donors (unconditional)",
    len(female_cohort),
)

# ---------------------------------------------------------------------------
# 05E/05F: age-sex design-level confounding
# ---------------------------------------------------------------------------
# 3x2 contingency: age band (Young<18 / Mid / Old>=40) x sex (F/M)
band = donors["binary_task_group"].map(
    {"Young_binary": "Young_<18", "Mid_age_excluded_from_binary": "Mid_18-39", "Old_binary": "Old_>=40"}
)
ct = pd.crosstab(band, donors["sex"])
ct = ct.reindex(index=["Young_<18", "Mid_18-39", "Old_>=40"], columns=["Female", "Male"], fill_value=0)
ct.to_csv(META / "05E_age_sex_contingency.csv")

# Fisher exact test on Young vs Old 2x2 table
def fisher_2x2(df2):
    a = df2.iloc[0, 0]; b = df2.iloc[0, 1]
    c = df2.iloc[1, 0]; d = df2.iloc[1, 1]
    return stats.fisher_exact([[a, b], [c, d]])

f_young_old, p_young_old = fisher_2x2(ct.loc[["Young_<18", "Old_>=40"]])
# NOTE: the 3x2 contingency table is descriptive only; no second Fisher test needed.

# Sex-specific age summaries
sex_sum = (
    donors.groupby("sex")["age_years"]
    .agg(n="count", age_min="min", age_median="median", age_max="max", age_mean="mean", age_std="std")
    .round(2)
)
sex_sum.to_csv(META / "05F_age_by_sex_summary.csv")

# ---------------------------------------------------------------------------
# 05G: sensitivity plan (design level)
# ---------------------------------------------------------------------------
sex_lines = []
for sex, row in sex_sum.iterrows():
    sex_lines.append(
        f"  {sex}: n={int(row['n'])}, age range {row['age_min']}-{row['age_max']}, "
        f"median {row['age_median']}, mean {row['age_mean']} (sd {row['age_std']})"
    )

sensitivity_plan = f"""AGE x SEX SENSITIVITY PLAN (design level, frozen at Step 05)
================================================================================
Cohort: {len(donors)} eligible thymus donors, GSE231906 (GPL24676)

Sex composition of full cohort:
{chr(10).join(sex_lines)}

Age-band x sex contingency (Fisher exact test):
{ct.to_string()}
  Young vs Old 2x2: odds_ratio={f_young_old:.3f}, p={p_young_old:.4f}

Interpretation rule:
- If p < 0.05, age and sex are associated at the design level; sex is a potential
  confounder of the age-outcome relationship and will be handled ONLY via
  sensitivity analysis (never as a primary conclusion).
- Expression-level age-sex sensitivity is performed later in Step 09
  (age_sex_expression_sensitivity.csv).

Handling in ML:
- Sex will NOT be used as a covariate in primary models (too few male donors,
  n=4, to fit reliably).
- Regression and binary models are run on the full frozen cohorts; a
  sensitivity rerun restricted to female donors (n={len(female_cohort)}) is
  ALWAYS produced (unconditionally frozen in 05G_female_only_sensitivity_cohort.csv),
  to check that primary findings are not driven by the small male subset.
- In Step 11, the female-only sensitivity re-runs the ENTIRE model (outer LODO,
  per-fold feature screening, inner tuning, refit); female-only binary may be
  skipped with a warning if either class has <2 donors.
================================================================================
"""
(META / "05G_age_sex_sensitivity_plan.txt").write_text(sensitivity_plan)

# ---------------------------------------------------------------------------
# 05_age_task_definition.txt
# ---------------------------------------------------------------------------
n_young = int((bin_cohort["label_binary"] == "Young").sum())
n_old = int((bin_cohort["label_binary"] == "Old").sum())
task_def = f"""AGE TASK DEFINITION (FROZEN) - Step 05
================================================================================
Date: {pd.Timestamp.now():%Y-%m-%d %H:%M:%S}
Parameters (predeclared, do not change):
  BINARY_YOUNG_MAX_AGE = {BINARY_YOUNG_MAX_AGE}  (strictly younger)
  BINARY_OLD_MIN_AGE   = {BINARY_OLD_MIN_AGE}    (or older)

PRIMARY TASK (continuous age regression)
  Cohort: all {len(reg_cohort)} eligible donors, age range {donors['age_years'].min():.0f}-{donors['age_years'].max():.0f} years
  Response: age_years (continuous)
  This is the main modeling task and the primary source of conclusions.

SECONDARY TASK (binary classification, EXPLORATORY)
  Young: age < {BINARY_YOUNG_MAX_AGE}  -> {n_young} donors
  Old:   age >= {BINARY_OLD_MIN_AGE}   -> {n_old} donors
  Total: {len(bin_cohort)} donors
  Position: exploratory validation of extreme-age discrimination.
  Metrics (AUC etc.) are supportive only and are NOT the primary evidence.

EXCLUDED FROM BINARY (regression / sensitivity only)
  Mid-age donors ({BINARY_YOUNG_MAX_AGE} <= age < {BINARY_OLD_MIN_AGE}): {len(mid_cohort)} donors
  Ages: {sorted(mid_cohort['age_years'].tolist())}

Downstream rules:
  - All ML/DL sample boundaries are donor-level.
  - Cohort files below are frozen; later scripts read these files and never
    re-define labels ad hoc:
      05B_final_regression_cohort.csv
      05C_final_binary_classification_cohort.csv
================================================================================
"""
(META / "05_age_task_definition.txt").write_text(task_def)

# ---------------------------------------------------------------------------
# Summary printout + log
# ---------------------------------------------------------------------------
LOGS.mkdir(parents=True, exist_ok=True)
summary = (
    f"Step 05 done: {len(reg_cohort)} regression donors; "
    f"binary {n_young} Young vs {n_old} Old; {len(mid_cohort)} mid-age excluded; "
    f"age-sex Fisher p={p_young_old:.4f}"
)
print(summary)
with (LOGS / "05_define_age_tasks.log").open("w") as fh:
    fh.write(summary + "\n")
    fh.write(f"fisher_young_old_odds={f_young_old:.4f}\n")
    fh.write(f"fisher_young_old_p={p_young_old:.4f}\n")

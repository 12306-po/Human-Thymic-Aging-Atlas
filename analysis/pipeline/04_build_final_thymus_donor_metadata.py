#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 04
Build clean THYMUS-ONLY donor metadata for the primary GSE231906 cohort.

Important:
1. Use LIBRARY-LEVEL GEO metadata from Step 03B.
2. Remove peripheral blood / PBMC libraries.
3. Keep only thymus libraries.
4. Parse numerical age.
5. Check platform and preparation consistency.
6. One donor remains one biological sample.
7. Do NOT assign final Young/Old groups yet.
"""

from pathlib import Path
import hashlib
import logging
import os
import re
import pandas as pd
import numpy as np

# ============================================================
# Paths — read from environment
# ============================================================

def _env(key):
    val = os.environ.get(key)
    if not val:
        raise RuntimeError(f"Required env var {key} is not set. Run: source scripts/00_project_config.sh")
    return Path(val)

PROJECT = _env("PROJ")
META = PROJECT / "01_raw_processing" / "metadata"
INPUT = META / "03B_GSE231906_primary_library_with_GEO_metadata.csv"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ============================================================
# Read
# ============================================================

if not INPUT.exists():
    raise FileNotFoundError(
        f"Input not found: {INPUT}"
    )

df = pd.read_csv(INPUT)

print("=" * 80)
print("STEP 04: THYMUS-ONLY COHORT CLEANING")
print("=" * 80)

print(
    "Input libraries:",
    len(df)
)

print(
    "Input donors:",
    df["donor_id"].nunique()
)


# ============================================================
# Functions
# ============================================================

def clean_string(x):

    if pd.isna(x):
        return ""

    return str(x).strip()


def parse_age(x):

    """
    Examples:
        56 years -> 56
        5 years  -> 5
        9 years  -> 9
    """

    if pd.isna(x):
        return np.nan

    s = str(x).strip()

    match = re.search(
        r"(\d+(?:\.\d+)?)",
        s
    )

    if not match:
        return np.nan

    return float(
        match.group(1)
    )


def is_thymus(row):

    tissue = clean_string(
        row.get(
            "char_tissue",
            ""
        )
    ).lower()

    source = clean_string(
        row.get(
            "source_name_ch1",
            ""
        )
    ).lower()

    title = clean_string(
        row.get(
            "title",
            ""
        )
    ).lower()

    # Prefer explicit GEO characteristic
    if tissue == "thymus":
        return True

    if tissue in [
        "peripheral blood",
        "blood",
        "pbmc"
    ]:
        return False

    # fallback only when char_tissue missing
    if not tissue:

        if (
            "thymus" in source
            or
            "thymic" in source
            or
            "thymus" in title
        ):
            return True

    return False


# ============================================================
# 1. Tissue audit BEFORE filtering
# ============================================================

print()
print("Raw tissue counts:")

if "char_tissue" in df.columns:

    print(
        df["char_tissue"]
        .fillna("NA")
        .value_counts()
        .to_string()
    )


# ============================================================
# 2. Mark thymus libraries
# ============================================================

df["is_thymus"] = df.apply(
    is_thymus,
    axis=1
)


thymus = df[
    df["is_thymus"]
].copy()


excluded_non_thymus = df[
    ~df["is_thymus"]
].copy()


print()
print(
    "Thymus libraries:",
    len(thymus)
)

print(
    "Non-thymus libraries removed:",
    len(excluded_non_thymus)
)

print(
    "Unique donors represented in thymus:",
    thymus["donor_id"].nunique()
)


# ============================================================
# 3. Parse age
# ============================================================

if "char_age" not in thymus.columns:

    raise RuntimeError(
        "char_age column not found."
    )


thymus["age_years"] = (
    thymus[
        "char_age"
    ]
    .apply(
        parse_age
    )
)


# ============================================================
# 4. Standardize sex
# ============================================================

if "char_sex" in thymus.columns:

    thymus["sex_clean"] = (

        thymus[
            "char_sex"
        ]

        .astype(str)

        .str.strip()

        .str.capitalize()
    )

else:

    thymus["sex_clean"] = "Unknown"


# ============================================================
# 5. Inspect preparation
# ============================================================

def prep_group(x):

    s = clean_string(
        x
    ).lower()

    if (
        "combined" in s
        and
        "cd45+" in s
        and
        "cd45-" in s
    ):

        return "combined_CD45pos_CD45neg_6to4"

    if "cd34" in s:

        return "CD34_enriched"

    if "epcam" in s:

        return "EpCAM_enriched"

    if "pbmc" in s:

        return "PBMC"

    if not s:

        return "Unknown"

    return "Other"


if "char_cell_type" in thymus.columns:

    thymus["preparation_group"] = (

        thymus[
            "char_cell_type"
        ]

        .apply(
            prep_group
        )
    )

else:

    thymus[
        "preparation_group"
    ] = "Unknown"


# ============================================================
# 6. Platform standardization
# ============================================================

if "platform_id" not in thymus.columns:

    thymus[
        "platform_id"
    ] = "Unknown"


# ============================================================
# 7. Primary-cohort eligibility
#
# Do not silently discard samples.
# Explicitly record reasons.
# ============================================================

EXPECTED_PLATFORM = "GPL24676"

EXPECTED_PREP = (
    "combined_CD45pos_CD45neg_6to4"
)


reasons = []


for _, row in thymus.iterrows():

    current = []

    if pd.isna(
        row["age_years"]
    ):

        current.append(
            "missing_age"
        )


    if clean_string(
        row[
            "platform_id"
        ]
    ) != EXPECTED_PLATFORM:

        current.append(
            "different_platform"
        )


    if (
        row[
            "preparation_group"
        ]
        != EXPECTED_PREP
    ):

        current.append(
            "different_preparation"
        )


    reasons.append(
        ";".join(current)
        if current
        else "eligible"
    )


thymus[
    "eligibility_reason"
] = reasons


thymus[
    "primary_eligible"
] = (

    thymus[
        "eligibility_reason"
    ]
    == "eligible"
)


# ============================================================
# 8. Save library-level thymus metadata
# ============================================================

library_out = (
    META
    / "04A_GSE231906_thymus_library_metadata.csv"
)


thymus.to_csv(
    library_out,
    index=False
)


excluded_out = (
    META
    / "04B_GSE231906_non_thymus_libraries_removed.csv"
)


excluded_non_thymus.to_csv(
    excluded_out,
    index=False
)


# ============================================================
# 9. Check thymus libraries per donor
# ============================================================

library_count = (

    thymus

    .groupby(
        [
            "donor_num",
            "donor_id"
        ]
    )

    .size()

    .reset_index(
        name="n_thymus_libraries"
    )
)


print()
print("Thymus libraries per donor:")

print(
    library_count
    .to_string(
        index=False
    )
)


# ============================================================
# 10. Build donor-level table
# ============================================================

donor_rows = []


for (
    donor_num,
    donor_id
), sub in thymus.groupby(
    [
        "donor_num",
        "donor_id"
    ]
):

    ages = (
        sub[
            "age_years"
        ]
        .dropna()
        .unique()
    )

    sexes = (
        sub[
            "sex_clean"
        ]
        .dropna()
        .unique()
    )

    platforms = (
        sub[
            "platform_id"
        ]
        .dropna()
        .astype(str)
        .unique()
    )

    preps = (
        sub[
            "preparation_group"
        ]
        .dropna()
        .astype(str)
        .unique()
    )

    gsm_ids = (
        sub[
            "gsm_id"
        ]
        .dropna()
        .astype(str)
        .tolist()
    )


    if len(ages) == 1:

        age = float(
            ages[0]
        )

    else:

        age = np.nan


    sex = (

        sexes[0]

        if len(sexes) == 1

        else "|".join(
            sorted(
                map(
                    str,
                    sexes
                )
            )
        )
    )


    platform = "|".join(
        sorted(
            map(
                str,
                platforms
            )
        )
    )


    preparation = "|".join(
        sorted(
            map(
                str,
                preps
            )
        )
    )


    donor_eligible = bool(
        sub[
            "primary_eligible"
        ].all()
    )


    eligibility_reasons = "|".join(
        sorted(
            sub[
                "eligibility_reason"
            ]
            .astype(str)
            .unique()
        )
    )


    donor_rows.append({

        "donor_num":
            donor_num,

        "donor_id":
            donor_id,

        "age_years":
            age,

        "sex":
            sex,

        "n_thymus_libraries":
            len(sub),

        "gsm_ids":
            "|".join(
                gsm_ids
            ),

        "platform_id":
            platform,

        "preparation_group":
            preparation,

        "cell_type_original":
            " | ".join(
                sorted(
                    sub[
                        "char_cell_type"
                    ]
                    .dropna()
                    .astype(str)
                    .unique()
                )
            ),

        "tissue":
            "thymus",

        "total_cells_preQC":
            int(
                sub[
                    "n_barcodes"
                ].sum()
            ),

        "primary_eligible":
            donor_eligible,

        "eligibility_reason":
            eligibility_reasons
    })


donor = pd.DataFrame(
    donor_rows
)


# ============================================================
# 10b. Consistency check (age/sex/platform/preparation must be unique per donor)
# ============================================================

# P1 (2026-09-15): each donor must have EXACTLY ONE valid value for
# age/sex/platform/preparation. Zero valid values (all missing/blank/Unknown)
# is NOT consistency — it is flagged as MISSING so the donor is not eligible.
_INVALID_TOKENS = {"", "unknown", "na", "nan", "none", "n/a", "?"}


def _valid_values(group_row, field):
    """Return the unique *valid* values of a field for one donor's rows."""
    series = group_row[field]
    if pd.api.types.is_numeric_dtype(series):
        return series.dropna().unique()
    cleaned = (
        series.dropna().astype(str).str.strip()
        .loc[lambda v: ~v.str.lower().isin(_INVALID_TOKENS)]
    )
    return cleaned.unique()


for _, row in thymus.groupby(["donor_num", "donor_id"]):
    d_id = row["donor_id"].iloc[0]
    conflicts = []
    for field in ["age_years", "sex_clean", "platform_id", "preparation_group"]:
        vals = _valid_values(row, field)
        if len(vals) == 0:
            conflicts.append(f"{field}=MISSING")
        elif len(vals) > 1:
            conflicts.append(f"{field}={vals.tolist()}")
    if conflicts:
        log.warning("Donor %s has conflicting fields: %s — marking donor_consistency_ok=False", d_id, conflicts)
        donor.loc[donor["donor_id"] == d_id, "donor_consistency_ok"] = False
        donor.loc[donor["donor_id"] == d_id, "donor_conflict_reason"] = "; ".join(conflicts)
    else:
        donor.loc[donor["donor_id"] == d_id, "donor_consistency_ok"] = True
        donor.loc[donor["donor_id"] == d_id, "donor_conflict_reason"] = ""

# Mark ineligible if consistency check fails
if "donor_consistency_ok" in donor.columns:
    donor["primary_eligible"] = donor["primary_eligible"] & donor["donor_consistency_ok"].fillna(True)


donor = donor.sort_values(
    "age_years"
)


# ============================================================
# 11. Add descriptive age bins
#
# These are for cohort inspection only.
# NOT yet the final binary ML labels.
# ============================================================

def descriptive_age_bin(age):

    if pd.isna(age):
        return "Unknown"

    if age < 18:
        return "<18"

    if age < 40:
        return "18-39"

    if age < 60:
        return "40-59"

    return ">=60"


donor[
    "age_bin_descriptive"
] = donor[
    "age_years"
].apply(
    descriptive_age_bin
)


# ============================================================
# 12. Save donor table
# ============================================================

donor_out = (
    META
    / "04C_GSE231906_thymus_donor_metadata_candidate.csv"
)


donor.to_csv(
    donor_out,
    index=False
)


eligible = donor[
    donor[
        "primary_eligible"
    ]
].copy()


eligible_out = (
    META
    / "04D_GSE231906_primary_eligible_donors.csv"
)


eligible.to_csv(
    eligible_out,
    index=False
)


excluded_donors = donor[
    ~donor[
        "primary_eligible"
    ]
].copy()


excluded_donor_out = (
    META
    / "04E_GSE231906_excluded_donors.csv"
)


excluded_donors.to_csv(
    excluded_donor_out,
    index=False
)


# ============================================================
# 12b. Frozen cohort freeze files (04G, 04H)
# ============================================================

freeze_out = META / "04G_final_cohort_freeze.csv"
eligible.to_csv(freeze_out, index=False)
log.info("Frozen cohort written to %s (%d donors)", freeze_out, len(eligible))

# SHA-256 of the freeze file for provenance
sha = hashlib.sha256()
with open(freeze_out, "rb") as f:
    while True:
        chunk = f.read(1 << 20)
        if not chunk:
            break
        sha.update(chunk)
sha_out = META / "04H_final_cohort_sha256.txt"
sha_out.write_text(f"{sha.hexdigest()}  {freeze_out.name}\n")
log.info("Cohort SHA-256: %s", sha.hexdigest())


# ============================================================
# 13. Age distribution
# ============================================================

age_table = (

    eligible[
        [
            "donor_num",
            "donor_id",
            "age_years",
            "sex",
            "age_bin_descriptive",
            "n_thymus_libraries",
            "total_cells_preQC"
        ]
    ]

    .sort_values(
        "age_years"
    )
)


age_out = (
    META
    / "04F_GSE231906_age_distribution.csv"
)


age_table.to_csv(
    age_out,
    index=False
)


# ============================================================
# 14. Summary report
# ============================================================

summary = []


summary.append(
    "GSE231906 THYMUS-ONLY COHORT AUDIT"
)

summary.append(
    "=" * 72
)

summary.append(
    f"Input libraries: {len(df)}"
)

summary.append(
    f"Thymus libraries retained: {len(thymus)}"
)

summary.append(
    f"Non-thymus libraries removed: {len(excluded_non_thymus)}"
)

summary.append(
    f"Unique thymus donors: {donor['donor_id'].nunique()}"
)

summary.append(
    f"Primary eligible donors: {len(eligible)}"
)

summary.append(
    f"Excluded donors: {len(excluded_donors)}"
)

summary.append("")


summary.append(
    "Platform counts among thymus libraries:"
)

for key, value in (
    thymus[
        "platform_id"
    ]
    .value_counts(
        dropna=False
    )
    .items()
):

    summary.append(
        f"  {key}: {value}"
    )


summary.append("")


summary.append(
    "Preparation counts among thymus libraries:"
)

for key, value in (
    thymus[
        "preparation_group"
    ]
    .value_counts(
        dropna=False
    )
    .items()
):

    summary.append(
        f"  {key}: {value}"
    )


summary.append("")


summary.append(
    "Age distribution of eligible donors:"
)

for _, row in age_table.iterrows():

    summary.append(
        f"  {row['donor_id']}: "
        f"{row['age_years']} years, "
        f"{row['sex']}, "
        f"{row['total_cells_preQC']} cells"
    )


summary.append("")


summary.append(
    "IMPORTANT:"
)

summary.append(
    "- PBMC/peripheral-blood libraries were removed."
)

summary.append(
    "- One donor is one biological ML sample."
)

summary.append(
    "- Descriptive age bins are NOT yet the final Young/Old labels."
)

summary.append(
    "- Final binary age thresholds will be chosen only after examining "
    "the complete age distribution."
)

summary.append(
    "- Continuous age regression should be retained as an important task."
)


summary_file = (
    META
    / "04_GSE231906_thymus_cohort_audit.txt"
)


summary_file.write_text(
    "\n".join(
        summary
    ),
    encoding="utf-8"
)


# ============================================================
# Console
# ============================================================

print()
print("=" * 80)
print("FINAL THYMUS DONOR CANDIDATES")
print("=" * 80)

print(
    donor[
        [
            "donor_id",
            "age_years",
            "sex",
            "n_thymus_libraries",
            "platform_id",
            "preparation_group",
            "total_cells_preQC",
            "primary_eligible",
            "eligibility_reason"
        ]
    ]
    .to_string(
        index=False
    )
)


print()
print("=" * 80)
print("ELIGIBLE AGE DISTRIBUTION")
print("=" * 80)

print(
    age_table
    .to_string(
        index=False
    )
)


print()
print(
    "Saved:"
)

for f in [
    library_out,
    excluded_out,
    donor_out,
    eligible_out,
    excluded_donor_out,
    age_out,
    summary_file
]:

    print(
        f
    )

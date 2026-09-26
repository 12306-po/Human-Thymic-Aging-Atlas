#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 03
Parse GSE231906 GEO Series Matrix metadata and join it to the already validated
10X libraries.

Inputs
------
/data/zxy/raw_data/GSE231906/GSE231906_matrix/GSE231906-GPL24676_series_matrix.txt.gz
/data/zxy/raw_data/GSE231906/GSE231906_matrix/GSE231906-GPL34284_series_matrix.txt.gz
/data/zxy/projects/human_thymus_age_ML_DL/01_raw_processing/metadata/
    02C_GSE231906_primary_donor1_18_libraries.csv

Outputs
-------
03A_GSE231906_GEO_sample_metadata.csv
03B_GSE231906_primary_library_with_GEO_metadata.csv
03C_GSE231906_primary_donor_metadata_candidate.csv
03D_GSE231906_primary_metadata_conflicts.csv
03E_GSE231906_primary_cohort_screening.csv
03_GSE231906_metadata_summary.txt

Notes
-----
- donor1-18 is still provisional at this step.
- Final training donors must be selected only after checking age, preparation/sorting,
  tissue/source and platform.
- No raw data are modified.
"""

from __future__ import annotations

import csv
import gzip
import logging
import os
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths — read from environment (set by source 00_project_config.sh)
# ---------------------------------------------------------------------------
def _env(key):
    val = os.environ.get(key)
    if not val:
        raise RuntimeError(f"Required env var {key} is not set. Run: source scripts/00_project_config.sh")
    return Path(val)

PROJECT = _env("PROJ")
RAW = _env("HUMAN_PRIMARY_RAW")
META_DIR = PROJECT / "01_raw_processing" / "metadata"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

SERIES_FILES = [
    RAW / "GSE231906_matrix" / "GSE231906-GPL24676_series_matrix.txt.gz",
    RAW / "GSE231906_matrix" / "GSE231906-GPL34284_series_matrix.txt.gz",
]

PRIMARY_LIBRARIES = (
    META_DIR / "02C_GSE231906_primary_donor1_18_libraries.csv"
)

META_DIR.mkdir(parents=True, exist_ok=True)


def clean_value(x: str) -> str:
    x = x.strip()
    if len(x) >= 2 and x[0] == '"' and x[-1] == '"':
        x = x[1:-1]
    return x.replace(r"\"", '"').strip()


def safe_colname(x: str) -> str:
    x = x.strip().lower()
    x = re.sub(r"[^a-z0-9]+", "_", x)
    x = re.sub(r"_+", "_", x).strip("_")
    return x or "unknown"


def parse_series_matrix(path: Path) -> pd.DataFrame:
    """
    Parse all !Sample_* rows from one GEO series-matrix file.
    Repeated rows (especially characteristics_ch1) are retained.
    """
    if not path.exists():
        print(f"[WARN] Missing series matrix: {path}")
        return pd.DataFrame()

    sample_rows = defaultdict(list)

    opener = gzip.open if str(path).endswith(".gz") else open

    with opener(path, "rt", encoding="utf-8", errors="replace", newline="") as fh:
        reader = csv.reader(fh, delimiter="\t", quotechar='"')

        for fields in reader:
            if not fields:
                continue

            key = fields[0].strip()

            if key == "!series_matrix_table_begin":
                break

            if not key.startswith("!Sample_"):
                continue

            field_name = key[len("!Sample_"):]

            values = [clean_value(v) for v in fields[1:]]
            sample_rows[field_name].append(values)

    if "geo_accession" not in sample_rows:
        raise RuntimeError(
            f"No !Sample_geo_accession row was found in {path}"
        )

    gsm_values = sample_rows["geo_accession"][0]
    n = len(gsm_values)

    out = pd.DataFrame({
        "gsm_id": gsm_values,
        "series_matrix_file": path.name,
    })

    # Store all sample fields. Repeated GEO fields get __1, __2, ...
    for field_name, occurrences in sample_rows.items():
        if field_name == "geo_accession":
            continue

        for i, vals in enumerate(occurrences, start=1):
            if len(vals) != n:
                # GEO metadata should align by sample; keep script strict.
                print(
                    f"[WARN] {path.name}: {field_name} occurrence {i} "
                    f"has {len(vals)} values; expected {n}. Skipping."
                )
                continue

            col = safe_colname(field_name)
            if len(occurrences) > 1:
                col = f"{col}__{i}"

            out[col] = vals

    # Convert "key: value" characteristics into explicit char_* columns.
    char_cols = [c for c in out.columns if c.startswith("characteristics_ch1")]

    for char_col in char_cols:
        for row_idx, raw_value in out[char_col].items():
            if not isinstance(raw_value, str) or ":" not in raw_value:
                continue

            k, v = raw_value.split(":", 1)
            k = safe_colname(k)
            v = v.strip()

            if not k:
                continue

            target = f"char_{k}"

            if target not in out.columns:
                out[target] = ""

            old = str(out.at[row_idx, target]).strip()

            if not old:
                out.at[row_idx, target] = v
            elif v and v not in old.split(" | "):
                out.at[row_idx, target] = old + " | " + v

    # Explicit platform fallback from filename if needed.
    platform_from_name = None
    m = re.search(r"(GPL\d+)", path.name, re.I)
    if m:
        platform_from_name = m.group(1).upper()

    if "platform_id" not in out.columns:
        out["platform_id"] = platform_from_name
    else:
        out["platform_id"] = out["platform_id"].replace("", pd.NA)
        out["platform_id"] = out["platform_id"].fillna(platform_from_name)

    return out


def collapse_unique(series: pd.Series) -> str:
    vals = []
    for x in series:
        if pd.isna(x):
            continue
        x = str(x).strip()
        if not x:
            continue
        if x not in vals:
            vals.append(x)
    return " | ".join(vals)


def count_unique_nonempty(series: pd.Series) -> int:
    vals = {
        str(x).strip()
        for x in series
        if not pd.isna(x) and str(x).strip()
    }
    return len(vals)


# ---------------------------------------------------------------------
# 1. Parse GEO metadata from both platforms
# ---------------------------------------------------------------------

geo_parts = [parse_series_matrix(p) for p in SERIES_FILES]
geo_parts = [x for x in geo_parts if not x.empty]

if not geo_parts:
    raise SystemExit("No GEO sample metadata could be parsed.")

geo = pd.concat(geo_parts, ignore_index=True, sort=False)

# Detect duplicate GSM rows across platform files.
CRITICAL_GSM_FIELDS = ["char_age", "char_sex", "char_tissue", "char_cell_type", "platform_id", "source_name_ch1"]

# P1 (2026-09-16): audits from THIS run must never be appended to stale CSVs
# from a previous failed run. Remove the outputs we are about to rebuild and
# collect everything in memory, writing once at the end.
_audit_fnames = ["03F_duplicate_GSM_audit.csv", "03G_critical_metadata_conflicts.csv"]
for _fname in _audit_fnames:
    _p = META_DIR / _fname
    if _p.exists():
        _p.unlink()

dup_gsm = geo[geo["gsm_id"].duplicated(keep=False)].copy()
dup_audit_rows = []
conflict_rows = []
if not dup_gsm.empty:
    log.warning("Duplicate GSM accessions found across series-matrix files.")
    dup_ids = dup_gsm["gsm_id"].unique()
    for gsm in dup_ids:
        rows = geo[geo["gsm_id"] == gsm]
        conflicts = []
        for field in CRITICAL_GSM_FIELDS:
            if field in rows.columns:
                vals = rows[field].dropna().unique().tolist()
                if len(vals) > 1:
                    conflicts.append((field, vals))
        if conflicts:
            conflict_rows.extend([
                {"gsm_id": gsm, "field": f, "values": str(v)}
                for f, v in conflicts
            ])
            log.error("CRITICAL: GSM %s has conflicting critical fields: %s", gsm, conflicts)
        # collect duplicate audit rows for all dup GSM rows
        dup_audit_rows.append(
            rows[["gsm_id", "series_matrix_file"]
                 + [c for c in CRITICAL_GSM_FIELDS if c in rows.columns]].copy()
        )

# Write this run's audits exactly once (overwrite, never append).
if dup_audit_rows:
    pd.concat(dup_audit_rows, ignore_index=True).to_csv(
        META_DIR / "03F_duplicate_GSM_audit.csv", index=False)
if conflict_rows:
    pd.DataFrame(conflict_rows).to_csv(
        META_DIR / "03G_critical_metadata_conflicts.csv", index=False)

# If conflicts found in critical fields, fail-fast (do NOT silently keep first).
if conflict_rows:
    _first = conflict_rows[0]
    raise RuntimeError(
        f"Duplicate GSM {_first['gsm_id']} has conflicting "
        f"'{_first['field']}': {_first['values']}. Resolve manually. "
        "See 03F_duplicate_GSM_audit.csv and 03G_critical_metadata_conflicts.csv."
    )

# Keep one row per GSM only after verifying no critical conflicts.
geo = geo.drop_duplicates(subset=["gsm_id"], keep="first")

geo_out = META_DIR / "03A_GSE231906_GEO_sample_metadata.csv"
geo.to_csv(geo_out, index=False)


# ---------------------------------------------------------------------
# 2. Join validated 10X libraries to GEO metadata
# ---------------------------------------------------------------------

if not PRIMARY_LIBRARIES.exists():
    raise FileNotFoundError(PRIMARY_LIBRARIES)

libs = pd.read_csv(PRIMARY_LIBRARIES)

joined = libs.merge(
    geo,
    how="left",
    on="gsm_id",
    validate="many_to_one",
)

joined["geo_metadata_found"] = joined["series_matrix_file"].notna()

# Useful heuristic ONLY for manual review, never final truth.
review_text_cols = [
    c for c in joined.columns
    if c not in {
        "matrix_path", "features_path", "barcodes_path", "vdj_contig_path"
    }
]

def row_text(row) -> str:
    return " ".join(
        str(row[c])
        for c in review_text_cols
        if c in row.index and not pd.isna(row[c])
    ).lower()

selection_hints = []
thymus_hints = []

for _, row in joined.iterrows():
    text = row_text(row)

    if "unsorted" in text or "unselected" in text:
        selection = "LIKELY_UNSORTED"
    elif re.search(
        r"\bcd34\b|\bepcam\b|sorted|enrich|enriched|facs|macs|purif",
        text,
    ):
        selection = "POSSIBLY_SORTED_OR_ENRICHED"
    else:
        selection = "UNKNOWN"

    if "thymus" in text or "thymic" in text:
        tissue = "LIKELY_THYMUS"
    else:
        tissue = "UNKNOWN"

    selection_hints.append(selection)
    thymus_hints.append(tissue)

joined["selection_review_hint"] = selection_hints
joined["tissue_review_hint"] = thymus_hints

joined_out = META_DIR / "03B_GSE231906_primary_library_with_GEO_metadata.csv"
joined.to_csv(joined_out, index=False)


# ---------------------------------------------------------------------
# 3. Donor-level metadata candidate table
# ---------------------------------------------------------------------

# Metadata columns worth collapsing to donor level.
exclude_cols = {
    "library_prefix",
    "gsm_id",
    "matrix_path",
    "features_path",
    "barcodes_path",
    "vdj_contig_path",
    "mtx_nnz",
    "n_barcodes",
    "n_features",
    "mtx_rows",
    "mtx_cols",
    "matrix_copy_count",
    "features_copy_count",
    "barcodes_copy_count",
    "contig_copy_count",
    "library_within_donor",
}

metadata_cols = [
    c for c in joined.columns
    if c not in exclude_cols
    and c not in {"donor_num", "donor_id"}
]

donor_rows = []
conflict_rows = []

for (donor_num, donor_id), sub in joined.groupby(
    ["donor_num", "donor_id"], dropna=False
):
    row = {
        "donor_num": donor_num,
        "donor_id": donor_id,
        "n_libraries": len(sub),
        "gsm_ids": collapse_unique(sub["gsm_id"]),
        "total_cells_preQC": int(sub["n_barcodes"].sum()),
        "geo_metadata_found_for_all_libraries": bool(
            sub["geo_metadata_found"].all()
        ),
    }

    for c in metadata_cols:
        row[c] = collapse_unique(sub[c])

        # Report conflicts for fields that ideally should be donor-consistent.
        # Sample title/GSM-specific fields are allowed to differ.
        if c in {
            "title",
            "series_matrix_file",
            "selection_review_hint",
        }:
            continue

        n_unique = count_unique_nonempty(sub[c])
        if n_unique > 1:
            conflict_rows.append({
                "donor_num": donor_num,
                "donor_id": donor_id,
                "field": c,
                "n_unique_values": n_unique,
                "values": collapse_unique(sub[c]),
            })

    donor_rows.append(row)

donor_df = pd.DataFrame(donor_rows).sort_values("donor_num")

donor_out = META_DIR / "03C_GSE231906_primary_donor_metadata_candidate.csv"
donor_df.to_csv(donor_out, index=False)

conflicts = pd.DataFrame(conflict_rows)
conflict_out = META_DIR / "03D_GSE231906_primary_metadata_conflicts.csv"
conflicts.to_csv(conflict_out, index=False)


# ---------------------------------------------------------------------
# 4. Compact cohort-screening table
# ---------------------------------------------------------------------

keywords = [
    "age", "sex", "gender", "source", "tissue", "organ",
    "cell", "sort", "selection", "enrich", "cd34", "epcam",
    "platform", "title", "status",
]

screen_cols = [
    "donor_num",
    "donor_id",
    "n_libraries",
    "gsm_ids",
    "total_cells_preQC",
    "geo_metadata_found_for_all_libraries",
]

for c in donor_df.columns:
    lc = c.lower()
    if c in screen_cols:
        continue
    if any(k in lc for k in keywords):
        screen_cols.append(c)

screen_cols = [c for c in screen_cols if c in donor_df.columns]

screen = donor_df[screen_cols].copy()
screen_out = META_DIR / "03E_GSE231906_primary_cohort_screening.csv"
screen.to_csv(screen_out, index=False)


# ---------------------------------------------------------------------
# 5. Summary
# ---------------------------------------------------------------------

summary = []

summary.append("GSE231906 GEO METADATA PARSING SUMMARY")
summary.append("=" * 76)
summary.append(f"GEO samples parsed: {len(geo)}")
summary.append(f"Primary libraries joined: {len(joined)}")
summary.append(
    f"Libraries with GEO metadata: "
    f"{int(joined['geo_metadata_found'].sum())}/{len(joined)}"
)
summary.append(f"Provisional donor1-18 donor rows: {len(donor_df)}")
summary.append(f"Metadata conflict rows: {len(conflicts)}")
summary.append("")
summary.append("IMPORTANT")
summary.append(
    "- 18 donor IDs exist in the filenames, but donor1-18 is still provisional."
)
summary.append(
    "- Final training donors must be selected after checking age, "
    "sorting/preparation, tissue and platform."
)
summary.append(
    "- Multiple libraries from the same donor are technical/library units; "
    "they must never be treated as independent ML samples."
)
summary.append(
    "- filtered_contig_annotations.csv.gz is TCR/V(D)J information, not donor metadata."
)
summary.append(
    "- The GEO series_matrix files are sufficient for metadata parsing; R is not "
    "required at this step."
)

summary_out = META_DIR / "03_GSE231906_metadata_summary.txt"
summary_out.write_text("\n".join(summary), encoding="utf-8")

print("\n".join(summary))
print("\nCompact donor screening table:\n")
print(screen.to_string(index=False))

print("\nOutputs:")
for p in [
    geo_out,
    joined_out,
    donor_out,
    conflict_out,
    screen_out,
    summary_out,
]:
    print(p)

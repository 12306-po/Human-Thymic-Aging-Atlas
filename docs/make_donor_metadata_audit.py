#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regenerate docs/donor_metadata_audit.csv from release/data/donor_summary.csv.

Usage (repo root):  python3 docs/make_donor_metadata_audit.py
Output:            docs/donor_metadata_audit.csv
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "release" / "data" / "donor_summary.csv"
OUT = ROOT / "docs" / "donor_metadata_audit.csv"

MISSING_COLS = [
    "diagnosis", "surgical_collection_indication", "health_comorbidity",
    "procurement_site", "procurement_method", "processing_time",
    "library_batch", "sequencing_batch",
]


def audit_status(v):
    if pd.isna(v) or "not available" in str(v):
        return "NA"
    return "PARSEABLE"


def main():
    df = pd.read_csv(SRC)
    rows = []
    for _, r in df.iterrows():
        row = {
            "donor_id": r["donor_id"],
            "age_years": r["age_years"],
            "sex": r["sex"],
            "platform": r["platform"],
            "sample_preparation": r["sample_preparation"],
            "n_thymus_libraries": r["n_thymus_libraries"],
            "has_replicate_libraries": r["has_replicate_libraries"],
            "total_cells_preQC": r["total_cells_preQC"],
            "total_cells_postQC": r["total_cells_postQC"],
            "included_primary_analysis": r["included_primary_analysis"],
            "exclusion_reason": r["exclusion_reason"],
            "gsm_ids": r["gsm_ids"],
            "outer_fold": r["outer_fold"],
        }
        for c in MISSING_COLS:
            row[f"audit__{c}"] = audit_status(r[c])
        row["n_missing_clinical_batch_fields"] = sum(
            1 for c in MISSING_COLS if row[f"audit__{c}"] == "NA")
        rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print(f"wrote {OUT} ({len(out)} donors)")


if __name__ == "__main__":
    main()
#（注：内容由AI生成）

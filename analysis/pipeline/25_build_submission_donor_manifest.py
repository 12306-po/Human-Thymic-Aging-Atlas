#!/usr/bin/env python
"""Step 25: build a submission-grade donor manifest without inventing metadata.

The script consolidates the frozen 18-donor cohort, library-level GEO fields,
pre/post-QC cell counts, inclusion status, and explicit missing-data reasons.
Unavailable clinical/procurement variables remain NA and are labelled as such.

Outputs (01_raw_processing/metadata/submission_manifest/):
  donor_manifest.csv
  donor_manifest.xlsx                         (when openpyxl is installed)
  donor_manifest_missingness.tsv
  donor_manifest_age_confounding_audit.tsv
  donor_manifest_field_provenance.tsv
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


MISSING_TEXT = "not available in current public metadata"
EMPTY = {"", "na", "n/a", "nan", "none", "unknown", "?", "not reported"}


FIELD_ALIASES = {
    "diagnosis": ("diagnosis", "disease", "disease_state", "health_status"),
    "surgical_collection_indication": (
        "surgical_collection_indication", "surgical_indication",
        "collection_indication", "reason_for_surgery", "procedure",
    ),
    "health_comorbidity": (
        "health_comorbidity", "comorbidity", "comorbidities", "clinical_history",
    ),
    "procurement_site": (
        "procurement_site", "collection_site", "anatomic_site", "sampling_site",
    ),
    "procurement_method": (
        "procurement_method", "collection_method", "tissue_procurement",
    ),
    "processing_time": (
        "processing_time", "processing_delay", "ischemia_time", "time_to_processing",
    ),
    "library_batch": ("library_batch", "library_prep_batch", "batch"),
    "sequencing_batch": ("sequencing_batch", "seq_batch", "run_id"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project",
        type=Path,
        default=Path(os.environ.get("PROJ", "/data/zxy/projects/human_thymus_age_ML_DL")),
    )
    return parser.parse_args()


def clean_values(series: pd.Series) -> list[str]:
    values: list[str] = []
    for value in series.dropna().astype(str):
        value = value.strip()
        if value.lower() not in EMPTY and value not in values:
            values.append(value)
    return values


def collapse_aliases(frame: pd.DataFrame, aliases: tuple[str, ...]) -> tuple[object, str]:
    columns = {column.lower(): column for column in frame.columns}
    for alias in aliases:
        if alias.lower() in columns:
            column = columns[alias.lower()]
            values = clean_values(frame[column])
            if values:
                return "|".join(values), column
    return pd.NA, ""


def bh_adjust(pvalues: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=pvalues.index, dtype=float)
    valid = pvalues.dropna().sort_values()
    if valid.empty:
        return out
    n = len(valid)
    ranked = valid.to_numpy() * n / np.arange(1, n + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out.loc[valid.index] = np.minimum(ranked, 1.0)
    return out


def age_association_audit(manifest: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    age = pd.to_numeric(manifest["age_years"], errors="coerce")
    excluded = {
        "donor_id", "age_years", "included_primary_analysis", "missing_data_reasons",
        "exclusion_reason", "gsm_ids", "replicate_library_ids",
    }
    for column in manifest.columns:
        if column in excluded or column.startswith("missing_reason__") or column.startswith("source__"):
            continue
        values = manifest[column]
        numeric = pd.to_numeric(values, errors="coerce")
        n_numeric = int((numeric.notna() & age.notna()).sum())
        if n_numeric >= 5 and numeric.dropna().nunique() >= 3:
            mask = numeric.notna() & age.notna()
            estimate, pvalue = stats.spearmanr(age[mask], numeric[mask])
            rows.append({
                "covariate": column,
                "method": "Spearman(age, numeric covariate)",
                "n_donors": int(mask.sum()),
                "n_levels": int(numeric[mask].nunique()),
                "effect_or_statistic": float(estimate),
                "p_value": float(pvalue),
                "status": "estimated",
            })
            continue

        categorical = values.astype("string").str.strip()
        categorical = categorical.mask(categorical.str.lower().isin(EMPTY))
        mask = categorical.notna() & age.notna()
        levels = categorical[mask].value_counts()
        if len(levels) >= 2 and int((levels >= 2).sum()) >= 2:
            kept_levels = levels[levels >= 2].index
            groups = [age[mask & categorical.eq(level)].to_numpy() for level in kept_levels]
            statistic, pvalue = stats.kruskal(*groups)
            rows.append({
                "covariate": column,
                "method": "Kruskal-Wallis(age by categorical covariate)",
                "n_donors": int(sum(len(group) for group in groups)),
                "n_levels": len(groups),
                "effect_or_statistic": float(statistic),
                "p_value": float(pvalue),
                "status": "estimated; descriptive only",
            })
        else:
            rows.append({
                "covariate": column,
                "method": "not estimable",
                "n_donors": int(mask.sum()),
                "n_levels": int(levels.size),
                "effect_or_statistic": np.nan,
                "p_value": np.nan,
                "status": "missing, single-level, or insufficient group size",
            })
    audit = pd.DataFrame(rows)
    audit["q_value_BH"] = bh_adjust(audit["p_value"])
    return audit


def main() -> None:
    project = parse_args().project.resolve()
    meta_dir = project / "01_raw_processing" / "metadata"
    output = meta_dir / "submission_manifest"
    output.mkdir(parents=True, exist_ok=True)

    freeze_path = meta_dir / "04G_final_cohort_freeze.csv"
    library_path = meta_dir / "04A_GSE231906_thymus_library_metadata.csv"
    excluded_path = meta_dir / "04E_GSE231906_excluded_donors.csv"
    qc_path = project / "02_pseudobulk" / "pseudobulk_qc_summary_all.csv"
    for required in (freeze_path, library_path):
        if not required.exists():
            raise FileNotFoundError(f"Required input is missing: {required}")

    freeze = pd.read_csv(freeze_path)
    libraries = pd.read_csv(library_path)
    freeze["donor_id"] = freeze["donor_id"].astype(str)
    libraries["donor_id"] = libraries["donor_id"].astype(str)
    if freeze["donor_id"].duplicated().any():
        raise ValueError("04G freeze must contain one row per donor")

    excluded = pd.read_csv(excluded_path) if excluded_path.exists() else pd.DataFrame()
    if not excluded.empty and "donor_id" in excluded:
        excluded["donor_id"] = excluded["donor_id"].astype(str)

    postqc: dict[str, int] = {}
    if qc_path.exists():
        qc = pd.read_csv(qc_path, index_col=0)
        count_columns = [column for column in qc.columns if str(column).startswith("n_cells_")]
        if count_columns:
            postqc = qc[count_columns].sum(axis=1).round().astype(int).to_dict()

    rows: list[dict[str, object]] = []
    provenance: list[dict[str, str]] = []
    for _, frozen in freeze.sort_values("age_years").iterrows():
        donor_id = str(frozen["donor_id"])
        donor_libraries = libraries[libraries["donor_id"].eq(donor_id)]
        if donor_libraries.empty:
            raise ValueError(f"Frozen donor has no library metadata: {donor_id}")

        gsm_column = "gsm_id" if "gsm_id" in donor_libraries else "gsm"
        gsm_ids = clean_values(donor_libraries[gsm_column]) if gsm_column in donor_libraries else []
        row: dict[str, object] = {
            "donor_id": donor_id,
            "gsm_ids": "|".join(gsm_ids) if gsm_ids else pd.NA,
            "age_years": frozen.get("age_years", pd.NA),
            "sex": frozen.get("sex", pd.NA),
            "platform": frozen.get("platform_id", pd.NA),
            "sample_preparation": frozen.get("preparation_group", pd.NA),
            "n_thymus_libraries": int(frozen.get("n_thymus_libraries", len(donor_libraries))),
            "replicate_library_ids": "|".join(gsm_ids) if len(gsm_ids) > 1 else pd.NA,
            "has_replicate_libraries": bool(len(gsm_ids) > 1),
            "total_cells_preQC": frozen.get("total_cells_preQC", pd.NA),
            "total_cells_postQC": postqc.get(donor_id, pd.NA),
            "included_primary_analysis": True,
            "exclusion_reason": pd.NA,
        }
        for field, aliases in FIELD_ALIASES.items():
            value, source = collapse_aliases(donor_libraries, aliases)
            row[field] = value
            row[f"source__{field}"] = source if source else pd.NA
            provenance.append({
                "donor_id": donor_id,
                "field": field,
                "source_column": source,
                "source_file": library_path.name,
                "status": "available" if source else MISSING_TEXT,
            })

        missing_reasons: list[str] = []
        for field in FIELD_ALIASES:
            reason_column = f"missing_reason__{field}"
            if pd.isna(row[field]):
                row[reason_column] = MISSING_TEXT
                missing_reasons.append(f"{field}: {MISSING_TEXT}")
            else:
                row[reason_column] = pd.NA
        if pd.isna(row["total_cells_postQC"]):
            row["missing_reason__total_cells_postQC"] = "Step 08 QC summary not found or unreadable"
            missing_reasons.append("total_cells_postQC: Step 08 QC summary unavailable")
        else:
            row["missing_reason__total_cells_postQC"] = pd.NA
        row["missing_data_reasons"] = "; ".join(missing_reasons) if missing_reasons else pd.NA
        rows.append(row)

    manifest = pd.DataFrame(rows)
    if len(manifest) != 18 or manifest["donor_id"].nunique() != 18:
        raise AssertionError(f"Expected 18 unique frozen donors, found {len(manifest)} rows")

    manifest_path = output / "donor_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    try:
        manifest.to_excel(output / "donor_manifest.xlsx", index=False)
    except ImportError:
        pass

    missingness = []
    for column in manifest.columns:
        missing = int(manifest[column].isna().sum())
        missingness.append({
            "field": column,
            "n_missing": missing,
            "n_available": int(len(manifest) - missing),
            "missing_fraction": missing / len(manifest),
        })
    pd.DataFrame(missingness).to_csv(
        output / "donor_manifest_missingness.tsv", sep="\t", index=False
    )
    pd.DataFrame(provenance).to_csv(
        output / "donor_manifest_field_provenance.tsv", sep="\t", index=False
    )
    age_association_audit(manifest).to_csv(
        output / "donor_manifest_age_confounding_audit.tsv", sep="\t", index=False
    )

    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    metadata = {
        "analysis_unit": "human donor",
        "n_donors": 18,
        "source_freeze": str(freeze_path),
        "source_libraries": str(library_path),
        "post_qc_source": str(qc_path),
        "missing_metadata_policy": "retain NA and explicit missing reason; never infer",
        "donor_manifest_sha256": digest,
    }
    (output / "donor_manifest_audit.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"Wrote {manifest_path} ({len(manifest)} donors; sha256={digest})")


if __name__ == "__main__":
    main()

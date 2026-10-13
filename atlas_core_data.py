"""Corrected, prediction-free data for the three-function atlas website."""
from __future__ import annotations

import gzip
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd

GEO_DONOR17 = "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM8945775"
LEVELS = ["atlas_level1", "atlas_level2", "atlas_level3"]
DONOR_FIELDS = [
    "donor_id", "gsm_ids", "age_years", "sex", "platform", "sample_preparation",
    "n_thymus_libraries", "replicate_library_ids", "has_replicate_libraries",
    "total_cells_preQC", "total_cells_postQC", "included_primary_analysis",
    "exclusion_reason", "diagnosis", "surgical_collection_indication",
    "health_comorbidity", "procurement_site", "procurement_method",
    "processing_time", "library_batch", "sequencing_batch", "missing_data_reasons",
    "age_source_raw", "age_source_url", "age_unit_status",
]


def donor_metadata(source: pd.DataFrame) -> pd.DataFrame:
    required = {"donor_id", "gsm_ids", "age_years", "sex"}
    if not required.issubset(source):
        raise ValueError("Donor metadata lacks donor_id, gsm_ids, age_years or sex")
    frame = source[[column for column in DONOR_FIELDS if column in source]].copy()
    if frame.donor_id.isna().any():
        raise ValueError("Missing donor ID")
    frame["donor_id"] = frame.donor_id.astype(str).str.strip()
    if not frame.donor_id.is_unique:
        raise ValueError("Duplicate donor IDs")
    frame["age_years"] = pd.to_numeric(frame.age_years, errors="raise")
    if not np.isfinite(frame.age_years).all() or frame.age_years.lt(0).any():
        raise ValueError("Invalid donor ages")
    hit = frame.donor_id.eq("donor17")
    if hit.sum() != 1 or "GSM8945775" not in str(frame.loc[hit, "gsm_ids"].iloc[0]):
        raise ValueError("donor17 does not match the audited GEO sample GSM8945775")
    age = float(frame.loc[hit, "age_years"].iloc[0])
    if not (np.isclose(age, 21.0) or np.isclose(age, 1.75)):
        raise ValueError("Unexpected donor17 age; reconcile the source record")
    frame.loc[hit, "age_years"] = 21.0 / 12.0
    if "age_source_raw" not in frame:
        frame["age_source_raw"] = ""
    if "age_source_url" not in frame:
        frame["age_source_url"] = ""
    if "age_unit_status" not in frame:
        frame["age_unit_status"] = "years_in_source_metadata"
    frame.loc[hit, "age_source_raw"] = "21 months"
    frame.loc[hit, "age_source_url"] = GEO_DONOR17
    frame.loc[hit, "age_unit_status"] = "months_converted_to_years"
    return frame.sort_values(["age_years", "donor_id"]).reset_index(drop=True)


def atlas_cells(source: pd.DataFrame, donors: pd.DataFrame) -> pd.DataFrame:
    required = {"donor_id", "umap_1", "umap_2", *LEVELS}
    if not required.issubset(source):
        raise ValueError("Atlas cells lack coordinates, donor IDs or hierarchy labels")
    cells = source[["umap_1", "umap_2", "donor_id", *LEVELS]].copy()
    if cells.donor_id.isna().any():
        raise ValueError("Atlas has missing donor IDs")
    cells["donor_id"] = cells.donor_id.astype(str).str.strip()
    if not set(cells.donor_id).issubset(set(donors.donor_id)):
        raise ValueError("Atlas includes donors absent from the corrected metadata")
    for column in ("umap_1", "umap_2"):
        cells[column] = pd.to_numeric(cells[column], errors="raise")
        if not np.isfinite(cells[column]).all():
            raise ValueError("Atlas has invalid coordinates")
    for column in LEVELS:
        cells[column] = cells[column].astype("object").fillna("Unresolved").astype(str)
    return cells.merge(donors[["donor_id", "age_years", "sex"]], on="donor_id",
                       how="left", validate="many_to_one")


def coverage(cells: pd.DataFrame, level: str) -> pd.DataFrame:
    if level not in LEVELS:
        raise ValueError("Unknown hierarchy level")
    keys = LEVELS[:LEVELS.index(level) + 1]
    return cells.groupby(keys, observed=True, dropna=False).agg(
        n_cells=("donor_id", "size"), n_donors=("donor_id", "nunique"),
        min_age_years=("age_years", "min"), max_age_years=("age_years", "max"),
    ).reset_index().sort_values("n_cells", ascending=False)


def donor_counts(cells: pd.DataFrame) -> pd.DataFrame:
    result = cells.groupby(["donor_id", *LEVELS[:2]], observed=True).size().rename("n_cells").reset_index()
    totals = cells.groupby("donor_id").size()
    result["captured_atlas_denominator"] = result.donor_id.map(totals)
    result["fraction_of_captured_atlas"] = result.n_cells / result.captured_atlas_denominator
    meta = cells[["donor_id", "age_years", "sex"]].drop_duplicates("donor_id")
    return result.merge(meta, on="donor_id", validate="many_to_one")


def load_core(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    canonical = root / "release/core_v1/data/donor_metadata.csv"
    source = canonical if canonical.is_file() else root / "release/data/donor_summary.csv"
    donors = donor_metadata(pd.read_csv(source))
    cells_path = root / "release/biology_atlas_v2/data/hierarchical_umap_source.csv.gz"
    cells = atlas_cells(pd.read_csv(cells_path), donors)
    return donors, cells


def csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False).encode("utf-8")


def correction_manifest() -> dict:
    return {
        "site_functions": ["atlas_browsing", "donor_evidence", "results_and_code_downloads"],
        "donor17": {"gsm_id": "GSM8945775", "source_age": "21 months",
                    "age_years": 1.75, "source_url": GEO_DONOR17},
        "scope": "Corrected donor metadata and descriptive atlas coverage/composition",
        "excluded_from_current_site": ["old_age_predictions", "old_prediction_errors",
                                       "old_age_gaps", "unverified_age_association_results"],
        "composition_scope": "Fractions of captured atlas cells, not intact-tissue abundances",
    }


def result_files(donors: pd.DataFrame, cells: pd.DataFrame) -> dict[str, bytes]:
    files = {
        "release/core_v1/data/donor_metadata.csv": csv_bytes(donors),
        "release/biology_atlas_v2/data/hierarchical_umap_source.csv.gz": gzip.compress(csv_bytes(cells), mtime=0),
        "results/donor_cellstate_counts.csv": csv_bytes(donor_counts(cells)),
    }
    for level in LEVELS:
        files[f"results/{level}_coverage.csv"] = csv_bytes(coverage(cells, level))
    manifest = correction_manifest()
    manifest["file_sha256"] = {name: sha256(data).hexdigest() for name, data in files.items()}
    files["release/core_v1/manifest.json"] = json.dumps(manifest, indent=2, ensure_ascii=False).encode("utf-8")
    return files


def zip_bytes(files: dict[str, bytes]) -> bytes:
    buffer = BytesIO()
    with ZipFile(buffer, "w", compression=ZIP_DEFLATED) as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    return buffer.getvalue()


def source_files(root: Path) -> dict[str, bytes]:
    names = ["app.py", "atlas_core_data.py", "requirements-core.txt", "docs/core_site.md"]
    return {name: (root / name).read_bytes() for name in names}

"""Validation and export for session-only community result submissions.

This module never reads or writes the frozen release. It accepts standardized
result tables only; it does not ingest single-cell objects or raw sequencing data.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import json
import math
import re
from typing import Mapping
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd

MAX_FILE_BYTES = 10 * 1024 * 1024
MAX_ROWS = 200_000
EXPECTED_FILES = ("metadata.csv", "cell_annotation.csv", "gene_age_effects.csv")
REQUIRED = {
    "metadata.csv": {"donor_id", "age_years", "sex", "study_id", "health_status", "platform", "tissue"},
    "cell_annotation.csv": {"cell_id", "donor_id", "cell_type"},
    "gene_age_effects.csv": {"gene", "cell_type", "beta_age", "q_value", "n_donors"},
}
ALLOWED_CELL_TYPES = (
    "B", "DC", "DN", "DP", "Fibroblast", "GammaDelta_T", "NKT_like",
    "SP_CD4", "SP_CD8", "TEC", "Treg",
)
ALIASES = {
    "b cell": "B", "b cells": "B", "dendritic cell": "DC", "dendritic cells": "DC",
    "double negative": "DN", "double positive": "DP", "fibroblasts": "Fibroblast",
    "gamma delta t": "GammaDelta_T", "gd t": "GammaDelta_T", "nkt": "NKT_like",
    "cd4 sp": "SP_CD4", "cd8 sp": "SP_CD8", "tec": "TEC", "tregs": "Treg",
    "regulatory t": "Treg", "whole thymus": "whole_thymus",
}
_CANONICAL = {label.casefold(): label for label in (*ALLOWED_CELL_TYPES, "whole_thymus")}
_FORMULA_PREFIX = re.compile(r"^[\s]*[=+@\-\t\r]")


@dataclass
class ValidationResult:
    issues: list[dict[str, str]]
    tables: dict[str, pd.DataFrame]
    mappings: pd.DataFrame
    summary: dict[str, object]

    @property
    def valid(self) -> bool:
        return not any(issue["level"] == "error" for issue in self.issues)


def _issue(issues: list[dict[str, str]], level: str, file: str, message: str) -> None:
    issues.append({"level": level, "file": file, "message": message})


def read_csv_upload(raw: bytes, filename: str) -> pd.DataFrame:
    if filename not in EXPECTED_FILES:
        raise ValueError("Only the three named standardized CSV result files are accepted.")
    if not raw or len(raw) > MAX_FILE_BYTES or b"\x00" in raw:
        raise ValueError("File is empty, exceeds 10 MiB, or is not a text CSV.")
    try:
        frame = pd.read_csv(BytesIO(raw), dtype=str, encoding="utf-8-sig", keep_default_na=False,
                            nrows=MAX_ROWS + 1, on_bad_lines="error")
    except (UnicodeError, ValueError, OverflowError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise ValueError("Invalid UTF-8 CSV file.") from exc
    if len(frame) > MAX_ROWS:
        raise ValueError(f"More than {MAX_ROWS:,} rows; split or summarize results first.")
    if frame.columns.duplicated().any() or any(not str(c).strip() or str(c).startswith("Unnamed:") for c in frame.columns):
        raise ValueError("CSV headers must be unique and nonempty.")
    frame.columns = [str(c).strip() for c in frame.columns]
    return frame.apply(lambda col: col.str.strip())


def _numbers(frame: pd.DataFrame, column: str, file: str, issues: list[dict[str, str]],
             minimum: float | None = None, maximum: float | None = None,
             integer: bool = False) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce")
    bad = values.isna() | ~values.map(math.isfinite)
    if minimum is not None:
        bad |= values.lt(minimum)
    if maximum is not None:
        bad |= values.gt(maximum)
    if integer:
        bad |= values.ne(values.round())
    if bad.any():
        _issue(issues, "error", file, f"{column}: {int(bad.sum())} missing or invalid values; expected "
               f"{'integer ' if integer else ''}{minimum if minimum is not None else '-infinity'} to "
               f"{maximum if maximum is not None else 'infinity'}.")
    return values


def validate_submission(files: Mapping[str, bytes], manual_map: Mapping[str, str] | None = None) -> ValidationResult:
    issues: list[dict[str, str]] = []
    tables: dict[str, pd.DataFrame] = {}
    manual_map = manual_map or {}
    for name in EXPECTED_FILES:
        if name not in files:
            _issue(issues, "error", name, "Required file is missing.")
            continue
        try:
            frame = read_csv_upload(files[name], name)
        except ValueError as exc:
            _issue(issues, "error", name, str(exc))
            continue
        missing = REQUIRED[name] - set(frame.columns)
        if missing:
            _issue(issues, "error", name, "Missing columns: " + ", ".join(sorted(missing)))
            continue
        extra = set(frame.columns) - REQUIRED[name]
        if extra:
            _issue(issues, "error", name, "Unexpected columns: " + ", ".join(sorted(extra)) + ". Use only the template fields to reduce disclosure risk.")
            continue
        if frame.empty:
            _issue(issues, "error", name, "At least one data row is required.")
            continue
        for column in REQUIRED[name]:
            blank = frame[column].eq("")
            if blank.any():
                _issue(issues, "error", name, f"{column}: {int(blank.sum())} blank values.")
        # Reject spreadsheet formulas in all text fields before CSV download.
        text_columns = set(frame.columns) - {"age_years", "beta_age", "q_value", "n_donors"}
        for column in text_columns:
            if frame[column].str.match(_FORMULA_PREFIX).any():
                _issue(issues, "error", name, f"{column} contains spreadsheet formula-like values.")
        tables[name] = frame

    if "metadata.csv" in tables:
        metadata = tables["metadata.csv"]
        age = _numbers(metadata, "age_years", "metadata.csv", issues, 0, 120)
        metadata["age_years"] = age
        if metadata["donor_id"].duplicated().any():
            _issue(issues, "error", "metadata.csv", "donor_id must be unique within the submission.")
        if metadata["study_id"].nunique() != 1:
            _issue(issues, "error", "metadata.csv", "Use one study_id per submission.")
        if not metadata["sex"].str.lower().isin({"m", "f", "male", "female", "unknown", "not reported"}).all():
            _issue(issues, "warning", "metadata.csv", "Review nonstandard sex labels before manual acceptance.")
        if len(metadata) < 3 or age.nunique() < 2:
            _issue(issues, "error", "metadata.csv", "Age-effect submissions need at least 3 donors and 2 distinct ages.")

    mappings = []
    raw_labels = sorted({label for name in ("cell_annotation.csv", "gene_age_effects.csv")
                         if name in tables for label in tables[name]["cell_type"].unique()})
    for label in raw_labels:
        key = re.sub(r"[\s_-]+", " ", label.strip()).casefold()
        target = manual_map.get(label) or _CANONICAL.get(label.casefold()) or ALIASES.get(key, "")
        if target and target not in (*ALLOWED_CELL_TYPES, "whole_thymus"):
            target = ""
        mappings.append({"submitted_label": label, "mapped_label": target,
                         "method": "manual" if label in manual_map else "automatic" if target else "unresolved"})
    mapping_frame = pd.DataFrame(mappings, columns=["submitted_label", "mapped_label", "method"])
    mapping_dict = dict(zip(mapping_frame["submitted_label"], mapping_frame["mapped_label"]))
    if mapping_frame["mapped_label"].eq("").any():
        _issue(issues, "error", "cell_type", "Unresolved cell-type labels require an explicit mapping.")

    for name in ("cell_annotation.csv", "gene_age_effects.csv"):
        if name in tables:
            tables[name]["cell_type_original"] = tables[name]["cell_type"]
            tables[name]["cell_type"] = tables[name]["cell_type"].map(mapping_dict)
    if "cell_annotation.csv" in tables:
        cells = tables["cell_annotation.csv"]
        if cells["cell_id"].duplicated().any():
            _issue(issues, "error", "cell_annotation.csv", "cell_id must be unique.")
        if cells["cell_type"].eq("whole_thymus").any():
            _issue(issues, "error", "cell_annotation.csv", "whole_thymus is an effect context, not a cell label.")
        if "metadata.csv" in tables:
            missing_donors = set(cells["donor_id"]) - set(tables["metadata.csv"]["donor_id"])
            if missing_donors:
                _issue(issues, "error", "cell_annotation.csv", f"{len(missing_donors)} donor IDs absent from metadata.")
            unrepresented = set(tables["metadata.csv"]["donor_id"]) - set(cells["donor_id"])
            if unrepresented:
                _issue(issues, "warning", "metadata.csv", f"{len(unrepresented)} donors have no annotated cells.")

    if "gene_age_effects.csv" in tables:
        effects = tables["gene_age_effects.csv"]
        effects["beta_age"] = _numbers(effects, "beta_age", "gene_age_effects.csv", issues)
        effects["q_value"] = _numbers(effects, "q_value", "gene_age_effects.csv", issues, 0, 1)
        effects["n_donors"] = _numbers(effects, "n_donors", "gene_age_effects.csv", issues, 3, integer=True)
        if effects.duplicated(["gene", "cell_type"]).any():
            _issue(issues, "error", "gene_age_effects.csv", "Each gene × mapped cell context must occur once.")
        if "metadata.csv" in tables and effects["n_donors"].gt(len(tables["metadata.csv"])).any():
            _issue(issues, "error", "gene_age_effects.csv", "n_donors cannot exceed metadata donor count.")
        if "cell_annotation.csv" in tables:
            labels = set(tables["cell_annotation.csv"]["cell_type"])
            absent = set(effects["cell_type"]) - labels - {"whole_thymus", ""}
            if absent:
                _issue(issues, "error", "gene_age_effects.csv", "Effect contexts missing from cell annotations: " + ", ".join(sorted(absent)))
            covered = tables["cell_annotation.csv"].groupby("cell_type")["donor_id"].nunique()
            for context, group in effects.groupby("cell_type"):
                if context in covered.index and group["n_donors"].gt(covered[context]).any():
                    _issue(issues, "error", "gene_age_effects.csv", f"n_donors exceeds annotated donor coverage for {context}.")

    summary = {
        "donors": len(tables["metadata.csv"]) if "metadata.csv" in tables else 0,
        "age_min": float(tables["metadata.csv"]["age_years"].min()) if "metadata.csv" in tables and tables["metadata.csv"]["age_years"].notna().any() else None,
        "age_max": float(tables["metadata.csv"]["age_years"].max()) if "metadata.csv" in tables and tables["metadata.csv"]["age_years"].notna().any() else None,
        "cells": len(tables["cell_annotation.csv"]) if "cell_annotation.csv" in tables else 0,
        "gene_effects": len(tables["gene_age_effects.csv"]) if "gene_age_effects.csv" in tables else 0,
        "mapped_cell_types": int(mapping_frame["mapped_label"].replace("", pd.NA).nunique()),
    }
    return ValidationResult(issues, tables, mapping_frame, summary)


def make_download_package(submission_id: str, result: ValidationResult) -> bytes:
    if not result.valid:
        raise ValueError("Invalid submissions cannot be exported as normalized results.")
    report = {"submission_id": submission_id, "status": "Pending manual review",
              "receipt": "Local session only; export is not a server submission or publication.",
              "summary": result.summary, "issues": result.issues,
              "official_atlas_modified": False}
    output = BytesIO()
    with ZipFile(output, "w", ZIP_DEFLATED) as archive:
        archive.writestr("validation_report.json", json.dumps(report, indent=2, ensure_ascii=False))
        archive.writestr("cell_type_mapping.csv", result.mappings.to_csv(index=False))
        for name, frame in result.tables.items():
            archive.writestr(name, frame.to_csv(index=False))
    return output.getvalue()

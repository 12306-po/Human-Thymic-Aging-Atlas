"""Strict, session-only inference with a verified frozen T05 donor model."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from io import StringIO
import json
import math
from pathlib import Path
import re

import numpy as np


MAX_UPLOAD_BYTES = 256 * 1024
N_FEATURES = 542
_DONOR_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


@dataclass(frozen=True)
class FrozenAgeModel:
    feature_names: tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray
    coefficients: np.ndarray
    intercept: float
    training_min: np.ndarray
    training_max: np.ndarray
    training_age_min: float
    training_age_max: float
    alpha: float
    reproduction_max_error: float


def load_frozen_model(path: Path) -> FrozenAgeModel:
    """Load JSON only; never deserialize a pickle supplied by a user."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if (data.get("schema_version") != "1.0"
            or data.get("model_name") != "scgpt_global_plus_composition__ridge"
            or data.get("n_training_donors") != 18
            or data.get("n_features") != N_FEATURES
            or data.get("preprocessing") != "StandardScaler"
            or data.get("verified_against_t05c_oof") is not True):
        raise ValueError("Frozen T05 model metadata failed validation")
    names = data.get("feature_names")
    if (not isinstance(names, list) or len(names) != N_FEATURES
            or any(not isinstance(name, str) or not name
                   or any(character in name for character in "\t\r\n")
                   for name in names)
            or "donor_id" in names
            or len(set(names)) != N_FEATURES):
        raise ValueError("Frozen T05 model has invalid feature names")

    def vector(key: str) -> np.ndarray:
        values = np.asarray(data[key], dtype=float)
        if values.shape != (N_FEATURES,) or not np.isfinite(values).all():
            raise ValueError(f"Frozen T05 model has invalid {key}")
        return values

    mean = vector("scaler_mean")
    scale = vector("scaler_scale")
    coefficients = vector("ridge_coefficients")
    training_min = vector("training_feature_min")
    training_max = vector("training_feature_max")
    intercept = float(data["ridge_intercept"])
    alpha = float(data["ridge_alpha"])
    age_min = float(data["training_age_min"])
    age_max = float(data["training_age_max"])
    reproduction_error = float(data["oof_reproduction_max_abs_error_years"])
    scalars = (intercept, alpha, age_min, age_max, reproduction_error)
    if (not all(math.isfinite(value) for value in scalars)
            or np.any(scale <= 0) or np.any(training_min > training_max)
            or alpha <= 0 or not 0 <= age_min < age_max <= 120
            or reproduction_error < 0 or reproduction_error > 1e-3):
        raise ValueError("Frozen T05 model numeric checks failed")
    return FrozenAgeModel(tuple(names), mean, scale, coefficients, intercept,
                          training_min, training_max, age_min, age_max,
                          alpha, reproduction_error)


def feature_template(model: FrozenAgeModel) -> bytes:
    columns = ("donor_id", *model.feature_names)
    return ("\t".join(columns) + "\n" + "\t" * (len(columns) - 1) + "\n").encode("utf-8")


def parse_feature_upload(raw: bytes, filename: str,
                         model: FrozenAgeModel) -> tuple[str, np.ndarray]:
    if not filename.lower().endswith(".tsv"):
        raise ValueError("Upload a UTF-8 .tsv feature table")
    if not raw or len(raw) > MAX_UPLOAD_BYTES or b"\x00" in raw:
        raise ValueError("File is empty, too large, or not a text TSV")
    try:
        rows = list(csv.reader(StringIO(raw.decode("utf-8-sig")), delimiter="\t"))
    except UnicodeError as exc:
        raise ValueError("Feature table must be UTF-8 text") from exc
    if len(rows) != 2:
        raise ValueError("Feature table must have one header and exactly one donor row")
    header, values = rows
    if len(header) != N_FEATURES + 1 or len(set(header)) != len(header):
        raise ValueError("Feature table has duplicate or incorrect columns")
    expected = {"donor_id", *model.feature_names}
    if set(header) != expected or len(values) != len(header):
        missing = expected - set(header)
        extra = set(header) - expected
        raise ValueError(
            f"Feature names do not match the frozen model: "
            f"{len(missing)} missing, {len(extra)} unexpected"
        )
    row = dict(zip(header, values))
    donor_id = row.pop("donor_id")
    if not _DONOR_ID.fullmatch(donor_id):
        raise ValueError("donor_id must be 1–64 ASCII letters, digits, '_' or '-'")
    try:
        features = np.asarray([float(row[name]) for name in model.feature_names], dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("All 542 model features must be numeric and present") from exc
    if not np.isfinite(features).all():
        raise ValueError("Model features must be finite, with no missing values")
    return donor_id, features


def predict_age(model: FrozenAgeModel, features: np.ndarray) -> tuple[float, int]:
    if features.shape != (N_FEATURES,) or not np.isfinite(features).all():
        raise ValueError("Invalid feature vector")
    standardized = (features - model.mean) / model.scale
    predicted = float(np.dot(standardized, model.coefficients) + model.intercept)
    if not math.isfinite(predicted):
        raise ValueError("Model produced a non-finite prediction")
    outside_range = int(np.count_nonzero(
        (features < model.training_min) | (features > model.training_max)
    ))
    return predicted, outside_range

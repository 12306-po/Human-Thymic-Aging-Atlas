import json

import numpy as np
import pytest

from age_prediction_inference import (
    feature_template, load_frozen_model, parse_feature_upload, predict_age,
)


def model_file(tmp_path):
    names = [f"feature_{i:03d}" for i in range(542)]
    bundle = {
        "schema_version": "1.0",
        "model_name": "scgpt_global_plus_composition__ridge",
        "n_training_donors": 18,
        "n_features": 542,
        "preprocessing": "StandardScaler",
        "verified_against_t05c_oof": True,
        "feature_names": names,
        "scaler_mean": [0] * 542,
        "scaler_scale": [1] * 542,
        "ridge_coefficients": [1] + [0] * 541,
        "ridge_intercept": 35,
        "training_feature_min": [-2] * 542,
        "training_feature_max": [2] * 542,
        "training_age_min": 4,
        "training_age_max": 69,
        "ridge_alpha": 100,
        "oof_reproduction_max_abs_error_years": 0.00001,
    }
    path = tmp_path / "frozen_t05_model.json"
    path.write_text(json.dumps(bundle), encoding="utf-8")
    return path


def test_one_donor_inference(tmp_path):
    model = load_frozen_model(model_file(tmp_path))
    header = ["donor_id", *reversed(model.feature_names)]
    values = ["sample-1", *("1" if name == "feature_000" else "0" for name in header[1:])]
    raw = ("\t".join(header) + "\n" + "\t".join(values) + "\n").encode()
    donor, features = parse_feature_upload(raw, "donor_features.tsv", model)
    predicted, outside = predict_age(model, features)
    assert donor == "sample-1"
    assert predicted == pytest.approx(36)
    assert outside == 0
    assert feature_template(model).decode().startswith("donor_id\tfeature_000")


def test_reject_missing_feature_and_second_donor(tmp_path):
    model = load_frozen_model(model_file(tmp_path))
    header = ["donor_id", *model.feature_names]
    row = ["donor_1", *(["0"] * 542)]
    raw = ("\t".join(header) + "\n" + "\t".join(row) + "\n").encode()
    with pytest.raises(ValueError, match="one header and exactly one donor"):
        parse_feature_upload(raw + ("\t".join(row) + "\n").encode(),
                             "donor_features.tsv", model)
    with pytest.raises(ValueError, match="Feature names do not match"):
        parse_feature_upload(raw.replace(b"feature_000", b"wrong_name"),
                             "donor_features.tsv", model)
    with pytest.raises(ValueError, match="finite"):
        parse_feature_upload(raw.replace(b"\t0\n", b"\tNaN\n"),
                             "donor_features.tsv", model)


def test_reject_unverified_model(tmp_path):
    path = model_file(tmp_path)
    bundle = json.loads(path.read_text())
    bundle["verified_against_t05c_oof"] = False
    path.write_text(json.dumps(bundle))
    with pytest.raises(ValueError, match="metadata"):
        load_frozen_model(path)

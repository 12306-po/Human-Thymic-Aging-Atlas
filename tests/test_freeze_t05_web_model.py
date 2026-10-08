import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from age_prediction_inference import load_frozen_model
from freeze_t05_web_model import select_alpha_by_lodo_mae


SCRIPT = Path(__file__).resolve().parents[1] / "freeze_t05_web_model.py"


def write_training_tables(tmp_path):
    rng = np.random.default_rng(371)
    features = rng.normal(size=(18, 542))
    ages = np.linspace(4, 69, 18)
    donors = [f"donor{i}" for i in range(1, 19)]
    feature_names = [f"feature_{i:03d}" for i in range(542)]
    feature_path = tmp_path / "features.tsv.gz"
    outcome_path = tmp_path / "outcomes.tsv"
    oof_path = tmp_path / "oof.tsv"
    pd.DataFrame(features, columns=feature_names).assign(donor_id=donors).to_csv(
        feature_path, sep="\t", index=False,
    )
    pd.DataFrame({"donor_id": donors, "age_years": ages}).to_csv(
        outcome_path, sep="\t", index=False,
    )
    predictions = []
    for held_out in range(18):
        train = np.arange(18) != held_out
        scaler = StandardScaler().fit(features[train])
        regressor = Ridge(alpha=100).fit(scaler.transform(features[train]), ages[train])
        predictions.append(regressor.predict(scaler.transform(features[held_out:held_out + 1]))[0])
    pd.DataFrame({
        "donor_id": donors, "chronological_age": ages,
        "predicted_age": predictions,
        "selected_parameters": [json.dumps({"regressor__alpha": 100.0})] * 18,
    }).to_csv(oof_path, sep="\t", index=False)
    return feature_path, outcome_path, oof_path


def run_freeze(paths, output, alpha_grid=("100",), verify_only=False):
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--features", str(paths[0]),
         "--outcomes", str(paths[1]), "--oof", str(paths[2]),
         "--output", str(output), "--alpha-grid", *alpha_grid,
         *(["--verify-only"] if verify_only else [])],
        capture_output=True, text=True,
    )


def test_verified_model_export(tmp_path):
    paths = write_training_tables(tmp_path)
    output = tmp_path / "model"
    result = run_freeze(paths, output)
    assert result.returncode == 0, result.stderr
    model = load_frozen_model(output / "frozen_t05_model.json")
    assert model.reproduction_max_error < 1e-8
    assert len(model.feature_names) == 542


def test_verify_only_does_not_write_model(tmp_path):
    paths = write_training_tables(tmp_path)
    output = tmp_path / "model"
    result = run_freeze(paths, output, verify_only=True)
    assert result.returncode == 0, result.stderr
    assert "Verification only" in result.stdout
    assert not output.exists()


def test_export_refuses_different_oof_pipeline(tmp_path):
    paths = write_training_tables(tmp_path)
    oof = pd.read_csv(paths[2], sep="\t")
    oof.loc[0, "predicted_age"] += 1
    oof.to_csv(paths[2], sep="\t", index=False)
    output = tmp_path / "model"
    result = run_freeze(paths, output)
    assert result.returncode != 0
    assert "REFUSING TO EXPORT" in result.stderr
    assert not (output / "frozen_t05_model.json").exists()


def test_export_refuses_alpha_outside_reconstructed_grid(tmp_path):
    paths = write_training_tables(tmp_path)
    oof = pd.read_csv(paths[2], sep="\t")
    oof.loc[0, "selected_parameters"] = json.dumps({"regressor__alpha": 1000.0})
    oof.to_csv(paths[2], sep="\t", index=False)
    output = tmp_path / "model"
    result = run_freeze(paths, output)
    assert result.returncode != 0
    assert "outside --alpha-grid" in result.stderr
    assert not (output / "frozen_t05_model.json").exists()


def test_multicandidate_grid_reconstructs_tuning_and_selects_final_alpha(tmp_path):
    paths = write_training_tables(tmp_path)
    x = pd.read_csv(paths[0], sep="\t").drop(columns="donor_id").to_numpy(float)
    y = pd.read_csv(paths[1], sep="\t")["age_years"].to_numpy(float)
    oof = pd.read_csv(paths[2], sep="\t")
    for held_out in range(len(y)):
        train = np.arange(len(y)) != held_out
        alpha, _ = select_alpha_by_lodo_mae(x[train], y[train], (100.0, 1000.0))
        scaler = StandardScaler().fit(x[train])
        model = Ridge(alpha=alpha).fit(scaler.transform(x[train]), y[train])
        oof.loc[held_out, "selected_parameters"] = json.dumps({"regressor__alpha": alpha})
        oof.loc[held_out, "predicted_age"] = model.predict(
            scaler.transform(x[held_out:held_out + 1])
        )[0]
    oof.to_csv(paths[2], sep="\t", index=False)
    output = tmp_path / "model"
    result = run_freeze(paths, output, ("100", "1000"))
    assert result.returncode == 0, result.stderr
    bundle = json.loads((output / "frozen_t05_model.json").read_text())
    assert bundle["alpha_candidate_grid"] == [100.0, 1000.0]
    assert bundle["t05c_outer_fold_alpha_selections_reproduced"] is True
    assert bundle["ridge_alpha"] in (100.0, 1000.0)

import gzip
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path

import pandas as pd
import pytest

from atlas_gene_data import corrected_gene_effects, gene_references, reference_files
from import_corrected_gene_evidence import import_evidence


def test_real_gene_references_exclude_old_age_and_prediction_columns():
    root = Path(__file__).resolve().parents[1]
    index, expression, development = gene_references(root)
    assert {"FOXN1", "CD3D"}.issubset(set(index.gene))
    assert len(index) > 10000
    assert not {"logFC_age_perSD", "P.Value", "adj.P.Val", "beta", "q"} & set(expression)
    assert not {"beta", "q", "predicted_age"} & set(development)
    assert not expression.loc[expression.gene.eq("FOXN1")].empty
    files = reference_files(index, expression, development)
    downloaded = pd.read_csv(BytesIO(gzip.decompress(files[
        "release/core_v1/data/gene_expression_reference.csv.gz"])))
    assert set(downloaded) == set(expression)


def evidence(tmp_path, age=1.75):
    directory = tmp_path / "release/core_v1/gene_evidence"
    directory.mkdir(parents=True)
    frame = pd.DataFrame({"study_id": ["GSE231906"], "gene": ["FOXN1"],
                          "atlas_level2": ["cTEC"], "beta": [0.2], "p": [0.1],
                          "q": [0.2], "n_donors": [7]})
    path = directory / "gene_age_effects.csv.gz"
    frame.to_csv(path, index=False)
    (directory / "manifest.json").write_text(json.dumps({
        "donor17_age_years": age, "effects_sha256": sha256(path.read_bytes()).hexdigest(),
    }), encoding="utf-8")
    return path


def test_old_age_gene_evidence_is_rejected(tmp_path):
    evidence(tmp_path, age=21)
    with pytest.raises(ValueError, match="provenance"):
        corrected_gene_effects(tmp_path)


def test_changed_gene_evidence_is_rejected(tmp_path):
    path = evidence(tmp_path)
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="hash"):
        corrected_gene_effects(tmp_path)


def test_corrected_gene_evidence_is_readable(tmp_path):
    evidence(tmp_path)
    assert corrected_gene_effects(tmp_path).gene.tolist() == ["FOXN1"]


def test_importer_rejects_old_stage_age_before_writing(tmp_path):
    stage = tmp_path / "stage"
    stage.mkdir()
    (stage / "provenance.json").write_text(json.dumps({
        "status": "completed_corrected_age_effects_only", "donor17_corrected_age_years": 21,
    }), encoding="utf-8")
    root = tmp_path / "site"
    with pytest.raises(ValueError, match="1.75"):
        import_evidence(stage, root)
    assert not (root / "release/core_v1/gene_evidence").exists()


def test_importer_rejects_changed_stage_results(tmp_path):
    stage = tmp_path / "stage"
    (stage / "config").mkdir(parents=True)
    (stage / "data").mkdir()
    donors = stage / "config/donor_metadata_master_corrected.csv"
    donors.write_text("donor_id,study_id,exact_age\ndonor17,GSE231906,1.75\n", encoding="utf-8")
    effects = stage / "data/GSE231906_study_specific_age_effects.csv"
    effects.write_text("changed", encoding="utf-8")
    (stage / "provenance.json").write_text(json.dumps({
        "status": "completed_corrected_age_effects_only", "donor17_corrected_age_years": 1.75,
        "new_sha256": {"donors": sha256(donors.read_bytes()).hexdigest(), "gene_effects": "old"},
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="differs"):
        import_evidence(stage, tmp_path / "site")

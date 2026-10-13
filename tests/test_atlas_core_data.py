from io import BytesIO
import gzip
import json
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
import pytest

from atlas_core_data import atlas_cells, coverage, donor_metadata, result_files, zip_bytes


def data():
    donors = pd.DataFrame({"donor_id": ["donor17", "donor3"],
                           "gsm_ids": ["GSM8945775", "GSM8945751"],
                           "age_years": [21., 24.], "sex": ["Female", "Female"],
                           "absolute_error": [1.2, 3.4], "chronological_age": [21., 24.],
                           "elasticnet_OOF_predicted_age": [22., 27.]})
    cells = pd.DataFrame({"donor_id": ["donor17", "donor17", "donor3"],
                          "umap_1": [1., 2., 3.], "umap_2": [2., 3., 4.],
                          "atlas_level1": ["T"] * 3, "atlas_level2": ["DP", "SP", "DP"],
                          "atlas_level3": ["DP_Q", "SP_CD4", "DP_Q"],
                          "age_years": [21., 21., 24.]})
    return donors, cells


def test_one_correct_age_source_for_donors_cells_and_coverage():
    source, source_cells = data()
    donors = donor_metadata(source)
    cells = atlas_cells(source_cells, donors)
    assert donors.loc[donors.donor_id.eq("donor17"), "age_years"].iloc[0] == 1.75
    assert cells.loc[cells.donor_id.eq("donor17"), "age_years"].eq(1.75).all()
    assert coverage(cells, "atlas_level2").min_age_years.min() == 1.75
    assert not {"absolute_error", "chronological_age", "elasticnet_OOF_predicted_age"} & set(donors)
    assert source.loc[0, "age_years"] == 21.


def test_reject_wrong_gsm_instead_of_silently_correcting():
    source, _ = data()
    source.loc[0, "gsm_ids"] = "GSM_WRONG"
    with pytest.raises(ValueError, match="does not match"):
        donor_metadata(source)


def test_reject_unrecognized_donor17_age():
    source, _ = data()
    source.loc[0, "age_years"] = 30
    with pytest.raises(ValueError, match="Unexpected donor17 age"):
        donor_metadata(source)


def test_download_bundle_is_consistent_and_excludes_old_model_columns():
    source, source_cells = data()
    donors = donor_metadata(source)
    cells = atlas_cells(source_cells, donors)
    with ZipFile(BytesIO(zip_bytes(result_files(donors, cells)))) as archive:
        downloaded = pd.read_csv(BytesIO(archive.read("release/core_v1/data/donor_metadata.csv")))
        downloaded_cells = pd.read_csv(BytesIO(gzip.decompress(archive.read(
            "release/biology_atlas_v2/data/hierarchical_umap_source.csv.gz"))))
        assert downloaded.loc[downloaded.donor_id.eq("donor17"), "age_years"].iloc[0] == 1.75
        assert downloaded_cells.loc[downloaded_cells.donor_id.eq("donor17"), "age_years"].eq(1.75).all()
        assert "elasticnet_OOF_predicted_age" not in downloaded
        manifest = json.loads(archive.read("release/core_v1/manifest.json"))
        assert manifest["donor17"]["age_years"] == 1.75


def test_real_app_has_only_three_tabs_and_correct_donor_age():
    from streamlit.testing.v1 import AppTest
    root = Path(__file__).resolve().parents[1]
    app = AppTest.from_file(str(root / "app.py"), default_timeout=120).run()
    assert not app.exception, str(app.exception)
    assert len(app.tabs) == 3
    assert [tab.label for tab in app.tabs] == [
        "图谱浏览 · Atlas", "供者级证据查询 · Donors", "结果与代码下载 · Downloads"]
    assert any(metric.label == "年龄" and metric.value == "1.75 岁" for metric in app.metric)
    assert not any("predicted" in metric.label.lower() for metric in app.metric)
    assert not app.get("file_uploader")
    next(radio for radio in app.radio if radio.label == "图谱着色").set_value("年龄").run()
    assert not app.exception, str(app.exception)
    next(selector for selector in app.selectbox if selector.label == "选择供者").set_value("donor3").run()
    assert any(metric.label == "年龄" and metric.value == "24 岁" for metric in app.metric)
    assert not app.exception, str(app.exception)
    app.multiselect[0].set_value([]).run()
    assert not app.exception, str(app.exception)
    next(radio for radio in app.radio if radio.label == "浏览内容").set_value("基因查询").run()
    assert not app.exception, str(app.exception)
    assert len(app.tabs) == 3
    gene_selector = next(selector for selector in app.selectbox if selector.label == "基因符号")
    assert gene_selector.value == "CD3D"
    gene_selector.set_value("FOXN1").run()
    assert not app.exception, str(app.exception)
    assert len(app.get("imgs")) >= 2

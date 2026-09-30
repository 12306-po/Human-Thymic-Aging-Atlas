from __future__ import annotations

from io import BytesIO
import json
import unittest
from zipfile import ZipFile

from community_submission import make_download_package, validate_submission


def sample_files():
    return {
        "metadata.csv": b"donor_id,age_years,sex,study_id,health_status,platform,tissue\nD1,4,F,S1,healthy,10x,thymus\nD2,30,M,S1,healthy,10x,thymus\nD3,69,F,S1,healthy,10x,thymus\n",
        "cell_annotation.csv": b"cell_id,donor_id,cell_type\nC1,D1,B cell\nC2,D2,B cell\nC3,D3,B cell\n",
        "gene_age_effects.csv": b"gene,cell_type,beta_age,q_value,n_donors\nFOXN1,B cell,-0.5,0.02,3\n",
    }


class CommunitySubmissionTests(unittest.TestCase):
    def test_valid_package_is_separate_and_mapped(self):
        result = validate_submission(sample_files())
        self.assertTrue(result.valid, result.issues)
        self.assertEqual(result.tables["cell_annotation.csv"]["cell_type"].tolist(), ["B"] * 3)
        self.assertEqual(result.summary["donors"], 3)
        with ZipFile(BytesIO(make_download_package("CSA-TEST", result))) as archive:
            report = json.loads(archive.read("validation_report.json"))
            self.assertEqual(report["status"], "Pending manual review")
            self.assertFalse(report["official_atlas_modified"])
            self.assertIn(b"cell_type_original", archive.read("cell_annotation.csv"))

    def test_invalid_effect_and_donor_coverage_block_export(self):
        files = sample_files()
        files["gene_age_effects.csv"] = files["gene_age_effects.csv"].replace(b"-0.5,0.02,3", b"inf,1.2,4")
        result = validate_submission(files)
        self.assertFalse(result.valid)
        with self.assertRaises(ValueError):
            make_download_package("CSA-TEST", result)

    def test_unknown_labels_require_explicit_mapping(self):
        files = sample_files()
        files["cell_annotation.csv"] = files["cell_annotation.csv"].replace(b"B cell", b"Novel cell")
        files["gene_age_effects.csv"] = files["gene_age_effects.csv"].replace(b"B cell", b"Novel cell")
        self.assertFalse(validate_submission(files).valid)
        self.assertTrue(validate_submission(files, {"Novel cell": "B"}).valid)

    def test_formula_and_extra_columns_are_rejected(self):
        files = sample_files()
        files["metadata.csv"] = files["metadata.csv"].replace(b"D1,4", b"=1+1,4")
        self.assertFalse(validate_submission(files).valid)
        files = sample_files()
        files["metadata.csv"] = files["metadata.csv"].replace(b"tissue\n", b"tissue,patient_name\n")
        self.assertFalse(validate_submission(files).valid)


if __name__ == "__main__":
    unittest.main()

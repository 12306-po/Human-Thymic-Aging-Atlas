"""Exercise the deployed page flow, including upload and clearing state."""

from __future__ import annotations

from pathlib import Path
import unittest

from streamlit.testing.v1 import AppTest


APP = Path(__file__).resolve().parents[1] / "app.py"
FILES = {
    "metadata.csv": (
        b"donor_id,age_years,sex,study_id,health_status,platform,tissue\n"
        b"D1,4,F,S1,healthy,10x,thymus\n"
        b"D2,30,M,S1,healthy,10x,thymus\n"
        b"D3,69,F,S1,healthy,10x,thymus\n"
    ),
    "cell_annotation.csv": (
        b"cell_id,donor_id,cell_type\n"
        b"C1,D1,Novel cell\nC2,D2,Novel cell\nC3,D3,Novel cell\n"
    ),
    "gene_age_effects.csv": (
        b"gene,cell_type,beta_age,q_value,n_donors\n"
        b"FOXN1,Novel cell,-0.5,0.02,3\n"
    ),
}


class AppSubmissionTests(unittest.TestCase):
    def test_upload_mapping_change_and_clear(self):
        app = AppTest.from_file(str(APP)).run(timeout=120)
        self.assertFalse(app.exception)
        self.assertIn("Contribute data", [tab.label for tab in app.tabs])

        for uploader in app.file_uploader:
            uploader.set_value((uploader.label, FILES[uploader.label], "text/csv"))
        app.run(timeout=120)
        self.assertFalse(app.exception)
        next(button for button in app.button if button.label == "Clear this session's submission").click()
        app.run(timeout=120)
        self.assertTrue(all(uploader.value is None for uploader in app.file_uploader))
        for uploader in app.file_uploader:
            uploader.set_value((uploader.label, FILES[uploader.label], "text/csv"))
        app.run(timeout=120)

        mapping = next(box for box in app.selectbox if box.label == "Map Novel cell")
        mapping.set_value("B")
        app.checkbox[0].set_value(True)
        app.run(timeout=120)
        next(button for button in app.button if button.label == "Validate and prepare submission package").click()
        app.run(timeout=120)
        self.assertFalse(app.exception)
        self.assertTrue(any(button.label == "Clear this session's submission" for button in app.button))
        self.assertTrue(any(button.label == "Download normalized results and validation report"
                            for button in app.download_button))

        next(box for box in app.selectbox if box.label == "Map Novel cell").set_value("SP_CD4")
        app.run(timeout=120)
        self.assertFalse(app.exception)
        self.assertFalse(any(button.label == "Download normalized results and validation report"
                             for button in app.download_button))

        next(button for button in app.button if button.label == "Validate and prepare submission package").click()
        app.run(timeout=120)
        next(button for button in app.button if button.label == "Clear this session's submission").click()
        app.run(timeout=120)
        self.assertFalse(app.exception)
        self.assertTrue(all(uploader.value is None for uploader in app.file_uploader))
        self.assertFalse(app.checkbox[0].value)
        self.assertFalse(any(button.label == "Download normalized results and validation report"
                             for button in app.download_button))


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HAS_ANALYSIS_STACK = all(
    importlib.util.find_spec(name) is not None
    for name in ("numpy", "pandas", "scipy", "sklearn")
)


def load_script(name: str):
    path = ROOT / name
    spec = importlib.util.spec_from_file_location(name.replace(".", "_"), path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class SubmissionCompletionContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if HAS_ANALYSIS_STACK:
            cls.ml = load_script("26_fully_nested_donor_ML.py")
            cls.comp = load_script("27_composition_aware_age.py")

    @unittest.skipUnless(HAS_ANALYSIS_STACK, "server analysis stack is not installed locally")
    def test_fully_nested_inner_screening_and_oof_identity(self):
        import numpy as np
        import pandas as pd
        from sklearn.linear_model import ElasticNet

        rng = np.random.default_rng(12)
        donors = [f"d{i}" for i in range(9)]
        columns = [f"gc_{i}" for i in range(12)] + [f"wt_{i}" for i in range(8)] + [f"p_{i}" for i in range(4)]
        features = pd.DataFrame(rng.normal(size=(9, len(columns))), index=donors, columns=columns)
        ages = np.linspace(5, 65, 9) + rng.normal(scale=1.0, size=9)
        blocks = {
            "all_features": columns,
            "gene_celltype_only": columns[:12],
            "whole_thymus_only": columns[12:20],
            "composition_only": columns[20:],
        }
        models = {
            "ElasticNet": (
                ElasticNet(max_iter=5000, random_state=self.ml.SEED),
                {"alpha": [0.1], "l1_ratio": [0.5]},
            )
        }
        predictions, _, inner, _, ablation = self.ml.run_nested(
            features, ages, blocks, models, top_k=5,
            cohort="synthetic", run_ablation=True,
        )
        elastic = predictions[predictions["model"].eq("ElasticNet")]
        self.assertEqual(len(elastic), 9)
        self.assertEqual(elastic["donor_id"].nunique(), 9)
        screening = inner[inner["screening_scope"].notna()]
        self.assertTrue(screening["screening_scope"].eq("inner-training donors only").all())
        self.assertEqual(ablation["feature_block"].nunique(), 4)

    @unittest.skipUnless(HAS_ANALYSIS_STACK, "server analysis stack is not installed locally")
    def test_clr_is_closed_and_zero_safe(self):
        import numpy as np
        import pandas as pd

        proportions = pd.DataFrame(
            [[0.0, 0.4, 0.6], [0.2, 0.0, 0.8], [0.1, 0.3, 0.6]],
            index=["d1", "d2", "d3"], columns=["A", "B", "C"],
        )
        totals = pd.Series([100, 200, 300], index=proportions.index)
        replaced = self.comp.zero_replace_and_close(proportions, totals)
        self.assertTrue(np.allclose(replaced.sum(axis=1), 1.0))
        self.assertTrue((replaced.to_numpy() > 0).all())
        transformed = self.comp.clr(replaced)
        self.assertTrue(np.allclose(transformed.sum(axis=1), 0.0))

    def test_cross_species_script_uses_adjusted_beta_and_no_animal_test(self):
        text = (ROOT / "28_cross_species_sex_adjusted.R").read_text(encoding="utf-8")
        self.assertIn("logFC_age_perSD", text)
        self.assertIn("descriptive counts only; no animal-level test", text)
        self.assertNotIn("binom.test", text)


if __name__ == "__main__":
    unittest.main()

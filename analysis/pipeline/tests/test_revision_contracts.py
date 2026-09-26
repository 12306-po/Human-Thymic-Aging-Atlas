"""Small synthetic contract tests; never stand in for frozen-data validation."""
from __future__ import annotations

import os
import runpy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fig_paper import fig2, fig4, fig5  # noqa: E402
import figure_common  # noqa: E402


class RevisionContracts(unittest.TestCase):
    def test_paper_font_is_times_new_roman(self):
        self.assertEqual(figure_common.PAPER_FONT, "Times New Roman")
        self.assertEqual(matplotlib.rcParams["font.family"], ["serif"])
        self.assertEqual(matplotlib.rcParams["font.serif"], ["Times New Roman"])
        self.assertEqual(matplotlib.rcParams["mathtext.rm"], "Times New Roman")

    def test_figure4_bipolar_color_map(self):
        loc = pd.DataFrame([
            {"gene": gene, "source_object": "Sort1", "developmental_stage": stage,
             "mean_zscore": z, "pct_expr": 50.0}
            for gene, z in (("NEG", -0.7), ("POS", 0.8))
            for stage in fig4.STAGES["Sort1"]
        ])
        norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
        f, ax = plt.subplots()
        fig4._facet(ax, loc, ["NEG", "POS"], "Sort1", True, norm)
        self.assertEqual(ax.collections[0].get_cmap().name, "RdBu_r")
        self.assertEqual(ax.collections[0].norm.vmin, -1)
        self.assertLess(norm(-0.7), 0.5)
        self.assertGreater(norm(0.8), 0.5)
        plt.close(f)

    def test_figure4_stage_delta_is_within_object_only(self):
        loc = pd.DataFrame([
            {"gene": "G", "source_object": src, "developmental_stage": stage,
             "mean_zscore": value}
            for src, stages, values in (
                ("Sort1", fig4.STAGES["Sort1"], [0.1, 0.2, 0.5]),
                ("Sort2", fig4.STAGES["Sort2"], [-0.2, 0.0, 0.1]))
            for stage, value in zip(stages, values)
        ])
        with tempfile.TemporaryDirectory() as tmp, patch.object(fig4, "FIGF", Path(tmp)):
            table = fig4._stage_delta_table(loc, ["G"])
        self.assertEqual(len(table), 2)
        self.assertAlmostEqual(table.loc[table.source_object.eq("Sort1"),
                                         "delta_z_late_minus_early"].iloc[0], 0.4)
        self.assertAlmostEqual(table.loc[table.source_object.eq("Sort2"),
                                         "delta_z_late_minus_early"].iloc[0], 0.3)

    def test_composite_keeps_declared_canvas(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(figure_common, "FIGF", Path(tmp)):
            f = plt.figure(figsize=(4, 3))
            f.text(-0.2, 1.2, "outside canvas")
            figure_common.save_composite(f, "test")
            with Image.open(Path(tmp) / "test.png") as img:
                self.assertEqual(img.size, (1200, 900))

    def test_figure2_count_legend_uses_real_positive_integers(self):
        lm = pd.DataFrame({"gene": list("ABCDEFGH"),
                           "adj.P.Val": [.001, .002, .003, .004, .8, .8, .8, .8]})
        with tempfile.TemporaryDirectory() as tmp:
            gmt = Path(tmp) / "test.gmt"
            gmt.write_text("HALLMARK_A\tdesc\tA\tB\tC\tE\n"
                           "HALLMARK_B\tdesc\tA\tB\tC\tD\n", encoding="utf-8")
            f, ax = plt.subplots()
            with patch.dict(os.environ, {"HALLMARK_GMT": str(gmt)}), \
                 patch.object(fig2, "FIGF", Path(tmp)), \
                 patch.object(fig2.pd, "read_csv", return_value=lm):
                fig2._hallmark_dotplot(ax)
            source = pd.read_csv(Path(tmp) / "fig2f_hallmark_enrichment_source.tsv", sep="\t")
            self.assertTrue(source.Count.ge(1).all())
            self.assertTrue(all(int(t.get_text()) >= 1 for t in ax.get_legend().get_texts()))
            plt.close(f)

    def test_figure5_four_distinct_states(self):
        self.assertEqual(fig5._direction_state(True, True), "support")
        self.assertEqual(fig5._direction_state(True, False), "discordant")
        self.assertEqual(fig5._direction_state(False, None), "not_assessed")
        with self.assertRaises(ValueError):
            fig5._direction_state(True, None)
        self.assertFalse(fig5._bool("False"))

    def test_figure5_regulatory_counts_are_row_based_not_funnel(self):
        net = pd.DataFrame({"human_target_gene": ["A", "A", None, "B"],
                            "motif_direction_support": [True, False, True, False],
                            "full_chain_supported": [False, False, False, False]})
        with tempfile.TemporaryDirectory() as tmp:
            f, ax = plt.subplots()
            with patch.object(fig5, "FIGF", Path(tmp)), \
                 patch.object(fig5.pd, "read_csv", return_value=net):
                fig5._tf_attrition(ax)
            widths = [bar.get_width() for bar in ax.patches]
            self.assertEqual(widths, [4, 3, 2, 0])
            self.assertIn("not a funnel", ax.get_xlabel())
            plt.close(f)

    def test_manifest_points_to_existing_plot_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(figure_common, "FIGF", Path(tmp)), \
                 patch.object(figure_common, "PROJ", Path(tmp)):
                runpy.run_path(str(Path(__file__).resolve().parents[1] /
                                   "99b_source_manifest.py"))
            manifest = pd.read_csv(Path(tmp) / "figure_source_manifest.tsv", sep="\t")
            self.assertTrue(manifest.plot_script_exists.all())
            self.assertIn("S08b", set(manifest.panel))


if __name__ == "__main__":
    unittest.main()

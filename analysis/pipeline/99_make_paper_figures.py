#!/usr/bin/env python
"""Third-round revision runner; reads frozen results, writes a separate folder.

Example:
    python 99_make_paper_figures.py --project /data/.../human_thymus_age_ML_DL \
        --output /data/.../human_thymus_age_ML_DL/figures_final_revised \
        --hallmark-gmt /data/.../h.all.v2023.2.Hs.symbols.gmt
"""
from __future__ import annotations

import argparse
import importlib
import logging
import os
import runpy
import shutil
import sys
import traceback
from pathlib import Path

# scripts/ on path for the fig_paper package
sys.path.insert(0, str(Path(__file__).resolve().parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("fig99")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=os.environ.get("PROJ"),
                        required="PROJ" not in os.environ)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--hallmark-gmt", type=Path, default=None)
    parser.add_argument("--main-only", action="store_true")
    parser.add_argument("--supp-only", action="store_true")
    args = parser.parse_args()
    if args.main_only and args.supp_only:
        parser.error("--main-only and --supp-only cannot be combined")
    project = Path(args.project).expanduser().resolve()
    if not project.is_dir():
        parser.error(f"Frozen project directory not found: {project}")
    output = (args.output or project / "figures_final_revised").expanduser().resolve()
    if output == (project / "figures_final").resolve():
        parser.error("Refusing to overwrite the frozen original figures_final")
    os.environ["PROJ"] = str(project)
    os.environ["FIG_OUTPUT"] = str(output)
    if args.hallmark_gmt:
        os.environ["HALLMARK_GMT"] = str(args.hallmark_gmt.resolve())
    from figure_common import FIGF  # imported after setting project/output
    FIGF.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(Path(__file__).resolve().parent / "figure_legends_revision_notes.md",
                    FIGF / "figure_legends_revision_notes.md")
    jobs = [
        ("Figure 1", "fig_paper.fig1", "build"),
        ("Figure 2", "fig_paper.fig2", "build"),
        ("Figure 3", "fig_paper.fig3", "build"),
        ("Figure 4", "fig_paper.fig4", "build"),
        ("Figure 5", "fig_paper.fig5", "build"),
    ]
    failed = []
    if not args.supp_only:
        for label, mod_name, fn in jobs:
            try:
                mod = importlib.import_module(mod_name)
                getattr(mod, fn)()
                log.info("%s written", label)
            except Exception:  # noqa: BLE001
                failed.append(label)
                log.error("%s FAILED:\n%s", label, traceback.format_exc())

    # supplementary (best effort; report failures individually)
    from fig_paper import fig_supp1, fig_supp2, fig_supp3
    supp_jobs = [
            ("S01 age task", fig_supp1.s01_age_task),
            ("S02 batch UMAP", fig_supp1.s02_batch_umap),
            ("S03 marker dotplot", fig_supp1.s03_marker_dotplot),
            ("S04 QC/scrublet/doublet", fig_supp2.s04_qc_scrublet_doublet),
            ("S05 annotation", fig_supp2.s05_annotation),
            ("S06 full ML/female", fig_supp2.s06_full_ml),
            ("S07 binary", fig_supp3.s07_classification),
            ("S08 DL diagnostics", fig_supp3.s08_deep_learning),
            ("S09 mouse programs", fig_supp3.s09_mouse_programs)]
    if not args.main_only:
        for label, fn in supp_jobs:
            try:
                fn()
                log.info("%s written", label)
            except Exception:  # noqa: BLE001
                failed.append(label)
                log.error("%s FAILED:\n%s", label, traceback.format_exc())
    try:
        runpy.run_path(str(Path(__file__).resolve().parent / "99b_source_manifest.py"))
    except Exception:  # noqa: BLE001
        failed.append("source manifest")
        log.error("Source manifest FAILED:\n%s", traceback.format_exc())
    if failed:
        log.error("Incomplete rebuild: %s", ", ".join(failed))
        raise SystemExit(1)
    log.info("All requested figures written -> %s", FIGF)


if __name__ == "__main__":
    main()

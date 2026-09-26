#!/usr/bin/env python
"""Step 32: assemble the human thymic aging atlas source-data layer.

The atlas is a set of traceable donor-level result tables, not a new statistical
test. Each record retains its analysis level and generating source file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project", type=Path,
        default=Path(os.environ.get("PROJ", "/data/zxy/projects/human_thymus_age_ML_DL")),
    )
    return parser.parse_args()


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    project = parse_args().project.resolve()
    output = project / "10_results" / "human_thymic_aging_atlas"
    output.mkdir(parents=True, exist_ok=True)
    limma_dir = project / "03_feature_selection" / "limma_adjusted"

    required = {
        "composition": project / "03_feature_selection" / "composition_adjusted" / "composition_results.tsv",
        "developmental_context": project / "07_external_validation" / "17_candidate_gene_localization.csv",
        "donor_score": project / "05_deep_learning" / "thymic_aging_score" / "thymic_aging_score_internal.tsv",
        "donor_manifest": project / "01_raw_processing" / "metadata" / "submission_manifest" / "donor_manifest.csv",
    }
    missing = [str(path) for path in required.values() if not path.exists() or path.stat().st_size == 0]
    if missing:
        raise FileNotFoundError("Atlas prerequisites missing/empty:\n" + "\n".join(missing))

    expression_frames = []
    expression_sources = []
    for path in sorted(limma_dir.glob("*_limma_age_sex.csv")):
        if path.name == "limma_celltype_coverage.csv":
            continue
        table = pd.read_csv(path)
        if not {"gene", "logFC_age_perSD", "P.Value", "adj.P.Val"}.issubset(table.columns):
            continue
        tag = path.name.replace("_limma_age_sex.csv", "")
        table.insert(0, "cell_context", "whole_thymus" if tag == "whole_thymus" else tag)
        table.insert(0, "analysis_level", "whole_thymus" if tag == "whole_thymus" else "celltype_pseudobulk")
        table["source_file"] = str(path)
        expression_frames.append(table)
        expression_sources.append(path)
    if not expression_frames:
        raise RuntimeError(f"No valid limma result tables found in {limma_dir}")
    expression = pd.concat(expression_frames, ignore_index=True)
    expression.to_csv(
        output / "atlas_gene_age_effects.tsv.gz", sep="\t", index=False, compression="gzip"
    )

    composition = pd.read_csv(required["composition"], sep="\t")
    composition = composition[composition["term"].eq("age_z")].copy()
    composition["effect_interpretation"] = "CLR log-ratio effect per 1 SD age; sex-adjusted"
    composition["source_file"] = str(required["composition"])
    composition.to_csv(output / "atlas_composition_age_effects.tsv", sep="\t", index=False)

    pathway_frames = []
    pathway_sources = []
    for name, label in (
        ("16_pathway_enrichment_ML.csv", "ML_candidate_enrichment"),
        ("16_pathway_enrichment_DL.csv", "HumanThymusFormer_candidate_enrichment"),
    ):
        path = project / "06_interpretation" / name
        if path.exists() and path.stat().st_size:
            table = pd.read_csv(path)
            table.insert(0, "evidence_layer", label)
            table["source_file"] = str(path)
            pathway_frames.append(table)
            pathway_sources.append(path)
    if pathway_frames:
        pd.concat(pathway_frames, ignore_index=True).to_csv(
            output / "atlas_pathways.tsv", sep="\t", index=False
        )
    else:
        pd.DataFrame(columns=["evidence_layer", "status"]).to_csv(
            output / "atlas_pathways.tsv", sep="\t", index=False
        )

    developmental = pd.read_csv(required["developmental_context"])
    developmental["evidence_scope"] = "external human developmental localization; not aging replication"
    developmental["source_file"] = str(required["developmental_context"])
    developmental.to_csv(output / "atlas_developmental_context.tsv", sep="\t", index=False)

    score = pd.read_csv(required["donor_score"], sep="\t")
    score.to_csv(output / "atlas_donor_scores.tsv", sep="\t", index=False)
    manifest = pd.read_csv(required["donor_manifest"])
    manifest.to_csv(output / "atlas_donor_manifest.tsv", sep="\t", index=False)

    sources = {
        **required,
        **{f"expression_{index:02d}": path for index, path in enumerate(expression_sources, start=1)},
        **{f"pathway_{index:02d}": path for index, path in enumerate(pathway_sources, start=1)},
    }
    pd.DataFrame([
        {
            "module": name,
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": digest(path),
        }
        for name, path in sources.items()
    ]).to_csv(output / "atlas_source_manifest.tsv", sep="\t", index=False)

    contract = {
        "atlas_name": "Donor-resolved continuous-age human thymic aging atlas",
        "independent_unit": "human donor",
        "primary_inference": "sex-adjusted donor-level limma-voom and composition-aware age+sex CLR",
        "modules": [
            "cell composition", "whole-thymus pseudobulk", "cell-type-specific pseudobulk",
            "age-associated genes", "pathways", "developmental-state context",
            "internal thymic aging score",
        ],
        "developmental_context_limit": "GSE195812 is not an aging validation cohort",
        "score_limit": "internal proof-of-concept, not a clinical clock",
    }
    (output / "atlas_contract.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")
    print(f"Human thymic aging atlas packet written to {output}")


if __name__ == "__main__":
    main()

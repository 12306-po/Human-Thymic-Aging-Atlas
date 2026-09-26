#!/usr/bin/env python
"""Step 30: validate Steps 25-29 and build a manuscript-update packet.

This step never edits the manuscript. It fails closed when required outputs are
missing and records exact source paths/checksums for every value summarized.
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    project = parse_args().project.resolve()
    output = project / "10_results" / "submission_completion"
    output.mkdir(parents=True, exist_ok=True)
    required = {
        "donor_manifest": project / "01_raw_processing" / "metadata" / "submission_manifest" / "donor_manifest.csv",
        "ml_predictions": project / "04_machine_learning" / "fully_nested" / "oof_predictions_all_models.tsv",
        "ml_metrics": project / "04_machine_learning" / "fully_nested" / "model_metrics.tsv",
        "ml_ci": project / "04_machine_learning" / "fully_nested" / "model_metrics_bootstrap_ci.tsv",
        "ml_vs_null": project / "04_machine_learning" / "fully_nested" / "paired_model_vs_null_MAE.tsv",
        "ml_ablation": project / "04_machine_learning" / "fully_nested" / "nested_ablation_results.tsv",
        "composition": project / "03_feature_selection" / "composition_adjusted" / "composition_results.tsv",
        "composition_sensitivity": project / "03_feature_selection" / "composition_adjusted" / "composition_sensitivity_results.tsv",
        "composition_dirichlet": project / "03_feature_selection" / "composition_adjusted" / "composition_dirichlet_sensitivity.tsv",
        "cross_species": project / "08_mouse_validation" / "sex_adjusted_descriptive" / "human_mouse_ortholog_comparison.tsv",
        "cross_species_summary": project / "08_mouse_validation" / "sex_adjusted_descriptive" / "28_cross_species_descriptive_summary.tsv",
        "expression_sensitivity": project / "03_feature_selection" / "expression_sensitivity" / "expression_sensitivity_summary.tsv",
    }
    missing = [str(path) for path in required.values() if not path.exists() or path.stat().st_size == 0]
    if missing:
        raise FileNotFoundError("Required completion outputs are missing/empty:\n" + "\n".join(missing))

    manifest = pd.read_csv(required["donor_manifest"])
    metrics = pd.read_csv(required["ml_metrics"], sep="\t")
    intervals = pd.read_csv(required["ml_ci"], sep="\t")
    paired = pd.read_csv(required["ml_vs_null"], sep="\t")
    composition = pd.read_csv(required["composition"], sep="\t")
    cross_summary = pd.read_csv(required["cross_species_summary"], sep="\t")
    expression = pd.read_csv(required["expression_sensitivity"], sep="\t")

    if len(manifest) != 18 or manifest["donor_id"].nunique() != 18:
        raise AssertionError("Donor manifest is not the frozen 18-donor cohort")
    primary = metrics[metrics["cohort"].eq("primary_18_donors")]
    elastic = primary[primary["model"].eq("ElasticNet")]
    null = primary[primary["model"].eq("MeanAgeNull")]
    if len(elastic) != 1 or len(null) != 1:
        raise AssertionError("Exactly one primary ElasticNet and MeanAgeNull metric row is required")
    elastic_ci = intervals[
        intervals["cohort"].eq("primary_18_donors") & intervals["model"].eq("ElasticNet")
    ]
    if len(elastic_ci) != 1:
        raise AssertionError("Exactly one primary ElasticNet CI row is required")

    age_composition = composition[composition["term"].eq("age_z")]
    significant_composition = age_composition[age_composition["q_value_BH"] < 0.05]
    elastic_vs_null = paired[paired["model"].eq("ElasticNet")]

    source_rows = [
        {
            "logical_name": name,
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for name, path in required.items()
    ]
    pd.DataFrame(source_rows).to_csv(
        output / "submission_source_data_manifest.tsv", sep="\t", index=False
    )

    e = elastic.iloc[0]
    n = null.iloc[0]
    ci = elastic_ci.iloc[0]
    paired_row = elastic_vs_null.iloc[0] if len(elastic_vs_null) == 1 else None
    lines = [
        "# Final-analysis result packet (Steps 25–30)",
        "",
        "This file is a gated summary for manuscript revision. It does not change the Word manuscript.",
        "",
        "## Donor manifest",
        "",
        f"- Frozen cohort: {len(manifest)} unique human donors.",
        f"- Donors with one or more explicit missing metadata fields: {int(manifest['missing_data_reasons'].notna().sum())}.",
        "- Missing clinical/procurement values remain NA; no ethics or clinical fields were inferred.",
        "",
        "## Fully nested internal machine learning",
        "",
        f"- Elastic Net OOF R²: {e.R2:.3f} (donor-bootstrap 95% CI {ci.R2_ci_low:.3f} to {ci.R2_ci_high:.3f}).",
        f"- Elastic Net OOF MAE: {e.MAE:.2f} years (95% CI {ci.MAE_ci_low:.2f} to {ci.MAE_ci_high:.2f}).",
        f"- Mean-age null OOF R²: {n.R2:.3f}; MAE: {n.MAE:.2f} years.",
    ]
    if paired_row is not None:
        lines.append(
            f"- Paired ΔMAE (Elastic Net − null): {paired_row.delta_MAE_model_minus_null:.2f} years "
            f"(95% CI {paired_row.ci_low:.2f} to {paired_row.ci_high:.2f})."
        )
    lines.extend([
        "- Interpretation remains internal donor-level validation; there is no independent adult-aging test cohort.",
        "",
        "## Composition-aware analysis",
        "",
        f"- CLR age+sex models tested {len(age_composition)} cell-type components.",
        f"- Components with BH q<0.05: {len(significant_composition)}.",
    ])
    if len(significant_composition):
        lines.append("- Significant components: " + ", ".join(significant_composition["cell_type"].astype(str)))
    lines.extend([
        "- Effects are relative log-ratio effects, not absolute cell-count changes.",
        "",
        "## Sex-adjusted human–mouse comparison",
        "",
        "- Human direction is based on sex-adjusted limma β_age.",
        "- Mouse evidence remains a one-Young/one-Aged descriptive comparison and is supplementary/hypothesis-generating only.",
        "",
    ])
    for _, row in cross_summary.iterrows():
        if row["subset"] == "all_mapped":
            fraction = row["concordance_fraction"]
            fraction_text = "NA" if pd.isna(fraction) else f"{fraction:.1%}"
            lines.append(
                f"- {row['evidence_type']}: {int(row['n_concordant'])}/{int(row['n_assessable'])} "
                f"assessable rows concordant ({fraction_text}); descriptive only."
            )
    lines.extend([
        "",
        "## Expression sensitivity",
        "",
    ])
    for _, row in expression.iterrows():
        lines.append(
            f"- {row['analysis']}: β_age correlation with primary={row['spearman_beta_vs_primary']:.3f}; "
            f"sign concordance={row['sign_concordance_all']:.1%}; n={int(row['n_donors'])}."
        )
    lines.extend([
        "",
        "## Manuscript update gate",
        "",
        "The old provisional numeric ML and descriptive composition text may now be replaced only with values traced to the files in `submission_source_data_manifest.tsv`. Claims of external validation, clinical prediction, or conserved mouse mechanism remain disallowed by the study design.",
    ])
    (output / "final_result_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    gate = {
        "safe_to_replace_provisional_numeric_results": True,
        "required_outputs_verified": list(required),
        "n_manifest_donors": 18,
        "remaining_claim_limits": [
            "internal validation only",
            "composition effects are relative",
            "mouse comparison is single-pair and descriptive",
            "ethics identifiers must be copied from authoritative source records",
        ],
    }
    (output / "manuscript_update_gate.json").write_text(
        json.dumps(gate, indent=2), encoding="utf-8"
    )
    print(f"Submission result packet written to {output}")


if __name__ == "__main__":
    main()

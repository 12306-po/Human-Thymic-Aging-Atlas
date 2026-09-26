#!/usr/bin/env python
"""Step 34: validate signed human aging effects across standardized datasets.

External effect files must already be estimated with the animal/sample/pool as
the independent unit. Cells, genes, and peaks are never treated as replicates.
No automatic download or metadata guessing is performed here.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_EXTERNAL_COLUMNS = {
    "dataset_id", "species", "modality", "feature_type", "feature_id",
    "effect_age", "standard_error", "p_value", "q_value",
    "n_biological_units", "biological_unit", "contrast",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project", type=Path,
        default=Path(os.environ.get("PROJ", "/data/zxy/projects/human_thymus_age_ML_DL")),
    )
    parser.add_argument(
        "--registry", type=Path,
        default=Path(__file__).with_name("external_validation_registry_20260923.tsv"),
    )
    parser.add_argument("--allow-empty", action="store_true")
    return parser.parse_args()


def normalize_bool(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def map_mouse_gene(effect: pd.DataFrame, ortholog: pd.DataFrame) -> pd.DataFrame:
    effect = effect.copy()
    effect["mouse_lower"] = effect["feature_id"].astype(str).str.lower()
    mapped = effect.merge(
        ortholog[["mouse_lower", "human_gene"]], on="mouse_lower", how="left",
        validate="many_to_one",
    )
    mapped["human_feature_id"] = mapped["human_gene"]
    mapped["mapping_status"] = np.where(mapped["human_gene"].notna(), "mapped_best_ortholog", "not_mapped")
    return mapped


def main() -> None:
    args = parse_args()
    project = args.project.resolve()
    input_dir = project / "08_external_evidence" / "standardized_effects"
    output = project / "08_external_evidence" / "validation"
    output.mkdir(parents=True, exist_ok=True)
    input_dir.mkdir(parents=True, exist_ok=True)

    reference_path = project / "08_external_evidence" / "human_reference_effects.tsv"
    ortholog_path = (
        project / "08_mouse_validation" / "orthologs" / "human_mouse_orthologs_reconciled.csv"
    )
    if not reference_path.exists():
        raise FileNotFoundError(f"Run Step 33 first: {reference_path}")
    if not args.registry.exists():
        raise FileNotFoundError(f"Dataset registry missing: {args.registry}")
    reference = pd.read_csv(reference_path, sep="\t")
    registry = pd.read_csv(args.registry, sep="\t")
    if registry["dataset_id"].duplicated().any():
        raise ValueError("Dataset registry contains duplicate dataset_id values")

    effect_files = sorted(input_dir.glob("*.tsv"))
    if not effect_files:
        template = pd.DataFrame(columns=sorted(REQUIRED_EXTERNAL_COLUMNS))
        template.to_csv(input_dir / "EXTERNAL_EFFECT_TEMPLATE.tsv", sep="\t", index=False)
        if args.allow_empty:
            print(f"No external effects yet; template written to {input_dir}")
            return
        raise FileNotFoundError(
            f"No standardized external effect files in {input_dir}. "
            "A template has been created; metadata must be frozen before analysis."
        )

    ortholog = pd.read_csv(ortholog_path)
    ortholog["is_best_ortholog"] = ortholog["is_best_ortholog"].astype(str).str.lower().isin(
        ["true", "1", "yes"]
    )
    ortholog = ortholog[ortholog["is_best_ortholog"]].copy()
    ortholog["mouse_lower"] = ortholog["mouse_gene"].astype(str).str.lower()
    if ortholog["mouse_lower"].duplicated().any():
        # Mouse-to-human can be many-to-one/one-to-many. Retain only mappings
        # unique in the reverse direction to prevent effect-dependent selection.
        ortholog = ortholog[~ortholog["mouse_lower"].duplicated(keep=False)].copy()

    evidence_frames = []
    audit_rows = []
    for path in effect_files:
        if path.name == "EXTERNAL_EFFECT_TEMPLATE.tsv":
            continue
        external = pd.read_csv(path, sep="\t")
        missing = REQUIRED_EXTERNAL_COLUMNS.difference(external.columns)
        if missing:
            raise ValueError(f"{path.name} lacks columns: {', '.join(sorted(missing))}")
        ids = set(external["dataset_id"].astype(str))
        if len(ids) != 1:
            raise ValueError(f"{path.name} must contain exactly one dataset_id")
        dataset_id = next(iter(ids))
        if dataset_id not in set(registry["dataset_id"].astype(str)):
            raise ValueError(f"Dataset {dataset_id} is not in the frozen registry")
        if external.duplicated(["feature_type", "feature_id"]).any():
            raise ValueError(f"{path.name} has duplicate feature effects")
        for numeric_column in (
            "effect_age", "standard_error", "p_value", "q_value", "n_biological_units"
        ):
            external[numeric_column] = pd.to_numeric(
                external[numeric_column], errors="coerce"
            )
        if external["n_biological_units"].isna().any() or (
            external["n_biological_units"] < 2
        ).any():
            raise ValueError(
                f"{path.name} has missing/invalid biological-unit counts or effects with <2 units"
            )

        is_mouse = external["species"].astype(str).eq("Mus musculus")
        is_gene_like = external["feature_type"].isin(["gene", "tf"])
        external["human_feature_id"] = external["feature_id"].astype(str)
        external["mapping_status"] = "direct_human_or_standardized_program"
        if (is_mouse & is_gene_like).any():
            mapped = map_mouse_gene(external[is_mouse & is_gene_like], ortholog)
            unchanged = external[~(is_mouse & is_gene_like)].copy()
            external = pd.concat([mapped, unchanged], ignore_index=True, sort=False)

        reference_join = reference.rename(columns={"feature_id": "human_feature_id"})
        joined = external.merge(
            reference_join,
            on=["feature_type", "human_feature_id"], how="left",
            suffixes=("_external", "_human"), validate="many_to_one",
        )
        joined["assessable"] = (
            joined["human_effect_age"].notna()
            & joined["effect_age"].notna()
            & joined["human_effect_age"].ne(0)
            & joined["effect_age"].ne(0)
        )
        joined["direction_concordant"] = pd.NA
        assessable = joined["assessable"]
        joined.loc[assessable, "direction_concordant"] = (
            np.sign(joined.loc[assessable, "human_effect_age"])
            == np.sign(joined.loc[assessable, "effect_age"])
        )
        joined["external_significant_q05"] = joined["q_value"].lt(0.05)
        joined["human_significant_q05"] = joined["human_q_value"].lt(0.05)
        joined["evidence_status"] = "not_assessable"
        joined.loc[assessable & joined["direction_concordant"].eq(True), "evidence_status"] = (
            "directional_support_nonsignificant"
        )
        joined.loc[
            assessable & joined["direction_concordant"].eq(True)
            & joined["external_significant_q05"], "evidence_status"
        ] = "replicated_external_support"
        joined.loc[
            assessable & joined["direction_concordant"].eq(False)
            & joined["external_significant_q05"], "evidence_status"
        ] = "replicated_external_discordance"
        joined.loc[
            assessable & joined["direction_concordant"].eq(False)
            & ~joined["external_significant_q05"], "evidence_status"
        ] = "directional_discordance_nonsignificant"

        registry_row = registry[registry["dataset_id"].astype(str).eq(dataset_id)].iloc[0]
        joined["inference_tier"] = registry_row["inference_tier"]
        joined["proposed_role"] = registry_row["proposed_role"]
        joined["source_effect_file"] = str(path)
        evidence_frames.append(joined)
        audit_rows.append({
            "dataset_id": dataset_id,
            "source_effect_file": str(path),
            "n_rows": len(joined),
            "n_assessable": int(joined["assessable"].sum()),
            "n_external_q05": int(joined["external_significant_q05"].sum()),
            "biological_unit": ";".join(sorted(joined["biological_unit"].astype(str).unique())),
            "minimum_n_biological_units": int(joined["n_biological_units"].min()),
        })

    evidence = pd.concat(evidence_frames, ignore_index=True)
    evidence.to_csv(output / "cross_dataset_evidence_long.tsv.gz", sep="\t", index=False, compression="gzip")
    pd.DataFrame(audit_rows).to_csv(output / "cross_dataset_input_audit.tsv", sep="\t", index=False)

    summary_rows = []
    for (dataset_id, feature_type), table in evidence.groupby(["dataset_id", "feature_type"]):
        assessed = table[table["assessable"]]
        significant = assessed[assessed["external_significant_q05"]]
        summary_rows.append({
            "dataset_id": dataset_id,
            "feature_type": feature_type,
            "inference_tier": table["inference_tier"].iloc[0],
            "n_total": len(table),
            "n_assessable": len(assessed),
            "n_concordant": int(assessed["direction_concordant"].eq(True).sum()),
            "concordance_fraction": assessed["direction_concordant"].eq(True).mean() if len(assessed) else np.nan,
            "n_external_q05": len(significant),
            "n_external_q05_concordant": int(significant["direction_concordant"].eq(True).sum()),
            "claim_scope": "cross-dataset support; not human score validation",
        })
    pd.DataFrame(summary_rows).to_csv(output / "cross_dataset_evidence_summary.tsv", sep="\t", index=False)

    integrated_input = evidence[
        evidence["inference_tier"].isin(["A", "B"]) & evidence["assessable"]
    ].copy()
    integrated = (
        integrated_input
        .groupby(["feature_type", "human_feature_id"], as_index=False)
        .agg(
            n_datasets_assessed=("dataset_id", "nunique"),
            n_replicated_support=("evidence_status", lambda x: int((x == "replicated_external_support").sum())),
            n_replicated_discordance=("evidence_status", lambda x: int((x == "replicated_external_discordance").sum())),
            n_directional_support=("direction_concordant", lambda x: int(pd.Series(x).eq(True).sum())),
            datasets=("dataset_id", lambda x: ";".join(sorted(set(map(str, x))))),
            modalities=("modality", lambda x: ";".join(sorted(set(map(str, x))))),
        )
    )
    integrated["evidence_class"] = np.select(
        [
            (integrated["n_replicated_support"] >= 2) & integrated["n_replicated_discordance"].eq(0),
            integrated["n_replicated_support"].ge(1) & integrated["n_replicated_discordance"].eq(0),
            integrated["n_replicated_discordance"].ge(1),
        ],
        ["multi-dataset replicated support", "single-dataset replicated support", "replicated discordance"],
        default="directional/descriptive only",
    )
    integrated.to_csv(output / "cross_dataset_integrated_evidence.tsv", sep="\t", index=False)

    contract = {
        "human_reference": "sex-adjusted donor-level human age effects",
        "external_unit": "animal, sample, or independent biological pool defined by each frozen dataset",
        "cells_genes_peaks_are_replicates": False,
        "integration": "tiered evidence counts; no pooling across incomparable modalities",
        "score_validation": "not established by mouse or developmental datasets",
    }
    (output / "cross_dataset_validation_contract.json").write_text(
        json.dumps(contract, indent=2), encoding="utf-8"
    )
    print(f"Cross-dataset evidence outputs written to {output}")


if __name__ == "__main__":
    main()

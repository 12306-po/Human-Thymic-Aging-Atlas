from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

VENDOR = Path(__file__).resolve().parent / "_vendor"
if VENDOR.exists():
    sys.path.insert(0, str(VENDOR))


KEY_CONTEXTS = ["whole_thymus", "TEC", "Fibroblast", "DP", "SP_CD4", "SP_CD8"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_csv(path, sep="\t")


def compact_join(values, limit: int = 8) -> str:
    cleaned = [str(value) for value in values if pd.notna(value) and str(value).strip()]
    return "; ".join(dict.fromkeys(cleaned[:limit]))


def split_gene_members(value: object) -> list[str]:
    if pd.isna(value):
        return []
    return [item.strip().upper() for item in str(value).replace(",", ";").split(";") if item.strip()]


def build_gene_pathway_links(pathways: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, record in pathways.iterrows():
        genes = split_gene_members(record.get("Genes"))
        for gene in genes:
            rows.append(
                {
                    "gene": gene,
                    "evidence_layer": record.get("evidence_layer"),
                    "candidate_set": record.get("candidate_set"),
                    "gene_set_library": record.get("Gene_set"),
                    "pathway": record.get("Term"),
                    "pathway_p_value": record.get("P-value"),
                    "pathway_q_value": record.get("Adjusted_P_value", record.get("Adjusted P-value")),
                    "odds_ratio": record.get("Odds_Ratio", record.get("Odds Ratio")),
                    "combined_score": record.get("Combined_Score", record.get("Combined Score")),
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=[
                "gene",
                "evidence_layer",
                "candidate_set",
                "gene_set_library",
                "pathway",
                "pathway_p_value",
                "pathway_q_value",
                "odds_ratio",
                "combined_score",
            ]
        )
    result = pd.DataFrame(rows)
    result["pathway_q_value"] = pd.to_numeric(result["pathway_q_value"], errors="coerce")
    return result.sort_values(["gene", "pathway_q_value", "pathway"])


def build_gene_summary(
    effects: pd.DataFrame,
    candidate: pd.DataFrame,
    attribution: pd.DataFrame,
    developmental: pd.DataFrame,
    gene_pathways: pd.DataFrame,
) -> pd.DataFrame:
    effects = effects.copy()
    effects["gene"] = effects["gene"].astype(str).str.upper()
    genes = pd.DataFrame({"gene": sorted(effects["gene"].unique())})

    for context in KEY_CONTEXTS:
        subset = effects.loc[effects["cell_context"].eq(context)].drop_duplicates("gene")
        subset = subset.set_index("gene")
        prefix = context.lower()
        genes[f"{prefix}_beta_age"] = genes["gene"].map(subset["logFC_age_perSD"])
        genes[f"{prefix}_q_value"] = genes["gene"].map(subset["adj.P.Val"])
        genes[f"{prefix}_p_value"] = genes["gene"].map(subset["P.Value"])
        genes[f"{prefix}_n_donors"] = genes["gene"].map(subset["n_donors"])

    candidate = candidate.copy()
    candidate["gene"] = candidate["gene"].astype(str).str.upper()
    candidate_columns = [
        "gene",
        "evidence_tier",
        "evidence_score",
        "n_evidence_layers",
        "direction_summary",
        "in_ML_consensus",
        "ml_best_median_pct",
        "ml_n_features",
        "in_DL_top100",
        "dl_mean_rank",
        "mouse_RNA_direction",
        "mouse_RNA_concordant",
        "mouse_ATAC_detectable",
        "mouse_ATAC_concordant",
        "mouse_regulatory_support",
        "external_top_state",
        "mechanism_celltypes",
    ]
    candidate_columns = [column for column in candidate_columns if column in candidate.columns]
    genes = genes.merge(candidate[candidate_columns].drop_duplicates("gene"), on="gene", how="left")

    attribution = attribution.copy()
    attribution["gene"] = attribution["gene"].astype(str).str.upper()
    attribution["IG_importance"] = pd.to_numeric(attribution.get("IG_importance"), errors="coerce")
    attribution["attention_importance"] = pd.to_numeric(
        attribution.get("attention_importance"), errors="coerce"
    )
    best_ig = (
        attribution.sort_values("IG_rank")
        .drop_duplicates("gene")
        .set_index("gene")
    )
    best_attention = (
        attribution.sort_values("attention_rank")
        .drop_duplicates("gene")
        .set_index("gene")
    )
    genes["transformer_best_ig_celltype"] = genes["gene"].map(best_ig.get("celltype"))
    genes["transformer_best_ig_rank"] = genes["gene"].map(best_ig.get("IG_rank"))
    genes["transformer_best_ig_importance"] = genes["gene"].map(best_ig.get("IG_importance"))
    genes["transformer_best_attention_celltype"] = genes["gene"].map(best_attention.get("celltype"))
    genes["transformer_best_attention_rank"] = genes["gene"].map(best_attention.get("attention_rank"))
    genes["transformer_best_attention_importance"] = genes["gene"].map(
        best_attention.get("attention_importance")
    )

    developmental = developmental.copy()
    developmental["gene"] = developmental["gene"].astype(str).str.upper()
    best_stage = (
        developmental.sort_values(["gene", "mean_zscore"], ascending=[True, False])
        .drop_duplicates("gene")
        .set_index("gene")
    )
    genes["developmental_top_stage"] = genes["gene"].map(best_stage.get("developmental_stage"))
    genes["developmental_top_stage_z"] = genes["gene"].map(best_stage.get("mean_zscore"))
    genes["developmental_top_stage_pct_expr"] = genes["gene"].map(best_stage.get("pct_expr"))

    if not gene_pathways.empty:
        top_pathways = (
            gene_pathways.sort_values(["gene", "pathway_q_value", "pathway"])
            .groupby("gene", observed=True)["pathway"]
            .apply(lambda series: compact_join(series, 8))
        )
        genes["top_pathways"] = genes["gene"].map(top_pathways)
    else:
        genes["top_pathways"] = ""

    genes["sp_effects"] = (
        "SP_CD4 beta="
        + genes["sp_cd4_beta_age"].map(lambda x: "NA" if pd.isna(x) else f"{x:.4g}")
        + "; SP_CD8 beta="
        + genes["sp_cd8_beta_age"].map(lambda x: "NA" if pd.isna(x) else f"{x:.4g}")
    )
    return genes


def build_cell_type_outputs(
    effects: pd.DataFrame,
    composition: pd.DataFrame,
    attribution: pd.DataFrame,
    gene_pathways: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = effects.loc[effects["analysis_level"].eq("celltype_pseudobulk")].copy()
    rows["gene"] = rows["gene"].astype(str).str.upper()
    rows["direction"] = np.select(
        [rows["adj.P.Val"].lt(0.05) & rows["logFC_age_perSD"].gt(0),
         rows["adj.P.Val"].lt(0.05) & rows["logFC_age_perSD"].lt(0)],
        ["age_up", "age_down"],
        default="not_significant",
    )
    rows = rows.sort_values(["cell_context", "adj.P.Val", "logFC_age_perSD"], ascending=[True, True, False])

    attribution = attribution.copy()
    attribution["gene"] = attribution["gene"].astype(str).str.upper()
    attr_summary = (
        attribution.groupby("celltype", observed=True)
        .apply(
            lambda group: pd.Series(
                {
                    "top_gene_celltype_predictors": compact_join(
                        group.sort_values("IG_rank")["gene"], 10
                    ),
                    "n_transformer_tokens": len(group),
                }
            ),
            include_groups=False,
        )
        .reset_index()
        .rename(columns={"celltype": "cell_type"})
    )

    comp = composition.copy().rename(columns={"cell_type": "cell_type"})
    comp = comp.loc[comp["term"].eq("age_z")].copy()
    summaries: list[dict[str, object]] = []
    for cell_type, group in rows.groupby("cell_context", observed=True):
        up = group.loc[group["direction"].eq("age_up")]
        down = group.loc[group["direction"].eq("age_down")]
        summaries.append(
            {
                "cell_type": cell_type,
                "n_genes_tested": len(group),
                "n_age_up_q05": len(up),
                "n_age_down_q05": len(down),
                "top_age_up_genes": compact_join(up["gene"], 12),
                "top_age_down_genes": compact_join(down["gene"], 12),
                "min_donor_coverage": group["n_donors"].min(),
                "max_donor_coverage": group["n_donors"].max(),
            }
        )
    summary = pd.DataFrame(summaries)
    summary = summary.merge(
        comp[
            [
                "cell_type",
                "estimate",
                "standard_error_HC3",
                "ci_low_95",
                "ci_high_95",
                "p_value",
                "q_value_BH",
                "n_donors",
                "effect_interpretation",
            ]
        ].rename(
            columns={
                "estimate": "composition_age_effect",
                "standard_error_HC3": "composition_standard_error_HC3",
                "ci_low_95": "composition_ci_low_95",
                "ci_high_95": "composition_ci_high_95",
                "p_value": "composition_p_value",
                "q_value_BH": "composition_q_value_BH",
                "n_donors": "composition_n_donors",
            }
        ),
        on="cell_type",
        how="left",
    )
    summary = summary.merge(attr_summary, on="cell_type", how="left")

    overlap_rows: list[dict[str, object]] = []
    significant = rows.loc[rows["direction"].isin(["age_up", "age_down"])]
    if not gene_pathways.empty:
        joined = significant[["cell_context", "gene", "direction"]].merge(
            gene_pathways, on="gene", how="inner"
        )
        if not joined.empty:
            for keys, group in joined.groupby(
                ["cell_context", "direction", "evidence_layer", "candidate_set", "pathway"],
                observed=True,
            ):
                cell_type, direction, evidence_layer, candidate_set, pathway = keys
                overlap_rows.append(
                    {
                        "cell_type": cell_type,
                        "direction": direction,
                        "evidence_layer": evidence_layer,
                        "candidate_set": candidate_set,
                        "pathway": pathway,
                        "n_overlap_genes": group["gene"].nunique(),
                        "overlap_genes": compact_join(sorted(group["gene"].unique()), 30),
                        "source_pathway_q_value": group["pathway_q_value"].min(),
                        "interpretation": "descriptive overlap with existing ML/DL pathway evidence; not de novo cell-type enrichment",
                    }
                )
    overlap = pd.DataFrame(overlap_rows)
    if not overlap.empty:
        overlap = overlap.sort_values(
            ["cell_type", "n_overlap_genes", "source_pathway_q_value"],
            ascending=[True, False, True],
        )
        top_overlap = (
            overlap.groupby("cell_type", observed=True)["pathway"]
            .apply(lambda series: compact_join(series, 8))
        )
        summary["pathway_evidence_overlap"] = summary["cell_type"].map(top_overlap)
    else:
        summary["pathway_evidence_overlap"] = ""
    return summary.sort_values("cell_type"), rows, overlap


def build_donor_outputs(
    manifest: pd.DataFrame,
    scores: pd.DataFrame,
    composition_path: Path | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest = manifest.copy()
    manifest["donor_id"] = manifest["donor_id"].astype(str)
    scores = scores.copy()
    scores["donor_id"] = scores["donor_id"].astype(str)
    donor = manifest.merge(scores, on="donor_id", how="left", suffixes=("", "_score"))

    if composition_path and composition_path.exists():
        wide = pd.read_csv(composition_path)
        if "donor_id" not in wide.columns:
            wide = wide.rename(columns={wide.columns[0]: "donor_id"})
        composition = wide.melt(id_vars="donor_id", var_name="cell_type", value_name="captured_fraction")
        composition["composition_scope"] = "all QC-passed captured cells; not native-tissue composition"
        donor["composition_status"] = "available"
    else:
        composition = pd.DataFrame(
            {
                "donor_id": donor["donor_id"],
                "cell_type": pd.NA,
                "captured_fraction": pd.NA,
                "composition_scope": "source table not present in the local bundle; export donor_celltype_proportion_all.csv from the server project",
            }
        )
        donor["composition_status"] = "source table not present in local bundle"
    return donor, composition


def write_h5ad(
    output_path: Path,
    effects: pd.DataFrame,
    gene_summary: pd.DataFrame,
    donor_summary: pd.DataFrame,
    cell_summary: pd.DataFrame,
) -> None:
    try:
        import anndata as ad
    except ImportError as exc:
        raise RuntimeError(
            "anndata is required to write the results-level H5AD. Install requirements.txt and rerun."
        ) from exc

    beta = effects.pivot_table(index="gene", columns="cell_context", values="logFC_age_perSD")
    qval = effects.pivot_table(index="gene", columns="cell_context", values="adj.P.Val").reindex(
        index=beta.index, columns=beta.columns
    )
    pval = effects.pivot_table(index="gene", columns="cell_context", values="P.Value").reindex(
        index=beta.index, columns=beta.columns
    )
    ndonors = effects.pivot_table(index="gene", columns="cell_context", values="n_donors").reindex(
        index=beta.index, columns=beta.columns
    )
    obs = gene_summary.set_index("gene").reindex(beta.index)
    obs = obs.select_dtypes(exclude=["object"]).copy()
    var = pd.DataFrame(index=beta.columns.astype(str))
    adata = ad.AnnData(
        X=np.nan_to_num(beta.to_numpy(dtype=np.float32), nan=0.0),
        obs=obs,
        var=var,
    )
    adata.layers["q_value"] = np.nan_to_num(qval.to_numpy(dtype=np.float32), nan=1.0)
    adata.layers["p_value"] = np.nan_to_num(pval.to_numpy(dtype=np.float32), nan=1.0)
    adata.layers["n_donors"] = np.nan_to_num(ndonors.to_numpy(dtype=np.float32), nan=0.0)
    adata.uns["atlas_name"] = "Human Thymic Aging Atlas"
    adata.uns["matrix_semantics"] = (
        "Results-level matrix: genes x cell contexts; X stores sex-adjusted beta_age per SD, not expression."
    )
    adata.uns["composition_scope"] = (
        "Captured-library fractions after nominal 6:4 CD45-positive/CD45-negative recombination; not native tissue."
    )
    adata.uns["score_scope"] = "Descriptive internally evaluated donor deviation; not a clinical clock."
    adata.uns["donor_summary_json"] = donor_summary.to_json(orient="records")
    adata.uns["cell_type_summary_json"] = cell_summary.to_json(orient="records")
    adata.write_h5ad(output_path, compression="gzip")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Human Thymic Aging Atlas release bundle.")
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "release")
    parser.add_argument("--donor-composition", type=Path, default=None)
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    output = args.output.resolve()
    data_dir = output / "data"
    manifest_dir = output / "manifest"
    source_dir = output / "source_files"
    figure_dir = output / "figures"
    data_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    source_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    atlas = source_root / "10_results" / "human_thymic_aging_atlas"
    candidate_path = source_root / "10_results" / "final_candidate_panel.csv"
    attribution_path = source_root / "05_deep_learning" / "interpretation" / "gene_celltype_importance.csv"
    source_paths = {
        "gene_effects": atlas / "atlas_gene_age_effects.tsv.gz",
        "composition_effects": atlas / "atlas_composition_age_effects.tsv",
        "pathways": atlas / "atlas_pathways.tsv",
        "developmental_context": atlas / "atlas_developmental_context.tsv",
        "donor_manifest": atlas / "atlas_donor_manifest.tsv",
        "donor_scores": atlas / "atlas_donor_scores.tsv",
        "candidate_panel": candidate_path,
        "transformer_attribution": attribution_path,
        "server_source_manifest": atlas / "atlas_source_manifest.tsv",
    }
    missing = [str(path) for path in source_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required source files:\n" + "\n".join(missing))

    effects = pd.read_csv(source_paths["gene_effects"], sep="\t")
    effects["gene"] = effects["gene"].astype(str).str.upper()
    composition = pd.read_csv(source_paths["composition_effects"], sep="\t")
    pathways = pd.read_csv(source_paths["pathways"], sep="\t")
    developmental = pd.read_csv(source_paths["developmental_context"], sep="\t")
    manifest = pd.read_csv(source_paths["donor_manifest"], sep="\t")
    scores = pd.read_csv(source_paths["donor_scores"], sep="\t")
    candidate = pd.read_csv(candidate_path)
    attribution = pd.read_csv(attribution_path)

    gene_pathways = build_gene_pathway_links(pathways)
    gene_summary = build_gene_summary(effects, candidate, attribution, developmental, gene_pathways)
    cell_summary, cell_genes, cell_pathway_overlap = build_cell_type_outputs(
        effects, composition, attribution, gene_pathways
    )

    donor_composition_path = args.donor_composition
    if donor_composition_path is None:
        candidate_composition = source_root / "02_pseudobulk" / "donor_celltype_proportion_all.csv"
        donor_composition_path = candidate_composition if candidate_composition.exists() else None
    donor_summary, donor_composition = build_donor_outputs(
        manifest, scores, donor_composition_path
    )

    cohort_registry = pd.DataFrame(
        [
            {
                "cohort_id": "GSE231906",
                "role": "primary_discovery",
                "n_independent_donors": 18,
                "age_range": "4-69 years",
                "health_status_audit": "unresolved in reviewed public donor metadata",
                "eligible_for_model_training": True,
                "status": "included",
            },
            {
                "cohort_id": "E-MTAB-8581",
                "role": "candidate_external_biology",
                "n_independent_donors": pd.NA,
                "age_range": "prenatal, pediatric and adult",
                "health_status_audit": "pending donor-level audit",
                "eligible_for_model_training": False,
                "status": "not imported",
            },
            {
                "cohort_id": "GSE147520",
                "role": "candidate_external_biology",
                "n_independent_donors": 5,
                "age_range": "fetal to 25 years",
                "health_status_audit": "pending donor-level audit; one adult sample",
                "eligible_for_model_training": False,
                "status": "not imported",
            },
            {
                "cohort_id": "HRA007984",
                "role": "candidate_external_validation",
                "n_independent_donors": 16,
                "age_range": "prenatal to geriatric",
                "health_status_audit": "reported healthy; exact donor metadata audit pending",
                "eligible_for_model_training": False,
                "status": "not imported",
            },
        ]
    )

    outputs = {
        "gene_summary.csv.gz": gene_summary,
        "gene_context_effects.csv.gz": effects,
        "gene_pathway_links.csv.gz": gene_pathways,
        "developmental_context.csv.gz": developmental,
        "cell_type_summary.csv": cell_summary,
        "cell_type_genes.csv.gz": cell_genes,
        "cell_type_pathway_overlap.csv.gz": cell_pathway_overlap,
        "donor_summary.csv": donor_summary,
        "donor_composition.csv": donor_composition,
        "cohort_registry.csv": cohort_registry,
    }
    for name, frame in outputs.items():
        frame.to_csv(data_dir / name, index=False, compression="gzip" if name.endswith(".gz") else None)

    h5ad_path = data_dir / "human_thymic_aging_atlas_results.h5ad"
    write_h5ad(h5ad_path, effects, gene_summary, donor_summary, cell_summary)

    local_manifest_rows = []
    for logical_name, path in source_paths.items():
        local_manifest_rows.append(
            {
                "scope": "local_source",
                "logical_name": logical_name,
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "status": "verified_local_file",
            }
        )
    if donor_composition_path and donor_composition_path.exists():
        local_manifest_rows.append(
            {
                "scope": "local_source",
                "logical_name": "donor_composition",
                "path": str(donor_composition_path),
                "bytes": donor_composition_path.stat().st_size,
                "sha256": sha256(donor_composition_path),
                "status": "verified_local_file",
            }
        )
    else:
        local_manifest_rows.append(
            {
                "scope": "server_source",
                "logical_name": "donor_composition",
                "path": "/data/zxy/projects/human_thymus_age_ML_DL/02_pseudobulk/donor_celltype_proportion_all.csv",
                "bytes": pd.NA,
                "sha256": pd.NA,
                "status": "not_present_in_local_bundle",
            }
        )

    server_manifest = pd.read_csv(source_paths["server_source_manifest"], sep="\t")
    server_manifest = server_manifest.rename(columns={"module": "logical_name"})
    server_manifest.insert(0, "scope", "server_source")
    server_manifest["status"] = "hash_recorded_by_upstream_pipeline"
    combined_manifest = pd.concat(
        [pd.DataFrame(local_manifest_rows), server_manifest], ignore_index=True, sort=False
    )
    combined_manifest.to_csv(manifest_dir / "source_manifest.csv", index=False)

    release_metadata = {
        "atlas_name": "Human Thymic Aging Atlas",
        "release": "v1.0-18donor",
        "primary_cohort": "GSE231906",
        "n_donors": 18,
        "n_gene_context_rows": int(len(effects)),
        "n_unique_genes": int(effects["gene"].nunique()),
        "n_cell_contexts": int(effects["cell_context"].nunique()),
        "donor_composition_available": bool(donor_composition_path and donor_composition_path.exists()),
        "limitations": [
            "Captured-library composition follows nominal 6:4 CD45-positive/CD45-negative recombination and is not native-tissue composition.",
            "Donor deviation is descriptive and internally evaluated; it is not a clinical biological-age score.",
            "External human cohorts are registered but not combined until donor-level audit and overlap checks are complete.",
        ],
    }
    (manifest_dir / "release_metadata.json").write_text(
        json.dumps(release_metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    checks = {
        "gene_summary_has_FOXN1": bool(gene_summary["gene"].eq("FOXN1").any()),
        "gene_effect_rows": int(len(effects)),
        "unique_genes": int(effects["gene"].nunique()),
        "donors": int(donor_summary["donor_id"].nunique()),
        "cell_types": int(cell_summary["cell_type"].nunique()),
        "duplicate_gene_context_rows": int(effects.duplicated(["gene", "cell_context"]).sum()),
        "donor_composition_available": bool(donor_composition_path and donor_composition_path.exists()),
    }
    (manifest_dir / "build_checks.json").write_text(
        json.dumps(checks, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if checks["donors"] != 18 or checks["duplicate_gene_context_rows"] != 0:
        raise RuntimeError(f"Atlas integrity checks failed: {checks}")

    delivery_sources = [
        source_root / "manuscript_C000_atlas_revision" / "Figure3_final_OOF_source_data.xlsx",
        source_root / "manuscript_C000_atlas_revision" / "Figure6_source_data.xlsx",
        source_root / "manuscript_C000_atlas_revision" / "build_revision.py",
        source_root / "图片代码_第三轮修订" / "32_build_human_thymic_aging_atlas.py",
        source_root / "manuscript_D000_revision" / "revise_d000.py",
        source_root / "10_results" / "human_thymic_aging_atlas" / "atlas_source_manifest.tsv",
        source_root / "10_results" / "submission_completion" / "submission_source_data_manifest.tsv",
    ]
    for path in delivery_sources:
        if path.exists():
            shutil.copy2(path, source_dir / path.name)
    delivery_figures = [
        source_root / "manuscript_D000_revision" / "Figure3_workflow_and_results.png",
        source_root / "manuscript_D000_revision" / "Figure5_atlas_residual_revised.png",
    ]
    for path in delivery_figures:
        if path.exists():
            shutil.copy2(path, figure_dir / path.name)

    hash_targets = sorted(
        path for path in output.rglob("*") if path.is_file() and path.name != "SHA256SUMS.txt"
    )
    hash_lines = [f"{sha256(path)}  {path.relative_to(output).as_posix()}" for path in hash_targets]
    (manifest_dir / "SHA256SUMS.txt").write_text("\n".join(hash_lines) + "\n", encoding="utf-8")

    project_root = Path(__file__).resolve().parent
    archive_path = project_root / "Human_Thymic_Aging_Atlas_v1.0-18donor.zip"
    if archive_path.exists():
        archive_path.unlink()
    release_files = [path for path in output.rglob("*") if path.is_file()]
    code_files = [
        project_root / "app.py",
        project_root / "build_atlas.py",
        project_root / "export_server_inputs.py",
        project_root / "run_atlas.ps1",
        project_root / "requirements.txt",
        project_root / "README.md",
        project_root / ".streamlit" / "config.toml",
        project_root / "schemas" / "donor_manifest_template.csv",
        project_root / "schemas" / "gene_context_effects_template.csv",
        project_root / "schemas" / "README.md",
    ]
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in release_files + code_files:
            archive.write(path, arcname=path.relative_to(project_root).as_posix())
    print(json.dumps(checks, indent=2))
    print(output)


if __name__ == "__main__":
    main()

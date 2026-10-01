#!/usr/bin/env python3
"""Build frozen-model-compatible donor pseudobulk features from an H5AD file.

The program is intentionally fail-closed. Donor identities/ages and cell-type
crosswalks must be supplied in reviewed TSV files. Raw counts are required;
normalized or integrated values are rejected. External ages are copied to the
output manifest but are never used to construct expression features.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
from pathlib import Path
from typing import Iterable

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

APPROVED = {"APPROVED", "AUTHOR_APPROVED"}


def truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def examples(values: pd.Series, n: int = 8) -> str:
    vals = pd.Series(values).dropna().astype(str).drop_duplicates().head(n)
    return " | ".join(vals)


def write_tsv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, sep="\t", index=False)


def parse_matrix_source(adata: ad.AnnData, source: str):
    if source == "raw":
        if adata.raw is None:
            raise RuntimeError("--matrix-source raw requested but adata.raw is absent")
        return adata.raw.X, adata.raw.var.copy(), pd.Index(adata.raw.var_names)
    if source == "X":
        return adata.X, adata.var.copy(), pd.Index(adata.var_names)
    if source.startswith("layer:"):
        layer = source.split(":", 1)[1]
        if layer not in adata.layers:
            raise RuntimeError(f"Requested layer is absent: {layer}")
        return adata.layers[layer], adata.var.copy(), pd.Index(adata.var_names)
    raise ValueError("--matrix-source must be raw, X, or layer:<name>")


def sampled_nonzero_values(matrix, limit: int = 100_000) -> np.ndarray:
    if sparse.issparse(matrix):
        values = np.asarray(matrix.data)
    else:
        values = np.asarray(matrix).reshape(-1)
        values = values[values != 0]
    if values.size > limit:
        positions = np.linspace(0, values.size - 1, limit, dtype=int)
        values = values[positions]
    return values.astype(float, copy=False)


def verify_raw_counts(matrix) -> dict[str, object]:
    values = sampled_nonzero_values(matrix)
    if values.size == 0:
        raise RuntimeError("Selected expression matrix contains no non-zero values")
    finite = values[np.isfinite(values)]
    if finite.size != values.size or np.min(finite) < 0:
        raise RuntimeError("Count matrix contains non-finite or negative values")
    integer_fraction = float(np.mean(np.isclose(finite, np.round(finite), atol=1e-6)))
    audit = {
        "sampled_nonzero_values": int(values.size),
        "minimum_sampled_value": float(np.min(finite)),
        "maximum_sampled_value": float(np.max(finite)),
        "integer_like_fraction": integer_fraction,
    }
    if integer_fraction < 0.999:
        raise RuntimeError(
            "Selected matrix does not look like raw counts "
            f"(integer-like fraction={integer_fraction:.5f}). Choose raw or a counts layer."
        )
    return audit


def resolve_gene_symbols(var: pd.DataFrame, var_names: pd.Index, column: str | None) -> pd.Index:
    if column:
        if column not in var.columns:
            raise KeyError(f"Gene-symbol column not found in var: {column}")
        symbols = pd.Index(var[column].astype(str))
    else:
        symbols = pd.Index(var_names.astype(str))
    symbols = pd.Index([re.sub(r"\.\d+$", "", value.strip()) for value in symbols])
    if (symbols == "").any():
        raise RuntimeError("Empty gene symbols were found")
    return symbols


def validate_manifest(path: Path, cohort: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest = pd.read_csv(path, sep="\t", dtype=str).fillna("")
    required = {"external_donor_id", "donor_id", "include", "review_status"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Donor manifest lacks columns: {sorted(missing)}")
    manifest["include_bool"] = manifest["include"].map(truthy)
    approved = manifest[
        manifest.include_bool & manifest.review_status.str.upper().isin(APPROVED)
    ].copy()
    if approved.empty:
        raise RuntimeError("No donor manifest row is both include=TRUE and APPROVED")
    if approved.external_donor_id.duplicated().any():
        raise RuntimeError("Approved external_donor_id values must be unique")
    if "cohort" not in approved:
        approved["cohort"] = cohort
    if not (approved["cohort"].astype(str) == cohort).all():
        raise RuntimeError("Approved donor manifest rows do not match --cohort")
    return manifest, approved


def validate_cell_map(path: Path) -> pd.DataFrame:
    mapping = pd.read_csv(path, sep="\t", dtype=str, comment="#").fillna("")
    required = {"external_label", "broad_cell_type", "include", "review_status"}
    missing = required - set(mapping.columns)
    if missing:
        raise ValueError(f"Cell-type map lacks columns: {sorted(missing)}")
    if mapping.external_label.duplicated().any():
        raise RuntimeError("Cell-type map external_label values must be unique")
    mapping["include_bool"] = mapping["include"].map(truthy)
    return mapping


def one_value(frame: pd.DataFrame, column: str, donor: str) -> object:
    if column not in frame:
        return np.nan
    values = frame[column].replace("", np.nan).dropna().drop_duplicates()
    if len(values) > 1:
        raise RuntimeError(f"Inconsistent {column} entries for standardized donor {donor}")
    return values.iloc[0] if len(values) else np.nan


def sum_rows(matrix, rows: np.ndarray) -> np.ndarray:
    if rows.size == 0:
        raise RuntimeError("Cannot pseudobulk an empty cell set")
    block = matrix[rows, :]
    return np.asarray(block.sum(axis=0)).reshape(-1).astype(float, copy=False)


def log_cpm_for_genes(
    matrix,
    rows: np.ndarray,
    gene_to_indices: dict[str, np.ndarray],
    needed_genes: Iterable[str],
) -> dict[str, float]:
    sums = sum_rows(matrix, rows)
    library_size = float(sums.sum())
    if not np.isfinite(library_size) or library_size <= 0:
        raise RuntimeError("A donor/context pseudobulk has zero library size")
    result: dict[str, float] = {}
    for gene in needed_genes:
        indices = gene_to_indices.get(gene)
        if indices is not None:
            count = float(sums[indices].sum())
            result[gene] = float(np.log1p(count / library_size * 1e6))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5ad", type=Path, required=True)
    parser.add_argument("--metadata-h5ad", type=Path)
    parser.add_argument("--cell-metadata", type=Path)
    parser.add_argument("--cell-id-column", default="cell_barcode")
    parser.add_argument("--selected-dictionary", type=Path, required=True)
    parser.add_argument("--donor-manifest", type=Path, required=True)
    parser.add_argument("--celltype-map", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--donor-col", default="donor_id")
    parser.add_argument("--celltype-col", default="cell_type")
    parser.add_argument("--matrix-source", default="raw")
    parser.add_argument("--gene-symbol-column")
    parser.add_argument("--subset-column")
    parser.add_argument("--subset-value", action="append", default=[])
    parser.add_argument("--scope", choices=["full_thymus", "stromal_sensitivity"], default="full_thymus")
    parser.add_argument("--compatible-celltypes", default="TEC,Fibroblast")
    parser.add_argument("--min-cells-per-context", type=int, default=20)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    for path in (args.h5ad, args.selected_dictionary, args.donor_manifest, args.celltype_map):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.metadata_h5ad and args.cell_metadata:
        raise ValueError("Use only one of --metadata-h5ad and --cell-metadata")
    for optional_path in (args.metadata_h5ad, args.cell_metadata):
        if optional_path and not optional_path.is_file():
            raise FileNotFoundError(optional_path)

    adata = ad.read_h5ad(args.h5ad)
    n_count_cells_before_alignment = int(adata.n_obs)
    if args.metadata_h5ad:
        metadata_adata = ad.read_h5ad(args.metadata_h5ad, backed="r")
        metadata_obs = metadata_adata.obs.copy()
        metadata_obs.index = metadata_obs.index.astype(str)
        count_names = pd.Index(adata.obs_names.astype(str))
        keep = count_names.isin(metadata_obs.index)
        if not keep.any():
            raise RuntimeError("No cell barcodes overlap between count and metadata H5AD files")
        adata = adata[keep, :].copy()
        obs = metadata_obs.reindex(adata.obs_names.astype(str)).copy()
        metadata_source = str(args.metadata_h5ad.resolve())
        metadata_adata.file.close()
    elif args.cell_metadata:
        separator = "," if args.cell_metadata.suffix.lower() == ".csv" else "\t"
        metadata_obs = pd.read_csv(args.cell_metadata, sep=separator, dtype=str)
        if args.cell_id_column not in metadata_obs:
            raise KeyError(f"Cell metadata lacks ID column: {args.cell_id_column}")
        metadata_obs = metadata_obs.set_index(args.cell_id_column, verify_integrity=True)
        metadata_obs.index = metadata_obs.index.astype(str)
        count_names = pd.Index(adata.obs_names.astype(str))
        keep = count_names.isin(metadata_obs.index)
        if not keep.any():
            raise RuntimeError("No cell barcodes overlap between count H5AD and metadata table")
        adata = adata[keep, :].copy()
        obs = metadata_obs.reindex(adata.obs_names.astype(str)).copy()
        metadata_source = str(args.cell_metadata.resolve())
    else:
        obs = adata.obs.copy()
        metadata_source = "embedded adata.obs"
    obs.index = obs.index.astype(str)
    obs_audit = pd.DataFrame({
        "column": obs.columns,
        "dtype": [str(obs[col].dtype) for col in obs.columns],
        "n_unique": [int(obs[col].nunique(dropna=True)) for col in obs.columns],
        "example_values": [examples(obs[col]) for col in obs.columns],
    })
    write_tsv(obs_audit, args.output / "metadata_column_audit.tsv")
    if args.donor_col not in obs or args.celltype_col not in obs:
        raise KeyError(
            f"Required obs columns not found: donor={args.donor_col}, celltype={args.celltype_col}. "
            "Review metadata_column_audit.tsv."
        )

    cell_mask = np.ones(adata.n_obs, dtype=bool)
    if args.subset_column:
        if args.subset_column not in obs:
            raise KeyError(f"Subset column not found: {args.subset_column}")
        if not args.subset_value:
            raise ValueError("--subset-column requires at least one --subset-value")
        cell_mask &= obs[args.subset_column].astype(str).isin(args.subset_value).to_numpy()
    if not cell_mask.any():
        raise RuntimeError("No cells survived the requested dataset subset")

    _, approved_donors = validate_manifest(args.donor_manifest, args.cohort)
    donor_lookup = approved_donors.set_index("external_donor_id")["donor_id"].to_dict()
    external_ids = obs[args.donor_col].astype(str)
    observed_donor_audit = (
        external_ids[cell_mask].value_counts().rename_axis("external_donor_id")
        .reset_index(name="n_cells")
    )
    observed_donor_audit["manifest_donor_id"] = observed_donor_audit.external_donor_id.map(donor_lookup)
    observed_donor_audit["approved_for_analysis"] = observed_donor_audit.manifest_donor_id.notna()
    write_tsv(observed_donor_audit, args.output / "observed_donor_audit.tsv")
    missing_donors = sorted(set(donor_lookup) - set(external_ids[cell_mask]))
    if missing_donors:
        raise RuntimeError(
            "Approved manifest donor IDs were not found in the H5AD: " + ", ".join(missing_donors)
        )

    target_mask = cell_mask & external_ids.isin(donor_lookup).to_numpy()
    mapping = validate_cell_map(args.celltype_map)
    label_lookup = mapping.set_index("external_label")
    observed_labels = pd.Series(obs.loc[target_mask, args.celltype_col].astype(str).unique(), name="external_label")
    mapping_audit = observed_labels.to_frame().merge(mapping, on="external_label", how="left")
    mapping_audit["n_cells"] = mapping_audit.external_label.map(
        obs.loc[target_mask, args.celltype_col].astype(str).value_counts()
    )
    write_tsv(mapping_audit, args.output / "observed_celltype_mapping_audit.tsv")
    incomplete = mapping_audit[
        mapping_audit.review_status.fillna("").str.upper().map(lambda x: x not in APPROVED)
        | mapping_audit["include"].isna()
    ]
    if args.audit_only:
        print(f"Audit files written to {args.output}; no feature matrix was generated")
        return
    if not incomplete.empty:
        labels = incomplete.external_label.astype(str).tolist()
        raise RuntimeError(
            "Every observed target-donor cell label must be explicitly reviewed. "
            f"Unreviewed labels: {labels}. See observed_celltype_mapping_audit.tsv."
        )

    approved_cell_map = mapping[
        mapping.include_bool & mapping.review_status.str.upper().isin(APPROVED)
    ].set_index("external_label")["broad_cell_type"].to_dict()
    broad = obs[args.celltype_col].astype(str).map(approved_cell_map)
    analysis_mask = target_mask & broad.notna().to_numpy()
    if not analysis_mask.any():
        raise RuntimeError("No cells survived the reviewed donor and cell-type mappings")

    matrix, var, var_names = parse_matrix_source(adata, args.matrix_source)
    matrix_audit = verify_raw_counts(matrix)
    gene_symbols = resolve_gene_symbols(var, var_names, args.gene_symbol_column)
    matrix_audit.update({
        "cohort": args.cohort,
        "source_file": str(args.h5ad.resolve()),
        "source_file_sha256": sha256(args.h5ad),
        "metadata_source": metadata_source,
        "matrix_source": args.matrix_source,
        "n_cells_total": int(adata.n_obs),
        "n_count_cells_before_metadata_alignment": n_count_cells_before_alignment,
        "n_count_cells_after_metadata_alignment": int(adata.n_obs),
        "n_features_total": int(matrix.shape[1]),
        "n_cells_in_dataset_subset": int(cell_mask.sum()),
        "n_cells_target_donors": int(target_mask.sum()),
        "n_cells_mapped_for_analysis": int(analysis_mask.sum()),
        "scope": args.scope,
    })
    (args.output / "matrix_audit.json").write_text(
        json.dumps(matrix_audit, indent=2), encoding="utf-8"
    )

    dictionary = pd.read_csv(args.selected_dictionary, sep="\t", dtype=str).fillna("")
    required_dict = {"feature", "feature_type", "gene", "celltype"}
    if not required_dict.issubset(dictionary.columns):
        raise ValueError(f"Selected dictionary lacks: {sorted(required_dict - set(dictionary.columns))}")
    if dictionary.feature.duplicated().any():
        raise RuntimeError("Selected feature dictionary contains duplicate feature IDs")

    needed_genes = set(dictionary.loc[dictionary.gene.ne(""), "gene"])
    gene_to_indices: dict[str, np.ndarray] = {}
    gene_positions: dict[str, list[int]] = {}
    for index, symbol in enumerate(gene_symbols):
        if symbol in needed_genes:
            gene_positions.setdefault(symbol, []).append(index)
    for gene, positions in gene_positions.items():
        gene_to_indices[gene] = np.asarray(positions, dtype=int)

    compatible_celltypes = {
        item.strip() for item in args.compatible_celltypes.split(",") if item.strip()
    }
    compatibility_reason = []
    compatible = []
    for row in dictionary.itertuples(index=False):
        if args.scope == "full_thymus":
            is_compatible = True
            reason = "full-thymus feature definition"
        elif row.feature_type == "gene_celltype" and row.celltype in compatible_celltypes:
            is_compatible = True
            reason = "cell-context feature available in stromal dataset"
        else:
            is_compatible = False
            reason = "not comparable in a stromal-enriched dataset"
        compatible.append(is_compatible)
        compatibility_reason.append(reason)
    dictionary["scope_compatible"] = compatible
    dictionary["compatibility_reason"] = compatibility_reason
    dictionary["gene_present_in_export"] = dictionary.gene.map(lambda x: x in gene_to_indices if x else True)
    write_tsv(dictionary, args.output / "selected_feature_compatibility.tsv")

    standardized_donor = external_ids.map(donor_lookup)
    donor_ids = approved_donors.donor_id.drop_duplicates().tolist()
    feature_matrix = pd.DataFrame(np.nan, index=donor_ids, columns=dictionary.feature, dtype=float)
    coverage_rows: list[dict[str, object]] = []
    cell_rows: list[dict[str, object]] = []
    for donor in donor_ids:
        donor_rows = np.flatnonzero(analysis_mask & standardized_donor.eq(donor).to_numpy())
        if donor_rows.size == 0:
            raise RuntimeError(f"Approved donor has no mapped cells: {donor}")
        donor_broad = broad.iloc[donor_rows].astype(str)
        counts_by_type = donor_broad.value_counts()
        wt = (
            log_cpm_for_genes(matrix, donor_rows, gene_to_indices, needed_genes)
            if args.scope == "full_thymus" else {}
        )
        context_cache: dict[str, dict[str, float] | None] = {}
        contexts = dictionary.loc[
            dictionary.feature_type.eq("gene_celltype") & dictionary.scope_compatible,
            "celltype",
        ].drop_duplicates()
        for celltype in contexts:
            rows = donor_rows[donor_broad.to_numpy() == celltype]
            context_cache[celltype] = (
                log_cpm_for_genes(matrix, rows, gene_to_indices, needed_genes)
                if rows.size >= args.min_cells_per_context else None
            )
        for row in dictionary.itertuples(index=False):
            if not row.scope_compatible:
                continue
            value = np.nan
            if row.feature_type == "whole_thymus":
                value = wt.get(row.gene, np.nan)
            elif row.feature_type == "gene_celltype":
                context = context_cache.get(row.celltype)
                if context is not None:
                    value = context.get(row.gene, np.nan)
            elif row.feature_type == "proportion":
                value = float(counts_by_type.get(row.celltype, 0) / donor_rows.size)
            else:
                raise RuntimeError(f"Unknown feature_type: {row.feature_type}")
            feature_matrix.loc[donor, row.feature] = value
        coverage_rows.append({
            "cohort": args.cohort,
            "donor_id": donor,
            "scope": args.scope,
            "n_cells": int(donor_rows.size),
            "n_selected_features": int(dictionary.shape[0]),
            "n_scope_compatible_features": int(dictionary.scope_compatible.sum()),
            "n_observed_selected_features": int(feature_matrix.loc[donor].notna().sum()),
            "observed_fraction": float(feature_matrix.loc[donor].notna().mean()),
        })
        for celltype, n_cells in counts_by_type.items():
            cell_rows.append({"cohort": args.cohort, "donor_id": donor,
                              "broad_cell_type": celltype, "n_cells": int(n_cells)})

    standardized_rows = []
    for donor, group in approved_donors.groupby("donor_id", sort=False):
        row = {"cohort": args.cohort, "donor_id": donor, "scope": args.scope}
        for column in ("age_years", "age_lower", "age_upper", "sex", "health", "age_type", "source_url"):
            row[column] = one_value(group, column, donor)
        row["external_donor_ids"] = "|".join(group.external_donor_id.astype(str))
        row["n_cells_eligible"] = int(sum(r["n_cells"] for r in coverage_rows if r["donor_id"] == donor))
        standardized_rows.append(row)
    output_manifest = pd.DataFrame(standardized_rows)

    with gzip.open(args.output / "external_feature_matrix.tsv.gz", "wt", encoding="utf-8", newline="") as handle:
        feature_matrix.rename_axis("donor_id").reset_index().to_csv(handle, sep="\t", index=False, na_rep="NA")
    write_tsv(output_manifest, args.output / "external_donor_manifest.tsv")
    write_tsv(pd.DataFrame(coverage_rows), args.output / "pseudobulk_feature_coverage.tsv")
    write_tsv(pd.DataFrame(cell_rows), args.output / "celltype_counts_by_donor.tsv")

    cell_audit = pd.DataFrame({
        "cell_barcode": obs.index[analysis_mask],
        "external_donor_id": external_ids[analysis_mask].to_numpy(),
        "donor_id": standardized_donor[analysis_mask].to_numpy(),
        "external_celltype": obs.loc[analysis_mask, args.celltype_col].astype(str).to_numpy(),
        "broad_cell_type": broad[analysis_mask].to_numpy(),
    })
    with gzip.open(args.output / "cell_metadata_audit.tsv.gz", "wt", encoding="utf-8", newline="") as handle:
        cell_audit.to_csv(handle, sep="\t", index=False)
    print(
        f"Prepared {len(donor_ids)} {args.cohort} donors, {feature_matrix.shape[1]} frozen features; "
        f"scope={args.scope}"
    )


if __name__ == "__main__":
    main()

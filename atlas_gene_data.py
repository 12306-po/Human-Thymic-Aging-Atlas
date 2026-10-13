"""Gene expression references and explicitly audited corrected-age evidence."""
from __future__ import annotations

import gzip
from hashlib import sha256
import json
from pathlib import Path

import pandas as pd

EXPRESSION_COLUMNS = ["gene", "cell_context", "AveExpr", "n_donors", "analysis_level"]
DEVELOPMENT_COLUMNS = ["gene", "source_object", "developmental_stage", "n_cells",
                       "mean_expr", "median_expr", "pct_expr", "mean_zscore", "median_zscore", "evidence_scope"]


def gene_references(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    core = root / "release/core_v1/data"
    old = root / "release/data"
    index_path = core / "gene_index.csv"
    if not index_path.is_file():
        index_path = old / "gene_summary.csv.gz"
    index = pd.read_csv(index_path, usecols=["gene"]).dropna()
    index["gene"] = index.gene.astype(str).str.upper()
    expression_path = core / "gene_expression_reference.csv.gz"
    if not expression_path.is_file():
        expression_path = old / "gene_context_effects.csv.gz"
    expression = pd.read_csv(expression_path, usecols=EXPRESSION_COLUMNS)
    expression["gene"] = expression.gene.astype(str).str.upper()
    developmental_path = core / "gene_development_reference.csv.gz"
    if not developmental_path.is_file():
        developmental_path = old / "developmental_context.csv.gz"
    development = pd.read_csv(developmental_path, usecols=DEVELOPMENT_COLUMNS)
    development["gene"] = development.gene.astype(str).str.upper()
    index = pd.concat([index, expression[["gene"]], development[["gene"]]])
    index = index.drop_duplicates().sort_values("gene").reset_index(drop=True)
    # Only expression summaries and coverage are retained. Old beta, p/q and ML/DL
    # columns are not carried into these reference tables.
    return index, expression[EXPRESSION_COLUMNS], development[DEVELOPMENT_COLUMNS]


def reference_files(index: pd.DataFrame, expression: pd.DataFrame,
                    development: pd.DataFrame) -> dict[str, bytes]:
    prefix = "release/core_v1/data/"
    return {
        prefix + "gene_index.csv": index.to_csv(index=False).encode("utf-8"),
        prefix + "gene_expression_reference.csv.gz": gzip.compress(
            expression.to_csv(index=False).encode("utf-8"), mtime=0),
        prefix + "gene_development_reference.csv.gz": gzip.compress(
            development.to_csv(index=False).encode("utf-8"), mtime=0),
    }


def corrected_gene_effects(root: Path) -> pd.DataFrame | None:
    directory = root / "release/core_v1/gene_evidence"
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("donor17_age_years") != 1.75:
        raise ValueError("Corrected gene evidence lacks donor17=1.75 provenance")
    table = directory / "gene_age_effects.csv.gz"
    if sha256(table.read_bytes()).hexdigest() != manifest.get("effects_sha256"):
        raise ValueError("Corrected gene evidence hash does not match its manifest")
    frame = pd.read_csv(table)
    required = {"study_id", "gene", "atlas_level2", "beta", "p", "q", "n_donors"}
    if not required.issubset(frame):
        raise ValueError("Corrected gene evidence lacks required columns")
    if not frame.study_id.astype(str).eq("GSE231906").all():
        raise ValueError("Unexpected study in corrected gene evidence")
    frame["gene"] = frame.gene.astype(str).str.upper()
    return frame


def corrected_evidence_files(root: Path) -> dict[str, bytes]:
    if corrected_gene_effects(root) is None:
        return {}
    names = ["release/core_v1/gene_evidence/manifest.json",
             "release/core_v1/gene_evidence/gene_age_effects.csv.gz"]
    return {name: (root / name).read_bytes() for name in names}

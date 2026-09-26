#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 02 — Pair GSE231906 10X libraries + copy-consistency audit

- Scans $HUMAN_PRIMARY_RAW for 10x triplet files (matrix/features/barcodes).
- For each library, if multiple copies exist (extracted/ vs untar/ etc.),
  computes sha256 for each copy of matrix/features/barcodes and compares.
  * all copies identical  -> pick copy by priority (extracted > matrix > untar)
  * any copy differs      -> raise RuntimeError (never silently pick one)
- Emits 02_copy_consistency_audit.csv and the standard manifest outputs.

Paths come from environment variables (source 00_project_config.sh).
"""

import gzip
import hashlib
import logging
import os
import re
from pathlib import Path
from collections import defaultdict

import pandas as pd


# ============================================================
# Configuration — environment variables (fail-fast)
# ============================================================

def _env(key):
    val = os.environ.get(key)
    if not val:
        raise RuntimeError(
            f"Required env var {key} is not set. "
            f"Run: source scripts/00_project_config.sh"
        )
    return Path(val)


ROOT = _env("HUMAN_PRIMARY_RAW")   # /data/zxy/raw_data/GSE231906
PROJECT = _env("PROJ")

OUTDIR = PROJECT / "01_raw_processing" / "metadata"
OUTDIR.mkdir(parents=True, exist_ok=True)

LOGDIR = PROJECT / "10_results" / "logs"
LOGDIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOGDIR / "02_pairing.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)


# ============================================================
# Hash utility
# ============================================================

def sha256_file(path, chunk_size=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def gzip_content_sha256(path, chunk_size=1 << 20):
    """Hash the DECOMPRESSED content of a .gz file.

    Different gzip copies of identical data can carry different headers/mtimes,
    so the compressed-byte sha256 can spuriously differ. Copy consistency for
    10x .gz artifacts is therefore judged on decompressed content.
    """
    h = hashlib.sha256()
    with gzip.open(path, "rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


# ============================================================
# File patterns
# ============================================================

patterns = {

    "matrix":
        re.compile(
            r"^(?P<prefix>.+)_matrix\.mtx(?:\.gz)?$",
            re.I
        ),

    "features":
        re.compile(
            r"^(?P<prefix>.+)_features\.tsv(?:\.gz)?$",
            re.I
        ),

    "barcodes":
        re.compile(
            r"^(?P<prefix>.+)_barcodes\.tsv(?:\.gz)?$",
            re.I
        ),

    "contig":
        re.compile(
            r"^(?P<prefix>.+)_filtered_contig_annotations\.csv(?:\.gz)?$",
            re.I
        )
}


# ============================================================
# Functions
# ============================================================

def choose_path(paths):

    if not paths:
        return None

    def rank(path):

        s = str(path)

        # 优先使用 extracted 目录
        if "/GSE231906_RAW/extracted/" in s:
            priority = 0

        elif "/GSE231906_matrix/" in s:
            priority = 1

        elif "/GSE231906_RAW/untar/" in s:
            priority = 2

        else:
            priority = 3

        return (
            priority,
            len(s),
            s
        )

    return sorted(
        paths,
        key=rank
    )[0]


def verify_copy_consistency(paths, kind, library_prefix):
    """
    Compute hashes for every candidate copy of a 10x file (matrix/features/
    barcodes/contig).

    For .gz files BOTH the compressed-byte hash and the DECOMPRESSED content
    hash are recorded; copy consistency is judged on decompressed content (gzip
    headers/mtime can differ for identical data).

    A copy whose hash cannot be computed is a HARD FAILURE (never silently
    treated as consistent). Likewise any genuine content mismatch fails.

    This function does NOT raise: it returns (chosen_path, audit_rows,
    failure_reason). The caller writes the audit CSV first and only then raises
    if failure_reason is set, so the audit trail is always flushed to disk.
    """
    if not paths:
        return None, [], None

    compressed = {}
    content = {}
    hash_errors = {}
    for p in paths:
        try:
            compressed[p] = sha256_file(p)
        except Exception as exc:  # noqa: BLE001
            log.error("compressed sha256 failed for %s: %s", p, exc)
            compressed[p] = None
            hash_errors[p] = f"compressed_hash_error: {exc}"
        # decompressed content hash for gzip files
        if str(p).endswith(".gz"):
            try:
                content[p] = gzip_content_sha256(p)
            except Exception as exc:  # noqa: BLE001
                log.error("gzip content sha256 failed for %s: %s", p, exc)
                content[p] = None
                hash_errors[p] = f"content_hash_error: {exc}"
        else:
            content[p] = compressed.get(p)

    failure_reason = None
    # P1 (2026-09-15): any unreadable copy -> fail (do NOT compare only the
    # subset that hashed successfully).
    if any(v is None for v in content.values()) or any(v is None for v in compressed.values()):
        failure_reason = "hash computation failed for one or more copies"
    else:
        unique_content = {d for d in content.values()}
        if len(unique_content) > 1:
            failure_reason = "decompressed content sha256 differs across copies"

    # consistency is reported against decompressed content; None hash -> False
    consistent = failure_reason is None

    chosen = choose_path(paths)

    audit_rows = []
    for p in paths:
        audit_rows.append({
            "library_prefix": library_prefix,
            "file_type": kind,
            "copy_path": str(p),
            "compressed_sha256": compressed[p],
            "content_sha256": content[p],
            "hash_error": hash_errors.get(p, ""),
            "copy_consistent": consistent,
            "is_selected": (p == chosen),
        })

    if failure_reason is not None:
        log.error(
            "COPY-CONSISTENCY FAILURE: library=%s file_type=%s (%d copies): %s. "
            "Audit flushed to 02_copy_consistency_audit.csv; refusing to pick a "
            "copy silently.",
            library_prefix, kind, len(paths), failure_reason,
        )

    return chosen, audit_rows, failure_reason


def open_text(path):

    if str(path).endswith(".gz"):

        return gzip.open(
            path,
            "rt"
        )

    return open(
        path,
        "rt"
    )


def count_lines(path):

    if path is None:
        return None

    n = 0

    with open_text(path) as f:

        for _ in f:
            n += 1

    return n


def read_mtx_header(path):

    """
    只读取 Matrix Market 文件头，
    不把整个表达矩阵读入内存。

    返回：
        rows
        columns
        nnz
    """

    if path is None:
        return (
            None,
            None,
            None
        )

    with open_text(path) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            if line.startswith("%"):
                continue

            parts = line.split()

            if len(parts) >= 3:

                return (
                    int(parts[0]),
                    int(parts[1]),
                    int(parts[2])
                )

    return (
        None,
        None,
        None
    )


# ============================================================
# Scan files
# ============================================================

grouped = defaultdict(
    lambda: defaultdict(list)
)


for path in ROOT.rglob("*"):

    if not path.is_file():
        continue

    for kind, pattern in \
            patterns.items():

        match = pattern.match(
            path.name
        )

        if match:

            prefix = match.group(
                "prefix"
            )

            grouped[
                prefix
            ][
                kind
            ].append(
                path
            )

            break


print(
    "Unique prefixes:",
    len(grouped)
)


# ============================================================
# Build library manifest
# ============================================================

rows = []

copy_audit_rows = []
copy_failures = []  # (prefix, kind, reason)


def _flush_copy_audit():
    """Write the copy-consistency audit to disk (called before any fail-fast)."""
    if copy_audit_rows:
        pd.DataFrame(copy_audit_rows).to_csv(
            OUTDIR / "02_copy_consistency_audit.csv", index=False)


for prefix, files in \
        sorted(grouped.items()):

    matrix, _rows, _fail = verify_copy_consistency(
        files.get("matrix", []),
        "matrix",
        prefix,
    )
    copy_audit_rows.extend(_rows)
    if _fail:
        copy_failures.append((prefix, "matrix", _fail))

    features, _rows, _fail = verify_copy_consistency(
        files.get("features", []),
        "features",
        prefix,
    )
    copy_audit_rows.extend(_rows)
    if _fail:
        copy_failures.append((prefix, "features", _fail))

    barcodes, _rows, _fail = verify_copy_consistency(
        files.get("barcodes", []),
        "barcodes",
        prefix,
    )
    copy_audit_rows.extend(_rows)
    if _fail:
        copy_failures.append((prefix, "barcodes", _fail))

    contig, _rows, _fail = verify_copy_consistency(
        files.get("contig", []),
        "contig",
        prefix,
    )
    copy_audit_rows.extend(_rows)
    if _fail:
        copy_failures.append((prefix, "contig", _fail))

    # fail fast AFTER flushing the audit trail for any inconsistent library
    if copy_failures:
        _flush_copy_audit()
        raise RuntimeError(
            "10x copy-consistency failure(s); audit written to "
            "02_copy_consistency_audit.csv: "
            + "; ".join(f"{p}/{k}: {r}" for p, k, r in copy_failures)
        )


    # --------------------------------------------------------
    # GSM ID
    # --------------------------------------------------------

    gsm_match = re.search(
        r"(GSM\d+)",
        prefix,
        re.I
    )

    gsm_id = (
        gsm_match
        .group(1)
        .upper()
        if gsm_match
        else None
    )


    # --------------------------------------------------------
    # donor ID
    #
    # donor10-1:
    # donor_num = 10
    # library_within_donor = 1
    # --------------------------------------------------------

    donor_match = re.search(
        r"donor(\d+)(?:-(\d+))?",
        prefix,
        re.I
    )

    if donor_match:

        donor_num = int(
            donor_match.group(1)
        )

        if donor_match.group(2):

            library_within_donor = int(
                donor_match.group(2)
            )

        else:

            library_within_donor = 1

    else:

        donor_num = None
        library_within_donor = None


    # --------------------------------------------------------
    # Check completeness
    # --------------------------------------------------------

    complete = all([
        matrix is not None,
        features is not None,
        barcodes is not None
    ])


    # --------------------------------------------------------
    # Matrix dimensions
    # --------------------------------------------------------

    n_features = (
        count_lines(features)
        if features
        else None
    )

    n_barcodes = (
        count_lines(barcodes)
        if barcodes
        else None
    )

    (
        mtx_rows,
        mtx_cols,
        mtx_nnz
    ) = read_mtx_header(
        matrix
    )


    orientation = "NA"

    dimension_ok = False


    if (
        complete
        and
        None not in [
            n_features,
            n_barcodes,
            mtx_rows,
            mtx_cols
        ]
    ):

        # 标准10X：
        # rows = genes
        # columns = cells

        if (
            mtx_rows == n_features
            and
            mtx_cols == n_barcodes
        ):

            orientation = \
                "features_x_cells"

            dimension_ok = True


        elif (
            mtx_rows == n_barcodes
            and
            mtx_cols == n_features
        ):

            orientation = \
                "cells_x_features"

            dimension_ok = True


        else:

            orientation = \
                "dimension_mismatch"


    rows.append({

        "library_prefix":
            prefix,

        "gsm_id":
            gsm_id,

        "donor_num":
            donor_num,

        "donor_id":
            (
                f"donor{donor_num}"
                if donor_num
                is not None
                else None
            ),

        "library_within_donor":
            library_within_donor,

        "complete_10x_triplet":
            complete,

        "dimension_ok":
            dimension_ok,

        "orientation":
            orientation,

        "n_features":
            n_features,

        "n_barcodes":
            n_barcodes,

        "mtx_rows":
            mtx_rows,

        "mtx_cols":
            mtx_cols,

        "mtx_nnz":
            mtx_nnz,

        "matrix_path":
            (
                str(matrix)
                if matrix
                else None
            ),

        "features_path":
            (
                str(features)
                if features
                else None
            ),

        "barcodes_path":
            (
                str(barcodes)
                if barcodes
                else None
            ),

        # TCR/V(D)J信息
        # 当前不用于RNA模型训练
        "vdj_contig_path":
            (
                str(contig)
                if contig
                else None
            ),

        # 检查有没有重复拷贝
        "matrix_copy_count":
            len(
                files.get(
                    "matrix",
                    []
                )
            ),

        "features_copy_count":
            len(
                files.get(
                    "features",
                    []
                )
            ),

        "barcodes_copy_count":
            len(
                files.get(
                    "barcodes",
                    []
                )
            ),

        "contig_copy_count":
            len(
                files.get(
                    "contig",
                    []
                )
            )
    })


manifest = pd.DataFrame(
    rows
)


# ============================================================
# Copy-consistency audit output
# ============================================================

copy_audit = pd.DataFrame(copy_audit_rows)

if not copy_audit.empty:
    copy_audit = copy_audit.sort_values(
        ["library_prefix", "file_type", "copy_path"]
    ).reset_index(drop=True)

copy_audit.to_csv(
    OUTDIR / "02_copy_consistency_audit.csv",
    index=False
)

log.info(
    "Copy-consistency audit: %d copy rows written "
    "(all consistent — any conflict would have raised earlier).",
    len(copy_audit),
)


# ============================================================
# Save all libraries
# ============================================================

manifest.to_csv(

    OUTDIR
    / "02A_GSE231906_library_manifest_all.csv",

    index=False
)


# ============================================================
# Keep complete libraries
# ============================================================

complete = manifest[

    manifest[
        "complete_10x_triplet"
    ].fillna(False)

    &

    manifest[
        "dimension_ok"
    ].fillna(False)

].copy()


complete.to_csv(

    OUTDIR
    / "02B_GSE231906_complete_10x_libraries.csv",

    index=False
)


# ============================================================
# Primary cohort:
# donor 1–18
# ============================================================

primary = complete[

    complete[
        "donor_num"
    ].between(
        1,
        18,
        inclusive="both"
    )

].copy()


primary.to_csv(

    OUTDIR
    / "02C_GSE231906_primary_donor1_18_libraries.csv",

    index=False
)


# ============================================================
# donor summary
# ============================================================

donor_summary = (

    primary

    .groupby(
        [
            "donor_num",
            "donor_id"
        ],
        dropna=False
    )

    .agg(

        n_libraries=(
            "library_prefix",
            "size"
        ),

        total_cells=(
            "n_barcodes",
            "sum"
        ),

        total_nnz=(
            "mtx_nnz",
            "sum"
        )
    )

    .reset_index()

    .sort_values(
        "donor_num"
    )
)


donor_summary.to_csv(

    OUTDIR
    / "02D_GSE231906_primary_donor_summary.csv",

    index=False
)


# ============================================================
# Refined metadata candidates
# ============================================================

technical_pattern = re.compile(

    r"_matrix\.mtx"
    r"|_barcodes\.tsv"
    r"|_features\.tsv"
    r"|_filtered_contig_annotations\.csv",

    re.I
)


meta_rows = []


for path in ROOT.rglob("*"):

    if not path.is_file():
        continue


    # 排除RNA技术文件和VDJ文件
    if technical_pattern.search(
        path.name
    ):
        continue


    name = path.name.lower()


    if (

        any(
            keyword in name
            for keyword in [
                "meta",
                "sample",
                "donor",
                "age",
                "clinical",
                "phenotype",
                "series_matrix",
                "family",
                "geo"
            ]
        )

        or

        name.endswith(".rds")

        or

        name.endswith(".txt")

        or

        name.endswith(".txt.gz")

    ):

        meta_rows.append({

            "filename":
                path.name,

            "size_bytes":
                path.stat().st_size,

            "absolute_path":
                str(path)
        })


refined_metadata = pd.DataFrame(
    meta_rows
)


if not refined_metadata.empty:

    refined_metadata = (
        refined_metadata
        .drop_duplicates()
        .sort_values(
            "absolute_path"
        )
    )


refined_metadata.to_csv(

    OUTDIR
    / "02E_GSE231906_refined_metadata_candidates.csv",

    index=False
)


# ============================================================
# Summary
# ============================================================

summary = []


summary.append(
    "GSE231906 10X LIBRARY PAIRING"
)

summary.append(
    "=" * 72
)


summary.append(
    f"Unique library prefixes: "
    f"{len(manifest)}"
)


summary.append(
    f"Complete dimension-valid libraries: "
    f"{len(complete)}"
)


summary.append(
    f"Primary donor1-18 libraries: "
    f"{len(primary)}"
)


summary.append(
    f"Primary unique donors: "
    f"{len(donor_summary)}"
)


if not donor_summary.empty:

    donor_list = ",".join(

        donor_summary[
            "donor_num"
        ]

        .astype(str)

        .tolist()
    )


    summary.append(
        f"Donor numbers: "
        f"{donor_list}"
    )


summary.append("")

summary.append(
    "NOTE:"
)

summary.append(
    "donor1-18 is currently the selected primary cohort."
)

summary.append(
    "Age / sorting / platform will be confirmed from GEO metadata next."
)

summary.append(
    "filtered_contig_annotations.csv.gz is VDJ/TCR data, not RNA metadata."
)


summary_text = "\n".join(
    summary
)


summary_file = (

    OUTDIR
    / "02_GSE231906_pairing_summary.txt"
)


summary_file.write_text(

    summary_text,

    encoding="utf-8"
)


print()
print(
    summary_text
)


print()
print(
    "PRIMARY DONOR SUMMARY"
)

print(
    "=" * 72
)


if donor_summary.empty:

    print(
        "No donor1-18 libraries found."
    )

else:

    print(
        donor_summary
        .to_string(
            index=False
        )
    )


print()
print(
    "Saved to:"
)

print(
    OUTDIR
)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 01 — Human data inventory + provenance audit (READ-ONLY)

This script scans raw source directories, computes file hashes,
identifies duplicate copies, and audits copy consistency.
It NEVER moves, deletes, or renames any raw files.

Paths are read from environment variables (set by source 00_project_config.sh).
"""

from pathlib import Path
import os
import hashlib
import re
import logging
import pandas as pd
from collections import defaultdict

# ============================================================
# Paths — read from environment (fail-fast if missing)
# ============================================================

def _env(key):
    val = os.environ.get(key)
    if not val:
        raise RuntimeError(
            f"Required env var {key} is not set. "
            f"Run: source scripts/00_project_config.sh"
        )
    return Path(val)

PROJECT  = _env("PROJ")
GSE_ROOT = _env("HUMAN_PRIMARY_RAW")  # /data/zxy/raw_data/GSE231906

EXT_ROOTS = []
_ext = os.environ.get("HUMAN_EXTERNAL_RAW")
if _ext and Path(_ext).exists():
    EXT_ROOTS.append(Path(_ext))

ALL_RAW_ROOTS = [GSE_ROOT] + EXT_ROOTS

OUTDIR = PROJECT / "01_raw_processing" / "metadata"
OUTDIR.mkdir(parents=True, exist_ok=True)

LOGDIR = PROJECT / "10_results" / "logs"
LOGDIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOGDIR / "01_inventory.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)


# ============================================================
# Hash utilities
# ============================================================

def sha256_file(path, chunk_size=1 << 20):
    """Compute SHA-256 hex digest for a file (streaming, memory-safe)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


# ============================================================
# Utility functions (kept from original)
# ============================================================

def human_size(nbytes):
    if nbytes < 0:
        return "NA"
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if nbytes < 1024:
            return f"{nbytes:.2f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.2f} PB"


def classify_file(path):
    name = path.name.lower()
    if name.endswith(".h5ad"):          return "H5AD"
    if name.endswith(".h5seurat"):      return "H5SEURAT"
    if name.endswith(".rds"):           return "RDS"
    if name.endswith(".loom"):          return "LOOM"
    if name.endswith((".mtx", ".mtx.gz")): return "MTX"
    if name.endswith((".h5", ".hdf5")):
        if any(x in name for x in ["feature_bc_matrix", "filtered", "raw_feature", "matrix"]):
            return "10X_H5"
        return "HDF5"
    if name.endswith((".csv", ".csv.gz")):  return "CSV"
    if name.endswith((".tsv", ".tsv.gz")):  return "TSV"
    if name.endswith((".txt", ".txt.gz")):  return "TXT"
    if any(name.endswith(x) for x in [".tar.gz", ".tgz", ".tar", ".zip"]):
        return "ARCHIVE"
    if any(name.endswith(x) for x in [".fastq", ".fastq.gz", ".fq", ".fq.gz"]):
        return "FASTQ"
    return "OTHER"


def sample_guess(path):
    name = path.name
    for suffix in [
        ".fastq.gz", ".fq.gz", ".mtx.gz", ".tsv.gz", ".csv.gz", ".txt.gz",
        ".tar.gz", ".h5ad", ".h5seurat", ".loom", ".rds", ".hdf5", ".h5",
        ".mtx", ".csv", ".tsv", ".txt", ".zip",
    ]:
        if name.lower().endswith(suffix):
            name = name[: -len(suffix)]
            break
    name = re.sub(
        r"(_filtered_feature_bc_matrix|_raw_feature_bc_matrix|_matrix)$",
        "", name, flags=re.IGNORECASE,
    )
    return name


def metadata_candidate(path):
    if classify_file(path) not in ["CSV", "TSV", "TXT"]:
        return False
    keywords = [
        "meta", "metadata", "annotation", "annot", "donor", "sample", "age",
        "sex", "celltype", "cell_type", "cellinfo", "cell_info", "phenotype",
        "clinical", "cluster", "label",
    ]
    return any(x in path.name.lower() for x in keywords)


# ============================================================
# 1. Scan all files (+ sha256 hashing)
# ============================================================

rows = []

for root in ALL_RAW_ROOTS:

    log.info(f"[SCAN] {root}")

    if not root.exists():

        log.warning("directory does not exist: %s", root)

        continue

    for dirpath, dirnames, filenames in os.walk(root):

        dirnames[:] = [
            d for d in dirnames
            if d not in [".git", "__pycache__", ".ipynb_checkpoints"]
        ]

        for filename in filenames:

            path = Path(dirpath) / filename

            try:
                size = path.stat().st_size
            except Exception:
                size = -1

            try:
                digest = sha256_file(path)
            except Exception as exc:
                log.warning("sha256 failed for %s: %s", path, exc)
                digest = None

            rows.append({
                "dataset_root": str(root),
                "relative_path": str(path.relative_to(root)),
                "filename": path.name,
                "parent_directory": str(path.parent),
                "sample_guess": sample_guess(path),
                "file_class": classify_file(path),
                "size_bytes": size,
                "size_human": human_size(size),
                "sha256": digest,
                "absolute_path": str(path),
            })


inventory = pd.DataFrame(rows)

inventory.to_csv(OUTDIR / "01_human_public_data_file_inventory.csv", index=False)


# ============================================================
# 1b. Duplicate audit (READ-ONLY — no file is moved/deleted/renamed)
# ============================================================

def _preferred_rank(p):
    """Preferred-copy priority (consistent with Step02):
    extracted/ > GSE231906_matrix/ > untar/ > other."""
    s = str(p)
    if "/GSE231906_RAW/extracted/" in s:
        return 0
    if "/GSE231906_matrix/" in s:
        return 1
    if "/GSE231906_RAW/untar/" in s:
        return 2
    return 3


def _duplicate_decision(group_df):
    """
    Assign per-file decision within a duplicate group:
      preferred           = the file Step02 will actually consume
      redundant_identical = byte-identical copy of the preferred file
      conflict            = same basename/group key but different content
    Selection rule (kept consistent with Step02 priority):
      extracted/ > GSE231906_matrix/ > untar/ > other
    """
    def rank(p):
        return _preferred_rank(p)

    if group_df["sha256"].nunique() == 1:
        # all identical → pick one as preferred
        idx = group_df["absolute_path"].map(rank).idxmin()
        decision = ["redundant_identical"] * len(group_df)
        decision[group_df.index.get_loc(idx)] = "preferred"
        group_df = group_df.copy()
        group_df["decision"] = decision
        group_df["preferred_copy"] = group_df["absolute_path"] == group_df.loc[idx, "absolute_path"]
    else:
        # conflicting content → all are conflicts; human must resolve
        group_df = group_df.copy()
        group_df["decision"] = "conflict"
        group_df["preferred_copy"] = False
    return group_df


if not inventory.empty:
    dup_key = inventory.groupby(
        ["dataset_root", "filename", "file_class"], dropna=False
    )["sha256"].transform("size")
    # files that have a sibling with the same (root, filename) but different content
    dup_group = inventory[dup_key > 1].copy()

    if dup_group.empty:
        duplicate_audit = pd.DataFrame(
            columns=[
                "dataset_root", "filename", "file_class", "size_bytes", "sha256",
                "absolute_path", "duplicate_group", "n_copies", "decision",
                "preferred_copy",
            ]
        )
    else:
        duplicate_audit = dup_group.groupby(
            ["dataset_root", "filename", "file_class"], dropna=False
        ).apply(_duplicate_decision, include_groups=False).reset_index()
        # drop stray group-key/index columns leaked by groupby().apply()
        duplicate_audit = duplicate_audit.drop(
            columns=[c for c in duplicate_audit.columns if c.startswith("level_")],
            errors="ignore",
        )
        duplicate_audit["duplicate_group"] = (
            duplicate_audit["dataset_root"] + "::" +
            duplicate_audit["filename"] + "::" +
            duplicate_audit["file_class"]
        )
        duplicate_audit["n_copies"] = duplicate_audit.groupby("duplicate_group")["absolute_path"].transform("size")

    # P1 (2026-09-15): ALSO group by raw content hash so that identical content
    # stored under DIFFERENT filenames is detected. Same-name + different hash is
    # a conflict (handled above); same hash + different name is a redundant copy.
    hashed = inventory[inventory["sha256"].notna() & (inventory["sha256"] != "")]
    content_sizes = hashed.groupby("sha256")["absolute_path"].transform("size")
    content_dup = hashed[content_sizes > 1].copy()
    content_rows = []
    if not content_dup.empty:
        for digest, grp in content_dup.groupby("sha256"):
            names = grp["filename"].nunique()
            idx = grp["absolute_path"].map(_preferred_rank).idxmin()
            for i, (rid, r) in enumerate(grp.iterrows()):
                content_rows.append({
                    "dataset_root": r["dataset_root"],
                    "filename": r["filename"],
                    "file_class": r["file_class"],
                    "size_bytes": r["size_bytes"],
                    "sha256": digest,
                    "absolute_path": r["absolute_path"],
                    "duplicate_group": f"content::{digest[:16]}",
                    "n_copies": len(grp),
                    "decision": ("preferred" if rid == idx
                                 else "redundant_identical"),
                    "preferred_copy": bool(rid == idx),
                    "group_basis": ("content_and_name" if names == 1
                                    else "same_content_different_name"),
                })
    content_audit = pd.DataFrame(content_rows)

    duplicate_audit["group_basis"] = "same_name"
    if not content_audit.empty:
        # union: same-name audit + content groups not already captured by name
        duplicate_audit = pd.concat(
            [duplicate_audit, content_audit], ignore_index=True
        ).drop_duplicates(subset=["absolute_path", "duplicate_group"])

    duplicate_audit.to_csv(OUTDIR / "01_duplicate_audit.csv", index=False)
    if not content_audit.empty:
        content_audit.to_csv(OUTDIR / "01_content_duplicate_audit.csv", index=False)

    # Also emit a plain sha256 list (gitignore-friendly)
    with open(OUTDIR / "01_checksums_sha256.txt", "w", encoding="utf-8") as fh:
        for _, r in inventory.sort_values("absolute_path").iterrows():
            fh.write(f"{r['sha256']}  {r['absolute_path']}\n")

    n_conflicts = int((duplicate_audit["decision"] == "conflict").sum())
    n_diff_name = int((duplicate_audit["group_basis"]
                       == "same_content_different_name").sum()) if "group_basis" in duplicate_audit else 0
    if n_diff_name:
        log.info("Found %d files that are byte-identical copies under different names.",
                 n_diff_name)
    if n_conflicts:
        log.error("FOUND %d conflicting duplicate files — see 01_duplicate_audit.csv", n_conflicts)
    else:
        log.info("Duplicate audit clean (no content conflicts).")

else:
    log.warning("No files found; skipping duplicate audit.")


# ============================================================
# 2. Detect proper 10X MTX directories
# ============================================================

tenx_rows = []

for root in ALL_RAW_ROOTS:

    if not root.exists():
        continue

    for dirpath, dirnames, filenames \
            in os.walk(root):

        names = set(
            filenames
        )

        matrix_file = next(
            (
                x for x in [
                    "matrix.mtx.gz",
                    "matrix.mtx"
                ]
                if x in names
            ),
            None
        )

        feature_file = next(
            (
                x for x in [
                    "features.tsv.gz",
                    "features.tsv",
                    "genes.tsv.gz",
                    "genes.tsv"
                ]
                if x in names
            ),
            None
        )

        barcode_file = next(
            (
                x for x in [
                    "barcodes.tsv.gz",
                    "barcodes.tsv"
                ]
                if x in names
            ),
            None
        )

        if (
            matrix_file
            and feature_file
            and barcode_file
        ):

            d = Path(
                dirpath
            )

            tenx_rows.append({

                "dataset_root":
                    str(root),

                "sample_guess":
                    d.parent.name,

                "input_type":
                    "10X_MTX",

                "matrix_directory":
                    str(d),

                "matrix_file":
                    str(
                        d
                        / matrix_file
                    ),

                "feature_file":
                    str(
                        d
                        / feature_file
                    ),

                "barcode_file":
                    str(
                        d
                        / barcode_file
                    )
            })


tenx = pd.DataFrame(
    tenx_rows
)

tenx.to_csv(
    OUTDIR
    / "02_detected_10x_matrix_directories.csv",
    index=False
)


# ============================================================
# 1c. Copy-consistency audit (raw copy vs source)
#     READ-ONLY — mismatch ⇒ fail-fast (raise)
# ============================================================

def copy_consistency_audit():
    """
    Compare extracted/ vs untar/ copies of the same 10x triplet files.
    For each logical file (GSM_prefix + file_kind) that exists in more than
    one location, verify the SHA-256 digests agree.

    Returns a DataFrame of per-copy rows with a 'consistent' flag.
    Raises RuntimeError if any logical file has disagreeing copies.
    """
    if inventory.empty:
        return pd.DataFrame()

    # Identify 10x triplet files: matrix / features / barcodes / contig
    t10 = inventory[
        inventory["file_class"].isin(["MTX", "TSV", "CSV"])
        & inventory["filename"].str.contains(
            r"(_matrix\.mtx|_features\.tsv|_barcodes\.tsv|_filtered_contig_annotations\.csv)",
            regex=True, case=False, na=False,
        )
    ].copy()

    # logical key = (GSM prefix, file kind)
    def logical_key(row):
        name = row["filename"].lower()
        if "_matrix.mtx" in name:
            kind = "matrix"
        elif "_features.tsv" in name:
            kind = "features"
        elif "_barcodes.tsv" in name:
            kind = "barcodes"
        elif "_filtered_contig_annotations.csv" in name:
            kind = "contig"
        else:
            kind = "other"
        prefix = re.sub(r"_(matrix|features|barcodes|filtered_contig_annotations)\..*$", "", row["filename"], flags=re.I)
        return (prefix, kind)

    t10["logical_key"] = t10.apply(logical_key, axis=1)
    t10["n_copies"] = t10.groupby("logical_key")["absolute_path"].transform("size")

    # Only rows where the same logical file appears more than once
    multi = t10[t10["n_copies"] > 1].copy()

    if multi.empty:
        log.info("Copy-consistency audit: no duplicated 10x copies found.")
        return pd.DataFrame()

    # Consistency per logical_key: all sha256 equal?
    multi["consistent"] = multi.groupby("logical_key")["sha256"].transform(
        lambda s: s.nunique() == 1
    )

    audit = multi[
        [
            "logical_key", "dataset_root", "filename", "file_class",
            "size_bytes", "sha256", "absolute_path", "n_copies", "consistent",
        ]
    ].copy()
    audit["logical_key"] = audit["logical_key"].astype(str)
    audit.to_csv(OUTDIR / "01_copy_consistency_audit.csv", index=False)

    bad = audit[~audit["consistent"]]
    if not bad.empty:
        log.error(
            "COPY-CONSISTENCY FAILURE: %d files disagree with their copies. "
            "See 01_copy_consistency_audit.csv. Resolve before Step 02.",
            len(bad),
        )
        raise RuntimeError(
            f"Copy-consistency audit failed for {len(bad)} files "
            f"(see 01_copy_consistency_audit.csv)."
        )

    log.info("Copy-consistency audit clean (%d duplicated copies all identical).", len(audit))
    return audit


copy_audit = copy_consistency_audit()


# ============================================================
# 3. Candidate processed single-cell objects
# ============================================================

object_types = [
    "H5AD",
    "H5SEURAT",
    "RDS",
    "LOOM",
    "10X_H5"
]

if inventory.empty:

    objects = pd.DataFrame()

else:

    objects = inventory[
        inventory[
            "file_class"
        ].isin(
            object_types
        )
    ][
        [
            "dataset_root",
            "sample_guess",
            "file_class",
            "size_human",
            "absolute_path"
        ]
    ].copy()


objects.to_csv(
    OUTDIR
    / "03_candidate_singlecell_objects.csv",
    index=False
)


# ============================================================
# 4. Candidate metadata files
# ============================================================

meta_rows = []

if not inventory.empty:

    for _, row in \
            inventory.iterrows():

        path = Path(
            row[
                "absolute_path"
            ]
        )

        if metadata_candidate(
            path
        ):

            meta_rows.append({

                "dataset_root":
                    row[
                        "dataset_root"
                    ],

                "filename":
                    row[
                        "filename"
                    ],

                "file_class":
                    row[
                        "file_class"
                    ],

                "size_human":
                    row[
                        "size_human"
                    ],

                "absolute_path":
                    row[
                        "absolute_path"
                    ]
            })


metadata = pd.DataFrame(
    meta_rows
)

metadata.to_csv(
    OUTDIR
    / "04_candidate_metadata_files.csv",
    index=False
)


# ============================================================
# 5. Archive files
# ============================================================

if inventory.empty:

    archives = pd.DataFrame()

else:

    archives = inventory[
        inventory[
            "file_class"
        ] == "ARCHIVE"
    ][
        [
            "dataset_root",
            "filename",
            "size_human",
            "absolute_path"
        ]
    ].copy()


archives.to_csv(
    OUTDIR
    / "05_archive_files.csv",
    index=False
)


# ============================================================
# 6. FASTQ summary
# ============================================================

if inventory.empty:

    fastq = pd.DataFrame()

else:

    fastq = inventory[
        inventory[
            "file_class"
        ] == "FASTQ"
    ][
        [
            "dataset_root",
            "filename",
            "size_human",
            "absolute_path"
        ]
    ].copy()


fastq.to_csv(
    OUTDIR
    / "06_fastq_files.csv",
    index=False
)


# ============================================================
# 7. Summary
# ============================================================

summary = []

summary.append(
    "HUMAN THYMUS DATA INVENTORY"
)

summary.append(
    "=" * 70
)

summary.append("")


for root in ALL_RAW_ROOTS:

    summary.append(
        f"ROOT: {root}"
    )

    summary.append(
        f"EXISTS: {root.exists()}"
    )

    summary.append("")


summary.append(
    f"TOTAL FILES: {len(inventory)}"
)

summary.append("")


if not inventory.empty:

    summary.append(
        "FILE TYPES:"
    )

    counts = (
        inventory[
            "file_class"
        ]
        .value_counts()
    )

    for key, value \
            in counts.items():

        summary.append(
            f"{key}: {value}"
        )


summary.append("")

summary.append(
    f"10X MTX directories: {len(tenx)}"
)

summary.append(
    f"Candidate objects: {len(objects)}"
)

summary.append(
    f"Candidate metadata files: {len(metadata)}"
)

summary.append(
    f"Archive files: {len(archives)}"
)

summary.append(
    f"FASTQ files: {len(fastq)}"
)

summary.append("")

summary.append(
    "IMPORTANT:"
)

summary.append(
    "sample_guess is NOT final donor_id."
)

summary.append(
    "Formal donor/age/celltype metadata "
    "must be verified in Step 02."
)

summary.append("")

summary.append(
    "PROVENANCE AUDIT:"
)

if not inventory.empty:
    summary.append(
        f"sha256 computed for all {len(inventory)} files "
        f"(see 01_checksums_sha256.txt)"
    )

    n_dup = len(duplicate_audit)
    n_conf = int((duplicate_audit["decision"] == "conflict").sum()) if n_dup else 0
    n_red = int((duplicate_audit["decision"] == "redundant_identical").sum()) if n_dup else 0
    summary.append(
        f"duplicate files: {n_dup} "
        f"(conflict={n_conf}, redundant_identical={n_red})"
    )

    if copy_audit is not None and not copy_audit.empty:
        summary.append(
            f"copy-consistency: {len(copy_audit)} duplicated 10x copies verified identical"
        )
    else:
        summary.append(
            "copy-consistency: no duplicated 10x copies found"
        )

summary.append(
    "Step01 is READ-ONLY: no raw file was moved/deleted/renamed."
)


summary_file = (
    OUTDIR
    / "00_inventory_summary.txt"
)


summary_file.write_text(
    "\n".join(summary),
    encoding="utf-8"
)


# ============================================================
# 8. Console report
# ============================================================

print()
print("=" * 80)
print("STEP 01 INVENTORY COMPLETE")
print("=" * 80)

print(
    "Total files:",
    len(inventory)
)

print(
    "10X MTX directories:",
    len(tenx)
)

print(
    "Candidate objects:",
    len(objects)
)

print(
    "Candidate metadata:",
    len(metadata)
)

print(
    "Archives:",
    len(archives)
)

print(
    "FASTQ:",
    len(fastq)
)


if not objects.empty:

    print()
    print(
        "Candidate single-cell objects:"
    )

    print(
        objects
        .head(30)
        .to_string(
            index=False
        )
    )


if not tenx.empty:

    print()
    print(
        "Detected 10X directories:"
    )

    print(
        tenx
        .head(30)
        .to_string(
            index=False
        )
    )


if not metadata.empty:

    print()
    print(
        "Candidate metadata files:"
    )

    print(
        metadata
        .head(50)
        .to_string(
            index=False
        )
    )


print()
print(
    "Results saved to:"
)

print(
    OUTDIR
)

print("=" * 80)

# P1 (2026-09-16): formal provenance gate. All audits/CSVs/summaries are now on
# disk; raise LAST so nothing is silently skipped, but Step02 must not run while
# same-name/different-content conflicts remain unresolved.
if not inventory.empty and len(duplicate_audit):
    _n_conflict_gate = int((duplicate_audit["decision"] == "conflict").sum())
    if _n_conflict_gate:
        raise RuntimeError(
            f"Raw provenance conflict detected ({_n_conflict_gate} conflicting "
            "files). All audits were written; resolve "
            "01_duplicate_audit.csv (decision=conflict) before running Step02."
        )

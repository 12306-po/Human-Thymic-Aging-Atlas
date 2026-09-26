#!/usr/bin/env python
"""Step 18a: Human -> Mouse ortholog mapping (reconciled mygene + Ensembl BioMart).

Two independent sources are queried; raw results are written to disk immediately
after each remote query so a network interruption never loses prior work, and the
script is resumable (already-queried genes are skipped on restart).

  * mygene.info        : human symbol -> homologene -> mouse (taxid 10090) NCBI ids
                         -> mouse symbol.
  * Ensembl BioMart    : human symbol -> mouse homolog (ensembl id, associated gene
                         name, orthology type, %id, orthology confidence, GOC score).

P0/P1 fixes per Codex review 2026-09-14:
  - P0: DL top-100 genes are read from the FROZEN 06_interpretation/16_DL_top100_
    genes.txt (Step 16 output); head(100) is not recomputed here.
  - P1: best-ortholog ranking uses a priority tuple
    (orthology_type_one2one > orthology_confidence > both-sources > %identity >
     GOC > symbol_match tie-break); the previous "symbol_match x 100000" is gone.
  - P1: 18a_unmapped_human_genes.csv is written for every gene without a best
    ortholog (human_gene / in_ML_consensus / in_DL_top100 / mygene_found /
    biomart_found / status / reason).

Inputs:
  * 04_machine_learning/ML_consensus_genes.txt
  * 06_interpretation/16_DL_top100_genes.txt  (frozen from Step 16)

Outputs (08_mouse_validation/orthologs/):
  * human_mouse_ortholog_mygene_raw.csv
  * human_mouse_ortholog_biomart_raw.csv
  * human_mouse_orthologs_reconciled.csv
  * 18a_unmapped_human_genes.csv
  * ortholog_mapping_summary.txt
"""
from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

PROJ = Path(os.environ["PROJ"])
OUT = PROJ / "08_mouse_validation" / "orthologs"
OUT.mkdir(parents=True, exist_ok=True)
LOGS = PROJ / "10_results" / "logs"
LOGS.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("step18a")
logger.setLevel(logging.INFO)
_ch = logging.StreamHandler(sys.stdout)
_fh = logging.FileHandler(LOGS / "18a_ortholog_mapping.log", mode="w")
_fmt = logging.Formatter("%(asctime)s  %(levelname)s  %(message)s")
_ch.setFormatter(_fmt)
_fh.setFormatter(_fmt)
logger.addHandler(_ch)
logger.addHandler(_fh)
logger.info("=== Step 18a: Human->Mouse ortholog mapping (mygene + BioMart, reconciled) ===")

MYGENE = "https://mygene.info/v3/query"
BIOMART_MIRRORS = [
    "https://useast.ensembl.org/biomart/martservice",
    "https://www.ensembl.org/biomart/martservice",
    "https://asia.ensembl.org/biomart/martservice",
]
BIOMART_CHUNK = 100


def mygene_query(ids, scopes, species, fields):
    """Batch mygene query; returns list of hit dicts."""
    r = requests.post(
        MYGENE,
        data={
            "q": json.dumps(list(ids)),
            "scopes": scopes,
            "species": species,
            "fields": fields,
        },
        timeout=90,
    )
    r.raise_for_status()
    return r.json()


def biomart_query(symbols, retries=3):
    """Query Ensembl BioMart martservice for human->mouse homologs.

    Returns raw TSV text (header + rows) or None on total failure.
    Tries mirrors in order; bounded retries so we never hang indefinitely.
    """
    if not symbols:
        return None
    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE Query>
<Query virtualSchemaName="default" formatter="TSV" header="1" uniqueRows="1" count="" datasetConfigVersion="0.6">
  <Dataset name="hsapiens_gene_ensembl" interface="default">
    <Filter name="external_gene_name" value="{','.join(symbols)}"/>
    <Attribute name="external_gene_name"/>
    <Attribute name="mmusculus_homolog_ensembl_gene"/>
    <Attribute name="mmusculus_homolog_associated_gene_name"/>
    <Attribute name="mmusculus_homolog_orthology_type"/>
    <Attribute name="mmusculus_homolog_perc_id"/>
    <Attribute name="mmusculus_homolog_perc_id_r1"/>
    <Attribute name="mmusculus_homolog_orthology_confidence"/>
    <Attribute name="mmusculus_homolog_goc_score"/>
  </Dataset>
</Query>"""

    last_err = None
    for attempt in range(retries):
        for url in BIOMART_MIRRORS:
            try:
                r = requests.post(url, data={"query": xml}, timeout=60)
                if r.status_code == 200 and not r.text.startswith("Query ERROR"):
                    return r.text
                if r.status_code != 200:
                    last_err = f"{url} -> HTTP {r.status_code}"
                else:
                    last_err = f"{url} -> {r.text.strip()[:200]}"
            except Exception as e:  # noqa: BLE001 - mirror fallback needs broad catch
                last_err = f"{url} -> {type(e).__name__}: {e}"
        if attempt < retries - 1:
            logger.warning("  BioMart attempt %d failed (%s); retrying...", attempt + 1, last_err)
            time.sleep(5 * (attempt + 1))
    logger.error("  BioMart query FAILED after %d attempts: %s", retries, last_err)
    return None


def is_standard_symbol(s):
    """HGNC-like: skip contigs (AC*, AL*, AP*, CT*), versioned, underscored."""
    if "." in s or "_" in s:
        return False
    if not re.match(r"^[A-Z][A-Za-z0-9-]*$", s):
        return False
    if re.match(r"^(AC|AL|AP|CT)[0-9]", s):
        return False
    if re.search(r"orf\d", s):
        return False
    return True


# ----------------------------------------------------------------------
# 1. Candidate universe
# ----------------------------------------------------------------------
ml_genes = [
    ln.strip()
    for ln in (PROJ / "04_machine_learning" / "ML_consensus_genes.txt").read_text().splitlines()
    if ln.strip()
]
# P0: DL top-100 read from the Step 16 FROZEN file (never recomputed here)
dl_top100_file = PROJ / "06_interpretation" / "16_DL_top100_genes.txt"
if not dl_top100_file.exists():
    raise FileNotFoundError(
        f"Frozen DL top-100 missing: {dl_top100_file} — run Step 16 (rev.2) first.")
dl_top100 = [ln.strip() for ln in dl_top100_file.read_text().splitlines() if ln.strip()]
logger.info("DL top-100 read from frozen %s (%d genes)", dl_top100_file.name, len(dl_top100))
gene_universe = sorted(set(ml_genes) | set(dl_top100))
ml_set, dl_set = set(ml_genes), set(dl_top100)

standard_symbols = [g for g in gene_universe if is_standard_symbol(g)]
nonstandard = sorted(set(gene_universe) - set(standard_symbols))
logger.info("ML genes: %d, DL top100: %d, union: %d", len(ml_genes), len(dl_top100), len(gene_universe))
logger.info("Standard symbols queried: %d; skipped non-standard: %d", len(standard_symbols), len(nonstandard))

# ----------------------------------------------------------------------
# 2. Source A: mygene (symbol -> homologene -> mouse symbol)
# ----------------------------------------------------------------------
def _mygene_query_pairs(query_symbols):
    """Query mygene homologene for a list of human symbols -> resolved pairs."""
    if not query_symbols:
        return pd.DataFrame(columns=[
            "human_gene", "human_symbol", "homologene_id",
            "mouse_ncbi_id", "mouse_gene"])
    hits = mygene_query(sorted(query_symbols), "symbol", "human",
                        "symbol,homologene")
    query_to_hit = {h.get("query"): h for h in hits if h}
    rows = []
    for g in query_symbols:
        h = query_to_hit.get(g, {})
        human_symbol = h.get("symbol")
        hg = h.get("homologene")
        if not hg:
            continue
        homologene_id = hg.get("id")
        mouse_ids = [gid for tid, gid in hg.get("genes", []) if tid == 10090]
        for mid in mouse_ids:
            rows.append({
                "human_gene": g,
                "human_symbol": human_symbol,
                "homologene_id": homologene_id,
                "mouse_ncbi_id": str(mid),
            })
    pairs = pd.DataFrame(rows)
    if pairs.empty:
        return pairs
    # resolve mouse NCBI id -> symbol
    all_mouse_ids = sorted(pairs["mouse_ncbi_id"].astype(str).unique())
    logger.info("Distinct mouse NCBI ids to resolve: %d", len(all_mouse_ids))
    mouse_hits = mygene_query(all_mouse_ids, "entrezgene", "mouse", "symbol")
    ncbi_to_symbol = {str(h.get("query")): h.get("symbol") for h in mouse_hits if h}
    pairs["mouse_gene"] = pairs["mouse_ncbi_id"].astype(str).map(ncbi_to_symbol)
    pairs = pairs.dropna(subset=["mouse_gene", "human_symbol"])
    return pairs


MYGENE_RAW = OUT / "human_mouse_ortholog_mygene_raw.csv"
# Resume like BioMart: load cache (a superset from prior universes), then query
# only current-universe standard symbols not already present, and append.
if MYGENE_RAW.exists():
    mygene_pairs = pd.read_csv(MYGENE_RAW, dtype={"mouse_ncbi_id": str})
    cached_symbols = set(mygene_pairs["human_gene"].unique())
    todo_mg = [g for g in standard_symbols if g not in cached_symbols]
    logger.info("mygene raw exists (%d genes); resuming %d missing",
                len(cached_symbols), len(todo_mg))
    if todo_mg:
        new_pairs = _mygene_query_pairs(todo_mg)
        if not new_pairs.empty:
            mygene_pairs = pd.concat(
                [mygene_pairs, new_pairs], ignore_index=True)
            mygene_pairs.to_csv(MYGENE_RAW, index=False)
            logger.info("mygene resume added %d pairs (cache now %d rows)",
                        len(new_pairs), len(mygene_pairs))
else:
    mygene_pairs = _mygene_query_pairs(standard_symbols)
    mygene_pairs.to_csv(MYGENE_RAW, index=False)
    logger.info("Wrote mygene raw: %d pairs", len(mygene_pairs))

# ----------------------------------------------------------------------
# 3. Source B: BioMart (human symbol -> mouse homolog)
# ----------------------------------------------------------------------
BIOMART_RAW = OUT / "human_mouse_ortholog_biomart_raw.csv"
BIOMART_COLS = [
    "human_gene", "mouse_ensembl_gene", "mouse_gene", "orthology_type",
    "perc_id", "perc_id_r1", "orthology_confidence", "goc_score",
]

already_done = set()
if BIOMART_RAW.exists():
    done_df = pd.read_csv(BIOMART_RAW)
    already_done = set(done_df["human_gene"].unique())
    logger.info("BioMart raw exists with %d genes; will resume remaining", len(already_done))

todo = [g for g in standard_symbols if g not in already_done]
logger.info("BioMart genes to query: %d / %d", len(todo), len(standard_symbols))

if todo:
    # create header if the file is fresh
    if not BIOMART_RAW.exists():
        pd.DataFrame(columns=BIOMART_COLS).to_csv(BIOMART_RAW, index=False)

    n_fail = 0
    for i in range(0, len(todo), BIOMART_CHUNK):
        chunk = todo[i : i + BIOMART_CHUNK]
        txt = biomart_query(chunk)
        if txt is None:
            n_fail += len(chunk)
            logger.warning("  chunk %d-%d failed; continuing with next chunk", i, i + len(chunk))
            continue
        try:
            df = pd.read_csv(pd.io.common.StringIO(txt), sep="\t")
            df = df.rename(columns={
                "Gene name": "human_gene",
                "Mouse gene stable ID": "mouse_ensembl_gene",
                "Mouse gene name": "mouse_gene",
                "Mouse homology type": "orthology_type",
                "%id. target Mouse gene identical to query gene": "perc_id",
                "%id. query gene identical to target Mouse gene": "perc_id_r1",
                "Mouse orthology confidence [0 low, 1 high]": "orthology_confidence",
                "Mouse Gene-order conservation score": "goc_score",
            })
            df = df[BIOMART_COLS]
            df.to_csv(BIOMART_RAW, mode="a", header=False, index=False)
            logger.info("  chunk %d-%d: %d rows saved", i, i + len(chunk), len(df))
        except Exception as e:  # noqa: BLE001
            n_fail += len(chunk)
            logger.warning("  chunk %d-%d parse failed (%s)", i, i + len(chunk), e)
    if n_fail:
        logger.warning("BioMart: %d genes failed to query", n_fail)

biomart_pairs = pd.read_csv(BIOMART_RAW)
biomart_pairs = biomart_pairs.dropna(subset=["mouse_gene"])
logger.info("BioMart raw pairs (with mouse gene name): %d", len(biomart_pairs))

# P1 (2026-09-15): the raw caches are resume caches and may retain pairs from a
# previous, larger candidate universe. Restrict BOTH sources to the CURRENT
# frozen gene universe before reconciling, so the reconciled best-ortholog table
# never contains stale genes. Raw cache files on disk are left untouched.
_univ = set(gene_universe)
_n_mg0, _n_bm0 = len(mygene_pairs), len(biomart_pairs)
mygene_pairs = mygene_pairs[mygene_pairs["human_gene"].isin(_univ)].copy()
biomart_pairs = biomart_pairs[biomart_pairs["human_gene"].isin(_univ)].copy()
logger.info(
    "Restricted ortholog caches to current universe (%d genes): "
    "mygene %d -> %d rows; biomart %d -> %d rows",
    len(_univ), _n_mg0, len(mygene_pairs), _n_bm0, len(biomart_pairs))

# ----------------------------------------------------------------------
# 4. Reconcile
# ----------------------------------------------------------------------
mg = mygene_pairs[["human_gene", "mouse_gene"]].copy()
mg["mouse_gene_lower"] = mg["mouse_gene"].str.lower()
mg = mg.drop_duplicates(subset=["human_gene", "mouse_gene_lower"])

bm = biomart_pairs.copy()
bm["mouse_gene_lower"] = bm["mouse_gene"].str.lower()

# union of (human_gene, mouse_gene_lower) pairs across both sources
merged = pd.merge(mg, bm, on=["human_gene", "mouse_gene_lower"], how="outer", suffixes=("_mg", "_bm"))
merged["mouse_gene"] = merged["mouse_gene_bm"].fillna(merged["mouse_gene_mg"])
merged["in_mygene"] = merged["mouse_gene_mg"].notna()
merged["in_biomart"] = merged["mouse_gene_bm"].notna()

# ranking for best-ortholog selection — P1: priority tuple
#   one2one > orthology_confidence > both-sources > %identity > GOC > symbol_match
# (the old "symbol_match x 100000" weighting is removed; symbol match is a
#  tie-break only, never a dominant term)
merged["symbol_match"] = merged["mouse_gene"].str.lower() == merged["human_gene"].str.lower()
conf = pd.to_numeric(merged["orthology_confidence"], errors="coerce").fillna(-1.0)
perc = pd.to_numeric(merged["perc_id"], errors="coerce").fillna(-1.0)
goc = pd.to_numeric(merged["goc_score"], errors="coerce").fillna(-1.0)
merged["_one2one"] = (merged["orthology_type"] == "ortholog_one2one").astype(int)
merged["_both_sources"] = (merged["in_mygene"] & merged["in_biomart"]).astype(int)
merged["_rank_conf"] = conf
merged["_rank_perc"] = perc
merged["_rank_goc"] = goc
merged["_rank_symbol"] = merged["symbol_match"].astype(int)
merged = merged.sort_values(
    ["human_gene", "_one2one", "_rank_conf", "_both_sources",
     "_rank_perc", "_rank_goc", "_rank_symbol"],
    ascending=[True, False, False, False, False, False, False],
)
merged = merged.drop(columns=["_one2one", "_both_sources", "_rank_conf",
                              "_rank_perc", "_rank_goc", "_rank_symbol"])
merged["is_best_ortholog"] = ~merged.duplicated(subset=["human_gene"])

# Acceptance (2026-09-15 review): a human candidate must map to AT MOST ONE
# frozen best ortholog. `duplicated(...)` already selects exactly one best row
# per human_gene present; assert the invariant explicitly so Steps 18/19/20 can
# rely on one-to-one frozen mapping (never re-select by mouse effect downstream).
best = merged.loc[merged["is_best_ortholog"], "human_gene"]
if best.duplicated().any():
    dup = best[best.duplicated()].unique().tolist()
    raise RuntimeError(f"Multiple best orthologs for human genes: {dup[:10]}")

# P1 (2026-09-16): ortholog_priority is recomputed AFTER sorting (and kept
# here, before the final column selection). The earlier `conf` Series is aligned
# to the pre-sort index; np.where is positional and would mislabel rows after
# sort_values. Re-derive the confidence from the current (sorted/merged) frame.
merged["in_ML_consensus"] = merged["human_gene"].isin(ml_set)
merged["in_DL_top100"] = merged["human_gene"].isin(dl_set)

# homologene_id / mouse_ncbi_id only exist on the mygene side; merge them back in
mg_ids = mygene_pairs[["human_gene", "mouse_gene", "homologene_id", "mouse_ncbi_id"]].copy()
mg_ids["mouse_gene_lower"] = mg_ids["mouse_gene"].str.lower()
mg_ids = mg_ids.drop(columns=["mouse_gene"]).drop_duplicates(subset=["human_gene", "mouse_gene_lower"])
merged = merged.merge(mg_ids, on=["human_gene", "mouse_gene_lower"], how="left")
n_symbol_match = int(merged["symbol_match"].sum())

# P1: provenance label recomputed on the FINAL row order (post-sort/merge).
_conf_final = pd.to_numeric(
    merged["orthology_confidence"], errors="coerce").fillna(-1.0)
merged["ortholog_priority"] = np.where(
    merged["symbol_match"], "symbol_match",
    np.where(_conf_final >= 1.0, "high_confidence", "homolog_only"),
)
# audit columns recording the actual ranking drivers (not just the label)
merged["priority_one2one"] = (merged["orthology_type"] == "ortholog_one2one")
merged["priority_both_sources"] = merged["in_mygene"] & merged["in_biomart"]

out_cols = [
    "human_gene", "mouse_gene", "is_best_ortholog", "ortholog_priority",
    "in_mygene", "in_biomart",
    "homologene_id", "mouse_ncbi_id", "mouse_ensembl_gene",
    "orthology_type", "perc_id", "perc_id_r1", "orthology_confidence", "goc_score",
    "in_ML_consensus", "in_DL_top100",
]
merged = merged[[c for c in out_cols if c in merged.columns]]

reconciled_path = OUT / "human_mouse_orthologs_reconciled.csv"
merged.to_csv(reconciled_path, index=False)

# ----------------------------------------------------------------------
# 5. Summary
# ----------------------------------------------------------------------
n_genes = merged["human_gene"].nunique()
n_best = int(merged["is_best_ortholog"].sum())
n_both = int((merged["in_mygene"] & merged["in_biomart"]).sum())
n_mygene_only = int((merged["in_mygene"] & ~merged["in_biomart"]).sum())
n_biomart_only = int((merged["in_biomart"] & ~merged["in_mygene"]).sum())
summary = (
    f"union genes: {len(gene_universe)}\n"
    f"standard symbols queried: {len(standard_symbols)}\n"
    f"non-standard skipped: {len(nonstandard)}\n"
    f"genes with >=1 ortholog (reconciled): {n_genes}\n"
    f"best-ortholog rows: {n_best}\n"
    f"total ortholog pairs: {len(merged)}\n"
    f"pairs in both sources: {n_both}\n"
    f"pairs mygene-only: {n_mygene_only}\n"
    f"pairs biomart-only: {n_biomart_only}\n"
    f"symbol-match pairs: {n_symbol_match}\n"
    f"unmapped: {len(standard_symbols) - n_genes}\n"
)
(OUT / "ortholog_mapping_summary.txt").write_text(summary)
logger.info("\n%s", summary)

# P1: unmapped genes CSV (human_gene / membership / source findings / status)
mapped_symbols = set(merged["human_gene"].unique())
unmapped_rows = []
for g in sorted(gene_universe):
    if g in mapped_symbols:
        continue
    mg_hit = bool(mygene_pairs["human_gene"].eq(g).any()) if not mygene_pairs.empty else False
    bm_hit = bool(biomart_pairs["human_gene"].eq(g).any()) if not biomart_pairs.empty else False
    reason = []
    if not is_standard_symbol(g):
        reason.append("non_standard_symbol")
    if not mg_hit and not bm_hit:
        reason.append("no_ortholog_in_any_source")
    elif mg_hit and not bm_hit:
        reason.append("mygene_hit_but_no_mouse_gene_resolved")
    elif not mg_hit and bm_hit:
        reason.append("biomart_hit_but_no_mouse_gene_resolved")
    unmapped_rows.append({
        "human_gene": g,
        "in_ML_consensus": g in ml_set,
        "in_DL_top100": g in dl_set,
        "mygene_found": mg_hit,
        "biomart_found": bm_hit,
        "status": "NOT_MAPPED",
        "reason": "; ".join(reason) if reason else "unknown",
    })
unmapped_df = pd.DataFrame(unmapped_rows)
if not unmapped_df.empty:
    unmapped_df.to_csv(OUT / "18a_unmapped_human_genes.csv", index=False)
    logger.info("Unmapped genes: %d (written to 18a_unmapped_human_genes.csv)", len(unmapped_df))
else:
    logger.info("All standard genes mapped successfully.")

logger.info("=== Step 18a COMPLETE ===")

"""Figure 4: external human fetal thymus developmental-stage localization.

This is independent developmental-state contextualization (GSE195812), NOT an
aging replication. Within-object gene-wise z-scores are shown separately for
Sort1 (DN1-DN3) and Sort2 (ISP/DP); raw magnitudes are never compared across
the two objects. Point size = percent expressing (0-100%).
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.cm import ScalarMappable

from figure_common import *

STAGES = {"Sort1": ("DN1", "DN2", "DN3"),
          "Sort2": ("ISP", "DP_CD3min", "DP_CD3plus")}


def _preselect_genes(loc, max_main=20):
    """Freeze an auditable exploratory candidate list, not a significance set.

    The original implementation constructed a union and then returned only
    ML-ranked genes, silently dropping IG/null-only candidates. Here the
    inclusion rule is explicit and every displayed gene has its route logged.
    """
    available = set(loc.gene.astype(str))
    cons = pd.read_csv(ML / "ML_consensus_gene_stability.csv")
    ig = pd.read_csv(DL / "strict_oof" / "oof_IG_gene_consistency.csv")
    null = read_gene_recurrence_null()
    for name, table in (("ML", cons), ("IG", ig), ("null", null)):
        if table.gene.duplicated().any():
            raise ValueError(f"Figure 4: duplicate gene in {name} source")
    rank = cons.set_index("gene")["best_rank_pct"]
    folds = ig.set_index("gene")["n_folds_top50"]
    q = null.set_index("gene")["q_recurrence_BH"]
    passed = null.set_index("gene")[NULL_COL_CORRECTED].fillna(False).astype(bool)
    rows = []
    for gene in sorted(available):
        ml = bool(gene in rank.index and rank[gene] <= 16)
        ig_route = bool(gene in folds.index and folds[gene] >= IG_STABLE_FOLDS)
        null_route = bool(gene in passed.index and passed[gene])
        if not (ml or ig_route or null_route):
            continue
        reasons = [label for flag, label in ((ml, "ML top-16%"),
                   (ig_route, f"OOF-IG >= {IG_STABLE_FOLDS}/18"),
                   (null_route, "permutation BH q<0.05")) if flag]
        rows.append({"gene": gene, "ML_consensus": ml,
                     "ML_best_rank_pct": rank.get(gene, np.nan),
                     "selection_recurrence_q": q.get(gene, np.nan),
                     "passes_permutation_BH": null_route,
                     "OOF_IG_fold_count": folds.get(gene, np.nan),
                     "detected_Sort1": bool(((loc.gene == gene) &
                         (loc.source_object == "Sort1") & (loc.pct_expr > 0)).any()),
                     "detected_Sort2": bool(((loc.gene == gene) &
                         (loc.source_object == "Sort2") & (loc.pct_expr > 0)).any()),
                     "included_reason": "; ".join(reasons)})
    audit = pd.DataFrame(rows)
    if audit.empty:
        raise ValueError("No Figure 4 candidates meet the declared selection routes")
    audit = audit.sort_values(["passes_permutation_BH", "ML_best_rank_pct",
                               "OOF_IG_fold_count", "gene"],
                              ascending=[False, True, False, True],
                              na_position="last").reset_index(drop=True)
    audit["displayed_main"] = np.arange(len(audit)) < max_main
    FIGF.mkdir(parents=True, exist_ok=True)
    audit.to_csv(FIGF / "fig4_candidate_selection_audit.tsv", sep="\t", index=False)
    return audit.loc[audit.displayed_main, "gene"].tolist(), audit


def _facet(ax, loc, genes, src, show_y, norm):
    sub = loc[(loc.source_object == src) & (loc.gene.isin(genes))]
    mat_z = sub.pivot_table(index="gene", columns="developmental_stage",
                            values="mean_zscore").reindex(
                                genes, columns=STAGES[src])
    mat_p = sub.pivot_table(index="gene", columns="developmental_stage",
                            values="pct_expr").reindex(
                                genes, columns=STAGES[src])
    for i, g in enumerate(genes):
        for j, st in enumerate(STAGES[src]):
            z = mat_z.loc[g, st]
            p = mat_p.loc[g, st]
            if pd.isna(z):
                ax.scatter(j, i, marker="x", s=10, color=GREY)
                continue
            # Fixed marker area stays below the 20-row pitch at print size.
            ax.scatter(j, i, s=10 + 95 * float(p) / 100.0,
                       c=[z], cmap="RdBu_r", norm=norm,
                       edgecolor="k", linewidth=0.3)
    ax.set_xticks(range(len(STAGES[src])))
    ax.set_xticklabels([s.replace("_", "\n") for s in STAGES[src]],
                       fontsize=6)
    ax.set_yticks(range(len(genes)))
    if show_y:
        ax.set_yticklabels(genes, fontsize=6.4)
    else:
        ax.set_yticklabels([])
    ax.set_xlim(-0.6, len(STAGES[src]) - 0.4)
    ax.set_ylim(len(genes) - 0.5, -0.5)
    ax.set_title(f"{src}: within-object z-score", fontsize=7, pad=6)
    ax.grid(axis="x", which="minor", color="#EEEEEE", lw=0.5)
    ax.tick_params(axis="x", which="minor", length=0)
    ax.tick_params(labelsize=6.0)
    return ax


def _coverage(ax, dc):
    """Reference-state sample coverage, never an aging sample-size claim."""
    order = [(src, stage) for src, stages in STAGES.items() for stage in stages]
    rows = (dc.set_index(["source_object", "developmental_stage"])
            .reindex(pd.MultiIndex.from_tuples(
                order, names=["source_object", "developmental_stage"])))
    if rows[["n_cells_state", "n_donors"]].isna().any().any():
        raise ValueError("Figure 4A: missing donor/cell coverage for a plotted state")
    rows = rows.reset_index()
    rows.to_csv(FIGF / "fig4_state_coverage_audit.tsv", sep="\t", index=False)
    y = np.arange(len(rows))[::-1]
    colors = ["#5B8DB8" if src == "Sort1" else "#9B77AD"
              for src in rows.source_object]
    vals = rows.n_cells_state.to_numpy(dtype=float)
    ax.barh(y, vals, color=colors, height=0.68)
    for yi, r in zip(y, rows.itertuples()):
        ax.text(r.n_cells_state + max(vals) * 0.025, yi,
                f"{int(r.n_cells_state):,} cells; {int(r.n_donors)} donors",
                va="center", fontsize=5.7)
    labels = [f"{r.source_object} · {r.developmental_stage.replace('_', ' ')}"
              for r in rows.itertuples()]
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=5.8)
    ax.set_xlim(0, max(vals) * 1.65)
    ax.set_xlabel("Cells in external developmental state (descriptive)", fontsize=6)
    ax.set_title("External reference coverage", fontsize=7, pad=6)
    ax.tick_params(labelsize=5.5)


def _stage_delta_table(loc, genes):
    """Within-object late-minus-early stage means; no cross-object contrast."""
    rows = []
    for src, (early, _, late) in STAGES.items():
        sub = loc[(loc.source_object == src) & loc.gene.isin(genes)]
        mat = sub.pivot_table(index="gene", columns="developmental_stage",
                              values="mean_zscore")
        for gene in genes:
            first = mat.loc[gene, early] if gene in mat.index and early in mat else np.nan
            last = mat.loc[gene, late] if gene in mat.index and late in mat else np.nan
            rows.append({"gene": gene, "source_object": src,
                         "early_state": early, "late_state": late,
                         "early_mean_z": first, "late_mean_z": last,
                         "delta_z_late_minus_early": last - first,
                         "comparison_type": "descriptive within-object"})
    audit = pd.DataFrame(rows)
    audit.to_csv(FIGF / "fig4_stage_delta_audit.tsv", sep="\t", index=False)
    return audit


def _stage_delta(ax, audit, src):
    sub = (audit[audit.source_object.eq(src)]
           .dropna(subset=["delta_z_late_minus_early"])
           .assign(abs_delta=lambda d: d.delta_z_late_minus_early.abs())
           .sort_values(["abs_delta", "gene"], ascending=[False, True])
           .head(8).sort_values("delta_z_late_minus_early"))
    if sub.empty:
        raise ValueError(f"Figure 4: no observed stage contrast for {src}")
    vals = sub.delta_z_late_minus_early.to_numpy(dtype=float)
    y = np.arange(len(sub))
    ax.barh(y, vals, color=["#8E6C9E" if v > 0 else "#5B8DB8"
                             for v in vals], height=0.67)
    ax.axvline(0, color="#444444", lw=0.65)
    ax.set_yticks(y); ax.set_yticklabels(sub.gene, fontsize=5.7)
    limit = max(0.2, float(np.max(np.abs(vals))) * 1.18)
    ax.set_xlim(-limit, limit)
    early, _, late = STAGES[src]
    ax.set_xlabel(f"Δ mean z: {late.replace('_', ' ')} − {early}", fontsize=6)
    ax.set_title(f"{src} stage contrast (top 8 |Δ|; descriptive)",
                 fontsize=6.7, pad=5)
    ax.tick_params(labelsize=5.5)


def build():
    loc = pd.read_csv(EXT / "17_candidate_gene_localization.csv")

    if loc.duplicated(["gene", "source_object", "developmental_stage"]).any():
        raise ValueError("Figure 4: duplicate gene/object/state localization rows")

    # Hard range assert (review P0): pct_expr is a percentage in [0,100].
    assert loc["pct_expr"].min() >= 0, "pct_expr < 0 detected"
    assert loc["pct_expr"].max() <= 100, "pct_expr > 100 detected"

    # Merge per-state donor counts (review 3.3: source table needs n_donors)
    dc = pd.read_csv(EXT / "17_state_donor_counts.csv")
    dc = dc.rename(columns={"object": "source_object",
                            "state": "developmental_stage"})
    if dc.duplicated(["source_object", "developmental_stage"]).any():
        raise ValueError("Figure 4: duplicate object/state donor-count rows")
    loc = loc.merge(dc[["source_object", "developmental_stage", "n_cells_state",
                        "n_donors"]],
                    on=["source_object", "developmental_stage"], how="left",
                    validate="many_to_one")
    if loc.n_donors.isna().any():
        raise ValueError("Figure 4: donor counts missing for some displayed states")

    genes, audit = _preselect_genes(loc)
    print(f"[fig4] showing {len(genes)} exploratory genes; "
          f"BH-corrected candidates: {int(audit.passes_permutation_BH.sum())}")

    # Export source table (review 3.3): gene × state with expression metrics
    export_cols = ["gene", "developmental_stage", "source_object",
                   "n_cells", "n_donors", "mean_expr", "mean_zscore", "pct_expr"]
    loc_export = (loc[export_cols].drop_duplicates()
                  .rename(columns={"developmental_stage": "state",
                                   "source_object": "object",
                                   "mean_expr": "avg_expr"}))
    loc_export.to_csv(FIGF / "fig4_dotplot_source.tsv", sep="\t", index=False)
    print(f"[fig4] exported fig4_dotplot_source.tsv ({len(loc_export)} rows)")

    displayed = loc[loc.gene.isin(genes)].mean_zscore.dropna().abs()
    if displayed.empty:
        raise ValueError("Figure 4: no finite mean_zscore in displayed genes")
    vmax = max(0.2, np.ceil(float(displayed.max()) * 5) / 5)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    fig = plt.figure(figsize=(W_DOUBLE, 285 / 25.4))
    gs = gridspec.GridSpec(4, 2, figure=fig, left=0.20, right=0.87,
                           bottom=0.055, top=0.935, hspace=0.52, wspace=0.38,
                           height_ratios=[1.0, 3.5, 1.3, 1.3])
    axA = fig.add_subplot(gs[0, :]); letter(axA, "A", dx=-0.20)
    _coverage(axA, dc)
    ax1 = fig.add_subplot(gs[1, 0]); letter(ax1, "B", dx=-0.20)
    _facet(ax1, loc, genes, "Sort1", True, norm)
    ax2 = fig.add_subplot(gs[1, 1]); letter(ax2, "C", dx=-0.12)
    _facet(ax2, loc, genes, "Sort2", False, norm)
    # Never derive a colour bar from the last scatter collection: that last
    # collection may be an empty legend handle with a default 0-1 colormap.
    sm = ScalarMappable(norm=norm, cmap="RdBu_r")
    sm.set_array([])
    cax = fig.add_axes([0.90, 0.44, 0.018, 0.27])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("within-object gene z-score", fontsize=6)
    cb.ax.tick_params(labelsize=5.5)

    # Shared size legend lives in the reserved right margin.
    for s in [10, 50, 90]:
        ax2.scatter([], [], s=10 + 95 * s / 100.0, color="#BBBBBB",
                    edgecolor="k", linewidth=0.3, label=f"{s}%")
    lg = fig.legend(*ax2.get_legend_handles_labels(), title="% expressing",
                    loc="upper left", bbox_to_anchor=(0.88, 0.79),
                    fontsize=5.4, title_fontsize=5.5, frameon=False,
                    labelspacing=0.9, borderaxespad=0)
    for h in lg.legend_handles:
        h.set_alpha(1.0)

    delta = _stage_delta_table(loc, genes)
    axD = fig.add_subplot(gs[2, :]); letter(axD, "D", dx=-0.20)
    _stage_delta(axD, delta, "Sort1")
    axE = fig.add_subplot(gs[3, :]); letter(axE, "E", dx=-0.20)
    _stage_delta(axE, delta, "Sort2")

    fig.suptitle("External fetal thymus context (exploratory; not aging replication)",
                 fontsize=8, y=0.985)
    pd.DataFrame({"color_limit_abs_z": [vmax], "colormap": ["RdBu_r"],
                  "displayed_genes": [len(genes)]}).to_csv(
        FIGF / "fig4_color_scale_audit.tsv", sep="\t", index=False)
    save_composite(fig, "Figure4")

"""Shared style / data / stats helpers for publication figure rebuild v2
(figure audit 2026-09-16).

Conventions:
  - composite Figures are built at final print width: 183 mm (double column)
    or 89 mm (single column); minimum font 7 pt; Times New Roman.
  - Okabe-Ito colour semantics: age-up vermillion #D55E00, age-down blue
    #0072B2; concordant #009E73, discordant #D55E00, not-assessed light grey.
  - one point = one donor / animal unless stated; genes/peaks are never
    treated as independent biological replicates.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Patch
from scipy import stats

PROJ = Path(os.environ.get("PROJ", "/data/zxy/projects/human_thymus_age_ML_DL"))
META = PROJ / "01_raw_processing" / "metadata"
PSEUDO = PROJ / "02_pseudobulk"
FEAT = PROJ / "03_feature_selection"
ML = PROJ / "04_machine_learning"
DL = PROJ / "05_deep_learning"
INTERP = PROJ / "06_interpretation"
EXT = PROJ / "07_external_validation"
MOUSE = PROJ / "08_mouse_validation"
RES = PROJ / "10_results"
FILT = PROJ / "01_raw_processing" / "filtered"
# Keep the frozen results read-only.  FIG_OUTPUT can point at a separate
# revision directory; the original figures_final is never overwritten by the
# revised runner unless the author explicitly chooses that location.
FIGF = Path(os.environ.get("FIG_OUTPUT", str(PROJ / "figures_final_revised")))
PANELDIR = FIGF / "panels"
SUPPDIR = FIGF / "Supplementary"

MM = 1 / 25.4
W_DOUBLE = 183 * MM   # 7.205 in
W_SINGLE = 89 * MM

# ----------------------------------------------------------------- pre-declared
# Stability thresholds (frozen 2026-09-17; used identically in Fig4 / Fig5F /
# S08 / legends so no panel can re-define "stable" on its own):
IG_TOP_K = 50            # OOF-IG top-k window
IG_STABLE_FOLDS = 5      # folds (of 18) required before a gene is called recurrent
NULL_COL_CORRECTED = "above_null_BH_q05"   # per-gene BH-corrected permutation q
NULL_COL_FWER = "above_null_FWER_maxstat"  # family-wise max-statistic flag

# ----------------------------------------------------------------- style
# Journal typography is deliberately fail-closed.  Matplotlib otherwise falls
# back silently to DejaVu when Times New Roman is unavailable, producing a file
# that appears to have run successfully but does not meet the requested font
# specification.  PAPER_FONT_PATH may point to an authorised local font file
# or a directory containing the regular/bold/italic font files on a server
# where the family is not installed system-wide.
PAPER_FONT = os.environ.get("PAPER_FONT", "Times New Roman")
PAPER_FONT_PATH = os.environ.get("PAPER_FONT_PATH")
if PAPER_FONT_PATH:
    _font_file = Path(PAPER_FONT_PATH).expanduser().resolve()
    if _font_file.is_dir():
        _font_files = sorted(
            p for pattern in ("*.ttf", "*.ttc", "*.otf")
            for p in _font_file.glob(pattern)
        )
    elif _font_file.is_file():
        _font_files = [_font_file]
    else:
        raise FileNotFoundError(f"PAPER_FONT_PATH does not exist: {_font_file}")
    if not _font_files:
        raise FileNotFoundError(f"No TTF/TTC/OTF fonts found in: {_font_file}")
    for _font_path in _font_files:
        font_manager.fontManager.addfont(str(_font_path))
try:
    PAPER_FONT_FILE = font_manager.findfont(
        FontProperties(family=[PAPER_FONT]), fallback_to_default=False)
except ValueError as exc:
    raise RuntimeError(
        "Times New Roman is required for the paper figures but is not visible "
        "to Matplotlib. Install/register the font, or set PAPER_FONT_PATH to "
        "an authorised Times New Roman TTF/TTC file before running the figure "
        "builder."
    ) from exc

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": [PAPER_FONT],
    "mathtext.fontset": "custom",
    "mathtext.rm": PAPER_FONT,
    "mathtext.it": f"{PAPER_FONT}:italic",
    "mathtext.bf": f"{PAPER_FONT}:bold",
    "mathtext.fallback": "stix",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 7.5,
    "axes.titlesize": 8,
    "axes.labelsize": 7.5,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 6.5,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "lines.linewidth": 0.9,
    "axes.grid": False,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# Okabe-Ito semantics
UP = "#D55E00"      # age-up / warm
DOWN = "#0072B2"    # age-down / cool
CONC = "#009E73"    # concordant / support
DISC = "#D55E00"    # discordant
GREY = "#999999"
LGREY = "#DDDDDD"
NA_GREY = "#E8E8E8"
NEUTRAL = "#FFFFFF"
FEMALE = "#CC79A7"
MALE = "#0072B2"
AGEGROUP = {"Young": "#0072B2", "Mid": "#BBBBBB", "Old": "#D55E00"}

# Fixed cell-type palette (T-lineage blue family; stromal green/brown;
# unresolved always grey). Used identically in UMAP/composition/fraction.
CELLTYPE_COLORS = {
    "DN": "#0B3D91", "DP": "#2E6FB7", "SP_CD4": "#5B9BD5",
    "SP_CD8": "#9DC3E6", "Treg": "#7030A0",
    "GammaDelta_T": "#882255", "NKT_like": "#B07AA1",
    "NK": "#56B4E9", "B": "#E69F00", "Plasma": "#F0E442",
    "DC": "#009E73", "Monocyte": "#D55E00", "Macrophage": "#A0522D",
    "Myeloid": "#E5AE7D",
    "TEC": "#117733", "Endothelial": "#88CCEE", "Fibroblast": "#6F5840",
    "Unknown": "#D9D9D9", "SP_unresolved": "#EAEAEA",
    "Innate_T_unresolved": "#EAEAEA",
}
CT_ORDER = ["DN", "DP", "SP_CD4", "SP_CD8", "Treg", "GammaDelta_T",
            "NKT_like", "NK", "B", "Plasma", "DC", "Monocyte", "Macrophage",
            "Myeloid", "TEC", "Endothelial", "Fibroblast",
            "SP_unresolved", "Unknown"]


def ct_palette(types):
    """Return colour list for an iterable of cell types (stable fallback)."""
    import colorsys
    out = []
    used = set(CELLTYPE_COLORS.values())
    fallback = [c for c in []]
    for t in types:
        if t in CELLTYPE_COLORS:
            out.append(CELLTYPE_COLORS[t])
        else:
            out.append("#9C9C9C")
    return out


# Canonical markers per major cell type (gene symbols; filtered to present).
MARKERS = {
    "DN": ["DNTT", "RAG1", "IL7R", "CD7"],
    "DP": ["CD1A", "CD4", "CD8A", "RAG1"],
    "SP_CD4": ["CD4", "CCR7", "IL7R"],
    "SP_CD8": ["CD8A", "CD8B", "CCR7"],
    "Treg": ["FOXP3", "IL2RA", "CTLA4"],
    "GammaDelta_T": ["TRDC", "TRGV9", "NKG7"],
    "NKT_like": ["NKG7", "KLRD1", "ZBTB16"],
    "NK": ["NKG7", "KLRD1", "GNLY", "NCAM1"],
    "B": ["MS4A1", "CD79A", "CD19"],
    "Plasma": ["MZB1", "JCHAIN", "SDC1"],
    "DC": ["FCER1A", "CLEC10A", "HLA-DRA"],
    "Monocyte": ["LYZ", "S100A8", "CD14"],
    "Macrophage": ["CD68", "AIF1", "C1QA", "LYZ"],
    "TEC": ["EPCAM", "KRT8", "KRT5", "FOXN1"],
    "Endothelial": ["PECAM1", "VWF"],
    "Fibroblast": ["COL1A1", "COL3A1", "DCN", "PDGFRA"],
}

# ----------------------------------------------------------------- helpers
def letter(ax, txt, dx=-0.14, dy=1.06):
    ax.text(dx, dy, txt, transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="bottom", ha="left")


def save_fig(fig, group, name):
    d = PANELDIR / group
    d.mkdir(parents=True, exist_ok=True)
    fig.savefig(d / f"{name}.pdf")
    fig.savefig(d / f"{name}.png", dpi=300)
    plt.close(fig)


def save_supp(fig, name):
    SUPPDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(SUPPDIR / f"{name}.pdf")
    fig.savefig(SUPPDIR / f"{name}.png", dpi=300)
    plt.close(fig)


def save_composite(fig, name):
    FIGF.mkdir(parents=True, exist_ok=True)
    # Keep the declared physical canvas.  The global tight-bbox style can
    # silently change panel sizes when an annotation or legend protrudes.
    fig.savefig(FIGF / f"{name}.pdf", bbox_inches=fig.bbox_inches)
    fig.savefig(FIGF / f"{name}.png", dpi=300, bbox_inches=fig.bbox_inches)
    plt.close(fig)


def load_freeze():
    f = pd.read_csv(META / "04G_final_cohort_freeze.csv")
    f = f.sort_values("age_years").reset_index(drop=True)
    f["age_group"] = pd.cut(
        f.age_years, [-1, 17.999, 39.999, 100],
        labels=["Young", "Mid", "Old"])
    return f


def rho_ci(x, y, n_boot=2000, seed=371):
    """Donor-level bootstrap 95% CI for Spearman rho."""
    rng = np.random.default_rng(seed)
    x, y = np.asarray(x), np.asarray(y)
    n = len(x)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        if len(np.unique(x[i])) < 2 or len(np.unique(y[i])) < 2:
            continue
        vals.append(stats.spearmanr(x[i], y[i]).statistic)
    r = stats.spearmanr(x, y).statistic
    return r, np.quantile(vals, [.025, .975])


def bh(p):
    return stats.false_discovery_control(np.asarray(p, float), method="bh")


def short(donor):
    return str(donor).replace("donor", "d")


def read_gene_recurrence_null():
    """Read the permutation-null table without upgrading legacy flags.

    A legacy ``above_null`` value is not a BH- or FWER-corrected result and
    must not be silently promoted to either statistical claim.
    """
    f = ML / "permutation_null" / "gene_recurrence_null.csv"
    g = pd.read_csv(f)
    required = {"gene", "obs_n_folds", "null_q95_folds", "p_recurrence", "q_recurrence_BH",
                NULL_COL_CORRECTED, NULL_COL_FWER}
    missing = required - set(g.columns)
    if missing:
        raise ValueError(f"{f} lacks corrected null fields: {sorted(missing)}. "
                         "Regenerate with 23_permutation_null_stability.py; "
                         "legacy above_null cannot stand in for BH/FWER.")
    if g.gene.duplicated().any():
        raise ValueError(f"Duplicate genes in {f}")
    return g

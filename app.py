from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "release" / "data"
MANIFEST = ROOT / "release" / "manifest"
ASSETS = ROOT / "assets"
st.set_page_config(page_title="Human Thymic Aging Atlas", page_icon="🧬", layout="wide")
st.markdown(
    """
    <style>
    :root {--ink:#102a2d;--muted:#5d7072;--teal:#0b756d;--paper:#fbfdfc;}
    .stApp {background:radial-gradient(circle at 8% 0%,rgba(93,189,169,.16),transparent 28rem),radial-gradient(circle at 98% 8%,rgba(88,129,180,.12),transparent 32rem),var(--paper);}
    .block-container {max-width:1280px;padding-top:1.4rem;padding-bottom:4rem;}
    h1,h2,h3,h4 {color:var(--ink);letter-spacing:-.02em;}
    .hero-kicker {display:inline-flex;padding:.38rem .72rem;border-radius:999px;color:#08665f;background:#dff3ec;border:1px solid #b9ded2;font-size:.78rem;font-weight:700;letter-spacing:.08em;text-transform:uppercase;}
    .hero-title {font-size:clamp(2.55rem,5vw,4.8rem);line-height:.98;margin:1rem 0 .9rem;font-weight:760;}
    .hero-copy {font-size:1.12rem;line-height:1.7;color:var(--muted);max-width:46rem;}
    .hero-panel {padding:2.4rem 2.5rem;border-radius:28px;background:linear-gradient(135deg,rgba(255,255,255,.94),rgba(238,248,245,.9));border:1px solid rgba(21,96,89,.14);box-shadow:0 24px 65px rgba(35,80,76,.10);}
    .pill-row {display:flex;flex-wrap:wrap;gap:.55rem;margin-top:1.35rem;}.pill {padding:.42rem .72rem;border-radius:999px;background:#fff;border:1px solid #d7e5e1;color:#38585a;font-size:.82rem;}
    .section-label {color:#0b756d;text-transform:uppercase;letter-spacing:.12em;font-size:.76rem;font-weight:800;margin-bottom:.25rem;}
    .atlas-card {height:100%;padding:1.15rem 1.2rem;border-radius:18px;background:rgba(255,255,255,.84);border:1px solid #dfe9e6;box-shadow:0 10px 28px rgba(35,73,70,.055);}
    .atlas-card h4 {margin:.15rem 0 .4rem;font-size:1.05rem;}.atlas-card p {margin:0;color:#667779;line-height:1.55;font-size:.9rem;}
    .scope-note {padding:1rem 1.15rem;border-left:4px solid #0b756d;border-radius:10px;background:#edf7f4;color:#365759;line-height:1.55;}
    div[data-testid="stMetric"] {background:rgba(255,255,255,.86);border:1px solid #dde8e5;padding:.9rem 1rem;border-radius:16px;box-shadow:0 8px 22px rgba(34,70,67,.05);}
    div[data-testid="stDataFrame"] {border:1px solid #e0e8e6;border-radius:14px;overflow:hidden;}
    .stTabs [data-baseweb="tab-list"] {gap:.35rem;background:#edf4f2;padding:.38rem;border-radius:14px;}
    .stTabs [data-baseweb="tab"] {height:2.75rem;border-radius:10px;padding:0 1rem;color:#486163;}
    .stTabs [aria-selected="true"] {background:white;color:#075f59;box-shadow:0 4px 14px rgba(30,80,74,.08);}
    .image-credit {font-size:.74rem;color:#78888a;line-height:1.45;}.footer {margin-top:2rem;padding-top:1.2rem;border-top:1px solid #dce7e4;color:#718183;font-size:.82rem;}
    </style>
    """,
    unsafe_allow_html=True,
)
@st.cache_data(show_spinner=False)
def load_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA / name)
def fmt(value: object, digits: int = 4) -> str:
    if pd.isna(value):
        return "NA"
    if isinstance(value, (float, np.floating)):
        return f"{value:.{digits}g}"
    return str(value)
def scope_note() -> None:
    st.markdown(
        '<div class="scope-note"><strong>Interpretation boundary.</strong> Release v1 summarizes donor-resolved analyses of GSE231906 (18 donors). Captured-library fractions follow nominal 6:4 CD45-positive/CD45-negative recombination and are not native-tissue composition. Donor deviation is descriptive and is not a validated clinical biological-age score.</div>',
        unsafe_allow_html=True,
    )
gene_summary = load_csv("gene_summary.csv.gz")
effects = load_csv("gene_context_effects.csv.gz")
gene_pathways = load_csv("gene_pathway_links.csv.gz")
developmental = load_csv("developmental_context.csv.gz")
cell_summary = load_csv("cell_type_summary.csv")
cell_genes = load_csv("cell_type_genes.csv.gz")
cell_pathways = load_csv("cell_type_pathway_overlap.csv.gz")
donors = load_csv("donor_summary.csv")
donor_composition = load_csv("donor_composition.csv")
cohorts = load_csv("cohort_registry.csv")
hero_left, hero_right = st.columns([1.35, .65], gap="large")
with hero_left:
    st.markdown('<div class="hero-panel"><span class="hero-kicker">Open results resource · Release v1</span><div class="hero-title">Human Thymic<br>Aging Atlas</div><div class="hero-copy">Explore continuous-age signals across genes, cell contexts and donors. The atlas connects sex-adjusted pseudobulk effects with machine-learning selection, Transformer attribution, developmental context and cross-dataset directional support.</div><div class="pill-row"><span class="pill">18 independent donors</span><span class="pill">17,181 genes</span><span class="pill">11 cell contexts</span><span class="pill">downloadable source tables</span></div></div>', unsafe_allow_html=True)
with hero_right:
    image = ASSETS / "healthy_human_t_cell.jpg"
    if image.exists():
        st.image(str(image), width="stretch")
        st.markdown('<div class="image-credit">Healthy human T lymphocyte, scanning electron micrograph. NIAID/NIH, public domain. Decorative context; not study data.</div>', unsafe_allow_html=True)
st.write("")
scope_note()
st.write("")
overview_tab, gene_tab, cell_tab, donor_tab, download_tab = st.tabs(["Overview", "Gene explorer", "Cell-type explorer", "Donor explorer", "Downloads & provenance"])
with overview_tab:
    st.markdown('<div class="section-label">Resource architecture</div>', unsafe_allow_html=True)
    st.subheader("From organ context to donor-resolved evidence")
    left, right = st.columns([.9, 1.1], gap="large")
    with left:
        lobule = ASSETS / "thymus_lobule_nih_bioart.svg"
        if lobule.exists():
            st.image(str(lobule), width="stretch")
            st.markdown('<div class="image-credit">Thymus lobule © Human Reference Atlas / NIAID NIH BioArt, CC BY 4.0. Decorative anatomical context; not study data.</div>', unsafe_allow_html=True)
    with right:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="atlas-card"><h4>Gene evidence</h4><p>Age effects by context, q values, pathways, ML selection, Transformer attribution and mouse directional support.</p></div>', unsafe_allow_html=True)
            st.write("")
            st.markdown('<div class="atlas-card"><h4>Donor evidence</h4><p>Age, sex, QC, captured-library composition, out-of-fold prediction and age-adjusted deviation.</p></div>', unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="atlas-card"><h4>Cell-type evidence</h4><p>Composition associations, donor coverage, age-up and age-down genes, pathways and top predictors.</p></div>', unsafe_allow_html=True)
            st.write("")
            st.markdown('<div class="atlas-card"><h4>Reproducibility</h4><p>CSV, H5AD, source workbooks, code, schemas, provenance manifests and SHA-256 checksums.</p></div>', unsafe_allow_html=True)
    st.write("")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Genes indexed", f"{gene_summary['gene'].nunique():,}")
    m2.metric("Gene × context rows", f"{len(effects):,}")
    m3.metric("Cell contexts", f"{cell_summary['cell_type'].nunique():,}")
    m4.metric("Primary donors", f"{donors['donor_id'].nunique():,}")
    counts = pd.DataFrame({"Layer":["Human age association","ML consensus","Transformer top-100","Mouse RNA concordance","Mouse ATAC concordance"],"Genes":[int(gene_summary["whole_thymus_q_value"].lt(.05).sum()),int(gene_summary["in_ML_consensus"].fillna(False).astype(bool).sum()),int(gene_summary["in_DL_top100"].fillna(False).astype(bool).sum()),int(gene_summary["mouse_RNA_concordant"].fillna(False).astype(bool).sum()),int(gene_summary["mouse_ATAC_concordant"].fillna(False).astype(bool).sum())]})
    fig = px.bar(counts, x="Genes", y="Layer", orientation="h", color="Layer", color_discrete_sequence=["#0b756d","#3c8b81","#5e9ca0","#315f91","#c95858"])
    fig.update_layout(showlegend=False, height=360, margin=dict(l=10,r=15,t=10,b=10))
    st.plotly_chart(fig, use_container_width=True)
with gene_tab:
    st.markdown('<div class="section-label">Gene-level lookup</div>', unsafe_allow_html=True)
    st.subheader("Follow one gene across evidence layers")
    options = gene_summary["gene"].dropna().astype(str).sort_values().tolist()
    selected_gene = st.selectbox("Gene symbol", options, index=options.index("FOXN1") if "FOXN1" in options else 0)
    record = gene_summary.loc[gene_summary["gene"].eq(selected_gene)].iloc[0]
    rows = effects.loc[effects["gene"].eq(selected_gene)].copy()
    rows["direction"] = np.where(rows["logFC_age_perSD"].ge(0), "Age-up", "Age-down")
    rows["significant_q05"] = rows["adj.P.Val"].lt(.05)
    contexts = ["whole_thymus","TEC","Fibroblast","DP","SP_CD4","SP_CD8"]
    for col, context in zip(st.columns(6), contexts):
        row = rows.loc[rows["cell_context"].eq(context)]
        col.metric(context, "NA" if row.empty else fmt(row.iloc[0]["logFC_age_perSD"]), None if row.empty else f"q={fmt(row.iloc[0]['adj.P.Val'])}; n={int(row.iloc[0]['n_donors'])}")
    left, right = st.columns([1.35,1])
    with left:
        st.markdown("#### Age effects by cell context")
        if not rows.empty:
            fig = px.scatter(rows.sort_values("logFC_age_perSD"), x="logFC_age_perSD", y="cell_context", color="direction", symbol="significant_q05", hover_data={"adj.P.Val":":.3g","P.Value":":.3g","n_donors":True}, color_discrete_map={"Age-up":"#c95858","Age-down":"#315f91"}, labels={"logFC_age_perSD":"Sex-adjusted beta_age per 1 SD","cell_context":""})
            fig.add_vline(x=0,line_color="#879395",line_width=1); fig.update_layout(height=430,legend_title_text="")
            st.plotly_chart(fig,use_container_width=True)
        st.dataframe(rows[["cell_context","logFC_age_perSD","P.Value","adj.P.Val","n_donors","direction"]].sort_values("adj.P.Val"),use_container_width=True,hide_index=True)
    with right:
        st.markdown("#### Integrated evidence")
        labels = ["ML consensus selection","ML feature count","Transformer top-100","Best IG cell type","Best IG rank","Mouse RNA direction","Mouse RNA concordant","Mouse ATAC concordant","Developmental top state"]
        keys = ["in_ML_consensus","ml_n_features","in_DL_top100","transformer_best_ig_celltype","transformer_best_ig_rank","mouse_RNA_direction","mouse_RNA_concordant","mouse_ATAC_concordant","developmental_top_stage"]
        st.dataframe(pd.DataFrame({"Evidence":labels,"Value":[fmt(record.get(k)) for k in keys]}),use_container_width=True,hide_index=True)
    p1,p2=st.columns(2)
    with p1:
        st.markdown("#### Pathway evidence"); path_rows=gene_pathways.loc[gene_pathways["gene"].eq(selected_gene)]
        if path_rows.empty:
            st.caption("No pathway membership in current outputs.")
        else:
            st.dataframe(path_rows[["evidence_layer","gene_set_library","pathway","pathway_q_value","odds_ratio"]].head(100),use_container_width=True,hide_index=True)
    with p2:
        st.markdown("#### Developmental localization"); dev=developmental.loc[developmental["gene"].astype(str).str.upper().eq(selected_gene)]
        if dev.empty: st.caption("Not available in the current GSE195812 localization table.")
        else: st.plotly_chart(px.bar(dev.sort_values("developmental_stage"),x="developmental_stage",y="mean_zscore",color="source_object",hover_data={"pct_expr":":.2f","n_cells":True},labels={"mean_zscore":"Mean within-object z-score","developmental_stage":"Stage"}),use_container_width=True)
with cell_tab:
    st.markdown('<div class="section-label">Cell-context view</div>',unsafe_allow_html=True); st.subheader("Composition and transcriptional evidence")
    selected_cell=st.selectbox("Cell type",cell_summary["cell_type"].sort_values().tolist()); rec=cell_summary.loc[cell_summary["cell_type"].eq(selected_cell)].iloc[0]
    metrics=st.columns(5); metrics[0].metric("Composition beta_age",fmt(rec.get("composition_age_effect"))); metrics[1].metric("Composition q",fmt(rec.get("composition_q_value_BH"))); metrics[2].metric("Age-up genes",int(rec.get("n_age_up_q05",0))); metrics[3].metric("Age-down genes",int(rec.get("n_age_down_q05",0))); metrics[4].metric("Donor coverage",f"{int(rec.get('min_donor_coverage',0))}-{int(rec.get('max_donor_coverage',0))}")
    st.caption("Composition effect is a sex-adjusted CLR association within captured libraries; it does not estimate intact-thymus abundance.")
    sig=cell_genes.loc[(cell_genes["cell_context"].eq(selected_cell))&(cell_genes["direction"].ne("not_significant"))]
    left,right=st.columns([1.35,1])
    with left: st.markdown("#### Significant age-associated genes"); st.dataframe(sig[["gene","direction","logFC_age_perSD","adj.P.Val","P.Value","n_donors"]].sort_values(["direction","adj.P.Val"]),use_container_width=True,hide_index=True,height=470)
    with right:
        st.markdown("#### Top Gene × CellType predictors"); predictors=str(rec.get("top_gene_celltype_predictors","")).split("; "); pred=gene_summary.loc[gene_summary["gene"].isin(predictors)]
        if pred.empty:
            st.caption("No Transformer token was available for this cell type.")
        else:
            st.dataframe(pred[["gene","transformer_best_ig_rank","transformer_best_ig_importance","in_ML_consensus"]].sort_values("transformer_best_ig_rank"),use_container_width=True,hide_index=True)
        st.markdown("#### Pathway evidence overlap"); cp=cell_pathways.loc[cell_pathways["cell_type"].eq(selected_cell)]
        if cp.empty:
            st.caption("No overlap with the current pathway evidence table.")
        else:
            st.dataframe(cp[["direction","pathway","n_overlap_genes","overlap_genes","source_pathway_q_value"]].head(50),use_container_width=True,hide_index=True,height=350)
with donor_tab:
    st.markdown('<div class="section-label">Donor-resolved view</div>',unsafe_allow_html=True); st.subheader("Metadata, QC and out-of-fold prediction")
    selected=st.selectbox("Donor",donors["donor_id"].astype(str).sort_values().tolist()); donor=donors.loc[donors["donor_id"].astype(str).eq(selected)].iloc[0]
    metrics=st.columns(6); metrics[0].metric("Age",f"{fmt(donor.get('age_years'))} y"); metrics[1].metric("Sex",fmt(donor.get("sex"))); metrics[2].metric("Post-QC cells",fmt(donor.get("total_cells_postQC"))); metrics[3].metric("OOF predicted age",fmt(donor.get("elasticnet_OOF_predicted_age"))); metrics[4].metric("Absolute error",fmt(donor.get("absolute_error"))); metrics[5].metric("Age-adjusted deviation",fmt(donor.get("thymic_age_deviation_years")))
    left,right=st.columns([1,1.3])
    with left:
        fields=["gsm_ids","platform","sample_preparation","total_cells_preQC","total_cells_postQC","n_thymus_libraries","has_replicate_libraries","included_primary_analysis","composition_status"]
        st.markdown("#### Metadata and QC"); st.dataframe(pd.DataFrame({"Field":fields,"Value":[fmt(donor.get(f)) for f in fields]}),use_container_width=True,hide_index=True)
    with right:
        st.markdown("#### Captured-library cell composition"); comp=donor_composition.loc[donor_composition["donor_id"].astype(str).eq(selected)].copy(); comp["captured_fraction"]=pd.to_numeric(comp.get("captured_fraction"),errors="coerce"); comp=comp.dropna(subset=["captured_fraction"])
        if comp.empty: st.warning("The donor-level composition source table is not present in the local bundle. Run export_server_inputs.py on the analysis server, then rebuild the release.")
        else: st.plotly_chart(px.bar(comp.sort_values("captured_fraction",ascending=False),x="cell_type",y="captured_fraction",labels={"captured_fraction":"Fraction of captured QC-passed cells","cell_type":""},color_discrete_sequence=["#0b756d"]),use_container_width=True)
with download_tab:
    st.markdown('<div class="section-label">Open-science release</div>',unsafe_allow_html=True); st.subheader("Files, cohorts and provenance")
    st.dataframe(cohorts,use_container_width=True,hide_index=True); st.caption("Only GSE231906 is included in release v1. Candidate external cohorts remain excluded until unique-donor, age, health, preparation and overlap audits are complete.")
    downloadable=sorted(DATA.glob("*"))+sorted(MANIFEST.glob("*"))+sorted((ROOT/"release"/"source_files").glob("*"))+sorted((ROOT/"release"/"figures").glob("*")); columns=st.columns(3)
    for i,path in enumerate(downloadable): columns[i%3].download_button(label=f"Download {path.name}",data=path.read_bytes(),file_name=path.name,mime="application/octet-stream",key=f"download_{path.name}")
    st.markdown("#### Image credits"); st.markdown("- **Healthy Human T Cell** — NIAID/NIH, public domain, via Wikimedia Commons.\n- **Thymus Lobule** — Human Reference Atlas / NIAID NIH BioArt, CC BY 4.0, via Wikimedia Commons.\n\nBoth images are visual context only; neither is a result from this study.")
st.markdown('<div class="footer">Human Thymic Aging Atlas · donor-resolved research resource · <a href="https://github.com/12306-po/Human-Thymic-Aging-Atlas" target="_blank">GitHub source repository</a></div>',unsafe_allow_html=True)
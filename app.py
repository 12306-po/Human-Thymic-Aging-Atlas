from __future__ import annotations

from pathlib import Path
from hashlib import sha256
import json
from urllib.parse import quote

import pandas as pd
import plotly.express as px
import streamlit as st

from atlas_core_data import (
    GEO_DONOR17, LEVELS, coverage, csv_bytes, donor_counts,
    load_core, result_files, source_files, zip_bytes,
)
from atlas_gene_data import (
    corrected_evidence_files, corrected_gene_effects, gene_references, reference_files,
)

ROOT = Path(__file__).resolve().parent
TABS = ["图谱浏览 · Atlas", "供者级证据查询 · Donors", "结果与代码下载 · Downloads"]
PALETTE = ["#0b756d", "#c95858", "#315f91", "#6e9b89", "#d7a14b", "#9477a8", "#5e9ca0"]


@st.cache_data(show_spinner=False)
def core_data():
    return load_core(ROOT)


@st.cache_data(show_spinner=False)
def download_data(donors, cells):
    files = result_files(donors, cells)
    files.update(reference_files(*gene_references(ROOT)))
    files.update(corrected_evidence_files(ROOT))
    manifest_path = "release/core_v1/manifest.json"
    manifest = json.loads(files[manifest_path])
    manifest["gene_reference_scope"] = "Expression summaries and external developmental localization, not old age statistics"
    manifest["file_sha256"] = {name: sha256(data).hexdigest() for name, data in files.items() if name != manifest_path}
    files[manifest_path] = json.dumps(manifest, indent=2, ensure_ascii=False).encode("utf-8")
    return files


@st.cache_data(show_spinner=False)
def gene_data():
    return gene_references(ROOT)


def render_gene_browser():
    st.subheader("基因查询")
    index, expression, development = gene_data()
    options = index.gene.tolist()
    if not options:
        st.info("当前版本没有基因参考表。")
        return
    requested = str(st.query_params.get("gene", "")).upper()
    default = requested if requested in options else ("CD3D" if "CD3D" in options else options[0])
    selected_gene = st.selectbox("基因符号", options, index=options.index(default), key="gene_symbol")
    st.markdown(f"[分享当前基因查询](?view=gene&gene={quote(selected_gene, safe='')})")
    gene_expression = expression.loc[expression.gene.eq(selected_gene)].copy()
    dev = development.loc[development.gene.eq(selected_gene)].copy()
    st.caption(f"索引包含 {len(options):,} 个基因。输入基因符号可检索表达参考、供者覆盖和发育定位。")
    left, right = st.columns([1.1, 1])
    with left:
        st.markdown("#### 各细胞类型的表达参考")
        if gene_expression.empty:
            st.info("该基因没有可用的表达汇总。")
        else:
            fig = px.bar(
                gene_expression.sort_values("AveExpr", ascending=False),
                x="cell_context", y="AveExpr", color="cell_context",
                hover_data=["n_donors", "analysis_level"], color_discrete_sequence=PALETTE,
                labels={"cell_context": "源表细胞类型", "AveExpr": "AveExpr（源表表达汇总值）"},
            )
            fig.update_layout(showlegend=False, height=390)
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(gene_expression, use_container_width=True, hide_index=True)
        st.caption("这里保留表达汇总与供者覆盖，使用原分析的粗粒度细胞标签；未展示旧年龄模型的 beta、P 值或 FDR。")
    with right:
        st.markdown("#### T 细胞发育阶段定位")
        if dev.empty:
            st.info("当前发育参考仅覆盖既有候选基因，该基因尚未覆盖。")
        else:
            metrics = {"标准化表达": "mean_zscore", "表达汇总": "mean_expr", "检出比例（%）": "pct_expr"}
            chosen = st.radio("发育定位指标", list(metrics), horizontal=True)
            fig = px.line(
                dev, x="developmental_stage", y=metrics[chosen], color="source_object", markers=True,
                category_orders={"developmental_stage": ["DN1", "DN2", "DN3", "ISP", "DP_CD3min", "DP_CD3plus"]},
                color_discrete_sequence=PALETTE,
                labels={"developmental_stage": "发育阶段", metrics[chosen]: chosen, "source_object": "参考对象"},
                hover_data=["n_cells", "pct_expr"],
            )
            fig.update_layout(height=390)
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(dev.drop(columns=["evidence_scope"]), use_container_width=True, hide_index=True)
        st.caption("发育定位来自外部人类发育参考；覆盖范围不代表修正后的衰老筛选结果，定位曲线也不是年龄轨迹。")
    st.markdown("#### 修正后的供者级年龄关联")
    try:
        effects = corrected_gene_effects(ROOT)
    except (OSError, ValueError, KeyError) as exc:
        st.error(f"年龄关联数据未通过来源校验：{exc}")
        effects = None
    if effects is None:
        st.info("修正年龄后的基因关联结果尚未导入网站。表达参考可正常查询；年龄效应将在核验新版分析后显示。")
    else:
        rows = effects.loc[effects.gene.eq(selected_gene)]
        if rows.empty:
            st.info("修正后的年龄关联表没有该基因。")
        else:
            st.dataframe(rows.sort_values("q"), use_container_width=True, hide_index=True)
            st.caption("统计单元为供者。beta 使用源模型尺度；此表的显著性不等于已通过 HC3 或影响供者稳定性审计。")
    st.download_button("下载当前基因表达参考", csv_bytes(gene_expression),
                       file_name=f"{selected_gene}_expression_reference.csv", mime="text/csv")
    if not dev.empty:
        st.download_button("下载当前基因发育定位", csv_bytes(dev),
                           file_name=f"{selected_gene}_development_reference.csv", mime="text/csv")


def main():
    st.set_page_config(page_title="Human Thymic Aging Atlas", page_icon="🧬", layout="wide")
    st.markdown("""
    <style>
    :root {--ink:#102a2d;--muted:#5d7072;--teal:#0b756d;}
    .stApp {background:radial-gradient(circle at 8% 0%,rgba(93,189,169,.16),transparent 28rem),radial-gradient(circle at 98% 8%,rgba(88,129,180,.12),transparent 32rem),#fbfdfc;}
    .block-container {max-width:1280px;padding-top:1.5rem;padding-bottom:3rem;}
    h1,h2,h3 {color:var(--ink);letter-spacing:-.02em;}
    .hero-panel {min-height:350px;padding:2rem 2.2rem;border-radius:26px;background:linear-gradient(135deg,#ffffffed,#eef8f5);border:1px solid #d0e4dd;box-shadow:0 20px 55px #23504c13;}
    .hero-kicker {display:inline-flex;padding:.4rem .75rem;border-radius:999px;color:#08665f;background:#dff3ec;font-size:.78rem;font-weight:700;letter-spacing:.05em;}
    .hero-title {font-size:clamp(2.2rem,4.2vw,3.65rem);line-height:1.16;margin:1rem 0;font-weight:760;}
    .hero-copy {font-size:1.02rem;line-height:1.8;color:var(--muted);}
    .pill-row {display:flex;flex-wrap:wrap;gap:.5rem;margin-top:1rem;}
    .pill {padding:.38rem .65rem;border-radius:999px;background:white;border:1px solid #d7e5e1;color:#38585a;font-size:.78rem;}
    .image-credit {font-size:.72rem;color:#78888a;line-height:1.5;margin-top:.3rem;}
    div[data-testid="stImage"] img {border-radius:22px;}
    div[data-testid="stMetric"] {background:#ffffffdd;border:1px solid #dde8e5;padding:.9rem 1rem;border-radius:16px;box-shadow:0 8px 22px #2246430d;}
    .stTabs [data-baseweb="tab-list"] {gap:.35rem;background:#edf4f2;padding:.35rem;border-radius:14px;margin-top:1rem;}
    .stTabs [data-baseweb="tab"] {height:2.85rem;border-radius:10px;padding:0 1rem;color:#486163;}
    .stTabs [aria-selected="true"] {background:white;color:#075f59;box-shadow:0 4px 14px #1e504a14;}
    div[data-testid="stDataFrame"] {border:1px solid #e0e8e6;border-radius:14px;overflow:hidden;}
    </style>
    """, unsafe_allow_html=True)
    hero_left, hero_right = st.columns([1.35, .65], gap="large")
    with hero_left:
        st.markdown(
            '<div class="hero-panel"><span class="hero-kicker">HUMAN THYMIC AGING ATLAS</span>'
            '<div class="hero-title">人胸腺单细胞图谱<br>与供者证据查询</div>'
            '<div class="hero-copy">从细胞状态到基因表达，从供者覆盖到数据来源。'
            '探索分层注释、表达参考和供者级证据，下载可复现的数据与代码。</div>'
            '<div class="pill-row"><span class="pill">GSE231906</span>'
            '<span class="pill">3 个核心入口</span><span class="pill">细胞与基因浏览</span>'
            '<span class="pill">可追溯来源</span></div></div>', unsafe_allow_html=True,
        )
    with hero_right:
        image = ROOT / "assets/healthy_human_t_cell.jpg"
        if image.is_file():
            st.image(str(image), use_column_width=True)
            st.markdown(
                '<div class="image-credit">健康人 T 淋巴细胞显微图 · NIAID/NIH，公共领域。'
                '用于视觉示意，不是本研究结果。</div>', unsafe_allow_html=True,
            )
    st.write("")

    try:
        donors, cells = core_data()
    except (OSError, ValueError, KeyError) as exc:
        st.error(f"图谱数据未通过校验：{exc}")
        st.stop()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("主队列供者", len(donors))
    m2.metric("图谱捕获细胞", f"{len(cells):,}")
    m3.metric("二级细胞状态", cells.atlas_level2.nunique())
    m4.metric("供者年龄范围", f"{donors.age_years.min():g}–{donors.age_years.max():g} 岁")
    st.caption(
        "donor17：GEO 原始年龄 21 months，换算为 1.75 岁。页面、年龄覆盖统计和下载数据使用同一年龄表。"
        "细胞比例描述捕获的细胞库；原取样包含约 6:4 的 CD45+/CD45− 重组，不代表完整胸腺组织比例。"
    )
    atlas_tab, donor_tab, download_tab = st.tabs(TABS)

    with atlas_tab:
        intro_left, intro_right = st.columns([.6, 1.4], gap="large")
        with intro_left:
            lobule = ROOT / "assets/thymus_lobule_nih_bioart.svg"
            if lobule.is_file():
                st.image(str(lobule), use_column_width=True)
                st.caption("胸腺小叶示意 · Human Reference Atlas / NIAID NIH BioArt · CC BY 4.0。非研究数据。")
        with intro_right:
            st.markdown("### 从细胞状态到基因表达")
            st.write("细胞图谱展示分层标签与供者覆盖；基因查询展示表达汇总、发育阶段定位和经过核验的年龄关联。")
            st.caption("基因浏览位于图谱入口内部，网站仍保留三个核心功能。")
        gene_view = st.query_params.get("view") == "gene" or bool(st.query_params.get("gene"))
        mode = st.radio("浏览内容", ["细胞图谱", "基因查询"], index=1 if gene_view else 0, horizontal=True)
        if mode == "基因查询":
            render_gene_browser()
        else:
            st.subheader("分层图谱浏览")
            options = sorted(cells.atlas_level1.unique())
            broad = st.multiselect("一级细胞标签", options, default=options)
            view = cells.loc[cells.atlas_level1.isin(broad)].copy()
            donor_options = donors.donor_id.tolist()
            donor_filter = st.multiselect("供者筛选", donor_options, default=donor_options)
            view = view.loc[view.donor_id.isin(donor_filter)]
            color_options = {"一级标签": "atlas_level1", "二级状态": "atlas_level2",
                             "三级细分状态": "atlas_level3", "供者": "donor_id", "年龄": "age_years"}
            color_label = st.radio("图谱着色", list(color_options), horizontal=True)
            color = color_options[color_label]
            if view.empty:
                st.info("当前筛选没有细胞，请选择细胞标签和供者。")
            else:
                limit = st.select_slider("最多绘制的细胞点", options=[5000, 10000, 20000, 40000], value=20000)
                plotted = view.sample(limit, random_state=371) if len(view) > limit else view
                figure = px.scatter(
                    plotted, x="umap_1", y="umap_2", color=color, render_mode="webgl",
                    hover_data=["donor_id", "age_years", "sex", *LEVELS],
                    labels={"umap_1": "UMAP 1", "umap_2": "UMAP 2", "age_years": "年龄（岁）"},
                    opacity=.65, color_continuous_scale="Teal", color_discrete_sequence=PALETTE,
                )
                figure.update_traces(marker={"size": 2.5})
                figure.update_layout(height=610, margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(figure, use_container_width=True)
                st.caption(f"筛选细胞 {len(view):,} 个；绘制 {len(plotted):,} 个。下面的覆盖统计使用全部筛选细胞。")
                level = st.selectbox("查看标签覆盖", LEVELS, index=1)
                st.dataframe(coverage(view, level), use_container_width=True, hide_index=True)
                st.caption("标签来自现有图谱注释；供者覆盖和年龄范围帮助判断数据可用性，不替代标记基因验证。")

    with donor_tab:
        st.subheader("供者级证据查询")
        selected = st.selectbox("选择供者", donors.donor_id.tolist(),
                                index=donors.donor_id.tolist().index("donor17"))
        donor = donors.loc[donors.donor_id.eq(selected)].iloc[0]
        selected_cells = cells.loc[cells.donor_id.eq(selected)]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("年龄", f"{donor.age_years:g} 岁")
        c2.metric("性别", str(donor.sex))
        c3.metric("该供者图谱细胞", f"{len(selected_cells):,}")
        qc = donor.get("total_cells_postQC")
        c4.metric("源记录 QC 后细胞", "NA" if pd.isna(qc) else f"{int(qc):,}")
        if selected == "donor17":
            st.info("年龄单位已核正：21 months ÷ 12 = 1.75 岁。")
            st.markdown(f"[查看 donor17 的 GEO 来源]({GEO_DONOR17})")

        left, right = st.columns([1, 1.35])
        with left:
            fields = [
                "gsm_ids", "platform", "sample_preparation", "n_thymus_libraries",
                "has_replicate_libraries", "total_cells_preQC", "total_cells_postQC",
                "included_primary_analysis", "age_source_raw", "age_unit_status",
                "diagnosis", "surgical_collection_indication", "library_batch", "sequencing_batch",
            ]
            metadata = pd.DataFrame({
                "字段": fields,
                "值": ["未提供" if pd.isna(donor.get(field)) or str(donor.get(field, "")) == ""
                       else str(donor.get(field)) for field in fields],
            })
            st.dataframe(metadata, use_container_width=True, hide_index=True)
        with right:
            st.markdown("#### 捕获细胞组成")
            counts = donor_counts(cells)
            selected_counts = counts.loc[counts.donor_id.eq(selected)]
            if selected_counts.empty:
                st.info("当前图谱中没有该供者的细胞记录。")
            else:
                chart = px.bar(
                    selected_counts.sort_values("fraction_of_captured_atlas", ascending=False),
                    x="atlas_level2", y="fraction_of_captured_atlas", color="atlas_level1",
                    hover_data=["n_cells", "captured_atlas_denominator"], color_discrete_sequence=PALETTE,
                    labels={"atlas_level2": "细胞状态", "fraction_of_captured_atlas": "占该供者捕获图谱细胞的比例"},
                )
                chart.update_layout(height=420, legend_title_text="")
                st.plotly_chart(chart, use_container_width=True)

        st.markdown("#### 同一细胞状态在各供者中的覆盖")
        pairs = sorted(set(zip(cells.atlas_level1, cells.atlas_level2)))
        pair_options = {" / ".join(item): item for item in pairs}
        pair = pair_options[st.selectbox("细胞状态", list(pair_options))]
        selected_state = cells.loc[cells.atlas_level1.eq(pair[0]) & cells.atlas_level2.eq(pair[1])]
        state_counts = selected_state.groupby("donor_id").size()
        totals = cells.groupby("donor_id").size()
        comparison = donors[["donor_id", "age_years", "sex"]].copy()
        comparison["n_cells"] = comparison.donor_id.map(state_counts).fillna(0).astype(int)
        comparison["captured_atlas_denominator"] = comparison.donor_id.map(totals)
        comparison["fraction_of_captured_atlas"] = comparison.n_cells / comparison.captured_atlas_denominator
        comparison["selected_donor"] = comparison.donor_id.eq(selected)
        comparison["point_size"] = comparison.selected_donor.map({True: 12, False: 6})
        comparison["donor_role"] = comparison.selected_donor.map({True: "当前供者", False: "其他供者"})
        fig = px.scatter(
            comparison, x="age_years", y="fraction_of_captured_atlas", color="donor_role",
            color_discrete_map={"当前供者": "#c95858", "其他供者": "#5e9ca0"},
            size="point_size", size_max=14,
            hover_data=["donor_id", "sex", "n_cells", "captured_atlas_denominator"],
            labels={"age_years": "年龄（岁）", "fraction_of_captured_atlas": "占该供者捕获细胞的比例",
                    "donor_role": "供者"},
        )
        fig.update_layout(height=360)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(comparison.drop(columns=["point_size", "selected_donor"]), use_container_width=True, hide_index=True)
        st.caption("每个点是一位供者。此处显示描述性组成，不将细胞视为独立生物学重复，也不展示未校正的年龄效应。")
        st.download_button("下载当前供者元数据", csv_bytes(donors.loc[donors.donor_id.eq(selected)]),
                           file_name=f"{selected}_metadata.csv", mime="text/csv")
        st.download_button("下载当前供者细胞组成", csv_bytes(donor_counts(cells).query("donor_id == @selected")),
                           file_name=f"{selected}_cellstate_counts.csv", mime="text/csv")

    with download_tab:
        st.subheader("结果与代码下载")
        st.write("下载修正供者表、图谱坐标、分层覆盖、供者级细胞组成与基因表达参考。")
        files = download_data(donors, cells)
        source = source_files(ROOT)
        a, b, c = st.columns(3)
        a.download_button("下载结果数据包", zip_bytes(files),
                          file_name="thymus_atlas_core_results.zip", mime="application/zip")
        b.download_button("下载网站源代码", zip_bytes(source),
                          file_name="thymus_atlas_core_code.zip", mime="application/zip")
        c.download_button("下载完整复现包", zip_bytes({**files, **source}),
                          file_name="thymus_atlas_core_reproducible.zip", mime="application/zip")
        with st.expander("下载单独的供者表与覆盖表"):
            st.download_button("修正后的供者表 CSV", csv_bytes(donors),
                               file_name="donor_metadata_corrected.csv", mime="text/csv")
            for level in LEVELS:
                st.download_button(f"{level} 覆盖表 CSV", csv_bytes(coverage(cells, level)),
                                   file_name=f"{level}_coverage.csv", mime="text/csv")
        st.markdown("#### 版本与数据来源")
        st.write("donor17 的原始记录为 21 months；当前版本统一换算为 1.75 岁。")
        st.write("当前下载包不包含旧年龄训练的预测、误差、年龄偏差，也不包含尚未导入修正版本的年龄关联结果。")
        st.markdown(f"[GEO donor17]({GEO_DONOR17}) · [GSE231906](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE231906)")
        st.markdown("[网站源码与版本历史](https://github.com/12306-po/Human-Thymic-Aging-Atlas)")
        st.caption("复现：解压完整复现包，安装 requirements-core.txt，然后执行 streamlit run app.py。")

    st.caption("Human Thymic Aging Atlas · 供者级研究资源")


if __name__ == "__main__":
    main()

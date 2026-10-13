from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from atlas_core_data import (
    GEO_DONOR17, LEVELS, coverage, csv_bytes, donor_counts,
    load_core, result_files, source_files, zip_bytes,
)

ROOT = Path(__file__).resolve().parent
TABS = ["图谱浏览 · Atlas", "供者级证据查询 · Donors", "结果与代码下载 · Downloads"]


@st.cache_data(show_spinner=False)
def core_data():
    return load_core(ROOT)


@st.cache_data(show_spinner=False)
def download_data(donors, cells):
    return result_files(donors, cells)


def main():
    st.set_page_config(page_title="Human Thymic Aging Atlas", page_icon="🧬", layout="wide")
    st.markdown("""
    <style>
    .stApp {background:#f7faf9;}
    .block-container {max-width:1280px;padding-top:1.6rem;}
    h1,h2,h3 {color:#163d3b;}
    div[data-testid="stMetric"] {background:#fff;border:1px solid #dce8e4;border-radius:12px;padding:12px;}
    .stTabs [data-baseweb="tab-list"] {gap:14px;}
    .stTabs [data-baseweb="tab"] {height:50px;}
    </style>
    """, unsafe_allow_html=True)
    st.title("人胸腺单细胞图谱与供者证据查询")
    st.caption("Human Thymic Aging Atlas · GSE231906 · 分层标签、供者覆盖与可复现的数据")

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
                opacity=.65, color_continuous_scale="Teal",
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
                    hover_data=["n_cells", "captured_atlas_denominator"],
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
        fig = px.scatter(
            comparison, x="age_years", y="fraction_of_captured_atlas", color="selected_donor",
            size="point_size", size_max=14,
            hover_data=["donor_id", "sex", "n_cells", "captured_atlas_denominator"],
            labels={"age_years": "年龄（岁）", "fraction_of_captured_atlas": "占该供者捕获细胞的比例",
                    "selected_donor": "当前供者"},
        )
        fig.update_layout(height=360)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(comparison.drop(columns=["point_size"]), use_container_width=True, hide_index=True)
        st.caption("每个点是一位供者。此处显示描述性组成，不将细胞视为独立生物学重复，也不展示未校正的年龄效应。")
        st.download_button("下载当前供者元数据", csv_bytes(donors.loc[donors.donor_id.eq(selected)]),
                           file_name=f"{selected}_metadata.csv", mime="text/csv")
        st.download_button("下载当前供者细胞组成", csv_bytes(donor_counts(cells).query("donor_id == @selected")),
                           file_name=f"{selected}_cellstate_counts.csv", mime="text/csv")

    with download_tab:
        st.subheader("结果与代码下载")
        st.write("下载与当前页面一致的修正供者表、图谱坐标、分层覆盖和供者级细胞组成。")
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

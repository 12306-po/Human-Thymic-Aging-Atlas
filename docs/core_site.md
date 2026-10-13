# 人胸腺单细胞图谱与供者证据查询系统

当前网站保留三个功能：图谱浏览、供者级证据查询、结果与代码下载。

图谱入口内部包含“细胞图谱”和“基因查询”两个视图。恢复健康人 T 细胞显微图与胸腺小叶示意图；它们是署名清楚的装饰/解剖示意，不是研究结果。页面使用青绿色主题。

## 使用

图谱页支持三级标签、供者与年龄着色，并显示按当前筛选细胞计算的供者覆盖和年龄范围。供者页显示取样信息、QC、细胞组成及各供者的描述性数据点。所有年龄字段来自同一修正供者表。

donor17 的 GEO GSM8945775 原始年龄为 21 months，换算为 1.75 年。代码只在核对该供者与该 GSM 对应关系后执行已审计的单位修正，拒绝无法识别的年龄值。

当前版取消新样本预测、PBMC 预测研究入口、社区上传及独立的旧年龄模型页面。旧预测、误差、年龄偏差和未核正的年龄关联结果不进入当前页面或下载包。历史分析文件保留在仓库，但不构成当前版本的校正结果。

细胞组成以捕获图谱细胞为分母；取样涉及约 6:4 的 CD45+/CD45− 重组，因此不代表完整胸腺原生组成。描述性图中每个点是一位供者；不报告从细胞数推断出的群体年龄显著性。

## 数据与复现

完整复现包包含页面代码、最小依赖、修正供者表、图谱坐标、供者级组成表、三级覆盖表及数据 SHA256。

```bash
python -m pip install -r requirements-core.txt
streamlit run app.py
```

供者表：`release/core_v1/data/donor_metadata.csv`。
图谱坐标：`release/biology_atlas_v2/data/hierarchical_umap_source.csv.gz`。

若修正供者表不存在，代码会从旧供者汇总提取允许的元数据列并进行单位修正；旧预测字段不会随之进入页面。覆盖表从细胞与修正供者表重新计算，不沿用旧年龄范围。

源记录：https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM8945775

## 基因查询

基因索引、细胞类型表达汇总与外部发育阶段定位均可浏览和下载。表达参考只保留 gene、cell_context、AveExpr、n_donors、analysis_level，不沿用旧模型的年龄 beta、P 值或 FDR。

发育定位参考仅覆盖原有候选基因；它的覆盖范围不代表修正后的衰老基因筛选，也不构成年龄轨迹。表达参考使用源表粗粒度细胞标签，与图谱二级状态不是逐一对应关系。

以下文件为干净的参考包：`gene_index.csv`、`gene_expression_reference.csv.gz`、`gene_development_reference.csv.gz`，位于 `release/core_v1/data`。

## 导入修正后的基因年龄效应

服务器已有完成的 Stage03 修正结果时，可运行：

```bash
PROJECT=/data/zxy/projects/human_thymus_age_ML_DL
REPO="$PROJECT/Human-Thymic-Aging-Atlas"
PYTHON=/data/zxy/environments/micromamba_envs/human_thymus_age_ml_dl/bin/python
"$PYTHON" "$REPO/import_corrected_gene_evidence.py" \
  --root "$REPO" \
  --stage "$REPO/release/multicohort_atlas_v1/stage03_corrected_age_v1"
```

导入器核对 Stage03 provenance 状态、donor17=1.75、全部供者年龄，以及源供者表和基因结果 SHA256。通过后才生成 `release/core_v1/gene_evidence`；将该目录提交到网站仓库即可显示修正后的年龄关联。导入不会把普通 FDR 自动解释成 HC3 稳健性结论。

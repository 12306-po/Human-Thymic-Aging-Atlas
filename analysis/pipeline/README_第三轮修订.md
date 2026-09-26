# 第三轮投稿图：修订版代码与验收说明

这是一套**保留原版、独立输出**的修订源码。原目录 `F:\论文材料\图片代码` 和已有 `figures_final` 图像均未覆盖。当前交付解决了已能从代码定位的逻辑与制图错误；它**不是已完成的数据核验或投稿终稿**。

## 运行

在拥有原始冻结分析项目的环境中，先安装项目原环境依赖（至少 Python、numpy、pandas、scipy、matplotlib、anndata；主图 1/补图 S02–S03 需要 anndata），然后运行：

```bash
python 99_make_paper_figures.py \
  --project /data/zxy/projects/human_thymus_age_ML_DL \
  --output /data/zxy/projects/human_thymus_age_ML_DL/figures_final_revised \
  --hallmark-gmt /absolute/path/h.all.v2023.2.Hs.symbols.gmt
```

`--main-only` 或 `--supp-only` 可分别重画。程序拒绝把输出直接写入原 `figures_final`，单图失败时返回非零退出码；不会把旧图当新图交付。代码路径记录在修订版 `figure_source_manifest.tsv`。最小代码检查：

```bash
python -m compileall -q .
python -m unittest discover -s tests -v
```

## 已实施的修改

| 对象 | 代码上的修复 | 自动审计输出 |
|---|---|---|
| Figure 1 | A 缩成四步短流程；B 供者 ID 独立 y 轴并用形状区分性别；C 独占整行；F 取消拥挤的图内群名并用完整图例；G 扩为 2×3。 | — |
| Figure 2D/F | D 修复基因/供者矩阵方向，限 16 个实际存在的基因、缺失专色、年龄/性别独立顶栏；F 从真实 `Count` 计算点面积和正整数图例，`GeneRatio=Count/全部年龄相关基因`，对全部测试的 Hallmark 集合做 BH 校正。 | `fig2d_heatmap_row_z_source.tsv`、`fig2f_all_hallmark_tests.tsv`、`fig2f_hallmark_enrichment_source.tsv` |
| Figure 3 | A/B/F 重排；MeanAgeNull 不再用 Spearman ρ 暗示生物关联；核查每模型 18 个唯一、同一批 OOF 供者，并重算 R²/MAE 对照结果表；置换矩阵与运行摘要核对，缺少 BH/FWER 字段直接报错。 | `fig3_oof_predictions_audit.tsv`、`fig3d_ablation_oof_audit.tsv`、`fig3e_null_summary_audit.json`、`fig3g_oof_donor_residual_audit.tsv` |
| Figure 4 | 候选改为明示 ML/OOF-IG/BH 三条路线并输出入选理由；主图只展示排序前 20 个，不把未通过校正的候选称为“校正集”；两个对象共用明确的双向 `RdBu_r` 和 `TwoSlopeNorm`，色标由同一 `ScalarMappable` 生成，绝不从末尾空散点取色标。 | `fig4_candidate_selection_audit.tsv`、`fig4_color_scale_audit.tsv`、`fig4_dotplot_source.tsv` |
| Figure 5 | RNA/ATAC 从同一重建母表取值并断言本轮 160/257、122/256；E 全部改为“网络记录行数”的独立摘要而非混单位漏斗；F 逐基因区分同向、反向、已评估无支持、未评估并附排除理由；F/G 顺序连续。 | `fig5_cross_species_mother_audit.tsv`、`fig5e_regulatory_row_audit.tsv`、`fig5f_candidate_evidence_audit.tsv`、`fig5f_state_counts.tsv` |
| S01–S08 | 供者标签引线、图例外移；S04 敏感性改表；S06 去掉各图重复的固定预测置换 P；S07 AUC 限于 0–1 且显示 0 特异度的原始计数；S08 的泄漏诊断单独导出为 S08b。 | `s04_doublet_sensitivity_summary.tsv` 等 |

## 必须由原始数据负责人完成的验收

1. 运行上述命令并确认所有 Figure 1–5、S01–S09 及 S08b 都实际生成，且没有失败日志；本机不能完成这步，因为 `F:\论文材料` 内缺少原项目冻结结果表/H5AD/GMT，本机绘图环境也未安装 `anndata`。
2. 用输出的审计表逐行核对 Figure 2F 的 Count、GeneRatio、BH q；Figure 3 的 OOF 预测、消融和置换；Figure 4 的实际绘出候选与色标；Figure 5 的母表映射与每格状态。不要仅凭新 PNG 认可数值。
3. 以目标期刊的**最终印刷尺寸**逐页检查文字、图例、坐标和字母；本次没有原始数据可做最终渲染检查。若仍挤，应拆到补图，而不是继续缩字。
4. 同步修订正文与图注：Figure 2A 是 sex-adjusted limma，2B/5 人端是未校正 Spearman；Figure 3 是 n=18 的内部 LODO；Figure 4 是外部**发育背景定位**而非衰老复制；Figure 5 小鼠每年龄组仅 1 只，不能把基因/peak 当生物学重复或称机制验证。
5. 如果重新分析改变了上述冻结计数，先更新审查记录并重新签核；不要只绕过代码断言。

修订按 `第三轮投稿级图片审查与修改指令.md` 实施，采用 [Kassis 等《Scientific Agent Skills》](https://doi.org/10.48550/arXiv.2609.00065)的图表溯源与证据边界原则；该方法学参考不构成任何本研究数值的证据。
# Figure 5C：Sort2 DP 列修正版

旧绘图代码将源表中的 `DP_CD3min`、`DP_CD3plus` 错写成带空格的
`DP CD3min`、`DP CD3plus`，导致 `reindex()` 生成两列全 NA。修正版脚本：

```bash
PY_BIN=/data/zxy/environments/microbiome_envs/human_thymus_age_ml_dl/bin/python
project=/data/zxy/projects/human_thymus_age_ML_DL

"$PY_BIN" \
  "$project/scripts/40_plot_figure5_corrected.py" \
  --project "$project" \
  --font-dir "$HOME/fonts/times_new_roman"
```

默认输出目录：

```text
$project/10_results/figures_manuscript/
```

主要输出：`Figure5_corrected.png/pdf`、`Figure5C_corrected.png/pdf`、
`Figure5C_source_table.tsv` 和 `Figure5C_validation.tsv`。

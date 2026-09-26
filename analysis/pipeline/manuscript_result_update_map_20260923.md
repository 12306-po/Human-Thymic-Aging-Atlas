# 修订稿最终结果回填映射

本文件只定义回填位置和数据来源，不预填尚未运行的数值。

| 稿件位置 | 当前临时内容 | Step 30 后的数据来源 | 回填原则 |
|---|---|---|---|
| Abstract，第 2 段 | `Elastic Net R²=0.67, MAE=9.2 years`；fully nested 尚未完成 | `04_machine_learning/fully_nested/model_metrics.tsv`、`model_metrics_bootstrap_ci.tsv`、`paired_model_vs_null_MAE.tsv` | 换成新 OOF 点估计、CI 和配对 ΔMAE；仍称 internal donor-level validation |
| Abstract，第 2 段 | composition 为未调整 Spearman | `03_feature_selection/composition_adjusted/composition_results.tsv`、`composition_dirichlet_sensitivity.tsv` | 报 CLR age+sex 结果和方向；说明相对组成，不写绝对细胞减少/增加 |
| Abstract，第 3–4 段 | 人鼠使用 unadjusted human age correlation；结论 provisional | `08_mouse_validation/sex_adjusted_descriptive/28_cross_species_descriptive_summary.tsv` | 可更新方向计数，但必须保留 single-pair、descriptive、hypothesis-generating |
| Results：Study design and donor cohort | donor manifest 尚未完成 | `01_raw_processing/metadata/submission_manifest/donor_manifest.csv`、missingness/provenance/audit 表 | 只写已核实字段；不可用临床/取材字段继续标 NA 和 residual confounding |
| Results：composition | composition 尚需 age+sex 模型 | Step 27 全部输出 | 主文优先 CLR age+sex；Dirichlet 仅 sensitivity；加入 donor stability |
| Results：machine learning | 当前 Step 11 为探索性 | Step 26 全部输出 | 替换模型性能、ablation、模型-空模型配对差、donor influence、age-error pattern |
| Methods：Donor metadata | manifest 应补建 | Step 25 contract 与 provenance | 写明来源字段和缺失策略；不推断诊断、取材、批次或伦理号 |
| Methods：Composition | 最终应加 log-ratio/Dirichlet | `composition_analysis_contract.json` | 写明 all-QC denominator、half-cell zero replacement、CLR、age_z+sex、HC3、BH |
| Methods：Machine learning | screening 未完全嵌入 inner folds | `fully_nested_analysis_contract.json`、inner/outer audit | 明确每个 inner-training fold 内筛选、插补、标准化、调参；outer donor 始终隔离 |
| Methods：Human–mouse | 使用 WT Spearman 方向 | `28_analysis_contract.txt` | 改为 sex-adjusted limma `β_age`；不做 animal-level P value |
| Figure 2 及图注 | composition panel 为 Spearman | Step 27 source table | 重画为 CLR age+sex effect/CI，或将旧 Spearman 明确保留为 descriptive supplement |
| Figure 3 及图注 | 当前探索性 ML 数值 | Step 26 source tables | 主图所有性能和 ablation 必须统一来自 fully nested 输出 |
| Supplementary Fig. S2 | 当前 female-only/模型敏感性 | Step 26 sensitivity cohort 输出 | 更新数值和 n；注明 restricted-age 规则 |
| Supplementary Fig. S4 | human WT Spearman 与 mouse | Step 28 source tables | x 轴改为 sex-adjusted human limma `β_age`；标题与图注保留 single-pair warning |
| Data/Code availability | 待冻结 source data 与代码 | `10_results/submission_completion/submission_source_data_manifest.tsv` | 只在仓库 commit/release/DOI 实际生成后填入；不得预写 DOI |

## 不因重跑而取消的限制

- 18 名供者仍是小型、性别不均衡的内部队列。
- 没有独立成人胸腺衰老验证队列。
- HumanThymusFormer 仍是 post-hoc representation and attribution model。
- GSE195812 仍只提供 developmental context。
- 小鼠仍为 1 Young + 1 Aged，不支持 population-level validation 或 conserved mechanism。
- 伦理审批号、知情同意和动物伦理信息必须从原论文/GEO/数据提供方准确复制，不能由分析代码生成。

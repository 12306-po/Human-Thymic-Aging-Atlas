# 统计审查后补充分析（Steps 25–30）

本补丁依据 `manuscript_pre_submission_action_list.docx` 和修订稿中的明确边界编写。它保留 Steps 00–24 及其探索性结果，所有新结果写入独立目录；在 Step 30 通过前，不应删除稿件中的 `provisional`、`exploratory` 或 `should be rerun`。

## 新增步骤

| Step | 脚本 | 目的 | 主要输出 |
|---|---|---|---|
| 25 | `25_build_submission_donor_manifest.py` | 整合 18 名供者、GSM/文库、年龄、性别、平台、前后 QC 细胞数；缺失临床/取材信息保持 NA 并记录原因 | `01_raw_processing/metadata/submission_manifest/` |
| 26 | `26_fully_nested_donor_ML.py` | outer LODO；每个 inner-training fold 内重新做缺失过滤、方差过滤、特征筛选、插补、标准化和调参 | `04_machine_learning/fully_nested/` |
| 27 | `27_composition_aware_age.py` | CLR age+sex 主模型、HC3 不确定性、供者稳定性、年龄范围/非线性/female-only，以及 joint Dirichlet sensitivity | `03_feature_selection/composition_adjusted/` |
| 28 | `28_cross_species_sex_adjusted.R` | 用 sex-adjusted human limma `β_age` 重算人鼠方向；只输出描述性计数和不可评估原因 | `08_mouse_validation/sex_adjusted_descriptive/` |
| 29 | `29_expression_age_shape_sensitivity.R` | 表达层去除极端供者、18–64 岁、female-only、二次年龄项和逐供者影响 | `03_feature_selection/expression_sensitivity/` |
| 30 | `30_build_submission_result_packet.py` | 检查 25–29 输出，生成带 SHA-256 的 source-data manifest 和稿件回填摘要；不直接改 Word | `10_results/submission_completion/` |

## 关键统计边界

- Step 26 是内部 donor-level validation，不是独立外部验证，也不支持临床年龄预测结论。
- Step 27 的系数是相对组成的 log-ratio 效应，不是绝对细胞数变化。
- Step 28 的小鼠设计仍为 1 Young + 1 Aged；基因和峰不是生物学重复，不做动物水平显著性检验，不使用 `validation` 或 `conserved mechanism` 作为结论。
- Step 25 不推断诊断、手术原因、取材、批次或伦理号。公共记录中没有的字段保持 NA。
- HumanThymusFormer 的定位不变：post-hoc representation and attribution model，不与 Elastic Net 做性能优劣竞争。

## 服务器运行

先把本目录新增的 `25`–`30` 脚本和两个 runner 同步到：

`/data/zxy/projects/human_thymus_age_ML_DL/figures-0920/`

然后执行只读检查：

```bash
bash /data/zxy/projects/human_thymus_age_ML_DL/figures-0920/run_submission_completion_25_to_30.sh --check
```

检查通过后运行：

```bash
bash /data/zxy/projects/human_thymus_age_ML_DL/figures-0920/run_submission_completion_25_to_30.sh --execute
```

默认 Python 与 R 环境分别为：

```text
/data/zxy/environments/micromamba_envs/human_thymus_age_ml_dl/bin/python
/data/zxy/environments/micromamba_envs/r_validation/bin/Rscript
```

如果实际路径不同，可在命令前设置 `PY_BIN` 或 `R_BIN`。Step 26 计算量最大；Step 27 的 Dirichlet donor bootstrap 默认 500 次。不要并行启动两份相同流程。

## 完成判定

成功结束后，应首先查看：

```text
/data/zxy/projects/human_thymus_age_ML_DL/10_results/submission_completion/final_result_summary.md
/data/zxy/projects/human_thymus_age_ML_DL/10_results/submission_completion/submission_source_data_manifest.tsv
/data/zxy/projects/human_thymus_age_ML_DL/10_results/submission_completion/manuscript_update_gate.json
```

只有 `manuscript_update_gate.json` 生成且 `safe_to_replace_provisional_numeric_results=true` 时，才用 `final_result_summary.md` 中的最终数值替换稿件的临时 ML、组成和人鼠方向描述。即使通过，以下限定仍保留：内部验证、相对组成效应、单对小鼠描述性证据、伦理信息必须来自权威原始记录。

## 复现与审计

- 随机种子：371。
- ML inner model selection：平均 inner-fold MAE。
- composition 主分析：all-QC denominator，donor-specific half-cell zero replacement，CLR，`age_z + sex`，HC3 标准误，BH 校正。
- 年龄范围敏感性：18–64 岁；另单独去除最年轻、最年长及两端供者。
- 所有稿件回填值均由 Step 30 记录源文件路径和 SHA-256。

本次科学写作与报告边界审计参考了 Scientific Writing Agent Skills 的可追溯性原则：Kassis T, et al. *Scientific Writing Agent Skills: A Benchmark and Evidence-Traceable Framework for Scientific Writing Agents*. arXiv:2609.00065v2 (2026). doi:10.48550/arXiv.2609.00065。

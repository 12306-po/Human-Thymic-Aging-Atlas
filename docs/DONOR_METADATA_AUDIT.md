# Donor Metadata Audit — GSE231906 Primary Cohort（供者元数据审计）

> 生成日期：2026-10-03 · 数据来源：`release/data/donor_summary.csv`（发布版冻结队列）
> 审计表：`docs/donor_metadata_audit.csv`（18 供者 × 24 字段，机器可读）

---

## 1. 审计结论（Executive Summary）

| 类别 | 字段 | 状态 |
|---|---|---|
| **已解析（队列层面可得）** | age_years、sex、platform（GPL24676）、sample_preparation（combined_CD45pos_CD45neg_6to4）、n_thymus_libraries、has_replicate_libraries、total_cells_preQC/postQC、gsm_ids、included_primary_analysis | 全部 18 供者一致，无冲突 |
| **未解析（公共元数据不可得）** | diagnosis、surgical_collection_indication、health_comorbidity、procurement_site、procurement_method、processing_time、library_batch、sequencing_batch | 全部 18 供者 **8/8 缺失** |
| 混杂检验能力 | age–batch / 年龄–诊断 / 年龄–取材 关联 | **不可检验**（数据不可得性，非遗漏） |

**核心结论**：GEO 公共层（GSE231906 的 `char_*` characteristic 与平台注册信息）仅提供组织、年龄、性别、细胞类型（制备）、平台等 5 类特性；诊断、手术指征、取材方式、处理时间、文库批次、测序批次 8 个字段在该数据集的公共元数据中根本不存在。因此"age–batch 混杂无法排除"不是分析缺口，而是**公开数据可达性（data availability）硬约束**，投稿时应作为数据限制在 Discussion 中正面声明（建议措辞见 §3），并在文中给出本审计表。

> 验证依据：`analysis/pipeline/04_build_final_thymus_donor_metadata.py` 逐库解析
> `03B_GSE231906_primary_library_with_GEO_metadata.csv` 后生成 04A–04H；
> 发布版 `donor_summary.csv` 对上述 8 字段统一标记
> `not available in current public metadata`（`missing_data_reasons` 字段留痕）。

---

## 2. 供者级审计表（18 供者）

完整机器可读表见 `docs/donor_metadata_audit.csv`。概览如下（QC 数字来自发布版冻结队列）：

| donor | age | sex | GPL | prep | n_lib | preQC | postQC | primary | outer_fold | 8 个临床/批次字段 |
|---|---|---|---|---|---|---|---|---|---|---|
| donor13 | 4 | F | GPL24676 | combined | 1 | 7,494 | 6,653 | True | 5 | 全 NA |
| donor4 | 5 | F | GPL24676 | combined | 1 | 11,544 | 9,921 | True | 13 | 全 NA |
| donor11 | 9 | M | GPL24676 | combined | 1 | 7,479 | 6,620 | True | 3 | 全 NA |
| donor5 | 12 | F | GPL24676 | combined | 1 | 8,342 | 7,146 | True | 14 | 全 NA |
| donor17 | 21 | F | GPL24676 | combined | 1 | 7,007 | 6,127 | True | 9 | 全 NA |
| donor12 | 22 | M | GPL24676 | combined | 1 | 5,252 | 4,696 | True | 4 | 全 NA |
| donor3 | 24 | F | GPL24676 | combined | 1 | 7,766 | 5,629 | True | 12 | 全 NA |
| donor9 | 24 | F | GPL24676 | combined | 1 | 5,124 | 3,794 | True | 18 | 全 NA |
| donor10 | 35 | F | GPL24676 | combined | 1 | 3,247 | 2,768 | True | 2 | 全 NA |
| donor8 | 35 | F | GPL24676 | combined | 1 | 6,605 | 5,442 | True | 17 | 全 NA |
| donor6 | 35 | F | GPL24676 | combined | 1 | 11,258 | 8,508 | True | 15 | 全 NA |
| donor16 | 46 | F | GPL24676 | combined | 1 | 7,138 | 6,263 | True | 8 | 全 NA |
| donor2 | 48 | F | GPL24676 | combined | 1 | 9,147 | 7,403 | True | 11 | 全 NA |
| donor15 | 53 | M | GPL24676 | combined | 1 | 5,681 | 4,968 | True | 7 | 全 NA |
| donor7 | 56 | F | GPL24676 | combined | 1 | 7,251 | 6,532 | True | 16 | 全 NA |
| donor1 | 56 | F | GPL24676 | combined | 1 | 5,513 | 4,864 | True | 1 | 全 NA |
| donor18 | 64 | F | GPL24676 | combined | 1 | 12,298 | 11,141 | True | 10 | 全 NA |
| donor14 | 69 | M | GPL24676 | combined | 1 | 7,447 | 6,482 | True | 6 | 全 NA |

> 8 个未解析字段：diagnosis、surgical_collection_indication、health_comorbidity、
> procurement_site、procurement_method、processing_time、library_batch、sequencing_batch。

---

## 3. 投稿 Discussion 建议段落（可整段引用）

### 3.1 英文版（English, ready-to-use）

> **Limitations regarding donor-level confounding.** The primary cohort is derived from
> GSE231906, whose public GEO metadata provide only donor age, sex, tissue (thymus),
> preparation (combined CD45-positive/CD45-negative at a 6:4 ratio) and platform
> (GPL24676) for the 18 donors included. Fields that would permit direct assessment of
> confounding by clinical or technical factors — donor diagnosis, surgical collection
> indication, comorbidity, procurement site, procurement method, processing time,
> library batch and sequencing batch — are not available in the public metadata of
> this dataset (full donor-level audit table in the repository:
> `docs/donor_metadata_audit.csv`). Consequently, potential confounding between
> chronological age and technical batch, or between age and clinical indication for
> thymectomy, cannot be formally tested or excluded in this cohort, and the reported
> age-associated signatures should be interpreted with this caveat in mind. We note
> several mitigating design choices: (i) all 18 donors were profiled on a single
> platform (GPL24676) with a single, uniformly prepared combined CD45+/CD45− library
> layout, which limits the most obvious platform- and preparation-driven batch
> structure; (ii) age-stratified, donor-disjoint fit/validation splits and leave-one-
> donor-out cross-validation ensure that no donor contributes to both training and
> held-out evaluation, so batch structure shared across donors cannot inflate the
> reported out-of-fold estimates; and (iii) the strict fold-internal evaluation
> (fold-specific token selection, scalers and early stopping on training donors only)
> further guards against feature-selection leakage. Nevertheless, residual confounding
> by unmeasured clinical or technical covariates cannot be ruled out, and direct
> validation on an independent cohort with complete donor-level metadata remains a
> priority. We encourage deposition of donor-level clinical and batch metadata by the
> data owners to enable formal confounding checks in future analyses.

### 3.2 中文版（Chinese, for reference）

> **供者层面混杂的限制。** 主队列来自 GSE231906，其公共 GEO 元数据仅提供 18 名供者的
> 年龄、性别、组织（胸腺）、制备方式（CD45 阳性/阴性 6:4 混合）与平台（GPL24676）。
> 可用于直接检验临床或技术混杂的字段——供者诊断、手术取材指征、合并症、取材部位、
> 取材方法、处理时间、文库批次与测序批次——在该数据集的公共元数据中不可得（供者级
> 完整审计表见仓库 `docs/donor_metadata_audit.csv`）。因此，本队列无法正式检验或排除
> 实际年龄与技术批次之间、或年龄与胸腺切除临床指征之间的混杂，所报告的年龄相关特征
> 应在此前提下解读。本研究的若干设计可部分缓解该风险：（i）全部 18 名供者均使用同一
> 平台（GPL24676）与同一统一的 CD45+/CD45− 6:4 混合文库方案，消除了最明显的平台与
> 制备驱动的批次结构；（ii）年龄分层、供者不相交的 fit/validation 划分与留一供者交叉
> 验证确保任一供者不会同时进入训练与留出评估，跨供者共享的批次结构无法抬高所报告的
> 折外指标；（iii）严格折内评估（仅在训练供者上做折内 token 选择、缩放与早停）进一步
> 防止特征选择泄漏。然而，未测临床或技术协变量的残余混杂仍无法排除，在拥有完整供者
> 级元数据的独立队列上直接验证仍是优先事项。我们建议数据所有者补充提交供者级临床与
> 批次元数据，以便在后续分析中开展正式混杂检验。

---

## 4. 可执行的后续动作（按优先级）

1. **投稿时**：在 Methods（数据来源）与 Discussion 引用本审计表与 §3.1 段落（建议同时附
   `donor_metadata_audit.csv` 为补充材料）。
2. **联系数据所有者**：GSE231906 提交者/关联原始发表（胸腺单细胞图谱）作者，索取 18 供者
   的诊断、取材与批次信息；取得后更新 `04` 系列脚本并在复现批次中补充年龄–批次相关性检验
   （如 Kruskal–Wallis 或 bootstrap 置换检验）。
3. **敏感性分析（可选，投稿加分项）**：以"供者实际测序的 GSM 的 GEO 提交时间/文库编号"
   为可得的近似批次代理，检验年龄与近似批次的 Spearman 相关性（若成立需在文中声明代理性质）。
4. **独立验证**：外部队列 HRA007984 的元数据如含诊断/取材信息，可先行完成跨队列的
   混杂可检验性声明（现有 4 供者外部验证见 `analysis/external_validation/`）。

---

## 5. 审计口径与方法

- 已解析字段的判定：`donor_summary.csv` 中非空、非 `not available` 且跨库一致（由
  `04_build_final_thymus_donor_metadata.py` 的 `donor_consistency_ok` 保证）。
- 未解析字段的判定：值为 `NaN` 或 `not available in current public metadata`
  （与 `missing_data_reasons` 文本一致）。
- 生成脚本（可复跑）：仓库内 `docs/make_donor_metadata_audit.py` 从
  `release/data/donor_summary.csv` 重新生成 `docs/donor_metadata_audit.csv`。

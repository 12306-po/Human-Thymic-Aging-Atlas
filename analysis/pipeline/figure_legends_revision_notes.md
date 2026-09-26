# 图注同步修改清单（需数据负责人用新图与审计表签核）

- Figure 1：B 的性别编码为圆/方，颜色为年龄组；G 图内仅显示 ρ 与 BH q，bootstrap CI 请在数据表或图注中列出。UMAP 颜色查外置全图例。
- Figure 2：A 的前 6 个 BH q 排名基因保存在 `fig2a_top_six_genes_audit.tsv`，不再用穿越标题的引线标注；D 为实际匹配到表达矩阵的前 16 行，灰色是缺失/不可标准化，并非表达量 0；F 的 `Count` 是年龄相关基因与通路重叠数，`GeneRatio=Count/全部年龄相关基因数`，背景为 limma 中受检基因，BH 对所有测试的 Hallmark 集合计算，年龄上调与下调基因合并分析。需明确 GMT 的版本和文件校验值。
- Figure 3：B 的 MeanAgeNull 不报告可误解为生物关系的 ρ；E 说明零值在线性区、其余在对数区的 symlog 轴及全流程置换次数；F 是**观察到的复现次数**，不是经多重校正的稳定 biomarker。R²/CI 来自内部 LODO，非外部验证。
- Figure 4：A 为每个发育状态的细胞数和实际人体供者数；B/C 为 Sort1/Sort2 各对象内的候选定位点图；D/E 只展示同一对象内较早与较晚状态的均值 z-score 差值绝对值前 8 个基因，**描述性展示、没有显著性检验**。用 `fig4_candidate_selection_audit.tsv` 中的实际路线解释 20 个主图候选；若 BH 路线为 0，不得写“来自 permutation-null-corrected gene set”。颜色来自各对象内所有细胞的基因级 z-score 后的各状态均值，固定双向色标由审计表给出；不得据此比较 Sort1 与 Sort2 的原始表达幅度，也不得把 D/E 称作增龄趋势。`fig4_state_coverage_audit.tsv` 和 `fig4_stage_delta_audit.tsv` 保留全部源数值。根据 `17_external_human_validation.R`，表达输入是 Seurat `RNA` assay 的标准化 `data` layer，`% expressing = mean(data > 0) × 100`，分母是对应对象/状态的细胞数；人体供者数另报，细胞数不是独立供者重复。
- Figure 5：A/B 与 C/D 分别是不同可评估集合，按母表确认 160/257 与 122/256；绿色是同向、灰色条的剩余部分是**已评估非同向**，不是未评估。E 是相同单位的独立 network-row 摘要，非逐级存活或完整机制链。F 标题为候选证据可用性，灰/白/橙/绿四种状态及每列计数见审计表；G 是互斥的规则化探索分组，不是验证等级。小鼠为 1 young/1 aged，禁止动物层推断。
- S06/S07/S08：固定预测-标签置换不等于全训练管线的模型显著性；S07 的零特异度用原始正确阴性数/全部阴性数呈现；S08b 的 global-token 图仅为泄漏诊断，不与严格 OOF 当作等价模型比较。IG ≥5/18 只是描述性展示阈值。

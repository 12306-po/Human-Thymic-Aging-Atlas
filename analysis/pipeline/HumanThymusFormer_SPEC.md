# HumanThymusFormer — 复现规格说明书（Model Specification）

> 本文档为投稿与复现的唯一权威配置声明（2026-10-03 冻结）。
> 所有取值逐项核验自以下已冻结脚本，与代码实现一一对应，不虚构任何超参：
> - `analysis/pipeline/13_build_HumanThymusFormer_dataset.py`
> - `analysis/pipeline/14_train_HumanThymusFormer.py`
> - `analysis/pipeline/14b_strict_oof_HumanThymusFormer.py`
> - `analysis/pipeline/15_interpret_HumanThymusFormer.py`
>
> 复现环境依赖见仓库根目录 `requirements-dl.txt`。

---

## 0. 一页速览（投稿引用表）

| 项目 | 取值 | 出处（代码位置） |
|---|---|---|
| 模型全称 | HumanThymusFormer（基因×细胞类型 token Transformer） | 13/14/14b/15 |
| 任务 | 供者级年龄回归（连续年龄，标准化后 MSE） | 14 L106–145 |
| Token 上限 | 500（两套构造逻辑见 §3） | 13 L111；14b L55/136–145 |
| 架构 | dim=64 · 2 层 · 2 头 · FFN=128 · dropout=0.2 · GELU · post-norm | 14 RUN_CONFIGS L273–275；14b L95–107 |
| 优化器 | Adam，lr=1e-3，weight_decay=1e-4，betas=(0.9,0.999)，eps=1e-8（PyTorch 默认） | 14 L205/428；14b L200 |
| 学习率调度 | CosineAnnealingLR，T_max=n_epochs（300；final 400） | 14 L206；14b L201 |
| 随机种子 | SEED=371（torch + numpy） | 14 L85–86；14b L54–57；15 L73–74 |
| 早停 | 基于验证 RMSE（年），改善阈值 1e-4，patience=40（final 60） | 14 L157–247；14b L203–226 |
| 归一化 | 每折仅用 fit 供者（观测 token）的 mean/std；缺失 token 置 0 | 14 L164–187；14b L148–156 |
| IG 基线 | 标准化空间零向量（即训练均值表达基线） | 15 L280–296；14b L244 |
| IG 积分步数 | 最终模型 n_steps=50；折内/跨折 n_steps=20 | 15 L304/341；14b L244 |
| 归因聚合 | 见 §6（token→基因/细胞类型/基因×细胞类型，跨折 rank 一致性） | 15 L372–403；14b L282–296 |
| 分析定位 | 后验解释性归因（post-hoc），非泛化性能基准 | 13 L4–24；14 L18–22 |

---

## 1. 模型架构（Architecture）

定义在 `14_train_HumanThymusFormer.py` L116–145（`HumanThymusFormer` 类），`14b`（L94–118）与 `15`（L88–118）中为逐字相同的类定义。投稿引用时以本文档为准。

### 1.1 Token 表示

每个 token 对应一个 **基因×细胞类型**（或 whole-thymus）特征，其输入向量由五个分量的逐元素加和构成：

```
token_t = gene_emb(gene_id) + ct_emb(celltype_id) + pos_t + expr_proj(expr_t) + mask_proj(mask_t)
```

| 分量 | 实现 | 维度 |
|---|---|---|
| gene_emb | `nn.Embedding(n_genes, 64)` | 基因查表 |
| ct_emb | `nn.Embedding(n_celltypes, 64)` | 细胞类型查表 |
| pos | `nn.Parameter(randn(1, n_tokens, 64) * 0.02)` | 可学习位置嵌入（按 token 序号，非序列位置） |
| expr_proj | `nn.Linear(1, 64)` | 标准化 log-CPM 表达量投影 |
| mask_proj | `nn.Linear(1, 64)` | 观测掩码（1=观测，0=结构性缺失）投影 |

- 表达缺失（mask=0）的 token：原始值在标准化后被置 0（见 §5），并同时通过注意力掩码排除。
- Token 顺序固定为该折 token 清单顺序（跨供者共享同一套 token 顺序）。

### 1.2 Transformer 编码器

- `nn.TransformerEncoderLayer(d_model=64, nhead=2, dim_feedforward=128, dropout=0.2, batch_first=True, activation="gelu")`
- **Post-norm**（`norm_first=False`，PyTorch 默认）：自注意力作用于层输入，层归一化在各子层之后。
- `nn.TransformerEncoder(num_layers=2)`。
- 训练与解释时均传入 `src_key_padding_mask`（True=缺失 token，不参与注意力）。

### 1.3 池化与预测头

- **Masked mean pooling**：`pooled = sum_t(out_t · mask_t) / (sum_t mask_t + 1e-6)`，仅对观测 token 取均值。
- 预测头：`LayerNorm(64) → Linear(64, 1)`，输出为标准化年龄预测，反标准化后得到年龄（年）。

### 1.4 参数量与输入规模

- 输入：供者 × 500 token 的表达矩阵；18 名供者（GSE231906 主队列）。
- 基因/细胞类型词表：随 token 集确定（全局版见 13 L174–178；折内版见 14b L171–174）。

---

## 2. 训练流程（Training Protocol）

### 2.1 优化器与调度

- 优化器：**Adam**，`lr=1e-3`，`weight_decay=1e-4`。
  - `betas=(0.9, 0.999)`、`eps=1e-8` 为 PyTorch `torch.optim.Adam` 默认值（脚本未显式传入，特此声明采用默认）。
- 学习率调度：`CosineAnnealingLR(optimizer, T_max=n_epochs)`（LODO 折 n_epochs=300；最终模型 Stage 1 为 400）。
- 损失：`nn.MSELoss()`（标准化年龄空间）。

### 2.2 交叉验证结构

| 环节 | 配置 |
|---|---|
| 外层 | 留一供者（LODO），18 折；每折 test=1 供者，train=其余 17 |
| 内层验证 | 从 17 个外圈训练供者中按年龄排序取 **3 个等距供者** 作为 validation（`order[::max(1,len//3)][:3]`），其余为 fit |
| 早停 | 监控 validation RMSE（标准化空间 × y_std，单位年）；改善 ≥1e-4 视为进步；patience=40（最终模型 Stage 1 为 60，fit=15 / val=3） |
| 批次 | full-batch（batch_size = len(fit_idx)），`shuffle=True`（数据顺序依赖全局 RNG，见 §4） |
| 轮数上限 | 300（最终模型 Stage 1：400） |

### 2.3 最终解释模型（两阶段）

- Stage 1：15 fit / 3 val（年龄排序等距取 3）确定 `best_epoch`。
- Stage 2：**全部 18 供者** 从零重训固定 `best_epoch+1` 轮（无早停），scaler 在全部 18 供者上重算。
- 阶段溯源写入 checkpoint（`stage1_fit_donors` / `stage1_val_donors` / `stage2_train_donors` / `best_epoch_source`，14 L449–463）。

### 2.4 训练曲线与日志

- 每折逐 epoch 记录 `train_loss / val_mae / val_rmse`（`training_history_per_fold.csv`）。
- 折内与最终模型的 checkpoint 均内嵌该折 scaler 与供者划分（`best_model_cfgB_foldN.pt`、`final_interp_model.pt`）。

---

## 3. Token 构造逻辑（Token Construction）

仓库内存在**两套** token 构造，用途严格分离，投稿必须分别引用：

### 3.1 全局共识版（post-hoc，`Step 13`）

- 输入：`Step 12` 的 `ML_consensus_expression_features.csv`（`is_consensus==True`；仅 `feature_type ∈ {gene_celltype, whole_thymus}`）。
- 排序：按 `(n_models_selected DESC, best_median_rank_pct ASC)`，取前 **500**（`MAX_DL_TOKENS`）。
- 特征解析：以 `Step 10 feature_dictionary.csv` 为权威字典，缺失即 `KeyError`（禁止启发式猜测，13 L92–105）。
- 表达值：whole-thymus token 读 `donor_matrix/log_normalized_combined.csv.gz`；cell-type token 读 `celltype_matrix/log_normalized/{donor}_{CT}.csv.gz`；文件或基因缺失记 NaN → mask=0 → 值置 0。
- 用途：`Step 14` 训练与 `Step 15` 归因的 token 空间。
- **泄漏声明**：该 token 集由全 18 供者的全局共识生成，每个外圈 test 供者间接参与了 token 选择。因此 **Step 14 的 DL 指标为后验解释性结果，不得与 ML OOF 泛化指标等价比较**（见 `05_deep_learning/dataset/dl_scope_statement.txt`）。

### 3.2 折内严格版（strict fold-internal，`Step 14b`）

- 输入：`Step 11` 折级特征重要性（`metrics/fold_level_importance.csv`，模型取 ElasticNet ∪ LinearSVR）。
- 每折 token 集 = 该折**外圈训练供者**模型的重要性 top 500（仅基因×细胞类型与 whole-thymus 特征）。
- 排序：按 `(模型支持数 DESC, min(rank_pct) ASC)`（14b L136–145）。
- 折对齐：Step 11 的折按**字典序供者顺序**（sklearn LeaveOneGroupOut 行为），本脚本 L71–83 显式校正了旧版错位问题。
- 用途：`Step 14b` 严格折内 OOF 预测与折内 IG；held-out 供者从未参与 token 选择、缩放或调参。

> 论文正文"strict fold-internal HumanThymusFormer R²=0.13 / MAE=15.0 年"对应 **3.2 折内版**；"post-hoc 拟合产生的高表观 R²"对应 **3.1 全局版泄漏诊断**（14 L63–65）。

---

## 4. 随机种子（Random Seeds）

- **SEED = 371**，全流程统一：
  - `Step 14`：脚本开头 `torch.manual_seed(371); np.random.seed(371)`（L85–86）。
  - `Step 14b`：SEED 常量（L54），且**每折训练前再次 `torch.manual_seed(SEED)`**（L198），保证折间独立可复现。
  - `Step 15`：脚本开头同 Step 14（L73–74）。
  - `14b` bootstrap：`np.random.default_rng(SEED)`（L267），2,000 次供者重采样。
- 作用域声明：`DataLoader(shuffle=True)` 依赖全局 RNG；因此完整复现要求**按脚本顺序在单一进程中依次执行** 13 → 14 → 15（14b 独立运行亦可复现，因每折重置种子）。
- 未引入 CUDA 确定性设置（`deterministic/cudnn.benchmark`）；CPU 推理完全确定，GPU 数值顺序差异不影响定性结论。

---

## 5. 数据归一化（Normalization）

- 每折（或最终模型每阶段）独立计算 scaler，**仅用 fit 供者**：
  - `token_mean/std`：逐 token 在 `mask==1` 的观测值上计算；std<1e-6 置 1.0；该 token 在 fit 供者中无观测则保持 mean=0/std=1（14 L164–181；14b L148–156）。
  - `y_mean/y_std`：fit 供者年龄的均值/标准差（std=0 时置 1.0）。
- 标准化后，**结构性缺失 token（mask=0）强制置 0**，与训练/归因输入完全一致（14 L187；14b L186；15 L151–163）。
- 归因时必须使用 checkpoint 内嵌的该折 scaler，禁止用全 18 供者重算（15 L9–12）。

---

## 6. 可解释性归因（Interpretation Protocol）

### 6.1 注意力归因（`Step 15`）

- 仅取最后一层 `self_attn`：`need_weights=True, average_attn_weights=False` → 每头 (B, T, T)。
- 复现 post-norm 路径：自注意力输入为**层原始输入**（非 norm1 后），与训练前向一致（15 L166–206）。
- 掩码：注意力矩阵按 **query 与 key 双侧有效掩码**（`valid_pair = valid_q & valid_k`）清零缺失位置，再做 per-query 重归一化。
- token 重要性 = 在 (heads, queries) 两轴上的均值。

### 6.2 Integrated Gradients（IG）

- 实现：captum `IntegratedGradients`，包一层 `AttrModel`（仅对表达通道插值，mask/ID/掩码固定）。
- **基线**：标准化空间零向量（即"训练均值表达基线"，15 L280–296）。
- **积分步数**：最终模型 `n_steps=50`（15 L304）；跨折一致性 `n_steps=20`（15 L341）；严格折内 `n_steps=20`（14b L244）。
- 逐供者独立计算（每供者构建自己的 mask 上下文）；`return_convergence_delta=True` 记录收敛性。
- 数值口径：保留有符号 IG（乘该折 `y_std` 得"年"单位，供生物学方向解释），并同时输出绝对值与排名（14b L247–257）。

### 6.3 聚合方式（Aggregation）

| 层级 | 规则 | 输出文件 |
|---|---|---|
| token（最终模型） | 注意力均值 / IG 绝对值跨供者均值 → 排名 | `final_attention_token_importance.csv`、`final_IG_token_importance.csv` |
| 基因 / 细胞类型 / 基因×细胞类型 | 对 token 层均值聚合（mean/max/n_tokens） | `gene_importance.csv`、`celltype_importance.csv`、`gene_celltype_importance.csv` |
| 跨折一致性（18 折） | 每折内 rank（`method=first`）→ rank_pct；按特征聚合：median/min rank_pct、n_folds_in_top50、mean_importance | `IG_across_fold_consistency.csv`（15）；`oof_IG_gene_consistency.csv`（14b） |

---

## 7. 运行顺序与环境

### 7.1 运行顺序

```
# post-hoc 解释性管线（全局共识 token）
Step 11 (折级重要性) → 12 (ML 共识) → 13 (构建 DL 数据集) → 14 (LODO 训练) → 15 (归因)
# 严格折内管线（投稿 R²=0.13 结果）
Step 11 (折级重要性) → 14b (严格折内 OOF + IG)
```

### 7.2 环境

- 依赖清单：仓库根目录 `requirements-dl.txt`（torch、captum、scikit-learn、scipy、pandas、numpy、matplotlib；与 `requirements.txt` 的 Streamlit 运行环境分离）。
- 目标环境：分析服务器（原始数据与 `Step 01–12` 中间产物在 `PROJ` 下，通过 `00_project_config.sh` 设置）。
- 确定性说明：固定 SEED=371 后，CPU 复现确定；GPU 下损失曲线可能有数值级微小差异，不影响定性与排名结论。

---

## 8. 投稿时的引用写法（Suggested Reporting）

建议在 Methods / Data availability 中引用本文档与脚本编号：

> "HumanThymusFormer hyperparameters, token-construction rules, normalization, Integrated Gradients baseline and integration steps, and aggregation procedures are fully specified in `analysis/pipeline/HumanThymusFormer_SPEC.md` (repository: github.com/12306-po/Human-Thymic-Aging-Atlas) and implemented in pipeline Steps 13, 14, 14b and 15 (SEED=371)."

---

## 9. 已知未纳入本规格的项（透明声明）

- CUDA 确定性配置（`torch.backends.cudnn`）未启用：数值结果在 GPU 间可能微小差异（不影响结论）。
- `Step 11` 折级重要性的特征排序细节（ElasticNet/LinearSVR 各自参数）属于机器学习管线规范，不在此处重复，见 `11_nested_donor_ML.py` 与 `HumanThymusFormer_SPEC` 之外的管线文档。

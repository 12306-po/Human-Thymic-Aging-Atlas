# 2026-09-21 版式修订与 Figure 4 扩展

本轮仅修改绘图层（`figure_common.py`、`fig_paper/fig1.py`–`fig5.py`、来源清单、图注说明和测试），不更改 00–24 的分析结果。

- Figure 1：独立横排 QC 坐标；UMAP 保持等比例；图例使用预留区域。
- Figure 2：密集的细胞类型统计、热图、组成相关和通路图各占整行；颜色、面积图例有独立区域；前六个 q 排名基因另存 `fig2a_top_six_genes_audit.tsv`。
- Figure 3：预测散点等比例；模型比较、消融、置换和残差分行；流程箭头方向修正。
- Figure 4：由两分面扩展为五分面。A 为外部发育状态的细胞/供者覆盖度；B/C 为 Sort1/Sort2 定位；D/E 是**同一对象内**较晚状态减较早状态的均值 z-score 差值绝对值前八个基因。D/E 为描述性展示，没有显著性检验，不是增龄验证。新增 `fig4_state_coverage_audit.tsv` 与 `fig4_stage_delta_audit.tsv`。
- Figure 5：按 A–G 顺序排版，长轴标签缩短，证据矩阵独占整行。小鼠每年龄组 1 只的证据边界不变。

复合图导出固定物理画布，防止外置图例触发 `bbox_inches='tight'` 自动扩张尺寸。改版后仍须用服务器上的真实冻结结果重新运行绘图入口并逐张检查 PDF/PNG；本地只重画了 Figure 4 的旧审计表预览，不能代替真实项目重绘。

将本目录修改过的文件同步到服务器的 `/data/zxy/projects/human_thymus_age_ML_DL/figures-0920/`（尤其是 `figure_common.py`、整个 `fig_paper/`、`99b_source_manifest.py`、`figure_legends_revision_notes.md`）。Windows 本地修改不会自动更新服务器文件。随后调用 `99_make_paper_figures.py`，并指定新的 `--output` 目录，不能覆盖原图。

本地最小验证：

```bash
python -B -m unittest discover -s tests -v
```

After the server redraw, run `python 100_check_paper_figures.py --output /path/to/new/figures` to check the physical canvas and required exports. This mechanical check does not detect every label collision; inspect the PDFs at journal print size.

## Figure font

All main and supplementary figure scripts inherit **Times New Roman** from
`figure_common.py`, including Matplotlib math text. The configuration is
fail-closed: figure generation stops instead of silently substituting DejaVu
when Times New Roman is unavailable. On a server where the authorised regular,
bold and italic font files are stored in a directory but are not installed
system-wide, set:

```bash
export PAPER_FONT_PATH=/path/to/times-new-roman-font-files
```

PDF and PostScript output uses embedded TrueType fonts (`fonttype = 42`).

投稿前还必须核对图内数值、审计表、目标期刊尺寸与图注；排版通过不等于科学结果已被验证。

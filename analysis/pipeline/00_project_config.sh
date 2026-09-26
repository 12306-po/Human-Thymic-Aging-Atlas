#!/bin/bash
# 00_project_config.sh — 统一路径配置（export/env 方式）
# 用法：只在 shell 层 source 一次（source scripts/00_project_config.sh），
#       随后逐脚本运行 python/Rscript；Python/R 脚本内部【不】source 本文件，
#       分别用 os.environ / Sys.getenv 读取（未设置时 fail-fast）。

# --- raw source 路径（只读，绝不修改） ---
export HUMAN_PRIMARY_RAW=/data/zxy/raw_data/GSE231906
export HUMAN_EXTERNAL_RAW=/data/zxy/raw_data/human_thymus/external
export MOUSE_SCRNA_RAW=/data/zxy/raw_data/mouse_scRNA
export MOUSE_SCATAC_RAW=/data/zxy/raw_data/mouse_scATAC
export MOUSE_DERIVED=/data/zxy/projects/mouse_thymus/mouse_thymus_atlas

# --- 项目根 ---
export PROJ=/data/zxy/projects/human_thymus_age_ML_DL

# 环境变量未设置时的 fail-fast 检查
for var in HUMAN_PRIMARY_RAW HUMAN_EXTERNAL_RAW MOUSE_SCRNA_RAW MOUSE_SCATAC_RAW MOUSE_DERIVED PROJ; do
    if [ -z "${!var}" ]; then
        echo "ERROR: required env var ${var} is not set" >&2
        exit 1
    fi
done

echo "Project: ${PROJ}"
echo "Raw data: ${HUMAN_PRIMARY_RAW}"
echo "Ready"
#!/usr/bin/env bash
# --- path config: override with env vars (defaults = original experiment machine) ---
: "${OPENPI_ROOT:=/root/gpufree-data/openpi}"
: "${RBM_ASSETS_ROOT:=/root/gpufree-data/rbm_eval_assets}"
: "${SHARE_ROOT:=/root/gpufree-share}"
: "${AUTODL_ROOT:=/root/autodl-tmp}"
# ============================================================
# OpenPI 环境变量配置（第 3.4 节：把所有缓存/数据/权重放到大容量磁盘）
# 用法：source /root/gpufree-data/openpi/my_env.sh
# ============================================================

# 本文件有两个使用位置，PROJECT_ROOT 按位置自动判断：
#   1) 复制成 $OPENPI_ROOT/my_env.sh 后 source → 所在目录就是项目根
#   2) 直接从 repo/scripts_local/ 执行 → 回退到文件顶部定义的 OPENPI_ROOT
_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ "$(basename "$_HERE")" = "scripts_local" ]; then
    export PROJECT_ROOT="$OPENPI_ROOT"
else
    export PROJECT_ROOT="$_HERE"
fi
export CACHE_DIR="$PROJECT_ROOT/.cache"

# ---------- 依赖下载缓存 ----------
export HF_HOME="$CACHE_DIR/huggingface"          # HF 数据集 / 模型权重
# lerobot 0.1.0 的默认数据集根目录是 HF_LEROBOT_HOME/<repo_id>，
# 与 HF Hub 的 hub/datasets--xxx 布局不同，必须显式对齐否则训练会重复下载
export HF_LEROBOT_HOME="$HF_HOME/lerobot"
export PIP_CACHE_DIR="$CACHE_DIR/pip"
export UV_CACHE_DIR="$CACHE_DIR/uv"

# ---------- OpenPI 权重缓存 ----------
export OPENPI_DATA_HOME="$CACHE_DIR/openpi"      # pi05_base 的 GCS 重定向落点

# ---------- Libero 仿真 ----------
# libero 只认 $LIBERO_CONFIG_PATH/config.yaml（默认 ~/.libero），在 import 时会交互式索要路径。
# 注意：网上流传的 LIBERO_DATASET_PATH 环境变量在 libero 包里并不存在，无法跳过提示。
# 这里把配置放在工作区里（config.yaml 已预生成），避免写入系统盘和交互等待。
export LIBERO_CONFIG_PATH="$CACHE_DIR/libero"

# ---------- HuggingFace 镜像 ----------
# 本机 huggingface.co 直连超时，hf-mirror.com 可达（已实测 HTTP 200）
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DOWNLOAD_TIMEOUT=60

# ---------- Git 修复 ----------
# 本机 git 不支持 openssl 后端，且 hub.fastgit.org 不可达；
# 用工作区内的 gitconfig 覆盖全局配置（不改动 ~/.gitconfig）
if [ -f "$CACHE_DIR/gitconfig" ]; then
    export GIT_CONFIG_GLOBAL="$CACHE_DIR/gitconfig"
fi

# ---------- WandB（可选，参考第 10.6 节）----------
# export WANDB_API_KEY=你的API密钥

# 注意：不能含未定义变量，否则在 `set -u` 的脚本里会直接报错
# （历史上这里残留过 $LIBERO_DATASET_PATH，导致每次 source 都打印 `mkdir: 无法创建目录 ""`）
mkdir -p "$HF_HOME" "$PIP_CACHE_DIR" "$UV_CACHE_DIR" "$OPENPI_DATA_HOME" "$LIBERO_CONFIG_PATH"

echo "✅ OpenPI 环境变量已就绪"
echo "   PROJECT_ROOT = $PROJECT_ROOT"
echo "   CACHE_DIR    = $CACHE_DIR"
echo "   HF_ENDPOINT  = $HF_ENDPOINT"

# ---- 防止系统盘被缓存写满（2026-11-07 崩溃后加）----
# 所有 HF / 临时缓存统一放到数据盘或 JuiceFS，不写 /root/.cache
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${SHARE_ROOT}/hf_datasets_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TMPDIR="${TMPDIR:-${SHARE_ROOT}/tmp}"
mkdir -p "$HF_DATASETS_CACHE" "$HF_HUB_CACHE" "$TMPDIR" 2>/dev/null || true

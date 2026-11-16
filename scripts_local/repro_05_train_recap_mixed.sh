#!/usr/bin/env bash
# --- path config: override with env vars (defaults = original experiment machine) ---
: "${OPENPI_ROOT:=/root/gpufree-data/openpi}"
: "${RBM_ASSETS_ROOT:=/root/gpufree-data/rbm_eval_assets}"
: "${SHARE_ROOT:=/root/gpufree-share}"
: "${AUTODL_ROOT:=/root/autodl-tmp}"
# mixed RECAP-lite: 90% SFT demo + 10% in-domain rollout（3 个 shard）
set -uo pipefail
RPO=${OPENPI_ROOT}/robobenchmart-openpi-rlinf-ppo-main
CKPT=${MODEL_PATH:-${OPENPI_ROOT}/pytorch_ckpts/pi05_rbm_official_pt}
SFT_DS=${SFT_DS:-${OPENPI_ROOT}/.cache/huggingface/lerobot/rbm_dataset}
ROLL_DS=${ROLL_DS:-${OPENPI_ROOT}/rollouts}
TAG=${TAG:-rbm_recap_mixed}
cd "$RPO"
if [ -f "${OPENPI_ROOT}/my_env.sh" ]; then
  source "${OPENPI_ROOT}/my_env.sh" >/dev/null 2>&1
else
  echo "[warn] 未找到 ${OPENPI_ROOT}/my_env.sh，环境变量需自行设置" >&2
  echo "       可执行: cp scripts_local/repro_00_env.sh ${OPENPI_ROOT}/my_env.sh" >&2
fi
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl ROBOT_PLATFORM=LIBERO
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=1
export PYTHONPATH="$RPO:${OPENPI_ROOT}/.cache/rbm/repo:${OPENPI_ROOT}/src:${PYTHONPATH:-}"
export PATH="$RPO/.venv_rlinf/bin:$PATH"
export REPO_PATH="$RPO"
export PROJECT="$RPO"
export SFT_DATASET="$SFT_DS" ROLLOUT_DATASET="$ROLL_DS" TAG
export MAX_STEPS=${STEPS:-150} GLOBAL_BATCH_SIZE=${GLOBAL:-4}
export SFT_WEIGHT=${SFT_WEIGHT:-0.9} ROLLOUT_WEIGHT=${ROLLOUT_WEIGHT:-0.1}
export POSITIVE_FRACTION=${POSITIVE_FRACTION:-0.67} QUOTA_CYCLE_SIZE=${CYCLE:-60}
export MIN_EPISODES_PER_SHARD=${MIN_EP:-10}
export MODEL_PATH="$CKPT"
bash scripts_local/11_rbm_recap_cfg_mixed.sh \
  actor.model.model_path="$CKPT" \
  actor.micro_batch_size=1 \
  data.num_workers=0 \
  runner.save_interval=100 \
  "$@"

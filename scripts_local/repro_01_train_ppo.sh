#!/usr/bin/env bash
# --- path config: override with env vars (defaults = original experiment machine) ---
: "${OPENPI_ROOT:=/root/gpufree-data/openpi}"
: "${RBM_ASSETS_ROOT:=/root/gpufree-data/rbm_eval_assets}"
: "${SHARE_ROOT:=/root/gpufree-share}"
: "${AUTODL_ROOT:=/root/autodl-tmp}"
set -uo pipefail
RPO=${OPENPI_ROOT}/robobenchmart-openpi-rlinf-ppo-main
RBM=${OPENPI_ROOT}/.cache/rbm/repo
CKPT=${MODEL_PATH:-${OPENPI_ROOT}/pytorch_ckpts/pi05_rbm_official_pt}
cd "$RPO"
if [ -f "${OPENPI_ROOT}/my_env.sh" ]; then
  source "${OPENPI_ROOT}/my_env.sh" >/dev/null 2>&1
else
  echo "[warn] 未找到 ${OPENPI_ROOT}/my_env.sh，环境变量需自行设置" >&2
  echo "       可执行: cp scripts_local/repro_00_env.sh ${OPENPI_ROOT}/my_env.sh" >&2
fi
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
export MS_ASSET_DIR=${RBM_ASSETS_ROOT}/ms_assets
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl ROBOT_PLATFORM=LIBERO
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1
export HF_DATASETS_CACHE=${SHARE_ROOT}/hf_datasets_cache
export RBM_INIT_STATE_DIR=${OPENPI_ROOT}/.cache/rbm/eval_states/fanta
export PYTHONPATH="$RPO:$RBM:${OPENPI_ROOT}/src:${PYTHONPATH:-}"
export PATH="$RPO/.venv_rlinf/bin:$PATH"
bash examples/embodiment/run_embodiment.sh rbm_ppo_local LIBERO \
  actor.model.model_path="$CKPT" \
  rollout.model.model_path="$CKPT" \
  env.train.project_path="$RBM" \
  env.eval.project_path="$RBM" \
  runner.max_epochs="${MAX_EPOCHS:-1}" \
  runner.save_interval=${SAVE_EVERY:-5} \
  runner.val_check_interval=-1 \
  runner.logger.experiment_name="${EXP:-rbm_ppo_smoke_local}" \
  actor.optim.critic_warmup_steps="${WARMUP:-12}" \
  "$@"

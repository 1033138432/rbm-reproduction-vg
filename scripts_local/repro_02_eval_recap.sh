#!/usr/bin/env bash
# --- path config: override with env vars (defaults = original experiment machine) ---
: "${OPENPI_ROOT:=/root/gpufree-data/openpi}"
: "${RBM_ASSETS_ROOT:=/root/gpufree-data/rbm_eval_assets}"
: "${SHARE_ROOT:=/root/gpufree-share}"
: "${AUTODL_ROOT:=/root/autodl-tmp}"
# 用 RLinf 匹配协议评测 mixed RECAP checkpoint（3 ID × 30 episodes）
set -uo pipefail
RPO=${OPENPI_ROOT}/robobenchmart-openpi-rlinf-ppo-main
RBM=${OPENPI_ROOT}/.cache/rbm/repo
CKPT_DIR=${CKPT_DIR:?需要 CKPT_DIR}
BASE=${MODEL_PATH:-${OPENPI_ROOT}/pytorch_ckpts/pi05_rbm_official_pt}
FULL=$CKPT_DIR/actor/model_state_dict/full_weights.pt
# HEADLESS=1 用于无 value/noise head 的 RECAP 断点；PPO 断点保持默认（带 head）
if [ "${HEADLESS:-0}" = "1" ]; then
  HEAD_ARGS="actor.model.add_value_head=False actor.model.openpi.add_value_head=False actor.model.openpi.noise_method=flow_ode"
else
  HEAD_ARGS=""
fi
[ -f "$FULL" ] || { echo "找不到 $FULL"; exit 1; }
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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=1
export PYTHONPATH="$RPO:$RBM:${OPENPI_ROOT}/src:${PYTHONPATH:-}"
export PATH="$RPO/.venv_rlinf/bin:$PATH"
bash examples/embodiment/run_embodiment.sh rbm_ppo_local LIBERO \
  runner.only_eval=True runner.max_epochs=1 runner.val_check_interval=-1 \
  runner.ckpt_path="$FULL" \
  ${HEAD_ARGS:-} \
  actor.model.model_path="$BASE" rollout.model.model_path="$BASE" \
  actor.enable_sft_co_train=False \
  env.eval.project_path="$RBM" \
  env.eval.total_num_envs=3 \
  env.eval.max_steps_per_rollout_epoch=600 env.eval.max_episode_steps=600 \
  env.eval.auto_reset=False env.eval.ignore_terminations=False \
  env.eval.init_params.sim_backend=cpu \
  algorithm.eval_rollout_epoch=30 \
  runner.logger.experiment_name=eval_recap_mixed "$@"

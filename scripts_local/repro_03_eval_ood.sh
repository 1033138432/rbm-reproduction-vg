#!/usr/bin/env bash
# --- path config: override with env vars (defaults = original experiment machine) ---
: "${OPENPI_ROOT:=/root/gpufree-data/openpi}"
: "${RBM_ASSETS_ROOT:=/root/gpufree-data/rbm_eval_assets}"
: "${SHARE_ROOT:=/root/gpufree-share}"
: "${AUTODL_ROOT:=/root/autodl-tmp}"
# OOD 评测：Nestle + Slam，各 30 episodes，seed 42000-42029
set -uo pipefail
RPO=${OPENPI_ROOT}/robobenchmart-openpi-rlinf-ppo-main
RBM=${OPENPI_ROOT}/.cache/rbm/repo
CKPT_DIR=${CKPT_DIR:-}
BASE=${BASE:-${OPENPI_ROOT}/pytorch_ckpts/pi05_rbm_official_pt}
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
HEAD_ARGS=""
[ "${HEADLESS:-0}" = "1" ] && HEAD_ARGS="actor.model.add_value_head=False actor.model.openpi.add_value_head=False actor.model.openpi.noise_method=flow_ode"
CKPT_ARGS=""
if [ -n "$CKPT_DIR" ]; then CKPT_ARGS="runner.ckpt_path=$CKPT_DIR/actor/model_state_dict/full_weights.pt"; fi
bash examples/embodiment/run_embodiment.sh rbm_ppo_local LIBERO \
  runner.only_eval=True runner.max_epochs=1 runner.val_check_interval=-1 \
  $CKPT_ARGS $HEAD_ARGS \
  actor.model.model_path="$BASE" rollout.model.model_path="$BASE" \
  actor.enable_sft_co_train=False \
  env.eval.project_path="$RBM" \
  'env.eval.init_params.id=[PickToBasketContNestleEnv,PickToBasketContSlamEnv]' \
  "env.eval.init_params.config_dir_path=${OPENPI_ROOT}/.cache/rbm/demo_envs/test_unseen_items_pick_to_basket" \
  "env.eval.init_params.init_state_dir=${OPENPI_ROOT}/.cache/rbm/eval_states/ood" \
  'env.eval.episode_seed_lists=null' \
  'env.eval.episode_seed_start=42000' \
  env.eval.total_num_envs=2 \
  rollout.pipeline_stage_num=2 \
  env.eval.max_steps_per_rollout_epoch=600 env.eval.max_episode_steps=600 \
  env.eval.auto_reset=False env.eval.ignore_terminations=False \
  env.eval.init_params.sim_backend=cpu \
  algorithm.eval_rollout_epoch=30 \
  runner.logger.experiment_name=eval_ood "$@"

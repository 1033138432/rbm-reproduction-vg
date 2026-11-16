#!/bin/bash
# --- path config: override with env vars (defaults = original experiment machine) ---
: "${OPENPI_ROOT:=/root/gpufree-data/openpi}"
: "${RBM_ASSETS_ROOT:=/root/gpufree-data/rbm_eval_assets}"
: "${SHARE_ROOT:=/root/gpufree-share}"
: "${AUTODL_ROOT:=/root/autodl-tmp}"
# RoboBenchMart Stage-2: Train3 KL-PPO small-run probe (1 epoch)
# Adapted from docs/rbm_stage1_rlinf_ppo_smoke_adaptation.md section 15.1
# to the current machine paths (rbm_ppo_local config, verified working 2026-11-07 15:04 eval-only run).
set -x
cd ${OPENPI_ROOT}/robobenchmart-openpi-rlinf-ppo-main

source .venv_rlinf/bin/activate
export EMBODIED_PATH="$PWD/examples/embodiment"
export REPO_PATH="$PWD"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTHONPATH="$REPO_PATH:${PYTHONPATH:-}"

MODEL_PATH=${OPENPI_ROOT}/pytorch_ckpts/pi05_rbm_official_pt
RBMCACHE=${OPENPI_ROOT}/.cache/rbm/repo
TS=$(date +%Y%m%d-%H:%M:%S)
LOGDIR="$PWD/logs/${TS}-rbm_ppo_local"
mkdir -p "$LOGDIR"
echo "$LOGDIR" > /tmp/rbm_stage2_logdir

exec python examples/embodiment/train_embodied_agent.py \
  --config-path "$PWD/examples/embodiment/config/" \
  --config-name rbm_ppo_local \
  runner.logger.log_path="$LOGDIR" \
  runner.logger.experiment_name=rbm_ppo_train3_probe \
  actor.model.model_path="$MODEL_PATH" \
  rollout.model.model_path="$MODEL_PATH" \
  env.train.project_path="$RBMCACHE" \
  env.eval.project_path="$RBMCACHE" \
  runner.max_epochs=2 \
  runner.save_interval=1 \
  runner.val_check_interval=-1 \
  env.train.total_num_envs=12 \
  env.train.max_steps_per_rollout_epoch=300 \
  env.eval.total_num_envs=3 \
  env.eval.max_steps_per_rollout_epoch=600 \
  algorithm.update_epoch=1 \
  algorithm.kl_beta=0.001 \
  actor.optim.lr=1.0e-6 \
  actor.optim.value_lr=5.0e-5

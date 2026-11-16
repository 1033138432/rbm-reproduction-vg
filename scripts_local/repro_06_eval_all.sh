#!/usr/bin/env bash
# --- path config: override with env vars (defaults = original experiment machine) ---
: "${OPENPI_ROOT:=/root/gpufree-data/openpi}"
: "${RBM_ASSETS_ROOT:=/root/gpufree-data/rbm_eval_assets}"
: "${SHARE_ROOT:=/root/gpufree-share}"
: "${AUTODL_ROOT:=/root/autodl-tmp}"
# 复现官方评测协议：3 个 ID 任务 + 2 个 OOD 任务，每任务 30 episodes
# 用法: bash eval_all.sh <checkpoint_dir> [port]
set -euo pipefail
CKPT="${1:?用法: bash eval_all.sh <checkpoint_dir> [port]}"
PORT="${2:-8200}"
N=30
OPENPI=${OPENPI_ROOT}
RBM=$OPENPI/.cache/rbm
EVAL=$RBM/eval_states
DSPY=$RBM/.dsynth-env/bin/python
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
export PYTHONPATH=$RBM/repo
export MS_ASSET_DIR=${RBM_ASSETS_ROOT}/ms_assets

cd "$OPENPI"
setsid nohup env CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 uv run scripts/serve_policy.py \
  --port "$PORT" policy:checkpoint --policy.config pi05_eval_rbm --policy.dir "$CKPT" \
  > /tmp/serve_eval_repro.log 2>&1 &
echo "等待策略服务..."
for i in $(seq 1 90); do grep -q "server listening" /tmp/serve_eval_repro.log 2>/dev/null && break; sleep 2; done
grep -q "server listening" /tmp/serve_eval_repro.log && echo "服务已就绪" || { echo "服务启动失败"; tail -20 /tmp/serve_eval_repro.log; exit 1; }

MP=$RBM/demo_envs_mp/pick_to_basket/demos/motionplanning
run() {
  local env_id=$1 sub=$2 scene=$3 init=$4; shift 4
  echo "===== $sub ($env_id) ====="
  ( cd "$RBM" && "$DSPY" repo/scripts/eval_policy_client.py -e "$env_id" --scene-dir "$scene" \
      --host 127.0.0.1 --port "$PORT" --num-traj "$N" --max-horizon 600 \
      --exec-horizon 10 --clip-actions --init-state-dir "$init" --save-video \
      --eval-subdir "repro_$sub" "$@" ) 2>&1 | grep -E "Results|success_rate" | tail -2
}

run PickToBasketContFantaEnv fanta_id "$RBM/demo_envs/pick_to_basket" \
    "$EVAL/fanta" --json-path "$MP/pick_to_basket_fanta_248traj_4workers.rgbd.pd_joint_pos.physx_cpu.json"
run PickToBasketContNiveaEnv nivea_id "$RBM/demo_envs/pick_to_basket" \
    "$EVAL/nivea" --json-path "$MP/pick_to_basket_nivea_248traj_4workers.rgbd.pd_joint_pos.physx_cpu.json"
run PickToBasketContStarsEnv stars_id "$RBM/demo_envs/pick_to_basket" \
    "$EVAL/stars" --json-path "$MP/pick_to_basket_stars_248traj_4workers.rgbd.pd_joint_pos.physx_cpu.json"
run PickToBasketContNestleEnv nestle_ood "$RBM/demo_envs/test_unseen_items_pick_to_basket" \
    "$EVAL/ood" --start-seed 42000
run PickToBasketContSlamEnv slam_ood "$RBM/demo_envs/test_unseen_items_pick_to_basket" \
    "$EVAL/ood" --start-seed 42000

for p in $(ps -eo pid,args | awk '/serve_policy/ && !/awk/ {print $1}'); do kill -9 "$p" 2>/dev/null || true; done
echo "评测结束"

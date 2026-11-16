#!/usr/bin/env bash
# --- path config: override with env vars (defaults = original experiment machine) ---
: "${OPENPI_ROOT:=/root/gpufree-data/openpi}"
: "${RBM_ASSETS_ROOT:=/root/gpufree-data/rbm_eval_assets}"
: "${SHARE_ROOT:=/root/gpufree-share}"
: "${AUTODL_ROOT:=/root/autodl-tmp}"
# 用 openpi 服务 + 评测客户端录视频（不依赖 Ray）
# 用法: bash rec_openpi.sh <tag> <port> <ckpt_dir> <eval_subdir>
set -uo pipefail
M=${1:?}; PORT=${2:?}; DIR=${3:?}; SUB=${4:?}
O=${OPENPI_ROOT}
WS=${OPENPI_ROOT}/.cache/rbm            # 工作目录（含 demo_envs / eval_states）
REPO=$WS/repo                                      # 代码目录
PY=$WS/.dsynth-env/bin/python                      # 仿真解释器
MP=$WS/demo_envs_mp/pick_to_basket/demos/motionplanning
cd $O
if [ -f my_env.sh ]; then
  source my_env.sh >/dev/null 2>&1
else
  echo "[warn] 未找到 $O/my_env.sh，环境变量需自行设置" >&2
  echo "       可执行: cp scripts_local/repro_00_env.sh $O/my_env.sh" >&2
fi
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
export MS_ASSET_DIR=${RBM_ASSETS_ROOT}/ms_assets
export PYTHONPATH=$REPO
CUDA_VISIBLE_DEVICES=0 setsid nohup uv run scripts/serve_policy.py --port $PORT policy:checkpoint \
  --policy.config pi05_eval_rbm --policy.dir "$DIR" > $O/logs/serve_$M.log 2>&1 &
for i in $(seq 1 120); do grep -q "listening" $O/logs/serve_$M.log 2>/dev/null && break; sleep 2; done
cd $WS
$PY $REPO/scripts/eval_policy_client.py -e PickToBasketContFantaEnv \
  --scene-dir $WS/demo_envs/pick_to_basket --host 127.0.0.1 --port $PORT \
  --num-traj 10 --max-horizon 600 --exec-horizon 10 --clip-actions \
  --init-state-dir $WS/eval_states/fanta \
  --json-path $MP/pick_to_basket_fanta_248traj_4workers.rgbd.pd_joint_pos.physx_cpu.json \
  --save-video --eval-subdir "$SUB" > $O/logs/rec_$M.log 2>&1
grep -oE "^Results: \[[^]]*\]" $O/logs/rec_$M.log | tail -1
for p in $(ps -eo pid,args | awk '/serve_policy/ && !/awk/ {print $1}'); do kill -9 $p 2>/dev/null; done

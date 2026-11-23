# RoboBenchMart × π0.5 后训练项目：从零复现操作指南

> 目标读者：未来的你，或任何拿到一台 46 GB 单卡机器、想从零重做这套流程的人。
> 本文所有命令和改动都在 L40 46 GB + 128 核 + 197 GB 数据盘上实测通过。
>
> **路径说明**：文中的 `/root/gpufree-data`、`/root/autodl-tmp`、`/root/gpufree-share` 是原实验机器的
> 路径，按实跑记录原样保留。换机器时不用改脚本，按 README 的「路径配置」用环境变量覆盖即可；
> 只有 3 处 YAML 必须手改（README 里有表）。
>
> 阅读顺序：先看 §0 的硬件门槛，再按 §1→§11 顺序执行。§12 是每步的验收标准，§13 是坑清单索引。

---

## 0. 前置条件

| 项 | 要求 | 检查命令 |
|---|---|---|
| GPU | ≥ 46 GB（LoRA / expert-only 够；全参需 95 GB）| `nvidia-smi` |
| 磁盘 | 数据盘 ≥ 120 GB 可用 | `df -h` |
| 内存 | cgroup 限制 ≥ 100 GB | `cat /sys/fs/cgroup/memory.max` |
| 网络 | 能访问 hf-mirror.com | `curl -I https://hf-mirror.com` |

**磁盘预算**（必须提前算）：

```
h5 数据（一次一个）    27 GB
LeRobot 数据集         26 GB
JAX SFT 断点           15 GB / 个
PyTorch 权重           6.8 GB / 个
RLinf venv             9.3 GB
PPO 断点               19 GB / 个
Arrow 缓存             25 GB（训练时临时，指向 JuiceFS）
```

---

## 1. 目录规划

```bash
export WORK=/root/gpufree-data/openpi          # 数据盘
export RBM=$WORK/.cache/rbm                    # RoboBenchMart 工作目录
export SHARE=/root/gpufree-share               # JuiceFS，放大文件
```

最终结构：

```
$WORK/
├── src/openpi/                openpi（已打 RBM 补丁）
├── checkpoints/               JAX 训练断点
├── pytorch_ckpts/             JAX→PyTorch 转换结果
├── .cache/huggingface/
│   ├── rbm_pi05_official/     官方 JAX 权重 + norm_stats
│   └── lerobot/rbm_dataset/   744 条 LeRobot 数据集
├── .cache/rbm/
│   ├── repo/                  RoboBenchMart 代码
│   ├── .dsynth-env/           仿真 Python 环境（Python 3.10）
│   ├── demo_envs*/            场景配置
│   └── eval_states/           评测用初始姿态（导出物）
└── robobenchmart-openpi-rlinf-ppo-main/   RLinf（PPO/RECAP）
    └── .venv_rlinf/           训练环境（Python 3.11）
```

**注意**：仿真用 Python 3.10（`.dsynth-env`），训练用 Python 3.11（`.venv_rlinf`），**不能合并**。

---

## 2. 环境搭建（4 步，约 3 小时）

### 2.1 openpi（Python 3.11）

```bash
cd $WORK
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -e .
```

### 2.2 仿真环境（Python 3.10）

```bash
cd $RBM
uv venv .dsynth-env --python 3.10
source .dsynth-env/bin/activate
uv pip install torch mani_skill==3.0.1 sapien==3.0.3 \
  scene-synthesizer==1.15.0 trimesh==4.5.3 mujoco \
  h5py zarr gymnasium opencv-python imageio
```

> **版本号说明**：`torch` 与 `mujoco` 未锁死小版本，请按本机 CUDA 版本安装后用
> `pip freeze > requirements-dsynth.txt` 固定环境。注意 `.dsynth-env`（Python 3.10）
> 与 §2.3 的 RLinf 训练环境（`.venv_rlinf`，Python 3.11）为两套独立环境，切勿混用依赖版本。

### 2.3 RLinf（Python 3.11）

**坑 ①：RLinf 要求 Python ≤3.11.14，本机只有 3.11.18**

```bash
cd $WORK/robobenchmart-openpi-rlinf-ppo-main
# 改 pyproject.toml 第 11 行
sed -i 's/requires-python = ">=3.10,<=3.11.14"/requires-python = ">=3.10,<=3.11.18"/' pyproject.toml
```

**坑 ②：uv 会偷偷用 3.10 建 venv，然后装 openpi 失败**

```bash
export UV_PYTHON=/root/.local/share/uv/python/cpython-3.11-linux-x86_64-gnu/bin/python3.11
uv venv .venv_rlinf --python "$UV_PYTHON"
source .venv_rlinf/bin/activate
export VIRTUAL_ENV=$PWD/.venv_rlinf
uv sync --extra embodied --active            # 约 20 分钟，6.4 GB
```

**坑 ③：lerobot 从 github 拉，本机 git 不支持 openssl 后端**

准备一个独立 gitconfig（不要污染 `~/.gitconfig`）：

```bash
cat > $WORK/.cache/gitconfig <<'EOF'
[http]
	sslbackend = gnutls
[url "https://ghproxy.net/https://github.com/"]
	insteadOf = https://hub.fastgit.org/
EOF
export GIT_CONFIG_GLOBAL=$WORK/.cache/gitconfig
```

**坑 ④：hydra 版本不兼容**

```bash
uv pip install --python .venv_rlinf/bin/python "hydra-core==1.3.2"
# 现象：hydra.errors.HydraException: version_base='1.1' is not supported in Hydra 1.4
```

**坑 ⑤：pyarrow 要求 numpy≥2，但 ManiSkill 要 numpy 1.26**

```bash
uv pip install --python .venv_rlinf/bin/python "pyarrow==16.1.0" "numpy==1.26.4" \
  "scene-synthesizer==1.15.0" "trimesh==4.5.3" "mujoco" \
  "mani_skill==3.0.1" "sapien==3.0.3"
```

**收尾：装 openpi 包 + 覆盖 transformers 补丁 + 建软链**

```bash
uv pip install -e $WORK --python .venv_rlinf/bin/python
cp -r $WORK/src/openpi/models_pytorch/transformers_replace/* \
      .venv_rlinf/lib/python3.11/site-packages/transformers/
ln -sfn .venv_rlinf .venv_openpi      # 他们的脚本硬编码了这个路径
```

**验收**：

```bash
VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json \
PYTHONPATH=$PWD:$RBM/repo:$WORK/src \
.venv_rlinf/bin/python -c "
import torch, rlinf, openpi, mani_skill, dsynth.envs, dsynth.robots
print('OK', torch.__version__, mani_skill.__version__)"
# 期望：OK 2.6.0+cu124 3.0.1
```

### 2.4 统一环境变量（建议写进 `$WORK/my_env.sh`）

```bash
export HF_HOME=$WORK/.cache/huggingface
export HF_LEROBOT_HOME=$HF_HOME/lerobot
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
export OPENPI_DATA_HOME=$WORK/.cache/openpi
export MS_ASSET_DIR=/root/gpufree-data/rbm_eval_assets/ms_assets
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json

# 【必须】把缓存挪出系统盘，否则训练时 / 会被写满（我们就因此崩过一次）
export HF_DATASETS_CACHE=$SHARE/hf_datasets_cache
export HF_HUB_CACHE=$HF_HOME/hub
export TMPDIR=$SHARE/tmp
mkdir -p "$HF_DATASETS_CACHE" "$HF_HUB_CACHE" "$TMPDIR"
```

---

## 3. 数据准备

### 3.1 下载（三个商品，一次一个）

```bash
cd $WORK && source my_env.sh
uv run python .cache/tools/download_hf_parallel.py \
  --repo emb-ai/RoboBenchMart_demo_envs_mp --repo-type dataset \
  --local-dir .cache/rbm/demo_envs_mp \
  --include "pick_to_basket/demos/motionplanning/pick_to_basket_fanta_248traj_4workers.rgbd.pd_joint_pos.physx_cpu.h5" \
  --workers 16
```

文件清单（**注意在 `_mp` 这个 repo，不在 `demo_envs`**）：

```
fanta  27.36 GB
nivea  26.95 GB
stars  27.45 GB
```

### 3.2 转换前必须打的 3 个补丁

文件：`robobenchmart-openpi-rlinf-ppo-main/sft/convert_pick_to_basket_h5_to_lerobot.py`

#### 补丁 A：批量读取（提速 31 倍，**不打这个补丁全量转换要 12 小时**）

**位置**：`convert()` 函数内，`frame_data = {...}` 之前

```python
# 改前（逐帧索引 → 每次解压整个 chunk，140 ms/帧）
actions = traj["actions"]
qpos = traj["obs/agent/qpos"]
left = traj["obs/sensor_data/left_base_camera_link/rgb"]
right = traj["obs/sensor_data/right_base_camera_link/rgb"]
wrist = traj["obs/sensor_data/fetch_hand/rgb"]

# 改后（整条轨迹批量读，4.4 ms/帧）
actions = np.asarray(traj["actions"])
qpos = np.asarray(traj["obs/agent/qpos"])
left = np.asarray(traj["obs/sensor_data/left_base_camera_link/rgb"])
right = np.asarray(traj["obs/sensor_data/right_base_camera_link/rgb"])
wrist = np.asarray(traj["obs/sensor_data/fetch_hand/rgb"])
```

#### 补丁 B：append 模式（磁盘放不下 3 个 h5，必须下一个转一个）

**改动 1**：`convert()` 签名加参数

```python
def convert(
    input_h5s, *, repo_id, root, fps, max_episodes, max_frames_per_episode,
    overwrite, image_format, jpeg_quality,
    append: bool = False,          # ← 新增
) -> None:
```

**改动 2**：`convert()` 开头替换这一段

```python
output_path = root / repo_id
episode_offset = 0
frame_offset = 0
existing_tasks: dict[str, int] = {}
if output_path.exists():
    if append:
        import json as _json
        ep_file = output_path / "meta/episodes.jsonl"
        if ep_file.exists():
            lines = [ln for ln in ep_file.read_text().splitlines() if ln.strip()]
            episode_offset = len(lines)
            frame_offset = sum(_json.loads(ln)["length"] for ln in lines)
        tk = output_path / "meta/tasks.jsonl"
        if tk.exists():
            for ln in tk.read_text().splitlines():
                if ln.strip():
                    rec = _json.loads(ln)
                    existing_tasks[rec["task"]] = rec["task_index"]
    elif not overwrite:
        raise FileExistsError(f"{output_path} already exists. Pass --overwrite to replace it.")
    elif overwrite:
        shutil.rmtree(output_path)
(output_path / "meta").mkdir(parents=True, exist_ok=True)   # ← exist_ok 必须加
```

**改动 3**：初始化与循环

```python
task_to_index: dict[str, int] = dict(existing_tasks)
global_frame_index = frame_offset
written_frames = 0
# ...
for _local_index, (h5_path, traj_key) in enumerate(tqdm(episode_plan, ...)):
    episode_index = episode_offset + _local_index
```

**改动 4**：info 计数（**这里最容易错**）

```python
# 改前（会重复加偏移，出现 744 而非 496 的 bug）
info["total_episodes"] = episode_offset + completed_episodes
info["total_frames"] = written_frames

# 改后（completed_episodes 已含偏移）
info["total_episodes"] = completed_episodes
info["total_frames"] = frame_offset + written_frames
```

**改动 5**：CLI 加参数

```python
parser.add_argument("--append", action="store_true")
# main() 里：append=args.append,
```

### 3.3 执行转换（一次一个商品）

```bash
cd $WORK && source my_env.sh
H5=$PWD/.cache/rbm/demo_envs_mp/pick_to_basket/demos/motionplanning/pick_to_basket_fanta_248traj_4workers.rgbd.pd_joint_pos.physx_cpu.h5

# 第一条：--overwrite
uv run python robobenchmart-openpi-rlinf-ppo-main/sft/convert_pick_to_basket_h5_to_lerobot.py \
  --input-h5 "$H5" --root "$HF_LEROBOT_HOME" --repo-id rbm_dataset \
  --image-format jpeg --jpeg-quality 95 --fps 10 --overwrite

rm -f "$H5"       # 立刻删，腾 27 GB

# 第二、三条：--append
# （nivea.h5 / stars.h5 同样先下载，再 --append，再删）
```

### 3.4 验收（**必须做**）

```bash
python3 -c "
import json
d=json.load(open('$HF_LEROBOT_HOME/rbm_dataset/meta/info.json'))
print('episodes:', d['total_episodes'], 'frames:', d['total_frames'])
"
# 期望：episodes: 744  frames: 307319     ← 与官方文档一帧不差
```

---

## 4. openpi 配置改动（SFT 用）

文件：`src/openpi/training/config.py`

在 `_CONFIGS` 列表里、`# Debugging configs.` 之前插入：

```python
    TrainConfig(
        name="pi05_rbm_sft_expertonly",
        # 冻结 VLM，只训 action expert（427.9M = 12.8%）
        # 全参需 95 GB 显存，46 GB 做不了；这是最接近的替代
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=50,
            discrete_state_input=False,
            paligemma_variant="gemma_2b",
            action_expert_variant="gemma_300m",
        ),
        data=LeRobotRBMDataConfig(
            repo_id="rbm_dataset",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=16,          # 32 会 OOM
        num_workers=8,          # 容器 cgroup 内存限制
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=1e-5,
            decay_steps=1_000_000,
            decay_lr=1e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/root/gpufree-data/openpi/.cache/huggingface/rbm_pi05_official/params"
        ),
        num_train_steps=20_000,
        # ★ 核心：冻结除 ._1 后缀（action expert）以外的全部参数
        freeze_filter=nnx.All(nnx.Param, nnx.Not(nnx_utils.PathRegex(".*llm.*_1.*"))),
    ),
```

**同时确认文件头部有这行 import**：

```python
import openpi.shared.nnx_utils as nnx_utils
```

**验证**：

```bash
uv run python -c "
import openpi.training.config as c
cfg = c.get_config('pi05_rbm_sft_expertonly')
print(cfg.name, cfg.batch_size, cfg.num_train_steps)"
```

---

## 5. 训练 SFT（约 14 小时）

```bash
cd $WORK && source my_env.sh

# norm_stats 必须用官方的（自己重算会让 dim12 越界 → 模型输出 0）
mkdir -p assets/pi05_rbm_sft_expertonly/rbm_dataset
cp .cache/huggingface/rbm_pi05_official/assets/rbm_dataset/norm_stats.json \
   assets/pi05_rbm_sft_expertonly/rbm_dataset/

# 清孤儿 dataloader（上次训练被内核杀死的元凶）
ps -eo pid,comm | awk '$2=="pt_data_worker"{print $1}' | xargs -r kill -9

setsid nohup env CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
  uv run scripts/train.py pi05_rbm_sft_expertonly \
    --exp-name rbm_sft_20k --num-train-steps 20000 \
    --checkpoint-base-dir "$PWD/checkpoints" \
    --save-interval 5000 --keep-period 10000 \
    --overwrite --no-wandb-enabled \
  > logs/train_sft_20k.log 2>&1 &
```

**期望**：2.5 s/step、显存 34.7 GB、loss 0.0665→0.0055、grad_norm 0.17→0.04。

---

## 6. 评测方案 A：openpi 自带服务 + 自建客户端

### 6.1 导出评测初始姿态（**这一步决定成败**）

本机 `env.reset()` 给的是 pose B，而演示数据从 pose A 开始，**必须注入**，否则一切为 0。

#### 6.1.1 为什么对方不用注入、我们必须注入（根因）

这不是"我们多此一举"，而是**两个 RoboBenchMart 快照的机器人默认初始姿态不同**。

**证据 1：演示数据生成时没有传任何姿态参数**

演示 JSON 里记录了 `reset_kwargs`：

```json
"reset_kwargs": {"options": {"reconfigure": true}, "seed": 4}
```

只有 `seed`。所以对方那边 `env.reset(seed=4)` 得到的姿态**就等于**演示起点（pose A）。

**证据 2：我们仓库的默认关键帧是另一套姿态**

文件 `dsynth/robots/ds_fetch.py`：

```python
keyframes = dict(
    rest=Keyframe(
        qpos=np.array([0,0,0, 0.386, 0,0,0, -np.pi/4, 0, np.pi/4, 0, np.pi/3, 0, 0.015, 0.015])
    )
)
#                    torso=0.386    j7=-0.785   j9=0.785   j11=1.047   ← pose B
```

对比：

| | torso | j7 | j9 | j11 | 语义 |
|---|---|---|---|---|---|
| 演示起点（pose A）| 0.340 | **1.400** | 0.760 | **−2.094** | 手高举，准备够货架 |
| 我们 keyframe（pose B）| 0.386 | **−0.785** | 0.785 | **+1.047** | 手垂在身侧 |

手臂差了近 π。对方文档也记录过这个坑（"初始姿态错会让成功率全 0"），
他们的修法是让 `env.reset()` 直接返回 pose A；我们只能靠注入来等价实现。

**证据 3：注入等价于注入一个常量，不存在信息泄漏**

```
30 条演示的初始姿态：均值 [0.340, 1.400, 0.760, -2.094]
                    标准差 [0.000, 0.000, 0.000, 0.000]
```

所有 episode 起点完全相同——演示本来就是从同一个固定 home 姿态开始的。

#### 6.1.2 更干净的替代方案：直接改 keyframe（推荐）

如果不打算复用对方的 RLinf 流程，**改这一行比注入更干净**，因为 PPO 的 env wrapper 就不用打补丁了：

```python
# 文件：dsynth/robots/ds_fetch.py
# 把 rest 关键帧的 qpos 改成演示起点（pose A）
keyframes = dict(
    rest=Keyframe(
        pose=sapien.Pose(),
        qpos=np.array([0, 0, 0, 0.34, 0, 0, 0, 1.4, 0, 0.76, 0, -2*np.pi/3, 0, 0.015, 0.015]),
    )
)
```

改完后 `env.reset()` 原生返回 pose A，**§8.3 的补丁 B 就可以完全跳过**。

代价：修改了仿真器的默认行为，任何依赖"默认 rest 姿态"的流程（比如重采演示）都会跟着变。
本指南保留注入方案，是因为它不动仿真器、只在评测侧生效，更好回滚。

```bash
cd $WORK && source my_env.sh
uv run python - <<'PY'
import numpy as np, pathlib
from datasets import disable_progress_bars; disable_progress_bars()
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

ds = LeRobotDataset("rbm_dataset", root=pathlib.Path(".cache/huggingface/lerobot/rbm_dataset"))
eps = ds.meta.episodes; items = list(eps.values()) if hasattr(eps,"values") else eps
offs = np.concatenate([[0], np.cumsum([int(m["length"]) for m in items])])
out = pathlib.Path(".cache/rbm/eval_states")
for name, base in {"fanta": 0, "nivea": 248, "stars": 496}.items():
    d = out/name; d.mkdir(parents=True, exist_ok=True)
    for i in range(30):
        np.save(d/f"ep{i}_state0.npy", ds[int(offs[base+i])]["state"].numpy().astype(np.float32))
# OOD 没有对应演示，统一用 pose A
d = out/"ood"; d.mkdir(parents=True, exist_ok=True)
q = np.load(out/"fanta"/"ep0_state0.npy")
for i in range(30): np.save(d/f"ep{i}_state0.npy", q)
print("done")
PY
```

**验收**：`q[3]=0.340, q[7]=1.400, q[11]=-2.094`。

### 6.2 启动策略服务

```bash
cd $WORK && source my_env.sh
setsid nohup env CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
  uv run scripts/serve_policy.py --port 8200 policy:checkpoint \
    --policy.config pi05_eval_rbm \
    --policy.dir .cache/huggingface/rbm_pi05_official \
  > logs/serve.log 2>&1 &
# 等 "server listening on 0.0.0.0:8200"
```

### 6.3 跑评测（3 ID × 30 + 2 OOD × 30）

```bash
cd $RBM
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json PYTHONPATH=$PWD/repo
export MS_ASSET_DIR=/root/gpufree-data/rbm_eval_assets/ms_assets
MP=$PWD/demo_envs_mp/pick_to_basket/demos/motionplanning

# ID（用官方 seed 列表 + 注入姿态）
.dsynth-env/bin/python repo/scripts/eval_policy_client.py \
  -e PickToBasketContFantaEnv --scene-dir $PWD/demo_envs/pick_to_basket \
  --host 127.0.0.1 --port 8200 --num-traj 30 --max-horizon 600 \
  --exec-horizon 10 --clip-actions \
  --init-state-dir $PWD/eval_states/fanta \
  --json-path $MP/pick_to_basket_fanta_248traj_4workers.rgbd.pd_joint_pos.physx_cpu.json \
  --save-video --eval-subdir my_eval_fanta

# OOD（seed 42000-42029）
.dsynth-env/bin/python repo/scripts/eval_policy_client.py \
  -e PickToBasketContNestleEnv --scene-dir $PWD/demo_envs/test_unseen_items_pick_to_basket \
  --host 127.0.0.1 --port 8200 --num-traj 30 --max-horizon 600 \
  --exec-horizon 10 --clip-actions --init-state-dir $PWD/eval_states/ood \
  --start-seed 42000 --eval-subdir my_eval_nestle
```

**期望**（官方权重）：ID 69/90 = 76.7%、OOD 2/60 = 3.3%。

---

## 7. JAX → PyTorch 转换

```bash
cd $WORK/robobenchmart-openpi-rlinf-ppo-main && source $WORK/my_env.sh
setsid nohup env PYTHONPATH=$PWD:$WORK/src JAX_PLATFORMS=cpu \
  .venv_rlinf/bin/python rlinf/utils/ckpt_convertor/convert_openpi_jax_to_python.py \
    --checkpoint-dir $WORK/checkpoints/pi05_rbm_sft_expertonly/rbm_sft_20k/19999 \
    --config-name pi05_rbm_sft_expertonly \
    --output-path $WORK/pytorch_ckpts/pi05_rbm_sft_pt \
    --precision bfloat16 > $WORK/logs/convert.log 2>&1 &
```

**两个坑**：
- `--checkpoint-dir` 要传**父目录**（含 `params/` 的那层），不是 `params/` 本身
- `--config-name` 必须是 openpi 里存在的配置名（`pi05_rbm` 是 RLinf 的，会报 "not found"）

**转换后必须补 norm_stats**：

```bash
cp $WORK/.cache/huggingface/rbm_pi05_official/assets/rbm_dataset/norm_stats.json \
   $WORK/pytorch_ckpts/pi05_rbm_sft_pt/rbm_dataset/
```

---

## 8. RLinf 接入（3 个补丁，缺一个 PPO 就是 0 成功）

### 8.1 配置本地化

复制两份配置，改成自己的路径：

```bash
cd $WORK/robobenchmart-openpi-rlinf-ppo-main/examples/embodiment/config

sed -e 's|project_path: /root/autodl-tmp/projects/RoboBenchMart|project_path: /root/gpufree-data/openpi/.cache/rbm/repo|' \
    -e 's|config_dir_path: /root/autodl-tmp/projects/RoboBenchMart/demo_envs/pick_to_basket|config_dir_path: /root/gpufree-data/openpi/.cache/rbm/demo_envs/pick_to_basket\n  init_state_dir: /root/gpufree-data/openpi/.cache/rbm/eval_states/fanta|' \
    -e 's|  sim_backend: auto|  sim_backend: cpu|' \
    env/robobenchmart_pick_to_basket_train3.yaml > env/robobenchmart_pick_to_basket_train3_local.yaml

sed -e 's|env/robobenchmart_pick_to_basket_train3@|env/robobenchmart_pick_to_basket_train3_local@|g' \
    -e 's|sft_data_path: /root/autodl-tmp/lerobot_data|sft_data_path: /root/gpufree-data/openpi/.cache/huggingface/lerobot|' \
    rbm_pick_to_basket_ppo_openpi_pi05.yaml > rbm_ppo_local.yaml
```

### 8.2 补丁 A：assets 用绝对路径

文件：`rlinf/models/embodiment/openpi/dataconfig/__init__.py`

```python
# 改前（Ray worker 的 CWD 不同，相对路径会找不到 → "Normalization stats not found"）
assets=AssetsConfig(assets_dir="checkpoints/torch/pi05_rbm/assets"),

# 改后
assets=AssetsConfig(assets_dir="/<绝对路径>/robobenchmart-openpi-rlinf-ppo-main/checkpoints/torch/pi05_rbm/assets"),
```

同时把 norm_stats 放两处（路径被 model_path 覆盖）：

```bash
mkdir -p checkpoints/torch/pi05_rbm/assets/rbm_dataset
cp $WORK/.cache/huggingface/rbm_pi05_official/assets/rbm_dataset/norm_stats.json \
   checkpoints/torch/pi05_rbm/assets/rbm_dataset/
cp $WORK/.cache/huggingface/rbm_pi05_official/assets/rbm_dataset/norm_stats.json \
   $WORK/pytorch_ckpts/<你的 pt 权重目录>/rbm_dataset/
```

### 8.3 补丁 B：姿态注入（**核心**）

文件：`rlinf/envs/robobenchmart/robobenchmart_env.py`

**改动 1**：`__init__` 里，`cfg = _select_robobenchmart_task(cfg, seed_offset)` 之后

```python
        # init_state_dir 不是 ManiSkill env 的参数，先摘出来，
        # 否则 gym.make(**init_params) 会报 unexpected keyword argument
        self._rbm_init_state_dir = None
        _ip = getattr(cfg, "init_params", None)
        if _ip is not None and "init_state_dir" in _ip:
            self._rbm_init_state_dir = str(_ip["init_state_dir"])
            with open_dict(cfg):
                del cfg.init_params["init_state_dir"]
```

**改动 2**：`reset()` 里，`extracted_obs, infos = super().reset(...)` 之后

```python
        init_dir = getattr(self, "_rbm_init_state_dir", None) or os.environ.get("RBM_INIT_STATE_DIR")
        if full_reset and init_dir:
            import glob as _glob
            import numpy as _np
            files = sorted(_glob.glob(os.path.join(init_dir, "*_state0.npy")))
            if files:
                robot = self.env.unwrapped.agent.robot
                n = int(self.num_envs)
                base = self._rbm_reset_count
                qs = [_np.load(files[(base + i) % len(files)]).astype(_np.float32) for i in range(n)]
                import torch as _torch
                robot.set_qpos(_torch.tensor(_np.stack(qs), device=robot.qpos.device.type))

                # ★★★ 最关键的一行 ★★★
                # 注入发生在 reset 返回观测之后，必须重新取一次观测，
                # 否则策略第一帧仍是注入前的姿态（实测 j7=-0.785 而非 +1.400），
                # 输出被压到 [-1,1]，机器人几乎不动，成功率恒为 0。
                extracted_obs = self._wrap_obs(self.env.unwrapped.get_obs(), infos=infos)
```

**这一行就是"0/12 → 4/12"的全部原因。**

---

## 9. 训练 PPO（含 KL + SFT rehearsal）

### 9.1 启动

```bash
cd $WORK
cat > run_ppo.sh <<'EOF'
#!/usr/bin/env bash
RPO=/root/gpufree-data/openpi/robobenchmart-openpi-rlinf-ppo-main
RBM=/root/gpufree-data/openpi/.cache/rbm/repo
CKPT=${MODEL_PATH:-/root/gpufree-data/openpi/pytorch_ckpts/pi05_rbm_official_pt}
cd "$RPO"
source /root/gpufree-data/openpi/my_env.sh
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
export MS_ASSET_DIR=/root/gpufree-data/rbm_eval_assets/ms_assets
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl ROBOT_PLATFORM=LIBERO
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=1
export PYTHONPATH="$RPO:$RBM:/root/gpufree-data/openpi/src:${PYTHONPATH:-}"
export PATH="$RPO/.venv_rlinf/bin:$PATH"
bash examples/embodiment/run_embodiment.sh rbm_ppo_local LIBERO \
  actor.model.model_path="$CKPT" rollout.model.model_path="$CKPT" \
  env.train.project_path="$RBM" env.eval.project_path="$RBM" \
  runner.max_epochs="${MAX_EPOCHS:-5}" \
  runner.save_interval="${SAVE_EVERY:-5}" \
  runner.val_check_interval=-1 \
  runner.logger.experiment_name="${EXP:-rbm_ppo}" \
  actor.optim.critic_warmup_steps="${WARMUP:-12}" "$@"
EOF
chmod +x run_ppo.sh

# 先启动 Ray（RLinf 必须有）
setsid nohup $RPO/.venv_rlinf/bin/ray start --head --port=6379 --num-cpus=32 --num-gpus=1 > /tmp/ray.log 2>&1 &

MAX_EPOCHS=5 WARMUP=12 SAVE_EVERY=5 setsid nohup bash run_ppo.sh > logs/ppo.log 2>&1 &
```

### 9.2 三个必知的坑

| 坑 | 现象 | 解决 |
|---|---|---|
| **save_interval=1** | 每 epoch 写 19 GB 断点，5 epoch = 95 GB → 磁盘满 → 断点被截断（`failed finding central directory`）| `SAVE_EVERY=5` |
| **epoch 数太少** | 只跑 2 个 epoch 时：warmup 吃掉 epoch 1，若 epoch 2 成功数 <2 被门控拦住 → **KL 和 SFT rehearsal 一次都不触发** | 至少 5 个 epoch |
| **Ray 未启动** | `ConnectionError: Could not find any running Ray instance` | 先 `ray start --head` |

### 9.3 验收（TensorBoard 必须看到这两个指标）

```bash
source $WORK/my_env.sh
$RPO/.venv_rlinf/bin/python - <<'PY'
import glob
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
f = sorted(glob.glob("<logs目录>/tensorboard/events*"))[-1]
ea = EventAccumulator(f); ea.Reload()
tags = ea.Tags()["scalars"]
need = ["train/actor/ref_kl_abs", "train/actor/ref_kl_beta", "train/sft_loss"]
for k in need:
    print(k, "✅" if k in tags else "❌ 缺失（说明机制没触发）")
PY
```

**期望值**：`ref_kl_abs ≈ 0.0012~0.0020`、`sft_loss ≈ 0.03`、`ref_kl_beta_next = beta/1.5`。

---

## 10. 评测方案 B：RLinf 90 集协议

```bash
cat > run_eval_rlinf.sh <<'EOF'
#!/usr/bin/env bash
set -uo pipefail
RPO=/root/gpufree-data/openpi/robobenchmart-openpi-rlinf-ppo-main
RBM=/root/gpufree-data/openpi/.cache/rbm/repo
CKPT_DIR=${CKPT_DIR:?}
BASE=/root/gpufree-data/openpi/pytorch_ckpts/pi05_rbm_official_pt
FULL=$CKPT_DIR/actor/model_state_dict/full_weights.pt
cd "$RPO"
source /root/gpufree-data/openpi/my_env.sh
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json MS_ASSET_DIR=/root/gpufree-data/rbm_eval_assets/ms_assets
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl ROBOT_PLATFORM=LIBERO OMP_NUM_THREADS=1
export PYTHONPATH="$RPO:$RBM:/root/gpufree-data/openpi/src:${PYTHONPATH:-}"
export PATH="$RPO/.venv_rlinf/bin:$PATH"
# RECAP 断点没有 value/noise head，PPO 断点有 → 用 HEADLESS 区分
[ "${HEADLESS:-0}" = "1" ] && HEAD="actor.model.add_value_head=False actor.model.openpi.add_value_head=False actor.model.openpi.noise_method=flow_ode" || HEAD=""
bash examples/embodiment/run_embodiment.sh rbm_ppo_local LIBERO \
  runner.only_eval=True runner.max_epochs=1 runner.val_check_interval=-1 \
  runner.ckpt_path="$FULL" $HEAD \
  actor.model.model_path="$BASE" rollout.model.model_path="$BASE" \
  actor.enable_sft_co_train=False env.eval.project_path="$RBM" \
  env.eval.total_num_envs=3 \
  env.eval.max_steps_per_rollout_epoch=600 env.eval.max_episode_steps=600 \
  env.eval.auto_reset=False env.eval.ignore_terminations=False \
  env.eval.init_params.sim_backend=cpu \
  algorithm.eval_rollout_epoch=30 "$@"
EOF
```

**OOD 变体**（2 个任务，注意 `pipeline_stage_num` 要跟着改）：

```bash
  'env.eval.init_params.id=[PickToBasketContNestleEnv,PickToBasketContSlamEnv]' \
  "env.eval.init_params.config_dir_path=$WORK/.cache/rbm/demo_envs/test_unseen_items_pick_to_basket" \
  'env.eval.init_params.init_state_dir=/root/gpufree-data/openpi/.cache/rbm/eval_states/ood' \
  'env.eval.episode_seed_lists=null' 'env.eval.episode_seed_start=42000' \
  env.eval.total_num_envs=2 rollout.pipeline_stage_num=2 \
  algorithm.eval_rollout_epoch=30
```

**期望**（官方权重）：ID 50/90 = 55.6%、OOD 2/60 = 3.3%。

---

## 11. RECAP-lite（三步）

### 11.1 打标签（全 positive 或 success-column）

```bash
cd $RPO && source $WORK/my_env.sh
# SFT 数据：全 positive
PYTHONPATH=$PWD:$WORK/src .venv_rlinf/bin/python examples/recap/process/make_recap_lite_advantages.py \
  --dataset-path $WORK/.cache/huggingface/lerobot/rbm_dataset \
  --tag rbm_recap_mixed --dataset-type sft --label-mode all-positive --overwrite
```

### 11.2 采集 in-domain rollout（90 条，约 26 分钟）

```bash
# 用 09 脚本或自己调用 run_embodiment：
+env.eval.data_collection.enabled=True \
+env.eval.data_collection.save_dir=/root/gpufree-data/openpi/rollouts \
+env.eval.data_collection.export_format=lerobot \
+env.eval.data_collection.robot_type=ds_fetch_basket \
+env.eval.data_collection.fps=10 \
+env.eval.data_collection.only_success=False \
+env.eval.data_collection.finalize_interval=0 \
algorithm.eval_rollout_epoch=30
```

然后对每个 shard 打 success 标签：

```bash
for d in rollouts/rank_*/id_0; do
  PYTHONPATH=$PWD:$WORK/src .venv_rlinf/bin/python examples/recap/process/make_recap_lite_advantages.py \
    --dataset-path "$d" --tag rbm_recap_mixed --dataset-type rollout \
    --label-mode success-column --overwrite
done
```

**验收**：每个 shard 的 `positive_rate` 应介于 0 和 1 之间（我们的：31.0% / 44.8% / 60.5%）。

### 11.3 mixed 训练（配额必须能被整除）

```bash
SFT_DATASET=$WORK/.cache/huggingface/lerobot/rbm_dataset \
ROLLOUT_DATASET=$WORK/rollouts \
TAG=rbm_recap_mixed MAX_STEPS=150 GLOBAL_BATCH_SIZE=4 \
QUOTA_CYCLE_SIZE=60 POSITIVE_FRACTION=0.67 \
bash scripts_local/11_rbm_recap_cfg_mixed.sh \
  actor.model.model_path=<pt权重> actor.micro_batch_size=1 data.num_workers=0
```

**配额约束（会直接报错）**：

```
cycle=60 → counts = [54, 2, 2, 2]（SFT 0.9 + 每 shard 0.1/3）✅
MAX_STEPS × GLOBAL_BATCH_SIZE 必须是 cycle 的整数倍
  150 × 4 = 600 = 60 × 10 ✅
```

---

## 12. 每步验收清单（照着对，错一步后面全废）

| 步骤 | 命令 | 期望 |
|---|---|---|
| 环境 | `import torch, rlinf, dsynth.envs` | 无报错 |
| 数据 | `cat meta/info.json` | `total_episodes: 744, total_frames: 307319` |
| 初始姿态 | `np.load(ep0_state0.npy)` | `[...,0.340,...,1.400,...,-2.094]` |
| SFT 训练 | 日志 `Step 0` | `loss≈0.066, grad_norm≈0.17` |
| SFT 收敛 | 日志最后 | `loss≈0.0055` |
| 官方评测 | `Results:` | ID 69/90（协议 A）|
| 转换 | `ls pytorch_ckpts/*/model.safetensors` | 6.8 GB |
| **PPO rollout** | `env/success_once` | **≥ 0.17（即 ≥2/12）**，否则查姿态注入 |
| **KL 生效** | TensorBoard | `ref_kl_abs`、`sft_loss` 存在 |
| RECAP 标签 | `positive_rate` | 0 < rate < 1 |

---

## 13. 踩过的所有坑

完整清单见 [`repro_02_pitfalls.md`](repro_02_pitfalls.md)：27 个坑，按类别分组，每条都是
「现象 → 根因 → 解法」。这里不再维护第二份表格，避免两边编号对不上。

照本文顺序执行时最容易撞上的几个已经写进对应步骤：§2.3 的坑 ①–⑤（Python 版本、uv 建 venv、
git ssl、hydra、pyarrow/numpy）、§3.2 的补丁 A/B（批量读、append 与 info 计数）、
§6.1.1（初始姿态为什么必须注入）。

---

## 14. 时间与资源预算

| 阶段 | 时间 | 显存峰值 | 磁盘 |
|---|---|---|---|
| 环境 | 3 h | — | 20 GB |
| 数据（3 商品）| 3.5 h | — | 27 GB（峰值）+ 26 GB 成品 |
| SFT 20k | 14 h | 34.7 GB | 30 GB 断点 |
| 评测 A（全量）| 3 h | 25 GB | 1 GB 视频 |
| 转换 | 0.5 h | CPU | 6.8 GB |
| PPO（5 epoch）| 40 min | 22.6 GB | 19 GB |
| RECAP | 1.5 h | 14 GB | 20 GB |
| **合计** | **约 1.5 天** | | **约 120 GB** |

---

## 15. 三条经验（比命令更重要）

1. **先验证环境，再烧算力**：我因为没验证初始姿态，白跑了一整轮 SFT 训练。
2. **每个阶段都要有独立的验收指标**：数据看帧数、训练看 loss、评测看已知基线、PPO 看 rollout 成功数。任何一个数字不对，后面的都不可信。
3. **数字不可比时要说出来**：我们注入 pose A 让 BASE 从 47.8% 变 55.6%。不声明这一点，整份对比就是误导。

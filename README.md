# RoboBenchMart × π0.5 后训练复现

> 基于 [RLinf](https://github.com/RLinf/RLinf)、[OpenPI](https://github.com/Physical-Intelligence/openpi) 与 RoboBenchMart 构建。
> 本仓库是**复现与适配工作**，不是上述项目的官方发布。上游版权与许可声明全部保留，见 `LICENSE` 与 `NOTICE`。

在 RoboBenchMart 的 `PickToBasket`（把商品抓进篮子）任务上，用 π0.5 完整走通
**SFT → KL-PPO → RECAP-lite → 双协议评测**，并逐项对照原项目的数字。

**结论**：链路全部跑通，定位并修复了上游一个真实 bug；但**未能复现原项目的绝对成功率**——
两套实验的初始条件不同，绝对值不可直接比较（见「局限」）。

---

## 我做了什么

1. **定位并修复上游 bug**：`reset()` 返回观测之后才注入初始姿态，导致策略看到错误的第一帧。
   修复前 PPO rollout **0/12** 成功，修复后 **4/12**。
   （`rlinf/envs/robobenchmart/robobenchmart_env.py`）
2. **实现真 SFT-reference KL-PPO**：上游 embodied PPO 只有 PPO clip（约束 current vs rollout-old），
   缺少对初始 SFT 策略的约束。补上冻结 reference、CPU 权重换入换出、自适应 β、
   critic warmup、成功数门控与 SFT rehearsal。（`rlinf/workers/actor/fsdp_actor_worker.py`、
   `rlinf/utils/embodied_training_safety.py`）
3. **重写 PickToBasket 奖励**：事件式 shaped reward + episode 内「只奖励第一次」状态缓存，
   压制「抬起来就一直举着」的 reward hacking。
4. **RECAP-lite**：用 episode 成败把自主 rollout 分成正/负样本，通过
   `Advantage: positive/negative` 条件化 CFG flow-matching 继续训练；
   采样端改为 episode 均衡 + 确定性配额。
5. **数据管线**：H5 → LeRobot v2.1，批量读 + 增量追加，
   **140 ms/帧 → 4.4 ms/帧（31×）**，使 26 GB 数据可以分三次下载、逐批转换。
6. **两套评测协议**：A（openpi 服务 + RHC-10 闭环）与 B（RLinf 开环 50 步），同种子逐集对照。

改动集中在 35 个文件（见文末「代码地图」）。

## 结果

协议 B（RLinf 开环 50 步，3 个 ID 任务 × 30 集）：

| 模型 | ID 成功 | 协议 A | 原项目文档 |
|---|---:|---:|---:|
| 官方 BASE | 55.6%（50/90） | **76.7%**（69/90） | 47.8% |
| 本次 SFT | 27.8%（25/90） | 56.7%（51/90） | 74.4%（67/90） |
| PPO（step 5） | 58.9%（53/90） | — | 66.7% |
| RECAP mixed | 58.9%（53/90） | — | 57.8%（52/90） |

OOD（2 个未见商品，60 集）：BASE 2/60、SFT 2/60、PPO 2/60、RECAP 4/60。

**同一切始条件下的相对排序**：PPO ≈ RECAP ≈ BASE > SFT。
但**绝对值与原项目不可比**——原项目用环境默认初始姿态，本次从演示导出姿态注入。

## 没做成的部分（负结果）

- **LoRA 微调 12 轮里 11 轮闭环 0%**：数据 60→217 条、步数 3k→8k、裁剪、RHC-1 全部无效；
  离线 MAE 0.05、真值回放 3/3 都正常，闭环仍为 0，结论是行为克隆的曝光偏差。
  唯一非零的 v10 是从**官方权重**继续训得到 45%（9/20），而同协议下官方基线是 85%——
  属于退化，不是提升。这也是后来改用 **expert-only 全参微调（冻 VLM，仅训 427.9M）** 的原因。
- **PPO 提升有限**：上游项目 59/90 → 60/90（+1）；本次复现 50/90 → 53/90（+3）。
  两者都小于任务间的波动幅度，critic 解释方差≈0，不足以支撑「PPO 有效」的结论。
- **RECAP mixed 没有超过 SFT/BASE**：上游 quota 重采样后 step1→step5 权重 relative L2 仅
  `5.36e-9`，本次复现结果为 53/90（= PPO，仅比 BASE +3）。更新量不足以产生能力变化，
  小样本差异应视为采样噪声。

## 复现

环境、数据、路径依赖请先读 [`docs/repro_03_runbook.md`](docs/repro_03_runbook.md)。

**第 0 步：准备环境脚本。** 其余脚本开头都会 `source $OPENPI_ROOT/my_env.sh`，
而仓库里没有这个文件——它就是 `repro_00_env.sh` 的副本：

```bash
export OPENPI_ROOT=/your/path/to/openpi          # 其余变量见下方「路径配置」
cp scripts_local/repro_00_env.sh $OPENPI_ROOT/my_env.sh
```

没复制也能跑：脚本会打印一条 `[warn] 未找到 .../my_env.sh` 然后继续，
但 HF 缓存、`PROJECT_ROOT` 等会落到默认值，可能写满系统盘。

```bash
source scripts_local/repro_00_env.sh        # 让当前 shell 的环境变量生效
bash scripts_local/repro_01_train_ppo.sh    # KL-PPO（5 epoch）
bash scripts_local/repro_04_collect_rollouts.sh
bash scripts_local/repro_05_train_recap_mixed.sh
bash scripts_local/repro_02_eval_recap.sh   # 协议 B
bash scripts_local/repro_03_eval_ood.sh     # OOD
```

`repro_00_env.sh` 按自身位置推断 `PROJECT_ROOT`：放在 `$OPENPI_ROOT/my_env.sh` 时用所在目录；
放在 `scripts_local/` 里直接执行时回退到 `OPENPI_ROOT`——两种用法都不会把缓存建到仓库里。

数据、模型权重、仿真资产均不在仓库内（合计 >100 GB），需自行准备。

**Python 工具必须用 `export`。** 下面几个脚本通过 `os.environ` 读路径，
只在 shell 里赋值（`VAR=x python ...`）不会传给子进程：

```bash
export RBM_DATASET_ROOT=/data/rbm_dataset
python scripts_local/repro_08_offline_mae.py ...
```

涉及 `LEROBOT_ROOT`、`RBM_PROJECT`、`RBM_SCENE_DIR`、`RBM_DATASET_ROOT`、`RBM_ASSETS_DIR`。

**OpenPI patch 里另有两处绝对路径。** `sft/openpi/openpi_rbm_sft_config.patch`
应用到 openpi 后，会把 `assets_dir` 和 `weight_loader` 指向
`/root/autodl-tmp/projects/RoboBenchMart/...`，打补丁前需先改这两行
（patch 第 151、157 行；只改路径文本、不增删行，`git apply` 不受影响）。

**可执行位。** 从 Windows 提交的 `.sh` 可能没有 `+x`，克隆到 Linux 后先执行一次：

```bash
chmod +x scripts_local/*.sh run_stage2_probe.sh
```

### 路径配置

脚本里的绝对路径已改成**可覆盖的环境变量**，默认值仍是原实验机器的路径，
换机器时**脚本本身不用改**，设置变量即可：

| 变量 | 作用 | 默认值 |
|---|---|---|
| `OPENPI_ROOT` | openpi 工作目录 | `/root/gpufree-data/openpi` |
| `RBM_ASSETS_ROOT` | ManiSkill 资产 | `/root/gpufree-data/rbm_eval_assets` |
| `SHARE_ROOT` | 缓存/临时盘 | `/root/gpufree-share` |
| `AUTODL_ROOT` | 早期实验数据盘 | `/root/autodl-tmp` |
| `LEROBOT_ROOT` | LeRobot 数据集根 | `/root/autodl-tmp/lerobot_data` |
| `RBM_PROJECT` / `RBM_SCENE_DIR` | RoboBenchMart 仓库 / 场景目录 | `/root/autodl-tmp/projects/RoboBenchMart` |
| `RBM_DATASET_ROOT` / `RBM_ASSETS_DIR` | 数据集 / openpi 权重资产 | 见对应源码 |

**但 YAML 配置必须手动改 3 处**——它们没有被任何脚本用 `key=value` 覆盖，
只设上面的环境变量不会生效：

| 配置文件 | 键 | 行 |
|---|---|---:|
| `examples/embodiment/config/rbm_ppo_local.yaml` | `actor.sft_data_path` | 150 |
| `examples/embodiment/config/env/robobenchmart_pick_to_basket_train3_local.yaml` | `init_params.config_dir_path` | 35 |
| 同上 | `init_params.init_state_dir` | 36 |

其中前两处影响 PPO 训练与训练场景构建，第三处决定 rollouts 的初始姿态来源，
改错会直接改变成功率——这正是复现里最敏感的那个变量。

> `examples/**` 下共 22 处绝对路径（12 个文件），其余多数会被脚本的 `key=value` 覆盖；
> 用 `grep -rn "/root/" examples/ --include=*.yaml` 可以列出全部。
> `docs/**` 里的路径是当时的实跑记录，保持原样未动。

## 代码地图

上游 RLinf 的 610 个 `.py` 保持原样，我的改动集中在 35 个文件：

```text
rlinf/envs/robobenchmart/            环境 wrapper、proxy 任务、观测刷新修复、奖励
rlinf/models/embodiment/openpi/      RBM 三视角 policy 与 data config
rlinf/models/embodiment/openpi_cfg/  CFG 条件模型（RECAP-lite）
rlinf/workers/actor/                 KL-PPO（SFT reference、自适应 β、门控）
rlinf/workers/sft/fsdp_cfg_worker.py CFG-SFT 训练循环与配额采样
rlinf/data/datasets/recap/           advantage 对齐与混合数据集
rlinf/utils/embodied_training_safety.py  训练安全纯函数（有单测）
sft/                                 H5→LeRobot 转换、OpenPI 补丁
docs/repro_*.md                      复现记录（结果 / 27 个坑 / 手册 / 偏差分析）
```

## 局限

- **绝对值不可比**：初始姿态注入 vs 环境默认姿态，这是两套初始条件（详见 `docs/repro_04_deviations.md`）。
- **RECAP-lite ≠ RECAP**：用 episode 二值成败标签，没有 value 模型、没有专家 correction。
- **单种子**：每个配置只跑一次，没有置信区间。
- **未做**：JAX/PyTorch 动作轨迹逐帧对齐、proxy-mix 消融、断点续训验证。

## 上游与许可

- `LICENSE`：Apache 2.0（上游原文）
- `NOTICE`：衍生关系与归属
- 610 个上游 `.py` 的版权头全部保留；`pyproject.toml` 中的包名与作者元数据（`rlinf` / `RLinf Contributors`）
  描述的是上游包本身，未做修改。

## 硬件

单卡 NVIDIA L40（46 GB）｜SFT 20,000 步 / 14 小时 09 分 / 34.7 GB 显存｜PPO 5 epoch、断点 19 GB｜
RECAP 150 步混合训练｜数据 744 集 / 307,319 帧 / 26 GB。

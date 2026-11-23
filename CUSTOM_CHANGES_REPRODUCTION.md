# 本次复现的改动索引

本文列出在**上游完整源码之上**做的修改，方便代码审查与面试讲解快速定位。

> 基线：上游发布版本（641 个 .py 全部保留）
> 本文档对应的改动已直接体现在源码中，无需打补丁

---

## 1. 核心 bug 修复（最重要）

**文件**：`rlinf/envs/robobenchmart/robobenchmart_env.py`

### 问题

PPO rollout **12 条全部失败（0/12）**，但 SFT 模型在同一环境能成功。

### 根因（两层）

**第一层：初始姿态未被注入**

RoboBenchMart 的演示从固定 home 姿态开始（`torso=0.340, j7=1.400, j9=0.760, j11=-2.094`），
但本机 `env.reset()` 返回的是另一套姿态（`torso=0, j7=-0.785, j9=0.785, j11=1.047`），
配置里的 `robot_init_pose_seed` 在本机无效。

**第二层（真正的 bug）：注入发生在 reset 返回观测之后**

```python
extracted_obs, infos = super().reset(seed=seed, options=options)   # ← 观测已生成
robot.set_qpos(...)                                                 # ← 此时才注入
# extracted_obs 仍是注入前的姿态！
```

策略看到的第一帧是 `j7=-0.785`（错误姿态），却要输出针对 `j7=+1.400` 的动作。

### 修复

```python
if full_reset and init_dir:
    ...
    robot.set_qpos(torch.tensor(np.stack(qs), device=...))
    print(f"[RBM-INJECT] reset#{base} 注入 {n} 个状态, 首个 torso={qs[0][3]:+.3f} j7={qs[0][7]:+.3f}")
    # 【关键】注入后必须重新取观测
    extracted_obs = self._wrap_obs(self.env.unwrapped.get_obs(), infos=infos)
```

另外把 `init_state_dir` 从 `cfg.init_params` 摘出来（它不是 ManiSkill env 的参数）。

### 效果

```
修复前：PPO rollout 0/12 成功
修复后：4/12 成功
```

---

## 2. 数据转换器改造

**文件**：`sft/convert_pick_to_basket_h5_to_lerobot.py`

### 2.1 批量读（140 ms/帧 → 4.4 ms/帧，31 倍）

原实现逐帧索引 h5，每次都解压整个 chunk：

```python
# 原：每次取一帧都要解压 chunk
obs = h5f[f"traj_{i}/obs/..."][frame_idx]

# 改：整条轨迹一次性读入
traj = {k: h5f[f"traj_{i}/{k}"][:] for k in keys}
```

### 2.2 增量追加（--append）

三个商品各 27 GB，磁盘放不下同时存。加 `--append` 支持"下一个、转一个、删一个"：

```python
if append:
    ep_file = output_path / "meta/episodes.jsonl"
    if ep_file.exists():
        lines = [ln for ln in ep_file.read_text().splitlines() if ln.strip()]
        episode_offset = len(lines)
        frame_offset = sum(json.loads(ln)["length"] for ln in lines)
```

**踩过的坑**：追加时 `info.json` 的 `total_episodes` 写成 744 而非 496，
原因是偏移量加了两次。修正为使用 `completed_episodes`（已含偏移）。

---

## 3. 配置与环境适配

| 文件 | 改动 |
|---|---|
| `rlinf/models/embodiment/openpi/dataconfig/__init__.py` | 资产路径改绝对路径 |
| `pyproject.toml` | Python 上限 `3.11.14` → `3.11.18` |

### 新增的配置

| 文件 | 作用 |
|---|---|
| `examples/embodiment/config/rbm_ppo_local.yaml` | PPO 训练（5 epoch，save_interval 5）|
| `examples/embodiment/config/env/robobenchmart_pick_to_basket_train3_local.yaml` | 环境配置，**含 `init_state_dir`** |
| `examples/recap/cfg/config/rbm_recap_local.yaml` | RECAP-lite 混合训练 |
| `run_stage2_probe.sh` | 第二阶段探针（1 环境 1 步，快速验证链路）|

---

## 4. 复现脚本

`scripts_local/repro_*.sh`（与上游的 `scripts_local/0X_*.sh` 区分）：

| 脚本 | 作用 |
|---|---|
| `repro_00_env.sh` | 环境变量（HF 缓存、Vulkan、TMPDIR 全指到数据盘）|
| `repro_01_train_ppo.sh` | PPO 训练（`SAVE_EVERY=5`）|
| `repro_02_eval_recap.sh` | 协议 B 评测 |
| `repro_03_eval_ood.sh` | OOD 评测（`pipeline_stage_num=2`）|
| `repro_04_collect_rollouts.sh` | 采集 90 条 rollout |
| `repro_05_train_recap_mixed.sh` | RECAP-lite 混合训练 |
| `repro_06_eval_all.sh` | 批量评测 |
| `repro_07_record_videos.sh` | 录制评测视频 |
| `repro_08_offline_mae.py` | 离线预测诊断（比较策略输出与专家动作）|

---

## 5. 训练配置要点

### 5.1 SFT：expert-only 微调（不是 LoRA）

```python
freeze_filter = nnx.All(nnx.Param, nnx.Not(nnx_utils.PathRegex(".*llm.*_1.*")))
```

- 冻结 VLM 主干，只训练动作专家
- 可训练参数 **427.9 M（12.8%）**
- batch 16 / 20,000 步 / LR 1e-5 + warmup 1000 / 14 小时 09 分
- loss 0.0665 → 0.0055

### 5.2 PPO：真 SFT-reference KL

```text
ref_logprobs = SFT_reference(forward_inputs)     # 冻结初始 SFT 作为 reference
loss += kl_beta * KL(current_logprobs, ref_logprobs)
自适应 β：target 0.01，高于 2×target 时 ×1.5，低于 0.5×target 时 ÷1.5，clamp [1e-4, 10]
critic warmup 12
actor_update_min_successes = 2
SFT rehearsal（混入 SFT loss 防止遗忘）
```

**实测**：`ref_kl_abs = 0.0012–0.0020`，`sft_loss = 0.0311` 正常触发。

### 5.3 RECAP-lite

- advantage 条件化 CFG-SFT，positive/negative prompt
- unconditional dropout 0.3
- quota cycle 60 = 54 SFT + 每 shard 2 条（1 正 1 负）
- 90 rollouts 采集 26 分钟；shard 正样本率 31.0% / 44.8% / 60.5%
- 混合训练 150 步 @ 1.61 秒/步

---

## 6. 评测协议

| 协议 | 实现 | 说明 |
|---|---|---|
| **A** | openpi 服务 + `eval_policy_client.py --exec-horizon 10` | RHC-10 闭环重规划 |
| **B** | RLinf `only_eval`，`data_horizon=50` | 开环整块执行 |

**为什么两套都跑**：两套协议数字完全不可比（A 闭环、B 开环）。
实测同一模型差 20 个百分点以上（BASE 76.7% vs 55.6%）。
**报告 VLA 成功率必须写清协议。**

---

## 7. 文档

| 文件 | 内容 |
|---|---|
| `docs/repro_00_lora_prelude.md` | 复现前的 12 轮 LoRA 探索（含初始姿态契约的发现）|
| `docs/repro_01_results.md` | 4 模型 × 2 协议评测 + OOD + 结果解读 |
| `docs/repro_02_pitfalls.md` | 27 个踩坑：现象 → 根因 → 解法 |
| `docs/repro_03_runbook.md` | 从零复现的完整步骤（环境 3 小时 → 数据 → 训练 → 评测）|
| `docs/repro_04_deviations.md` | 未能复现的部分、代理指标、不可比之处 |
| `results/experiments.csv` | 探索期 12 轮训练的原始记录 |
| `results/reproduction-comparison.csv` | 本次 4 模型 × 2 协议的实测数字（原项目数字不在这个 CSV 里，见 `repro_04` §1.2）|

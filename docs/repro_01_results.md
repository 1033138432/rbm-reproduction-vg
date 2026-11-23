# 复现结果：4 个模型 × 2 套评测协议

> 硬件：单卡 NVIDIA L40（46 GB）｜SFT 20,000 步 / 14 小时 09 分 / 34.7 GB 显存｜
> PPO 5 epoch、断点 19 GB｜RECAP 150 步混合训练
> 数据：744 集 / 307,319 帧 / 26 GB（H5 → LeRobot v2.1）
> 原始数字：[`../results/reproduction-comparison.csv`](../results/reproduction-comparison.csv)（本次 4 模型 × 2 协议的实测）、
> [`../results/experiments.csv`](../results/experiments.csv)（探索期 12 轮 LoRA 尝试）
> 与原项目文档的逐项对照见 [`repro_04_deviations.md`](repro_04_deviations.md) §1.2（原项目用的哪套协议未说明，
> 那边单独列，不和本文件的 A/B 协议混排）。

## 一、一句话结论

SFT → KL-PPO → RECAP-lite → 双协议评测整条链路跑通，并定位修复了上游一个真实 bug；
但**没有复现出原项目的绝对成功率**——两套实验的初始条件不同，绝对值不可直接比较（见 §三）。

## 二、ID 结果（3 个任务 × 30 集 = 90 集）

| 模型 | 协议 B（RLinf 开环 50 步） | 协议 A（openpi + RHC-10 闭环） | 原项目文档 |
|---|---:|---:|---:|
| 官方 BASE | 55.6%（50/90） | **76.7%**（69/90） | 47.8% |
| 本次 SFT（expert-only） | 27.8%（25/90） | 56.7%（51/90） | 74.4%（67/90） |
| PPO（step 5） | 58.9%（53/90） | — | 66.7% |
| RECAP mixed | 58.9%（53/90） | — | 57.8%（52/90） |

**OOD**（2 个未见商品 × 30 集 = 60 集，协议 B）：BASE 2/60、SFT 2/60、PPO 2/60、RECAP 4/60。
原项目文档里 BASE 5/60、SFT 2/60。

## 三、怎么读这两张表

1. **协议之间不可比。** 同一个 BASE，协议 A 76.7%、协议 B 55.6%，相差 21 个百分点。
   协议 A 每 10 步闭环重规划，协议 B 开环执行整块 50 步动作，两套数字完全不是一个口径。
2. **与原项目的绝对值不可比。** 原项目用环境默认初始姿态，本次从演示导出 `*_state0.npy` 注入
   （PPO rollout 0/12 那轮的根因），这是两套初始条件，只能说方向。
3. **能比较的是同一初始条件下的相对排序**：PPO ≈ RECAP ≈ BASE > SFT。
   本次 SFT 明显低于 BASE（27.8% vs 55.6%，协议 B），与「只训 12.8% 参数」的假设一致，
   但没有全参微调的对照，不能当成结论。

## 四、各阶段要点

| 阶段 | 关键数字 |
|---|---|
| 数据 | 744 集 / 307,319 帧 / 26 GB；转换 140 ms/帧 → 4.4 ms/帧（31×） |
| SFT | expert-only（冻结 VLM，可训练 427.9 M = 12.8%），20,000 步，loss 0.0665 → 0.0055 |
| PPO | 5 epoch、断点 19 GB；`ref_kl_abs` 0.0012–0.0020、`sft_loss` 0.0311 正常触发 |
| RECAP-lite | 90 条 rollout（采集 26 分钟，shard 正样本率 31.0% / 44.8% / 60.5%）；150 步 @ 1.61 秒/步 |
| 环境 bug | 注入初始姿态后没刷新观测，PPO rollout 0/12 → **4/12** |

## 五、负结果与不可比之处

- **PPO 提升不显著**：本次 50/90 → 53/90（+3），上游 59/90 → 60/90（+1）。
  两者都小于任务间波动，critic 解释方差 ≈ 0，不足以支撑「PPO 有效」。
- **RECAP mixed 没有超过 SFT/BASE**：quota 重采样后 step1→step5 权重 relative L2 仅 `5.36e-9`，
  更新量不足以产生能力变化，53/90 与 PPO 相同，应视为采样噪声。
- **SFT 低于 BASE**：见 §三 第 3 条，缺全参对照。
- **单种子**：每个配置只跑一次，没有置信区间。
- 完整的偏差清单、代理指标与「哪些话不能写」见 [`repro_04_deviations.md`](repro_04_deviations.md)。

## 六、相关文档

| 文件 | 内容 |
|---|---|
| [`../README.md`](../README.md) | 结论、复现步骤、代码地图、局限 |
| [`repro_00_lora_prelude.md`](repro_00_lora_prelude.md) | 复现前的 12 轮 LoRA 探索（含初始姿态契约的发现） |
| [`repro_02_pitfalls.md`](repro_02_pitfalls.md) | 27 个坑：现象 → 根因 → 解法 |
| [`repro_03_runbook.md`](repro_03_runbook.md) | 从零复现的完整步骤与每步验收 |
| [`repro_04_deviations.md`](repro_04_deviations.md) | 未复现的部分、代理指标、不可比之处 |

# 踩坑与失败分析

> 复现本项目时踩过的所有问题。每条给出：**现象 → 根因 → 解法**。
> 按类别分组，共 27 个。

---

## 一、环境类（7 个）

### 1. conda channel ToS 失效
**现象**：`CondaError: Terms of Service have not been accepted`
**解法**：改用 `uv`（`uv venv` + `uv pip install`），绕开 conda channel。

### 2. 下载资产被墙
**现象**：`download_asset RoboCasa` 硬编码 `huggingface.co`，连接超时。
**解法**：`HF_ENDPOINT=https://hf-mirror.com` + 手动下载 3.73 GB 资产包。

### 3. Vulkan 找不到 NVIDIA ICD
**现象**：SAPIEN 渲染退回软件模式，速度只有 3 帧/秒。
**解法**：显式 `export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json`。

### 4. ManiSkill 3 用 gymnasium，仓库用 gym
**现象**：`ModuleNotFoundError: No module named 'gym'`
**解法**：写 gym shim（`gym.py` 转发到 gymnasium）。

### 5. 环境注册不到机器人
**现象**：`gym.make` 报找不到 robot uid。
**解法**：显式 `from dsynth.robots import *` 触发注册。

### 6. 资产路径与文档不符
**现象**：文档写的目录名不存在。
**解法**：迁到数据盘 + `~/.maniskill` 软链。

### 7. 文档 5 处错误
**现象**：目录名、脚本路径、不存在的仓库——文档与实际代码不一致。
**解法**：逐一核实修正，并记录在 `repro_03_runbook.md`。

---

## 二、依赖版本冲突（4 个）

### 8. hydra 版本过高
**现象**：`hydra.errors.HydraException: version_base='1.1' is not supported in Hydra 1.4`
**根因**：仓库用 `@hydra.main(version_base="1.1")`，hydra 1.4 不再支持。
**解法**：降级到 `hydra-core==1.3.2`。

### 9. pyarrow 与 numpy 冲突
**现象**：`pyarrow requires NumPy 2.0 or later`（其他依赖要求 numpy 1.x）
**解法**：`pyarrow==16.1.0` + `numpy==1.26.4`。

### 10. 缺 scene_synthesizer
**现象**：`ModuleNotFoundError: No module named 'scene_synthesizer'`
**解法**：装 `scene_synthesizer==1.15.0` 到 `.venv_rlinf`。

### 11. Python 版本上限
**现象**：`RLinf requires Python <=3.11.14`，本机只有 3.11.18。
**解法**：改 `pyproject.toml` → `">=3.10,<=3.11.18"`（已包含在源码中）。

---

## 三、数据转换类（5 个）

### 12. 逐帧读 h5 极慢
**现象**：转换 1 帧要 140 ms，744 条要 8 小时。
**根因**：每次索引都解压整个 chunk。
**解法**：整条轨迹批量读入内存 → **4.4 ms/帧（31 倍）**。见 `sft/convert_pick_to_basket_h5_to_lerobot.py`。

### 13. 磁盘放不下三个 27 GB 的 h5
**现象**：三个商品各 27 GB，同时存超容量。
**解法**：加 `--append` 支持"下一个、转一个、删一个"。

### 14. 追加模式报 FileExistsError
**现象**：`FileExistsError: .../meta`
**解法**：`mkdir(exist_ok=True)`。

### 15. total_episodes 写成 744 而非 496
**现象**：转换第二个商品后 `info.json` 的 `total_episodes` 翻倍。
**根因**：偏移量加了两次。
**解法**：改用 `completed_episodes`（已含偏移）。

### 16. 转换脚本切换 h5 时卡死 35 分钟
**现象**：CPU 100%，无任何输出。
**解法**：改为单文件转换，不做循环。

---

## 四、训练与评测（8 个）

### 17. ★ PPO rollout 全部失败

**现象**：PPO rollout **0/12 成功**，但 SFT 模型在同一环境能成功。

**根因**（两层）：
1. 初始姿态未注入——演示从 `j7=1.400` 开始，环境 reset 给的是 `j7=-0.785`
2. **注入发生在 reset 返回观测之后**——策略看到的第一帧仍是注入前姿态

**解法**：注入后重新取观测（见 `rlinf/envs/robobenchmart/robobenchmart_env.py`）

```python
extracted_obs = self._wrap_obs(self.env.unwrapped.get_obs(), infos=infos)
```

**效果**：0/12 → **4/12**。

### 18. KL 与 SFT rehearsal 从未触发
**现象**：`ref_kl_abs` 和 `sft_loss` 在日志里完全没出现。
**根因**：`critic warmup = 12`，2 个 epoch 时 epoch 1 全被 warmup 占用，
epoch 2 只有 1 条成功 < `actor_update_min_successes=2`，actor 一次都没更新。
**解法**：跑 ≥5 个 epoch。之后 `ref_kl_abs = 0.0012–0.0020`、`sft_loss = 0.0311` 正常出现。

### 19. Normalization stats not found
**现象**：`ValueError: Normalization stats not found`
**根因**：openpi 在 `assets/{config_name}/{repo_id}/` 和断点目录两处找。
**解法**：资产路径改绝对路径（见 `rlinf/models/embodiment/openpi/dataconfig/__init__.py`）+ norm_stats 同时放到断点目录。

### 20. pipeline stage 数配置错误
**现象**：`env.eval.total_num_envs // pipeline_stage_num must be > 0`
**解法**：OOD 评测设 `rollout.pipeline_stage_num=2`，单任务设 `=1`。

### 21. 断点 key 不匹配（两类，方向相反）

| 现象 | 原因 |
|---|---|
| `Missing key: value_head/noise_head` | 用 RECAP 断点跑非 RECAP 配置 |
| `Unexpected key: value_head/noise_head` | 反之 |

**解法**：`HEADLESS=1` 只用于 RECAP 断点。

### 22. Missing key: embed_tokens.weight
**现象**：加载 SFT 断点时报缺这个 key。
**解法**：从 PPO 断点复制该张量进 SFT 的 `.pt`。

### 23. UnpicklingError
**现象**：`_pickle.UnpicklingError` 加载断点。
**根因**：把 safetensors 直接改名成 `.pt`。
**解法**：用 `torch.save` 真正序列化。

### 24. Ray 环境不稳定

**现象**：三种错误轮流出现
- `Found multiple active Ray instances`
- `InvalidAuthToken: Authentication token is missing`
- `Could not find any running Ray instance`

**根因**：
1. 上个会话的 `gcs_server` 还占着 6379 端口
2. `~/.ray/auth_token` 与新集群不匹配
3. **`pkill -f raylet` 会杀掉自己的命令行**（命令里含 "raylet" 字样）

**解法**：
```bash
ps -eo pid,args | awk '/gcs_server/ && !/awk/ {print $1}' | xargs -r kill -9
rm -rf /tmp/ray ~/.ray
# 或绕开 Ray：用 openpi 服务 + eval 客户端
```

---

## 五、资源类（3 个）

### 25. 磁盘写满两次

| 次数 | 原因 | 后果 |
|---|---|---|
| 第一次（数据盘）| 79 GB 旧 PPO 产物 + 12 GB base 权重 | PPO 断点截断（`failed finding central directory`）|
| 第二次（系统盘）| HF 缓存撑满 30 GB 系统盘 | 训练中断 |

**解法**：
- `HF_DATASETS_CACHE` / `TMPDIR` 重定向到数据盘
- 训练前算清 `断点大小 × 数量 + 数据 + 权重 < 可用空间`
- 加磁盘守护脚本（双盘都盯）

### 26. 训练进程被内核杀掉
**现象**：启动几分钟后无声消失，GPU 归零，无报错。
**根因**：容器 cgroup 上限 120 GB，**192 个孤儿 `pt_data_worker` 占了 115 GB**。
**解法**：清理孤儿进程。

### 27. 多进程争抢 GPU
**现象**：`CUDA out of memory`，但 nvidia-smi 显示显存没满。
**根因**：上一个进程的 CUDA context 未完全释放。
**解法**：`nvidia-smi --query-compute-apps=pid` 显式核对，逐个 kill。

---

## 六、经验总结

### 复现第三方项目的三条铁律

1. **先跑最小闭环**（1 环境 1 步），不要一上来就训练
2. **每个数字都要问"什么协议"**——实测同一模型两套协议差 20 个百分点
3. **对方文档里的路径/版本/命令，全部要自己验证**（本次修正 5 处文档错误）

### 判断"是不是自己的问题"的排查顺序

当 rollout 全败时，按这个顺序排查，在第 2 步找到根因：

```
1. 动作是否真的送到环境？     → 加打印，确认动作值
2. 策略看到的第一帧对不对？   → 打印 qpos，发现 j7=-0.785（应为 +1.400）★
3. 环境初始状态是否被覆盖？   → 追进 reset() 看注入时机
4. 模型权重是否加载正确？     → 对比 key 列表
```

**先确认"输入正确"，再怀疑"模型不对"。**

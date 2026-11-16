"""离线预测诊断：在数据集真实观测上比较策略输出与专家动作的误差。
用法: python offline_mae.py <config_name> <ckpt_dir> [n_samples]
"""
import os, sys, pathlib, numpy as np, torch
from openpi.policies import policy_config
from openpi.training import config as _config
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from datasets import disable_progress_bars; disable_progress_bars()

config_name, ckpt, n = sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv)>3 else 200
cfg = _config.get_config(config_name)
policy = policy_config.create_trained_policy(cfg, pathlib.Path(ckpt))
ds = LeRobotDataset(
    "rbm_dataset",
    root=pathlib.Path(
        os.environ.get(
            "RBM_DATASET_ROOT",
            "/root/gpufree-data/openpi/.cache/huggingface/lerobot/rbm_dataset",
        )
    ),
)
eps = ds.meta.episodes; items = list(eps.values()) if hasattr(eps,"values") else eps
starts = np.cumsum([0]+[int(m["length"]) for m in items])
idx = np.linspace(0, len(ds)-1, n).astype(int)
errs_1, errs_10, first = [], [], []
for i in idx:
    s = ds[int(i)]
    img = (s["image"].numpy().transpose(1,2,0)*255).astype(np.uint8)
    wri = (s["wrist_image"].numpy().transpose(1,2,0)*255).astype(np.uint8)
    ext = (s["extra_image"].numpy().transpose(1,2,0)*255).astype(np.uint8)
    out = policy.infer({
        "observation/image": img, "observation/wrist_image": wri,
        "observation/extra_image": ext,
        "observation/state": s["state"].numpy(), "prompt": s["task"],
    })
    pred = np.asarray(out["actions"])          # (50,13)
    # 取当前 episode 内接下来 50 帧的真值动作（LeRobot 不带 delta_timestamps 时只返回单帧）
    ep = int(np.searchsorted(starts, int(i), side="right") - 1)
    lo, hi = int(starts[ep]), int(starts[ep+1]) if ep+1 < len(starts) else len(ds)
    gt = np.stack([ds[int(min(int(i)+k, hi-1))]["actions"].numpy() for k in range(50)])
    errs_1.append(np.abs(pred[0]-gt[0]).mean())
    errs_10.append(np.abs(pred[:10]-gt[:10]).mean())
    first.append(np.abs(pred[0]-gt[0]))
first = np.array(first)
print(f"样本数 {len(idx)}")
print(f"第1步   MAE: {np.mean(errs_1):.4f}")
print(f"前10步  MAE: {np.mean(errs_10):.4f}")
print("逐维 MAE:", np.round(first.mean(0), 3))
print("最大单维误差维度:", int(first.mean(0).argmax()), "值", round(float(first.mean(0).max()),3))

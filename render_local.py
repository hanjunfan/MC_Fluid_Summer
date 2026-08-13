"""本机渲染脚本：读 conc_*.npy 用动态色标重新出图（无需 GPU）。"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

from velmc2024.gpu3d.run_smoke3d_gpu import render_frame  # noqa: E402

root = Path(__file__).resolve().parent
L = 3.0
frames = sorted(root.glob("conc_*.npy"))
print(f"找到 {len(frames)} 个浓度场")

for p in frames:
    q = np.load(p)
    step = int(p.stem.split("_")[1])
    render_frame(q, L, root / f"frame_{step:05d}.png", step * 0.02)
    print(f"渲染 step {step} 完成")

print("全部完成")

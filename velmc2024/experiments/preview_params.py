"""preview_params.py —— 同一帧用不同 vmax / 插值方式渲染成对比图，挑视觉参数。

用法：
    python velmc2024/experiments/preview_params.py --src results/cohomology_vort --step 207
    python velmc2024/experiments/preview_params.py --step 230
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

from velmc2024.solver.scenes import CohomologyScene     # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=str, default="results/cohomology_vort")
    ap.add_argument("--step", type=int, default=207)
    ap.add_argument("--out", type=str, default="results/param_preview.png")
    args = ap.parse_args()

    scene = CohomologyScene(grid_res=96, dt=0.05, time_steps=280)
    c = np.load(ROOT / args.src / f"conc_{args.step:05d}.npy")
    vmaxs = [0.6, 0.4, 0.28, 0.18]
    interps = ["bilinear", "none"]
    t = args.step * 0.05

    fig, axes = plt.subplots(2, len(vmaxs), figsize=(3.1 * len(vmaxs), 6.4))
    for r, interp in enumerate(interps):
        for k, vmax in enumerate(vmaxs):
            ax = axes[r][k]
            ax.imshow(c, extent=(-1.2, 1.2, -1.2, 1.2), origin="lower", cmap="RdBu_r",
                      vmin=-vmax, vmax=vmax, interpolation=interp)
            for ob in scene.obstacles:
                v = ob.vertices
                ax.plot(np.append(v[:, 0], v[0, 0]), np.append(v[:, 1], v[0, 1]), "k-", lw=2)
            ax.set_title(f"±{vmax}  {interp}", fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_aspect("equal")
    fig.suptitle(f"t={t:.2f} 参数对比（行=插值 列=色标）", y=0.99)
    fig.tight_layout()
    out = ROOT / args.out
    fig.savefig(out, dpi=150)
    print(f"已保存 {out}")


if __name__ == "__main__":
    main()

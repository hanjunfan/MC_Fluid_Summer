"""rerender_wake.py —— 用低色标重新渲染 cohomology 帧，显示弱尾丝。

用法：
    python velmc2024/experiments/rerender_wake.py --dir results/cohomology_vort --vmax 0.4
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
    ap.add_argument("--dir", type=str, default="results/cohomology_vort")
    ap.add_argument("--steps", type=str, default="92,115,161,230,276", help="逗号分隔的 step 号")
    ap.add_argument("--vmax", type=float, default=0.4)
    ap.add_argument("--out", type=str, default="results/wake_preview.png")
    args = ap.parse_args()

    scene = CohomologyScene(grid_res=96, dt=0.05, time_steps=280)
    out_dir = ROOT / args.dir
    steps = [int(x) for x in args.steps.split(",")]

    n = len(steps)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.4))
    if n == 1:
        axes = [axes]
    for ax, s in zip(axes, steps):
        c = np.load(out_dir / f"conc_{s:05d}.npy")
        ext = (-1.2, 1.2, -1.2, 1.2)
        ax.imshow(c, extent=ext, origin="lower", cmap="RdBu_r",
                  vmin=-args.vmax, vmax=args.vmax, interpolation="bilinear")
        for ob in scene.obstacles:
            v = ob.vertices
            ax.plot(np.append(v[:, 0], v[0, 0]), np.append(v[:, 1], v[0, 1]), "k-", lw=2)
        ax.set_title(f"t={s * 0.05:.2f}")
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"cohomology 低色标 ±{args.vmax}（显示弱尾丝）", y=0.98)
    fig.tight_layout()
    out = ROOT / args.out
    fig.savefig(out, dpi=140)
    print(f"已保存 {out}（{n} 帧，色标 ±{args.vmax}）")


if __name__ == "__main__":
    main()

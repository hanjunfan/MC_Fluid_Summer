"""rerender_all_wake.py —— 把 cohomology_vort 的 conc_*.npy 用低色标重渲染成 frame_*.png。

用法：
    python velmc2024/experiments/rerender_all_wake.py \
        --src results/cohomology_vort --out results/cohomology_wake --vmax 0.35
然后：
    python velmc2024/experiments/make_gif.py --dir results/cohomology_wake --fps 6
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
    ap.add_argument("--out", type=str, default="results/cohomology_wake")
    ap.add_argument("--vmax", type=float, default=0.35)
    args = ap.parse_args()

    src = ROOT / args.src
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    scene = CohomologyScene(grid_res=96, dt=0.05, time_steps=280)
    concs = sorted(src.glob("conc_*.npy"))
    for p in concs:
        c = np.load(p)
        step = int(p.stem.split("_")[1])
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ext = (-1.2, 1.2, -1.2, 1.2)
        ax.imshow(c, extent=ext, origin="lower", cmap="RdBu_r",
                  vmin=-args.vmax, vmax=args.vmax, interpolation="bilinear")
        for ob in scene.obstacles:
            v = ob.vertices
            ax.plot(np.append(v[:, 0], v[0, 0]), np.append(v[:, 1], v[0, 1]), "k-", lw=2)
        ax.set_title(f"cohomology  t={step * 0.05:.2f}  (色标 ±{args.vmax})")
        ax.set_aspect("equal")
        fig.tight_layout()
        fig.savefig(out / f"frame_{step:05d}.png", dpi=130)
        plt.close(fig)
    print(f"已渲染 {len(concs)} 帧 -> {out}（色标 ±{args.vmax}）")


if __name__ == "__main__":
    main()

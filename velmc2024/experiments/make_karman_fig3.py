"""make_karman_fig3.py —— 合成论文 Fig 3 卡门涡街三 Re 对比图（e-g）。

读三个 karman_fig3_re* 目录的浓度帧，在相同 t（或各自末帧）并排渲染成 1×3 对比。
用法：
    python velmc2024/experiments/make_karman_fig3.py --step 500
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

from velmc2024.solver.scenes import KarmanScene     # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--re25", type=str, default="results/karman_fig3_re25")
    ap.add_argument("--re250", type=str, default="results/karman_fig3_re250")
    ap.add_argument("--re25p", type=str, default="results/karman_fig3_re25p")
    ap.add_argument("--step", type=int, default=0, help="指定 step（0=各目录末帧）")
    ap.add_argument("--out", type=str, default="results/karman_fig3.png")
    args = ap.parse_args()

    dirs = [args.re25p, args.re25, args.re250]
    labels = ["Re=2.5  ν=0.1", "Re=25  ν=0.01", "Re=250  ν=0.001"]
    scene = KarmanScene(viscosity=0.01, grid_res=(80, 40), dt=0.02, time_steps=500)

    steps = []
    for d in dirs:
        concs = sorted((ROOT / d).glob("conc_*.npy"))
        if not concs:
            print(f"警告：{d} 无帧")
            steps.append(None)
        else:
            last = int(concs[-1].stem.split("_")[1])
            steps.append(args.step if args.step else last)

    fig, axes = plt.subplots(1, 3, figsize=(15, 3.2))
    for ax, d, label, s in zip(axes, dirs, labels, steps):
        if s is None:
            ax.text(0.5, 0.5, "无数据", ha="center", va="center")
            ax.axis("off")
            continue
        c = np.load(ROOT / d / f"conc_{s:05d}.npy")
        Lx, Ly = 4.0, 2.0
        ax.imshow(c, extent=(-Lx/2, Lx/2, -Ly/2, Ly/2), origin="lower",
                  cmap="RdBu_r", vmin=0, vmax=1.2, interpolation="bilinear")
        th = np.linspace(0, 2*np.pi, 100)
        ax.plot(-1.5 + 0.125*np.cos(th), 0.125*np.sin(th), "k-", lw=2)
        ax.set_title(f"{label}  t={s*0.02:.1f}", fontsize=11)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("卡门涡街（论文 Fig 3 e-g）：均匀来流绕圆柱，浓度条带", y=0.98)
    fig.tight_layout()
    out = ROOT / args.out
    fig.savefig(out, dpi=150)
    print(f"已保存 {out}（steps={steps}）")


if __name__ == "__main__":
    main()

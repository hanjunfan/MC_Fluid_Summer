"""run_smoke3d.py —— 三维浮力烟运行脚本 + 切片渲染。

用法：
    python velmc2024/experiments/run_smoke3d.py --grid 16 --steps 40 --smoke
    python velmc2024/experiments/run_smoke3d.py --grid 24 --steps 120 --nvol 120 --npsb 120
渲染三个正交切片（侧面 xy / 正面 yz / 俯视 xz）拼成一帧，输出 PNG + GIF。
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

from velmc2024.solver.smoke3d import Smoke3DSolver  # noqa: E402


def render_frame(q: np.ndarray, L: float, path: Path, t: float):
    """渲染浓度场三个正交切片。q shape (nz, ny, nx)，y 为垂直方向。"""
    nz, ny, nx = q.shape
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), constrained_layout=True)
    ext = (-L / 2, L / 2, -L / 2, L / 2)

    # 侧面视图（xz 平面切片？—— 垂直方向 y，取 x=中 的 yz 面 + z=中 的 xy 面）
    # 1) 侧面：xy 平面（z=中）—— 看烟沿 y 上升
    ax = axes[0]
    im = ax.imshow(q[:, :, nx // 2], extent=ext, origin="lower",
                   cmap="inferno", vmin=0, vmax=1)
    ax.set_title("侧面 (z=0)：烟柱上升")
    ax.set_xlabel("x"); ax.set_ylabel("y")

    # 2) 正面：yz 平面（x=中）
    ax = axes[1]
    ax.imshow(q[:, ny // 2, :], extent=ext, origin="lower",
              cmap="inferno", vmin=0, vmax=1)
    ax.set_title("正面 (x=0)")
    ax.set_xlabel("z"); ax.set_ylabel("y")

    # 3) 俯视：xz 平面（y=中）
    ax = axes[2]
    ax.imshow(q[nz // 2, :, :], extent=ext, origin="lower",
              cmap="inferno", vmin=0, vmax=1)
    ax.set_title("俯视 (y=0)")
    ax.set_xlabel("x"); ax.set_ylabel("z")

    fig.suptitle(f"3D 浮力烟  t={t:.2f}")
    fig.colorbar(im, ax=axes, shrink=0.7, label="浓度")
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, default=24)
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--nvol", type=int, default=150)
    ap.add_argument("--npsb", type=int, default=150)
    ap.add_argument("--smoke", action="store_true", help="冒烟测试（少量步）")
    ap.add_argument("--beta", type=float, default=1.0, help="浮力强度 buoyancy_beta")
    ap.add_argument("--relax", type=float, default=0.3, help="投影阻尼 projection_relax")
    ap.add_argument("--out", type=str, default="results/smoke3d")
    args = ap.parse_args()

    steps = 30 if args.smoke else args.steps
    solver = Smoke3DSolver(
        grid_res=args.grid, domain_size=3.0, dt=0.02,
        num_volume_samples=args.nvol, num_pseudo_boundary_samples=args.npsb,
        buoyancy_beta=args.beta, projection_relax=args.relax,
    )
    print(f"3D 烟: 网格 ({solver.nx},{solver.ny},{solver.nz}), dt={solver.dt}, "
          f"步数 {steps}, nvol={args.nvol}, npsb={args.npsb}, "
          f"β={args.beta}, relax={args.relax}")
    print(f"网格点数 {solver.nx * solver.ny * solver.nz}, "
          f"烟源盒中心 {solver.smoke_center}, 半宽 {solver.smoke_half}")

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    save_every = max(1, steps // 12)
    t0 = time.perf_counter()
    solver.run(steps, save_every=save_every, out_dir=str(out_dir))
    dt_wall = time.perf_counter() - t0
    print(f"总耗时 {dt_wall:.1f}s, 平均每步 {dt_wall / steps:.2f}s")

    print("渲染 PNG 帧...")
    L = solver.domain_size
    for p in sorted(out_dir.glob("conc_*.npy")):
        q = np.load(p)
        step = int(p.stem.split("_")[1])
        render_frame(q, L, out_dir / f"frame_{step:05d}.png", step * solver.dt)
    print(f"完成，帧图在 {out_dir}")


if __name__ == "__main__":
    main()

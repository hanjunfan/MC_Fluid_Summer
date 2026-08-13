"""run_smoke3d_gpu.py —— PyTorch GPU 高精度三维浮力烟运行脚本 + 切片渲染。

用法（在 GPU 机器上）：
    pip install torch numpy matplotlib
    python velmc2024/gpu3d/run_smoke3d_gpu.py --grid 128 --steps 500 --nvol 5000 --npsb 2000

渲染三个正交切片（侧面 xy / 正面 yz / 俯视 xz）拼成一帧，输出 PNG + GIF。
"""

import argparse
import time
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

from velmc2024.gpu3d.smoke3d_gpu import SmokeSolverGPU  # noqa: E402


def render_frame(q: np.ndarray, L: float, path: Path, t: float):
    """渲染浓度场三个正交切片。q shape (nz, ny, nx)，y 为垂直方向。"""
    nz, ny, nx = q.shape
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), constrained_layout=True)
    ext = (-L / 2, L / 2, -L / 2, L / 2)

    im = axes[0].imshow(q[:, :, nx // 2], extent=ext, origin="lower",
                        cmap="inferno", vmin=0, vmax=1)
    axes[0].set_title("侧面 (z=0)")
    axes[0].set_xlabel("x"); axes[0].set_ylabel("y")

    axes[1].imshow(q[:, ny // 2, :], extent=ext, origin="lower",
                   cmap="inferno", vmin=0, vmax=1)
    axes[1].set_title("正面 (x=0)")
    axes[1].set_xlabel("z"); axes[1].set_ylabel("y")

    axes[2].imshow(q[nz // 2, :, :], extent=ext, origin="lower",
                   cmap="inferno", vmin=0, vmax=1)
    axes[2].set_title("俯视 (y=0)")
    axes[2].set_xlabel("x"); axes[2].set_ylabel("z")

    fig.suptitle(f"3D 浮力烟 (GPU)  t={t:.2f}")
    fig.colorbar(im, ax=axes, shrink=0.7, label="浓度")
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, default=128)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--nvol", type=int, default=5000)
    ap.add_argument("--npsb", type=int, default=2000)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--relax", type=float, default=1.0)
    ap.add_argument("--out", type=str, default="results/smoke3d_gpu")
    args = ap.parse_args()

    solver = SmokeSolverGPU(
        grid_res=args.grid, nvol=args.nvol, npsb=args.npsb,
        buoyancy_beta=args.beta, projection_relax=args.relax,
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_every = max(1, args.steps // 20)
    t0 = time.perf_counter()
    solver.run(args.steps, save_every=save_every, out_dir=str(out_dir))
    wall = time.perf_counter() - t0
    print(f"总耗时 {wall:.1f}s ({wall/60:.1f}min), 平均每步 {wall/args.steps:.2f}s")

    print("渲染 PNG 帧...")
    L = solver.L
    for p in sorted(out_dir.glob("conc_*.npy")):
        q = np.load(p)
        step = int(p.stem.split("_")[1])
        render_frame(q, L, out_dir / f"frame_{step:05d}.png", step * solver.dt)
    print(f"完成，帧图在 {out_dir}")


if __name__ == "__main__":
    main()

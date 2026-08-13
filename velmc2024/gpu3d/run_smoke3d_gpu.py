"""run_smoke3d_gpu.py —— PyTorch GPU 高精度三维浮力烟运行脚本 + 切片渲染。

用法（在 GPU 机器上）：
    pip install torch numpy matplotlib
    python velmc2024/gpu3d/run_smoke3d_gpu.py --grid 128 --steps 500 --nvol 5000 --npsb 2000

渲染三个正交切片（侧面 xy / 正面 yz / 俯视 xz）拼成一帧，输出 PNG + GIF。
"""

import argparse
import sys
import time
from pathlib import Path

# 确保项目根目录在 sys.path 中（无论从哪个目录运行脚本都能找到 velmc2024 包）
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

from velmc2024.gpu3d.smoke3d_gpu import SmokeSolverGPU  # noqa: E402


def render_frame(q: np.ndarray, L: float, path: Path, t: float, crop: float = 0.75):
    """渲染浓度场三个正交切片。q shape (nz, ny, nx)=(z,y,x)，y 为垂直方向。

    烟柱在 x、z 方向很窄，裁剪到 [-crop, crop] 聚焦烟柱；y 方向保留全范围看上升。
    """
    nz, ny, nx = q.shape
    h = L / nx

    def idx(lo, hi, n):
        return slice(max(0, int((lo + L / 2) / h)), min(n, int((hi + L / 2) / h)))

    yr = slice(0, ny)              # y 全范围（垂直，看上升）
    xr = idx(-crop, crop, nx)      # x 裁剪
    zr = idx(-crop, crop, nz)      # z 裁剪
    ext_xz = (-crop, crop, -crop, crop)
    ext_side = (-crop, crop, -L / 2, L / 2)   # (x, y)
    ext_front = (-L / 2, L / 2, -crop, crop)  # (y, z)

    # 动态色标：按非零浓度的 95 分位设 vmax，避免低浓度烟柱渲染成一片黑
    qpos = q[q > 1e-4]
    vmax = float(np.percentile(qpos, 95)) if qpos.size > 0 else 1.0
    vmax = max(vmax, 0.02)

    fig, axes = plt.subplots(1, 3, figsize=(15, 6), constrained_layout=True)

    # 侧面：xy 平面（z=中），横轴 x，纵轴 y
    im = axes[0].imshow(q[nz // 2, yr, xr], extent=ext_side, origin="lower",
                        cmap="inferno", vmin=0, vmax=vmax)
    axes[0].set_title("侧面 (z=0)：烟柱上升")
    axes[0].set_xlabel("x"); axes[0].set_ylabel("y")

    # 正面：yz 平面（x=中），横轴 y，纵轴 z
    axes[1].imshow(q[zr, yr, nx // 2], extent=ext_front, origin="lower",
                   cmap="inferno", vmin=0, vmax=vmax)
    axes[1].set_title("正面 (x=0)")
    axes[1].set_xlabel("y"); axes[1].set_ylabel("z")

    # 俯视：xz 平面（y=中），横轴 x，纵轴 z
    axes[2].imshow(q[zr, ny // 2, xr], extent=ext_xz, origin="lower",
                   cmap="inferno", vmin=0, vmax=vmax)
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
    ap.add_argument("--beta", type=float, default=4.0, help="浮力强度（补偿平流耗散，4 能升到顶）")
    ap.add_argument("--relax", type=float, default=0.15, help="投影松弛（≤1，越小越稳）")
    ap.add_argument("--umax", type=float, default=1.5, help="速度钳制上限（防发散，0=关闭）")
    ap.add_argument("--out", type=str, default="results/smoke3d_gpu")
    args = ap.parse_args()

    solver = SmokeSolverGPU(
        grid_res=args.grid, nvol=args.nvol, npsb=args.npsb,
        buoyancy_beta=args.beta, projection_relax=args.relax,
        umax_clamp=args.umax,
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

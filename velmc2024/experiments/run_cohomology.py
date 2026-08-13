"""
run_cohomology.py —— 复现论文 cohomology 场景（两团气体穿过双六边形障碍间隙）

用法：
    python velmc2024/experiments/run_cohomology.py --grid 64 --steps 120 --paths 300 --smoke
    python velmc2024/experiments/run_cohomology.py --grid 96 --steps 300 --paths 800
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

from velmc2024.solver.scenes import CohomologyScene     # noqa: E402
from velmc2024.solver.mc_solver import MCFluidSolver    # noqa: E402


def render_frame(solver, path: Path, t: float):
    """画浓度场（红/蓝两团气体）+ 障碍轮廓 + 速度场。"""
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    q = solver.c_cache.q
    extent = (-solver.scene.domain_size / 2, solver.scene.domain_size / 2,
              -solver.scene.domain_size / 2, solver.scene.domain_size / 2)
    im = ax.imshow(q, extent=extent, origin="lower", cmap="RdBu_r",
                   vmin=-1.2, vmax=1.2, interpolation="bilinear")
    # 障碍轮廓
    for ob in solver.scene.obstacles:
        v = ob.vertices
        ax.plot(np.append(v[:, 0], v[0, 0]), np.append(v[:, 1], v[0, 1]), "k-", lw=2)
    # 速度场（抽样）
    nx, ny = solver.u_cache.nx, solver.u_cache.ny
    i = np.arange(0, nx, max(1, nx // 12))
    j = np.arange(0, ny, max(1, ny // 12))
    X, Y = solver.u_cache.idx_to_point(i[:, None], j[None, :])  # (len(i), len(j))
    U = solver.u_cache.u[np.ix_(j, i, [0])].squeeze().T
    V = solver.u_cache.u[np.ix_(j, i, [1])].squeeze().T
    ax.quiver(X, Y, U, V, color="gray", alpha=0.5, scale=30)
    ax.set_title(f"cohomology 浓度场  t={t:.1f}")
    ax.set_aspect("equal")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, default=64)
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--paths", type=int, default=300)
    ap.add_argument("--nvol", type=int, default=800)
    ap.add_argument("--smoke", action="store_true", help="只跑少量步冒烟测试")
    ap.add_argument("--out", type=str, default="results/cohomology")
    args = ap.parse_args()

    steps = 20 if args.smoke else args.steps
    scene = CohomologyScene(grid_res=args.grid, dt=0.05, time_steps=steps)
    solver = MCFluidSolver(scene, num_paths=args.paths,
                           num_volume_samples_direct=args.nvol,
                           num_pseudo_boundary_samples_direct=args.nvol,
                           verbose=True)
    print(scene.describe())
    print(f"求解器: paths={args.paths}, nvol={args.nvol}, grid={args.grid}")

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    save_every = max(1, steps // 12)
    t0 = time.perf_counter()
    solver.run(steps, save_every=save_every, out_dir=str(out_dir), prefix="conc")
    print(f"总耗时 {time.perf_counter()-t0:.1f}s, 平均每步 "
          f"{(time.perf_counter()-t0)/steps:.2f}s")

    # 渲染 PNG 序列
    print("渲染 PNG 帧...")
    for p in sorted(out_dir.glob("conc_*.npy")):
        q = np.load(p)
        solver.c_cache.q[...] = q
        step = int(p.stem.split("_")[1])
        # 若有该时刻速度场则载入（箭头对应时刻）
        vpath = out_dir / f"vel_{step:05d}.npy"
        if vpath.exists():
            solver.u_cache.u[...] = np.load(vpath)
        render_frame(solver, out_dir / f"frame_{step:05d}.png", step * scene.dt)
    print(f"完成，帧图在 {out_dir}")


if __name__ == "__main__":
    main()

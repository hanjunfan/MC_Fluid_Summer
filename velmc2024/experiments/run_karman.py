"""
run_karman.py —— 复现论文卡门涡街场景（均匀来流绕圆障碍，粘性可调）

用法：
    python velmc2024/experiments/run_karman.py --nu 0.01 --steps 200 --smoke
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

from velmc2024.solver.scenes import KarmanScene           # noqa: E402
from velmc2024.solver.mc_solver import MCFluidSolver      # noqa: E402
from velmc2024.core.geometry import circle_to_polygon     # noqa: E402


def render_frame(solver, path: Path, t: float):
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    q = solver.c_cache.q
    Lx, Ly = solver.u_cache.Lx, solver.u_cache.Ly
    extent = (-Lx / 2, Lx / 2, -Ly / 2, Ly / 2)
    im = ax.imshow(q, extent=extent, origin="lower", cmap="RdBu_r",
                   vmin=0, vmax=1.2, interpolation="bilinear")
    for ob in solver.scene.obstacles:
        if hasattr(ob, "vertices"):
            v = ob.vertices
            ax.plot(np.append(v[:, 0], v[0, 0]), np.append(v[:, 1], v[0, 1]), "k-", lw=2)
        else:
            th = np.linspace(0, 2 * np.pi, 100)
            ax.plot(ob.center[0] + ob.radius * np.cos(th), ob.center[1] + ob.radius * np.sin(th), "k-", lw=2)
    ax.set_title(f"Kármán 涡街 浓度  ν={solver.scene.viscosity}  t={t:.1f}")
    ax.set_aspect("equal")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nu", type=float, default=0.01)
    ap.add_argument("--grid", type=int, default=64)   # 64 表示 x 方向；y = x//2
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--paths", type=int, default=300)
    ap.add_argument("--nvol", type=int, default=800)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", type=str, default="results/karman")
    args = ap.parse_args()

    steps = 30 if args.smoke else args.steps
    nx = args.grid
    ny = max(2, nx // 2)
    scene = KarmanScene(viscosity=args.nu, grid_res=(nx, ny), dt=0.02, time_steps=steps)
    # 卡门场景有粘性+入流，涡量模式（平流ω重建速度）未实现这两项，用旧速度平流模式
    solver = MCFluidSolver(scene, num_paths=args.paths,
                           num_volume_samples_direct=args.nvol,
                           num_pseudo_boundary_samples_direct=args.nvol,
                           advect_vorticity=False)
    print(scene.describe())
    print(f"求解器: paths={args.paths}, nvol={args.nvol}, grid={scene.grid_res}")

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    save_every = max(1, steps // 15)
    t0 = time.perf_counter()
    solver.run(steps, save_every=save_every, out_dir=str(out_dir), prefix="conc")
    print(f"总耗时 {time.perf_counter()-t0:.1f}s, 平均每步 {(time.perf_counter()-t0)/steps:.2f}s")

    print("渲染 PNG 帧...")
    for p in sorted(out_dir.glob("conc_*.npy")):
        q = np.load(p)
        solver.c_cache.q[...] = q
        step = int(p.stem.split("_")[1])
        vpath = out_dir / f"vel_{step:05d}.npy"
        if vpath.exists():
            solver.u_cache.u[...] = np.load(vpath)
        render_frame(solver, out_dir / f"frame_{step:05d}.png", step * scene.dt)
    print(f"完成，帧图在 {out_dir}")


if __name__ == "__main__":
    main()

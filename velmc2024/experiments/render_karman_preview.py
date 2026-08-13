"""渲染 Karman 浓度场预览：跑 N 步后画一帧，检查浓度是否绕过圆柱。"""
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

from velmc2024.solver.scenes import KarmanScene        # noqa: E402
from velmc2024.solver.mc_solver import MCFluidSolver    # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nu", type=float, default=0.01)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--grid", type=int, default=80)
    ap.add_argument("--out", type=str, default="results/karman_preview")
    ap.add_argument("--save", type=int, default=0, help="每隔 N 步存一帧（0=只存最后）")
    args = ap.parse_args()

    scene = KarmanScene(viscosity=args.nu, grid_res=(args.grid, args.grid // 2),
                        dt=0.02, time_steps=args.steps)
    s = MCFluidSolver(scene, num_paths=200,
                      num_volume_samples_direct=600,
                      num_pseudo_boundary_samples_direct=600,
                      advect_vorticity=False)
    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    for k in range(args.steps):
        s.step()
        if args.save > 0 and (k + 1) % args.save == 0:
            _render(s, out_dir / f"frame_{k+1:04d}.png", (k + 1) * scene.dt)
            _save_npy(s, out_dir, k + 1)
    if args.save == 0 or args.steps % args.save != 0:
        _render(s, out_dir / f"frame_{args.steps:04d}.png", args.steps * scene.dt)
        _save_npy(s, out_dir, args.steps)
    print(f"完成，帧图在 {out_dir}")


def _save_npy(solver, out_dir: Path, step: int):
    np.save(out_dir / f"conc_{step:05d}.npy", solver.c_cache.q)
    np.save(out_dir / f"vel_{step:05d}.npy", solver.u_cache.u)


def _render(solver, path: Path, t: float):
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    q = solver.c_cache.q
    Lx, Ly = solver.u_cache.Lx, solver.u_cache.Ly
    extent = (-Lx / 2, Lx / 2, -Ly / 2, Ly / 2)
    im = ax.imshow(q, extent=extent, origin="lower", cmap="RdBu_r",
                   vmin=0, vmax=1.2, interpolation="bilinear")
    for ob in solver.scene.obstacles:
        v = ob.vertices
        ax.plot(np.append(v[:, 0], v[0, 0]), np.append(v[:, 1], v[0, 1]), "k-", lw=2)
    ax.set_title(f"Kármán 浓度  ν={solver.scene.viscosity}  t={t:.1f}")
    ax.set_aspect("equal")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()

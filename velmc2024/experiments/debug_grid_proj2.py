"""干净测试 v2：只测求解器的 _project（宽模板+4pin）。
- 只留边界环固体
- 边界衰减的平滑场
- 只测离固体≥2格的内部流体单元散度
"""
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from velmc2024.solver.scenes import CohomologyScene
from velmc2024.reference.grid_solver import GridFluidSolver


def main():
    nx = 64
    s = CohomologyScene(grid_res=nx, dt=0.05)
    g = GridFluidSolver(s)
    ring = np.zeros((nx, nx), dtype=bool)
    ring[0, :] = ring[-1, :] = ring[:, 0] = ring[:, -1] = True
    g.solid = ring
    g._enforce_solid()
    hx = hy = s.domain_size / nx

    def divergence(u):
        return ((np.roll(u[..., 0], -1, axis=1) - np.roll(u[..., 0], 1, axis=1)) / (2 * hx)
                + (np.roll(u[..., 1], -1, axis=0) - np.roll(u[..., 1], 1, axis=0)) / (2 * hy))

    interior = np.zeros((nx, nx), dtype=bool)
    interior[3:-3, 3:-3] = True

    pts = g._grid_points()
    xx = pts[:, 0]; yy = pts[:, 1]
    L = s.domain_size
    r0 = L / 4.0
    env = np.exp(-(xx ** 2 + yy ** 2) / r0 ** 2)
    base_u = np.stack([np.sin(np.pi * xx / L) * np.cos(np.pi * yy / L),
                       -np.cos(np.pi * xx / L) * np.sin(np.pi * yy / L)], axis=1) * env[:, None]
    perturb = np.stack([env * np.sin(np.pi * xx / L), np.zeros_like(xx)], axis=1)
    g.u_cache.u[...] = (base_u + 0.7 * perturb).reshape(nx, nx, 2)
    g._enforce_solid()

    d0 = np.abs(divergence(g.u_cache.u))[interior].max()
    print(f"投影前 内部 div max = {d0:.4f}  |u|max={np.abs(g.u_cache.u).max():.3f}")
    g._project()
    d1 = np.abs(divergence(g.u_cache.u))[interior].max()
    print(f"投影后 内部 div max = {d1:.8f}  |u|max={np.abs(g.u_cache.u).max():.3f}")
    print(f"降幅 = {d0/d1:.1f}x")


if __name__ == "__main__":
    main()

"""隔离测试网格基准的投影算子：u=(x,0) 投影一次，散度应大幅减小。"""
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from velmc2024.solver.scenes import CohomologyScene
from velmc2024.reference.grid_solver import GridFluidSolver


def divergence(u, hx, hy):
    return ((np.roll(u[..., 0], -1, axis=1) - np.roll(u[..., 0], 1, axis=1)) / (2 * hx)
            + (np.roll(u[..., 1], -1, axis=0) - np.roll(u[..., 1], 1, axis=0)) / (2 * hy))


def main():
    s = CohomologyScene(grid_res=48, dt=0.05)
    hx = hy = s.domain_size / 48

    def divergence(u):
        return ((np.roll(u[..., 0], -1, axis=1) - np.roll(u[..., 0], 1, axis=1)) / (2 * hx)
                + (np.roll(u[..., 1], -1, axis=0) - np.roll(u[..., 1], 1, axis=0)) / (2 * hy))

    g = GridFluidSolver(s)
    pts = g._grid_points()
    L = s.domain_size
    # 平滑测试场：u = (sin(π x/L)cos(π y/L), -cos(π x/L)sin(π y/L)) 泰勒-格林（无散度）
    # 再叠加一个发散分量 sin(π x/L)（制造散度）
    xx = pts[:, 0]; yy = pts[:, 1]
    g.u_cache.u[...] = 0.0
    g.u_cache.u[..., 0] = (np.sin(np.pi * xx / L) * np.cos(np.pi * yy / L) + 0.5 * np.sin(np.pi * xx / L)).reshape(48, 48)
    g.u_cache.u[..., 1] = (-np.cos(np.pi * xx / L) * np.sin(np.pi * yy / L)).reshape(48, 48)
    g._enforce_solid()
    d0 = np.abs(divergence(g.u_cache.u)).max()
    print("投影前 div max", round(d0, 4), " |u|max", round(np.abs(g.u_cache.u).max(), 3))
    g._project()
    u = g.u_cache.u
    d1 = np.abs(divergence(u)).max()
    print("投影后 div max", round(d1, 6), " |u|max", round(np.abs(u).max(), 3))
    print("div 降幅", round(d0 / d1, 1) if d1 > 0 else "inf")


if __name__ == "__main__":
    main()

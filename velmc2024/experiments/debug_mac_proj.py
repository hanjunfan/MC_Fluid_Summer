"""隔离测试 MAC 投影：干净盒子 + 平滑场，只测投影。"""
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
    # 只留边界环固体
    ring = np.zeros((nx, nx), dtype=bool)
    ring[0, :] = ring[-1, :] = ring[:, 0] = ring[:, -1] = True
    g.solid = ring
    g.solid_x[:] = False
    g.solid_y[:] = False
    g.solid_x[0, :] = g.solid_x[-1, :] = True
    g.solid_y[:, 0] = g.solid_y[:, -1] = True
    g.fluid = ~g.solid

    X, Y = g._cell_centers()
    L = s.domain_size
    r0 = L / 4.0
    env = np.exp(-(X ** 2 + Y ** 2) / r0 ** 2)
    # 面速度：无散度基（泰勒-格林）+ 发散扰动
    u_init = np.sin(np.pi * X / L) * np.cos(np.pi * Y / L) * env + 0.7 * np.sin(np.pi * X / L) * env
    v_init = -np.cos(np.pi * X / L) * np.sin(np.pi * Y / L) * env
    # 放到 x 面 / y 面
    Xu, Yu = g._xface_pos()
    g.u_f[...] = (np.sin(np.pi * Xu / L) * np.cos(np.pi * Yu / L) * np.exp(-(Xu ** 2 + Yu ** 2) / r0 ** 2)
                  + 0.7 * np.sin(np.pi * Xu / L) * np.exp(-(Xu ** 2 + Yu ** 2) / r0 ** 2))
    Xv, Yv = g._yface_pos()
    g.v_f[...] = -np.cos(np.pi * Xv / L) * np.sin(np.pi * Yv / L) * np.exp(-(Xv ** 2 + Yv ** 2) / r0 ** 2)
    g._enforce_solid()

    def div():
        return (g.u_f[1:, :] - g.u_f[:-1, :]) / g.hx + (g.v_f[:, 1:] - g.v_f[:, :-1]) / g.hy

    interior = np.zeros((nx, nx), dtype=bool)
    interior[3:-3, 3:-3] = True
    d0 = np.abs(div())[interior].max()
    print(f"投影前 内部 div max = {d0:.4f} |u|max={np.abs(g.u_f).max():.3f}")
    g._project()
    d1 = np.abs(div())[interior].max()
    print(f"投影后 内部 div max = {d1:.8f} |u|max={np.abs(g.u_f).max():.3f}")
    print(f"降幅 = {d0/d1:.1f}x")


if __name__ == "__main__":
    main()

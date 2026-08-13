"""逐步诊断 MAC 求解器：每步打印平流后/投影后的 |u|max 与 div。"""
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from velmc2024.solver.scenes import CohomologyScene
from velmc2024.reference.grid_solver import GridFluidSolver


def main():
    s = CohomologyScene(grid_res=48, dt=0.05)
    g = GridFluidSolver(s)

    def divmax():
        d = (g.u_f[1:, :] - g.u_f[:-1, :]) / g.hx + (g.v_f[:, 1:] - g.v_f[:, :-1]) / g.hy
        return np.abs(d)[g.fluid].max()

    print(f"init: |u|max={np.abs(g.u_f).max():.4f} div={divmax():.6f}")
    for step in range(5):
        nu, nv = g._advect_face_velocities(0.05)
        a_u = np.abs(nu).max()
        g.u_f, g.v_f = nu, nv
        d1 = divmax()
        g._enforce_solid()   # 与 step() 一致：投影前强制固体面
        g._project()
        print(f"step{step+1}: 平流后|u|max={a_u:.4f} div={d1:.5f} | 投影后|u|max={np.abs(g.u_f).max():.4f} div={divmax():.8f}")
        if np.abs(g.u_f).max() > 100:
            break


if __name__ == "__main__":
    main()

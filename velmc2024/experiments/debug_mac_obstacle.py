"""诊断：cohomology 初始场投影后压力/散度在哪放大。"""
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from velmc2024.solver.scenes import CohomologyScene
from velmc2024.reference.grid_solver import GridFluidSolver


def main():
    s = CohomologyScene(grid_res=64, dt=0.05)
    g = GridFluidSolver(s)

    def divfield():
        return (g.u_f[1:, :] - g.u_f[:-1, :]) / g.hx + (g.v_f[:, 1:] - g.v_f[:, :-1]) / g.hy

    d0 = divfield()
    print("初始 div: max(流体)", np.abs(d0)[g.fluid].max(),
          "在障碍附近(流体)占比", (np.abs(d0)[g.fluid] > 1).mean())
    # 障碍附近流体单元（相邻有固体）
    near_solid = np.zeros((g.nx, g.ny), dtype=bool)
    near_solid[1:, :] |= g.solid[:-1, :]
    near_solid[:-1, :] |= g.solid[1:, :]
    near_solid[:, 1:] |= g.solid[:, :-1]
    near_solid[:, :-1] |= g.solid[:, 1:]
    print("远离障碍的流体单元 div max:", np.abs(d0)[g.fluid & ~near_solid].max())

    g._project()
    # 手动重算压力看 p
    from scipy.sparse import lil_matrix
    from scipy.sparse.linalg import cg
    fluid = g.fluid
    div = divfield()
    b = -div.ravel()
    A = lil_matrix((g.nx * g.ny, g.nx * g.ny))
    idx = np.arange(g.nx * g.ny).reshape(g.nx, g.ny)
    for di, dj, coef in [(1, 0, 1.0 / g.hx ** 2), (-1, 0, 1.0 / g.hx ** 2),
                         (0, 1, 1.0 / g.hy ** 2), (0, -1, 1.0 / g.hy ** 2)]:
        ni, nj = np.broadcast_arrays(np.clip(np.arange(g.nx)[:, None] + di, 0, g.nx - 1),
                                     np.clip(np.arange(g.ny)[None, :] + dj, 0, g.ny - 1))
        m = fluid & fluid[ni, nj]
        for r, cc in zip(idx[m], idx[ni[m], nj[m]]):
            A[r, cc] = -coef
    dg = np.zeros((g.nx, g.ny))
    for di, dj, coef in [(1, 0, 1.0 / g.hx ** 2), (-1, 0, 1.0 / g.hx ** 2),
                         (0, 1, 1.0 / g.hy ** 2), (0, -1, 1.0 / g.hy ** 2)]:
        ni, nj = np.broadcast_arrays(np.clip(np.arange(g.nx)[:, None] + di, 0, g.nx - 1),
                                     np.clip(np.arange(g.ny)[None, :] + dj, 0, g.ny - 1))
        fn = fluid[ni, nj]
        dd = np.zeros((g.nx, g.ny))
        dd[fluid] = fn[fluid] * coef
        dg[fluid] += dd[fluid]
    A.setdiag(dg.ravel())
    A = A.tolil()
    for k in np.where(~fluid.ravel())[0]:
        A[k, :] = 0; A[k, k] = 1.0
    pin = int(np.argmax(fluid.ravel()))
    A[pin, :] = 0; A[pin, pin] = 1.0; b[pin] = 0
    A = A.tocsr()
    p, info = cg(A, b, rtol=1e-8, maxiter=1000)
    print("CG info", info, "p range", round(p.min(), 4), round(p.max(), 4))
    p = p.reshape(g.nx, g.ny)
    print("p 在障碍附近流体单元的最大值:", np.abs(p)[fluid & near_solid].max())
    print("p 在远离障碍流体单元的最大值:", np.abs(p)[fluid & ~near_solid].max())


if __name__ == "__main__":
    main()

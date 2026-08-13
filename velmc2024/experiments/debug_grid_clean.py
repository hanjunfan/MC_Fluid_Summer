"""干净测试：网格投影算子（窄 5 点模板）。
- 只保留边界环固体（清掉六边形固体），避免障碍边界伪散度干扰
- 边界处衰减的平滑场，制造已知散度
- 只在"离固体≥2格"的内部流体单元上测散度
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
    # 清掉障碍固体，只留边界环
    ring = np.zeros((nx, nx), dtype=bool)
    ring[0, :] = ring[-1, :] = ring[:, 0] = ring[:, -1] = True
    g.solid = ring
    g._enforce_solid()

    hx = hy = s.domain_size / nx

    def divergence(u):
        return ((np.roll(u[..., 0], -1, axis=1) - np.roll(u[..., 0], 1, axis=1)) / (2 * hx)
                + (np.roll(u[..., 1], -1, axis=0) - np.roll(u[..., 1], 1, axis=0)) / (2 * hy))

    # 内部流体单元（离固体 >= 2 格）
    interior = np.zeros((nx, nx), dtype=bool)
    interior[2:-2, 2:-2] = True

    pts = g._grid_points()
    xx = pts[:, 0]; yy = pts[:, 1]
    L = s.domain_size
    # 无散度基 + 发散扰动，都随 exp(-(r/r0)^2) 衰减（边界处≈0）
    r0 = L / 4.0
    env = np.exp(-(xx ** 2 + yy ** 2) / r0 ** 2)
    base_u = np.stack([np.sin(np.pi * xx / L) * np.cos(np.pi * yy / L),
                       -np.cos(np.pi * xx / L) * np.sin(np.pi * yy / L)], axis=1) * env[:, None]
    perturb = np.stack([env * np.sin(np.pi * xx / L), np.zeros_like(xx)], axis=1)  # 发散扰动
    u0 = base_u + 0.7 * perturb
    g.u_cache.u[...] = u0.reshape(nx, nx, 2)
    g._enforce_solid()

    d0 = np.abs(divergence(g.u_cache.u))[interior].max()
    print(f"投影前 内部 div max = {d0:.4f}")

    # ---- 手动复刻压力求解，检查每个环节 ----
    u = g.u_cache.u
    ny = nx
    fluid = ~g.solid
    div = np.zeros((ny, nx))
    div[fluid] = divergence(u)[fluid]
    print("div: max", np.abs(div).max(), "sum", div[fluid].sum())
    from scipy.sparse import lil_matrix
    from scipy.sparse.linalg import cg
    A = lil_matrix((ny * nx, ny * nx))
    idx = np.arange(ny * nx).reshape(ny, nx)
    for di, dj, coef in [(0, 1, 1.0 / hx ** 2), (0, -1, 1.0 / hx ** 2),
                         (1, 0, 1.0 / hy ** 2), (-1, 0, 1.0 / hy ** 2)]:
        ni, nj = np.broadcast_arrays(
            np.clip(np.arange(ny)[:, None] + di, 0, ny - 1),
            np.clip(np.arange(nx)[None, :] + dj, 0, nx - 1))
        m = fluid & fluid[ni, nj]
        for r, cc in zip(idx[m], idx[ni[m], nj[m]]):
            A[r, cc] = -coef
    diag_vals = np.zeros((ny, nx))
    for di, dj, coef in [(0, 1, 1.0 / hx ** 2), (0, -1, 1.0 / hx ** 2),
                         (1, 0, 1.0 / hy ** 2), (-1, 0, 1.0 / hy ** 2)]:
        ni, nj = np.broadcast_arrays(
            np.clip(np.arange(ny)[:, None] + di, 0, ny - 1),
            np.clip(np.arange(nx)[None, :] + dj, 0, nx - 1))
        fn = fluid[ni, nj]
        dd = np.zeros((ny, nx))
        dd[fluid] = fn[fluid] * coef
        diag_vals[fluid] += dd[fluid]
    A.setdiag(diag_vals.ravel())
    A = A.tolil()
    for k in np.where(~fluid.ravel())[0]:
        A[k, :] = 0
        A[k, k] = 1.0
    A = A.tocsr()
    b = -div.ravel()
    p, info = cg(A, b, rtol=1e-10, maxiter=2000)
    print("CG info", info, "p range", p.min(), p.max())
    res = np.abs(A @ p - b).max()
    print("线性残差 |Ap-b| max =", res)
    p2 = p.reshape(ny, nx)
    lap = (np.roll(p2, -1, axis=1) + np.roll(p2, 1, axis=1) - 2 * p2) / hx ** 2 \
          + (np.roll(p2, -1, axis=0) + np.roll(p2, 1, axis=0) - 2 * p2) / hy ** 2
    print("窄∇²p vs div: max diff(内部)", np.abs(lap[interior] - div[interior]).max())
    wlap = (np.roll(p2, -2, axis=1) + np.roll(p2, 2, axis=1) - 2 * p2) / (4 * hx ** 2) \
           + (np.roll(p2, -2, axis=0) + np.roll(p2, 2, axis=0) - 2 * p2) / (4 * hy ** 2)
    print("宽∇²p vs div: max diff(内部)", np.abs(wlap[interior] - div[interior]).max())

    g._project()


if __name__ == "__main__":
    main()

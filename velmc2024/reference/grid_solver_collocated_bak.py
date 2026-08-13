"""
grid_solver.py —— 传统网格法基准求解器（Stable Fluids 风格）

作为"正确结果"参考，与蒙特卡洛求解器对比：
  - 同位网格（collocated），速度/浓度/固体掩膜
  - RK3 半拉格朗日平流（与 MC 一致）
  - 显式/隐式粘性扩散（可选）
  - PCG 压力投影（∇²p = ∇·u，自由滑移 Neumann 边界 + 障碍物无穿透）
  - 入流/出流边界（卡门涡街）

论文 Fig 3(a) 的参考就是这种传统网格法（Batty et al. 2007 风格）。
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import cg

from ..core.velocity_cache import VelocityCache, ScalarCache


class GridFluidSolver:
    def __init__(self, scene, grid_res=None, domain_size=None, seed=0):
        self.scene = scene
        self.grid_res = tuple(grid_res) if grid_res else scene.grid_res
        self.domain_size = domain_size if domain_size else scene.domain_size
        self.nx, self.ny = self.grid_res
        self.u_cache = VelocityCache(self.nx, self.ny, self.domain_size)
        self.c_cache = ScalarCache(self.nx, self.ny, self.domain_size)
        self.rng = np.random.default_rng(seed)
        self.time = 0.0
        self.step_count = 0
        self._build_solid_mask()
        self._init_fields()

    # ------------------------------------------------------------------ #
    def _build_solid_mask(self):
        i = np.arange(self.nx)
        j = np.arange(self.ny)
        x, y = self.u_cache.idx_to_point(i[:, None], j[None, :])
        pts = np.stack([x.ravel(), y.ravel()], axis=1)
        solid = np.zeros(pts.shape[0], dtype=bool)
        for ob in self.scene.obstacles:
            solid |= ob.inside(pts)
        solid = solid.reshape(self.ny, self.nx)
        # 盒子外边界一圈视为固体（自由滑移墙），避免 np.roll 环绕
        solid[0, :] = True
        solid[-1, :] = True
        solid[:, 0] = True
        solid[:, -1] = True
        # 微膨胀障碍边界
        self.solid = _dilate(solid)

    def _grid_points(self):
        i = np.arange(self.nx)
        j = np.arange(self.ny)
        x, y = self.u_cache.idx_to_point(i[:, None], j[None, :])
        return np.stack([x.ravel(), y.ravel()], axis=1)

    def _init_fields(self):
        pts = self._grid_points()
        v = self.scene.initial_velocity(pts)
        c = self.scene.initial_concentration(pts)
        self.u_cache.u[...] = v.reshape(self.ny, self.nx, 2)
        self.c_cache.q[...] = c.reshape(self.ny, self.nx)
        self._enforce_solid()

    def _enforce_solid(self):
        solid = np.asarray(self.scene.solid_velocity, dtype=np.float64)
        self.u_cache.u[self.solid] = solid
        self.c_cache.q[self.solid] = 0.0

    # ------------------------------------------------------------------ #
    def step(self):
        scene = self.scene
        dt = scene.dt
        # 平流
        u_new = self._advect_vector()
        c_new = self._advect_scalar()
        # 扩散
        if scene.viscosity > 0:
            u_new = self._diffuse(u_new, scene.viscosity, dt)
            c_new = self._diffuse_scalar(c_new, scene.viscosity, dt)
        if hasattr(scene, "apply_inlet"):
            scene.apply_inlet(u_new, c_new)
        # 投影
        self.u_cache.u[...] = u_new.u
        self._project()
        self.c_cache.q[...] = c_new.q
        self._enforce_solid()
        self.time += dt
        self.step_count += 1

    def _advect_vector(self):
        """RK3 半拉格朗日平流速度场。"""
        from ..core.advection import advect_vector
        return advect_vector(self.u_cache, self.u_cache, self.scene.dt, "RK3")

    def _advect_scalar(self):
        from ..core.advection import advect_scalar
        return advect_scalar(self.c_cache, self.u_cache, self.scene.dt, "RK3")

    def _diffuse(self, u_new, nu, dt):
        from ..core.diffusion import diffuse_vector
        return diffuse_vector(u_new, nu, dt)

    def _diffuse_scalar(self, c_new, nu, dt):
        from ..core.diffusion import diffuse_scalar
        return diffuse_scalar(c_new, nu, dt)

    # ------------------------------------------------------------------ #
    def _project(self):
        """压力投影：∇²p = ∇·u，自由滑移 Neumann 边界，固体掩膜。"""
        u = self.u_cache.u
        ny, nx = self.ny, self.nx
        hx = self.u_cache.Lx / nx
        hy = self.u_cache.Ly / ny
        solid = self.solid
        fluid = ~solid

        # 散度（中心差分 2h，仅流体单元）
        div = np.zeros((ny, nx))
        div[fluid] = ((np.roll(u[:, :, 0], -1, axis=1) - np.roll(u[:, :, 0], 1, axis=1))[fluid] / (2 * hx)
                      + (np.roll(u[:, :, 1], -1, axis=0) - np.roll(u[:, :, 1], 1, axis=0))[fluid] / (2 * hy))

        # 压力矩阵用"宽模板拉普拉斯"，与中心差分 div/grad 完全自洽：
        #   ∇²p ≈ [p(i+2)+p(i-2)+p(j+2)+p(j-2) - 4p]/(4h²)
        # 这样 ∇·(∇p) = 宽拉普拉斯(p) = div，投影精确。
        # 注意：宽模板按 (i mod 2, j mod 2) 把网格解耦成 4 个子格，
        # 每个子格有各自的常数零空间，需每个子格固定一个压力零点。
        A = lil_matrix((ny * nx, ny * nx), dtype=np.float64)
        idx = np.arange(ny * nx).reshape(ny, nx)
        for di, dj, coef in [(0, 2, 1.0 / (4 * hx ** 2)), (0, -2, 1.0 / (4 * hx ** 2)),
                             (2, 0, 1.0 / (4 * hy ** 2)), (-2, 0, 1.0 / (4 * hy ** 2))]:
            ni, nj = np.broadcast_arrays(
                np.clip(np.arange(ny)[:, None] + di, 0, ny - 1),
                np.clip(np.arange(nx)[None, :] + dj, 0, nx - 1))
            mask = fluid & fluid[ni, nj]   # 邻居也是流体才连边（Neumann）
            rows = idx[mask]
            cols = idx[ni[mask], nj[mask]]
            vals = np.full(rows.size, -coef)
            for r, cc, vv in zip(rows, cols, vals):
                A[r, cc] = vv
        diag_vals = np.zeros((ny, nx))
        for di, dj, coef in [(0, 2, 1.0 / (4 * hx ** 2)), (0, -2, 1.0 / (4 * hx ** 2)),
                             (2, 0, 1.0 / (4 * hy ** 2)), (-2, 0, 1.0 / (4 * hy ** 2))]:
            ni, nj = np.broadcast_arrays(
                np.clip(np.arange(ny)[:, None] + di, 0, ny - 1),
                np.clip(np.arange(nx)[None, :] + dj, 0, nx - 1))
            fluid_nbr = fluid[ni, nj]
            diag_vals[fluid] += fluid_nbr[fluid] * coef
        A.setdiag(diag_vals.ravel())

        # 固体行置单位阵；每个奇偶子格固定一个流体单元（消除各自常数零空间）
        A = A.tolil()
        fluid_flat = fluid.ravel()
        b = -div.ravel()
        for k in np.where(~fluid_flat)[0]:
            A[k, :] = 0
            A[k, k] = 1.0
        yy, xx = np.indices((ny, nx))
        for pj in (0, 1):
            for pi in (0, 1):
                cls = fluid & (yy % 2 == pj) & (xx % 2 == pi)
                if cls.any():
                    k = int(idx[cls][0])
                    A[k, :] = 0
                    A[k, k] = 1.0
                    b[k] = 0.0
        A = A.tocsr()

        p, info = cg(A, b, rtol=1e-8, maxiter=1000)
        if info != 0:
            print(f"  [grid] PCG 未收敛 (info={info})")
        p = p.reshape(ny, nx)

        # u -= ∇p（中心差分 2h，固体单元不动）
        gx = np.zeros_like(p)
        gy = np.zeros_like(p)
        gx[fluid] = (np.roll(p, -1, axis=1) - np.roll(p, 1, axis=1))[fluid] / (2 * hx)
        gy[fluid] = (np.roll(p, -1, axis=0) - np.roll(p, 1, axis=0))[fluid] / (2 * hy)
        u[fluid, 0] -= gx[fluid]
        u[fluid, 1] -= gy[fluid]

    def run(self, n_steps, save_every=0, out_dir=None, prefix="frame"):
        import os
        if save_every > 0 and out_dir:
            os.makedirs(out_dir, exist_ok=True)
        saved = []
        for _ in range(n_steps):
            self.step()
            if save_every > 0 and self.step_count % save_every == 0 and out_dir:
                path = os.path.join(out_dir, f"{prefix}_{self.step_count:05d}.npy")
                np.save(path, self.c_cache.q)
                saved.append(path)
        return saved


def _dilate(mask: np.ndarray) -> np.ndarray:
    """对固体掩膜做一次 4-邻域膨胀（避免边界泄漏）。"""
    m = mask.copy()
    ny, nx = m.shape
    for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        ni = np.clip(np.arange(ny)[:, None] + di, 0, ny - 1)
        nj = np.clip(np.arange(nx)[None, :] + dj, 0, nx - 1)
        m |= mask[ni, nj]
    return m

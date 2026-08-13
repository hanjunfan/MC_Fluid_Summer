"""
grid_solver_mac.py —— 传统网格法基准求解器（MAC 交错网格，Stable Fluids 风格）

作为"正确结果"参考，与蒙特卡洛求解器对比：
  - MAC（交错网格）：u 在 x 面、v 在 y 面、压力/浓度在单元中心
  - RK3 半拉格朗日平流（与 MC 一致）
  - 显式/隐式粘性扩散（可选，高斯卷积近似）
  - PCG 压力投影（∇²p = ∇·u，自由滑移 Neumann 边界 + 障碍无穿透）
  - 入流/出流边界（卡门涡街）

MAC 网格上散度（面通量）与压力梯度（面压力差）天然自洽（互逆），
所以标准的 5 点压力矩阵就能精确消除散度（论文 Fig 3(a) 的参考正是这类网格法）。

为兼容对比接口，暴露 cell 中心的 u_cache / c_cache（插值自面速度）。
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import cg

from ..core.velocity_cache import VelocityCache, ScalarCache


def _dilate(mask: np.ndarray) -> np.ndarray:
    """4 邻域膨胀（微增厚固体/障碍边界）。"""
    out = mask.copy()
    out[:-1, :] |= mask[1:, :]
    out[1:, :] |= mask[:-1, :]
    out[:, :-1] |= mask[:, 1:]
    out[:, 1:] |= mask[:, :-1]
    return out


class GridFluidSolver:
    def __init__(self, scene, grid_res=None, domain_size=None, seed=0):
        self.scene = scene
        self.grid_res = tuple(grid_res) if grid_res else scene.grid_res
        self.domain_size = domain_size if domain_size else scene.domain_size
        self.nx, self.ny = self.grid_res
        if np.isscalar(self.domain_size):
            self.Lx = self.Ly = float(self.domain_size)
        else:
            self.Lx, self.Ly = float(self.domain_size[0]), float(self.domain_size[1])
        self.hx = self.Lx / self.nx
        self.hy = self.Ly / self.ny
        self.rng = np.random.default_rng(seed)
        self.time = 0.0
        self.step_count = 0
        self._build_solid_mask()
        self._init_fields()

    # ------------------------------------------------------------------ #
    # 几何/索引
    # ------------------------------------------------------------------ #
    def _cell_centers(self):
        i = np.arange(self.nx)
        j = np.arange(self.ny)
        x = (i - (self.nx - 1) / 2.0) * self.hx
        y = (j - (self.ny - 1) / 2.0) * self.hy
        X, Y = np.meshgrid(x, y, indexing="ij")
        return X, Y

    def _xface_pos(self):
        """x 面位置 (nx+1, ny)：x=(i-nx/2)hx, y=单元 j 中心。"""
        i = np.arange(self.nx + 1)
        j = np.arange(self.ny)
        x = (i - self.nx / 2.0) * self.hx
        y = (j - (self.ny - 1) / 2.0) * self.hy
        X, Y = np.meshgrid(x, y, indexing="ij")
        return X, Y

    def _yface_pos(self):
        i = np.arange(self.nx)
        j = np.arange(self.ny + 1)
        x = (i - (self.nx - 1) / 2.0) * self.hx
        y = (j - self.ny / 2.0) * self.hy
        X, Y = np.meshgrid(x, y, indexing="ij")
        return X, Y

    def _build_solid_mask(self):
        X, Y = self._cell_centers()
        pts = np.stack([X.ravel(), Y.ravel()], axis=1)
        solid = np.zeros(pts.shape[0], dtype=bool)
        for ob in self.scene.obstacles:
            solid |= ob.inside(pts)
        solid = solid.reshape(self.ny, self.nx).T  # 变成 (nx, ny)
        # 盒子外边界一圈固体（自由滑移墙）
        solid[0, :] = solid[-1, :] = True
        solid[:, 0] = solid[:, -1] = True
        self.solid = _dilate(solid)
        # 面掩膜：x 面 (nx+1, ny)，y 面 (nx, ny+1)；两邻单元都固体才算固体面
        self.solid_x = np.zeros((self.nx + 1, self.ny), dtype=bool)
        self.solid_x[1:-1, :] = self.solid[:-1, :] & self.solid[1:, :]
        self.solid_y = np.zeros((self.nx, self.ny + 1), dtype=bool)
        self.solid_y[:, 1:-1] = self.solid[:, :-1] & self.solid[:, 1:]
        self.fluid = ~self.solid

    def _init_fields(self):
        X, Y = self._xface_pos()
        pts = np.stack([X.ravel(), Y.ravel()], axis=1)
        v = self.scene.initial_velocity(pts)
        self.u_f = v[:, 0].reshape(self.nx + 1, self.ny)
        X, Y = self._yface_pos()
        pts = np.stack([X.ravel(), Y.ravel()], axis=1)
        v = self.scene.initial_velocity(pts)
        self.v_f = v[:, 1].reshape(self.nx, self.ny + 1)
        X, Y = self._cell_centers()
        pts = np.stack([X.ravel(), Y.ravel()], axis=1)
        self.c = self.scene.initial_concentration(pts).reshape(self.nx, self.ny)
        self._enforce_solid()
        self._sync_cell_caches()

    def _enforce_solid(self):
        """固体面上的法向速度 = 固体速度（无穿透）；浓度在固体单元置 0。"""
        solid_vel = np.asarray(self.scene.solid_velocity, dtype=np.float64)
        self.u_f[self.solid_x] = solid_vel[0]
        self.v_f[self.solid_y] = solid_vel[1]
        self.c[self.solid] = 0.0

    # ------------------------------------------------------------------ #
    # 速度插值（MAC 面 -> 任意点；三次 Catmull-Rom 减少数值耗散）
    # ------------------------------------------------------------------ #
    @staticmethod
    def _cr(t, f0, f1, f2, f3):
        t2 = t * t
        t3 = t2 * t
        return 0.5 * (2.0 * f1 + (-f0 + f2) * t
                      + (2.0 * f0 - 5.0 * f1 + 4.0 * f2 - f3) * t2
                      + (-f0 + 3.0 * f1 - 3.0 * f2 + f3) * t3)

    def _interp_x(self, field, pts):
        """u_f (nx+1, ny) 的双三次插值（CR）。"""
        x = pts[:, 0]; y = pts[:, 1]
        fi = x / self.hx + self.nx / 2.0
        fj = y / self.hy + (self.ny - 1) / 2.0
        i0 = np.clip(np.floor(fi).astype(int), 1, self.nx - 2)
        j0 = np.clip(np.floor(fj).astype(int), 1, self.ny - 3)
        tx = np.clip(fi - i0, 0, 1)
        ty = np.clip(fj - j0, 0, 1)
        f = field
        # x 方向三次（对每个 j 采样点）
        xw = [self._cr(tx, f[i0 - 1, j0 + d], f[i0, j0 + d], f[i0 + 1, j0 + d], f[i0 + 2, j0 + d])
              for d in (-1, 0, 1, 2)]
        # y 方向三次
        return self._cr(ty, xw[0], xw[1], xw[2], xw[3])

    def _interp_y(self, field, pts):
        """v_f (nx, ny+1) 的双三次插值（CR）。"""
        x = pts[:, 0]; y = pts[:, 1]
        fi = x / self.hx + (self.nx - 1) / 2.0
        fj = y / self.hy + self.ny / 2.0
        i0 = np.clip(np.floor(fi).astype(int), 1, self.nx - 3)
        j0 = np.clip(np.floor(fj).astype(int), 1, self.ny - 2)
        tx = np.clip(fi - i0, 0, 1)
        ty = np.clip(fj - j0, 0, 1)
        f = field
        xw = [self._cr(tx, f[i0 - 1, j0 + d], f[i0, j0 + d], f[i0 + 1, j0 + d], f[i0 + 2, j0 + d])
              for d in (-1, 0, 1, 2)]
        return self._cr(ty, xw[0], xw[1], xw[2], xw[3])

    def _interp_scalar(self, field, pts):
        """c (nx, ny) 的双三次插值（CR）。"""
        x = pts[:, 0]; y = pts[:, 1]
        fi = x / self.hx + (self.nx - 1) / 2.0
        fj = y / self.hy + (self.ny - 1) / 2.0
        i0 = np.clip(np.floor(fi).astype(int), 1, self.nx - 3)
        j0 = np.clip(np.floor(fj).astype(int), 1, self.ny - 3)
        tx = np.clip(fi - i0, 0, 1)
        ty = np.clip(fj - j0, 0, 1)
        f = field
        xw = [self._cr(tx, f[i0 - 1, j0 + d], f[i0, j0 + d], f[i0 + 1, j0 + d], f[i0 + 2, j0 + d])
              for d in (-1, 0, 1, 2)]
        return self._cr(ty, xw[0], xw[1], xw[2], xw[3])

    def _velocity_at(self, points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float64)
        u = self._interp_x(self.u_f, pts)
        v = self._interp_y(self.v_f, pts)
        return np.stack([u, v], axis=1)

    # ------------------------------------------------------------------ #
    # 平流（RK3 半拉格朗日，逐面）
    # ------------------------------------------------------------------ #
    def _backtrace(self, pos: np.ndarray, dt: float) -> np.ndarray:
        v1 = self._velocity_at(pos)
        p1 = pos - 0.5 * dt * v1
        v2 = self._velocity_at(p1)
        p2 = pos - dt * (2.0 * v2 - v1)
        v3 = self._velocity_at(p2)
        return pos - (dt / 6.0) * (v1 + 4.0 * v2 + v3)

    def _advect_face_velocities(self, dt: float):
        X, Y = self._xface_pos()
        u_pts = np.stack([X.ravel(), Y.ravel()], axis=1)
        X, Y = self._yface_pos()
        v_pts = np.stack([X.ravel(), Y.ravel()], axis=1)
        xu = self._backtrace(u_pts, dt)
        new_u = self._interp_x(self.u_f, xu).reshape(self.nx + 1, self.ny)
        xv = self._backtrace(v_pts, dt)
        new_v = self._interp_y(self.v_f, xv).reshape(self.nx, self.ny + 1)
        return new_u, new_v

    def _advect_scalar(self, dt: float):
        X, Y = self._cell_centers()
        c_pts = np.stack([X.ravel(), Y.ravel()], axis=1)
        xb = self._backtrace(c_pts, dt)
        return self._interp_scalar(self.c, xb).reshape(self.nx, self.ny)

    def _diffuse_faces(self, nu, dt):
        sigma_phys = np.sqrt(2.0 * nu * dt)
        from scipy import ndimage
        sg = (max(sigma_phys / self.hy, 1e-6), max(sigma_phys / self.hx, 1e-6))
        self.u_f = ndimage.gaussian_filter(self.u_f, sigma=sg, mode="reflect")
        self.v_f = ndimage.gaussian_filter(self.v_f, sigma=sg, mode="reflect")
        self.c = ndimage.gaussian_filter(self.c, sigma=sg, mode="reflect")

    # ------------------------------------------------------------------ #
    # 投影（MAC）
    # ------------------------------------------------------------------ #
    def _project(self):
        nx, ny = self.nx, self.ny
        hx, hy = self.hx, self.hy
        fluid = self.fluid
        div = np.zeros((nx, ny))
        div[fluid] = ((self.u_f[1:, :] - self.u_f[:-1, :]) / hx
                      + (self.v_f[:, 1:] - self.v_f[:, :-1]) / hy)[fluid]

        # 压力矩阵：标准 5 点，A·p = -∇²p；固体邻居贡献 0（Neumann）
        A = lil_matrix((nx * ny, nx * ny), dtype=np.float64)
        idx = np.arange(nx * ny).reshape(nx, ny)
        for di, dj, coef in [(1, 0, 1.0 / hx ** 2), (-1, 0, 1.0 / hx ** 2),
                             (0, 1, 1.0 / hy ** 2), (0, -1, 1.0 / hy ** 2)]:
            ni, nj = np.broadcast_arrays(
                np.clip(np.arange(nx)[:, None] + di, 0, nx - 1),
                np.clip(np.arange(ny)[None, :] + dj, 0, ny - 1))
            mask = fluid & fluid[ni, nj]
            rows = idx[mask]
            cols = idx[ni[mask], nj[mask]]
            for r, cc in zip(rows, cols):
                A[r, cc] = -coef
        diag_vals = np.zeros((nx, ny))
        for di, dj, coef in [(1, 0, 1.0 / hx ** 2), (-1, 0, 1.0 / hx ** 2),
                             (0, 1, 1.0 / hy ** 2), (0, -1, 1.0 / hy ** 2)]:
            ni, nj = np.broadcast_arrays(
                np.clip(np.arange(nx)[:, None] + di, 0, nx - 1),
                np.clip(np.arange(ny)[None, :] + dj, 0, ny - 1))
            fn = fluid[ni, nj]
            dd = np.zeros((nx, ny))
            dd[fluid] = fn[fluid] * coef
            diag_vals[fluid] += dd[fluid]
        A.setdiag(diag_vals.ravel())

        # 固体行单位阵；固定一个流体单元（消除 Neumann 常数零空间）
        A = A.tolil()
        fluid_flat = fluid.ravel()
        b = -div.ravel()
        for k in np.where(~fluid_flat)[0]:
            A[k, :] = 0
            A[k, k] = 1.0
        pin = int(np.argmax(fluid_flat))
        A[pin, :] = 0
        A[pin, pin] = 1.0
        b[pin] = 0.0
        A = A.tocsr()

        p, info = cg(A, b, rtol=1e-8, maxiter=1000)
        if info != 0:
            print(f"  [grid] PCG 未收敛 (info={info})")
        p = p.reshape(nx, ny)

        # 面压力梯度：u_f[i,j] -= (p[i,j]-p[i-1,j])/hx；v_f[i,j] -= (p[i,j]-p[i,j-1])/hy
        # 只修正"两侧都是流体"的面；流体-固体边界面不修正（保持固体速度=无穿透，
        # 对应 Neumann 条件 ∂p/∂n=0）
        fx = np.zeros((nx + 1, ny), dtype=bool)
        fx[1:-1, :] = fluid[:-1, :] & fluid[1:, :]
        self.u_f[fx] -= ((p[1:, :] - p[:-1, :]) / hx)[fx[1:-1, :]]
        fy = np.zeros((nx, ny + 1), dtype=bool)
        fy[:, 1:-1] = fluid[:, :-1] & fluid[:, 1:]
        self.v_f[fy] -= ((p[:, 1:] - p[:, :-1]) / hy)[fy[:, 1:-1]]
        self._enforce_solid()

    # ------------------------------------------------------------------ #
    # 主循环与输出
    # ------------------------------------------------------------------ #
    def step(self):
        scene = self.scene
        dt = scene.dt
        new_u, new_v = self._advect_face_velocities(dt)
        self.u_f, self.v_f = new_u, new_v
        self.c = self._advect_scalar(dt)
        if scene.viscosity > 0:
            self._diffuse_faces(scene.viscosity, dt)
        self._enforce_solid()  # 平流后先强制固体面无穿透
        if hasattr(scene, "apply_inlet_mac"):
            scene.apply_inlet_mac(self)  # 入流覆盖盒壁面速度
        self._project()
        self._sync_cell_caches()
        self.time += dt
        self.step_count += 1

    def _sync_cell_caches(self):
        """把 MAC 面速度合成 cell 中心速度，供渲染/对比使用。"""
        u_cell = 0.5 * (self.u_f[:-1, :] + self.u_f[1:, :])
        v_cell = 0.5 * (self.v_f[:, :-1] + self.v_f[:, 1:])
        self.u_cache = VelocityCache(self.nx, self.ny, (self.Lx, self.Ly))
        self.u_cache.u[...] = np.stack([u_cell, v_cell], axis=-1).transpose(1, 0, 2)
        self.c_cache = ScalarCache(self.nx, self.ny, (self.Lx, self.Ly))
        self.c_cache.q[...] = self.c.T

    def run(self, n_steps, save_every=0, out_dir=None, prefix="frame"):
        import os
        if save_every > 0 and out_dir:
            os.makedirs(out_dir, exist_ok=True)
        saved = []
        for _ in range(n_steps):
            self.step()
            if save_every > 0 and self.step_count % save_every == 0 and out_dir:
                path = os.path.join(out_dir, f"{prefix}_{self.step_count:05d}.npy")
                np.save(path, self.c)
                saved.append(path)
        return saved

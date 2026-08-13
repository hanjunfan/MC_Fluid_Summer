"""smoke3d.py —— 三维浮力烟求解器（论文 velocity_fluids_3d.cu 的速度法移植）。

场景：无边界盒域内，底部中心一个烟源盒持续注入浓度与温度；温度通过浮力
（Boussinesq 近似）驱动烟向上，速度平流 + 扩散 + MC 投影保持无散。

每步（对齐作者 velocity_fluids_3d.cu 顺序）：
    1) 平流（RK3 半拉格朗日）速度 / 浓度 / 温度
    2) 烟源注入（box source：浓度 + dt·rate，温度向 1 弛豫）
    3) 浮力（accel = (α·c − β·T)·gravity）
    4) 扩散（高斯卷积近似）
    5) 投影（3D 体积项 + 伪边界项，MC 估计）
"""

from __future__ import annotations

import numpy as np

from ..core3d.velocity_cache3d import VelocityCache3D, ScalarCache3D
from ..core3d.geometry3d import Box3D
from ..core3d.advection3d import advect_scalar, advect_vector
from ..core3d.diffusion3d import diffuse_scalar, diffuse_vector
from ..core3d.projection3d import project_batch_3d


class Smoke3DSolver:
    def __init__(self, grid_res=24, domain_size=3.0, dt=0.02,
                 num_volume_samples=150, num_pseudo_boundary_samples=150,
                 concentration_rate=2.0, temperature_rate=3.0,
                 buoyancy_alpha=0.0, buoyancy_beta=1.0, gravity=(0.0, -1.0, 0.0),
                 diffusion_sg_v=0.36, diffusion_sg_c=0.11, diffusion_sg_t=0.11,
                 projection_relax=0.3,
                 smoke_center=(0.0, -1.25, 0.0), smoke_half=(0.125, 0.125, 0.125),
                 seed=0, verbose=True):
        if np.isscalar(grid_res):
            grid_res = (int(grid_res), int(grid_res), int(grid_res))
        self.nx, self.ny, self.nz = (int(grid_res[0]), int(grid_res[1]), int(grid_res[2]))
        self.domain_size = float(domain_size) if np.isscalar(domain_size) else domain_size
        self.dt = float(dt)
        self.n_vol = int(num_volume_samples)
        self.n_pse = int(num_pseudo_boundary_samples)
        self.concentration_rate = float(concentration_rate)
        self.temperature_rate = float(temperature_rate)
        self.buoyancy_alpha = float(buoyancy_alpha)
        self.buoyancy_beta = float(buoyancy_beta)
        self.gravity = np.asarray(gravity, dtype=np.float64)
        self.sg_v = float(diffusion_sg_v)   # 速度扩散（按格数 sigma）
        self.sg_c = float(diffusion_sg_c)   # 浓度扩散（按格数 sigma）
        self.sg_t = float(diffusion_sg_t)   # 温度扩散（按格数 sigma）
        self.projection_relax = float(projection_relax)
        self.smoke_center = np.asarray(smoke_center, dtype=np.float64)
        self.smoke_half = np.asarray(smoke_half, dtype=np.float64)
        self.verbose = verbose
        self.rng = np.random.default_rng(seed)

        self.box = Box3D(self.domain_size)
        self.u_cache = VelocityCache3D(self.nx, self.ny, self.nz, self.domain_size)
        self.c_cache = ScalarCache3D(self.nx, self.ny, self.nz, self.domain_size)
        self.t_cache = ScalarCache3D(self.nx, self.ny, self.nz, self.domain_size)
        self.step_count = 0
        self.time = 0.0

    # ------------------------------------------------------------------ #
    def _smoke_mask(self, pts: np.ndarray) -> np.ndarray:
        rel = pts - self.smoke_center
        return ((np.abs(rel[:, 0]) <= self.smoke_half[0])
                & (np.abs(rel[:, 1]) <= self.smoke_half[1])
                & (np.abs(rel[:, 2]) <= self.smoke_half[2]))

    def _apply_source(self, c: np.ndarray, T: np.ndarray):
        """在烟源盒内注入浓度 + 温度（对网格节点直接作用）。"""
        pts = self.u_cache.grid_points()
        m = self._smoke_mask(pts).reshape(self.nz, self.ny, self.nx)
        c[m] = np.minimum(c[m] + self.dt * self.concentration_rate, 1.0)
        T[m] = T[m] + (1.0 - np.exp(-self.temperature_rate * self.dt)) * (1.0 - T[m])

    def _apply_buoyancy(self, u: np.ndarray, c: np.ndarray, T: np.ndarray):
        """Boussinesq 浮力：accel = (α·c − β·T)·g，作用于网格节点速度。"""
        if self.buoyancy_alpha == 0.0 and self.buoyancy_beta == 0.0:
            return
        accel = (self.buoyancy_alpha * c - self.buoyancy_beta * T)[..., None] * self.gravity
        u += self.dt * accel

    def _project(self):
        """对整个网格批量投影（分块，控制内存）。"""
        pts = self.u_cache.grid_points()

        def u_field(x):
            return self.u_cache.trilinear(x)

        chunk = 2048
        out = np.empty_like(self.u_cache.u.reshape(-1, 3))
        for s in range(0, pts.shape[0], chunk):
            sub = pts[s:s + chunk]
            out[s:s + chunk] = project_batch_3d(
                sub, u_field, self.box, self.n_vol, self.n_pse, self.rng,
                antithetic=True, relax=self.projection_relax)
        self.u_cache.u[...] = out.reshape(self.nz, self.ny, self.nx, 3)

    # ------------------------------------------------------------------ #
    def step(self):
        dt = self.dt
        u_adv = advect_vector(self.u_cache, self.u_cache, dt)
        c_adv = advect_scalar(self.c_cache, self.u_cache, dt)
        T_adv = advect_scalar(self.t_cache, self.u_cache, dt)

        self._apply_source(c_adv.q, T_adv.q)
        self._apply_buoyancy(u_adv.u, c_adv.q, T_adv.q)

        if self.sg_v > 0:
            u_adv.u[...] = diffuse_vector(u_adv.u, self.sg_v)
        if self.sg_c > 0:
            c_adv.q[...] = diffuse_scalar(c_adv.q, self.sg_c)
        if self.sg_t > 0:
            T_adv.q[...] = diffuse_scalar(T_adv.q, self.sg_t)

        self.u_cache.u[...] = u_adv.u
        self._project()
        self.c_cache.q[...] = c_adv.q
        self.t_cache.q[...] = T_adv.q

        self.step_count += 1
        self.time += dt

    def run(self, n_steps: int, save_every: int = 0, out_dir=None,
            umax_limit: float = 50.0):
        import os
        if save_every > 0 and out_dir:
            os.makedirs(out_dir, exist_ok=True)
        for _ in range(n_steps):
            self.step()
            # 发散保护：速度峰值超过阈值即停止（保留已保存帧）
            if float(np.abs(self.u_cache.u).max()) > umax_limit:
                if self.verbose:
                    print(f"  [发散保护] step={self.step_count} |u|max 超 {umax_limit}，提前停止")
                break
            if save_every > 0 and self.step_count % save_every == 0 and out_dir:
                path = os.path.join(out_dir, f"conc_{self.step_count:05d}.npy")
                np.save(path, self.c_cache.q)
                if self.verbose:
                    umax = float(np.abs(self.u_cache.u).max())
                    print(f"  t={self.time:.2f} step={self.step_count} 已保存 {path}  "
                          f"|u|max={umax:.3f}")

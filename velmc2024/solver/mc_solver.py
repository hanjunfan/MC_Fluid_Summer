"""
mc_solver.py —— 速度法蒙特卡洛主求解器（论文 §2 算子分裂 + caching 策略）

每时间步（论文 §2）：
    1) 平流   : RK3 半拉格朗日（速度 + 浓度同时平流）
    2) 外力   : 本实现两场景无外力
    3) 扩散   : 粘性场景用高斯卷积近似（§2.4）
    4) 投影   : VPL 缓存的 walk-on-boundary 蒙特卡洛投影（§2.3）

缓存策略（论文 Fig 3(b)）：投影后与平流后都缓存速度场。
"""

from __future__ import annotations

import numpy as np

from ..core.advection import advect_scalar, advect_vector
from ..core.diffusion import diffuse_scalar, diffuse_vector
from ..core.velocity_cache import VelocityCache, ScalarCache
from ..core.wob import (project_vpl_construct, project_grid_vpl_batched,
                        reconstruct_velocity_from_vorticity)


class MCFluidSolver:
    def __init__(self, scene, num_paths: int = 400, path_length: int = 4,
                 num_volume_samples_direct: int = 1000,
                 num_pseudo_boundary_samples_direct: int = 1000,
                 num_volume_samples_indirect: int = 10,
                 num_pseudo_boundary_samples_indirect: int = 10,
                 seed: int = 0, verbose: bool = True,
                 advect_vorticity: bool = True,
                 num_vorticity_samples: int = 400):
        self.scene = scene
        self.num_paths = int(num_paths)
        self.path_length = int(path_length)
        self.n_vol_dir = int(num_volume_samples_direct)
        self.n_pse_dir = int(num_pseudo_boundary_samples_direct)
        self.n_vol_ind = int(num_volume_samples_indirect)
        self.n_pse_ind = int(num_pseudo_boundary_samples_indirect)
        self.verbose = verbose
        self.advect_vorticity = bool(advect_vorticity)
        self.n_vort_samples = int(num_vorticity_samples)
        self.rng = np.random.default_rng(seed)

        nx, ny = scene.grid_res
        dom = scene.domain_size
        self.u_cache = VelocityCache(nx, ny, dom)
        self.c_cache = ScalarCache(nx, ny, dom)
        self.omega_cache = ScalarCache(nx, ny, dom)
        self.step_count = 0
        self.time = 0.0
        self._init_fields()

    # ------------------------------------------------------------------ #
    def _grid_points(self) -> np.ndarray:
        i = np.arange(self.u_cache.nx)
        j = np.arange(self.u_cache.ny)
        x, y = self.u_cache.idx_to_point(i[:, None], j[None, :])
        # y 主序：pts[k]=(x[i,j],y[i,j])，k=j*nx+i，使 reshape(ny,nx) 与缓存 [y,x] 读取约定一致
        return np.stack([x.T.ravel(), y.T.ravel()], axis=1)

    def _inside_mask(self, pts: np.ndarray) -> np.ndarray:
        w = np.zeros(pts.shape[0])
        for ob in self.scene.obstacles:
            w = w + ob.signed_winding(pts)
        return np.abs(w) >= 0.5

    def _init_fields(self):
        pts = self._grid_points()
        v = self.scene.initial_velocity(pts)
        c = self.scene.initial_concentration(pts)
        self.u_cache.u[...] = v.reshape(self.u_cache.ny, self.u_cache.nx, 2)
        self.c_cache.q[...] = c.reshape(self.u_cache.ny, self.u_cache.nx)
        if self.advect_vorticity:
            self._compute_initial_vorticity()
        self._apply_solid()

    def _compute_initial_vorticity(self):
        """从初始速度场算涡量 ω = ∂v/∂x - ∂u/∂y（中心差分），障碍内部置 0。"""
        u = self.u_cache.u  # (ny, nx, 2)：轴0=y，轴1=x
        hx = self.u_cache.Lx / self.u_cache.nx
        hy = self.u_cache.Ly / self.u_cache.ny
        dvdx = np.gradient(u[:, :, 1], hx, axis=1)   # ∂(uy)/∂x
        dudy = np.gradient(u[:, :, 0], hy, axis=0)   # ∂(ux)/∂y
        omega = dvdx - dudy
        pts = self._grid_points()
        inside = self._inside_mask(pts).reshape(self.u_cache.ny, self.u_cache.nx)
        omega[inside] = 0.0
        self.omega_cache.q[...] = omega

    def _apply_solid(self):
        """障碍内部速度置固体速度、浓度置 0。"""
        pts = self._grid_points()
        inside = self._inside_mask(pts)
        solid = np.asarray(self.scene.solid_velocity, dtype=np.float64)
        v = self.u_cache.u.reshape(-1, 2)
        v[inside] = solid
        self.c_cache.q.reshape(-1)[inside] = 0.0

    # ------------------------------------------------------------------ #
    def step(self):
        scene = self.scene
        dt = scene.dt
        solid = np.asarray(scene.solid_velocity, dtype=np.float64)

        if self.advect_vorticity:
            # ---- 涡量法（论文原法，保涡核）----
            # 1) 平流涡量标量场 ω（CR 下守恒好）
            omega_adv = advect_scalar(self.omega_cache, self.u_cache, dt, "RK3")
            self.omega_cache.q[...] = omega_adv.q
            # 2) 从涡量重建无散速度（Biot-Savart 体积项，保涡核强度）
            reconstruct_velocity_from_vorticity(
                self.omega_cache, self.u_cache, scene.obstacles, scene.sim_box,
                self.n_vort_samples, self.rng)
            # 3) 障碍内部速度置固体
            pts = self._grid_points()
            inside = self._inside_mask(pts)
            self.u_cache.u.reshape(-1, 2)[inside] = solid
            # 4) 平流浓度（用重建后的速度）
            c_adv = advect_scalar(self.c_cache, self.u_cache, dt, "RK3",
                                  interp=getattr(scene, "concentration_interp", "catmull_rom"))
        else:
            # ---- 旧法：平流速度场（粘性/入流场景用）----
            u_adv = advect_vector(self.u_cache, self.u_cache, dt, "RK3")
            c_adv = advect_scalar(self.c_cache, self.u_cache, dt, "RK3",
                                  interp=getattr(scene, "concentration_interp", "catmull_rom"))
            if scene.viscosity > 0:
                u_adv = diffuse_vector(u_adv, scene.viscosity, dt)
                c_adv = diffuse_scalar(c_adv, scene.viscosity, dt)
            if hasattr(scene, "apply_inlet"):
                scene.apply_inlet(u_adv, c_adv)
            pts = self._grid_points()
            inside = self._inside_mask(pts)
            u_adv.u.reshape(-1, 2)[inside] = solid
            self.u_cache.u[...] = u_adv.u

        self.c_cache.q[...] = c_adv.q.reshape(self.c_cache.ny, self.c_cache.nx)
        # 物理上浓度非负的场景（如 Karman 入流条带）钳掉 Catmull-Rom 过冲负值；
        # 两团场景（cohomology/cylinder，红+蓝-）浓度允许为负，不能钳制。
        if getattr(scene, "concentration_nonnegative", False):
            np.clip(self.c_cache.q, 0.0, None, out=self.c_cache.q)

        # 投影（VPL 共享）：强制障碍/盒子边界无穿透，保持涡量
        vpl_pos, vpl_val = project_vpl_construct(
            scene.obstacles, self.u_cache.bilinear, scene.sim_box,
            self.num_paths, self.path_length,
            self.n_vol_ind, self.n_pse_ind, self.rng, scene.solid_velocity)
        project_grid_vpl_batched(
            self.u_cache, scene.obstacles, scene.sim_box,
            vpl_pos, vpl_val, self.num_paths, self.path_length,
            self.n_vol_dir, self.n_pse_dir, self.rng, scene.solid_velocity,
            use_box_boundary=getattr(scene, "use_box_boundary", True),
            relax=getattr(scene, "projection_relax", 1.0))

        self._apply_solid()
        self.step_count += 1
        self.time += dt

    # ------------------------------------------------------------------ #
    def run(self, n_steps: int, save_every: int = 0, out_dir=None, prefix="frame",
            save_velocity: bool = True):
        """跑 n_steps 步；save_every>0 时周期性保存浓度场（.npy）。"""
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
                if save_velocity:
                    vpath = os.path.join(out_dir, f"vel_{self.step_count:05d}.npy")
                    np.save(vpath, self.u_cache.u)
                if self.verbose:
                    print(f"  t={self.time:.2f} step={self.step_count} 已保存 {path}")
        return saved

"""advection3d.py —— 三维 RK3 半拉格朗日平流（三线性插值）。

RK3（Ralston 三阶，对齐作者）：
    v1 = u(x)
    v2 = u(x - (dt/2) v1)
    v3 = u(x - dt (2 v2 - v1))
    x_prev = x - (dt/6)(v1 + 4 v2 + v3)
"""

from __future__ import annotations

import numpy as np

from .velocity_cache3d import VelocityCache3D, ScalarCache3D


def backtrace(points: np.ndarray, u_cache: VelocityCache3D, dt: float,
              mode: str = "RK3") -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    if mode == "Euler":
        return pts - dt * u_cache.trilinear(pts)
    v1 = u_cache.trilinear(pts)
    x1 = pts - 0.5 * dt * v1
    v2 = u_cache.trilinear(x1)
    x2 = pts - dt * (2.0 * v2 - v1)
    v3 = u_cache.trilinear(x2)
    return pts - (dt / 6.0) * (v1 + 4.0 * v2 + v3)


def advect_scalar(src: ScalarCache3D, u_cache: VelocityCache3D, dt: float,
                  mode: str = "RK3") -> ScalarCache3D:
    dst = ScalarCache3D(src.nx, src.ny, src.nz, src.L)
    grid_pts = _grid_points(src)
    x_prev = backtrace(grid_pts, u_cache, dt, mode)
    dst.q[...] = src.trilinear(x_prev).reshape(src.nz, src.ny, src.nx)
    return dst


def advect_vector(src: VelocityCache3D, u_cache: VelocityCache3D, dt: float,
                  mode: str = "RK3") -> VelocityCache3D:
    dst = VelocityCache3D(src.nx, src.ny, src.nz, src.L)
    grid_pts = _grid_points(src)
    x_prev = backtrace(grid_pts, u_cache, dt, mode)
    dst.u[...] = src.trilinear(x_prev).reshape(src.nz, src.ny, src.nx, 3)
    return dst


def _grid_points(cache) -> np.ndarray:
    """按缓存展平顺序（iz 最快）返回网格节点坐标。"""
    iz, iy, ix = np.meshgrid(np.arange(cache.nz), np.arange(cache.ny),
                             np.arange(cache.nx), indexing="ij")
    x = (ix - (cache.nx - 1) / 2.0) * (cache.Lx / cache.nx)
    y = (iy - (cache.ny - 1) / 2.0) * (cache.Ly / cache.ny)
    z = (iz - (cache.nz - 1) / 2.0) * (cache.Lz / cache.nz)
    return np.stack([x, y, z], axis=-1).reshape(-1, 3)

"""
advection.py —— 半拉格朗日平流（论文 §2.1，作者 advection_mode=RK3）

对每个网格求值点 x 反向追踪轨迹到 x_prev，然后 new_field(x) = old_field(x_prev)。
速度场用双线性插值（点态查询）。支持 Euler / RK3。

RK3（Ralston 三阶，作者使用）：
    v1 = u(x)
    v2 = u(x - (dt/2) v1)
    v3 = u(x - dt (2 v2 - v1))
    x_prev = x - (dt/6)(v1 + 4 v2 + v3)
"""

from __future__ import annotations

import numpy as np

from .velocity_cache import VelocityCache, ScalarCache


def backtrace(points: np.ndarray, u_cache: VelocityCache, dt: float,
              mode: str = "RK3") -> np.ndarray:
    """对 points (P,2) 反向追踪到上一时间步位置 (P,2)。

    速度用三次 Catmull-Rom 插值（比双线性耗散小很多），是降低半拉格朗日
    数值耗散、保持涡对/尖峰的关键。
    """
    pts = np.asarray(points, dtype=np.float64)
    if mode == "Euler":
        v1 = u_cache.catmull_rom(pts)
        return pts - dt * v1
    # RK3
    v1 = u_cache.catmull_rom(pts)
    x1 = pts - 0.5 * dt * v1
    v2 = u_cache.catmull_rom(x1)
    x2 = pts - dt * (2.0 * v2 - v1)
    v3 = u_cache.catmull_rom(x2)
    return pts - (dt / 6.0) * (v1 + 4.0 * v2 + v3)


def advect_scalar(src: ScalarCache, u_cache: VelocityCache, dt: float,
                  mode: str = "RK3", interp: str = "catmull_rom") -> ScalarCache:
    """平流标量场（浓度/温度），返回新缓存。

    interp: "catmull_rom"（默认，低耗散、保形好）或 "bilinear"（ClampToEdge 正确、
    无过冲，适合尖锐 0/1 条带如 Karman 入流浓度；CR 对尖锐阶跃会过冲出负值，
    且边界 clip 会让入口条带粘在边界无法流入内部）。
    """
    dst = ScalarCache(src.nx, src.ny, src.L)
    i = np.arange(src.nx)
    j = np.arange(src.ny)
    x, y = src.idx_to_point(i[:, None], j[None, :])
    # y 主序：pts[k]=(x[i,j],y[i,j])，k=j*nx+i，使 reshape(ny,nx) 与缓存 [y,x] 读取约定一致
    grid_pts = np.stack([x.T.ravel(), y.T.ravel()], axis=1)
    x_prev = backtrace(grid_pts, u_cache, dt, mode)
    if interp == "bilinear":
        dst.q[...] = src.bilinear(x_prev).reshape(src.ny, src.nx)
    else:
        dst.q[...] = src.catmull_rom(x_prev).reshape(src.ny, src.nx)
    return dst


def advect_vector(src: VelocityCache, u_cache: VelocityCache, dt: float,
                  mode: str = "RK3") -> VelocityCache:
    """平流速度场，返回新缓存。"""
    dst = VelocityCache(src.nx, src.ny, src.L)
    i = np.arange(src.nx)
    j = np.arange(src.ny)
    x, y = src.idx_to_point(i[:, None], j[None, :])
    # y 主序：pts[k]=(x[i,j],y[i,j])，k=j*nx+i，使 reshape(ny,nx,2) 与缓存 [y,x] 读取约定一致
    grid_pts = np.stack([x.T.ravel(), y.T.ravel()], axis=1)
    x_prev = backtrace(grid_pts, u_cache, dt, mode)
    dst.u[...] = src.catmull_rom(x_prev).reshape(src.ny, src.nx, 2)
    return dst

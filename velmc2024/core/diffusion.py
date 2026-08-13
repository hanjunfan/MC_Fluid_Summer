"""
diffusion.py —— 扩散步（论文 §2.4）

粘性流体：∂u/∂t = ν∇²u。无边界/小时间步时，高斯卷积是精确解：
    u(x, dt) = ∫ u0(y) G(x - y; 2νdt) dy,  G 为高斯核。

有障碍物时边界效应需 time-dependent walk-on-boundary（作者完整实现）。
本模块先用高斯卷积近似（无边界精确、含边界为近似），在报告中说明该简化。
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from .velocity_cache import VelocityCache, ScalarCache


def diffuse_scalar(src: ScalarCache, nu: float, dt: float) -> ScalarCache:
    """标量场高斯卷积扩散。"""
    dst = ScalarCache(src.nx, src.ny, src.L)
    sigma_phys = np.sqrt(2.0 * nu * dt)
    dx = src.Lx / src.nx
    dy = src.Ly / src.ny
    sigma_g = (max(sigma_phys / dy, 1e-6), max(sigma_phys / dx, 1e-6))  # (ny, nx) 轴序
    dst.q[...] = ndimage.gaussian_filter(src.q, sigma=sigma_g, mode="reflect")
    return dst


def diffuse_vector(src: VelocityCache, nu: float, dt: float) -> VelocityCache:
    """速度场（逐分量）高斯卷积扩散。"""
    dst = VelocityCache(src.nx, src.ny, src.L)
    sigma_phys = np.sqrt(2.0 * nu * dt)
    dx = src.Lx / src.nx
    dy = src.Ly / src.ny
    sigma_g = (max(sigma_phys / dy, 1e-6), max(sigma_phys / dx, 1e-6))
    for c in range(2):
        dst.u[..., c] = ndimage.gaussian_filter(src.u[..., c], sigma=sigma_g, mode="reflect")
    return dst

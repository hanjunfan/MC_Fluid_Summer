"""diffusion3d.py —— 三维扩散（高斯卷积近似）。

sigma 直接以"格数"给出（sigma_phys / h），保证不同网格分辨率下耗散强度一致。
论文原法用 WoB 扩散；3D 版本此处沿用高斯卷积近似以降低实现复杂度。
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def diffuse_vector(u: np.ndarray, sg: float) -> np.ndarray:
    """对速度场 (nz, ny, nx, 3) 做高斯卷积扩散（各分量独立，reflect 边界）。

    sg 为按格数的高斯 sigma。
    """
    if sg <= 0:
        return u
    out = u.copy()
    for c in range(3):
        out[..., c] = ndimage.gaussian_filter(u[..., c], sigma=sg, mode="reflect")
    return out


def diffuse_scalar(q: np.ndarray, sg: float) -> np.ndarray:
    """对标量场 (nz, ny, nx) 做高斯卷积扩散。sg 为按格数的高斯 sigma。"""
    if sg <= 0:
        return q
    return ndimage.gaussian_filter(q, sigma=sg, mode="reflect")


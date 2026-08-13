"""
sampling.py —— 蒙特卡洛采样原语（对齐作者 velocity_fluids_optix.cu）

- strongly_singular_ball_sample: 以 PDF ∝ 1/r^(d-1)（2D 为 1/r）在包围球内采样，
  用于投影的体积项（处理 Hessian 核的奇异积分）。2D 下 r 均匀、角均匀，
  inv_pdf = 2π R r。
- antithetic: 附加镜像点 2x - y（即 -r_vec），减小方差。
- uniform_boundary_sample: 在几何边界上按边长/弧长均匀采样，inv_pdf = 周长/弧长。
"""

from __future__ import annotations

import numpy as np


def strongly_singular_ball_sample(sampling_radius: float, n: int,
                                  rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """在半径 R 的圆盘内按 PDF ∝ 1/r 采样。

    2D: r = R * u（均匀），θ 均匀。
    p(r,θ) = 1/(2π R r)，inv_pdf = 2π R r。

    返回 (r_vec (n,2), inv_pdf (n,))。
    """
    R = float(sampling_radius)
    u = rng.random(n)
    theta = rng.uniform(0.0, 2.0 * np.pi, n)
    r = R * u
    r_vec = r[:, None] * np.stack([np.cos(theta), np.sin(theta)], axis=1)
    inv_pdf = 2.0 * np.pi * R * r
    return r_vec, inv_pdf


def antithetic_double(r_vec: np.ndarray, inv_pdf: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """对每个样本附加其关于原点的镜像点（2x - y => r_vec -> -r_vec）。

    输入 n 个样本，输出 2n 个（镜像点与其 inv_pdf 相同）。用于体积项方差缩减。
    """
    r_vec2 = np.concatenate([r_vec, -r_vec], axis=0)
    inv_pdf2 = np.concatenate([inv_pdf, inv_pdf], axis=0)
    return r_vec2, inv_pdf2


def uniform_dir_2d(n: int, rng: np.random.Generator) -> np.ndarray:
    """均匀随机方向（单位向量）。"""
    theta = rng.uniform(0.0, 2.0 * np.pi, n)
    return np.stack([np.cos(theta), np.sin(theta)], axis=1)


def uniform_boundary_sample(geometry, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, float]:
    """在 geometry（Polygon / Rectangle / Circle）边界上均匀采样。

    返回 (points (n,2), normals (n,2), inv_pdf=周长)。
    """
    pts, norms = geometry.sample_boundary(n, rng)
    return pts, norms, geometry.perimeter

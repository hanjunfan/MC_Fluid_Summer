"""sampling3d.py —— 三维蒙特卡洛采样原语。

- strongly_singular_ball_sample_3d: 在半径 R 的球内按 PDF ∝ 1/r 采样
  （处理 3D Hessian 核 1/r³ 的奇异积分）。r = R·sqrt(u)，方向均匀球面，
  inv_pdf = 2π R² r。
- antithetic_double: 附加镜像点 -r_vec 减小方差。
"""

from __future__ import annotations

import numpy as np


def strongly_singular_ball_sample_3d(sampling_radius: float, n: int,
                                     rng: np.random.Generator):
    """球内 PDF ∝ 1/r 采样。返回 (r_vec (n,3), inv_pdf (n,))。

    p(r) ∝ 1/r（体积密度 p(r)=1/(2πR²r)），CDF(r)=r²/R²，故 r=R·sqrt(u)。
    inv_pdf = 2π R² r。
    """
    R = float(sampling_radius)
    u = rng.random(n)
    r = R * np.sqrt(u)
    z = rng.uniform(-1.0, 1.0, n)
    phi = rng.uniform(0.0, 2.0 * np.pi, n)
    s = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    r_vec = np.stack([s * np.cos(phi), s * np.sin(phi), z], axis=1) * r[:, None]
    inv_pdf = 2.0 * np.pi * R * R * r
    return r_vec, inv_pdf


def antithetic_double(r_vec: np.ndarray, inv_pdf: np.ndarray):
    """对每个样本附加关于原点的镜像点（r_vec -> -r_vec），输出 2n 个。"""
    r_vec2 = np.concatenate([r_vec, -r_vec], axis=0)
    inv_pdf2 = np.concatenate([inv_pdf, inv_pdf], axis=0)
    return r_vec2, inv_pdf2

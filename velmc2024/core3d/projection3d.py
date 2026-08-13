"""projection3d.py —— 三维压力投影的蒙特卡洛估计器（论文 §2.3 的 3D 版）。

无边界域投影：
    ∇p(x) = -∫_{Ωs} S(x,y){u(y)-u(x)} dV
            -∫_{∂Ωs} ∇xG(x,y) n·{u(y)-u(x)} dA

3D 基本解与核（r_vec = y - x, r = |r_vec|）：
    G(x,y)   = 1/(4π r)
    ∇xG      = r_vec / (4π r³)          = r̂ / (4π r²)
    S(x,y)   = (3 r̂ r̂ᵀ - I) / (4π r³)    （Hessian 主部）

估计器：
    ⟨E_V⟩ = (1/N_V) Σ S(x,y_i) · {u(y_i)-u(x)} · inv_pdf_V
            P_V ∝ 1/r，inv_pdf_V = 2π R² r   （R = x 到盒最远角）
    ⟨E_A⟩ = (1/N_A) Σ ∇xG(x,y_j) · n·{u(y_j)-u(x)} · area_box
            P_A 均匀（盒面），inv_pdf_A = 盒总面积

全局速度平移 {u(y)-u(x)} 消去 1/r³ 奇异主项（论文 Eq.11 关键技巧）。
"""

from __future__ import annotations

import numpy as np


def project_batch_3d(points: np.ndarray, u_field, box,
                     num_volume_samples: int, num_pseudo_boundary_samples: int,
                     rng: np.random.Generator,
                     antithetic: bool = True,
                     dGdx_reg: float = 1e-5,
                     relax: float = 1.0) -> np.ndarray:
    """对 points (P,3) 批量投影，返回投影后的速度 (P,3)。

    u_field(y: (M,3)) -> (M,3) 为点态速度查询（绑定到缓存的三线性插值）。
    relax < 1 时对压力梯度修正做阻尼（低样本数下抑制 MC 噪声正反馈发散）。
    """
    pts = np.asarray(points, dtype=np.float64)
    P = pts.shape[0]
    u_x = u_field(pts)  # (P,3)
    R_p = np.array([box.max_corner_distance(p) for p in pts])  # (P,)

    # ---------------- 体积项 ---------------- #
    n_draw = max(1, num_volume_samples // 2) if antithetic else max(1, num_volume_samples)
    ur = rng.random((P, n_draw))
    r = R_p[:, None] * np.sqrt(ur)                       # 3D: r=R*sqrt(u)
    zz = rng.uniform(-1.0, 1.0, (P, n_draw))
    phi = rng.uniform(0.0, 2.0 * np.pi, (P, n_draw))
    s = np.sqrt(np.maximum(1.0 - zz * zz, 0.0))
    direction = np.stack([s * np.cos(phi), s * np.sin(phi), zz], axis=-1)  # (P,S,3)
    r_vec = direction * r[..., None]                     # (P,S,3)
    inv_pdf = 2.0 * np.pi * R_p[:, None] ** 2 * r        # (P,S)
    if antithetic:
        r_vec = np.concatenate([r_vec, -r_vec], axis=1)
        inv_pdf = np.concatenate([inv_pdf, inv_pdf], axis=1)
        S = n_draw * 2
    else:
        S = n_draw
    n_eff = max(num_volume_samples, 1)

    y = pts[:, None, :] + r_vec                          # (P,S,3)
    vel_y = u_field(y.reshape(-1, 3)).reshape(P, S, 3)
    vel_diff = vel_y - u_x[:, None, :]                   # (P,S,3)
    in_box = box.contains(y.reshape(-1, 3)).reshape(P, S)
    rr = np.maximum(np.linalg.norm(r_vec, axis=-1), 1e-12)
    r_hat = r_vec / rr[..., None]
    dot_ru = np.einsum("pij,pij->pi", r_hat, vel_diff)
    kernel = 3.0 * dot_ru[..., None] * r_hat - vel_diff  # 3 r̂(r̂·Δu) - Δu
    # ev = (1/N) Σ kernel · inv_pdf/(4π r³) = (1/N) Σ kernel · R²/(2 r²)
    contrib = in_box[..., None] * (R_p[:, None] ** 2 / (2.0 * rr * rr))[..., None] * kernel
    ev = contrib.sum(axis=1) / n_eff                     # (P,3)

    # ---------------- 伪边界项 ---------------- #
    ea = np.zeros((P, 3))
    if num_pseudo_boundary_samples > 0:
        bpts, bnorms = box.sample_boundary(num_pseudo_boundary_samples, rng)  # (B,3)
        vel_b = u_field(bpts)                            # (B,3)
        # ndot[p,b] = n_b · (u(b) - u(x_p))
        ndot = (np.einsum("ij,ij->i", bnorms, vel_b))[None, :] \
            - np.einsum("ij,pj->pi", bnorms, u_x)        # (P,B)
        r_vec_all = bpts[None, :, :] - pts[:, None, :]   # (P,B,3)
        r_all = np.maximum(np.linalg.norm(r_vec_all, axis=-1), dGdx_reg)
        g = r_vec_all / (4.0 * np.pi * (r_all ** 3))[..., None]  # ∇xG (P,B,3)
        ea = (box.area / num_pseudo_boundary_samples) \
            * np.einsum("pb,pbi->pi", ndot, g)           # (P,3)

    pressure_grad = -relax * (ev + ea)
    return u_x - pressure_grad

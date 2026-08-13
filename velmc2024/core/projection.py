"""
projection.py —— 压力投影的蒙特卡洛估计器（论文 Eq.11-14，对齐 velocity_fluids_optix.cu）

投影目标：对带散度的速度场 u3，找到 ∇p 使 u4 = u3 - ∇p 无散度（∇·u4 = 0）。

无边界情形（UnboundedDomain，或无障碍时）：
    ∇p(x) = -∫_{Ωs} S(x,y){u(y)-u(x)} dV
            -∫_{∂Ωs} ∇xG(x,y) n(y)·{u(y)-u(x)} dA
其中
    S(x,y) = (1/(2π r²)) (2 r̂ r̂ᵀ - I)        （2D Hessian 主部）
    G(x,y) = -(1/2π) log r                     （2D 基本解）
    ∇xG    = r̂ / (2π r)

估计器：
    ⟨E_V⟩ = -(1/N_V) Σ S(x,y_i)/P_V · {u(y_i)-u(x)}      P_V ∝ 1/r（重要性采样 + antithetic）
    ⟨E_A⟩ = -(1/N_A) Σ ∇xG(x,y_j)/P_A · n·{u(y_j)-u(x)}  P_A 均匀（inv_pdf=周长）

全局速度平移：用 {u(y)-u(x)} 消去奇异主项（Eq.11 的关键）。

有障碍物时的 WoB 边界积分项在 phase 2 加入（obstacles 参数暂为 []）。
"""

from __future__ import annotations

import numpy as np

from .sampling import strongly_singular_ball_sample, antithetic_double, uniform_boundary_sample


def dGdx_2d(r_vec: np.ndarray, regularization: float = 1e-5) -> np.ndarray:
    """2D 基本解的梯度 ∇xG(x,y) = (y-x)/|y-x|² · (1/2π)，|r| 用 regularization 截断。

    支持 (..., 2) 批量输入或 (2,) 单点输入（返回同形状）。
    """
    r_vec = np.asarray(r_vec, dtype=np.float64)
    single = r_vec.ndim == 1
    if single:
        r_vec = r_vec[None, :]
    r = np.linalg.norm(r_vec, axis=-1)
    r = np.maximum(r, regularization)
    out = r_vec / r[..., None] / (2.0 * np.pi * r[..., None])
    if single:
        return out[0]
    return out


def estimate_volume_term(x: np.ndarray, x_vel: np.ndarray, get_velocity,
                         sim_box, obstacles, num_samples: int,
                         rng: np.random.Generator,
                         antithetic: bool = True,
                         dGdx_reg: float = 1e-5) -> np.ndarray:
    """体积项 ⟨E_V⟩：2D 下采样半径 = x 到仿真盒最远角距离。

    返回 (2,) 向量（未加负号，调用方统一取负）。
    """
    R = sim_box.max_corner_distance(x)
    n_draw = num_samples // 2 if antithetic else num_samples
    r_vec, inv_pdf = strongly_singular_ball_sample(R, n_draw, rng)
    if antithetic:
        r_vec, inv_pdf = antithetic_double(r_vec, inv_pdf)
        n_contrib = n_draw * 2
    else:
        n_contrib = n_draw
    # 参与计数 = num_samples（antithetic 时 n_draw=num//2）
    n_eff = num_samples

    y = x[None, :] + r_vec
    # 乘子：在仿真盒内 且 不在障碍内
    inside_box = sim_box.contains(y).astype(np.float64)
    if obstacles:
        w = np.zeros(y.shape[0])
        for ob in obstacles:
            w = w + ob.signed_winding(y)
        multiplier = inside_box * (1.0 - (np.abs(w) >= 0.5))
    else:
        multiplier = inside_box

    vel_diff = get_velocity(y) - x_vel[None, :]  # (n,2)
    r = np.linalg.norm(r_vec, axis=1)
    r = np.maximum(r, 1e-12)
    r_hat = r_vec / r[:, None]
    inv_pdf_div = inv_pdf / (r * r)  # inv_pdf / |r|²
    # 2*(r̂·Δu)*r̂ - Δu
    dot_ru = np.einsum("ij,ij->i", r_hat, vel_diff)
    kernel = 2.0 * dot_ru[:, None] * r_hat - vel_diff
    contrib = multiplier[:, None] * inv_pdf_div[:, None] * kernel
    total = contrib.sum(axis=0)
    return total / (n_eff * 2.0 * np.pi)


def estimate_pseudo_boundary_term(x: np.ndarray, x_vel: np.ndarray, get_velocity,
                                  sim_box, obstacles, num_samples: int,
                                  rng: np.random.Generator,
                                  dGdx_reg: float = 1e-5) -> np.ndarray:
    """伪边界项 ⟨E_A⟩（无边界域仿真盒边界上的积分）。返回 (2,)。"""
    pts, norms, inv_pdf = uniform_boundary_sample(sim_box, num_samples, rng)
    # 伪边界采样点如果在障碍内则贡献为 0
    if obstacles:
        w = np.zeros(pts.shape[0])
        for ob in obstacles:
            w = w + ob.signed_winding(pts)
        mult = (np.abs(w) < 0.5).astype(np.float64)
    else:
        mult = np.ones(pts.shape[0])
    vel = get_velocity(pts) - x_vel[None, :]
    n_dot = np.einsum("ij,ij->i", norms, vel)
    g = dGdx_2d(pts - x[None, :], dGdx_reg)  # (n,2)
    contrib = mult[:, None] * inv_pdf * n_dot[:, None] * g
    return contrib.sum(axis=0) / num_samples


def project_pointwise(x: np.ndarray, get_velocity, sim_box, obstacles,
                      num_volume_samples: int, num_pseudo_boundary_samples: int,
                      rng: np.random.Generator,
                      antithetic: bool = True, dGdx_reg: float = 1e-5) -> np.ndarray:
    """单点投影：返回 u4(x)。无障碍版本（WoB 边界项在 phase 2）。"""
    x = np.asarray(x, dtype=np.float64)
    x_vel = get_velocity(x[None, :])[0]
    ev = estimate_volume_term(x, x_vel, get_velocity, sim_box, obstacles,
                              num_volume_samples, rng, antithetic, dGdx_reg)
    ea = estimate_pseudo_boundary_term(x, x_vel, get_velocity, sim_box, obstacles,
                                       num_pseudo_boundary_samples, rng, dGdx_reg)
    pressure_grad = -(ev + ea)
    return x_vel - pressure_grad


def project_grid(cache, sim_box, obstacles,
                 num_volume_samples: int, num_pseudo_boundary_samples: int,
                 rng: np.random.Generator, antithetic: bool = True,
                 dGdx_reg: float = 1e-5, solid_velocity=(0.0, 0.0)) -> None:
    """在整个网格缓存上执行投影（就地更新 cache.u）。

    get_velocity 闭包绑定到 cache 本身（双线性插值），与作者 get_velocity 一致。
    """
    nx, ny = cache.nx, cache.ny
    solid = np.asarray(solid_velocity, dtype=np.float64)

    def get_velocity(pts):
        return cache.bilinear(pts)

    # 对每个网格节点投影
    i = np.arange(nx)
    j = np.arange(ny)
    x, y = cache.idx_to_point(i[:, None], j[None, :])
    grid_pts = np.stack([x.ravel(), y.ravel()], axis=1)  # (nx*ny, 2)
    n_pts = grid_pts.shape[0]

    out = np.empty((n_pts, 2))
    chunk = max(1, int(2_000_000 / max(num_volume_samples, 1)))
    for s in range(0, n_pts, chunk):
        idx = slice(s, s + chunk)
        sub = grid_pts[idx]
        for k in range(sub.shape[0]):
            out[s + k] = project_pointwise(
                sub[k], get_velocity, sim_box, obstacles,
                num_volume_samples, num_pseudo_boundary_samples,
                rng, antithetic, dGdx_reg)

    # 障碍内部赋固体速度
    if obstacles:
        w = np.zeros(n_pts)
        for ob in obstacles:
            w = w + ob.signed_winding(grid_pts)
        inside = np.abs(w) >= 0.5
        out[inside] = solid

    cache.u[...] = out.reshape(ny, nx, 2)


# --------------------------------------------------------------------------- #
# 批量向量化版本：P 个求值点 × N 个样本全部向量化（CPU 性能关键）
# --------------------------------------------------------------------------- #
def project_batch_vectorized(points: np.ndarray, u_field,
                             sim_box, obstacles,
                             num_volume_samples: int, num_pseudo_boundary_samples: int,
                             rng: np.random.Generator,
                             antithetic: bool = True,
                             dGdx_reg: float = 1e-5) -> np.ndarray:
    """对 points (P,2) 批量投影。u_field(y: (M,2)) -> (M,2) 为点态速度查询（分析场或缓存）。

    返回 (P,2) 投影后的速度 u4。无边界版本（障碍物仅用于乘子截断，WoB 项 phase2）。
    """
    pts = np.asarray(points, dtype=np.float64)
    P = pts.shape[0]
    u_x = u_field(pts)  # (P,2)
    R_p = np.array([sim_box.max_corner_distance(p) for p in pts])  # (P,)

    # ---------------- 体积项 ---------------- #
    n_draw = max(1, num_volume_samples // 2) if antithetic else max(1, num_volume_samples)
    u_r = rng.random((P, n_draw))                       # 均匀半径系数
    th = rng.uniform(0, 2 * np.pi, (P, n_draw))
    r = R_p[:, None] * u_r                              # (P,S)
    r_vec = np.stack([r * np.cos(th), r * np.sin(th)], axis=-1)  # (P,S,2)
    inv_pdf = 2 * np.pi * R_p[:, None] * r              # (P,S)
    if antithetic:
        r_vec = np.concatenate([r_vec, -r_vec], axis=1)
        inv_pdf = np.concatenate([inv_pdf, inv_pdf], axis=1)
        th = np.concatenate([th, th + np.pi], axis=1)
        S = n_draw * 2
    else:
        S = n_draw
    n_eff = max(num_volume_samples, 1)

    y = pts[:, None, :] + r_vec                          # (P,S,2)
    vel_y = u_field(y.reshape(-1, 2)).reshape(P, S, 2)   # 查询所有样本
    vel_diff = vel_y - u_x[:, None, :]                   # (P,S,2)
    # 乘子：盒内 & 障碍外
    in_box = sim_box.contains(y.reshape(-1, 2)).reshape(P, S)
    if obstacles:
        w = np.zeros((P, S))
        for ob in obstacles:
            w = w + ob.signed_winding(y.reshape(-1, 2)).reshape(P, S)
        mult = in_box * (np.abs(w) < 0.5)
    else:
        mult = in_box
    rr = np.maximum(np.linalg.norm(r_vec, axis=-1), 1e-12)   # (P,S)
    r_hat = r_vec / rr[..., None]
    dot_ru = np.einsum("pij,pij->pi", r_hat, vel_diff)
    kernel = 2.0 * dot_ru[..., None] * r_hat - vel_diff
    contrib = mult[..., None] * (inv_pdf / (rr * rr))[..., None] * kernel
    ev = contrib.sum(axis=1) / (n_eff * 2.0 * np.pi)          # (P,2)

    # ---------------- 伪边界项 ---------------- #
    ea = np.zeros((P, 2))
    if num_pseudo_boundary_samples > 0:
        bpts, bnorms, b_inv = uniform_boundary_sample(sim_box, num_pseudo_boundary_samples, rng)
        if obstacles:
            wb = np.zeros(bpts.shape[0])
            for ob in obstacles:
                wb = wb + ob.signed_winding(bpts)
            bmult = (np.abs(wb) < 0.5).astype(np.float64)
        else:
            bmult = np.ones(bpts.shape[0])
        vel_b = u_field(bpts)  # (B,2)
        # 对每个求值点求和
        # dGdx(bpts - x_p) * inv_pdf * n·(u(bpts)-u(x_p))
        for p in range(P):
            g = dGdx_2d(bpts - pts[p][None, :], dGdx_reg)          # (B,2)
            ndot = np.einsum("ij,ij->i", bnorms, vel_b - u_x[p][None, :])
            ea[p] = np.sum((bmult * b_inv * ndot)[:, None] * g, axis=0) / num_pseudo_boundary_samples

    pressure_grad = -(ev + ea)
    return u_x - pressure_grad

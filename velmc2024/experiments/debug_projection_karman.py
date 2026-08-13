"""隔离投影各部分：对均匀 (1,0) 速度场投影一次，看哪项放大速度。

直接调用 wob 内部逻辑，分别计算 ev / ea / bsum 的贡献。
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from velmc2024.solver.scenes import KarmanScene        # noqa: E402
from velmc2024.solver.mc_solver import MCFluidSolver    # noqa: E402
from velmc2024.core import wob                           # noqa: E402
from velmc2024.core.projection import dGdx_2d           # noqa: E402


def main():
    scene = KarmanScene(viscosity=0.01, grid_res=(80, 40), dt=0.02, time_steps=100)
    s = MCFluidSolver(scene, num_paths=200,
                      num_volume_samples_direct=600,
                      num_pseudo_boundary_samples_direct=600,
                      advect_vorticity=False)
    # 初始速度应为均匀 (1,0)
    u0 = s.u_cache.u.copy()
    print("初始 ux min/max:", u0[:, :, 0].min(), u0[:, :, 0].max())

    # 手动复现 project_grid_vpl_batched 的单节点计算（取第 20 行、第 40 列的点）
    nx, ny = s.u_cache.nx, s.u_cache.ny
    i, j = np.array([40]), np.array([20])
    x, y = s.u_cache.idx_to_point(i, j)
    pts = np.stack([x, y], axis=1)  # (1,2)
    ux = s.u_cache.bilinear(pts)   # (1,2)
    Rc = np.array([scene.sim_box.max_corner_distance(p) for p in pts])

    rng = np.random.default_rng(12345)
    n_vol, n_pse, n_paths = 600, 600, 200

    # ---- ev 体积项 ----
    n_draw = max(1, n_vol // 2)
    ur = rng.random((1, n_draw))
    th = rng.uniform(0, 2 * np.pi, (1, n_draw))
    r = Rc[:, None] * ur
    r_vec = np.stack([r * np.cos(th), r * np.sin(th)], axis=-1)
    inv_pdf = 2 * np.pi * Rc[:, None] * r
    r_vec = np.concatenate([r_vec, -r_vec], axis=1)
    inv_pdf = np.concatenate([inv_pdf, inv_pdf], axis=1)
    S = 2 * n_draw
    ypts = pts[:, None, :] + r_vec
    yf = ypts.reshape(-1, 2)
    vel_y = s.u_cache.bilinear(yf).reshape(1, S, 2)
    vel_diff = vel_y - ux[:, None, :]
    in_box = scene.sim_box.contains(yf).reshape(1, S)
    wv = np.zeros(1 * S)
    for ob in scene.obstacles:
        wv = wv + ob.signed_winding(yf)
    mult = in_box * (np.abs(wv) < 0.5).reshape(1, S)
    rr = np.maximum(np.linalg.norm(r_vec, axis=-1), 1e-12)
    r_hat = r_vec / rr[..., None]
    dot_ru = np.einsum("pij,pij->pi", r_hat, vel_diff)
    kernel = 2.0 * dot_ru[..., None] * r_hat - vel_diff
    ev = np.sum(mult[..., None] * (inv_pdf / (rr * rr))[..., None] * kernel, axis=1) / (n_vol * 2 * np.pi)
    print("ev (体积项):", ev)

    # ---- ea 盒子边界项 ----
    bpts, bnorms, binv = wob._sample_union_boundary([scene.sim_box], n_pse, rng)
    gd = dGdx_2d(bpts[None, :, :] - pts[:, None, :], 1e-5)
    vel_b = s.u_cache.bilinear(bpts)
    ndot = np.einsum("bj,pbj->pb", bnorms, vel_b[None, :, :] - ux[:, None, :])
    ea = np.sum((binv * ndot)[:, :, None] * gd, axis=1) / n_pse
    print("ea (盒子边界项):", ea)

    # ---- bsum 障碍直接项 ----
    bpts, bnorms, binv = wob._sample_union_boundary(scene.obstacles, n_paths, rng)
    in_box_b = scene.sim_box.contains(bpts).astype(np.float64)
    gd = dGdx_2d(bpts[None, :, :] - pts[:, None, :], 1e-5)
    ndot = np.einsum("ij,pj->pi", bnorms, ux)
    dsum = np.sum((binv * in_box_b * ndot)[:, :, None] * gd, axis=1)
    bsum = dsum / n_paths
    print("bsum/n_paths (障碍项):", bsum)

    pg_full = -(ev + ea) - bsum
    print("pg 完整:", pg_full)
    print("out (xv-pg):", ux - pg_full)


if __name__ == "__main__":
    main()

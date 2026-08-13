"""
validate_wob.py —— 阶段2：WoB 障碍投影验证

均匀流 u=(1,0) 绕圆障碍（无边界域）：
  1) 圆边界上应满足 u·n ≈ 0（无穿透）
  2) 域外速度应与圆绕流解析势流解一致：
       u_x = U(1 - (a/r)^2 cos2θ), u_y = -U(a/r)^2 sin2θ
"""
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from velmc2024.core.geometry import Rectangle, Circle, circle_to_polygon
from velmc2024.core.wob import project_vpl_construct, project_vpl_gather


def main():
    rng = np.random.default_rng(2024)
    a = 0.3
    box = Rectangle(3.0)
    circle = Circle(center=(0.0, 0.0), radius=a, name="cylinder")
    obstacles = [circle_to_polygon(circle, n_segments=64)]
    U = 1.0
    solid = (0.0, 0.0)

    def u_uniform(pts):
        out = np.zeros((pts.shape[0], 2))
        out[:, 0] = U
        return out

    # ---- 构建共享 VPL ----
    print("构建 VPL 路径...")
    t0 = time.perf_counter()
    vpl_pos, vpl_val = project_vpl_construct(
        obstacles, u_uniform, box,
        num_paths=2000, path_length=4,
        num_volume_samples_indirect=10, num_pseudo_boundary_samples_indirect=10,
        rng=rng, solid_velocity=solid)
    print(f"  构建耗时 {time.perf_counter()-t0:.1f}s, 记录数={len(vpl_val)}")

    # ---- 在圆边界上检查 u·n ----
    bpts, bnorms = circle.sample_boundary(400, np.random.default_rng(3))
    print("检查圆边界 u·n（应≈0）...")
    unn = []
    for p, nrm in zip(bpts, bnorms):
        xv = u_uniform(p[None, :])[0]
        u4 = project_vpl_gather(
            p, xv, u_uniform, obstacles, box, vpl_pos, vpl_val,
            num_paths=2000, path_length=4,
            num_volume_samples_direct=3000, num_pseudo_boundary_samples_direct=3000,
            rng=rng, solid_velocity=solid)
        unn.append(abs(float(np.dot(u4, nrm))))
    unn = np.array(unn)
    print(f"  边界 |u·n|: mean={unn.mean():.4f}, median={np.median(unn):.4f}, "
          f"max={unn.max():.4f}  (均值应→0, 说明是方差而非偏差)")

    # ---- 域外点 vs 解析势流 ----
    print("域外点 vs 解析圆绕流...")
    xs = np.linspace(-1.2, 1.2, 13)
    X, Y = np.meshgrid(xs, xs)
    pts = np.stack([X.ravel(), Y.ravel()], axis=1)
    r = np.linalg.norm(pts, axis=1)
    mask = r > a + 0.02  # 域外
    pts = pts[mask]

    u4s = []
    for p in pts:
        xv = u_uniform(p[None, :])[0]
        u4 = project_vpl_gather(
            p, xv, u_uniform, obstacles, box, vpl_pos, vpl_val,
            num_paths=2000, path_length=4,
            num_volume_samples_direct=3000, num_pseudo_boundary_samples_direct=3000,
            rng=rng, solid_velocity=solid)
        u4s.append(u4)
    u4s = np.array(u4s)

    # 解析解
    rp = np.linalg.norm(pts, axis=1)
    th = np.arctan2(pts[:, 1], pts[:, 0])
    ux_a = U * (1.0 - (a / rp) ** 2 * np.cos(2 * th))
    uy_a = -U * (a / rp) ** 2 * np.sin(2 * th)
    err = np.sqrt(np.mean((u4s[:, 0] - ux_a) ** 2 + (u4s[:, 1] - uy_a) ** 2))
    rel = err / U
    print(f"  域外 RMSE vs 解析势流 = {err:.5f}  (相对 {rel*100:.2f}%)")
    # 远场应≈均匀流
    far = rp > 1.0
    far_err = np.sqrt(np.mean((u4s[far] - np.array([U, 0.0])) ** 2)) if far.sum() else float("nan")
    print(f"  远场(|r|>1) RMSE vs (1,0) = {far_err:.5f}")


if __name__ == "__main__":
    main()

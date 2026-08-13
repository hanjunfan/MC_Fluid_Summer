"""调试 WoB：逐项检查各贡献，定位边界项错误。"""
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from velmc2024.core.geometry import Rectangle, Circle, circle_to_polygon
from velmc2024.core.projection import (estimate_volume_term, estimate_pseudo_boundary_term, dGdx_2d)
from velmc2024.core.wob import project_vpl_construct


def main():
    rng = np.random.default_rng(5)
    a = 0.3
    box = Rectangle(3.0)
    circle = Circle(center=(0.0, 0.0), radius=a, name="cylinder")
    obstacles = [circle_to_polygon(circle, n_segments=64)]
    U = 1.0
    solid = (0.0, 0.0)

    def u_uniform(pts):
        out = np.zeros((pts.shape[0], 2)); out[:, 0] = U
        return out

    vpl_paths = project_vpl_construct(
        obstacles, u_uniform, box, num_paths=2000, path_length=4,
        num_volume_samples_indirect=10, num_pseudo_boundary_samples_indirect=10,
        rng=np.random.default_rng(99), solid_velocity=solid)
    # 展平 VPL 记录
    vpl_pos = np.array([rec[0] for path in vpl_paths for rec in path])
    vpl_val = np.array([rec[1] for path in vpl_paths for rec in path])
    print(f"VPL 记录数 = {len(vpl_pos)}, 非零 = {(np.abs(vpl_val) > 1e-12).sum()}")

    for x in [np.array([a, 0.0]), np.array([0.5, 0.0]), np.array([0.0, 0.5])]:
        xv = u_uniform(x[None, :])[0]
        ev = estimate_volume_term(x, xv, u_uniform, box, obstacles, 3000, rng, True)
        ea = estimate_pseudo_boundary_term(x, xv, u_uniform, box, obstacles, 3000, rng)
        # 直接边界项（num_paths 次，向量化）
        bpts, bnorms, bseg = obstacles[0].sample_boundary_with_idx(2000, rng)
        binv = obstacles[0].perimeter
        gd = dGdx_2d(bpts - x, 1e-5)
        direct = np.sum(binv * (bnorms @ xv)[:, None] * gd, axis=0)
        gv = dGdx_2d(vpl_pos - x, 1e-5)
        vpl_sum = vpl_val @ gv
        bound = (direct + vpl_sum) / 2000.0
        # 两种符号约定
        pg_orig = -(ev + ea) + bound
        pg_flip = -(ev + ea) - bound
        u4_orig = xv - pg_orig
        u4_flip = xv - pg_flip
        # 期望解析势流
        r = np.linalg.norm(x)
        th = np.arctan2(x[1], x[0])
        ux_a = U * (1.0 - (a / r) ** 2 * np.cos(2 * th)) if r > a - 1e-6 else 0.0
        uy_a = -U * (a / r) ** 2 * np.sin(2 * th) if r > a - 1e-6 else 0.0
        exp_u = np.array([ux_a, uy_a])
        print("=" * 60)
        print(f"x={x}, 期望 u4(解析)={np.round(exp_u, 4)}")
        print(f"  边界项={np.round(bound,4)}  (直接={np.round(direct/2000,4)}, VPL={np.round(vpl_sum/2000,4)})")
        print(f"  u4_原符号={np.round(u4_orig,4)}   |err|={np.linalg.norm(u4_orig-exp_u):.4f}")
        print(f"  u4_翻符号={np.round(u4_flip,4)}   |err|={np.linalg.norm(u4_flip-exp_u):.4f}")


if __name__ == "__main__":
    main()

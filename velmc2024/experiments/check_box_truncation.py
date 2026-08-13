"""验证：A2 残差来自盒截断假设（∇·u=0 在盒外）。盒子越大残差应越小。"""
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from velmc2024.core.geometry import Rectangle
from velmc2024.core.projection import estimate_volume_term, estimate_pseudo_boundary_term


def main():
    rng = np.random.default_rng(7)
    sigma = 0.3
    N = 100000

    def u_grad_local(pts):
        e = np.exp(-np.sum(pts ** 2, axis=1) / sigma ** 2)
        return -2.0 / sigma ** 2 * e[:, None] * pts

    test_pts = np.array([[0.2, 0.0], [0.0, 0.0], [-0.2, 0.2], [0.3, 0.1]])
    for box_size in [1.0, 1.5, 2.0, 3.0]:
        box = Rectangle(box_size)
        maxres = 0.0
        for x in test_pts:
            xv = u_grad_local(x[None, :])[0]
            ev = estimate_volume_term(x, xv, u_grad_local, box, [], N, rng, True)
            ea = estimate_pseudo_boundary_term(x, xv, u_grad_local, box, [], N, rng)
            u4 = xv + ev + ea
            maxres = max(maxres, float(np.linalg.norm(u4)))
        # 盒边界处散度大小（归一化）
        r = box_size / 2.0
        div_edge = abs((4 / sigma ** 4 * r ** 2 - 2 / sigma ** 2) * np.exp(-r ** 2 / sigma ** 2))
        print(f"box={box_size:>3}: max|u4| = {maxres:.5f}   盒边界|∇·u| = {div_edge:.3e}")


if __name__ == "__main__":
    main()

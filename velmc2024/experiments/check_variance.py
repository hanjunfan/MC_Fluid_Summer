"""测量投影体积项估计器的方差 vs 偏差：固定点多次独立运行。"""
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from velmc2024.core.geometry import Rectangle
from velmc2024.core.projection import estimate_volume_term, estimate_pseudo_boundary_term


def main():
    sigma = 0.3
    box = Rectangle(3.0)  # 大盒子：伪边界项≈0，截断误差≈0

    def u_grad_local(pts):
        e = np.exp(-np.sum(pts ** 2, axis=1) / sigma ** 2)
        return -2.0 / sigma ** 2 * e[:, None] * pts

    x = np.array([0.2, 0.0])
    xv = u_grad_local(x[None, :])[0]
    print(f"x={x}, u(x)={xv}, |u|={np.linalg.norm(xv):.4f}")
    print("期望：E_V ≈ -u(x)，即 u4 = u + E_V ≈ 0")

    for N in [1e4, 1e5, 5e5]:
        K = 8
        u4s = []
        for k in range(K):
            rng = np.random.default_rng(1000 + k)
            ev = estimate_volume_term(x, xv, u_grad_local, box, [], int(N), rng, True)
            ea = estimate_pseudo_boundary_term(x, xv, u_grad_local, box, [], int(N), rng)
            u4s.append(xv + ev + ea)
        u4s = np.array(u4s)
        mean = u4s.mean(axis=0)
        std = u4s.std(axis=0)
        print(f"N={int(N):>7}: mean(u4)={mean}, std(u4)={std}, |mean|={np.linalg.norm(mean):.4f}")


if __name__ == "__main__":
    main()

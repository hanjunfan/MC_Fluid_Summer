"""调试：检查投影体积项/伪边界项是否满足理论关系（定位 A2 偏差来源）。"""
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from velmc2024.core.geometry import Rectangle
from velmc2024.core.projection import (estimate_volume_term, estimate_pseudo_boundary_term,
                                       dGdx_2d)
from velmc2024.core.sampling import uniform_boundary_sample


def main():
    rng = np.random.default_rng(7)
    box = Rectangle(1.0)
    obstacles = []
    sigma = 0.3

    def u_grad_local(pts):
        e = np.exp(-np.sum(pts ** 2, axis=1) / sigma ** 2)
        return -2.0 / sigma ** 2 * e[:, None] * pts

    def u_rot(pts):
        return np.stack([-pts[:, 1], pts[:, 0]], axis=1)

    def make_get(u):
        return lambda pts: u(pts)

    test_pts = np.array([[0.0, 0.0], [0.2, 0.0], [-0.2, 0.2], [0.3, 0.1]])

    N = 50000
    for name, uf in [("A2 局域化梯度 u=grad(exp)", u_grad_local),
                     ("B  旋转 u=(-y,x)", u_rot)]:
        print("=" * 60)
        print(name)
        for x in test_pts:
            xv = uf(x[None, :])[0]
            get_u = make_get(uf)
            ev = estimate_volume_term(x, xv, get_u, box, obstacles, N, rng, True)
            ea = estimate_pseudo_boundary_term(x, xv, get_u, box, obstacles, N, rng)
            pg = -(ev + ea)   # 估计的 ∇p
            u4 = xv - pg
            print(f"  x={x}, |u|={np.linalg.norm(xv):.4f}")
            print(f"    E_V={ev}, E_A={ea}")
            print(f"    est grad p={pg}, 应={xv if 'grad' in name else '0'}")
            print(f"    u4={u4}, |u4|={np.linalg.norm(u4):.4f}  (纯梯度应→0; 旋转应=|u|={np.linalg.norm(xv):.4f})")

    # 额外：用大 N 缩小噪声再测一次 A2 的 u4
    print("=" * 60)
    print("A2 大 N=2e5 再测")
    N = 200000
    for x in test_pts:
        xv = u_grad_local(x[None, :])[0]
        ev = estimate_volume_term(x, xv, make_get(u_grad_local), box, obstacles, N, rng, True)
        ea = estimate_pseudo_boundary_term(x, xv, make_get(u_grad_local), box, obstacles, N, rng)
        u4 = xv + ev + ea
        print(f"  x={x}: u4={u4}, |u4|={np.linalg.norm(u4):.4f}")


if __name__ == "__main__":
    main()

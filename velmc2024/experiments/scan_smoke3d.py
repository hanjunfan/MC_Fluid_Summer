"""scan_smoke3d.py —— 3D 烟参数快速扫描（小网格找稳定组合）。

对 (beta, relax, viscosity) 组合在小网格上跑少量步，输出峰值速度与烟顶高度，
用于确定不发散、烟能上升的参数。运行很快（12³ 每步 ~0.1s）。
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from velmc2024.solver.smoke3d import Smoke3DSolver  # noqa: E402


def run_one(beta, relax, visc, grid=12, steps=40, nvol=100, npsb=100):
    s = Smoke3DSolver(grid_res=grid, dt=0.02,
                      num_volume_samples=nvol, num_pseudo_boundary_samples=npsb,
                      buoyancy_beta=beta, projection_relax=relax, viscosity=visc,
                      verbose=False)
    umax_peak = 0.0
    for _ in range(steps):
        s.step()
        umax_peak = max(umax_peak, float(np.abs(s.u_cache.u).max()))
    # 烟顶高度：浓度 > 0.05 的最高 y 坐标
    c = s.c_cache.q
    ymax = -1e9
    mask = c > 0.05
    if mask.any():
        yidx = np.where(mask)[1].max()
        ymax = (yidx - (s.ny - 1) / 2) * (s.domain_size / s.ny)
    return umax_peak, ymax


def main():
    combos = [
        (0.5, 0.2, 0.10), (0.5, 0.3, 0.10), (0.3, 0.2, 0.10), (0.3, 0.3, 0.10),
        (1.0, 0.15, 0.20), (0.5, 0.2, 0.20), (0.3, 0.25, 0.15), (0.2, 0.2, 0.10),
    ]
    print(f"{'beta':>5} {'relax':>6} {'visc':>5} | {'umax_peak':>10} {'烟顶y':>8}  评价")
    for beta, relax, visc in combos:
        umax, ytop = run_one(beta, relax, visc)
        verdict = "OK" if umax < 5 and ytop > -1.0 else ("发散" if umax >= 5 else "烟没起来")
        print(f"{beta:5.2f} {relax:6.2f} {visc:5.2f} | {umax:10.2f} {ytop:8.2f}  {verdict}")


if __name__ == "__main__":
    main()

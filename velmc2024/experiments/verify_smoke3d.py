"""verify_smoke3d.py —— 3D 烟稳定性验证：长期跑 + 高分辨率 + 耗时测量。

用于睡觉前确认参数稳定、烟能上升、并估算完整运行耗时。
"""

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from velmc2024.solver.smoke3d import Smoke3DSolver  # noqa: E402


def report(s, label):
    q = s.c_cache.q
    umax = float(np.abs(s.u_cache.u).max())
    mask = q > 0.05
    ytop = -1.5
    if mask.any():
        yidx = np.where(mask)[1].max()
        ytop = (yidx - (s.ny - 1) / 2) * (s.domain_size / s.ny)
    cmax = float(q.max())
    print(f"  {label}: t={s.time:.2f} |u|max={umax:7.2f} 烟顶y={ytop:6.2f} cmax={cmax:.3f}")


def main():
    params = dict(buoyancy_beta=1.0, projection_relax=0.15, viscosity=0.2)

    # 实验 1：grid 12 长期 120 步（看烟能否持续上升且不发散）
    print("== 实验1: grid 12, 120 步 (beta=1.0, relax=0.15, visc=0.2) ==")
    s = Smoke3DSolver(grid_res=12, dt=0.02, num_volume_samples=100,
                      num_pseudo_boundary_samples=100, verbose=False, **params)
    t0 = time.perf_counter()
    for step in range(1, 121):
        s.step()
        if step % 30 == 0:
            report(s, f"step {step:3d}")
    print(f"  grid12 耗时 {time.perf_counter()-t0:.1f}s, 每步 {(time.perf_counter()-t0)/120:.3f}s")

    # 实验 2：grid 16 短期 60 步（高分辨率稳定性 + 耗时）
    print("== 实验2: grid 16, 60 步 (同参数, nvol=120) ==")
    s2 = Smoke3DSolver(grid_res=16, dt=0.02, num_volume_samples=120,
                       num_pseudo_boundary_samples=120, verbose=False, **params)
    t1 = time.perf_counter()
    for step in range(1, 61):
        s2.step()
        if step % 20 == 0:
            report(s2, f"step {step:3d}")
    dt16 = (time.perf_counter() - t1) / 60
    print(f"  grid16 每步 {dt16:.3f}s")

    # 预测 grid 20 / 24 / 32 的单步耗时（O(n³) 缩放）
    for g in (20, 24, 32):
        scale = (g / 16) ** 3
        print(f"  预测 grid {g}^3: 每步约 {dt16*scale:.1f}s, "
              f"150 步约 {dt16*scale*150/60:.1f} 分钟")


if __name__ == "__main__":
    main()

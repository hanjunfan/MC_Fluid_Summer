"""验证 Catmull-Rom 负过冲修复：Karman 浓度条带应无负背景且向下游延伸。"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from velmc2024.solver.scenes import KarmanScene        # noqa: E402
from velmc2024.solver.mc_solver import MCFluidSolver    # noqa: E402


def main():
    scene = KarmanScene(viscosity=0.01, grid_res=(80, 40), dt=0.02, time_steps=100)
    s = MCFluidSolver(scene, num_paths=200,
                      num_volume_samples_direct=600,
                      num_pseudo_boundary_samples_direct=600,
                      advect_vorticity=False)
    for _ in range(20):
        s.step()
    q = s.c_cache.q
    print(f"min={q.min():.4f} max={q.max():.4f} sum={q.sum():.1f}")
    print(f"cells>0.5: {np.sum(q > 0.5)}")
    ys, xs = np.where(q > 0.5)
    if len(xs):
        print(f"x范围(>0.5): {xs.min()}..{xs.max()}, y范围: {ys.min()}..{ys.max()}")
    print("列和 x=0..4:", [round(float(q[:, i].sum()), 1) for i in range(5)])
    print("列和 x=10,20,30:", [round(float(q[:, i].sum()), 1) for i in (10, 20, 30)])
    # 用入口列对比：修复前下游应几乎全为负背景
    print("x=5 列 min/max:", round(float(q[:, 5].min()), 4), round(float(q[:, 5].max()), 4))


if __name__ == "__main__":
    main()

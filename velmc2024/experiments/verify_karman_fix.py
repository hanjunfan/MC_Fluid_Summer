"""验证 Karman 修复：速度稳定 + 浓度条带流入 + 无负背景。用法：python verify_karman_fix.py --nu 0.01 --steps 25 --grid 80"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from velmc2024.solver.scenes import KarmanScene        # noqa: E402
from velmc2024.solver.mc_solver import MCFluidSolver    # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nu", type=float, default=0.01)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--grid", type=int, default=80)
    args = ap.parse_args()
    scene = KarmanScene(viscosity=args.nu, grid_res=(args.grid, args.grid // 2),
                        dt=0.02, time_steps=args.steps)
    s = MCFluidSolver(scene, num_paths=200,
                      num_volume_samples_direct=600,
                      num_pseudo_boundary_samples_direct=600,
                      advect_vorticity=False)
    print(f"ν={args.nu} 初始浓度条带 cells>0.5:", np.sum(s.c_cache.q > 0.5))
    for k in range(args.steps):
        s.step()
        q = s.c_cache.q
        ux = s.u_cache.u[:, :, 0]
        if (k + 1) % 5 == 0 or k == 0:
            ys, xs = np.where(q > 0.5)
            xspan = f"{xs.min()}..{xs.max()}" if len(xs) else "无"
            print(f"step{k+1:2d}: ux[{ux.max():5.2f}/{ux.min():5.2f}]  "
                  f"c min={q.min():.3f} max={q.max():.3f} sum={q.sum():.1f}  "
                  f"cells>0.5={len(xs)} x范围={xspan}")


if __name__ == "__main__":
    main()

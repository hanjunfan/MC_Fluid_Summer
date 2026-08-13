"""调试：Karman 浓度条带为何不向下游平流。逐步检查速度场与浓度。"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from velmc2024.solver.scenes import KarmanScene        # noqa: E402
from velmc2024.solver.mc_solver import MCFluidSolver    # noqa: E402
from velmc2024.core.advection import advect_scalar      # noqa: E402


def main():
    scene = KarmanScene(viscosity=0.01, grid_res=(80, 40), dt=0.02, time_steps=100)
    s = MCFluidSolver(scene, num_paths=200,
                      num_volume_samples_direct=600,
                      num_pseudo_boundary_samples_direct=600,
                      advect_vorticity=False)
    print("初始: u 第20行前5列 =", s.u_cache.u[20, :5, 0])
    print("初始: c 第20行前5列 =", s.c_cache.q[20, :5])
    print("初始: c 第17-22行 x=0列 =", s.c_cache.q[17:23, 0])

    # 手动平流一次（用当前速度）
    c_adv = advect_scalar(s.c_cache, s.u_cache, scene.dt, "RK3")
    print("平流后: c 第20行前5列 =", np.round(c_adv.q[20, :5], 4))
    print("平流后: c 第17-22行 x=0..2列 =", np.round(c_adv.q[17:23, :3], 4))

    # 速度场分布
    ux = s.u_cache.u[:, :, 0]
    print("初始 ux 第20行前5列 =", np.round(ux[20, :5], 4))
    print("初始 ux min/max =", ux.min(), ux.max())

    # 跑 3 步看演变
    for k in range(3):
        s.step()
        q = s.c_cache.q
        print(f"step{k+1}: c 第20行前5列 =", np.round(q[20, :5], 4),
              " | sum =", round(float(q.sum()), 2),
              " | x>0 sum =", round(float(q[:, 1:].sum()), 2))
        ux = s.u_cache.u[:, :, 0]
        print(f"          ux 第20行前5列 =", np.round(ux[20, :5], 4))


if __name__ == "__main__":
    main()

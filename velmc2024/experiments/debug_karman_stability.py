"""隔离诊断：Karman 速度发散来自平流还是投影。

对比两条路径各 8 步：
  A) 标准 step()（平流+扩散+入口+投影）
  B) 手动：平流+扩散+入口（跳过投影）
观察 ux 最大/最小是否发散。
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from velmc2024.solver.scenes import KarmanScene        # noqa: E402
from velmc2024.solver.mc_solver import MCFluidSolver    # noqa: E402
from velmc2024.core.advection import advect_scalar, advect_vector  # noqa: E402
from velmc2024.core.diffusion import diffuse_vector, diffuse_scalar  # noqa: E402


def main():
    scene = KarmanScene(viscosity=0.01, grid_res=(80, 40), dt=0.02, time_steps=100)
    sA = MCFluidSolver(scene, num_paths=200,
                       num_volume_samples_direct=600,
                       num_pseudo_boundary_samples_direct=600,
                       advect_vorticity=False)
    sB = MCFluidSolver(scene, num_paths=200,
                       num_volume_samples_direct=600,
                       num_pseudo_boundary_samples_direct=600,
                       advect_vorticity=False)
    # C: 投影但跳过盒子边界项（num_pseudo_boundary_samples_direct=0）
    sC = MCFluidSolver(scene, num_paths=200,
                       num_volume_samples_direct=600,
                       num_pseudo_boundary_samples_direct=0,
                       num_pseudo_boundary_samples_indirect=0,
                       advect_vorticity=False)

    print("A=带投影 B=跳过投影 C=投影无盒子边界项 (ux max/min)")
    for k in range(8):
        sA.step()
        uxA = sA.u_cache.u[:, :, 0]
        sC.step()
        uxC = sC.u_cache.u[:, :, 0]
        dt = scene.dt
        u_adv = advect_vector(sB.u_cache, sB.u_cache, dt, "RK3")
        c_adv = advect_scalar(sB.c_cache, sB.u_cache, dt, "RK3")
        u_adv = diffuse_vector(u_adv, scene.viscosity, dt)
        c_adv = diffuse_scalar(c_adv, scene.viscosity, dt)
        scene.apply_inlet(u_adv, c_adv)
        inside = sB._inside_mask(sB._grid_points())
        u_adv.u.reshape(-1, 2)[inside] = (0.0, 0.0)
        sB.u_cache.u[...] = u_adv.u
        sB.c_cache.q[...] = c_adv.q.reshape(sB.c_cache.ny, sB.c_cache.nx)
        uxB = sB.u_cache.u[:, :, 0]
        print(f"step{k+1}: A ux[{uxA.max():.2f}/{uxA.min():.2f}]  "
              f"B ux[{uxB.max():.2f}/{uxB.min():.2f}]  "
              f"C ux[{uxC.max():.2f}/{uxC.min():.2f}]")


if __name__ == "__main__":
    main()

"""诊断 2：构造 step1 投影前速度场（平流+扩散+入口），测投影是否放大。"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from velmc2024.solver.scenes import KarmanScene        # noqa: E402
from velmc2024.solver.mc_solver import MCFluidSolver    # noqa: E402
from velmc2024.core.advection import advect_scalar, advect_vector  # noqa: E402
from velmc2024.core.diffusion import diffuse_vector, diffuse_scalar  # noqa: E402
from velmc2024.core.wob import project_vpl_construct, project_grid_vpl_batched  # noqa: E402


def main():
    scene = KarmanScene(viscosity=0.01, grid_res=(80, 40), dt=0.02, time_steps=100)
    s = MCFluidSolver(scene, num_paths=200,
                      num_volume_samples_direct=600,
                      num_pseudo_boundary_samples_direct=600,
                      advect_vorticity=False)
    dt = scene.dt
    # 模拟 step() 前半：平流+扩散+入口（投影前）
    u_adv = advect_vector(s.u_cache, s.u_cache, dt, "RK3")
    c_adv = advect_scalar(s.c_cache, s.u_cache, dt, "RK3")
    u_adv = diffuse_vector(u_adv, scene.viscosity, dt)
    c_adv = diffuse_scalar(c_adv, scene.viscosity, dt)
    scene.apply_inlet(u_adv, c_adv)
    inside = s._inside_mask(s._grid_points())
    u_adv.u.reshape(-1, 2)[inside] = (0.0, 0.0)
    s.u_cache.u[...] = u_adv.u
    s.c_cache.q[...] = c_adv.q.reshape(s.c_cache.ny, s.c_cache.nx)

    ux0 = s.u_cache.u[:, :, 0].copy()
    print("投影前 ux: min %.3f max %.3f  第20行前5列 %s" %
          (ux0.min(), ux0.max(), np.round(ux0[20, :5], 3)))

    # 投影（含盒子边界项，对照）
    vpl_pos, vpl_val = project_vpl_construct(
        scene.obstacles, s.u_cache.bilinear, scene.sim_box,
        10, 4, 10, 10, np.random.default_rng(7), scene.solid_velocity)
    project_grid_vpl_batched(
        s.u_cache, scene.obstacles, scene.sim_box,
        vpl_pos, vpl_val, 200, 4, 600, 600, np.random.default_rng(7),
        scene.solid_velocity, use_box_boundary=True)
    ux1 = s.u_cache.u[:, :, 0]
    print("投影后 ux: min %.3f max %.3f  第20行前5列 %s" %
          (ux1.min(), ux1.max(), np.round(ux1[20, :5], 3)))
    d = np.abs(ux1 - ux0)
    iy, ix = np.unravel_index(np.argmax(d), d.shape)
    print(f"增量 max {d.max():.3f} 位置 (y={iy}, x={ix})  "
          f"物理({s.u_cache.idx_to_point(ix, iy)})")
    # 沿 x 看第 20 行增量
    print("第20行增量前8列:", np.round(d[20, :8], 3))
    print("第20行增量 x=40附近:", np.round(d[20, 38:44], 3))
    # 看左右边界列的增量
    print("x=0列增量 max:", d[:, 0].max(), " x=79列增量 max:", d[:, -1].max())


if __name__ == "__main__":
    main()

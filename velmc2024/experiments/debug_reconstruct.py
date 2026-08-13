"""诊断：重建 → 投影 前后速度符号是否变化。"""
import sys
import numpy as np

sys.path.insert(0, r"c:\Users\jfhan\Desktop\MC_Fluid_Summer")
from velmc2024.solver.scenes import CohomologyScene
from velmc2024.solver.mc_solver import MCFluidSolver
from velmc2024.core.velocity_cache import VelocityCache
from velmc2024.core.wob import (reconstruct_velocity_from_vorticity,
                                project_vpl_construct, project_grid_vpl_batched)

GRID = 64
sc = CohomologyScene(grid_res=GRID)
sol = MCFluidSolver(sc, num_paths=200, num_volume_samples_direct=400,
                    num_pseudo_boundary_samples_direct=400, seed=0, verbose=False)

u_ana = sol.u_cache.u.copy()

def circ_cache(uarr, c, r=0.13):
    th = np.linspace(0, 2*np.pi, 120, endpoint=False); dth = 2*np.pi/120
    ct, st = np.cos(th), np.sin(th)
    vc = VelocityCache(GRID, GRID, 2.4); vc.u[...] = uarr
    pts = np.stack([c[0]+r*ct, c[1]+r*st], axis=1)
    u = vc.bilinear(pts)
    return float(((-u[:,0]*st + u[:,1]*ct)*r*dth).sum())

print("== 初始解析场 ==")
print(f"  红涡心 circ = {circ_cache(u_ana, [-0.7, 0.1667]):+.4f}")

# 1) 重建（无投影）
u_rec = VelocityCache(GRID, GRID, 2.4)
reconstruct_velocity_from_vorticity(sol.omega_cache, u_rec, sc.obstacles, sc.sim_box)
print("== 重建（无投影）==")
print(f"  红涡心 circ = {circ_cache(u_rec.u, [-0.7, 0.1667]):+.4f}")
rel = np.linalg.norm(u_rec.u - u_ana)/np.linalg.norm(u_ana)
print(f"  rel vs 解析 = {rel:.4f}")

# 2) 重建 + 投影
rng = np.random.default_rng(0)
vpl_pos, vpl_val = project_vpl_construct(sc.obstacles, u_rec.bilinear, sc.sim_box,
                                         200, 4, 10, 10, rng, (0, 0))
project_grid_vpl_batched(u_rec, sc.obstacles, sc.sim_box, vpl_pos, vpl_val,
                         200, 4, 400, 400, rng, (0, 0))
print("== 重建 + 投影 ==")
print(f"  红涡心 circ = {circ_cache(u_rec.u, [-0.7, 0.1667]):+.4f}")
rel2 = np.linalg.norm(u_rec.u - u_ana)/np.linalg.norm(u_ana)
print(f"  rel vs 解析 = {rel2:.4f}")

# 3) 完整跑 1 步，检查各阶段
from velmc2024.core.advection import advect_scalar
from velmc2024.core.velocity_cache import ScalarCache
om0 = sol.omega_cache.q.copy()
u_before = sol.u_cache.u.copy()
print("== 完整 step 1 分解 ==")
omega_adv = advect_scalar(sol.omega_cache, sol.u_cache, 0.05, "RK3")
print(f"  ω平流后: 红涡心附近 ω = {omega_adv.q[30:34,22:26].mean():+.3f} (原始 {om0[30:34,22:26].mean():+.3f})")
# 红涡在 (-0.7,0.167)，网格64 → i=(0.7+1.2)/0.0375≈50.7, j=(1.2-0.167)/0.0375≈27.5
iy = int((1.2-0.167)/0.0375); ix = int((-0.7+1.2)/0.0375)
print(f"  红涡格点 ({ix},{iy}): 平流前 ω={om0[iy,ix]:+.3f}, 平流后 ω={omega_adv.q[iy,ix]:+.3f}")
# 重建
uv = VelocityCache(GRID, GRID, 2.4)
reconstruct_velocity_from_vorticity(omega_adv, uv, sc.obstacles, sc.sim_box)
print(f"  重建速度 红涡心 circ = {circ_cache(uv.u, [-0.7, 0.1667]):+.4f}")
# 投影
rng2 = np.random.default_rng(1)
vpl_pos, vpl_val = project_vpl_construct(sc.obstacles, uv.bilinear, sc.sim_box, 200, 4, 10, 10, rng2, (0,0))
project_grid_vpl_batched(uv, sc.obstacles, sc.sim_box, vpl_pos, vpl_val, 200, 4, 400, 400, rng2, (0,0))
print(f"  投影后  红涡心 circ = {circ_cache(uv.u, [-0.7, 0.1667]):+.4f}")

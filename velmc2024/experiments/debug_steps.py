"""诊断：跑几步，跟踪浓度质心与速度方向。"""
import sys
import numpy as np

sys.path.insert(0, r"c:\Users\jfhan\Desktop\MC_Fluid_Summer")
from velmc2024.solver.scenes import CohomologyScene
from velmc2024.solver.mc_solver import MCFluidSolver
from velmc2024.core.velocity_cache import ScalarCache

GRID = 64
sc = CohomologyScene(grid_res=GRID)
sol = MCFluidSolver(sc, num_paths=200, num_volume_samples_direct=400,
                    num_pseudo_boundary_samples_direct=400, seed=1, verbose=False)

scache = ScalarCache(GRID, GRID, 2.4)
i = np.arange(GRID); j = np.arange(GRID)
x, y = scache.idx_to_point(i[:, None], j[None, :])
X = x.T; Y = y.T

def red_centroid():
    a = sol.c_cache.q
    q = np.where(a > 0, a, 0.0); w = q > 0.03
    if w.sum() == 0:
        return (float('nan'), float('nan')), float('nan')
    cx = float((X*w*q).sum()/(w*q).sum()); cy = float((Y*w*q).sum()/(w*q).sum())
    return (cx, cy), float(q.max())

def vel_at(c):
    pts = np.array([c])
    return sol.u_cache.bilinear(pts)[0]

print(f"step0: init, u at (-0.7,0.167)={vel_at([-0.7,0.1667])}")
for k in range(6):
    sol.step()
    (cx, cy), peak = red_centroid()
    u = vel_at([cx, cy])
    print(f"step{k+1}: red=({cx:+.3f},{cy:+.3f}) peak={peak:.3f} u_at_red={u[0]:+.4f},{u[1]:+.4f}")

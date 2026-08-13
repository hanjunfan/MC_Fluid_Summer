"""涡量模式快速验证：检查涡核环量是否随移动保持、浓度是否凝聚。"""
import sys, time
import numpy as np

sys.path.insert(0, r"c:\Users\jfhan\Desktop\MC_Fluid_Summer")
from velmc2024.solver.scenes import CohomologyScene
from velmc2024.solver.mc_solver import MCFluidSolver
from velmc2024.core.velocity_cache import ScalarCache

GRID = 96
sc = CohomologyScene(grid_res=GRID)
sol = MCFluidSolver(sc, num_paths=200, num_volume_samples_direct=400,
                    num_pseudo_boundary_samples_direct=400,
                    num_vorticity_samples=300, seed=1, verbose=False)

# 网格坐标（y 主序）
scache = ScalarCache(GRID, GRID, 2.4)
i = np.arange(GRID); j = np.arange(GRID)
x, y = scache.idx_to_point(i[:, None], j[None, :])
X = x.T; Y = y.T

th = np.linspace(0, 2*np.pi, 120, endpoint=False)
dth = 2*np.pi/120
ct, st = np.cos(th), np.sin(th)

def circ(c):
    pts = np.stack([c[0]+0.13*ct, c[1]+0.13*st], axis=1)
    u = sol.u_cache.bilinear(pts)
    tang = -u[:, 0]*st + u[:, 1]*ct
    return float((tang*0.13*dth).sum())

def centroid(a):
    q = np.where(a > 0, a, 0.0); w = q > 0.03
    if w.sum() == 0:
        return (float('nan'), float('nan'))
    return (float((X*w*q).sum()/(w*q).sum()), float((Y*w*q).sum()/(w*q).sum()))

def circ_track():
    qb = np.where(-sol.c_cache.q > 0, -sol.c_cache.q, 0.0)
    wb = qb > 0.03
    if wb.sum() == 0:
        return (0.0, 0.0)
    bc = ((X*wb*qb).sum()/(wb*qb).sum(), (Y*wb*qb).sum()/(wb*qb).sum())
    return circ(list(bc))

LOG = r"c:\Users\jfhan\Desktop\MC_Fluid_Summer\results\vorticity_test_log.txt"

def log(msg):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")
    print(msg, flush=True)

log(f"初始: omega=[{sol.omega_cache.q.min():.1f},{sol.omega_cache.q.max():.1f}] "
    f"red_circ(初始位置)={circ([-0.7, 0.1667]):+.4f}")
t0 = time.perf_counter()
try:
    for k in range(40):
        sol.step()
        if (k+1) % 10 == 0:
            rc = centroid(sol.c_cache.q)
            bc = centroid(-sol.c_cache.q)
            peak = float(np.abs(sol.c_cache.q).max())
            cr = circ(list(rc)) if np.isfinite(rc[0]) else float('nan')
            cb = circ(list(bc)) if np.isfinite(bc[0]) else float('nan')
            log(f"t={(k+1)*0.05:4.2f} peak={peak:.3f} red=({rc[0]:+.2f},{rc[1]:+.2f}) "
                f"circ_r={cr:+.3f} | blue=({bc[0]:+.2f},{bc[1]:+.2f}) circ_b={cb:+.3f}")
    log(f"40步耗时 {time.perf_counter()-t0:.1f}s ({(time.perf_counter()-t0)/40:.2f}s/步)")
except Exception as e:
    import traceback
    log("EXCEPTION: " + traceback.format_exc())

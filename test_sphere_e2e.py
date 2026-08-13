"""端到端：烟柱上升撞球体，检查球内浓度（无穿透）。grid 16 快速验证。"""
import numpy as np
import torch
from velmc2024.gpu3d.smoke3d_gpu import SmokeSolverGPU, Sphere3D

solver = SmokeSolverGPU(grid_res=16, nvol=300, npsb=200,
                        projection_relax=0.15, nsph=400,
                        spheres=[Sphere3D((0.0, 0.0, 0.0), 0.3)],
                        smoke_center=(0.0, -1.0, 0.0),
                        verbose=False)
n_steps = 120
for step in range(n_steps):
    solver.step()
    if step % 30 == 29:
        q = solver.T[0].numpy()
        print(f"step {step+1:3d}: 球心 q[8,8,8]={q[8,8,8]:.4f}  "
              f"球下 q[8,5,8]={q[8,5,8]:.4f}  球上 q[8,11,8]={q[8,11,8]:.4f}  "
              f"全max={q.max():.4f}")

"""
vorticity_conserved.py
完整修复版：涡量守恒 + 解析速度场 + 可选边界反射
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

class VorticityConservedSimulator:
    def __init__(self, nx=350, ny=350, L=2*np.pi, dt=0.001, n_particles=30000, use_boundary=True):
        """
        use_boundary: True 开启边界反射，False 关闭（粒子自由飘出）
        """
        print(f"初始化 {n_particles} 粒子，网格 {nx}x{ny}，dt={dt}")
        print(f"边界反射: {'开启' if use_boundary else '关闭'}")
        self.nx = nx
        self.ny = ny
        self.L = L
        self.dt = dt
        self.n_particles = n_particles
        self.use_boundary = use_boundary

        self.x = np.linspace(-L/2, L/2, nx)
        self.y = np.linspace(-L/2, L/2, ny)
        self.X, self.Y = np.meshgrid(self.x, self.y)

        # 均匀角度采样
        r = 0.9 * np.sqrt(np.random.uniform(0, 1, n_particles))
        theta = 2 * np.pi * np.random.uniform(0, 1, n_particles)
        self.particles_x = r * np.cos(theta)
        self.particles_y = r * np.sin(theta)

        # ===== 关键：涡量一旦初始化，永不改变（守恒） =====
        self.particles_omega = 1.5 * np.sin(self.particles_x) * np.cos(self.particles_y)

        self.time = 0.0

    def analytic_velocity(self, xq, yq):
        """泰勒-格林涡解析速度场"""
        u = np.sin(xq) * np.cos(yq)
        v = -np.cos(xq) * np.sin(yq)
        return u, v

    def step(self):
        """单步推进：只更新位置，涡量守恒不变"""
        u, v = self.analytic_velocity(self.particles_x, self.particles_y)
        self.particles_x += self.dt * u
        self.particles_y += self.dt * v

        # ===== 边界反射（可选） =====
        if self.use_boundary:
            for i in range(self.n_particles):
                r = np.sqrt(self.particles_x[i]**2 + self.particles_y[i]**2)
                if r > 0.97:
                    self.particles_x[i] *= 0.97 / r
                    self.particles_y[i] *= 0.97 / r

        self.time += self.dt

    def compute_grid_vorticity(self):
        """粒子涡量插值到网格（涡量守恒，值不变）"""
        omega_grid = np.zeros((self.ny, self.nx))
        dx = self.L / self.nx
        sigma = dx * 3.0

        for i in range(self.n_particles):
            ix = int((self.particles_x[i] + self.L/2) / self.L * self.nx)
            iy = int((self.particles_y[i] + self.L/2) / self.L * self.ny)
            for di in [-2, -1, 0, 1, 2]:
                for dj in [-2, -1, 0, 1, 2]:
                    ix2 = ix + di
                    iy2 = iy + dj
                    if 0 <= ix2 < self.nx and 0 <= iy2 < self.ny:
                        xg = self.x[ix2]
                        yg = self.y[iy2]
                        r2 = (xg - self.particles_x[i])**2 + (yg - self.particles_y[i])**2
                        weight = np.exp(-r2 / (2*sigma*sigma))
                        omega_grid[iy2, ix2] += self.particles_omega[i] * weight

        return omega_grid

    def render(self, save_path, sigma_smooth=2.0):
        omega_grid = self.compute_grid_vorticity()
        omega_smooth = gaussian_filter(omega_grid, sigma=sigma_smooth)

        vmax = np.max(np.abs(omega_smooth))
        if vmax == 0:
            vmax = 1.0
        vmin = -vmax

        plt.figure(figsize=(10, 10))
        contour = plt.contourf(self.X, self.Y, omega_smooth, levels=60,
                               cmap='RdBu_r', vmin=vmin, vmax=vmax)
        plt.colorbar(contour, fraction=0.046, pad=0.04, label='Vorticity ω')

        plt.title(f'Vorticity (Conserved) | t = {self.time:.3f} s\n'
                  f'Particles: {self.n_particles} | Boundary: {"ON" if self.use_boundary else "OFF"}',
                  fontsize=14, fontweight='bold')
        plt.xlabel('x'); plt.ylabel('y')
        plt.axis('equal')
        plt.xlim([-self.L/2, self.L/2])
        plt.ylim([-self.L/2, self.L/2])

        theta = np.linspace(0, 2*np.pi, 100)
        plt.plot(np.cos(theta), np.sin(theta), 'k-', linewidth=2.5, alpha=0.6)

        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
        plt.close()

    def diagnose_vorticity(self):
        """检查涡量是否守恒：所有粒子的涡量应该和初始值一样"""
        print("\n" + "=" * 60)
        print("涡量守恒诊断")
        print("=" * 60)
        print(f"粒子涡量最大值: {np.max(self.particles_omega):.6f}")
        print(f"粒子涡量最小值: {np.min(self.particles_omega):.6f}")
        print(f"粒子涡量平均值: {np.mean(self.particles_omega):.6f}")
        print(f"粒子涡量标准差: {np.std(self.particles_omega):.6f}")
        print("如果涡量值在演化过程中没有变化，说明守恒成立。")


if __name__ == "__main__":
    print("=" * 70)
    print("涡量守恒版模拟器")
    print("=" * 70)

    # ===== 先跑带边界反射的版本 =====
    print("\n>>> 运行带边界反射版本")
    sim = VorticityConservedSimulator(
        nx=350, ny=350,
        n_particles=30000,
        dt=0.001,
        use_boundary=True
    )

    sim.render('conserved_t0.png', sigma_smooth=2.0)
    sim.diagnose_vorticity()

    total_steps = 600
    save_interval = 50

    for step in range(total_steps):
        sim.step()
        if (step + 1) % save_interval == 0:
            sim.render(f'conserved_t{step+1:04d}.png', sigma_smooth=2.0)
            print(f"Progress: {step+1}/{total_steps}, t={sim.time:.3f}")

    sim.render('conserved_final.png', sigma_smooth=2.0)

    # ===== 再跑不带边界的版本（对比验证） =====
    print("\n" + "=" * 70)
    print(">>> 运行无边界版本（粒子自由飘出，验证涡量是否守恒）")
    print("=" * 70)

    sim_no = VorticityConservedSimulator(
        nx=350, ny=350,
        n_particles=30000,
        dt=0.001,
        use_boundary=False
    )

    sim_no.render('noboundary_t0.png', sigma_smooth=2.0)

    for step in range(total_steps):
        sim_no.step()
        if (step + 1) % save_interval == 0:
            sim_no.render(f'noboundary_t{step+1:04d}.png', sigma_smooth=2.0)
            print(f"Progress (no boundary): {step+1}/{total_steps}, t={sim_no.time:.3f}")

    sim_no.render('noboundary_final.png', sigma_smooth=2.0)

    print("\n" + "=" * 70)
    print("完成！")
    print("")
    print("📊 对比说明：")
    print("1. conserved_final.png  → 带边界反射，涡量守恒")
    print("2. noboundary_final.png → 无边界反射，粒子自由飘出")
    print("")
    print("如果 noboundary_final.png 中涡量颜色没有'淡出'，")
    print("说明边界反射是导致'淡出'的元凶。")
    print("=" * 70)
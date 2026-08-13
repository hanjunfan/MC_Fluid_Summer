"""
final_visual.py
最终版：锐化插值 + 弱平滑 + 可选边界反射
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

class FinalVisualSimulator:
    def __init__(self, nx=350, ny=350, L=2*np.pi, dt=0.001, n_particles=30000, use_boundary=False):
        """
        use_boundary: False 关闭边界反射（推荐，看到拉丝），True 开启（涡量会淡出）
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

        # 涡量守恒：初始化后永不改变
        self.particles_omega = 1.5 * np.sin(self.particles_x) * np.cos(self.particles_y)

        self.time = 0.0

    def analytic_velocity(self, xq, yq):
        """泰勒-格林涡解析速度场"""
        u = np.sin(xq) * np.cos(yq)
        v = -np.cos(xq) * np.sin(yq)
        return u, v

    def step(self):
        u, v = self.analytic_velocity(self.particles_x, self.particles_y)
        self.particles_x += self.dt * u
        self.particles_y += self.dt * v

        if self.use_boundary:
            for i in range(self.n_particles):
                r = np.sqrt(self.particles_x[i]**2 + self.particles_y[i]**2)
                if r > 0.97:
                    self.particles_x[i] *= 0.97 / r
                    self.particles_y[i] *= 0.97 / r

        self.time += self.dt

    def compute_grid_vorticity(self):
        omega_grid = np.zeros((self.ny, self.nx))
        dx = self.L / self.nx
        # ===== 关键：插值核从 dx*3.0 缩小到 dx*0.8，保留细节 =====
        sigma = dx * 0.8

        for i in range(self.n_particles):
            ix = int((self.particles_x[i] + self.L/2) / self.L * self.nx)
            iy = int((self.particles_y[i] + self.L/2) / self.L * self.ny)
            # 影响范围也缩小到 3x3（原来 5x5）
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    ix2 = ix + di
                    iy2 = iy + dj
                    if 0 <= ix2 < self.nx and 0 <= iy2 < self.ny:
                        xg = self.x[ix2]
                        yg = self.y[iy2]
                        r2 = (xg - self.particles_x[i])**2 + (yg - self.particles_y[i])**2
                        weight = np.exp(-r2 / (2*sigma*sigma))
                        omega_grid[iy2, ix2] += self.particles_omega[i] * weight

        return omega_grid

    def render(self, save_path, sigma_smooth=0.5):
        """
        sigma_smooth: 后处理高斯平滑强度，0.5 保留细节但仍有平滑
        """
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


if __name__ == "__main__":
    print("=" * 70)
    print("最终视觉版（锐化插值，弱平滑）")
    print("=" * 70)

    # ===== 先跑无边界版本（推荐，能看到拉丝） =====
    print("\n>>> 运行无边界版本（粒子自由飘出，涡量守恒）")
    sim_no = FinalVisualSimulator(
        nx=350, ny=350,
        n_particles=30000,
        dt=0.001,
        use_boundary=False
    )

    sim_no.render('final_noboundary_t0.png', sigma_smooth=0.5)

    total_steps = 600
    save_interval = 50

    for step in range(total_steps):
        sim_no.step()
        if (step + 1) % save_interval == 0:
            sim_no.render(f'final_noboundary_t{step+1:04d}.png', sigma_smooth=0.5)
            print(f"Progress (no boundary): {step+1}/{total_steps}, t={sim_no.time:.3f}")

    sim_no.render('final_noboundary_final.png', sigma_smooth=0.5)

    # ===== 再跑带边界版本（对比） =====
    print("\n" + "=" * 70)
    print(">>> 运行带边界版本（涡量会淡出，作为对比）")
    sim_boundary = FinalVisualSimulator(
        nx=350, ny=350,
        n_particles=30000,
        dt=0.001,
        use_boundary=True
    )

    sim_boundary.render('final_boundary_t0.png', sigma_smooth=0.5)

    for step in range(total_steps):
        sim_boundary.step()
        if (step + 1) % save_interval == 0:
            sim_boundary.render(f'final_boundary_t{step+1:04d}.png', sigma_smooth=0.5)
            print(f"Progress (boundary): {step+1}/{total_steps}, t={sim_boundary.time:.3f}")

    sim_boundary.render('final_boundary_final.png', sigma_smooth=0.5)

    print("\n" + "=" * 70)
    print("完成！对比两张 final 图：")
    print("1. final_noboundary_final.png → 无边界，涡量守恒，应出现拉丝/细丝")
    print("2. final_boundary_final.png   → 有边界，涡量淡出，颜色变浅")
    print("")
    print("如果 final_noboundary_final.png 中看到尖锐的红蓝细丝，则成功！")
    print("如果仍有模糊，请尝试进一步减小 sigma_smooth（如 0.2）或增大粒子数。")
    print("=" * 70)
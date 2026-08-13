"""
visualization_enhanced.py
视觉增强版：跑出论文级效果的图片
- 15000粒子
- 300x300网格
- 更小的dt
- 更平滑的高斯插值
- 更强的色彩映射
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from wos_core import WoS_Solver
import time


class VorticitySimulator2D_Enhanced:
    def __init__(self, nx=300, ny=300, L=2*np.pi, dt=0.002, nu=0.0, n_particles=15000):
        self.nx = nx
        self.ny = ny
        self.L = L
        self.dt = dt
        self.nu = nu
        self.n_particles = n_particles

        # 网格
        self.x = np.linspace(-L/2, L/2, nx)
        self.y = np.linspace(-L/2, L/2, ny)
        self.X, self.Y = np.meshgrid(self.x, self.y)

        # 初始化粒子位置（圆内均匀分布）
        print(f"生成 {n_particles} 个粒子...")
        self.particles_x = np.zeros(n_particles)
        self.particles_y = np.zeros(n_particles)
        for i in range(n_particles):
            while True:
                x_tmp = np.random.uniform(-0.95, 0.95)
                y_tmp = np.random.uniform(-0.95, 0.95)
                if x_tmp**2 + y_tmp**2 < 0.95**2:
                    self.particles_x[i] = x_tmp
                    self.particles_y[i] = y_tmp
                    break

        # 初始涡量（泰勒-格林涡，稍微放大振幅）
        self.particles_omega = 1.5 * np.sin(self.particles_x) * np.cos(self.particles_y)

        # 边界
        theta = np.linspace(0, 2*np.pi, 200)
        self.boundary_points = np.array([np.cos(theta), np.sin(theta)]).T
        self.boundary_values = np.zeros(200)
        self.wos_solver = WoS_Solver(self.boundary_points, self.boundary_values)

        self.n_escaped = 0
        self.time = 0.0

    def _initial_vorticity(self, x, y):
        return np.sin(x) * np.cos(y)

    def biot_savart(self, x_query, y_query):
        u = np.zeros_like(x_query)
        v = np.zeros_like(y_query)
        for i in range(len(x_query)):
            dx = self.particles_x - x_query[i]
            dy = self.particles_y - y_query[i]
            r2 = dx*dx + dy*dy + 1e-6
            weight = self.particles_omega / r2
            u[i] = -np.sum(weight * dy) / (2*np.pi)
            v[i] = np.sum(weight * dx) / (2*np.pi)
        return u, v

    def boundary_reflect(self, x, y):
        r = np.sqrt(x*x + y*y)
        if r > 0.98:
            nx = x / r
            ny = y / r
            rand_offset = 0.02 * np.random.randn(2)
            x_new = x - 2 * (r - 0.98) * nx + rand_offset[0]
            y_new = y - 2 * (r - 0.98) * ny + rand_offset[1]
            if x_new**2 + y_new**2 > 0.98**2:
                r_new = np.sqrt(x_new**2 + y_new**2)
                x_new = x_new / r_new * 0.96
                y_new = y_new / r_new * 0.96
            return x_new, y_new, True
        return x, y, False

    def step(self):
        u, v = self.biot_savart(self.particles_x, self.particles_y)
        self.particles_x += self.dt * u
        self.particles_y += self.dt * v

        if self.nu > 0:
            sigma = np.sqrt(2 * self.nu * self.dt)
            self.particles_x += sigma * np.random.randn(self.n_particles)
            self.particles_y += sigma * np.random.randn(self.n_particles)

        for i in range(self.n_particles):
            x_new, y_new, escaped = self.boundary_reflect(
                self.particles_x[i], self.particles_y[i]
            )
            if escaped:
                self.particles_x[i] = x_new
                self.particles_y[i] = y_new
                self.n_escaped += 1

        self.time += self.dt

    def compute_grid_vorticity(self, sigma_scale=1.5):
        """高斯核插值，sigma更大更平滑"""
        omega_grid = np.zeros((self.ny, self.nx))
        dx = self.L / self.nx
        sigma = dx * sigma_scale

        for i in range(self.n_particles):
            ix = int((self.particles_x[i] + self.L/2) / self.L * self.nx)
            iy = int((self.particles_y[i] + self.L/2) / self.L * self.ny)
            # 扩大影响范围到 5x5
            for di in range(-2, 3):
                for dj in range(-2, 3):
                    ix2 = ix + di
                    iy2 = iy + dj
                    if 0 <= ix2 < self.nx and 0 <= iy2 < self.ny:
                        xg = self.x[ix2]
                        yg = self.y[iy2]
                        r2 = (xg - self.particles_x[i])**2 + (yg - self.particles_y[i])**2
                        weight = np.exp(-r2 / (2*sigma*sigma))
                        omega_grid[iy2, ix2] += self.particles_omega[i] * weight

        return omega_grid

    def render_enhanced(self, save_path=None, cmap='RdBu_r', levels=50):
        """增强版渲染：更高分辨率、更平滑、更浓色彩"""
        omega = self.compute_grid_vorticity(sigma_scale=1.5)

        plt.figure(figsize=(10, 8))

        # 使用更多等高线层级，让颜色过渡更平滑
        contour = plt.contourf(self.X, self.Y, omega, levels=levels, cmap=cmap, extend='both')

        # Colorbar
        cbar = plt.colorbar(contour, fraction=0.046, pad=0.04)
        cbar.set_label('Vorticity ω', fontsize=12)

        plt.title(f'Vorticity Field at t = {self.time:.2f} s\n(N = {self.n_particles} particles)',
                  fontsize=14, fontweight='bold')
        plt.xlabel('x', fontsize=12)
        plt.ylabel('y', fontsize=12)
        plt.axis('equal')
        plt.xlim([-self.L/2, self.L/2])
        plt.ylim([-self.L/2, self.L/2])

        # 画边界圆（加粗）
        theta = np.linspace(0, 2*np.pi, 100)
        plt.plot(np.cos(theta), np.sin(theta), 'k-', linewidth=2.5, alpha=0.6)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=250, bbox_inches='tight', facecolor='white')
            print(f"Saved: {save_path}")
        plt.close()


if __name__ == "__main__":
    print("=" * 60)
    print("Visualization Enhanced - Paper-quality Output")
    print("=" * 60)

    # 参数：15000粒子，更高分辨率，更小步长
    sim = VorticitySimulator2D_Enhanced(
        nx=300, ny=300,
        n_particles=15000,
        dt=0.002,
        nu=0.0
    )

    # 初始状态
    sim.render_enhanced('vorticity_enhanced_t0.png')

    # 跑 200 步，每 20 步保存一张高分辨率图
    for step in range(200):
        sim.step()
        if step % 20 == 0:
            sim.render_enhanced(f'vorticity_enhanced_t{step+1:03d}.png')
            print(f"Step {step+1}/200, t = {sim.time:.2f}, escaped: {sim.n_escaped}")

    sim.render_enhanced('vorticity_enhanced_final.png')
    print("Done!")
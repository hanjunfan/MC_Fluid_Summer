"""
visual_fixed.py
修复了缩进错误 + 颜色映射范围问题
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.spatial import KDTree
from wos_core import WoS_Solver

class VisualCorrectSimulator:
    def __init__(self, nx=350, ny=350, L=2*np.pi, dt=0.001, n_particles=30000):
        print(f"初始化 {n_particles} 粒子，网格 {nx}x{ny}，dt={dt}...")
        self.nx = nx
        self.ny = ny
        self.L = L
        self.dt = dt
        self.n_particles = n_particles

        self.x = np.linspace(-L/2, L/2, nx)
        self.y = np.linspace(-L/2, L/2, ny)
        self.X, self.Y = np.meshgrid(self.x, self.y)

        # 粒子初始化（圆内均匀分布）
        self.particles_x = np.zeros(n_particles)
        self.particles_y = np.zeros(n_particles)
        for i in range(n_particles):
            while True:
                x_tmp = np.random.uniform(-0.9, 0.9)
                y_tmp = np.random.uniform(-0.9, 0.9)
                if x_tmp*x_tmp + y_tmp*y_tmp < 0.9*0.9:
                    self.particles_x[i] = x_tmp
                    self.particles_y[i] = y_tmp
                    break

        # 初始涡量（振幅1.5）
        self.particles_omega = 1.5 * np.sin(self.particles_x) * np.cos(self.particles_y)

        # 边界
        theta = np.linspace(0, 2*np.pi, 200)
        self.boundary_points = np.array([np.cos(theta), np.sin(theta)]).T
        self.boundary_values = np.zeros(200)
        self.wos_solver = WoS_Solver(self.boundary_points, self.boundary_values)

        self.n_escaped = 0
        self.time = 0.0

    def biot_savart(self, xq, yq):
        u = np.zeros_like(xq)
        v = np.zeros_like(yq)
        for i in range(len(xq)):
            dx = self.particles_x - xq[i]
            dy = self.particles_y - yq[i]
            r2 = dx*dx + dy*dy + 1e-6
            w = self.particles_omega / r2
            u[i] = np.sum(w * dy) / (2*np.pi)
            v[i] = -np.sum(w * dx) / (2*np.pi)
        return u, v

    def boundary_reflect(self, x, y):
        r = np.sqrt(x*x + y*y)
        if r > 0.97:
            nx = x / r
            ny = y / r
            rand_offset = 0.015 * np.random.randn(2)
            x_new = x - 2 * (r - 0.97) * nx + rand_offset[0]
            y_new = y - 2 * (r - 0.97) * ny + rand_offset[1]
            if x_new*x_new + y_new*y_new > 0.97*0.97:
                r2 = np.sqrt(x_new*x_new + y_new*y_new)
                x_new = x_new / r2 * 0.95
                y_new = y_new / r2 * 0.95
            return x_new, y_new, True
        return x, y, False

    def step(self):
        u, v = self.biot_savart(self.particles_x, self.particles_y)
        self.particles_x += self.dt * u
        self.particles_y += self.dt * v
        for i in range(self.n_particles):
            xn, yn, esc = self.boundary_reflect(self.particles_x[i], self.particles_y[i])
            if esc:
                self.particles_x[i] = xn
                self.particles_y[i] = yn
                self.n_escaped += 1
        self.time += self.dt

    def compute_grid_vorticity(self):
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

        # 修复颜色映射：使用全范围，不裁剪
        vmax = np.max(np.abs(omega_smooth))
        if vmax == 0:
            vmax = 1.0
        vmin = -vmax

        plt.figure(figsize=(10, 10))
        contour = plt.contourf(self.X, self.Y, omega_smooth, levels=60,
                               cmap='RdBu_r', vmin=vmin, vmax=vmax)
        plt.colorbar(contour, fraction=0.046, pad=0.04, label='Vorticity ω')

        plt.title(f'Vorticity at t = {self.time:.3f} s\n'
                  f'Particles: {self.n_particles} | dt = {self.dt}',
                  fontsize=14, fontweight='bold')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.axis('equal')
        plt.xlim([-self.L/2, self.L/2])
        plt.ylim([-self.L/2, self.L/2])

        theta = np.linspace(0, 2*np.pi, 100)
        plt.plot(np.cos(theta), np.sin(theta), 'k-', linewidth=2.5, alpha=0.6)

        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
        plt.close()

    def diagnose_velocity(self):
        """速度场诊断：检查是否有系统性偏差"""
        print("\n" + "=" * 60)
        print("速度场诊断报告")
        print("=" * 60)

        u0, v0 = self.biot_savart(np.array([0.0]), np.array([0.0]))
        print(f"原点速度: u = {u0[0]:.8f}, v = {v0[0]:.8f}")

        x_test = np.linspace(-0.5, 0.5, 10)
        y_test = np.linspace(-0.5, 0.5, 10)
        X_t, Y_t = np.meshgrid(x_test, y_test)
        x_flat = X_t.flatten()
        y_flat = Y_t.flatten()
        r_test = np.sqrt(x_flat**2 + y_flat**2)
        mask = r_test < 0.8
        x_flat = x_flat[mask]
        y_flat = y_flat[mask]
        u_flat, v_flat = self.biot_savart(x_flat, y_flat)
        u_mean = np.mean(u_flat)
        v_mean = np.mean(v_flat)
        u_std = np.std(u_flat)
        v_std = np.std(v_flat)

        print(f"\n测试点平均速度: u_mean = {u_mean:.8f}, v_mean = {v_mean:.8f}")
        print(f"u_std = {u_std:.8f}, v_std = {v_std:.8f}")

        omega_grid = self.compute_grid_vorticity()
        nx_half = self.nx // 2
        ny_half = self.ny // 2
        omega_center = omega_grid[ny_half-10:ny_half+10, nx_half-10:nx_half+10]
        omega_center_avg = np.mean(omega_center)
        print(f"\n中心区域涡量平均值: {omega_center_avg:.8f}")

        u_all, v_all = self.biot_savart(self.particles_x, self.particles_y)
        u_all_mean = np.mean(u_all)
        v_all_mean = np.mean(v_all)
        print(f"\n所有粒子平均速度: u = {u_all_mean:.8f}, v = {v_all_mean:.8f}")

        print("\n" + "-" * 40)
        if abs(v_mean) > 0.001 or abs(v_all_mean) > 0.001:
            print("⚠️ 警告：检测到显著的 y 方向净流动，可能符号组合仍需调整。")
        else:
            print("✅ 速度场平衡。")
        print("=" * 60)

        return {
            'u0': u0[0], 'v0': v0[0],
            'u_mean': u_mean, 'v_mean': v_mean,
            'u_std': u_std, 'v_std': v_std,
            'u_all_mean': u_all_mean, 'v_all_mean': v_all_mean,
            'omega_center_avg': omega_center_avg
        }


if __name__ == "__main__":
    print("=" * 70)
    print("视觉正确版 + 速度场诊断 (缩进已修复)")
    print("=" * 70)

    sim = VisualCorrectSimulator(nx=350, ny=350, n_particles=30000, dt=0.001)

    # 诊断
    sim.diagnose_velocity()

    # 初始帧
    sim.render('correct_t0.png', sigma_smooth=2.0)

    total_steps = 300
    save_interval = 30

    for step in range(total_steps):
        sim.step()
        if (step + 1) % save_interval == 0:
            sim.render(f'correct_t{step+1:04d}.png', sigma_smooth=2.0)
            print(f"Progress: {step+1}/{total_steps}, t={sim.time:.3f}, escaped={sim.n_escaped}")

    sim.render('correct_final.png', sigma_smooth=2.0)

    print("\n" + "=" * 70)
    print("完成！图片已保存。")
    print("=" * 70)
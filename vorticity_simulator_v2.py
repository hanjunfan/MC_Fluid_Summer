"""
vorticity_simulator_v2.py
改进版：软边界反射 + 更高分辨率 + 高斯插值
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from wos_core import WoS_Solver

class VorticitySimulator2D:
    def __init__(self, nx=200, ny=200, L=2*np.pi, dt=0.005, nu=0.0, n_particles=5000):
        self.nx = nx
        self.ny = ny
        self.L = L
        self.dt = dt
        self.nu = nu
        self.n_particles = n_particles
        
        self.x = np.linspace(-L/2, L/2, nx)
        self.y = np.linspace(-L/2, L/2, ny)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        
        # 初始化：更多粒子，分布在较大区域
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
        self.particles_omega = self._initial_vorticity(self.particles_x, self.particles_y)
        
        # 边界（单位圆）
        theta = np.linspace(0, 2*np.pi, 200)
        self.boundary_points = np.array([np.cos(theta), np.sin(theta)]).T
        self.boundary_values = np.zeros(200)
        self.wos_solver = WoS_Solver(self.boundary_points, self.boundary_values)
        
        # 统计边界穿透
        self.n_escaped = 0
        self.time = 0.0
    
    def _initial_vorticity(self, x, y):
        """泰勒-格林涡"""
        return np.sin(x) * np.cos(y)
    
    def biot_savart(self, x_query, y_query):
        """
        用Biot-Savart定律计算速度（优化版：向量化）
        """
        u = np.zeros_like(x_query)
        v = np.zeros_like(y_query)
        
        for i in range(len(x_query)):
            dx = self.particles_x - x_query[i]
            dy = self.particles_y - y_query[i]
            r2 = dx*dx + dy*dy + 1e-6
            weight = self.particles_omega / r2
            u[i] = np.sum(weight * dy) / (2*np.pi)
            v[i] = -np.sum(weight * dx) / (2*np.pi)
        
        return u, v
    
    def boundary_reflect(self, x, y):
        """
        软边界反射：把越界的粒子弹回圆内（沿径向）
        """
        r = np.sqrt(x*x + y*y)
        if r > 0.98:
            # 法线方向（径向）
            nx = x / r
            ny = y / r
            # 反射：沿法线方向弹回
            x_new = x - 2 * (r - 0.98) * nx
            y_new = y - 2 * (r - 0.98) * ny
            return x_new, y_new, True
        return x, y, False
    
    def step(self):
        """
        执行一个时间步
        """
        # 1. 计算速度
        u, v = self.biot_savart(self.particles_x, self.particles_y)
        
        # 2. 显式欧拉推进
        self.particles_x += self.dt * u
        self.particles_y += self.dt * v
        
        # 3. 粘性扩散
        if self.nu > 0:
            sigma = np.sqrt(2 * self.nu * self.dt)
            self.particles_x += sigma * np.random.randn(self.n_particles)
            self.particles_y += sigma * np.random.randn(self.n_particles)
        
        # 4. 边界反射（软处理）
        for i in range(self.n_particles):
            x_new, y_new, escaped = self.boundary_reflect(
                self.particles_x[i], self.particles_y[i]
            )
            if escaped:
                self.particles_x[i] = x_new
                self.particles_y[i] = y_new
                self.n_escaped += 1
        
        self.time += self.dt
    
    def compute_grid_vorticity(self):
        """
        用高斯核把粒子涡量插值到网格（平滑连续）
        """
        omega_grid = np.zeros((self.ny, self.nx))
        
        # 网格间距
        dx = self.L / self.nx
        sigma = dx * 1.0  # 高斯核宽度
        
        # 对每个粒子，只影响周围3x3网格（加速）
        for i in range(self.n_particles):
            ix = int((self.particles_x[i] + self.L/2) / self.L * self.nx)
            iy = int((self.particles_y[i] + self.L/2) / self.L * self.ny)
            
            # 周围3x3范围
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
        
        # 归一化
        return omega_grid
    
    def render(self, save_path=None):
        """
        可视化
        """
        omega = self.compute_grid_vorticity()
        
        plt.figure(figsize=(8, 6))
        # 用更平滑的 colormap
        plt.contourf(self.X, self.Y, omega, levels=30, cmap='RdBu_r')
        plt.colorbar(label='Vorticity')
        plt.title(f't = {self.time:.2f}  (particles: {self.n_particles})')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.axis('equal')
        plt.xlim([-self.L/2, self.L/2])
        plt.ylim([-self.L/2, self.L/2])
        
        # 画边界
        theta = np.linspace(0, 2*np.pi, 100)
        plt.plot(np.cos(theta), np.sin(theta), 'k--', linewidth=2)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        plt.close()


if __name__ == "__main__":
    print("=" * 60)
    print("2022 Vorticity Method Simulator (Improved)")
    print("=" * 60)
    
    # 参数：5000粒子，200x200网格，更小的dt
    sim = VorticitySimulator2D(
        nx=200, ny=200,
        n_particles=5000,
        dt=0.005,
        nu=0.0
    )
    
    sim.render('vorticity_v2_t=0.png')
    
from wos_core import WoS_Solver

# 选取一个采样点网格
x_sample = np.linspace(-1.5, 1.5, 20)
y_sample = np.linspace(-1.5, 1.5, 20)
X_s, Y_s = np.meshgrid(x_sample, y_sample)
x_flat = X_s.flatten()
y_flat = Y_s.flatten()

# 计算速度
u_flat, v_flat = sim.biot_savart(x_flat, y_flat)

# 把在圆外的点mask掉
r_sample = np.sqrt(x_flat**2 + y_flat**2)
mask = r_sample < 0.95

u_flat = u_flat[mask]
v_flat = v_flat[mask]
x_flat = x_flat[mask]
y_flat = y_flat[mask]

# 画图
plt.figure(figsize=(8, 8))
plt.quiver(x_flat, y_flat, u_flat, v_flat, 
           np.sqrt(u_flat**2 + v_flat**2), 
           cmap='viridis', scale=5, width=0.005)
plt.axis('equal')
plt.xlim([-1.8, 1.8])
plt.ylim([-1.8, 1.8])
plt.title('Velocity Field (Biot-Savart) at t=0')
theta = np.linspace(0, 2*np.pi, 100)
plt.plot(np.cos(theta), np.sin(theta), 'k--', linewidth=2)
plt.colorbar(label='Speed')
plt.savefig('velocity_field_t0.png')
print("Saved: velocity_field_t0.png")

    # 跑300步，每30步保存一张
    for step in range(300):
        sim.step()
        if step % 30 == 0:
            sim.render(f'vorticity_v2_t_{step+1:03d}.png')
            print(f"Step {step+1}/300, t = {sim.time:.2f}, escaped: {sim.n_escaped}")
    
    sim.render('vorticity_v2_final.png')
    print("Done!")
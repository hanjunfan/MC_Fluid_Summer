"""
vorticity_simulator_v3.py
包含速度向量图测试，用于检查 Biot-Savart 符号是否正确
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
        
        self.particles_x = np.random.uniform(-L/2*0.9, L/2*0.9, n_particles)
        self.particles_y = np.random.uniform(-L/2*0.9, L/2*0.9, n_particles)
        self.particles_omega = self._initial_vorticity(self.particles_x, self.particles_y)
        
        theta = np.linspace(0, 2*np.pi, 200)
        self.boundary_points = np.array([np.cos(theta), np.sin(theta)]).T
        self.boundary_values = np.zeros(200)
        self.wos_solver = WoS_Solver(self.boundary_points, self.boundary_values)
        
        self.n_escaped = 0
        self.time = 0.0
    
    def _initial_vorticity(self, x, y):
        return np.sin(x) * np.cos(y)
    
    def biot_savart(self, x_query, y_query):
        """向量化 Biot-Savart 计算速度 (u, v)"""
        u = np.zeros_like(x_query)
        v = np.zeros_like(y_query)
        for i in range(len(x_query)):
            dx = self.particles_x - x_query[i]
            dy = self.particles_y - y_query[i]
            r2 = dx*dx + dy*dy + 1e-6
            weight = self.particles_omega / r2
            # 符号修正：u = +, v = -
            u[i] = -np.sum(weight * dy) / (2*np.pi)
            v[i] = np.sum(weight * dx) / (2*np.pi)
        return u, v
    
    def boundary_reflect(self, x, y):
        r = np.sqrt(x*x + y*y)
        if r > 0.98:
            nx = x / r
            ny = y / r
            x_new = x - 2 * (r - 0.98) * nx
            y_new = y - 2 * (r - 0.98) * ny
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
    
    def compute_grid_vorticity(self):
        omega_grid = np.zeros((self.ny, self.nx))
        dx = self.L / self.nx
        sigma = dx * 1.0
        for i in range(self.n_particles):
            ix = int((self.particles_x[i] + self.L/2) / self.L * self.nx)
            iy = int((self.particles_y[i] + self.L/2) / self.L * self.ny)
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
    
    def render(self, save_path=None):
        omega = self.compute_grid_vorticity()
        plt.figure(figsize=(8, 6))
        plt.contourf(self.X, self.Y, omega, levels=30, cmap='RdBu_r')
        plt.colorbar(label='Vorticity')
        plt.title(f't = {self.time:.2f}  (particles: {self.n_particles})')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.axis('equal')
        plt.xlim([-self.L/2, self.L/2])
        plt.ylim([-self.L/2, self.L/2])
        theta = np.linspace(0, 2*np.pi, 100)
        plt.plot(np.cos(theta), np.sin(theta), 'k--', linewidth=2)
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        plt.close()


if __name__ == "__main__":
    import numpy as np
import matplotlib.pyplot as plt

# ========== 单粒子测试 ==========
# 一个粒子在原点，ω=1
particle_x = np.array([0.0])
particle_y = np.array([0.0])
particle_omega = np.array([1.0])

def biot_savart_test(x_query, y_query):
    u = np.zeros_like(x_query)
    v = np.zeros_like(y_query)
    for i in range(len(x_query)):
        dx = particle_x - x_query[i]
        dy = particle_y - y_query[i]
        r2 = dx*dx + dy*dy + 1e-6
        weight = particle_omega / r2
        # 方案A（之前的符号）
        u[i] = np.sum(weight * dy) / (2*np.pi)
        v[i] = -np.sum(weight * dx) / (2*np.pi)
    return u, v

# 采样点：围绕原点的圆环
theta = np.linspace(0, 2*np.pi, 20)
x_sample = 0.5 * np.cos(theta)
y_sample = 0.5 * np.sin(theta)

u_test, v_test = biot_savart_test(x_sample, y_sample)

plt.figure(figsize=(6, 6))
plt.quiver(x_sample, y_sample, u_test, v_test, 
           np.sqrt(u_test**2 + v_test**2), cmap='viridis', 
           scale=1.0, width=0.02)
plt.axis('equal')
plt.xlim([-1.0, 1.0])
plt.ylim([-1.0, 1.0])
plt.title('Single Particle Test: ω=1 at origin')
plt.grid(True)
plt.colorbar(label='Speed')
plt.savefig('single_particle_test.png')
print("Saved: single_particle_test.png")
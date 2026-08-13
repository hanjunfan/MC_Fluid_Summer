"""
vorticity_simulator.py
2022年论文《A Monte Carlo Method for Fluid Simulation》的2D涡量法实现
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from wos_core import WoS_Solver

class VorticitySimulator2D:
    def __init__(self, nx=100, ny=100, L=2*np.pi, dt=0.01, nu=0.0, n_particles=500):
        self.nx = nx
        self.ny = ny
        self.L = L
        self.dt = dt
        self.nu = nu
        self.n_particles = n_particles
        
        self.x = np.linspace(-L/2, L/2, nx)
        self.y = np.linspace(-L/2, L/2, ny)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        
        self.particles_x = np.random.uniform(-L/2*0.8, L/2*0.8, n_particles)
        self.particles_y = np.random.uniform(-L/2*0.8, L/2*0.8, n_particles)
        self.particles_omega = self._initial_vorticity(self.particles_x, self.particles_y)
        
        theta = np.linspace(0, 2*np.pi, 100)
        self.boundary_points = np.array([np.cos(theta), np.sin(theta)]).T
        self.boundary_values = np.zeros(100)
        self.wos_solver = WoS_Solver(self.boundary_points, self.boundary_values)
        self.time = 0.0
    
    def _initial_vorticity(self, x, y):
        return np.sin(x) * np.cos(y)
    
    def biot_savart(self, x_query, y_query, particles_x, particles_y, particles_omega):
        u = np.zeros_like(x_query)
        v = np.zeros_like(y_query)
        for i in range(len(x_query)):
            dx = particles_x - x_query[i]
            dy = particles_y - y_query[i]
            r2 = dx*dx + dy*dy + 1e-8
            u[i] = -np.sum(particles_omega * dy / r2) / (2*np.pi)
            v[i] = np.sum(particles_omega * dx / r2) / (2*np.pi)
        return u, v
    
    def step(self):
        u, v = self.biot_savart(
            self.particles_x, self.particles_y,
            self.particles_x, self.particles_y,
            self.particles_omega
        )
        self.particles_x += self.dt * u
        self.particles_y += self.dt * v
        
        for i in range(self.n_particles):
            r = np.sqrt(self.particles_x[i]**2 + self.particles_y[i]**2)
            if r > 0.95:
                self.particles_x[i] *= 0.9
                self.particles_y[i] *= 0.9
        
        if self.nu > 0:
            sigma = np.sqrt(2 * self.nu * self.dt)
            self.particles_x += sigma * np.random.randn(self.n_particles)
            self.particles_y += sigma * np.random.randn(self.n_particles)
        
        self.time += self.dt
    
    def compute_grid_vorticity(self):
        omega_grid = np.zeros((self.ny, self.nx))
        for i in range(self.n_particles):
            ix = int((self.particles_x[i] + self.L/2) / self.L * self.nx)
            iy = int((self.particles_y[i] + self.L/2) / self.L * self.ny)
            if 0 <= ix < self.nx and 0 <= iy < self.ny:
                omega_grid[iy, ix] += self.particles_omega[i]
        return omega_grid
    
    def render(self, save_path=None):
        omega = self.compute_grid_vorticity()
        plt.figure(figsize=(8, 6))
        plt.contourf(self.X, self.Y, omega, levels=20, cmap='RdBu_r')
        plt.colorbar(label='Vorticity')
        plt.title(f't = {self.time:.2f}')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.axis('equal')
        theta = np.linspace(0, 2*np.pi, 100)
        plt.plot(np.cos(theta), np.sin(theta), 'k--', linewidth=2)
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        plt.close()
if __name__ == "__main__":
    print("=" * 50)
    print("2022 Vorticity Method Simulator")
    print("=" * 50)
    
    # 参数升级：2000粒子，150x150网格，跑200步
    sim = VorticitySimulator2D(
        nx=150, ny=150,
        n_particles=2000,
        dt=0.01
    )
    
    sim.render('vorticity_t=0.png')
    
    for step in range(200):
        sim.step()
        if step % 20 == 0:
            sim.render(f'vorticity_t_{step+1:03d}.png')
            print(f"Step {step+1}/200, t = {sim.time:.2f}")
    
    sim.render('vorticity_t_final.png')
    print("Done!")
"""
mcm_fluid.py
2D涡量-速度蒙特卡洛求解器 (基于论文算法)
核心：速度由Biot-Savart积分从涡量场估计，完全无网格
边界：WoS流函数法 (自由滑移，涡量=0)
时间推进：显式欧拉，涡量守恒
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

# ========== 1. 边界处理：Walk-on-Spheres 流函数求解 ==========
class WalkOnSpheres2D:
    """2D WoS求解器：给定边界上流函数值，估计内部点流函数"""
    def __init__(self, boundary_points, boundary_psi):
        """
        boundary_points: (N,2) 边界采样点 (单位圆)
        boundary_psi: (N,) 边界上流函数值 (自由滑移时为常数)
        """
        self.boundary_points = boundary_points
        self.boundary_psi = boundary_psi
        self.tree = KDTree(boundary_points)
    
    def solve(self, x, y, max_steps=100, tol=1e-6):
        """从点(x,y)出发，执行WoS随机游走，返回边界流函数估计"""
        point = np.array([x, y])
        for _ in range(max_steps):
            dist, idx = self.tree.query(point)
            if dist < tol:
                return self.boundary_psi[idx]
            # 在半径为dist的球面上随机采样
            theta = np.random.uniform(0, 2*np.pi)
            point = point + dist * np.array([np.cos(theta), np.sin(theta)])
        # 超时返回最近边界值
        _, idx = self.tree.query(point)
        return self.boundary_psi[idx]

# ========== 2. 主模拟器 ==========
class MonteCarloVorticitySimulator:
    def __init__(self, n_particles=2000, dt=0.01, n_steps=100,
                 boundary_radius=1.0, nu=0.0):
        self.n_particles = n_particles
        self.dt = dt
        self.n_steps = n_steps
        self.boundary_radius = boundary_radius
        self.nu = nu
        
        # 初始化粒子位置（均匀分布在圆内）
        r = boundary_radius * np.sqrt(np.random.uniform(0, 1, n_particles))
        theta = 2*np.pi * np.random.uniform(0, 1, n_particles)
        self.x = r * np.cos(theta)
        self.y = r * np.sin(theta)
        
        # 初始涡量：泰勒-格林涡 (在圆域内)
        self.omega = 1.0 * np.sin(self.x) * np.cos(self.y)
        
        # 边界采样点 (单位圆)
        boundary_theta = np.linspace(0, 2*np.pi, 200)
        self.boundary_points = np.array([
            boundary_radius * np.cos(boundary_theta),
            boundary_radius * np.sin(boundary_theta)
        ]).T
        self.boundary_psi = np.zeros(200)  # 自由滑移：流函数常数
        
        # WoS求解器
        self.wos = WalkOnSpheres2D(self.boundary_points, self.boundary_psi)
        
        self.time = 0.0
        self.history = []  # 记录每帧涡量快照
    
    def biot_savart(self, x_query, y_query):
        """
        蒙特卡洛估计Biot-Savart积分：计算查询点的速度
        使用所有粒子的涡量求和（完全蒙特卡洛，无网格）
        2D Biot-Savart: u = -1/(2π) ∫ ω(y) × (x-y)/|x-y|² dy
        """
        u = np.zeros_like(x_query)
        v = np.zeros_like(y_query)
        for i in range(len(x_query)):
            dx = self.x - x_query[i]
            dy = self.y - y_query[i]
            r2 = dx*dx + dy*dy + 1e-8
            u[i] = -np.sum(self.omega * dy / r2) / (2*np.pi)
            v[i] = np.sum(self.omega * dx / r2) / (2*np.pi)
        return u, v
    
    def step(self):
        """一个时间步：计算速度 -> 推进粒子 -> 边界处理"""
        # 1. 计算速度 (蒙特卡洛Biot-Savart估计)
        u, v = self.biot_savart(self.x, self.y)
        
        # 2. 显式欧拉推进
        self.x += self.dt * u
        self.y += self.dt * v
        
        # 3. 粘性扩散 (如果nu>0)
        if self.nu > 0:
            sigma = np.sqrt(2*self.nu*self.dt)
            self.x += sigma * np.random.randn(self.n_particles)
            self.y += sigma * np.random.randn(self.n_particles)
        
        # 4. 边界处理：自由滑移 (涡量在边界处为0)
        # 使用WoS获得边界上的流函数，修正粒子的涡量
        # 简化：如果粒子跑出圆，将其拉回并设涡量为0
        for i in range(self.n_particles):
            r = np.sqrt(self.x[i]**2 + self.y[i]**2)
            if r > self.boundary_radius:
                # 拉回边界内
                self.x[i] = self.boundary_radius * self.x[i] / (r + 1e-8)
                self.y[i] = self.boundary_radius * self.y[i] / (r + 1e-8)
                # 自由滑移：涡量变为0
                self.omega[i] = 0.0
        
        self.time += self.dt
    
    def run(self):
        """运行完整模拟"""
        # 记录初始状态
        self.history.append((self.x.copy(), self.y.copy(), self.omega.copy()))
        for step in range(self.n_steps):
            self.step()
            if (step+1) % 10 == 0:
                self.history.append((self.x.copy(), self.y.copy(), self.omega.copy()))
            if (step+1) % 20 == 0:
                print(f"Step {step+1}/{self.n_steps}, t={self.time:.2f}")
    
    def render_field(self, resolution=100, save_path=None):
        """将粒子涡量插值到网格可视化"""
        nx, ny = resolution, resolution
        xg = np.linspace(-self.boundary_radius*1.2, self.boundary_radius*1.2, nx)
        yg = np.linspace(-self.boundary_radius*1.2, self.boundary_radius*1.2, ny)
        X, Y = np.meshgrid(xg, yg)
        omega_grid = np.zeros((ny, nx))
        
        # 高斯插值
        sigma = (xg[1]-xg[0]) * 2.0
        for i in range(self.n_particles):
            for xi in range(nx):
                for yi in range(ny):
                    r2 = (X[yi, xi] - self.x[i])**2 + (Y[yi, xi] - self.y[i])**2
                    weight = np.exp(-r2 / (2*sigma*sigma))
                    omega_grid[yi, xi] += self.omega[i] * weight
        return X, Y, omega_grid
    
    def plot_last_frame(self):
        """绘制最后一帧"""
        X, Y, omega = self.render_field()
        plt.figure(figsize=(8,6))
        plt.contourf(X, Y, omega, levels=40, cmap='RdBu_r')
        plt.colorbar(label='Vorticity')
        plt.title(f't={self.time:.2f}s, particles={self.n_particles}')
        plt.axis('equal')
        # 画边界
        theta = np.linspace(0, 2*np.pi, 100)
        plt.plot(self.boundary_radius*np.cos(theta), 
                 self.boundary_radius*np.sin(theta), 'k--')
        plt.show()

# ========== 主程序 ==========
if __name__ == "__main__":
    sim = MonteCarloVorticitySimulator(
        n_particles=3000,
        dt=0.005,
        n_steps=200,
        boundary_radius=1.0,
        nu=0.0
    )
    sim.run()
    sim.plot_last_frame()
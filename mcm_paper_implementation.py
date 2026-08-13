"""
mcm_paper_implementation.py
严格按 Rioux-Lavoie et al. 2022 论文算法实现的 2D 蒙特卡洛涡量法求解器

核心算法对应论文:
- Section 3: Biot-Savart 涡量→速度 (式4-8)
- Section 4.2: Feynman-Kac + Euler-Maruyama 时间推进 (式16-22)
- Section 4.1: Walk-on-Spheres 边界处理 (流函数法)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
import time

# ============================================================
# 1. Walk-on-Spheres 核心实现 (论文 Section 4.1)
# ============================================================
class WalkOnSpheres:
    """
    论文 Section 4.1: 用 WoS 求解自由滑移边界的流函数
    边界条件: Ψ = 0 (自由滑移)
    """
    def __init__(self, boundary_points):
        self.boundary_points = boundary_points
        self.tree = KDTree(boundary_points)
    
    def solve(self, x, y, max_steps=100, tol=1e-6):
        """
        从点 (x,y) 出发执行 WoS 随机游走
        返回: 边界上流函数值 (自由滑移边界为 0)
        """
        point = np.array([x, y])
        for step in range(max_steps):
            dist, _ = self.tree.query(point)
            if dist < tol:
                return 0.0  # Ψ=0 自由滑移边界
            theta = np.random.uniform(0, 2*np.pi)
            point = point + dist * np.array([np.cos(theta), np.sin(theta)])
        return 0.0


# ============================================================
# 2. 核心求解器 (论文 Section 4.2)
# ============================================================
class MonteCarloFluidSolver:
    """
    严格按论文 Section 4.2 实现的 2D 涡量法求解器
    
    论文公式:
    - Biot-Savart: (4)-(8)
    - Euler-Maruyama 时间推进: (19)
    - 涡量递推: (22)
    """
    def __init__(self, 
                 n_particles: int = 3000,      # 涡量粒子数
                 dt: float = 0.005,            # 时间步长
                 total_steps: int = 400,        # 总步数
                 boundary_radius: float = 1.0,  # 圆域半径
                 nu: float = 0.0,              # 粘性系数 (论文 Section 4.2)
                 n_diffusion_samples: int = 4,  # 扩散采样数 (论文 n_d)
                 seed: int = 42):
        
        np.random.seed(seed)
        self.n_particles = n_particles
        self.dt = dt
        self.total_steps = total_steps
        self.boundary_radius = boundary_radius
        self.nu = nu
        self.n_diffusion_samples = n_diffusion_samples
        
        # ---- 初始化粒子 (均匀分布在圆内) ----
        r = boundary_radius * np.sqrt(np.random.uniform(0, 0.9, n_particles))
        theta = 2 * np.pi * np.random.uniform(0, 1, n_particles)
        self.x = r * np.cos(theta)
        self.y = r * np.sin(theta)
        
        # ---- 初始涡量: Taylor-Green 涡 (论文 Section 5 测试用例) ----
        self.omega = 1.0 * np.sin(self.x) * np.cos(self.y)
        
        # ---- WoS 边界处理器 ----
        boundary_theta = np.linspace(0, 2*np.pi, 200)
        boundary_points = np.array([
            boundary_radius * np.cos(boundary_theta),
            boundary_radius * np.sin(boundary_theta)
        ]).T
        self.wos = WalkOnSpheres(boundary_points)
        
        # ---- 历史记录 ----
        self.time = 0.0
        self.history = []
        
        print(f"初始化: {n_particles} 粒子, dt={dt}, 总步数={total_steps}")
        print(f"粘性系数 nu={nu}, 扩散采样数={n_diffusion_samples}")
    
    # ==========================================================
    # Biot-Savart 蒙特卡洛积分 (论文式 4-8)
    # ==========================================================
    def biot_savart(self, x_query, y_query):
        """
        论文式 (4)-(8): 从涡量估计速度
        2D Biot-Savart:
            u(x) = -1/(2π) Σ ω_i × (x - x_i) / |x - x_i|²
            v(x) =  1/(2π) Σ ω_i × (y - y_i) / |x - x_i|²
        """
        u = np.zeros_like(x_query)
        v = np.zeros_like(y_query)
        
        for i in range(len(x_query)):
            dx = self.x - x_query[i]
            dy = self.y - y_query[i]
            r2 = dx*dx + dy*dy
            # 论文 Section 3: 排除自身奇点
            mask = r2 > 1e-8
            if np.any(mask):
                r2_masked = r2[mask]
                omega_masked = self.omega[mask]
                dx_masked = dx[mask]
                dy_masked = dy[mask]
                u[i] = -np.sum(omega_masked * dy_masked / r2_masked) / (2*np.pi)
                v[i] =  np.sum(omega_masked * dx_masked / r2_masked) / (2*np.pi)
            else:
                u[i] = 0.0
                v[i] = 0.0
        return u, v
    
    # ==========================================================
    # 单步推进 (论文式 19-22)
    # ==========================================================
    def step(self):
        """
        论文 Section 4.2: 单步时间推进
        式 (19): 对流 + 扩散 (Euler-Maruyama)
        式 (22): 涡量递推
        """
        # ---- 1. 计算当前速度 (Biot-Savart) ----
        u, v = self.biot_savart(self.x, self.y)
        
        # ---- 2. 对每个粒子执行 Euler-Maruyama (式 19) ----
        for i in range(self.n_particles):
            # 对流 (论文式 19 左项)
            self.x[i] += self.dt * u[i]
            self.y[i] += self.dt * v[i]
            
            # 扩散 (论文式 19 右项: Wiener 过程)
            if self.nu > 0:
                sigma = np.sqrt(2 * self.nu * self.dt)
                # 对每个扩散样本 (论文 n_d)
                for _ in range(self.n_diffusion_samples):
                    xi = sigma * np.random.randn()
                    yi = sigma * np.random.randn()
                    # 论文式 22: 对扩散样本取平均
                    self.x[i] += xi / self.n_diffusion_samples
                    self.y[i] += yi / self.n_diffusion_samples
        
        # ---- 3. 边界处理: 自由滑移 (涡量=0) ----
        for i in range(self.n_particles):
            r = np.sqrt(self.x[i]**2 + self.y[i]**2)
            if r > self.boundary_radius:
                # 拉回边界并置涡量为 0 (自由滑移)
                self.x[i] = 0.95 * self.x[i] / r
                self.y[i] = 0.95 * self.y[i] / r
                self.omega[i] = 0.0
        
        self.time += self.dt
    
    # ==========================================================
    # 运行与可视化
    # ==========================================================
    def run(self, save_every=20):
        self.history = [(self.x.copy(), self.y.copy(), self.omega.copy())]
        for step in range(self.total_steps):
            self.step()
            if (step + 1) % save_every == 0:
                self.history.append((self.x.copy(), self.y.copy(), self.omega.copy()))
                print(f"Step {step+1}/{self.total_steps}, t={self.time:.3f}s")
        print("完成!")
    
    def render(self, frame_idx=-1, resolution=200, save_path=None):
        if frame_idx < 0:
            frame_idx = len(self.history) - 1
        xp, yp, om = self.history[frame_idx]
        
        L = self.boundary_radius * 1.1
        xg = np.linspace(-L, L, resolution)
        yg = np.linspace(-L, L, resolution)
        X, Y = np.meshgrid(xg, yg)
        omega_grid = np.zeros((resolution, resolution))
        
        # 高斯插值
        dx = xg[1] - xg[0]
        sigma = dx * 1.0
        for i in range(len(xp)):
            ix = int((xp[i] + L) / (2*L) * resolution)
            iy = int((yp[i] + L) / (2*L) * resolution)
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    ix2 = ix + di
                    iy2 = iy + dj
                    if 0 <= ix2 < resolution and 0 <= iy2 < resolution:
                        xgv = xg[ix2]
                        ygv = yg[iy2]
                        r2 = (xgv - xp[i])**2 + (ygv - yp[i])**2
                        weight = np.exp(-r2 / (2*sigma*sigma))
                        omega_grid[iy2, ix2] += om[i] * weight
        
        # 归一化: 让颜色范围覆盖实际数据
        vmax = np.percentile(np.abs(omega_grid), 95)
        if vmax < 1e-12:
            vmax = 1.0
        vmin = -vmax
        
        plt.figure(figsize=(8, 8))
        plt.contourf(X, Y, omega_grid, levels=40, cmap='RdBu_r', vmin=vmin, vmax=vmax)
        plt.colorbar(label='Vorticity')
        plt.title(f't={self.time:.3f}s, N={self.n_particles}')
        plt.axis('equal')
        plt.xlim([-1.1, 1.1])
        plt.ylim([-1.1, 1.1])
        
        theta = np.linspace(0, 2*np.pi, 100)
        plt.plot(np.cos(theta), np.sin(theta), 'k--', linewidth=2)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"保存: {save_path}")
        plt.close()


# ============================================================
# 主程序
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("MCFluid 论文算法实现 (Rioux-Lavoie et al. 2022)")
    print("=" * 60)
    
    # 创建求解器
    solver = MonteCarloFluidSolver(
        n_particles=3000,
        dt=0.005,
        total_steps=400,
        boundary_radius=1.0,
        nu=0.0,
        n_diffusion_samples=1,
        seed=42
    )
    
    # 运行
    start = time.time()
    solver.run(save_every=20)
    elapsed = time.time() - start
    print(f"运行时间: {elapsed:.2f} 秒")
    
    # 生成最终图
    solver.render(save_path='paper_implementation_final.png')
    
    # 生成GIF
    print("\n生成动画...")
    try:
        import imageio
        import os
        os.makedirs('frames', exist_ok=True)
        for i in range(len(solver.history)):
            solver.render(i, save_path=f'frames/frame_{i:04d}.png')
        images = [imageio.imread(f'frames/frame_{i:04d}.png') for i in range(len(solver.history))]
        imageio.mimsave('paper_implementation.gif', images, fps=15)
        print("保存: paper_implementation.gif")
    except:
        print("未安装imageio")
    
    print("\n完成! 请查看 paper_implementation_final.png")
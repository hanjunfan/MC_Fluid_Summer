"""
wos_core.py - Walk-on-Spheres 核心函数（2D）
用于复现2022年涡量法论文的边界处理模块
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

class WoS_Solver:
    """
    Walk-on-Spheres 求解器（2D）
    功能：给定一个点，通过随机游走估算它在边界上的值
    """
    def __init__(self, boundary_points, boundary_values):
        """
        boundary_points: 边界上的点集 (N, 2)
        boundary_values: 边界上每个点的值 (N,)
        """
        self.tree = KDTree(boundary_points)
        self.boundary_values = boundary_values
        
    def distance_to_boundary(self, x):
        """计算点 x 到最近边界的距离"""
        dist, idx = self.tree.query(x)
        return dist, idx
    
    def walk(self, x0, max_steps=100, tol=1e-6):
        """
        从 x0 出发，执行 WoS 随机游走
        返回：估算的边界值
        """
        x = np.array(x0, dtype=float)
        for step in range(max_steps):
            # 1. 计算到边界的距离
            dist, idx = self.distance_to_boundary(x)
            
            # 2. 如果足够近，直接取边界值
            if dist < tol:
                return self.boundary_values[idx]
            
            # 3. 否则，在半径为 dist 的球面上随机采样一个新点
            theta = np.random.uniform(0, 2 * np.pi)
            x = x + dist * np.array([np.cos(theta), np.sin(theta)])
        
        # 如果超时，取最近边界值
        _, idx = self.tree.query(x)
        return self.boundary_values[idx]

# ============ 测试代码 ============
if __name__ == "__main__":
    import time
    
    # 构造一个简单的单位圆边界
    theta = np.linspace(0, 2*np.pi, 100)
    boundary_points = np.array([np.cos(theta), np.sin(theta)]).T
    boundary_values = np.sin(theta)
    
    solver = WoS_Solver(boundary_points, boundary_values)
    
    # ===== 跑100次取平均 =====
    n_trials = 100
    results = []
    for i in range(n_trials):
        results.append(solver.walk([0, 0]))
    
    avg = np.mean(results)
    std = np.std(results)
    print(f"原点处的估算值 (100次平均): {avg:.4f} ± {std:.4f}")
    print(f"解析解 (sin 平均值): {np.mean(boundary_values):.4f}")
    print(f"误差: {abs(avg - np.mean(boundary_values)):.4f}")
    
    # 可视化
    points = np.random.uniform(-0.8, 0.8, (100, 2))
    values = [solver.walk(p) for p in points]
    
    plt.figure(figsize=(6,6))
    plt.scatter(points[:,0], points[:,1], c=values, cmap='viridis')
    plt.colorbar(label='estimated value')
    plt.title('Walk-on-Spheres Estimation (2D)')
    plt.axis('equal')
    plt.savefig('wos_test.png')
    print("图片已保存为 wos_test.png")

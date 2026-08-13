"""
diagnose_uniform.py
均匀粒子分布诊断版
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

class DiagnosticSimulator:
    def __init__(self, nx=100, ny=100, L=2*np.pi, n_particles=5000):
        print(f"初始化 {n_particles} 粒子，网格 {nx}x{ny}...")
        self.nx = nx
        self.ny = ny
        self.L = L
        self.n_particles = n_particles

        self.x = np.linspace(-L/2, L/2, nx)
        self.y = np.linspace(-L/2, L/2, ny)
        self.X, self.Y = np.meshgrid(self.x, self.y)

        # ===== 均匀角度采样（完美均匀分布）=====
        r = 0.9 * np.sqrt(np.random.uniform(0, 1, n_particles))
        theta = 2 * np.pi * np.random.uniform(0, 1, n_particles)
        self.particles_x = r * np.cos(theta)
        self.particles_y = r * np.sin(theta)

        # 初始涡量
        self.particles_omega = 1.0 * np.sin(self.particles_x) * np.cos(self.particles_y)

        self.time = 0.0

    def biot_savart(self, xq, yq):
        u = np.zeros_like(xq)
        v = np.zeros_like(yq)
        for i in range(len(xq)):
            dx = self.particles_x - xq[i]
            dy = self.particles_y - yq[i]
            r2 = dx*dx + dy*dy + 1e-6
            w = self.particles_omega / r2
            # 符号组合 A
            u[i] = -np.sum(w * dy) / (2*np.pi)
            v[i] = np.sum(w * dx) / (2*np.pi)
        return u, v

    def diagnose_velocity(self):
        """速度场诊断"""
        print("\n" + "=" * 60)
        print("速度场诊断报告 (均匀粒子分布)")
        print("=" * 60)

        # 原点速度
        u0, v0 = self.biot_savart(np.array([0.0]), np.array([0.0]))
        print(f"原点速度: u = {u0[0]:.8f}, v = {v0[0]:.8f}")

        # 多个测试点平均
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

        print(f"\n测试点数量: {len(x_flat)}")
        print(f"u 平均值: {u_mean:.8f}, 标准差: {u_std:.8f}")
        print(f"v 平均值: {v_mean:.8f}, 标准差: {v_std:.8f}")

        # 所有粒子平均速度
        u_all, v_all = self.biot_savart(self.particles_x, self.particles_y)
        u_all_mean = np.mean(u_all)
        v_all_mean = np.mean(v_all)
        print(f"\n所有粒子平均速度: u = {u_all_mean:.8f}, v = {v_all_mean:.8f}")

        print("\n" + "-" * 40)
        if abs(v_mean) > 0.1 or abs(v_all_mean) > 0.1:
            print("⚠️ 警告：y 方向速度仍显著 > 0.1，可能 Biot-Savart 符号仍需调整。")
        else:
            print("✅ 速度场在统计意义上是平衡的！")
        print("=" * 60)


if __name__ == "__main__":
    sim = DiagnosticSimulator(n_particles=5000)
    sim.diagnose_velocity()
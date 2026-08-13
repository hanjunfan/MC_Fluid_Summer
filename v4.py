"""
test_velocity_plot.py
独立测试脚本：只画速度场，不跑主循环
用于调 quiver 参数，直到箭头清晰可见
"""

import numpy as np
import matplotlib.pyplot as plt

# ========== 设置粒子分布 ==========
n_particles = 5000
np.random.seed(42)  # 固定随机种子，可复现

# 粒子在圆内均匀分布
particles_x = np.zeros(n_particles)
particles_y = np.zeros(n_particles)
for i in range(n_particles):
    while True:
        x_tmp = np.random.uniform(-0.95, 0.95)
        y_tmp = np.random.uniform(-0.95, 0.95)
        if x_tmp**2 + y_tmp**2 < 0.95**2:
            particles_x[i] = x_tmp
            particles_y[i] = y_tmp
            break

# 初始涡量：泰勒-格林涡
particles_omega = np.sin(particles_x) * np.cos(particles_y)

# ========== Biot-Savart 函数 ==========
def biot_savart(x_query, y_query):
    u = np.zeros_like(x_query)
    v = np.zeros_like(y_query)
    for i in range(len(x_query)):
        dx = particles_x - x_query[i]
        dy = particles_y - y_query[i]
        r2 = dx*dx + dy*dy + 1e-6
        weight = particles_omega / r2
        u[i] = np.sum(weight * dy) / (2*np.pi)
        v[i] = -np.sum(weight * dx) / (2*np.pi)
    return u, v

# ========== 采样点：8x8 网格 ==========
x_sample = np.linspace(-0.8, 0.8, 8)
y_sample = np.linspace(-0.8, 0.8, 8)
X_s, Y_s = np.meshgrid(x_sample, y_sample)
x_flat = X_s.flatten()
y_flat = Y_s.flatten()

# 只保留圆内点
r_sample = np.sqrt(x_flat**2 + y_flat**2)
mask = r_sample < 0.8
x_flat = x_flat[mask]
y_flat = y_flat[mask]

# 计算速度
u_flat, v_flat = biot_savart(x_flat, y_flat)

# 归一化速度方向（只看方向，不看大小）
speed = np.sqrt(u_flat**2 + v_flat**2)
u_norm = u_flat / (speed + 1e-8)
v_norm = v_flat / (speed + 1e-8)

# ========== 画图 ==========
plt.figure(figsize=(8, 8))

# 方法1：归一化箭头（只看方向）
q1 = plt.quiver(x_flat, y_flat, u_norm, v_norm,
                color='blue', scale=1.5, width=0.02,
                pivot='mid', headwidth=4, headlength=5)

# 方法2：实际速度（看大小，scale调大）
# q2 = plt.quiver(x_flat, y_flat, u_flat, v_flat, speed,
#                 cmap='viridis', scale=50, width=0.015,
#                 pivot='mid', headwidth=3, headlength=4)

plt.axis('equal')
plt.xlim([-1.2, 1.2])
plt.ylim([-1.2, 1.2])
plt.title('Velocity Field (Normalized Arrows)')
plt.xlabel('x')
plt.ylabel('y')

# 画边界圆
theta = np.linspace(0, 2*np.pi, 100)
plt.plot(np.cos(theta), np.sin(theta), 'k--', linewidth=2)
plt.grid(True, linestyle=':', alpha=0.6)

plt.savefig('test_velocity_plot.png', dpi=150, bbox_inches='tight')
print("Saved: test_velocity_plot.png")

# 打印速度统计
print(f"平均速度大小: {np.mean(speed):.4f}")
print(f"最大速度大小: {np.max(speed):.4f}")
print(f"速度方向: 检查图片是否呈旋转环绕模式")
# ========== 复合图：涡量场 + 速度箭头 ==========
# 把粒子涡量插值到网格
nx, ny = 100, 100
x_grid = np.linspace(-1.5, 1.5, nx)
y_grid = np.linspace(-1.5, 1.5, ny)
X_g, Y_g = np.meshgrid(x_grid, y_grid)
omega_grid = np.zeros((ny, nx))

for i in range(n_particles):
    ix = int((particles_x[i] + 1.5) / 3.0 * nx)
    iy = int((particles_y[i] + 1.5) / 3.0 * ny)
    if 0 <= ix < nx and 0 <= iy < ny:
        omega_grid[iy, ix] += particles_omega[i]

# 画图
plt.figure(figsize=(8, 8))
# 红蓝涡量场
plt.contourf(X_g, Y_g, omega_grid, levels=30, cmap='RdBu_r', alpha=0.6)
plt.colorbar(label='Vorticity')
# 速度箭头（叠加在上面）
plt.quiver(x_flat, y_flat, u_norm, v_norm, color='black', scale=1.5, width=0.015, pivot='mid')
plt.axis('equal')
plt.xlim([-1.2, 1.2])
plt.ylim([-1.2, 1.2])
plt.title('Vorticity (RdBu) + Velocity Arrows')
plt.plot(np.cos(theta), np.sin(theta), 'k--', linewidth=2)
plt.savefig('composite_plot.png', dpi=150, bbox_inches='tight')
print("Saved: composite_plot.png")
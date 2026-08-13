"""
scenes.py —— 场景定义（对齐作者 VelMCFluids 的 configs/*.json 与 apps/velocity_fluids.cu）

两个场景：
  1) CohomologyScene  (scene 3, config_cohomology.json)
       两团涡量圆盘（浓度 ±1）产生的速度场把它们推过两个六边形障碍之间的间隙。
       无边界域；仿真盒 domain_size=2.4。
  2) KarmanScene  (scene 2, config_circle_karman_*.json)
       均匀来流 (1,0) 绕单个圆障碍（中心 (-1.5,0)，半径 0.125），粘性可调（Re 变化）。
       无边界域；仿真盒 [4.0, 2.0]，网格 256x128。
"""

from __future__ import annotations

import numpy as np

from ..core.geometry import Polygon, Rectangle, Circle, cohomology_obstacles, circle_to_polygon


# =========================================================================== #
#  Cohomology 场景
# =========================================================================== #
def _disk_biot_savart_quadrature(pts: np.ndarray, centers: list, signs: list,
                                 radius: float, r_div: int = 48, a_div: int = 48,
                                 chunk: int = 8192) -> np.ndarray:
    """用作者源码的数值积分计算多个均匀涡量圆盘的 Biot-Savart 速度。

    对齐 apps/velocity_fluids.cu case 3：
      u(x) = (1/(r_div*a_div)) Σ_{r,a} r * s_k * (-r_k.y, r_k.x)/|r_k|²
    其中 r 为圆盘内采样半径（径向 Jacobian 权重），s_k 为符号（+1/-1）。
    """
    P = pts.shape[0]
    out = np.zeros((P, 2))
    r_vals = (np.arange(r_div) + 0.5) * radius / r_div
    th = 2.0 * np.pi * np.arange(a_div) / a_div
    cth, sth = np.cos(th), np.sin(th)
    # 源点集合（相对圆盘中心）：(S,2)，权重为径向 Jacobian r
    r_rep = np.repeat(r_vals, a_div)
    src = np.stack([r_rep * np.tile(cth, r_div), r_rep * np.tile(sth, r_div)], axis=1)
    weights = r_rep
    for c, sgn in zip(centers, signs):
        c = np.asarray(c, dtype=np.float64)
        for s in range(0, P, chunk):
            p = pts[s:s + chunk]
            r_vec = p[:, None, :] - (c[None, None, :] + src[None, :, :])  # (C,S,2)
            r2 = np.sum(r_vec * r_vec, axis=-1) + 1e-12
            ker = np.stack([-r_vec[..., 1], r_vec[..., 0]], axis=-1) / r2[..., None]  # (C,S,2)
            out[s:s + chunk] += sgn * np.sum(weights[None, :, None] * ker, axis=1)
    return out / (r_div * a_div)


class CohomologyScene:
    """两团气体穿过双六边形障碍间隙（论文 Fig 1 / scene 3）。"""

    name = "cohomology"

    def __init__(self, grid_res: int = 128, domain_size: float = 2.4,
                 dt: float = 0.05, time_steps: int = 300):
        self.grid_res = (int(grid_res), int(grid_res))
        self.domain_size = float(domain_size)
        self.dt = float(dt)
        self.time_steps = int(time_steps)
        self.sim_box = Rectangle(domain_size)
        self.obstacles: list[Polygon] = cohomology_obstacles()
        self.solid_velocity = (0.0, 0.0)
        self.viscosity = 0.0  # 无粘性
        # 封闭盒子场景：投影含盒子边界项；浓度平滑用 CR；浓度可为负（红+蓝-两团）
        self.use_box_boundary = True
        self.concentration_interp = "catmull_rom"
        self.concentration_nonnegative = False

        # 两团涡量圆盘（作者 config/源码）
        self.c1 = np.array([-0.7, 1.0 / 6.0])
        self.c2 = np.array([-0.7, -1.0 / 6.0])
        self.disk_radius = 0.8 / 6.0

    def initial_velocity(self, pts: np.ndarray) -> np.ndarray:
        """初始速度 = 两圆盘 Biot-Savart（上盘 +, 下盘 -）。"""
        return _disk_biot_savart_quadrature(
            pts, [self.c1, self.c2], [+1.0, -1.0], self.disk_radius)

    def initial_concentration(self, pts: np.ndarray) -> np.ndarray:
        """浓度：上盘 +1（红），下盘 -1（蓝）。"""
        r1 = np.linalg.norm(pts - self.c1, axis=-1)
        r2 = np.linalg.norm(pts - self.c2, axis=-1)
        c = np.zeros(pts.shape[0])
        c[r1 <= self.disk_radius] = 1.0
        c[r2 <= self.disk_radius] = -1.0
        return c

    def describe(self) -> str:
        return (f"Cohomology: 盒 {self.domain_size}, 网格 {self.grid_res}, "
                f"dt={self.dt}, 步数 {self.time_steps}, "
                f"障碍=2 六边形, 涡量盘 中心 {self.c1}/{self.c2} 半径 {self.disk_radius:.3f}")


# =========================================================================== #
#  卡门涡街场景
# =========================================================================== #
class KarmanScene:
    """均匀来流绕圆障碍（论文 Fig 3 / scene 2），粘性可调。"""

    name = "karman"

    def __init__(self, viscosity: float = 0.01, grid_res=(256, 128),
                 domain_size=(4.0, 2.0), dt: float = 0.02, time_steps: int = 1000,
                 obstacle_center=(-1.5, 0.0), obstacle_radius: float = 0.125,
                 background_velocity: float = 1.0):
        self.grid_res = tuple(int(g) for g in grid_res)
        self.domain_size = tuple(float(d) for d in domain_size)
        self.dt = float(dt)
        self.time_steps = int(time_steps)
        self.sim_box = Rectangle(domain_size)
        self.obstacle = Circle(center=obstacle_center, radius=obstacle_radius, name="cylinder")
        # WoB/求解器需要多边形表示（作者 unit_circle.obj 也是多边形）
        self.obstacles = [circle_to_polygon(self.obstacle, n_segments=96)]
        self.solid_velocity = (0.0, 0.0)
        self.viscosity = float(viscosity)
        self.background_velocity = float(background_velocity)
        # 入流场景：投影跳过盒子边界项（入口 Dirichlet / 出口自由流出 / 上下壁自由滑移），
        # 浓度用双线性（CR 对尖锐 0/1 条带边界过冲出负值，且边界 clip 会让条带粘在入口）。
        # projection_relax<1：阻尼 MC 投影在圆柱迎风面的高方差修正，防止速度发散。
        self.use_box_boundary = False
        self.concentration_interp = "bilinear"
        self.concentration_nonnegative = True
        self.projection_relax = 0.15

    def initial_velocity(self, pts: np.ndarray) -> np.ndarray:
        """初始均匀来流 (1,0)。"""
        u = np.zeros((pts.shape[0], 2))
        u[:, 0] = self.background_velocity
        return u

    def initial_concentration(self, pts: np.ndarray) -> np.ndarray:
        """浓度：仅在左侧入流条带 |y| <= obstacle_radius（且 x 为最左列）置 1。"""
        c = np.zeros(pts.shape[0])
        # 最左列 x 判定：idx_to_point i=0 → x=-hx + Lx/(2*nx)（不是 -hx！旧判定永远 False）
        hx, hy = self.domain_size[0] / 2.0, self.domain_size[1] / 2.0
        x_min = -hx + self.domain_size[0] / (2.0 * self.grid_res[0])
        left_x = pts[:, 0] <= x_min + 1e-9
        strip = np.abs(pts[:, 1] - 0.0) <= self.obstacle.radius
        c[left_x & strip] = 1.0
        return c

    def apply_inlet(self, u_adv, c_adv) -> None:
        """每个时间步重置左侧入流列（idx x==0）：速度=(1,0)，浓度条带=1。"""
        u_adv.u[:, 0, 0] = self.background_velocity
        u_adv.u[:, 0, 1] = 0.0
        # 浓度条带：|y| <= obstacle_radius
        i, j = u_adv.idx_to_point(np.zeros(u_adv.ny), np.arange(u_adv.ny))
        strip = np.abs(j) <= self.obstacle.radius
        c_adv.q[strip, 0] = 1.0
        c_adv.q[~strip, 0] = 0.0

    def apply_inlet_mac(self, g) -> None:
        """MAC 网格版本的入口边界。

        - 左盒边 x 面 u_f[0,:] 固定为来流（Dirichlet，投影不修正边界外面）
        - 左端第一列流体单元注入浓度条带（|y|<=obstacle_radius 置 1）
        - 左端 v 面速度设为 0（法向入流无切向分量）
        """
        g.u_f[0, :] = self.background_velocity
        g.u_f[-1, :] = self.background_velocity
        g.v_f[0, :] = 0.0
        g.v_f[:, -1] = 0.0
        # 左侧第一列流体单元（i==2，两格墙之后）注入浓度条带
        strip = np.abs(g._cell_centers()[1][0, :]) <= self.obstacle.radius
        i_in = 2  # 固体墙厚 2 格
        g.c[i_in, strip] = 1.0
        g.c[i_in, ~strip] = 0.0

    def describe(self) -> str:
        return (f"Karman: 盒 {self.domain_size}, 网格 {self.grid_res}, dt={self.dt}, "
                f"步数 {self.time_steps}, ν={self.viscosity} (Re≈{0.25/self.viscosity:.0f}), "
                f"圆障碍 {self.obstacle.center} r={self.obstacle.radius}")


# =========================================================================== #
#  两团气体撞圆柱场景（扩展：CohomologyScene 的双六边形换成单圆柱）
# =========================================================================== #
class CylinderCollisionScene:
    """两团涡量圆盘从左侧撞向居中圆柱障碍，绕流后在下游形成尾流分布。

    与 CohomologyScene 几乎相同（涡量平流法、无粘无边界域），只是障碍换成单个圆柱，
    两团初始位于圆柱左侧，Biot-Savart 自驱动速度使它们向右运动并撞击/绕流圆柱。
    """

    name = "cylinder_collision"

    def __init__(self, grid_res: int = 96, domain_size: float = 2.4,
                 dt: float = 0.05, time_steps: int = 300,
                 cylinder_center=(0.0, 0.0), cylinder_radius: float = 0.15):
        self.grid_res = (int(grid_res), int(grid_res))
        self.domain_size = float(domain_size)
        self.dt = float(dt)
        self.time_steps = int(time_steps)
        self.sim_box = Rectangle(domain_size)
        self.obstacle = Circle(center=cylinder_center, radius=cylinder_radius, name="cylinder")
        self.obstacles = [circle_to_polygon(self.obstacle, n_segments=96)]
        self.solid_velocity = (0.0, 0.0)
        self.viscosity = 0.0  # 无粘性（涡量平流模式）
        # 两团场景：浓度可为负（红+1/蓝-1），不非负钳制；无边界投影盒子项
        self.use_box_boundary = True
        self.concentration_interp = "catmull_rom"
        self.concentration_nonnegative = False

        # 两团涡量圆盘：左侧上下分布（符号 +1/-1，像 cohomology）
        self.c1 = np.array([-0.6, 1.0 / 6.0])
        self.c2 = np.array([-0.6, -1.0 / 6.0])
        self.disk_radius = 0.8 / 6.0

    def initial_velocity(self, pts: np.ndarray) -> np.ndarray:
        """初始速度 = 两圆盘 Biot-Savart（上盘 +, 下盘 -），自驱动向右撞圆柱。"""
        return _disk_biot_savart_quadrature(
            pts, [self.c1, self.c2], [+1.0, -1.0], self.disk_radius)

    def initial_concentration(self, pts: np.ndarray) -> np.ndarray:
        """浓度：上盘 +1（红），下盘 -1（蓝）。"""
        r1 = np.linalg.norm(pts - self.c1, axis=-1)
        r2 = np.linalg.norm(pts - self.c2, axis=-1)
        c = np.zeros(pts.shape[0])
        c[r1 <= self.disk_radius] = 1.0
        c[r2 <= self.disk_radius] = -1.0
        return c

    def describe(self) -> str:
        return (f"CylinderCollision: 盒 {self.domain_size}, 网格 {self.grid_res}, "
                f"dt={self.dt}, 步数 {self.time_steps}, "
                f"圆柱 {self.obstacle.center} r={self.obstacle.radius}, "
                f"涡量盘 中心 {self.c1}/{self.c2} 半径 {self.disk_radius:.3f}")

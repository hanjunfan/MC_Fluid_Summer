"""
geometry.py —— 2D 几何基础（论文/作者源码的 Python 移植）

内容：
- Polygon：闭合多边形（障碍物）—— 缠绕数判定、按边长均匀采样边界、射线求交
- Rectangle：仿真域盒子（用于"伪边界"项采样）
- Circle：圆障碍（卡门涡街场景）

约定（对齐作者 CUDA 实现）：
- 障碍物多边形顶点按"顺时针"排列 => 内部点的缠绕数 = -1（作者仓库 visualize.py 里
  unbounded domain 用 winding < -0.5 判定障碍内部，与本约定一致）
- 边界法向量指向障碍内部（即"流体域的外法向"，从流体指向障碍）
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-12


def _signed_area(verts: np.ndarray) -> float:
    """多边形有向面积（CCW 为正）。"""
    x = verts[:, 0]
    y = verts[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - y * np.roll(x, -1))


class Polygon:
    """2D 闭合多边形障碍物。顶点按顺时针排列 => 内部缠绕数 -1。"""

    def __init__(self, vertices: np.ndarray, name: str = ""):
        self.name = name
        self.vertices = np.asarray(vertices, dtype=np.float64).reshape(-1, 2)
        assert self.vertices.shape[0] >= 3, "polygon needs >=3 vertices"
        self.n = self.vertices.shape[0]
        # 边向量与边长
        self.edges = np.roll(self.vertices, -1, axis=0) - self.vertices
        self.edge_len = np.linalg.norm(self.edges, axis=1)
        self.perimeter = float(self.edge_len.sum())
        self.edge_cdf = np.cumsum(self.edge_len) / self.perimeter  # 用于按边长 CDF 采样
        self.centroid = self.vertices.mean(axis=0)
        self.area = float(_signed_area(self.vertices))
        # 边中点与内法向（指向多边形内部；凸多边形可用质心判断）
        self.edge_mid = (self.vertices + np.roll(self.vertices, -1, axis=0)) / 2.0
        perp = np.stack([-self.edges[:, 1], self.edges[:, 0]], axis=1)
        perp /= np.linalg.norm(perp, axis=1, keepdims=True)
        # 指向质心一侧 => 内法向
        to_cent = self.edge_mid - self.centroid
        flip = np.einsum("ij,ij->i", perp, to_cent) < 0
        perp[flip] *= -1.0
        self.normals = perp  # 指向障碍内部（流体域外法向）

    # ------------------------------------------------------------------ #
    # 缠绕数 / 内外判定
    # ------------------------------------------------------------------ #
    def signed_winding(self, pts: np.ndarray) -> np.ndarray:
        """广义缠绕数，逐点向量化。内部点 = ±1（本约定为 -1），外部 = 0。

        w(p) = (1/2π) Σ_i angle( v_i - p, v_{i+1} - p )
        """
        pts = np.asarray(pts, dtype=np.float64)
        single = pts.ndim == 1
        if single:
            pts = pts[None, :]
        M = pts.shape[0]
        # a, b: (M, N, 2)
        a = self.vertices[None, :, :] - pts[:, None, :]
        b = np.roll(self.vertices, -1, axis=0)[None, :, :] - pts[:, None, :]
        cross = a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
        dot = a[..., 0] * b[..., 0] + a[..., 1] * b[..., 1]
        ang = np.arctan2(cross, dot)
        w = ang.sum(axis=1) / (2.0 * np.pi)
        if single:
            return w[0]
        return w

    def inside(self, pts: np.ndarray) -> np.ndarray:
        """是否在障碍内部（|winding| >= 0.5）。"""
        w = self.signed_winding(pts)
        return np.abs(w) >= 0.5

    # ------------------------------------------------------------------ #
    # 边界采样（按边长均匀）
    # ------------------------------------------------------------------ #
    def sample_boundary(self, n: int, rng: np.random.Generator | None = None,
                        jitter: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """在边界上均匀采样 n 个点，返回 (points (n,2), inward_normals (n,2))。

        inv_pdf = 周长（PDF = 1/周长）。
        """
        if rng is None:
            rng = np.random.default_rng()
        u = rng.random(n)
        edge_idx = np.searchsorted(self.edge_cdf, u)
        # 每条边上的局部参数（加上微小抖动避免顶点重复）
        s = rng.random(n) if jitter else np.full(n, 0.5)
        pts = self.vertices[edge_idx] + s[:, None] * self.edges[edge_idx]
        return pts, self.normals[edge_idx]

    def sample_boundary_with_idx(self, n: int, rng: np.random.Generator,
                                 jitter: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """同 sample_boundary，额外返回每条边索引 edge_idx (n,)（WoB 需要排除起点线段）。"""
        if rng is None:
            rng = np.random.default_rng()
        u = rng.random(n)
        edge_idx = np.searchsorted(self.edge_cdf, u)
        s = rng.random(n) if jitter else np.full(n, 0.5)
        pts = self.vertices[edge_idx] + s[:, None] * self.edges[edge_idx]
        return pts, self.normals[edge_idx], edge_idx

    # ------------------------------------------------------------------ #
    # 射线求交（WoB 路径扩展）
    # ------------------------------------------------------------------ #
    def ray_intersect(self, origin: np.ndarray, direction: np.ndarray) -> float | None:
        """返回射线 origin + t*direction 与多边形边界的第一个正交点距离 t（None 表示无交点）。

        direction 需为单位向量。
        """
        o = np.asarray(origin, dtype=np.float64)
        d = np.asarray(direction, dtype=np.float64)
        # 对每条边 (p1,p2)：o + t d = p1 + s (p2-p1)
        p1 = self.vertices
        p2 = np.roll(self.vertices, -1, axis=0)
        e = p2 - p1
        # 解线性方程组
        denom = e[:, 0] * d[1] - e[:, 1] * d[0]   # cross(e, d)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(np.abs(denom) > _EPS,
                         ((o[0] - p1[:, 0]) * e[:, 1] - (o[1] - p1[:, 1]) * e[:, 0]) / denom,
                         np.inf)
            s = np.where(np.abs(denom) > _EPS,
                         ((o[0] - p1[:, 0]) * d[1] - (o[1] - p1[:, 1]) * d[0]) / denom,
                         -1.0)
        valid = (t > _EPS) & (s >= 0.0) & (s <= 1.0)
        ts = t[valid]
        if ts.size == 0:
            return None
        return float(ts.min())

    def ray_intersect_point(self, origin: np.ndarray, direction: np.ndarray) -> np.ndarray | None:
        """射线与多边形边界的第一个交点坐标。"""
        t = self.ray_intersect(origin, direction)
        if t is None:
            return None
        return np.asarray(origin, dtype=np.float64) + t * np.asarray(direction, dtype=np.float64)

    def line_intersections(self, origin: np.ndarray, direction: np.ndarray,
                           skip_seg: int | None = None) -> list[tuple[float, np.ndarray, int]]:
        """沿整条直线（正负两个方向）与多边形边界的所有交点。

        返回 [(t, point, seg_idx), ...]，t 可为负（反方向）；可跳过某条线段（WoB 排除起点线段）。
        direction 需为单位向量。
        """
        o = np.asarray(origin, dtype=np.float64)
        d = np.asarray(direction, dtype=np.float64)
        p1 = self.vertices
        p2 = np.roll(self.vertices, -1, axis=0)
        e = p2 - p1
        denom = e[:, 0] * d[1] - e[:, 1] * d[0]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(np.abs(denom) > _EPS,
                         ((o[0] - p1[:, 0]) * e[:, 1] - (o[1] - p1[:, 1]) * e[:, 0]) / denom,
                         np.inf)
            s = np.where(np.abs(denom) > _EPS,
                         ((o[0] - p1[:, 0]) * d[1] - (o[1] - p1[:, 1]) * d[0]) / denom,
                         -1.0)
        hits = []
        for i in range(self.n):
            if skip_seg is not None and i == skip_seg:
                continue
            if np.abs(t[i]) > _EPS and 0.0 <= s[i] <= 1.0:
                hits.append((float(t[i]), o + t[i] * d, i))
        return hits

    def __repr__(self):
        return f"Polygon({self.name}, {self.n} verts, area={self.area:+.3f})"


class Rectangle:
    """仿真域盒子（[-L/2, L/2]^2），用作无边界域时的"伪边界"采样几何。"""

    def __init__(self, size: float | tuple[float, float], center=(0.0, 0.0)):
        if np.isscalar(size):
            size = (float(size), float(size))
        self.size = (float(size[0]), float(size[1]))
        self.center = np.asarray(center, dtype=np.float64)
        hx, hy = self.size[0] / 2.0, self.size[1] / 2.0
        # 顶点：逆时针（外法向朝外）
        self.vertices = np.array([
            [-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy],
        ], dtype=np.float64) + self.center
        self.perimeter = 2.0 * (self.size[0] + self.size[1])

    def sample_boundary(self, n: int, rng: np.random.Generator | None = None) -> tuple[np.ndarray, np.ndarray]:
        """在盒子边界均匀采样，返回 (points, outward_normals)。inv_pdf = 周长。"""
        if rng is None:
            rng = np.random.default_rng()
        hx, hy = self.size[0] / 2.0, self.size[1] / 2.0
        L = self.perimeter
        u = rng.random(n) * L
        pts = np.empty((n, 2))
        norms = np.empty((n, 2))
        lengths = np.array([2 * hx, 2 * hy, 2 * hx, 2 * hy])
        start = 0.0
        for k, (edge_len, (sx, sy, nx, ny)) in enumerate(zip(
                lengths,
                [(-hx, -hy, 0.0, -1.0), (hx, -hy, 1.0, 0.0), (hx, hy, 0.0, 1.0), (-hx, hy, -1.0, 0.0)])):
            mask = (u >= start) & (u < start + edge_len)
            local = (u[mask] - start) / edge_len
            if k == 0:
                pts[mask] = self.center + np.stack([sx + local * 2 * hx, np.full(local.size, sy)], axis=1)
            elif k == 1:
                pts[mask] = self.center + np.stack([np.full(local.size, sx), sy + local * 2 * hy], axis=1)
            elif k == 2:
                pts[mask] = self.center + np.stack([sx - local * 2 * hx, np.full(local.size, sy)], axis=1)
            else:
                pts[mask] = self.center + np.stack([np.full(local.size, sx), sy - local * 2 * hy], axis=1)
            norms[mask] = [nx, ny]
            start += edge_len
        return pts, norms

    def contains(self, pts: np.ndarray) -> np.ndarray:
        """点是否在盒子内（用于体积项/伪边界项的 box 截断）。"""
        pts = np.asarray(pts, dtype=np.float64)
        hx, hy = self.size[0] / 2.0, self.size[1] / 2.0
        rel = pts - self.center
        return (np.abs(rel[:, 0]) <= hx) & (np.abs(rel[:, 1]) <= hy)

    def max_corner_distance(self, x: np.ndarray) -> float:
        """点 x 到盒子最远角的距离（用于体积项采样半径）。"""
        x = np.asarray(x, dtype=np.float64)
        hx, hy = self.size[0] / 2.0, self.size[1] / 2.0
        corners = np.array([[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy]]) + self.center
        return float(np.max(np.linalg.norm(corners - x, axis=1)))


class Circle:
    """圆障碍（卡门涡街场景）。"""

    def __init__(self, center=(0.0, 0.0), radius: float = 1.0, name: str = "circle"):
        self.center = np.asarray(center, dtype=np.float64)
        self.radius = float(radius)
        self.name = name
        self.perimeter = 2.0 * np.pi * self.radius

    def inside(self, pts: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts, dtype=np.float64)
        return np.linalg.norm(pts - self.center, axis=-1) <= self.radius

    def sample_boundary(self, n: int, rng: np.random.Generator | None = None) -> tuple[np.ndarray, np.ndarray]:
        """均匀采样圆周，返回 (points, inward_normals)。"""
        if rng is None:
            rng = np.random.default_rng()
        theta = rng.uniform(0.0, 2.0 * np.pi, n)
        pts = self.center + self.radius * np.stack([np.cos(theta), np.sin(theta)], axis=1)
        norms = -np.stack([np.cos(theta), np.sin(theta)], axis=1)  # 指向圆心（障碍内部）
        return pts, norms

    def ray_intersect(self, origin: np.ndarray, direction: np.ndarray) -> float | None:
        o = np.asarray(origin, dtype=np.float64) - self.center
        d = np.asarray(direction, dtype=np.float64)
        b = o @ d
        c = o @ o - self.radius * self.radius
        disc = b * b - c
        if disc < 0:
            return None
        sq = np.sqrt(disc)
        t = -b - sq
        if t > _EPS:
            return float(t)
        t = -b + sq
        return float(t) if t > _EPS else None


def build_hexagon(center, circumradius: float) -> np.ndarray:
    """生成六边形顶点（顺时针排列），第一个顶点在右侧，与作者 hexagons.obj 一致。"""
    # 从 +x 轴开始、顺时针旋转
    angles = -np.linspace(0.0, 2.0 * np.pi, 7)[:-1]
    verts = np.stack([np.cos(angles), np.sin(angles)], axis=1) * circumradius + np.asarray(center, dtype=np.float64)
    return verts


def circle_to_polygon(circle: Circle, n_segments: int = 96) -> Polygon:
    """把圆离散成顺时针多边形（作者 unit_circle.obj 也是多边形近似）。"""
    angles = -np.linspace(0.0, 2.0 * np.pi, n_segments + 1)[:-1]
    verts = circle.center + circle.radius * np.stack([np.cos(angles), np.sin(angles)], axis=1)
    return Polygon(verts, name=circle.name)


def cohomology_obstacles() -> list[Polygon]:
    """论文 cohomology 场景的两个六边形障碍（顶点来自作者 hexagons.obj）。"""
    top = Polygon(np.array([
        [0.09622504486, 0.66666666666],
        [0.19245008973, 0.5],
        [0.09622504486, 0.33333333333],
        [-0.09622504486, 0.33333333333],
        [-0.19245008973, 0.5],
        [-0.09622504486, 0.66666666666],
    ]), name="hex_top")
    bot = Polygon(np.array([
        [0.09622504486, -0.33333333333],
        [0.19245008973, -0.5],
        [0.09622504486, -0.66666666666],
        [-0.09622504486, -0.66666666666],
        [-0.19245008973, -0.5],
        [-0.09622504486, -0.33333333333],
    ]), name="hex_bot")
    return [top, bot]

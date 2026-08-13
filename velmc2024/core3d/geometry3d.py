"""geometry3d.py —— 三维几何基础（仿真盒 Box3D）。

Box3D 用作无边界域的"伪边界"采样几何：contains（体积项盒截断）、
sample_boundary（6 面均匀采样）、max_corner_distance（体积项采样半径）。
"""

from __future__ import annotations

import numpy as np


class Box3D:
    """轴对齐三维盒 [-hx,hx]×[-hy,hy]×[-hz,hz]（中心 center）。"""

    def __init__(self, size, center=(0.0, 0.0, 0.0)):
        if np.isscalar(size):
            size = (float(size), float(size), float(size))
        self.size = (float(size[0]), float(size[1]), float(size[2]))
        self.center = np.asarray(center, dtype=np.float64)
        self.hx = self.size[0] / 2.0
        self.hy = self.size[1] / 2.0
        self.hz = self.size[2] / 2.0
        Lx, Ly, Lz = self.size
        self.area = 2.0 * (Lx * Ly + Lx * Lz + Ly * Lz)  # 6 面总面积

    def contains(self, pts: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts, dtype=np.float64)
        rel = pts - self.center
        return ((np.abs(rel[:, 0]) <= self.hx)
                & (np.abs(rel[:, 1]) <= self.hy)
                & (np.abs(rel[:, 2]) <= self.hz))

    def max_corner_distance(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=np.float64)
        hx, hy, hz = self.hx, self.hy, self.hz
        corners = np.array([
            [-hx, -hy, -hz], [hx, -hy, -hz], [hx, hy, -hz], [-hx, hy, -hz],
            [-hx, -hy, hz], [hx, -hy, hz], [hx, hy, hz], [-hx, hy, hz],
        ], dtype=np.float64) + self.center
        return float(np.max(np.linalg.norm(corners - x, axis=1)))

    def sample_boundary(self, n: int, rng: np.random.Generator | None = None):
        """在 6 个面上按面积均匀采样，返回 (points (n,3), outward_normals (n,3))。

        外法向（从盒指向外），inv_pdf = 总面积。
        """
        if rng is None:
            rng = np.random.default_rng()
        Lx, Ly, Lz = self.size
        hx, hy, hz = self.hx, self.hy, self.hz
        # 6 个面：-x, +x, -y, +y, -z, +z
        areas = np.array([Ly * Lz, Ly * Lz, Lx * Lz, Lx * Lz, Lx * Ly, Lx * Ly])
        cdf = np.cumsum(areas) / areas.sum()
        u = rng.random(n)
        face = np.searchsorted(cdf, u)
        pts = np.empty((n, 3))
        norms = np.empty((n, 3))
        # 每个面：两个切向坐标 + 固定法向坐标
        a = rng.uniform(-1, 1, n)  # 第一切向参数（× 半宽）
        b = rng.uniform(-1, 1, n)  # 第二切向参数
        cx, cy, cz = self.center
        for f, (nx, ny, nz) in enumerate([(-1, 0, 0), (1, 0, 0), (0, -1, 0),
                                          (0, 1, 0), (0, 0, -1), (0, 0, 1)]):
            m = face == f
            if not m.any():
                continue
            if f == 0 or f == 1:      # ±x 面：切向 y, z
                pts[m] = np.stack([np.full(m.sum(), nx * hx), a[m] * hy, b[m] * hz], axis=1)
            elif f == 2 or f == 3:    # ±y 面：切向 x, z
                pts[m] = np.stack([a[m] * hx, np.full(m.sum(), ny * hy), b[m] * hz], axis=1)
            else:                      # ±z 面：切向 x, y
                pts[m] = np.stack([a[m] * hx, b[m] * hy, np.full(m.sum(), nz * hz)], axis=1)
            norms[m] = [nx, ny, nz]
        return pts + self.center, norms

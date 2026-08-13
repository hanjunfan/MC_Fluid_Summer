"""velocity_cache3d.py —— 三维速度/标量场网格缓存与三线性点查询。

存储约定（对齐 2D 的 y 主序推广到 3D）：
    速度缓存 u 形状 (nz, ny, nx, 3)，u[iz, iy, ix] = 节点 (ix, iy, iz) 处的速度；
    展平（C 序）时 iz 变化最快、ix 最慢。
索引 <-> 坐标（对齐 2D / 作者实现）：
    x = (ix - (nx-1)/2) * (Lx/nx)      （y、z 同理）
"""

from __future__ import annotations

import numpy as np


class VelocityCache3D:
    """三维速度场网格缓存 (nz, ny, nx, 3)，支持批量三线性点查询（ClampToEdge）。"""

    def __init__(self, nx: int, ny: int, nz: int, domain_size):
        self.nx = int(nx)
        self.ny = int(ny)
        self.nz = int(nz)
        if np.isscalar(domain_size):
            self.Lx = self.Ly = self.Lz = float(domain_size)
        else:
            self.Lx, self.Ly, self.Lz = (float(domain_size[0]), float(domain_size[1]),
                                         float(domain_size[2]))
        self.L = (self.Lx, self.Ly, self.Lz)
        self.u = np.zeros((self.nz, self.ny, self.nx, 3), dtype=np.float64)

    def idx_to_point(self, i, j, k):
        i = np.asarray(i, dtype=np.float64)
        j = np.asarray(j, dtype=np.float64)
        k = np.asarray(k, dtype=np.float64)
        i, j, k = np.broadcast_arrays(i, j, k)
        x = (i - (self.nx - 1) / 2.0) * (self.Lx / self.nx)
        y = (j - (self.ny - 1) / 2.0) * (self.Ly / self.ny)
        z = (k - (self.nz - 1) / 2.0) * (self.Lz / self.nz)
        return x, y, z

    def point_to_idx(self, x, y, z):
        ix = np.asarray(x) * self.nx / self.Lx + (self.nx - 1) / 2.0
        iy = np.asarray(y) * self.ny / self.Ly + (self.ny - 1) / 2.0
        iz = np.asarray(z) * self.nz / self.Lz + (self.nz - 1) / 2.0
        return ix, iy, iz

    def grid_points(self) -> np.ndarray:
        """按缓存展平顺序（iz 最快）返回所有网格节点坐标 (nx*ny*nz, 3)。"""
        iz, iy, ix = np.meshgrid(np.arange(self.nz), np.arange(self.ny),
                                 np.arange(self.nx), indexing="ij")
        x = (ix - (self.nx - 1) / 2.0) * (self.Lx / self.nx)
        y = (iy - (self.ny - 1) / 2.0) * (self.Ly / self.ny)
        z = (iz - (self.nz - 1) / 2.0) * (self.Lz / self.nz)
        pts = np.stack([x, y, z], axis=-1)  # (nz, ny, nx, 3)
        return pts.reshape(-1, 3)

    def trilinear(self, points: np.ndarray) -> np.ndarray:
        """批量三线性插值（ClampToEdge）。points (M,3) -> (M,3)。"""
        pts = np.asarray(points, dtype=np.float64)
        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
        ix, iy, iz = self.point_to_idx(x, y, z)
        x0 = np.clip(np.floor(ix).astype(np.int64), 0, self.nx - 1)
        y0 = np.clip(np.floor(iy).astype(np.int64), 0, self.ny - 1)
        z0 = np.clip(np.floor(iz).astype(np.int64), 0, self.nz - 1)
        x1 = np.clip(x0 + 1, 0, self.nx - 1)
        y1 = np.clip(y0 + 1, 0, self.ny - 1)
        z1 = np.clip(z0 + 1, 0, self.nz - 1)
        tx = np.clip(ix - x0, 0.0, 1.0)
        ty = np.clip(iy - y0, 0.0, 1.0)
        tz = np.clip(iz - z0, 0.0, 1.0)
        u = self.u
        c000 = u[z0, y0, x0]
        c001 = u[z0, y0, x1]
        c010 = u[z0, y1, x0]
        c011 = u[z0, y1, x1]
        c100 = u[z1, y0, x0]
        c101 = u[z1, y0, x1]
        c110 = u[z1, y1, x0]
        c111 = u[z1, y1, x1]
        # x 方向插值（4 条边）
        c00 = (1 - tx)[:, None] * c000 + tx[:, None] * c001
        c01 = (1 - tx)[:, None] * c010 + tx[:, None] * c011
        c10 = (1 - tx)[:, None] * c100 + tx[:, None] * c101
        c11 = (1 - tx)[:, None] * c110 + tx[:, None] * c111
        # y 方向
        c0 = (1 - ty)[:, None] * c00 + ty[:, None] * c01
        c1 = (1 - ty)[:, None] * c10 + ty[:, None] * c11
        # z 方向
        return (1 - tz)[:, None] * c0 + tz[:, None] * c1


class ScalarCache3D:
    """三维标量场（浓度/温度）网格缓存 (nz, ny, nx)，三线性插值。"""

    def __init__(self, nx: int, ny: int, nz: int, domain_size):
        self.nx = int(nx)
        self.ny = int(ny)
        self.nz = int(nz)
        if np.isscalar(domain_size):
            self.Lx = self.Ly = self.Lz = float(domain_size)
        else:
            self.Lx, self.Ly, self.Lz = (float(domain_size[0]), float(domain_size[1]),
                                         float(domain_size[2]))
        self.L = (self.Lx, self.Ly, self.Lz)
        self.q = np.zeros((self.nz, self.ny, self.nx), dtype=np.float64)

    def idx_to_point(self, i, j, k):
        i = np.asarray(i, dtype=np.float64)
        j = np.asarray(j, dtype=np.float64)
        k = np.asarray(k, dtype=np.float64)
        i, j, k = np.broadcast_arrays(i, j, k)
        x = (i - (self.nx - 1) / 2.0) * (self.Lx / self.nx)
        y = (j - (self.ny - 1) / 2.0) * (self.Ly / self.ny)
        z = (k - (self.nz - 1) / 2.0) * (self.Lz / self.nz)
        return x, y, z

    def trilinear(self, points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float64)
        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
        ix = x * self.nx / self.Lx + (self.nx - 1) / 2.0
        iy = y * self.ny / self.Ly + (self.ny - 1) / 2.0
        iz = z * self.nz / self.Lz + (self.nz - 1) / 2.0
        x0 = np.clip(np.floor(ix).astype(np.int64), 0, self.nx - 1)
        y0 = np.clip(np.floor(iy).astype(np.int64), 0, self.ny - 1)
        z0 = np.clip(np.floor(iz).astype(np.int64), 0, self.nz - 1)
        x1 = np.clip(x0 + 1, 0, self.nx - 1)
        y1 = np.clip(y0 + 1, 0, self.ny - 1)
        z1 = np.clip(z0 + 1, 0, self.nz - 1)
        tx = np.clip(ix - x0, 0.0, 1.0)
        ty = np.clip(iy - y0, 0.0, 1.0)
        tz = np.clip(iz - z0, 0.0, 1.0)
        q = self.q
        c00 = (1 - tx) * q[z0, y0, x0] + tx * q[z0, y0, x1]
        c01 = (1 - tx) * q[z0, y1, x0] + tx * q[z0, y1, x1]
        c10 = (1 - tx) * q[z1, y0, x0] + tx * q[z1, y0, x1]
        c11 = (1 - tx) * q[z1, y1, x0] + tx * q[z1, y1, x1]
        c0 = (1 - ty) * c00 + ty * c01
        c1 = (1 - ty) * c10 + ty * c11
        return (1 - tz) * c0 + tz * c1

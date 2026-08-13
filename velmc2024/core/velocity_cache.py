"""
velocity_cache.py —— 速度场网格缓存与点态查询（对齐作者 utils / SampleBuffer）

论文方法按"点态估计"设计，但基础实现把中间速度场缓存到均匀网格（caching 策略）。
查询时用双线性插值；越界用 ClampToEdge（取最近缓存点值）。

网格节点坐标与索引（对齐作者）：
    idx = (grid_res / domain_size) * x + (grid_res - 1) / 2
    x   = (idx - (grid_res - 1) / 2) * (domain_size / grid_res)
即域 [-L/2, L/2] 映射到索引 [0, grid_res-1]。
"""

from __future__ import annotations

import numpy as np


def _catmull(t, f0, f1, f2, f3):
    """1D Catmull-Rom 插值：t∈[0,1] 位于控制点 f1 与 f2 之间。
    f 可带尾随维（如向量分量 (M,2)）；t 自动广播到 f 的维度。"""
    t = np.asarray(t, dtype=np.float64)
    while t.ndim < np.asarray(f0).ndim:
        t = t[..., None]
    t2 = t * t
    t3 = t2 * t
    return 0.5 * (2.0 * f1 + (-f0 + f2) * t
                  + (2.0 * f0 - 5.0 * f1 + 4.0 * f2 - f3) * t2
                  + (-f0 + 3.0 * f1 - 3.0 * f2 + f3) * t3)


class VelocityCache:
    """二维速度场网格缓存（形状 (ny, nx, 2)），支持批量双线性点查询。"""

    def __init__(self, nx: int, ny: int, domain_size):
        self.nx = int(nx)
        self.ny = int(ny)
        if np.isscalar(domain_size):
            self.Lx = float(domain_size)
            self.Ly = float(domain_size)
        else:
            self.Lx, self.Ly = float(domain_size[0]), float(domain_size[1])
        self.L = (self.Lx, self.Ly)
        self.u = np.zeros((self.ny, self.nx, 2), dtype=np.float64)

    # ------------------------------------------------------------------ #
    def idx_to_point(self, i: np.ndarray, j: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """网格节点索引 -> 域坐标（i, j 自动广播到相同形状）。"""
        i = np.asarray(i, dtype=np.float64)
        j = np.asarray(j, dtype=np.float64)
        i, j = np.broadcast_arrays(i, j)
        x = (i - (self.nx - 1) / 2.0) * (self.Lx / self.nx)
        y = (j - (self.ny - 1) / 2.0) * (self.Ly / self.ny)
        return x, y

    def point_to_idx(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """域坐标 -> 连续索引（float）。"""
        ix = (np.asarray(x) * self.nx / self.Lx) + (self.nx - 1) / 2.0
        iy = (np.asarray(y) * self.ny / self.Ly) + (self.ny - 1) / 2.0
        return ix, iy

    # ------------------------------------------------------------------ #
    def set_values(self, values: np.ndarray):
        """直接写入缓存 (ny, nx, 2)。"""
        self.u[...] = values

    def evaluate(self, func) -> None:
        """用函数 func(x, y) -> (..., 2) 在网格节点上求值填充缓存。"""
        i = np.arange(self.nx)
        j = np.arange(self.ny)
        x, y = self.idx_to_point(i[:, None], j[None, :])  # x:(nx,ny), y:(nx,ny)
        vals = func(x, y)  # (nx, ny, 2)
        self.u[...] = np.transpose(vals, (1, 0, 2))

    # ------------------------------------------------------------------ #
    def bilinear(self, points: np.ndarray) -> np.ndarray:
        """批量双线性插值（ClampToEdge）。points: (M, 2) -> (M, 2)。

        注意 u 数组形状为 (ny, nx, 2)：第一维是 y 索引，第二维是 x 索引。
        """
        pts = np.asarray(points, dtype=np.float64)
        x, y = pts[:, 0], pts[:, 1]
        ix, iy = self.point_to_idx(x, y)
        # ClampToEdge
        x0 = np.clip(np.floor(ix).astype(np.int64), 0, self.nx - 1)
        y0 = np.clip(np.floor(iy).astype(np.int64), 0, self.ny - 1)
        x1 = np.clip(x0 + 1, 0, self.nx - 1)
        y1 = np.clip(y0 + 1, 0, self.ny - 1)
        tx = np.clip(ix - x0, 0.0, 1.0)
        ty = np.clip(iy - y0, 0.0, 1.0)
        u = self.u
        v00 = u[y0, x0]  # (M,2)
        v01 = u[y0, x1]
        v10 = u[y1, x0]
        v11 = u[y1, x1]
        out = ((1 - ty)[:, None] * ((1 - tx)[:, None] * v00 + tx[:, None] * v01)
               + ty[:, None] * ((1 - tx)[:, None] * v10 + tx[:, None] * v11))
        return out

    def catmull_rom(self, points: np.ndarray) -> np.ndarray:
        """批量双三次 Catmull-Rom 插值（ClampToEdge）。points: (M,2) -> (M,2)。

        比 bilinear 的数值耗散低很多（网格基准已验证 ~3.5×），用于半拉格朗日平流。
        注意 u 数组形状为 (ny, nx, 2)：第一维是 y 索引，第二维是 x 索引。
        """
        pts = np.asarray(points, dtype=np.float64)
        x, y = pts[:, 0], pts[:, 1]
        ix, iy = self.point_to_idx(x, y)
        # 定位左下角单元并钳制，保证 4×4 邻域在 [0, nx-1]×[0, ny-1] 内
        i0 = np.clip(np.floor(ix).astype(np.int64), 1, self.nx - 3)
        j0 = np.clip(np.floor(iy).astype(np.int64), 1, self.ny - 3)
        tx = np.clip(ix - i0, 0.0, 1.0)
        ty = np.clip(iy - j0, 0.0, 1.0)
        di = np.array([-1, 0, 1, 2])
        dj = np.array([-1, 0, 1, 2])
        I = np.clip(i0[:, None] + di[None, :], 0, self.nx - 1)  # (M,4)
        J = np.clip(j0[:, None] + dj[None, :], 0, self.ny - 1)  # (M,4)
        # 4×4 邻域 (M,4,4,2)：行 J、列 I
        neigh = self.u[J[:, :, None], I[:, None, :]]
        # 沿 x 对 4 行分别做 1D CR
        rows = np.stack([
            _catmull(tx, neigh[:, r, 0], neigh[:, r, 1], neigh[:, r, 2], neigh[:, r, 3])
            for r in range(4)
        ], axis=1)  # (M,4,2)
        # 沿 y 对 4 行的结果做 1D CR
        return _catmull(ty, rows[:, 0], rows[:, 1], rows[:, 2], rows[:, 3])


class ScalarCache:
    """标量场（浓度/温度）网格缓存，同样的双线性插值。"""

    def __init__(self, nx: int, ny: int, domain_size):
        self.nx = int(nx)
        self.ny = int(ny)
        if np.isscalar(domain_size):
            self.Lx = float(domain_size)
            self.Ly = float(domain_size)
        else:
            self.Lx, self.Ly = float(domain_size[0]), float(domain_size[1])
        self.L = (self.Lx, self.Ly)
        self.q = np.zeros((self.ny, self.nx), dtype=np.float64)

    def set_values(self, values: np.ndarray):
        self.q[...] = values

    def evaluate(self, func) -> None:
        i = np.arange(self.nx)
        j = np.arange(self.ny)
        x, y = self.idx_to_point(i[:, None], j[None, :])
        self.q[...] = np.transpose(func(x, y), (1, 0))

    def idx_to_point(self, i: np.ndarray, j: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        i = np.asarray(i, dtype=np.float64)
        j = np.asarray(j, dtype=np.float64)
        i, j = np.broadcast_arrays(i, j)
        x = (i - (self.nx - 1) / 2.0) * (self.Lx / self.nx)
        y = (j - (self.ny - 1) / 2.0) * (self.Ly / self.ny)
        return x, y

    def bilinear(self, points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float64)
        x, y = pts[:, 0], pts[:, 1]
        ix = (x * self.nx / self.Lx) + (self.nx - 1) / 2.0
        iy = (y * self.ny / self.Ly) + (self.ny - 1) / 2.0
        x0 = np.clip(np.floor(ix).astype(np.int64), 0, self.nx - 1)
        y0 = np.clip(np.floor(iy).astype(np.int64), 0, self.ny - 1)
        x1 = np.clip(x0 + 1, 0, self.nx - 1)
        y1 = np.clip(y0 + 1, 0, self.ny - 1)
        tx = np.clip(ix - x0, 0.0, 1.0)
        ty = np.clip(iy - y0, 0.0, 1.0)
        q = self.q
        return ((1 - ty) * ((1 - tx) * q[y0, x0] + tx * q[y0, x1])
                + ty * ((1 - tx) * q[y1, x0] + tx * q[y1, x1]))

    def catmull_rom(self, points: np.ndarray) -> np.ndarray:
        """批量双三次 Catmull-Rom 插值（ClampToEdge）。points: (M,2) -> (M,)。"""
        pts = np.asarray(points, dtype=np.float64)
        x, y = pts[:, 0], pts[:, 1]
        ix = (x * self.nx / self.Lx) + (self.nx - 1) / 2.0
        iy = (y * self.ny / self.Ly) + (self.ny - 1) / 2.0
        i0 = np.clip(np.floor(ix).astype(np.int64), 1, self.nx - 3)
        j0 = np.clip(np.floor(iy).astype(np.int64), 1, self.ny - 3)
        tx = np.clip(ix - i0, 0.0, 1.0)
        ty = np.clip(iy - j0, 0.0, 1.0)
        di = np.array([-1, 0, 1, 2])
        dj = np.array([-1, 0, 1, 2])
        I = np.clip(i0[:, None] + di[None, :], 0, self.nx - 1)  # (M,4)
        J = np.clip(j0[:, None] + dj[None, :], 0, self.ny - 1)  # (M,4)
        neigh = self.q[J[:, :, None], I[:, None, :]]  # (M,4,4)
        rows = np.stack([
            _catmull(tx, neigh[:, r, 0], neigh[:, r, 1], neigh[:, r, 2], neigh[:, r, 3])
            for r in range(4)
        ], axis=1)  # (M,4)
        return _catmull(ty, rows[:, 0], rows[:, 1], rows[:, 2], rows[:, 3])

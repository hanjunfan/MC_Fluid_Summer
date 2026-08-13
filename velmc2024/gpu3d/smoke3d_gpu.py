"""smoke3d_gpu.py —— PyTorch GPU 高精度三维浮力烟求解器。

算法与 velmc2024/core3d（CPU 版）一致，核心算子用 CUDA 张量：

每步：平流(RK3) -> 烟源注入 -> 浮力(Boussinesq) -> 扩散(高斯卷积) -> 投影(MC)

3D 投影（论文 §2.3 的 d=3 版本）：
    G = 1/(4πr),  ∇xG = r̂/(4πr²),  S = (3 r̂ r̂ᵀ − I)/(4πr³)
    体积项：球内采样 PDF∝1/r，r=R·√u，inv_pdf=2πR²r，
            ev = (1/N) Σ (3 r̂(r̂·Δu) − Δu) · R²/(2r²)
    伪边界项：盒 6 面均匀采样，ea = (A/B) Σ ∇xG · (n·Δu)
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# 几何
# --------------------------------------------------------------------------- #
class Box3D:
    """轴对齐正方盒 [-h, h]³（无边界域伪边界采样几何）。"""

    def __init__(self, size: float):
        self.size = float(size)
        self.h = self.size / 2.0
        self.area = 6.0 * self.size * self.size

    def contains(self, pts: torch.Tensor) -> torch.Tensor:
        """pts (...,3) -> 布尔 (...,)。"""
        return (pts.abs() <= self.h).all(dim=-1)

    def max_corner_distance(self, pts: torch.Tensor) -> torch.Tensor:
        """pts (...,3) -> 到盒最远角距离 (...,)。"""
        return torch.sqrt(((self.h + pts.abs()) ** 2).sum(dim=-1))


def sample_box_boundary(n: int, box: Box3D, device, generator):
    """盒 6 面均匀采样，返回 (pts (n,3), outward_normals (n,3))。"""
    h = box.h
    dtype = torch.float32
    areas = torch.full((6,), 4.0 * h * h, device=device)
    cdf = torch.cumsum(areas, dim=0) / areas.sum()
    u = torch.rand(n, device=device, generator=generator)
    face = torch.searchsorted(cdf, u).clamp(0, 5)
    a = torch.rand(n, device=device, generator=generator) * 2.0 - 1.0
    b = torch.rand(n, device=device, generator=generator) * 2.0 - 1.0

    pts = torch.zeros(n, 3, device=device, dtype=dtype)
    norms = torch.zeros(n, 3, device=device, dtype=dtype)
    for f, (nx, ny, nz) in enumerate([(-1.0, 0.0, 0.0), (1.0, 0.0, 0.0),
                                      (0.0, -1.0, 0.0), (0.0, 1.0, 0.0),
                                      (0.0, 0.0, -1.0), (0.0, 0.0, 1.0)]):
        m = face == f
        if not m.any():
            continue
        if f in (0, 1):
            pts[m, 0] = nx * h
            pts[m, 1] = a[m] * h
            pts[m, 2] = b[m] * h
        elif f in (2, 3):
            pts[m, 0] = a[m] * h
            pts[m, 1] = ny * h
            pts[m, 2] = b[m] * h
        else:
            pts[m, 0] = a[m] * h
            pts[m, 1] = b[m] * h
            pts[m, 2] = nz * h
        norms[m] = torch.tensor([nx, ny, nz], device=device, dtype=dtype)
    return pts, norms


# --------------------------------------------------------------------------- #
# 三线性插值（grid_sample）
# --------------------------------------------------------------------------- #
def make_interpolator(grid: torch.Tensor, L: float):
    """返回 f(points (M,3) 物理坐标) -> (M,C) 三线性插值。

    grid: (C, nz, ny, nx)；物理域 [-L/2, L/2]³。
    align_corners=True 要求归一化坐标 (2i-(n-1))/(n-1) 才精确落在像素 i，
    而网格点物理坐标对应 (2i-(n-1))/n，故需乘 n/(n-1) 修正，否则每步插值
    有 ~0.5 像素偏移，平流会持续耗散场量。
    """
    C = grid.shape[0]
    nz, ny, nx = grid.shape[1], grid.shape[2], grid.shape[3]
    g = grid.unsqueeze(0)  # (1, C, D, H, W)
    scale = torch.tensor([nx / (nx - 1.0), ny / (ny - 1.0), nz / (nz - 1.0)],
                         dtype=grid.dtype, device=grid.device)  # (x, y, z)

    def f(points: torch.Tensor) -> torch.Tensor:
        M = points.shape[0]
        coords = points * (2.0 / L) * scale  # [-L/2,L/2] -> [-1,1]，顺序 (x,y,z)
        flow = coords.reshape(1, 1, 1, M, 3)  # (N, D_out, H_out, W_out, 3)
        out = F.grid_sample(g, flow, mode="bilinear", align_corners=True,
                            padding_mode="border")
        return out.reshape(C, M).t().contiguous()  # (M, C)

    return f


# --------------------------------------------------------------------------- #
# 3D 投影（批量 MC）
# --------------------------------------------------------------------------- #
def project_batch(points: torch.Tensor, u_field, box: Box3D,
                  nvol: int, npsb: int, generator,
                  antithetic: bool = True, relax: float = 1.0) -> torch.Tensor:
    """对 points (P,3) 批量投影，返回投影后速度 (P,3)。"""
    P = points.shape[0]
    device = points.device
    u_x = u_field(points)                      # (P,3)
    R_p = box.max_corner_distance(points)      # (P,)

    # ---------------- 体积项 ---------------- #
    n_draw = max(1, nvol // 2) if antithetic else max(1, nvol)
    ur = torch.rand((P, n_draw), device=device, generator=generator)
    r = R_p[:, None] * torch.sqrt(ur)
    z = torch.rand((P, n_draw), device=device, generator=generator) * 2.0 - 1.0
    phi = torch.rand((P, n_draw), device=device, generator=generator) * (2.0 * math.pi)
    s = torch.sqrt(torch.clamp(1.0 - z * z, min=0.0))
    direction = torch.stack([s * torch.cos(phi), s * torch.sin(phi), z], dim=-1)
    r_vec = direction * r[..., None]           # (P,n_draw,3)
    inv_pdf = (2.0 * math.pi) * (R_p[:, None] ** 2) * r
    if antithetic:
        r_vec = torch.cat([r_vec, -r_vec], dim=1)
        inv_pdf = torch.cat([inv_pdf, inv_pdf], dim=1)
        S = 2 * n_draw
    else:
        S = n_draw
    n_eff = max(nvol, 1)

    y = points[:, None, :] + r_vec             # (P,S,3)
    vel_y = u_field(y.reshape(-1, 3)).reshape(P, S, 3)
    vel_diff = vel_y - u_x[:, None, :]
    in_box = box.contains(y.reshape(-1, 3)).reshape(P, S)
    rr = r_vec.norm(dim=-1).clamp(min=1e-12)
    r_hat = r_vec / rr[..., None]
    dot_ru = (r_hat * vel_diff).sum(dim=-1)
    kernel = 3.0 * dot_ru[..., None] * r_hat - vel_diff
    contrib = in_box[..., None] * (R_p[:, None] ** 2 / (2.0 * rr * rr))[..., None] * kernel
    ev = contrib.sum(dim=1) / n_eff             # (P,3)

    # ---------------- 伪边界项 ---------------- #
    ea = torch.zeros_like(u_x)
    if npsb > 0:
        bpts, bnorms = sample_box_boundary(npsb, box, device, generator)
        vel_b = u_field(bpts)                  # (B,3)
        ndot = (bnorms * vel_b).sum(-1)[None, :] \
            - torch.einsum("bj,pj->pb", bnorms, u_x)   # (P,B)
        r_vec_all = bpts[None, :, :] - points[:, None, :]  # (P,B,3)
        r_all = r_vec_all.norm(dim=-1).clamp(min=1e-5)
        g = r_vec_all / (4.0 * math.pi * (r_all ** 3))[..., None]
        ea = (box.area / npsb) * (ndot[..., None] * g).sum(dim=1)

    pressure_grad = -relax * (ev + ea)
    return u_x - pressure_grad


# --------------------------------------------------------------------------- #
# 平流 / 扩散
# --------------------------------------------------------------------------- #
def backtrace(points: torch.Tensor, u_field, dt: float) -> torch.Tensor:
    v1 = u_field(points)
    x1 = points - 0.5 * dt * v1
    v2 = u_field(x1)
    x2 = points - dt * (2.0 * v2 - v1)
    v3 = u_field(x2)
    return points - (dt / 6.0) * (v1 + 4.0 * v2 + v3)


def _gaussian_kernel_1d(ksize: int, sg: float, device, dtype):
    x = torch.arange(ksize, device=device, dtype=dtype) - ksize // 2
    k = torch.exp(-(x * x) / (2.0 * sg * sg))
    return k / k.sum()


def gaussian_blur3d(q: torch.Tensor, sg: float) -> torch.Tensor:
    """可分离 3D 高斯卷积。q: (nz,ny,nx) 或 (nz,ny,nx,3)。"""
    if sg <= 0:
        return q
    is_vec = q.dim() == 4
    if is_vec:
        q4 = q.permute(3, 0, 1, 2).unsqueeze(0)   # (1,3,D,H,W)
        C = 3
    else:
        q4 = q.unsqueeze(0).unsqueeze(0)           # (1,1,D,H,W)
        C = 1
    ksize = int(math.ceil(sg * 3.0)) * 2 + 1
    k1 = _gaussian_kernel_1d(ksize, sg, q.device, q.dtype)
    k3 = (k1.view(1, 1, 1, 1, ksize) * k1.view(1, 1, 1, ksize, 1)
          * k1.view(1, 1, ksize, 1, 1))            # (1,1,kz,ky,kx)
    pad = ksize // 2
    out = F.conv3d(q4, k3.repeat(C, 1, 1, 1, 1), padding=pad, groups=C)
    if is_vec:
        return out.squeeze(0).permute(1, 2, 3, 0).contiguous()  # (D,H,W,3)
    return out.squeeze(0).squeeze(0)               # (D,H,W)


# --------------------------------------------------------------------------- #
# 求解器
# --------------------------------------------------------------------------- #
class SmokeSolverGPU:
    def __init__(self, grid_res=128, domain_size=3.0, dt=0.02,
                 nvol=5000, npsb=2000,
                 concentration_rate=2.0, temperature_rate=3.0,
                 buoyancy_beta=1.0, gravity=(0.0, -1.0, 0.0),
                 sg_v=0.36, sg_c=0.11, sg_t=0.11,
                 projection_relax=0.15,
                 umax_clamp=1.5,
                 smoke_center=(0.0, -1.25, 0.0), smoke_half=(0.125, 0.125, 0.125),
                 chunk=4096, seed=0, verbose=True):
        self.nx = self.ny = self.nz = int(grid_res)
        self.L = float(domain_size)
        self.dt = float(dt)
        self.nvol = int(nvol)
        self.npsb = int(npsb)
        self.concentration_rate = float(concentration_rate)
        self.temperature_rate = float(temperature_rate)
        self.buoyancy_beta = float(buoyancy_beta)
        self.sg_v, self.sg_c, self.sg_t = float(sg_v), float(sg_c), float(sg_t)
        self.relax = float(projection_relax)
        self.umax_clamp = float(umax_clamp)
        self.chunk = int(chunk)
        self.verbose = verbose
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gravity = torch.tensor(gravity, dtype=torch.float32, device=self.device)
        self.smoke_center = torch.tensor(smoke_center, dtype=torch.float32, device=self.device)
        self.smoke_half = torch.tensor(smoke_half, dtype=torch.float32, device=self.device)
        self.generator = torch.Generator(device=self.device).manual_seed(seed)

        self.box = Box3D(self.L)
        # 网格布局：(C, nz, ny, nx)
        self.u = torch.zeros((3, self.nz, self.ny, self.nx), device=self.device)
        self.c = torch.zeros((1, self.nz, self.ny, self.nx), device=self.device)
        self.T = torch.zeros((1, self.nz, self.ny, self.nx), device=self.device)

        self.u_field = make_interpolator(self.u, self.L)
        self.c_field = make_interpolator(self.c, self.L)
        self.T_field = make_interpolator(self.T, self.L)

        self._grid_pts = self._make_grid_points()
        self.step_count = 0
        self.time = 0.0
        if self.verbose:
            print(f"GPU 求解器: 网格 ({self.nx},{self.ny},{self.nz}), dt={self.dt}, "
                  f"nvol={self.nvol}, npsb={self.npsb}, β={self.buoyancy_beta}, "
                  f"relax={self.relax}, 设备={self.device}")

    def _make_grid_points(self):
        n = self.nx
        i = torch.arange(n, device=self.device, dtype=torch.float32)
        x = (i - (n - 1) / 2.0) * (self.L / n)
        Z, Y, X = torch.meshgrid(x, x, x, indexing="ij")
        # Z 沿 axis0、Y 沿 axis1、X 沿 axis2 → 展平顺序 iz 最快（与网格 (nz,ny,nx) 一致）
        return torch.stack([X, Y, Z], dim=-1).reshape(-1, 3)  # (n³,3)

    # ------------------------------------------------------------------ #
    def _smoke_mask(self, pts):
        rel = pts - self.smoke_center
        return (rel.abs() <= self.smoke_half).all(dim=-1)

    def _apply_source(self, c, T):
        m = self._smoke_mask(self._grid_pts).reshape(self.nz, self.ny, self.nx)
        c[0][m] = torch.minimum(c[0][m] + self.dt * self.concentration_rate,
                                torch.tensor(1.0, device=c.device))
        T[0][m] = T[0][m] + (1.0 - math.exp(-self.temperature_rate * self.dt)) * (1.0 - T[0][m])

    def _apply_buoyancy(self, u, T):
        # u: (3, nz, ny, nx)；T: (nz, ny, nx)；浮力 accel = -β·T·gravity
        g = (-self.gravity).view(3, 1, 1, 1)
        accel = self.buoyancy_beta * T.unsqueeze(0) * g  # (3, nz, ny, nx)
        u += self.dt * accel

    def _project(self):
        pts = self._grid_pts
        out = torch.empty_like(self.u)          # (3, nz, ny, nx)
        flat = self.u  # 投影写回
        n = pts.shape[0]
        for s in range(0, n, self.chunk):
            sub = pts[s:s + self.chunk]
            proj = project_batch(sub, self.u_field, self.box, self.nvol, self.npsb,
                                 self.generator, relax=self.relax)
            # proj (P,3) -> 写回 (3, nz, ny, nx) 对应位置
            idx = torch.arange(s, s + sub.shape[0], device=self.device)
            iz = idx // (self.ny * self.nx)
            iy = (idx // self.nx) % self.ny
            ix = idx % self.nx
            for c in range(3):
                self.u[c, iz, iy, ix] = proj[:, c]

    # ------------------------------------------------------------------ #
    def step(self):
        dt = self.dt
        pts = self._grid_pts
        # 平流
        u_field = self.u_field
        c_field = self.c_field
        T_field = self.T_field
        xb = backtrace(pts, u_field, dt)         # (n³,3)
        u_adv = u_field(xb).t()                  # (3, n³)
        c_adv = c_field(xb).t()                  # (1, n³)
        T_adv = T_field(xb).t()                  # (1, n³)
        u_new = u_adv.reshape(3, self.nz, self.ny, self.nx).clone()
        c_new = c_adv.reshape(1, self.nz, self.ny, self.nx).clone()
        T_new = T_adv.reshape(1, self.nz, self.ny, self.nx).clone()

        # 源 + 浮力
        self._apply_source(c_new, T_new)
        self._apply_buoyancy(u_new, T_new[0])

        # 扩散
        if self.sg_v > 0:
            u_new = gaussian_blur3d(u_new.permute(1, 2, 3, 0), self.sg_v).permute(3, 0, 1, 2)
        if self.sg_c > 0:
            c_new = gaussian_blur3d(c_new[0], self.sg_c).unsqueeze(0)
        if self.sg_t > 0:
            T_new = gaussian_blur3d(T_new[0], self.sg_t).unsqueeze(0)

        # 投影
        self.u.copy_(u_new)
        self._project()
        # 速度钳制（防浮力反馈发散）：按模长等比缩放到上限，保持方向
        if self.umax_clamp > 0:
            self._clamp_velocity()
        self.c.copy_(c_new)
        self.T.copy_(T_new)

        self.step_count += 1
        self.time += dt

    def _clamp_velocity(self):
        mag = self.u.norm(dim=0, keepdim=True)          # (1, nz, ny, nx)
        if float(mag.max()) <= self.umax_clamp:
            return
        scale = (self.umax_clamp / mag.clamp(min=1e-12)).clamp(max=1.0)
        self.u.mul_(scale)

    def run(self, n_steps, save_every=0, out_dir=None):
        import os
        import numpy as np
        if save_every > 0 and out_dir:
            os.makedirs(out_dir, exist_ok=True)
        for _ in range(n_steps):
            self.step()
            if save_every > 0 and self.step_count % save_every == 0 and out_dir:
                path = os.path.join(out_dir, f"conc_{self.step_count:05d}.npy")
                np.save(path, self.c[0].cpu().numpy())
                if self.verbose:
                    umax = float(self.u.abs().max().cpu())
                    print(f"  t={self.time:.2f} step={self.step_count} 已保存 {path}  "
                          f"|u|max={umax:.3f}")

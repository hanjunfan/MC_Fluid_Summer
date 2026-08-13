"""
wob.py —— walk-on-boundary 边界投影（论文 Eq.16-17 + VPL 缓存，翻译自 velocity_fluids_optix.cu）

有障碍物时，投影还需要障碍边界上的 Neumann 边界积分项（自由滑移 / 无穿透）。
作者用 Sugimoto et al. 2023 的 walk-on-boundary 方法求解该边界积分方程，并配合
VPL（virtual point light）缓存：先为所有路径构建边界子路径（project_vpl_construct），
再对每个求值点做 gather（project_vpl_gather），从而把昂贵的边界路径在所有求值点间共享。

这里的障碍物统一用 Polygon 表示（Circle 先离散成多边形）。
"""

from __future__ import annotations

import numpy as np

from .geometry import Polygon
from .projection import estimate_volume_term, estimate_pseudo_boundary_term, dGdx_2d
from .sampling import uniform_dir_2d
from .velocity_cache import ScalarCache


# --------------------------------------------------------------------------- #
# 涡量 → 速度重建（论文的核心：平流涡量标量场，再用 Biot-Savart 重建速度）
# --------------------------------------------------------------------------- #
def reconstruct_velocity_from_vorticity(omega: ScalarCache, out, obstacles: list[Polygon],
                                        sim_box, num_samples: int = 0, rng=None,
                                        chunk: int = 256, w_thresh: float = 1e-3) -> None:
    """从涡量标量场重建无散速度（就地写入 out.u，形状 (ny,nx,2)）。

    对每个网格求值点 x：
        u_vort(x) = ∫_Ω ω(y) K(x-y) dy,   K(r) = (-r_y, r_x)/(2π r²)
    用确定性求积：对网格上 |ω|>w_thresh 的源格点直接求和（Biot-Savart）。
    障碍无穿透边界由后续投影的 WoB 边界项处理；盒子是计算窗口，伪边界项负责。

    说明：MC 重要性采样（PDF∝1/r）对局域化涡量方差太大，确定性求积对 CPU 网格
    更稳健（ω 支撑通常紧凑，cost ≈ P×N_src）。
    """
    nx, ny = out.nx, out.ny
    hx = out.Lx / nx
    hy = out.Ly / ny
    w = omega.q  # (ny, nx)
    sy, sx = np.nonzero(np.abs(w) > w_thresh)  # y 索引, x 索引
    if sx.size == 0:
        out.u[...] = 0.0
        return
    src_x, src_y = out.idx_to_point(sx, sy)     # (x, y) 坐标
    src_w = w[sy, sx].astype(np.float64)
    dA = hx * hy
    i = np.arange(nx)
    j = np.arange(ny)
    x, y = out.idx_to_point(i[:, None], j[None, :])
    # y 主序：使 reshape(ny,nx,2) 与缓存 [y,x] 读取约定一致
    grid_pts = np.stack([x.T.ravel(), y.T.ravel()], axis=1)
    P = grid_pts.shape[0]
    S = sx.size
    out_arr = np.empty((P, 2))
    src = np.stack([src_x, src_y], axis=1)  # (S, 2)
    for s0 in range(0, P, chunk):
        xp = grid_pts[s0:s0 + chunk]  # (C, 2)
        r = xp[:, None, :] - src[None, :, :]  # (C, S, 2), r = x - y
        r2 = np.sum(r * r, axis=-1) + 1e-12
        kx = -r[..., 1] / r2                  # (C, S)
        ky = r[..., 0] / r2
        ux = (src_w[None, :] * kx).sum(axis=1)
        uy = (src_w[None, :] * ky).sum(axis=1)
        out_arr[s0:s0 + chunk] = np.stack([ux, uy], axis=1) * (dA / (2.0 * np.pi))
    out.u[...] = out_arr.reshape(ny, nx, 2)


# --------------------------------------------------------------------------- #
# 工具：整条直线与多个障碍物的交点（WoB 路径的射线求交）
# --------------------------------------------------------------------------- #
def line_intersection_boundary_sample(obstacles: list[Polygon], origin: np.ndarray,
                                      origin_ob: int, origin_seg: int,
                                      rng: np.random.Generator):
    """作者 line_intersection_boundary_sample 的 Python 版。

    从 origin（位于某障碍边界上）沿随机方向的整条直线求所有交点，
    返回 (point, num_intersections, ob_idx, seg_idx)。
    - 排除起点所在线段（origin_ob, origin_seg），避免 t≈0 自相交
    - 交点按等权均匀随机选一个作为返回点
    """
    direction = uniform_dir_2d(1, rng)[0]
    hits = []  # (|t|, point, ob_idx, seg_idx)
    for oi, ob in enumerate(obstacles):
        skip = origin_seg if oi == origin_ob else None
        for t, point, si in ob.line_intersections(origin, direction, skip_seg=skip):
            hits.append((abs(t), point, oi, si))
    if not hits:
        # 不应发生（origin 在障碍上，直线必与它再次相交）；返回安全值
        return None, 0, -1, -1
    hits.sort(key=lambda h: h[0])
    k = rng.integers(0, len(hits))
    _, point, oi, si = hits[k]
    return point, len(hits), oi, si


# --------------------------------------------------------------------------- #
# VPL 构建
# --------------------------------------------------------------------------- #
def project_vpl_construct(obstacles: list[Polygon], get_velocity, sim_box,
                          num_paths: int, path_length: int,
                          num_volume_samples_indirect: int,
                          num_pseudo_boundary_samples_indirect: int,
                          rng: np.random.Generator,
                          solid_velocity=(0.0, 0.0),
                          dGdx_reg: float = 1e-5):
    """构建 num_paths 条 VPL 路径。

    返回展平的 (vpl_pos (M,2), vpl_val (M,))，M = num_paths*(path_length+1)。
    """
    solid = np.asarray(solid_velocity, dtype=np.float64)
    positions = []
    values = []
    for _ in range(num_paths):
        # 从障碍边界均匀采样一个源点
        oi = rng.integers(0, len(obstacles))
        ob = obstacles[oi]
        pts, norms, segs = ob.sample_boundary_with_idx(1, rng)
        source_point = pts[0]
        source_normal = norms[0]
        source_seg = int(segs[0])
        source_inv_pdf = ob.perimeter
        if not sim_box.contains(source_point[None, :])[0]:
            source_inv_pdf = 0.0

        source_vel = get_velocity(source_point[None, :])[0]
        source_terms = (estimate_volume_term(source_point, source_vel, get_velocity, sim_box, obstacles,
                                             num_volume_samples_indirect, rng, True, dGdx_reg)
                        + estimate_pseudo_boundary_term(source_point, source_vel, get_velocity, sim_box, obstacles,
                                                        num_pseudo_boundary_samples_indirect, rng, dGdx_reg))

        # length 0.5 与 1 的贡献（vpl[0]）
        val0 = source_inv_pdf * float(np.dot(source_normal, source_vel + 2.0 * (source_terms - solid)))
        positions.append(source_point.copy())
        values.append(val0)

        # 射线求交 → 下一边界点
        point_x, num_int, ob_x, seg_x = line_intersection_boundary_sample(
            obstacles, source_point, oi, source_seg, rng)
        if point_x is None:
            for _ in range(path_length):
                positions.append(source_point.copy())
                values.append(0.0)
            continue
        if not sim_box.contains(point_x[None, :])[0]:
            num_int = 0
        normal_x = obstacles[ob_x].normals[seg_x]
        sign_x = 1.0 if float(np.dot(normal_x, point_x - source_point)) > 0 else -1.0
        si_x = sign_x * num_int
        vel_x = get_velocity(point_x[None, :])[0]

        mult_n_5 = source_inv_pdf * si_x * float(np.dot(source_normal, source_vel - vel_x))
        mult_n = source_inv_pdf * si_x * 2.0 * float(np.dot(source_normal, source_terms + source_vel - solid))

        # vpl[1]
        positions.append(point_x.copy())
        values.append(mult_n_5 + mult_n)

        point_y = point_x
        ob_y, seg_y = ob_x, seg_x
        for i in range(2, path_length + 1):
            point_x, num_int, ob_x, seg_x = line_intersection_boundary_sample(
                obstacles, point_y, ob_y, seg_y, rng)
            if point_x is None:
                positions.append(point_y.copy())
                values.append(0.0)
                break
            if not sim_box.contains(point_x[None, :])[0]:
                num_int = 0
            normal_x = obstacles[ob_x].normals[seg_x]
            sign_x = 1.0 if float(np.dot(normal_x, point_x - point_y)) > 0 else -1.0
            si_x = sign_x * num_int

            mult_n_5 *= si_x
            mult_n *= si_x
            if i == path_length - 1:
                mult_n *= 0.5
            elif i == path_length:
                mult_n_5 *= 0.5
                mult_n = 0.0
            positions.append(point_x.copy())
            values.append(mult_n_5 + mult_n)
            point_y = point_x
            ob_y, seg_y = ob_x, seg_x

        while len(values) < len(positions):
            positions.append(point_y.copy())
            values.append(0.0)
    return np.array(positions), np.array(values)


# --------------------------------------------------------------------------- #
# 直接边界项（length-0）与 VPL gather
# --------------------------------------------------------------------------- #
def estimate_boundary_term(x: np.ndarray, x_vel: np.ndarray, obstacles: list[Polygon],
                           sim_box, rng: np.random.Generator,
                           dGdx_reg: float = 1e-5) -> np.ndarray:
    """直接边界项：均匀采样一个边界点，贡献 = inv_pdf * n·(x_vel) * ∇G(b - x)。"""
    oi = rng.integers(0, len(obstacles))
    ob = obstacles[oi]
    pts, norms = ob.sample_boundary(1, rng)
    point = pts[0]
    normal = norms[0]
    inv_pdf = ob.perimeter
    if not sim_box.contains(point[None, :])[0]:
        inv_pdf = 0.0
    g = dGdx_2d(point - x, dGdx_reg)
    return inv_pdf * float(np.dot(normal, x_vel)) * g


def _sample_union_boundary(obstacles, n, rng):
    """在多个障碍边界的并集上均匀采样（概率 ∝ 周长），返回 (points, normals, inv_pdf=总周长)。"""
    perims = np.array([ob.perimeter for ob in obstacles])
    total = float(perims.sum())
    cdf = np.cumsum(perims) / total
    u = rng.random(n)
    oi = np.searchsorted(cdf, u)
    pts = np.empty((n, 2))
    norms = np.empty((n, 2))
    for idx, ob in enumerate(obstacles):
        m = oi == idx
        if m.any():
            p, nn = ob.sample_boundary(int(m.sum()), rng)
            pts[m] = p
            norms[m] = nn
    return pts, norms, total


def project_vpl_gather(x: np.ndarray, x_vel: np.ndarray, get_velocity,
                       obstacles: list[Polygon], sim_box,
                       vpl_pos, vpl_val, num_paths: int, path_length: int,
                       num_volume_samples_direct: int,
                       num_pseudo_boundary_samples_direct: int,
                       rng: np.random.Generator,
                       solid_velocity=(0.0, 0.0),
                       dGdx_reg: float = 1e-5) -> np.ndarray:
    """对单个求值点 x 用共享 VPL 做完整投影（含障碍边界项）。返回 u4(x)。

    压力梯度：pressure_grad = -(E_V + E_A) - (直接边界项 + VPL记录项)/num_paths
    （边界项负号已按解析势流测试校准）
    """
    x = np.asarray(x, dtype=np.float64)
    ev = estimate_volume_term(x, x_vel, get_velocity, sim_box, obstacles,
                              num_volume_samples_direct, rng, True, dGdx_reg)
    ea = estimate_pseudo_boundary_term(x, x_vel, get_velocity, sim_box, obstacles,
                                       num_pseudo_boundary_samples_direct, rng, dGdx_reg)
    pressure_grad = -(ev + ea)

    if obstacles and num_paths > 0:
        # 直接边界项（向量化）
        bpts, bnorms, binv = _sample_union_boundary(obstacles, num_paths, rng)
        in_box = sim_box.contains(bpts).astype(np.float64)
        gd = dGdx_2d(bpts - x[None, :], dGdx_reg)
        bsum = np.sum((binv * in_box * (bnorms @ x_vel))[:, None] * gd, axis=0)
        # VPL 记录（向量化）
        if vpl_val.size:
            gv = dGdx_2d(vpl_pos - x[None, :], dGdx_reg)
            bsum = bsum + vpl_val @ gv
        pressure_grad -= bsum / num_paths

    return x_vel - pressure_grad


def project_grid_vpl(cache, obstacles, sim_box, vpl_pos, vpl_val, num_paths, path_length,
                     num_volume_samples_direct, num_pseudo_boundary_samples_direct,
                     rng, solid_velocity=(0.0, 0.0), dGdx_reg=1e-5) -> None:
    """用共享 VPL 对整个网格缓存投影（就地更新 cache.u）。"""
    nx, ny = cache.nx, cache.ny
    solid = np.asarray(solid_velocity, dtype=np.float64)

    def get_velocity(pts):
        return cache.bilinear(pts)

    i = np.arange(nx)
    j = np.arange(ny)
    x, y = cache.idx_to_point(i[:, None], j[None, :])
    # y 主序：使 reshape(ny,nx,2) 与缓存 [y,x] 读取约定一致
    grid_pts = np.stack([x.T.ravel(), y.T.ravel()], axis=1)
    n_pts = grid_pts.shape[0]
    out = np.empty((n_pts, 2))
    for k in range(n_pts):
        p = grid_pts[k]
        xv = cache.bilinear(p[None, :])[0]
        out[k] = project_vpl_gather(p, xv, get_velocity, obstacles, sim_box,
                                    vpl_pos, vpl_val, num_paths, path_length,
                                    num_volume_samples_direct, num_pseudo_boundary_samples_direct,
                                    rng, solid, dGdx_reg)

    # 障碍内部赋固体速度
    if obstacles:
        w = np.zeros(n_pts)
        for ob in obstacles:
            w = w + ob.signed_winding(grid_pts)
        inside = np.abs(w) >= 0.5
        out[inside] = solid
    cache.u[...] = out.reshape(ny, nx, 2)


def project_grid_vpl_batched(cache, obstacles, sim_box, vpl_pos, vpl_val,
                             num_paths: int, path_length: int,
                             num_volume_samples_direct: int,
                             num_pseudo_boundary_samples_direct: int,
                             rng: np.random.Generator,
                             solid_velocity=(0.0, 0.0), dGdx_reg: float = 1e-5,
                             chunk: int = 128,
                             use_box_boundary: bool = True,
                             relax: float = 1.0) -> None:
    """批量向量化：对整个网格缓存投影（就地更新 cache.u）.

    把 chunk 个网格节点的所有样本合并成一次大 numpy 运算，CPU 上显著加速。
    """
    nx, ny = cache.nx, cache.ny
    solid = np.asarray(solid_velocity, dtype=np.float64)
    i = np.arange(nx)
    j = np.arange(ny)
    x, y = cache.idx_to_point(i[:, None], j[None, :])
    # y 主序：使 reshape(ny,nx,2) 与缓存 [y,x] 读取约定一致
    grid_pts = np.stack([x.T.ravel(), y.T.ravel()], axis=1)
    P = grid_pts.shape[0]
    ux = cache.bilinear(grid_pts)  # (P,2) 各节点原速度
    R_p = np.array([sim_box.max_corner_distance(p) for p in grid_pts])  # (P,)

    if obstacles:
        # 预计算障碍缠绕数（用于乘子/内部置 solid）
        w_all = np.zeros(P)
        for ob in obstacles:
            w_all = w_all + ob.signed_winding(grid_pts)
        inside_all = np.abs(w_all) >= 0.5
    else:
        inside_all = np.zeros(P, dtype=bool)

    out = np.empty((P, 2))
    for s in range(0, P, chunk):
        idx = slice(s, s + chunk)
        pts = grid_pts[idx]
        xv = ux[idx]
        Rc = R_p[idx]
        Pc = pts.shape[0]
        rng_local = np.random.default_rng(rng.integers(0, 2**31 - 1))

        # ---------------- 体积项（批量 + antithetic）---------------- #
        n_draw = max(1, num_volume_samples_direct // 2)
        ur = rng_local.random((Pc, n_draw))
        th = rng_local.uniform(0, 2 * np.pi, (Pc, n_draw))
        r = Rc[:, None] * ur
        r_vec = np.stack([r * np.cos(th), r * np.sin(th)], axis=-1)
        inv_pdf = 2 * np.pi * Rc[:, None] * r
        r_vec = np.concatenate([r_vec, -r_vec], axis=1)
        inv_pdf = np.concatenate([inv_pdf, inv_pdf], axis=1)
        S = 2 * n_draw
        ypts = pts[:, None, :] + r_vec
        yf = ypts.reshape(-1, 2)
        vel_y = cache.bilinear(yf).reshape(Pc, S, 2)
        vel_diff = vel_y - xv[:, None, :]
        in_box = sim_box.contains(yf).reshape(Pc, S)
        if obstacles:
            wv = np.zeros(Pc * S)
            for ob in obstacles:
                wv = wv + ob.signed_winding(yf)
            mult = in_box * (np.abs(wv) < 0.5).reshape(Pc, S)
        else:
            mult = in_box
        rr = np.maximum(np.linalg.norm(r_vec, axis=-1), 1e-12)
        r_hat = r_vec / rr[..., None]
        dot_ru = np.einsum("pij,pij->pi", r_hat, vel_diff)
        kernel = 2.0 * dot_ru[..., None] * r_hat - vel_diff
        ev = np.sum(mult[..., None] * (inv_pdf / (rr * rr))[..., None] * kernel, axis=1) / (
            num_volume_samples_direct * 2 * np.pi)

        # ---------------- 伪边界项（批量）---------------- #
        # 盒子边界项：对封闭盒（cohomology）强制 n·u 匹配（反射边界）。入流场景
        # （Karman）入口有法向来流 (1,0)，此项会强制入口法向速度→0，与 Dirichlet
        # 入流矛盾并逐时放大导致速度发散，故入流场景跳过（use_box_boundary=False）。
        ea = np.zeros((Pc, 2))
        if use_box_boundary and num_pseudo_boundary_samples_direct > 0:
            bpts, bnorms, binv = _sample_union_boundary([sim_box], num_pseudo_boundary_samples_direct, rng_local)
            gd = dGdx_2d(bpts[None, :, :] - pts[:, None, :], dGdx_reg)  # (Pc, B, 2)
            vel_b = cache.bilinear(bpts)  # (B,2)
            ndot = np.einsum("bj,pbj->pb", bnorms, vel_b[None, :, :] - xv[:, None, :])  # (Pc,B)
            ea = np.sum((binv * ndot)[:, :, None] * gd, axis=1) / num_pseudo_boundary_samples_direct

        # ---------------- 直接边界项（批量，每节点独立采样）---------------- #
        bsum = np.zeros((Pc, 2))
        if obstacles and num_paths > 0:
            # 直接边界项：对每个节点采样 num_paths 个边界点
            # 为控内存，分小批
            dsum = np.zeros((Pc, 2))
            sub = 256
            for b0 in range(0, num_paths, sub):
                b1 = min(b0 + sub, num_paths)
                nb = b1 - b0
                bpts, bnorms, binv = _sample_union_boundary(obstacles, nb, rng_local)
                in_box_b = sim_box.contains(bpts).astype(np.float64)
                gd = dGdx_2d(bpts[None, :, :] - pts[:, None, :], dGdx_reg)  # (Pc,nb,2)
                # n·u(x)：normals 与 各节点速度
                ndot = np.einsum("ij,pj->pi", bnorms, xv)  # (Pc,nb)
                dsum += np.sum((binv * in_box_b * ndot)[:, :, None] * gd, axis=1)
            # VPL 记录
            if vpl_val.size:
                gv = dGdx_2d(vpl_pos[None, :, :] - pts[:, None, :], dGdx_reg)  # (Pc,M,2)
                bsum = dsum + np.einsum("m,pmj->pj", vpl_val, gv)
            else:
                bsum = dsum
            pg = -(ev + ea) - bsum / num_paths
        else:
            pg = -(ev + ea)

        # relax 松弛：MC 投影方差在入流场景（Karman）会随平流正反馈放大而发散，
        # 用 relax<1 阻尼（保留障碍无穿透主体但抑制噪声放大）。
        out[idx] = xv - relax * pg

    out[inside_all] = solid
    cache.u[...] = out.reshape(ny, nx, 2)

"""
compare_error.py —— 误差分析：蒙特卡洛求解器 vs 传统网格法基准

对同一场景、同一初始条件分别运行两个求解器，在匹配时间点比较：
  - 浓度场 RMSE（流体区域）
  - 速度场 RMSE
  - 误差随时间的演化
输出：results/<scene>_error.png 与误差日志。

注意：两种方法是不同范式（随机 vs 确定性），误差来自多个来源：
MC 噪声（样本数）、投影盒截断、平流插值耗散等。报告中会讨论。
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

from velmc2024.solver.scenes import CohomologyScene, KarmanScene, CylinderCollisionScene  # noqa: E402
from velmc2024.solver.mc_solver import MCFluidSolver                  # noqa: E402
from velmc2024.reference.grid_solver import GridFluidSolver           # noqa: E402


def fluid_mask(solver):
    """流体区域掩膜：障碍外 + 比较盒内（用 MC 的仿真盒）。"""
    pts = None
    i = np.arange(solver.u_cache.nx)
    j = np.arange(solver.u_cache.ny)
    x, y = solver.u_cache.idx_to_point(i[:, None], j[None, :])
    # y 主序：与缓存 [y,x] 的 ravel 平铺索引一致
    pts = np.stack([x.T.ravel(), y.T.ravel()], axis=1)
    m = np.ones(pts.shape[0], dtype=bool)
    for ob in solver.scene.obstacles:
        m &= ~ob.inside(pts)
    # 与 MC 仿真盒对齐的比较区域
    L = getattr(solver.scene, "compare_domain_size", None) or (solver.scene.domain_size if np.isscalar(solver.scene.domain_size) else solver.scene.domain_size[0])
    if np.isscalar(L):
        hx = hy = L / 2
    else:
        hx, hy = L[0] / 2, L[1] / 2
    m &= (np.abs(pts[:, 0]) <= hx * 0.95) & (np.abs(pts[:, 1]) <= hy * 0.95)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", choices=["cohomology", "karman", "cylinder"], default="cohomology")
    ap.add_argument("--grid", type=int, default=56)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--mc-paths", type=int, default=250)
    ap.add_argument("--mc-nvol", type=int, default=700)
    ap.add_argument("--out", type=str, default="results/compare_cohomology")
    args = ap.parse_args()

    def make_scene():
        if args.scene == "cohomology":
            return CohomologyScene(grid_res=args.grid, dt=0.05, time_steps=args.steps)
        if args.scene == "cylinder":
            return CylinderCollisionScene(grid_res=args.grid, dt=0.05, time_steps=args.steps)
        return KarmanScene(viscosity=0.01, grid_res=(args.grid * 2, args.grid),
                           dt=0.02, time_steps=args.steps)

    scene0 = make_scene()
    print(f"场景: {scene0.describe()}")

    # MC 求解器（cohomology/cylinder 用涡量平流模式；karman 需粘性+入流用速度平流模式）
    mc = MCFluidSolver(make_scene(), num_paths=args.mc_paths,
                       num_volume_samples_direct=args.mc_nvol,
                       num_pseudo_boundary_samples_direct=args.mc_nvol,
                       advect_vorticity=(args.scene in ("cohomology", "cylinder")))
    # 网格基准（同分辨率；cohomology 用更大盒近似无边界；cylinder 封闭盒+圆柱障碍同盒对比）
    if args.scene == "cohomology":
        grid_scene = make_scene()
        grid_scene.domain_size = scene0.domain_size * 1.8
        grid_scene.grid_res = (int(args.grid * 1.8), int(args.grid * 1.8))
        grid = GridFluidSolver(grid_scene)
    else:
        grid = GridFluidSolver(make_scene())

    save_every = max(1, args.steps // 15)
    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    logfile = out_dir / "error.log"

    times, err_c, err_u, max_u, max_u_mc, err_w, rel_u = [], [], [], [], [], [], []
    t0 = time.perf_counter()
    last = {"m": None, "cm": None, "cg": None, "um": None, "ug": None}
    for s in range(1, args.steps + 1):
        mc.step()
        grid.step()
        if s % save_every == 0 or s == 1:
            # 把网格速度/浓度插值到 MC 网格
            m = fluid_mask(mc)
            pts = mc_grid_points(mc)
            cm = mc.c_cache.q.ravel()[m]
            cg = grid.c_cache.bilinear(pts).ravel()[m]
            u_m = mc.u_cache.u.reshape(-1, 2)[m]
            u_g = grid.u_cache.bilinear(pts).reshape(-1, 2)[m]
            rc = float(np.sqrt(np.mean((cm - cg) ** 2)))
            ru = float(np.sqrt(np.mean(np.sum((u_m - u_g) ** 2, axis=1))))
            norm = float(np.sqrt(np.mean(np.sum(u_g ** 2, axis=1))) + 1e-12)
            norm_mc = float(np.sqrt(np.mean(np.sum(u_m ** 2, axis=1))) + 1e-12)
            # 涡量（速度的旋度，中心差分）——无粘场景物理量守恒，更有判别力
            wm = _curl(mc.u_cache)
            wg_full = _curl_grid(grid)
            wm_m = wm.ravel()[m]
            wg_m = _interp_grid(grid, wg_full, pts).ravel()[m]
            rw = float(np.sqrt(np.mean((wm_m - wg_m) ** 2)))
            # 相对速度误差：仅在参考速度显著区域（|u|>0.05）
            strong = np.sum(u_g ** 2, axis=1) > 0.05 ** 2
            rrel = float(np.sqrt(np.mean(np.sum((u_m - u_g) ** 2, axis=1)[strong]))) if strong.any() else float("nan")
            times.append(mc.time)
            err_c.append(rc); err_u.append(ru); max_u.append(norm)
            max_u_mc.append(norm_mc); err_w.append(rw); rel_u.append(rrel)
            last["m"] = m; last["cm"] = cm; last["cg"] = cg; last["um"] = u_m; last["ug"] = u_g
            with open(logfile, "a", encoding="utf-8") as f:
                f.write(f"t={mc.time:.2f} step={s}: 浓度RMSE={rc:.5f}, 速度RMSE={ru:.5f}, "
                        f"|u|ref={norm:.4f}, |u|MC={norm_mc:.4f}, 涡量RMSE={rw:.5f}, 相对速度RMSE={rrel:.4f}\n")
            print(f"  t={mc.time:.2f} step={s}: 浓度RMSE={rc:.5f}, 速度RMSE={ru:.5f}, "
                  f"涡量RMSE={rw:.5f}, |u|MC={norm_mc:.4f} ({time.perf_counter()-t0:.0f}s)")
    print(f"总耗时 {time.perf_counter()-t0:.0f}s")

    # 画图（误差演化 + 最终空间误差分布）
    fig = plt.figure(figsize=(13, 6))
    gs = fig.add_gridspec(2, 4)
    ax1 = fig.add_subplot(gs[0, :2])
    ax2 = fig.add_subplot(gs[0, 2:])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    ax5 = fig.add_subplot(gs[1, 2:])
    ax1.plot(times, err_c, "o-", label="浓度 RMSE")
    ax1.set_xlabel("t"); ax1.set_ylabel("浓度 RMSE"); ax1.grid(alpha=0.3); ax1.legend()
    ax2.plot(times, err_u, "o-", label="速度 RMSE")
    ax2.plot(times, [e / n for e, n in zip(err_u, max_u)], "s--", label="速度RMSE/|u|ref")
    ax2.plot(times, max_u, "^-", label="|u|ref(网格)")
    ax2.plot(times, max_u_mc, "v--", label="|u|MC")
    ax2.set_xlabel("t"); ax2.set_ylabel("速度 RMSE"); ax2.grid(alpha=0.3); ax2.legend(fontsize=8)
    # 涡量误差
    ax5.plot(times, err_w, "o-", label="涡量 RMSE")
    ax5.plot(times, rel_u, "s--", label="相对速度RMSE(|u|>0.05)")
    ax5.set_xlabel("t"); ax5.set_ylabel("RMSE"); ax5.grid(alpha=0.3); ax5.legend(fontsize=8)
    # 最终空间误差图
    m = last["m"]; cm = last["cm"]; cg = last["cg"]
    nx, ny = mc.u_cache.nx, mc.u_cache.ny
    cmap = np.full((nx, ny), np.nan); cmap.ravel()[m] = cm - cg
    if cmap.shape[0] != 48 or cmap.shape[1] != 48:
        pass
    im3 = ax3.imshow(cmap.T, origin="lower", cmap="RdBu_r", vmin=-0.2, vmax=0.2,
                     extent=[-mc.scene.domain_size/2, mc.scene.domain_size/2] * 2 if np.isscalar(mc.scene.domain_size) else [-1.9, 1.9, -0.95, 0.95])
    ax3.set_title(f"最终浓度差 MC−grid (t={times[-1]:.1f})"); ax3.figure.colorbar(im3, ax=ax3, fraction=0.046)
    uerr = np.full((nx, ny), np.nan); uerr.ravel()[m] = np.sqrt(np.sum((last["um"] - last["ug"]) ** 2, axis=1))
    im4 = ax4.imshow(uerr.T, origin="lower", cmap="hot", vmin=0, vmax=0.5,
                     extent=[-mc.scene.domain_size/2, mc.scene.domain_size/2] * 2 if np.isscalar(mc.scene.domain_size) else [-1.9, 1.9, -0.95, 0.95])
    ax4.set_title("最终速度误差 |u_MC−u_grid|"); ax4.figure.colorbar(im4, ax=ax4, fraction=0.046)
    # 障碍物叠加
    for ax in (ax3, ax4):
        for ob in mc.scene.obstacles:
            xs = np.append(ob.vertices[:, 0], ob.vertices[0, 0])
            ys = np.append(ob.vertices[:, 1], ob.vertices[0, 1])
            ax.plot(xs, ys, "k-", lw=1.5)
    fig.suptitle(f"{args.scene} 误差：MC vs 网格基准（含涡量/相对误差/空间分布）")
    fig.tight_layout()
    fig.savefig(out_dir / "error.png", dpi=140)
    print(f"图已保存: {out_dir / 'error.png'}")


def _curl(u_cache):
    """cell 中心速度的旋度 ∂v/∂x − ∂u/∂y（MC 网格，(nx,ny)）。"""
    u = u_cache.u  # (nx, ny, 2)
    hx, hy = u_cache.Lx / u_cache.nx, u_cache.Ly / u_cache.ny
    dvdx = (u[2:, 1:-1, 1] - u[:-2, 1:-1, 1]) / (2 * hx)
    dudy = (u[1:-1, 2:, 0] - u[1:-1, :-2, 0]) / (2 * hy)
    w = np.zeros(u.shape[:2])
    w[1:-1, 1:-1] = dvdx - dudy
    return w


def _curl_grid(grid, pts=None):
    """网格 cell 中心速度旋度（与 _curl 同公式，返回 (nx,ny) 或给定点插值）。"""
    u_cell = 0.5 * (grid.u_f[:-1, :] + grid.u_f[1:, :])
    v_cell = 0.5 * (grid.v_f[:, :-1] + grid.v_f[:, 1:])
    hx, hy = grid.hx, grid.hy
    w = np.zeros((grid.nx, grid.ny))
    w[1:-1, 1:-1] = ((v_cell[2:, 1:-1] - v_cell[:-2, 1:-1]) / (2 * hx)
                     - (u_cell[1:-1, 2:] - u_cell[1:-1, :-2]) / (2 * hy))
    return w


def _interp_grid(grid, w, pts):
    """在 pts (N,2) 处双线性插值网格标量场 w (nx,ny)。"""
    x = pts[:, 0]; y = pts[:, 1]
    fi = x / grid.hx + (grid.nx - 1) / 2.0
    fj = y / grid.hy + (grid.ny - 1) / 2.0
    i0 = np.clip(np.floor(fi).astype(int), 0, grid.nx - 2)
    i1 = i0 + 1
    j0 = np.clip(np.floor(fj).astype(int), 0, grid.ny - 2)
    j1 = j0 + 1
    tx = np.clip(fi - i0, 0, 1)
    ty = np.clip(fj - j0, 0, 1)
    return ((1 - ty) * ((1 - tx) * w[i0, j0] + tx * w[i1, j0])
            + ty * ((1 - tx) * w[i0, j1] + tx * w[i1, j1]))


def mc_grid_points(mc):
    i = np.arange(mc.u_cache.nx)
    j = np.arange(mc.u_cache.ny)
    x, y = mc.u_cache.idx_to_point(i[:, None], j[None, :])
    # y 主序：与缓存 [y,x] 的 ravel 平铺索引一致
    return np.stack([x.T.ravel(), y.T.ravel()], axis=1)


if __name__ == "__main__":
    main()

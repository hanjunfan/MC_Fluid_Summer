"""
projection_convergence.py —— 阶段1：无边界 MC 投影验证 + 收敛测试（对照论文 Fig 6）

三组检验：
  A) 纯梯度场 u=(x,y)：投影后应 ≈ 0（投影算子杀死梯度分量）
  B) 无散度场 u=(-y,x)：投影后应 ≈ 不变（保持无散度分量）
  C) 论文 Fig 6 测试场 u=min(|x|,R)·x̂：RMSE vs 高样本参考，应呈 1/√N 收敛

运行：python velmc2024/experiments/projection_convergence.py
输出：results/projection_convergence.png
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# Windows 控制台默认 GBK，无法打印部分 Unicode；强制 UTF-8
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

from velmc2024.core.geometry import Rectangle          # noqa: E402
from velmc2024.core.projection import project_batch_vectorized  # noqa: E402


def log(msg: str = ""):
    print(msg)
    with open(ROOT / "results" / "projection_convergence.log", "a", encoding="utf-8") as f:
        f.write(msg + "\n")


# --------------------------------------------------------------------------- #
def make_eval_points(n_side: int, box_size: float) -> np.ndarray:
    """在盒子内生成 n_side x n_side 网格求值点（不含太靠近边界的点）。"""
    L = box_size
    margin = 0.02 * L
    x = np.linspace(-L / 2 + margin, L / 2 - margin, n_side)
    X, Y = np.meshgrid(x, x)
    return np.stack([X.ravel(), Y.ravel()], axis=1)


def rmse(a, b):
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))


def main():
    rng = np.random.default_rng(1234)
    box_size = 1.0
    sim_box = Rectangle(box_size)
    obstacles = []  # 阶段1 无边界

    R = 0.3

    def u_pre(pts):
        r = np.linalg.norm(pts, axis=-1)
        s = np.minimum(r, R) / np.maximum(r, 1e-12)
        return s[:, None] * pts

    def u_grad(pts):      # 纯梯度场（散度 = 2，不衰减）
        return pts

    def u_rot(pts):       # 无散度场（旋转）
        return np.stack([-pts[:, 1], pts[:, 0]], axis=1)

    def u_grad_local(pts):
        """局域化纯梯度场：u = ∇φ, φ = exp(-r²/σ²)，在盒边界≈0（无边界投影良定义，应→0）。"""
        sigma = 0.3
        e = np.exp(-np.sum(pts ** 2, axis=1) / sigma ** 2)
        return -2.0 / sigma ** 2 * e[:, None] * pts

    (ROOT / "results").mkdir(parents=True, exist_ok=True)
    logfile = ROOT / "results" / "projection_convergence.log"
    if logfile.exists():
        logfile.unlink()

    log("=" * 70)
    log("阶段1: 无边界蒙特卡洛投影验证（论文 §2.3.1 / Fig 6）")
    log("=" * 70)

    # ---------------- 检验 A/B：投影算子性质 + N 增大应 1/√N 收敛 ---------------- #
    log("\n[检验 A/B] 投影算子性质（N 增大误差应 ~1/√N 下降）")
    pts = make_eval_points(16, box_size)
    for name, uf, expect_zero in [("A1: 纯梯度(不衰减) u=(x,y)", u_grad, True),
                                  ("A2: 纯梯度(局域化) u=grad(exp(-r^2))", u_grad_local, True),
                                  ("B:  无散度 u=(-y,x)", u_rot, False)]:
        for nv in [1_000, 10_000, 50_000]:
            u_in = uf(pts)
            u_out = project_batch_vectorized(
                pts, uf, sim_box, obstacles,
                num_volume_samples=nv, num_pseudo_boundary_samples=nv,
                rng=rng, antithetic=True)
            if expect_zero:
                err = rmse(u_out, np.zeros_like(u_out))
                log(f"  {name} N={nv:>7}: |proj(u)|_rms = {err:.4e}   (应→0)")
            else:
                err = rmse(u_out, u_in)
                log(f"  {name} N={nv:>7}: |proj(u)-u|_rms = {err:.4e}   (应→0)")

    # ---------------- 检验 C：收敛测试（对照论文 Fig 6） ---------------- #
    log("\n[检验 C] 收敛测试：u=min(|x|,R)·x̂, R=0.3, 盒 [-0.5,0.5]^2")
    pts = make_eval_points(16, box_size)
    log("  计算高样本参考 (N_V=N_A=5e4)...")
    t0 = time.perf_counter()
    ref = project_batch_vectorized(
        pts, u_pre, sim_box, obstacles,
        num_volume_samples=50_000, num_pseudo_boundary_samples=50_000,
        rng=np.random.default_rng(99), antithetic=True)
    log(f"    参考计算耗时 {time.perf_counter()-t0:.1f}s")

    n_vols = [10, 100, 1000, 10_000, 50_000]
    n_areas = [10, 100, 1000, 10_000, 50_000]
    err_v, err_a = [], []
    for nv in n_vols:
        out = project_batch_vectorized(
            pts, u_pre, sim_box, obstacles,
            num_volume_samples=nv, num_pseudo_boundary_samples=50_000,
            rng=rng, antithetic=True)
        err_v.append(rmse(out, ref))
        log(f"  N_V={nv:>7}: RMSE={err_v[-1]:.4e}")
    for na in n_areas:
        out = project_batch_vectorized(
            pts, u_pre, sim_box, obstacles,
            num_volume_samples=50_000, num_pseudo_boundary_samples=na,
            rng=rng, antithetic=True)
        err_a.append(rmse(out, ref))
        log(f"  N_A={na:>7}: RMSE={err_a[-1]:.4e}")

    # ---------------- 画图 ---------------- #
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, ns, errs, lab in [(ax1, n_vols, err_v, "N_V"),
                              (ax2, n_areas, err_a, "N_A")]:
        ax.loglog(ns, errs, "o-", label=lab)
        # 参考 1/√N 斜率
        ax.loglog(ns, errs[0] * np.sqrt(ns[0] / np.array(ns, dtype=float)),
                  "--", color="gray", label="1/√N 参考斜率")
        ax.set_xlabel(lab)
        ax.set_ylabel("RMSE")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
    ax1.set_title("体积项收敛 (N_A=5e4)")
    ax2.set_title("伪边界项收敛 (N_V=5e4)")
    fig.suptitle("投影步蒙特卡洛收敛（对照论文 Fig 6）")
    fig.tight_layout()
    out_png = ROOT / "results" / "projection_convergence.png"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    log(f"\n图已保存: {out_png}")
    log(f"数据已保存: {logfile}")


if __name__ == "__main__":
    main()

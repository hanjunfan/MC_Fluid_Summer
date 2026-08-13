"""分析 Karman 浓度是否绕过圆柱（无穿透 + 两侧流动 + 下游尾流）。"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

RESULTS = ROOT / "results"


def main():
    for name in ["karman_preview_re25", "karman_preview_re250"]:
        d = RESULTS / name
        q = np.load(d / "conc_00080.npy")   # (40,80) y-major
        # 圆柱中心 (-1.5,0) → i=9.5, j=19.5；半径 0.125 → 2.5 格 → i∈[7,12], j∈[17,22]
        cyl = q[17:23, 7:13]
        above = q[:17, :].sum()
        below = q[23:, :].sum()
        right = q[:, 13:].sum()   # 下游 x>i=13
        print(f"[{name}] 圆柱内部浓度 max={cyl.max():.3f} (应为0=无穿透)")
        print(f"  上方总浓度={above:.1f} 下方={below:.1f} 下游(x>13)={right:.1f}")
        print(f"  全局 max={q.max():.2f} sum={q.sum():.1f}")
        # 检查圆柱下游是否有浓度尾流（>0.05 且连续）
        tail = (q[:, 14:30] > 0.05).sum()
        print(f"  下游条带区(i=14..30)浓度>0.05 的格数: {tail}")


if __name__ == "__main__":
    main()

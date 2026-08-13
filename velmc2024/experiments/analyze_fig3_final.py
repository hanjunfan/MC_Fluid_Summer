"""分析 Fig3 三个 Re 最终帧的涡街特征。"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

RESULTS = ROOT / "results"


def main():
    cases = [("karman_fig3_fixed_re25p", 300, "Re=2.5"),
             ("karman_fig3_fixed_re25", 495, "Re=25"),
             ("karman_fig3_fixed_re250", 390, "Re=250")]
    for name, step, label in cases:
        q = np.load(RESULTS / name / f"conc_{step:05d}.npy")
        neg = int((q < -1e-4).sum())
        cyl = q[17:23, 7:13].max()
        # 下游 x=14..45 分 6 段看上下优势
        seg = []
        for x0 in range(14, 45, 6):
            u = q[:19, x0:x0 + 6].sum()
            d = q[21:, x0:x0 + 6].sum()
            seg.append("上" if u > d else "下")
        print(f"{label} t={step*0.02:.1f}: 负浓度格={neg} 圆柱内max={cyl:.3f} "
              f"下游上下交替={''.join(seg)}")


if __name__ == "__main__":
    main()

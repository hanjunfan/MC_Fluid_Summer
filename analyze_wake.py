import numpy as np, os
d = r"c:\Users\jfhan\Desktop\MC_Fluid_Summer\results\cohomology_vort"
steps = sorted(int(f.split('_')[1].split('.')[0]) for f in os.listdir(d) if f.startswith('conc_'))
print(f"{'t':>5} {'核心sum':>7} {'弱丝sum':>7} {'弱/核':>6} {'弱丝y范围':>10} {'弱丝x范围':>10}")
for s in steps:
    c = np.load(os.path.join(d, f"conc_{s:05d}.npy"))
    core = np.abs(c) >= 0.3          # 核心团
    weak = (np.abs(c) >= 0.05) & (np.abs(c) < 0.3)  # 弱尾丝/晕
    core_sum = np.abs(c[core]).sum()
    weak_sum = np.abs(c[weak]).sum()
    t = s * 0.05
    if weak.any():
        ys, xs = np.where(weak)
        yr = f"{ys.min()}-{ys.max()}"
        xr = f"{xs.min()}-{xs.max()}"
    else:
        yr = xr = "无"
    print(f"{t:5.2f} {core_sum:7.1f} {weak_sum:7.1f} {weak_sum/core_sum:6.2f} {yr:>10} {xr:>10}")

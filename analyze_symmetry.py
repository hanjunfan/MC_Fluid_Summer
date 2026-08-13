import numpy as np, os
d = r"c:\Users\jfhan\Desktop\MC_Fluid_Summer\results\cohomology_vort"
steps = sorted(int(f.split('_')[1].split('.')[0]) for f in os.listdir(d) if f.startswith('conc_'))
print("帧数:", len(steps))
print(f"{'t':>5} {'红sum':>6} {'蓝sum':>6} {'比':>5} {'红y-48':>6} {'蓝y-48':>6} {'蓝x-红x':>6}")
for s in steps:
    c = np.load(os.path.join(d, f"conc_{s:05d}.npy"))
    red_sum = c[c > 0].sum()
    blue_sum = -c[c < 0].sum()
    def cm(mask):
        ys, xs = np.where(mask)
        if len(xs) == 0:
            return (np.nan, np.nan)
        w = np.abs(c[mask])
        return (np.average(xs, weights=w), np.average(ys, weights=w))
    rx, ry = cm(c > 0.3)
    bx, by = cm(c < -0.3)
    t = s * 0.05
    print(f"{t:5.2f} {red_sum:6.1f} {blue_sum:6.1f} {red_sum/blue_sum:5.3f} "
          f"{ry-48:6.1f} {by-48:6.1f} {bx-rx:6.1f}")

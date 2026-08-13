import numpy as np, os, sys
d = r"c:\Users\jfhan\Desktop\MC_Fluid_Summer\results\cohomology_vort"
steps = sorted(int(f.split('_')[1].split('.')[0]) for f in os.listdir(d) if f.startswith('conc_'))
print("frames:", steps)
for s in steps:
    c = np.load(os.path.join(d, f"conc_{s:05d}.npy"))
    red = c > 0.3
    blu = c < -0.3
    def cm(mask):
        ys, xs = np.where(mask)
        if len(xs) == 0:
            return (None, None, 0)
        w = np.abs(c[mask])
        return (round(float(np.average(xs, weights=w)), 1),
                round(float(np.average(ys, weights=w)), 1), len(xs))
    print(f"step {s:3d}: red(cm)={cm(red)}  blue(cm)={cm(blu)}  |c|max={np.abs(c).max():.2f}  sum|c|={np.abs(c).sum():.0f}")

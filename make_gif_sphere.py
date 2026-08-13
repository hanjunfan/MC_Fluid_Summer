"""把 grid 48 球体绕流帧合成 GIF 动画（本地）。"""
from pathlib import Path
from PIL import Image

root = Path(__file__).resolve().parent / "results" / "smoke3d_gpu"
frames = sorted(root.glob("frame_*.png"))
print(f"找到 {len(frames)} 帧: {root}")

if not frames:
    raise SystemExit("没有帧图，请先下载 results/smoke3d_gpu")

imgs = [Image.open(p) for p in frames]
out = root / "smoke3d_sphere.gif"
imgs[0].save(out, save_all=True, append_images=imgs[1:],
             duration=120, loop=0, optimize=False)
print(f"已保存 {out}，共 {len(imgs)} 帧，每帧 120ms")

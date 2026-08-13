"""把 grid 32 的烟柱帧合成 GIF 动画（本地）。"""
from pathlib import Path
from PIL import Image

root = Path(__file__).resolve().parent
frames = sorted(root.glob("frame_*.png"))
print(f"找到 {len(frames)} 帧")

imgs = [Image.open(p) for p in frames]
# 统一尺寸（应一致）
out = root / "smoke3d_grid32.gif"
imgs[0].save(out, save_all=True, append_images=imgs[1:],
             duration=120, loop=0, optimize=False)
print(f"已保存 {out}，共 {len(imgs)} 帧，每帧 120ms")

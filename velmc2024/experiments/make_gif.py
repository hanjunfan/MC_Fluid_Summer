"""make_gif.py —— 把 results/<dir>/frame_*.png 合成 GIF 动画（用 PIL）。"""
import argparse
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="含 frame_*.png 的目录")
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    d = ROOT / args.dir
    frames = sorted(d.glob("frame_*.png"))
    if not frames:
        print(f"未找到帧: {d}")
        return
    imgs = [Image.open(f) for f in frames]
    out = ROOT / (args.out or f"{args.dir}/animation.gif")
    imgs[0].save(out, save_all=True, append_images=imgs[1:], duration=1000 / args.fps, loop=0)
    print(f"已生成 {len(imgs)} 帧 -> {out}")


if __name__ == "__main__":
    main()

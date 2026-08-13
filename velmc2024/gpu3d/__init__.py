"""gpu3d —— PyTorch GPU 高精度三维浮力烟（论文 velocity_fluids_3d 的 GPU 移植）。

与 velmc2024/core3d 的 CPU 版算法完全一致，但核心计算（3D 投影、三线性插值、
RK3 平流、扩散）全部用 PyTorch CUDA 张量在 GPU 上执行，可跑到 128³~256³ 分辨率
与每点数万 MC 样本（接近论文 GPU 版量级）。

用法见 run_smoke3d_gpu.py。
"""

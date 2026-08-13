# GPU 3D 浮力烟（PyTorch / CUDA）

基于 SIGGRAPH 2024《Velocity-Based Monte Carlo Fluids》的无散度速度-涡量积分公式，
用 **PyTorch CUDA** 实现的高精度三维浮力烟求解器。适合在租用的 GPU 机器（如 RTX 4090）上运行。

## 环境要求

- Linux（Ubuntu 22.04 推荐）+ NVIDIA GPU（≥ 16GB 显存推荐，24GB 最佳）
- Python 3.10+
- PyTorch 2.x（CUDA 版）+ numpy + matplotlib

安装：

```bash
pip install torch numpy matplotlib
# 或按官方命令安装对应 CUDA 版本的 torch，例如 CUDA 12.8：
# pip install torch --index-url https://download.pytorch.org/whl/cu128
```

## 快速开始

```bash
python velmc2024/gpu3d/run_smoke3d_gpu.py \
    --grid 128 --steps 500 --nvol 5000 --npsb 2000 \
    --out results/smoke3d_gpu
```

输出到 `results/smoke3d_gpu/`：
- `conc_*.npy` —— 每一步保存的浓度场（步数较大时按需保存）
- `frame_*.png` —— 三个正交切片（侧面/正面/俯视）合成帧
- `animation.gif` —— 动画

## 参数说明

| 参数 | 默认 | 含义 |
|------|------|------|
| `--grid` | 128 | 网格分辨率（立方体，128³ ≈ 210 万格点） |
| `--steps` | 500 | 时间步数（dt=0.02，500 步 = 10 秒物理时间） |
| `--nvol` | 5000 | 投影体积项每点采样数 |
| `--npsb` | 2000 | 伪边界项每点采样数 |
| `--beta` | 1.0 | 浮力系数 |
| `--relax` | 1.0 | 投影松弛（≤1，用于抑制 MC 噪声累积） |
| `--out` | results/smoke3d_gpu | 输出目录 |

## 性能与显存预估（RTX 4090 / 24GB）

- 128³ 网格，体积项按 4096 个点分块计算，单块峰值约 2–3GB，总显存约 8–12GB。
- 单步约 10–20s（nvol=5000 / npsb=2000），500 步约 1.5–3 小时。
- 想快速试跑可先用小配置验证环境：

```bash
python velmc2024/gpu3d/run_smoke3d_gpu.py --grid 32 --steps 10 --nvol 500 --npsb 200
```

## 算法说明（与 CPU 版一致，仅换 GPU 后端）

算子分裂，每步依次：
1. **平流**：RK3 半拉格朗日反溯 + 三线性插值（`grid_sample`）；
2. **烟源**：箱体区域注入浓度与温度；
3. **浮力**：Boussinesq 近似 `a = -β·T·g`；
4. **扩散**：高斯核可分卷积（`F.conv3d`），用网格 σ 保证分辨率无关；
5. **投影**：MC 无散度投影（体积项 PDF∝1/r + 六面伪边界项）。

投影积分公式（已独立验证，旋转保持 ~1%、梯度项消除）：

$$
\mathbf{u}(\mathbf{x}) \leftarrow \sum_k \frac{3\hat{\mathbf{r}}(\hat{\mathbf{r}}\cdot\Delta\mathbf{u}) - \Delta\mathbf{u}}{2N r^2} R^2
  - \frac{A}{B}\sum_b \nabla_{\mathbf{x}}G \cdot (\mathbf{n}\cdot\Delta\mathbf{u})
$$

## 注意事项

- 代码会自动检测 CUDA，无 GPU 时退回 CPU（仅用于小网格调试）。
- 若出现数值发散（速度爆炸），先减小 `--relax`（如 0.3–0.5），或减小 `--beta`。
- 随机数种子固定（`seed=0`），结果可复现。

# 性能优化指南

pyOpenFOAM 基于 PyTorch 张量计算，支持 CPU、CUDA GPU 和 Apple MPS 加速。本文档涵盖 GPU 加速、内存优化和并行执行的最佳实践。

---

## GPU 加速

### 启用 GPU

pyOpenFOAM 自动检测可用的计算设备：

```python
from pyfoam.core.device import DeviceManager

dm = DeviceManager()
print(dm.capabilities)   # 查看可用设备
print(dm.device)         # 当前选中的设备
```

设备优先级：**CUDA > MPS > CPU**

手动切换设备：

```python
from pyfoam.core.device import device_context

# 强制使用 CPU
with device_context(device="cpu"):
    solver = RhoCentralFoam("case/")
    solver.run()

# 指定 CUDA 设备
with device_context(device="cuda:0"):
    solver = SimpleFoam("case/")
    solver.run()
```

### CUDA 使用

确保已安装 CUDA 版本的 PyTorch：

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

验证 GPU 可用性：

```python
import torch
print(torch.cuda.is_available())       # True
print(torch.cuda.get_device_name(0))   # GPU 名称
```

### 性能对比参考

| 求解器规模 | CPU (float64) | GPU (float64) | 加速比 |
|-----------|---------------|---------------|--------|
| 1K cells  | 0.5s          | 0.8s          | 0.6x (传输开销) |
| 10K cells | 5s            | 1.2s          | 4x |
| 100K cells| 50s           | 3s            | 17x |
| 1M cells  | 500s          | 15s           | 33x |

> **注意**: 小规模问题 (<10K cells) GPU 可能比 CPU 慢，因为数据传输开销超过了计算优势。

### 禁用 GPU（测试时）

运行测试时可通过环境变量强制 CPU：

```bash
CUDA_VISIBLE_DEVICES='' pytest tests/ -v
```

---

## 内存优化

### 精度选择

pyOpenFOAM 默认使用 `float64`（双精度），这是 CFD 数值稳定性的要求。对于探索性计算，可临时切换到 `float32`：

```python
import torch
from pyfoam.core.device import device_context

# 单精度可节省约 50% 内存，但可能影响收敛
with device_context(dtype=torch.float32):
    solver = IcoFoam("case/")
    solver.run()
```

> **警告**: float32 在迭代求解器中可能导致残差振荡或发散。仅在内存受限时使用，并验证结果正确性。

### 网格规模控制

内存消耗与网格规模线性相关。估算公式：

```
内存 ≈ n_cells × n_fields × 8 bytes (float64)
     + n_faces × n_face_fields × 8 bytes
     + 矩阵系数存储 (约 3x 场大小)
```

示例（不可压缩 SIMPLE 求解器）：
- 10K cells: ~10 MB
- 100K cells: ~100 MB
- 1M cells: ~1 GB
- 10M cells: ~10 GB

### 减少内存占用的策略

1. **减小 z 方向厚度**: 2D 问题使用最小的 z 方向单元数
2. **及时释放张量**: 不再需要的场变量及时删除并调用 `torch.cuda.empty_cache()`
3. **分步写入**: 避免在内存中保存所有时间步的场数据
4. **使用 purgeWrite**: 在 controlDict 中设置 `purgeWrite 1` 仅保留最新时间步

---

## 并行执行

### PyTorch 线程控制

PyTorch 自动使用多线程进行 CPU 上的张量运算。控制线程数：

```python
import torch
torch.set_num_threads(8)  # 设置为 CPU 核心数
```

或通过环境变量：

```bash
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES='' pytest tests/ -v
```

### 批量运行多个案例

独立案例可并行运行：

```python
import subprocess
import os

cases = ["case1/", "case2/", "case3/"]
procs = []
for case in cases:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    p = subprocess.Popen(
        ["python", "-m", "pyfoam.run", case],
        env=env,
    )
    procs.append(p)

for p in proc:
    p.wait()
```

### GPU 多流并行

对于多个小案例，可使用 CUDA streams 实现 GPU 并行：

```python
import torch

stream1 = torch.cuda.Stream()
stream2 = torch.cuda.Stream()

with torch.cuda.stream(stream1):
    solver1.run()
with torch.cuda.stream(stream2):
    solver2.run()

torch.cuda.synchronize()
```

> **注意**: 需确保每个求解器使用独立的张量，避免数据竞争。

---

## 精度 vs 性能权衡

| 设置 | 精度 | 速度 | 内存 |
|------|------|------|------|
| float64 + 细网格 | 最高 | 最慢 | 最大 |
| float64 + 粗网格 | 高 | 快 | 中 |
| float32 + 细网格 | 中等 | 中等 | 中 |
| float32 + 粗网格 | 低 | 最快 | 最小 |

---

## 常见性能问题

### 残差不收敛

| 现象 | 可能原因 | 解决方案 |
|------|----------|----------|
| 残差振荡 | 时间步过大 | 减小 deltaT 或 CFL |
| 残差不降 | 网格质量问题 | 检查网格正交性 |
| 残差发散 | 数值格式不当 | 使用更稳健的离散格式 |

### GPU 未被使用

```python
# 诊断检查
import torch
from pyfoam.core.device import DeviceManager

dm = DeviceManager()
print(f"CUDA available: {dm.capabilities.cuda}")
print(f"Current device: {dm.device}")

if dm.capabilities.cuda:
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
```

### 内存不足 (OOM)

1. 减小网格规模
2. 切换到 float32（谨慎使用）
3. 使用 `torch.cuda.empty_cache()` 清理缓存
4. 检查是否有不必要的张量副本

---

## 最佳实践总结

1. **规模匹配**: 小问题用 CPU，大问题用 GPU（>50K cells）
2. **默认 float64**: 除非明确验证过，否则保持双精度
3. **测试时禁用 GPU**: `CUDA_VISIBLE_DEVICES=''` 确保可重复性
4. **监控残差**: 使用 `ConvergenceMonitor` 确认收敛
5. **网格无关性验证**: 用至少 2-3 个网格级别确认结果收敛

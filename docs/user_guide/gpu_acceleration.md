# GPU 加速完全指南

pyOpenFOAM 基于 PyTorch，原生支持 CUDA 和 MPS (Apple Silicon) GPU 加速。本文档提供 GPU 使用的完整参考。详细性能数据见 [性能优化指南](performance.md)。

---

## 目录

- [环境检测与配置](#环境检测与配置)
- [CUDA 加速](#cuda-加速)
- [MPS 加速 (Apple Silicon)](#mps-加速-apple-silicon)
- [混合精度策略](#混合精度策略)
- [大规模网格的内存管理](#大规模网格的内存管理)
- [性能调优清单](#性能调优清单)

---

## 环境检测与配置

### 自动检测

```python
from pyfoam.core.device import DeviceManager

dm = DeviceManager()
print(f"可用设备: {dm.capabilities}")
print(f"当前设备: {dm.device}")
```

设备优先级: **CUDA > MPS > CPU**

### 手动指定设备

```python
from pyfoam.core.device import device_context

# 指定 CUDA 设备
with device_context(device="cuda:0"):
    solver = SimpleFoam("case/")
    solver.run()

# 指定 MPS (macOS)
with device_context(device="mps"):
    solver = SimpleFoam("case/")
    solver.run()

# 强制 CPU
with device_context(device="cpu"):
    solver = SimpleFoam("case/")
    solver.run()
```

### 环境变量控制

```bash
# 禁用 GPU（测试时必须）
CUDA_VISIBLE_DEVICES='' pytest tests/ -v

# 选择特定 GPU
CUDA_VISIBLE_DEVICES=0 python run_case.py

# 多 GPU 选择
CUDA_VISIBLE_DEVICES=1 python run_case.py
```

---

## CUDA 加速

### 安装 CUDA 版 PyTorch

```bash
# CUDA 12.x
pip install torch --index-url https://download.pytorch.org/whl/cu124

# CUDA 11.x
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### 验证

```python
import torch
print(f"CUDA 可用: {torch.cuda.is_available()}")
print(f"GPU 数量: {torch.cuda.device_count()}")
print(f"GPU 名称: {torch.cuda.get_device_name(0)}")
print(f"显存: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
```

### 使用示例

```python
from pyfoam.applications import SimpleFoam
from pyfoam.core.device import set_device

set_device("cuda:0")
solver = SimpleFoam("case/")
solver.run()
```

### 多 GPU 并行

```python
import subprocess, os

gpus = ["0", "1"]
cases = ["case_gpu0/", "case_gpu1/"]

procs = []
for gpu, case in zip(gpus, cases):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    p = subprocess.Popen(["python", "run_case.py", case], env=env)
    procs.append(p)

for p in procs:
    p.wait()
```

---

## MPS 加速 (Apple Silicon)

适用于 M1/M2/M3 芯片的 Mac。无需额外安装。

```python
from pyfoam.core.device import set_device

set_device("mps")
solver = SimpleFoam("case/")
solver.run()
```

**注意事项**:

- MPS 后端对 `float64` 支持有限，部分操作可能回退到 CPU
- 建议使用 `float32`（需验证精度）
- 确认 PyTorch 版本 >= 2.0

---

## 混合精度策略

### float64（默认，推荐）

CFD 标准精度。数值稳定性最好，残差可收敛到机器精度。

```python
from pyfoam.core.device import get_default_dtype
assert get_default_dtype() == torch.float64
```

### float32（探索性计算）

可节省约 50% 内存，加速矩阵运算，但需验证精度：

```python
import torch
from pyfoam.core.device import device_context

with device_context(dtype=torch.float32):
    solver = SimpleFoam("case/")
    result = solver.run()
    # 检查残差是否合理
```

**风险**: float32 下迭代求解器（SIMPLE/PISO）残差可能振荡或不收敛。

### 推荐策略

| 场景 | 精度 | 设备 |
|------|------|------|
| 生产 / 发表结果 | float64 | CPU 或 GPU |
| 探索 / 调试 | float64 | CPU |
| 大规模初步筛选 | float32 | GPU |
| 后处理 / 可视化 | float32 | GPU |

---

## 大规模网格的内存管理

### 内存估算

| 网格规模 | float64 内存 | float32 内存 | 推荐设备 |
|---------|-------------|-------------|----------|
| < 10K cells | < 10 MB | < 5 MB | CPU |
| 10K - 100K | 10-100 MB | 5-50 MB | CPU 或 GPU |
| 100K - 1M | 100 MB - 1 GB | 50 - 500 MB | GPU |
| > 1M cells | > 1 GB | > 500 MB | 必须 GPU |

### 内存优化技巧

```python
# 1. 及时释放不再需要的张量
del old_field
torch.cuda.empty_cache()

# 2. 使用 purgeWrite 限制磁盘输出
# controlDict: purgeWrite 1;

# 3. 监控 GPU 内存
print(f"已用: {torch.cuda.memory_allocated() / 1e6:.1f} MB")
print(f"缓存: {torch.cuda.memory_reserved() / 1e6:.1f} MB")

# 4. 使用 torch.no_grad() 禁用梯度追踪（非可微分场景）
with torch.no_grad():
    solver.run()
```

---

## 性能调优清单

### 选择正确设备

- [ ] 小问题 (< 50K cells): CPU 通常更快
- [ ] 中问题 (50K - 500K): GPU 开始有优势
- [ ] 大问题 (> 500K): GPU 必须

### PyTorch 线程优化

```bash
# 设置 CPU 线程数 = 物理核心数
export OMP_NUM_THREADS=8
```

```python
import torch
torch.set_num_threads(8)
```

### 避免不必要的数据传输

```python
# 不好：频繁 CPU ↔ GPU
for cell in range(n_cells):
    result[cell] = compute(data[cell])

# 好：批量张量运算
result = compute(data)  # 一次完成
```

### 使用 CUDA Events 计时

```python
import torch

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()
solver.run()
end.record()

torch.cuda.synchronize()
print(f"耗时: {start.elapsed_time(end):.1f} ms")
```

---

## 故障排除

### "CUDA out of memory"

1. 减小网格规模
2. 切换到 float32
3. 减少时间步内保存的场数据
4. 使用 `torch.cuda.empty_cache()` 清理

### GPU 未被使用

```python
import torch
print(torch.cuda.is_available())       # 应为 True
print(torch.cuda.device_count())       # 应 > 0
```

确认安装了 CUDA 版 PyTorch（不是 CPU-only 版本）:

```bash
python -c "import torch; print(torch.version.cuda)"
# 应输出 CUDA 版本号，而非 None
```

### MPS 相容性问题

部分 PyTorch 算子在 MPS 后端不可用。如遇错误，回退到 CPU：

```python
try:
    with device_context(device="mps"):
        solver.run()
except RuntimeError:
    print("MPS 不支持当前运算，回退到 CPU")
    with device_context(device="cpu"):
        solver.run()
```

---

## 进一步参考

- [性能优化指南](performance.md) — 详细性能数据和内存估算
- [高级主题](advanced_topics.md) — 可微分 CFD 和自定义模型
- [快速入门](getting_started.md) — 安装和基本使用

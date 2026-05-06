# GPU 加速指南

pyOpenFOAM 使用 PyTorch 作为张量后端，为所有 CFD 操作提供透明的 GPU 加速。本指南涵盖设备管理、性能优化和 GPU 加速 CFD 仿真的最佳实践。

## 设备检测

pyOpenFOAM 在启动时自动检测可用硬件：

```python
from pyfoam.core import DeviceManager

dm = DeviceManager()
print(dm.capabilities)
# DeviceCapabilities(cpu=True, cuda=True, mps=False, cuda_devices=1)

print(dm.capabilities.available_devices)
# ['cpu', 'cuda']

print(dm.device)
# device('cuda') — 自动选择最佳可用设备
```

### 支持的后端

| 后端 | 设备字符串 | 说明 |
|------|-----------|------|
| CPU | `"cpu"` | 始终可用，默认 float64 |
| CUDA | `"cuda"` | NVIDIA GPU，需要 CUDA 工具包 |
| CUDA（指定） | `"cuda:0"`、`"cuda:1"` | 多 GPU 选择 |
| MPS | `"mps"` | Apple Silicon（M1/M2/M3） |

### 优先级顺序

自动选择顺序为：**CUDA > MPS > CPU**。

```python
# 手动设备选择
dm.device = "cuda"    # 使用 NVIDIA GPU
dm.device = "mps"     # 使用 Apple Silicon
dm.device = "cpu"     # 强制使用 CPU
```

## 张量配置

### 默认精度

CFD 需要 **float64**（双精度）才能收敛。float32 会因压力-速度耦合算法中的累积舍入误差导致发散。

```python
from pyfoam.core import TensorConfig, get_default_dtype
import torch

print(get_default_dtype())  # torch.float64

config = TensorConfig()
print(config.dtype)    # torch.float64
print(config.device)   # device('cuda')（如果可用）
```

### 临时覆盖

使用上下文管理器进行临时设备/数据类型更改：

```python
from pyfoam.core import device_context, TensorConfig
import torch

config = TensorConfig()

# 临时 float32 用于性能测试
with config.override(dtype=torch.float32):
    t = config.zeros(1000)  # float32 张量
    print(t.dtype)  # torch.float32

# 恢复 float64
print(config.dtype)  # torch.float64

# 模块级上下文管理器
with device_context(device='cpu', dtype=torch.float32):
    # 所有操作在此使用 CPU + float32
    pass
```

## GPU 内存管理

### PyTorch 内存行为

PyTorch 使用缓存分配器管理 GPU 内存。关键行为：

1. **内存不会释放** — PyTorch 缓存已释放的内存以供重用。
2. **峰值内存重要** — 分配器跟踪峰值使用量，而非当前使用量。
3. **`torch.cuda.empty_cache()`** — 释放未使用的缓存内存（很少需要）。

### 监控 GPU 内存

```python
import torch

# 检查已分配内存
print(torch.cuda.memory_allocated())  # 字节
print(torch.cuda.memory_reserved())   # 字节（缓存）

# 检查最大已分配
print(torch.cuda.max_memory_allocated())

# 重置峰值统计
torch.cuda.reset_peak_memory_stats()
```

### 内存高效模式

```python
# 差：创建中间张量
result = a + b + c + d

# 好：原地操作减少峰值内存
result = a.clone()
result += b
result += c
result += d

# 差：在作用域中保留旧张量
for i in range(1000):
    temp = expensive_computation(i)
    results.append(temp)

# 好：释放中间变量
for i in range(1000):
    temp = expensive_computation(i)
    results.append(temp.clone())
    del temp
```

## 性能优化

### 1. 预计算几何

几何量是惰性计算的。在仿真循环前预计算它们：

```python
mesh.compute_geometry()  # 一次计算所有几何量
# 现在 mesh.cell_volumes、mesh.face_areas 等已被缓存
```

### 2. 批量操作

PyTorch 操作在批量处理时最快。避免对单元/面的 Python 循环：

```python
# 差：Python 循环遍历单元
for i in range(n_cells):
    result[i] = a[i] + b[i]

# 好：向量化张量操作
result = a + b  # 单次 GPU 内核启动

# 差：Python 循环遍历面
for f in range(n_internal_faces):
    flux[f] = phi[f] * (u[owner[f]] - u[neighbour[f]])

# 好：Gather + 向量化
u_owner = u[owner[:n_internal]]
u_neigh = u[neighbour]
flux = phi[:n_internal] * (u_owner - u_neigh)
```

### 3. 使用 scatter_add 和 gather

`scatter_add` 和 `gather` 操作是 FVM 组装的核心原语：

```python
from pyfoam.core import scatter_add, gather

# Gather：按索引收集值（边界查找、邻居访问）
x_owner = gather(x, owner)     # x[owner[f]]
x_neigh = gather(x, neighbour) # x[neighbour[f]]

# Scatter-add：将通量贡献累积到单元
y = scatter_add(flux, owner, n_cells)  # y[owner[f]] += flux[f]
```

### 4. 避免不必要的设备传输

在 CPU 和 GPU 之间移动张量很昂贵：

```python
# 差：频繁的设备传输
for i in range(n_cells):
    value = tensor[i].item()  # GPU → CPU（慢！）
    result = process(value)
    tensor[i] = result         # CPU → GPU（慢！）

# 好：所有操作在 GPU 上
result = process_gpu(tensor)   # 整个操作在 GPU 上
```

### 5. 求解时使用 CSR

在求解前将 LDU 矩阵转换为 CSR 格式：

```python
# 组装：COO 对增量插入很便宜
coo = ldu_matrix.to_sparse_coo()

# 求解：CSR 对矩阵-向量乘积更快
csr = ldu_matrix.to_sparse_csr()
```

## 多 GPU（未来）

通过 MPI 的多 GPU 支持已列入计划。架构支持通过以下方式实现：

1. **域分解** — 将网格拆分到多个 GPU。
2. **边界交换** — 通过 MPI 进行幽灵单元通信。
3. **并行组装** — 每个 GPU 组装其本地矩阵。
4. **全局求解** — 分布式线性求解器。

## 基准测试

### 计时 GPU 操作

```python
import torch
import time

# 计时前同步（GPU 操作是异步的）
torch.cuda.synchronize()
start = time.perf_counter()

# 运行计算
result = expensive_operation()

torch.cuda.synchronize()
elapsed = time.perf_counter() - start
print(f"耗时：{elapsed:.4f} 秒")
```

### 使用 PyTorch Profiler

```python
import torch

with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True,
) as prof:
    result = expensive_operation()

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

### 预期加速比

| 操作 | CPU（float64） | GPU（float64） | 加速比 |
|------|---------------|---------------|--------|
| 矩阵组装（10 万单元） | 50ms | 5ms | 10× |
| PCG 求解（1000 次迭代） | 2s | 200ms | 10× |
| GAMG 求解（100 个 V 循环） | 5s | 500ms | 10× |
| 梯度计算 | 20ms | 2ms | 10× |

*实际加速比取决于网格大小、GPU 型号和问题结构。*

## 常见问题

### "CUDA out of memory"

```python
# 减小网格大小或使用 CPU
dm = DeviceManager()
dm.device = 'cpu'

# 或释放 GPU 内存
import torch
torch.cuda.empty_cache()
```

### "Device 'cuda' is not available"

```python
# 检查 CUDA 安装
import torch
print(torch.cuda.is_available())  # 应为 True
print(torch.cuda.device_count())  # GPU 数量

# 安装启用 CUDA 的 PyTorch
# pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### GPU 性能缓慢

1. **检查张量大小** — 非常小的张量（< 1000 个元素）在 CPU 上可能更快。
2. **避免同步** — `tensor.item()`、`tensor.cpu()`、`tensor.numpy()` 会强制同步。
3. **使用 float64** — float32 更快但会导致 CFD 发散。
4. **先做性能分析** — 使用 `torch.profiler` 找到瓶颈。

### MPS（Apple Silicon）限制

MPS 后端有一些限制：

- 并非所有 PyTorch 操作都支持 MPS。
- 某些操作的 float64 支持有限。
- 对于大网格，性能可能不如 CUDA。

```python
# 检查 MPS 可用性
import torch
print(torch.backends.mps.is_available())  # 在 Apple Silicon 上为 True

# 如果 MPS 有问题则强制使用 CPU
dm.device = 'cpu'
```

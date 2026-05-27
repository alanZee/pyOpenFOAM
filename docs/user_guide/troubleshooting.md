# Troubleshooting Guide

## 常见错误与解决方案

### 安装与导入

| 错误 | 原因 | 解决方案 |
|------|------|----------|
| `ModuleNotFoundError: pyfoam` | 未安装或不在 Python 路径中 | `pip install -e .` 在项目根目录执行 |
| `torch.cuda.is_available()` 返回 False | CUDA 未正确安装 | 确认 NVIDIA 驱动 + CUDA toolkit 版本匹配；使用 `CUDA_VISIBLE_DEVICES=''` 强制 CPU |
| `ImportError: DLL load failed` (Windows) | PyTorch 版本与 CUDA 不兼容 | 从 PyTorch 官网选择正确的 wheel 安装 |

### 网格相关

| 错误 | 原因 | 解决方案 |
|------|------|----------|
| `ValueError: owner/neighbour length mismatch` | 面 owner/neighbour 数组长度不一致 | 检查 mesh 构建代码，确保内部面都有 owner 和 neighbour |
| `IndexError: face index out of range` | 面引用了不存在的节点 | 验证 face 列表中的节点索引在 `[0, n_points)` 范围内 |
| `AssertionError: n_internal_faces` | 内部面数量与 neighbour 长度不匹配 | 确保 neighbour 数组长度 == 内部面数 |
| 几何计算结果为 NaN | 退化单元（零体积或零面积面） | 使用 `check_mesh()` 工具检测质量问题 |

### 边界条件

| 错误 | 原因 | 解决方案 |
|------|------|----------|
| `KeyError: Unknown boundary condition type` | BC 类型未注册 | 检查拼写；用 `BoundaryCondition.available_types()` 查看所有已注册类型 |
| `RuntimeError: shape mismatch` | field 维度与 patch 面数不匹配 | 向量场 shape 应为 `(n, 3)`，标量场为 `(n,)` |
| BC 不生效（值不变） | 未调用 `apply()` 或 field 未传入 | 确认在时间循环内调用 `bc.apply(field, time=t)` |

### 求解器

| 错误 | 原因 | 解决方案 |
|------|------|----------|
| 残差不收敛 | 时间步过大或松弛因子不当 | 减小 `deltaT`；降低 under-relaxation 因子 |
| `RuntimeError: NaN detected` | 数值发散 | 检查初始条件；减小时间步；检查边界条件合理性 |
| SIMPLE 不满足连续性 | 压力-速度耦合问题 | 增加压力修正方程迭代次数；检查通量计算 |

---

## 调试技巧

### 启用详细日志

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### 检查字段值

```python
# 查看场的统计信息
print(f"min={field.min():.6f}, max={field.max():.6f}, mean={field.mean():.6f}")
print(f"NaN count: {torch.isnan(field).sum()}")
print(f"Inf count: {torch.isinf(field).sum()}")
```

### 使用 check_mesh 诊断网格

```python
from pyfoam.tools import check_mesh

result = check_mesh(mesh)
print(result)
# 关注: 非正交度 (non-orthogonality)、扭曲度 (skewness)、体积比
```

### 验证边界条件

```python
from pyfoam.boundary import BoundaryCondition

# 列出所有已注册的 BC 类型
print(BoundaryCondition.available_types())

# 确认特定类型已注册
assert "fixedValue" in BoundaryCondition.available_types()
```

### GPU 调试

```python
# 强制使用 CPU（排除 GPU 问题）
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# 查看当前设备
from pyfoam.core.device import get_device
print(f"Device: {get_device()}")
```

---

## 性能问题

### 内存占用过高

| 现象 | 可能原因 | 解决方案 |
|------|----------|----------|
| OOM (Out of Memory) | 网格过大、单精度/双精度选择不当 | 使用 `float32`；分批处理大网格 |
| 内存持续增长 | 未释放计算图 | 使用 `torch.no_grad()` 包裹非训练代码；及时 `.detach()` |

### 计算速度慢

| 现象 | 可能原因 | 解决方案 |
|------|----------|----------|
| 比预期慢 10x+ | 默认在 CPU 上运行 | 检查 `get_device()`；确认 CUDA 可用 |
| GPU 利用率低 | 数据传输瓶颈 | 批量传输数据，避免频繁 CPU↔GPU 拷贝 |
| 线性求解器慢 | 矩阵条件数差 | 改善网格质量；使用预条件器 |

### 验证计算正确性

```python
# 简单的收敛性测试
# 对同一问题，逐步减小时间步，观察结果是否趋于一致
for dt in [0.01, 0.005, 0.001]:
    result = run_simulation(deltaT=dt)
    print(f"dt={dt}: result={result:.8f}")
```

---

## FAQ

**Q: 如何在 Windows 上使用 pyOpenFOAM？**

A: 直接在 Windows + Python 环境中使用。所有功能均为纯 Python/PyTorch 实现，不依赖 OpenFOAM 安装。

**Q: 支持哪些 Python 版本？**

A: Python 3.10+。

**Q: 如何从 OpenFOAM 格式读取网格？**

A: 使用 `pyfoam.io` 模块中的 I/O 功能。详见 API 文档。

**Q: 可以和真实 OpenFOAM 结果对比吗？**

A: 可以。`validation/` 目录中提供了与 OpenFOAM 结果的对比验证案例。

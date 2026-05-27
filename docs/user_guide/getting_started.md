# 快速入门指南

本指南帮助您快速安装 pyOpenFOAM 并运行第一个 CFD 案例。

## 目录

- [环境要求](#环境要求)
- [安装](#安装)
- [运行第一个案例：icoFoam 方腔驱动流](#运行第一个案例icofoam-方腔驱动流)
- [理解求解器架构](#理解求解器架构)
- [GPU 加速](#gpu-加速)
- [可微分 CFD](#可微分-cfd)
- [下一步](#下一步)

---

## 环境要求

| 依赖 | 最低版本 |
|------|----------|
| Python | 3.10+ |
| PyTorch | 2.0+ |
| NumPy | 1.24+ |
| SciPy | 1.10+ |

可选依赖：
- **CUDA Toolkit** — GPU 加速（NVIDIA 显卡）
- **mpi4py** — MPI 并行计算

---

## 安装

### 从 PyPI 安装

```bash
# CPU 版本
pip install pyfoam-cfd

# GPU 版本（CUDA 12.x）
pip install pyfoam-cfd[gpu]

# MPI 版本
pip install pyfoam-cfd[mpi]

# 可视化工具
pip install pyfoam-cfd[viz]

# 开发版本（含测试依赖）
pip install pyfoam-cfd[dev]
```

### 从源码安装

```bash
git clone https://github.com/alanZee/pyOpenFOAM.git
cd pyOpenFOAM
pip install -e ".[dev]"
```

### Conda 环境安装

```bash
conda create -n pyopenfoam python=3.12
conda activate pyopenfoam

# 安装 PyTorch（根据您的 CUDA 版本选择）
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 安装 pyOpenFOAM
cd /path/to/pyOpenFOAM
pip install -e ".[dev]"
```

### 验证安装

```python
import pyfoam
print(pyfoam.__version__)  # 0.1.0

from pyfoam.core import DeviceManager
dm = DeviceManager()
print(dm.capabilities)  # DeviceCapabilities(cpu=True, cuda=False, ...)
print(dm.device)        # device('cpu')
```

---

## 运行第一个案例：icoFoam 方腔驱动流

方腔驱动流（Lid-Driven Cavity）是 CFD 领域最经典的验证案例。一个正方形腔体，顶壁以恒定速度运动，其余三壁固定。

### 方法一：使用应用级求解器（推荐）

这是最简单的方式，pyOpenFOAM 自动处理网格构建、场初始化和 I/O：

```python
from pyfoam.applications import IcoFoam

# 指向 OpenFOAM 案例目录（需包含 0/、constant/、system/ 三个子目录）
solver = IcoFoam("tutorials/incompressible/icoFoam/cavity")
solver.run()
```

### 方法二：使用底层 API 手动构建

当需要更细粒度的控制时，可以直接使用底层 API：

#### 第 1 步：加载案例和网格

```python
from pyfoam.io.case import Case
from pyfoam.mesh import FvMesh, PolyMesh

# 加载 OpenFOAM 案例
case = Case("tutorials/incompressible/icoFoam/cavity")

# 构建 FvMesh
mesh_data = case.mesh
mesh = PolyMesh.from_raw(
    points=mesh_data.points,
    faces=mesh_data.faces,
    owner=mesh_data.owner,
    neighbour=mesh_data.neighbour,
    boundary=mesh_data.boundary,
)
fv = FvMesh.from_poly_mesh(mesh)
fv.compute_geometry()
```

#### 第 2 步：创建场和边界条件

```python
import torch
from pyfoam.fields import volScalarField, volVectorField
from pyfoam.boundary import BoundaryCondition, Patch

# 读取初始场
U_data = case.read_field("U", time=0)
p_data = case.read_field("p", time=0)

# 创建场对象
U = volVectorField(fv, "U")
p = volScalarField(fv, "p")
U.assign(torch.zeros(fv.n_cells, 3))
p.assign(torch.zeros(fv.n_cells))
```

#### 第 3 步：配置并运行求解器

```python
from pyfoam.solvers import PISOSolver, PISOConfig

config = PISOConfig(n_correctors=2, n_non_orthogonal_correctors=0)
solver = PISOSolver(fv, config)

# 时间循环
dt = 0.005
for step in range(200):
    U_old = U.internal_field.clone()
    p_old = p.internal_field.clone()

    U_out, p_out, phi_out, convergence = solver.solve(
        U.internal_field, p.internal_field,
        torch.zeros(fv.n_faces),
        U_old=U_old, p_old=p_old,
    )
    U.assign(U_out)
    p.assign(p_out)
```

---

## 理解求解器架构

pyOpenFOAM 的架构分为四层：

```
+--------------------------------------------------+
|  applications/ (应用层)                            |
|  IcoFoam, SimpleFoam, InterFoam, ...              |
|  - 读取 OpenFOAM 案例目录                          |
|  - 构建网格、场、边界条件                           |
|  - 运行时间循环                                    |
+--------------------------------------------------+
|  solvers/ (算法层)                                  |
|  SIMPLE, PISO, PIMPLE                             |
|  - 压力-速度耦合                                   |
|  - 收敛控制                                        |
+--------------------------------------------------+
|  fields/ + boundary/ + discretisation/ (物理层)     |
|  场类、边界条件、离散格式                           |
|  - 有限体积离散                                    |
|  - 矩阵组装                                        |
+--------------------------------------------------+
|  core/ + mesh/ (基础层)                            |
|  设备管理、LDU 矩阵、网格、I/O                      |
|  - PyTorch 张量后端                                |
|  - GPU/MPI 支持                                    |
+--------------------------------------------------+
```

### 求解器调用流程

```
SolverBase.__init__()
    |
    v
Case(case_path)  -->  读取 controlDict, fvSchemes, fvSolution
    |
    v
_build_mesh()  -->  PolyMesh.from_raw()  -->  FvMesh.from_poly_mesh()
    |
    v
_read_fields()  -->  从 0/ 目录读取 U, p, (k, epsilon, ...)
    |
    v
solver.run()
    |
    v
TimeLoop:  while time < endTime:
    |
    +-> assemble_matrices()   组装动量/压力方程
    +-> solve_pressure()      求解压力 Poisson 方程
    +-> correct_velocity()    修正速度（满足连续性）
    +-> update_turbulence()   更新湍流量
    +-> write_fields()        按 writeInterval 输出
```

### 关键概念

| 概念 | 说明 |
|------|------|
| `SolverBase` | 所有求解器的基类，负责案例加载和基础设施搭建 |
| `TimeLoop` | 时间循环控制器，管理时间步进和写入间隔 |
| `ConvergenceMonitor` | 残差监控和收敛判断 |
| `FvMatrix` | 有限体积方程矩阵，支持源项、边界贡献和欠松弛 |
| `LduMatrix` | OpenFOAM 原生 LDU 稀疏矩阵格式 |

---

## GPU 加速

pyOpenFOAM 的所有场运算都基于 PyTorch 张量，天然支持 GPU 加速。

### 自动设备选择

```python
from pyfoam.core import DeviceManager

dm = DeviceManager()
print(dm.capabilities.available_devices)  # ['cpu'] 或 ['cpu', 'cuda']
print(dm.device)  # 如果有 CUDA 则为 device('cuda')，否则为 device('cpu')
```

### 手动指定设备

```python
from pyfoam.core import device_context

# 强制使用 CPU
with device_context("cpu"):
    solver = SimpleFoam("path/to/case")
    solver.run()

# 强制使用 GPU（需要 CUDA）
with device_context("cuda"):
    solver = SimpleFoam("path/to/case")
    solver.run()
```

### 性能建议

1. **使用 float64** — CFD 收敛需要双精度，float32 会导致发散
2. **预计算几何** — 在模拟循环前调用 `mesh.compute_geometry()`
3. **批量运算** — PyTorch 操作在批量处理时最快，避免 Python 循环遍历单元/面
4. **先 profile** — 使用 `torch.profiler` 识别瓶颈再优化

---

## 可微分 CFD

pyOpenFOAM 支持通过 `torch.autograd` 进行端到端可微分 CFD 模拟。

### 基本用法

```python
import torch
from pyfoam.differentiable import (
    DifferentiableGradient,
    DifferentiableDivergence,
    DifferentiableLaplacian,
    DifferentiableLinearSolve,
)

# 可微分梯度（支持 autograd）
phi = torch.randn(n_cells, requires_grad=True)
grad = DifferentiableGradient.apply(phi, mesh)

# 可微分 Laplacian
lap = DifferentiableLaplacian.apply(phi, mesh)

# 可微分线性求解（隐式微分）
A = torch.sparse_coo_tensor(indices, values, (n, n))
b = torch.randn(n, requires_grad=True)
x = DifferentiableLinearSolve.apply(A, b, tol=1e-6, max_iter=1000)

# 梯度可通过求解器反传
loss = x.sum()
loss.backward()
print(b.grad)  # d(loss)/d(b)
```

### 应用场景

- **形状优化** — 通过可微分 CFD 自动计算形状梯度
- **物理信息神经网络 (PINN)** — 将 CFD 算子嵌入神经网络
- **灵敏度分析** — 高效计算流场对设计变量的灵敏度

---

## 下一步

- [API 参考](../api/modules.md) — 完整的模块级 API 文档
- [求解器参考](solvers.md) — 所有求解器的详细说明
- [开发者指南](adding_features.md) — 如何扩展 pyOpenFOAM
- [GPU 指南](../en/gpu_guide.md) — 高级 GPU 使用和性能调优
- [迁移指南](../en/migration_guide.md) — 从 OpenFOAM 迁移

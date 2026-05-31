# pyOpenFOAM 快速入门指南

## 目录

- [环境要求](#环境要求)
- [安装](#安装)
- [快速开始：加载 OpenFOAM 算例](#快速开始加载-openfoam-算例)
- [创建简单算例](#创建简单算例)
- [运行求解器](#运行求解器)
- [后处理结果](#后处理结果)
- [GPU 加速](#gpu-加速)
- [下一步](#下一步)

---

## 环境要求

| 依赖 | 最低版本 |
|------|----------|
| Python | 3.10+ |
| PyTorch | 2.0+ |
| NumPy | 1.24+ |
| SciPy | 1.10+ |

可选依赖：CUDA Toolkit（NVIDIA GPU）、mpi4py（并行计算）

---

## 安装

### 从 PyPI

```bash
# CPU 版本
pip install pyfoam-cfd

# GPU 版本（CUDA 12.x）
pip install pyfoam-cfd[gpu]

# MPI 版本
pip install pyfoam-cfd[mpi]

# 开发版本
pip install pyfoam-cfd[dev]
```

### 从源码

```bash
git clone https://github.com/alanZee/pyOpenFOAM.git
cd pyOpenFOAM
pip install -e ".[dev]"
```

### 验证安装

```python
import pyfoam
print(pyfoam.__version__)

from pyfoam.core import DeviceManager
dm = DeviceManager()
print(dm.device)  # device('cpu') 或 device('cuda')
```

---

## 快速开始：加载 OpenFOAM 算例

pyOpenFOAM 可读取任意标准 OpenFOAM 算例目录：

```python
from pyfoam.io.case import Case

case = Case("path/to/incompressible/simpleFoam/pitzDaily")

# 检查配置
print(case.controlDict["application"])  # "simpleFoam"
print(case.get_start_time())            # 0.0
print(case.get_end_time())              # 100.0

# 列出时间目录和场
print(case.time_dirs)
print(case.list_fields(time=0))         # ['U', 'p', 'nut']

# 读取场数据
field_data = case.read_field("U", time=0)
```

---

## 创建简单算例

### 程序化创建网格

```python
from pyfoam.mesh import PolyMesh, FvMesh
import torch

# 2 单元 2D 四边形网格
mesh = PolyMesh.from_raw(
    points=[[0,0,0], [1,0,0], [2,0,0],
            [0,1,0], [1,1,0], [2,1,0]],
    faces=[[0,1,4,3], [1,2,5,4],  # 内部面
           [0,3], [2,5], [0,1], [3,4]],  # 边界面
    owner=[0, 1, 0, 1, 0, 0],
    neighbour=[1],
    boundary=[
        {"name": "left",   "type": "patch", "startFace": 2, "nFaces": 1},
        {"name": "right",  "type": "patch", "startFace": 3, "nFaces": 1},
        {"name": "bottom", "type": "wall",  "startFace": 4, "nFaces": 1},
        {"name": "top",    "type": "wall",  "startFace": 5, "nFaces": 1},
    ],
)

fv = FvMesh.from_poly_mesh(mesh)
fv.compute_geometry()
```

### 创建场和边界条件

```python
from pyfoam.fields import volScalarField, volVectorField
from pyfoam.boundary import BoundaryCondition, Patch

# 速度场和压力场
U = volVectorField(fv, "U")
U.assign(torch.zeros(fv.n_cells, 3))

p = volScalarField(fv, "p")
p.assign(torch.zeros(fv.n_cells))

# 入口固定速度
inlet = Patch(
    name="inlet",
    face_indices=torch.tensor([2]),
    face_normals=torch.tensor([[-1.0, 0.0, 0.0]]),
    face_areas=torch.tensor([1.0]),
    delta_coeffs=torch.tensor([100.0]),
    owner_cells=torch.tensor([0]),
)

inlet_bc = BoundaryCondition.create("fixedValue", inlet, coeffs={"value": 1.0})
```

---

## 运行求解器

### 方式一：使用应用级求解器（推荐）

```python
from pyfoam.applications import SimpleFoam

solver = SimpleFoam("path/to/case")
solver.run()
```

### 方式二：使用底层 SIMPLE 求解器

```python
from pyfoam.solvers.simple import SIMPLESolver, SIMPLEConfig

config = SIMPLEConfig(
    relaxation_factor_U=0.7,
    relaxation_factor_p=0.3,
)

solver = SIMPLESolver(fv, config)
U_result, p_result, phi_result, convergence = solver.solve(
    U.internal_field, p.internal_field, torch.zeros(fv.n_faces),
    max_outer_iterations=100,
    tolerance=1e-4,
)

print(f"收敛：{convergence.converged}")
print(f"迭代：{convergence.outer_iterations}")
print(f"连续性误差：{convergence.continuity_error:.6e}")
```

### 方式三：使用线性求解器

```python
from pyfoam.solvers.pcg import PCGSolver
from pyfoam.solvers.pbicgstab import PBiCGSTABSolver
from pyfoam.solvers.gamg import GAMGSolver

# 对称系统（压力方程）
pcg = PCGSolver(tolerance=1e-6, max_iter=1000, preconditioner="DIC")
solution, iters, res = pcg(matrix, source, x0, 1e-6, 1000)

# 非对称系统（动量方程）
bicg = PBiCGSTABSolver(tolerance=1e-6, max_iter=1000, preconditioner="DILU")

# 多重网格（大规模问题）
gamg = GAMGSolver(tolerance=1e-6, max_iter=100)
```

---

## 后处理结果

### 力和力矩

```python
from pyfoam.postprocessing import Forces

forces = Forces(mesh, patches=["wing"], rho_ref=1.225)
F = forces.compute(U, p)
print(f"阻力: {F.drag}, 升力: {F.lift}")
```

### y+ 计算

```python
from pyfoam.postprocessing import YPlus

yplus = YPlus(mesh, U, nut, nu=1.5e-5)
y = yplus.compute()
```

### VTK 输出（ParaView 可视化）

```python
from pyfoam.postprocessing import FoamToVTK

FoamToVTK("path/to/case").export()
# 用 ParaView 打开 path/to/case/VTK/*.vtk
```

### 场运算

```python
from pyfoam.postprocessing import FieldOperations

ops = FieldOperations(mesh)
grad_p = ops.gradient(p)
div_U = ops.divergence(U)
curl_U = ops.curl(U)
```

---

## GPU 加速

```python
from pyfoam.core import device_context

# 自动检测最佳设备
dm = DeviceManager()
print(dm.capabilities.available_devices)

# 强制 GPU
with device_context("cuda"):
    fv = FvMesh.from_poly_mesh(mesh)
    solver = SIMPLESolver(fv, config)
    # 所有张量自动在 GPU 上
```

### 性能提示

1. **使用 float64** — CFD 收敛需要双精度，float32 会发散
2. **预计算几何** — 仿真循环前调用 `mesh.compute_geometry()`
3. **批量操作** — PyTorch 批量处理最快，避免 Python 循环
4. **先做 profiling** — 用 `torch.profiler` 找瓶颈

---

## 下一步

| 文档 | 内容 |
|------|------|
| [API 索引](../api/README.md) | 24 个模块的完整 API 参考 |
| [模块详细 API](../api/modules.md) | 每个类和函数的详细说明 |
| [迁移指南](../migration_guide.md) | OpenFOAM 到 pyOpenFOAM 的映射 |
| [GPU 指南](gpu_guide.md) | 多 GPU、性能调优 |
| [架构设计](architecture.md) | 整体架构和设计决策 |

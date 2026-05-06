# pyOpenFOAM 入门指南

## 安装

### 前置条件

- Python 3.10 或更高版本
- PyTorch 2.0 或更高版本

### 从 PyPI 安装

```bash
# 基本安装（仅 CPU）
pip install pyfoam-cfd

# 含 GPU 支持（CUDA 12.x）
pip install pyfoam-cfd[gpu]

# 含 MPI 支持
pip install pyfoam-cfd[mpi]

# 含可视化工具
pip install pyfoam-cfd[viz]

# 开发安装
pip install pyfoam-cfd[dev]
```

### 从源码安装

```bash
git clone https://github.com/pyOpenFOAM/pyOpenFOAM.git
cd pyOpenFOAM
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

## 快速开始：加载 OpenFOAM 算例

pyOpenFOAM 可以读取任何标准 OpenFOAM 算例目录。以下是加载和检查算例的方法：

```python
from pyfoam.io.case import Case

# 加载 OpenFOAM 算例
case = Case("path/to/incompressible/simpleFoam/pitzDaily")

# 检查配置
print(case.controlDict["application"])  # "simpleFoam"
print(case.get_start_time())            # 0.0
print(case.get_end_time())              # 100.0
print(case.get_delta_t())               # 1.0

# 列出时间目录
print(case.time_dirs)  # ['0', '1', '2', ..., '100']

# 列出时间 0 处的场
print(case.list_fields(time=0))  # ['U', 'p', 'nut']

# 读取场
field_data = case.read_field("U", time=0)
print(field_data.dimensions)  # [0, 1, -1, 0, 0, 0, 0]  (m/s)
```

## 使用网格

### 加载网格

```python
from pyfoam.io.case import Case
from pyfoam.mesh import FvMesh

case = Case("path/to/case")
mesh_data = case.mesh  # 来自 polyMesh 目录的原始 MeshData

# 创建 FvMesh（自动计算几何）
fv_mesh = FvMesh(
    points=mesh_data.points,
    faces=mesh_data.faces,
    owner=mesh_data.owner,
    neighbour=mesh_data.neighbour,
    boundary=mesh_data.boundary,
)

# 或预计算所有几何量
fv_mesh.compute_geometry()
```

### 网格属性

```python
print(fv_mesh.n_points)           # 顶点数
print(fv_mesh.n_cells)            # 单元数
print(fv_mesh.n_faces)            # 总面数（内部 + 边界）
print(fv_mesh.n_internal_faces)   # 仅内部面

# 几何量（首次访问时惰性计算）
print(fv_mesh.cell_centres.shape)       # (n_cells, 3)
print(fv_mesh.cell_volumes.shape)       # (n_cells,)
print(fv_mesh.face_centres.shape)       # (n_faces, 3)
print(fv_mesh.face_areas.shape)         # (n_faces, 3) — 法向 × 面积
print(fv_mesh.face_weights.shape)       # (n_faces,) — 插值权重
print(fv_mesh.delta_coefficients.shape) # (n_faces,) — 1/距离

# 导出量
print(fv_mesh.face_normals.shape)       # (n_faces, 3) — 单位法向
print(fv_mesh.total_volume)             # 标量 — 所有单元体积之和
```

### 程序化创建网格

```python
from pyfoam.mesh import PolyMesh

# 定义一个简单的 2D 四边形网格（2 个单元）
mesh = PolyMesh.from_raw(
    points=[
        [0, 0, 0], [1, 0, 0], [2, 0, 0],
        [0, 1, 0], [1, 1, 0], [2, 1, 0],
    ],
    faces=[
        [0, 1, 4, 3],  # 内部面 0
        [1, 2, 5, 4],  # 内部面 1
        [0, 3],         # 边界面 2（左）
        [2, 5],         # 边界面 3（右）
        [0, 1],         # 边界面 4（底部）
        [3, 4],         # 边界面 5（顶部）
    ],
    owner=[0, 1, 0, 1, 0, 0],
    neighbour=[1],  # 仅内部面有邻居
    boundary=[
        {"name": "left",   "type": "patch", "startFace": 2, "nFaces": 1},
        {"name": "right",  "type": "patch", "startFace": 3, "nFaces": 1},
        {"name": "bottom", "type": "wall",  "startFace": 4, "nFaces": 1},
        {"name": "top",    "type": "wall",  "startFace": 5, "nFaces": 1},
    ],
)

print(mesh)  # PolyMesh(n_points=6, n_faces=6, n_cells=2, ...)
```

## 使用场

### 创建体积场

```python
import torch
from pyfoam.fields import volScalarField, volVectorField
from pyfoam.mesh import FvMesh

# 假设 mesh 是一个 FvMesh 实例
# 创建标量压力场
p = volScalarField(mesh, "p")
p.assign(torch.zeros(mesh.n_cells))  # 初始化为零

# 创建速度场
U = volVectorField(mesh, "U")
U.assign(torch.zeros(mesh.n_cells, 3))  # 初始化为零

# 设置均匀值
p.assign(torch.ones(mesh.n_cells) * 101325.0)  # 1 个大气压
```

### 场算术

```python
# 场支持标准算术运算
p1 = volScalarField(mesh, "p1")
p2 = volScalarField(mesh, "p2")
p1.assign(torch.ones(mesh.n_cells))
p2.assign(torch.ones(mesh.n_cells) * 2.0)

# 加法（量纲必须匹配）
p_sum = p1 + p2  # 值为 3.0 的新场

# 标量乘法（场必须无量纲才能乘以标量）
p_scaled = p1 * 2.0  # 值为 2.0 的新场

# 原地操作
p1 += p2       # p1 现在值为 3.0
p1 *= 0.5      # p1 现在值为 1.5
```

### 使用边界条件

```python
from pyfoam.boundary import BoundaryCondition, Patch

# 创建面片描述符
inlet_patch = Patch(
    name="inlet",
    face_indices=torch.tensor([0, 1, 2]),
    face_normals=torch.tensor([[-1.0, 0.0, 0.0]] * 3),
    face_areas=torch.tensor([0.01, 0.01, 0.01]),
    delta_coeffs=torch.tensor([100.0, 100.0, 100.0]),
    owner_cells=torch.tensor([0, 1, 2]),
)

# 使用 RTS（运行时选择）创建边界条件
inlet_bc = BoundaryCondition.create(
    "fixedValue",
    inlet_patch,
    coeffs={"value": 1.0},  # 均匀速度 1 m/s
)

# 应用到场
velocity = torch.zeros(10, 3)  # 10 个单元，3D 速度
inlet_bc.apply(velocity, patch_idx=0)

# 获取 FVM 组装的矩阵贡献
diag = torch.zeros(10)
source = torch.zeros(10)
diag, source = inlet_bc.matrix_contributions(velocity, 10, diag, source)
```

### 可用边界条件

| 类型名 | 类 | 描述 |
|--------|---|------|
| `fixedValue` | `FixedValueBC` | 给定值（罚函数法） |
| `zeroGradient` | `ZeroGradientBC` | 零法向梯度（Neumann） |
| `fixedGradient` | `FixedGradientBC` | 给定法向梯度 |
| `noSlip` | `NoSlipBC` | 零速度（壁面） |
| `cyclic` | `CyclicBC` | 周期耦合 |
| `symmetryPlane` | `SymmetryBC` | 对称面 |
| `inletOutlet` | `InletOutletBC` | 流向切换 |
| `nutkWallFunction` | `NutkWallFunctionBC` | 湍流粘性壁面函数 |
| `kqRWallFunction` | `KqRWallFunctionBC` | 湍流量壁面函数 |

## 运行 SIMPLE 求解器

SIMPLE（压力耦合方程的半隐式方法）算法是定常不可压缩流的标准求解器。

```python
import torch
from pyfoam.mesh import FvMesh
from pyfoam.fields import volScalarField, volVectorField
from pyfoam.solvers.simple import SIMPLESolver, SIMPLEConfig

# 创建网格和场（如上所示）
mesh = FvMesh(...)
U = volVectorField(mesh, "U")
p = volScalarField(mesh, "p")

# 配置求解器
config = SIMPLEConfig(
    relaxation_factor_U=0.7,   # 速度欠松弛
    relaxation_factor_p=0.3,   # 压力欠松弛
    n_correctors=1,            # 压力修正步数
)

# 创建求解器
solver = SIMPLESolver(mesh, config)

# 运行（U、p、phi 是原始张量）
U_tensor = U.internal_field
p_tensor = p.internal_field
phi_tensor = torch.zeros(mesh.n_faces)  # 面通量

U_result, p_result, phi_result, convergence = solver.solve(
    U_tensor, p_tensor, phi_tensor,
    max_outer_iterations=100,
    tolerance=1e-4,
)

print(f"收敛：{convergence.converged}")
print(f"迭代次数：{convergence.outer_iterations}")
print(f"连续性误差：{convergence.continuity_error:.6e}")
```

## 直接使用线性求解器

### PCG 求解器（对称系统）

```python
from pyfoam.core import LduMatrix
from pyfoam.solvers.pcg import PCGSolver

# 假设 matrix 是一个 LduMatrix 实例
solver = PCGSolver(
    tolerance=1e-6,
    rel_tol=0.01,
    max_iter=1000,
    preconditioner="DIC",  # 对角不完全 Cholesky
)

# 求解 A x = b
solution, iterations, residual = solver(matrix, source, x0, 1e-6, 1000)
```

### PBiCGStab 求解器（非对称系统）

```python
from pyfoam.solvers.pbicgstab import PBiCGSTABSolver

solver = PBiCGSTABSolver(
    tolerance=1e-6,
    max_iter=1000,
    preconditioner="DILU",  # 对角不完全 LU
)

solution, iterations, residual = solver(matrix, source, x0, 1e-6, 1000)
```

### GAMG 求解器（多重网格）

```python
from pyfoam.solvers.gamg import GAMGSolver

solver = GAMGSolver(
    tolerance=1e-6,
    max_iter=100,
    n_pre_smooth=2,
    n_post_smooth=2,
    max_levels=10,
)

solution, iterations, residual = solver(matrix, source, x0, 1e-6, 100)
```

## GPU 加速

### 自动设备选择

pyOpenFOAM 自动检测并使用最佳可用设备：

```python
from pyfoam.core import DeviceManager

dm = DeviceManager()
print(dm.capabilities.available_devices)  # ['cpu'] 或 ['cpu', 'cuda'] 等
print(dm.device)  # device('cuda')（如果可用），否则 device('cpu')
```

### 手动设备选择

```python
from pyfoam.core import device_context
import torch

# 强制 CPU 计算
with device_context(device='cpu'):
    mesh = FvMesh(...)  # 所有张量在 CPU 上
    solver = SIMPLESolver(mesh, ...)

# 强制 GPU 计算（需要 CUDA）
with device_context(device='cuda'):
    mesh = FvMesh(...)  # 所有张量在 GPU 上
    solver = SIMPLESolver(mesh, ...)
```

### 性能提示

1. **使用 float64** — CFD 收敛需要双精度。float32 会导致发散。
2. **预计算几何** — 在仿真循环前调用 `mesh.compute_geometry()`。
3. **批量操作** — PyTorch 操作在批量处理时最快；避免对单元/面的 Python 循环。
4. **先做性能分析** — 使用 `torch.profiler` 在优化前识别瓶颈。

## 下一步

- 阅读 [API 参考](api_reference.md) 获取完整的类和函数文档。
- 如果你来自 OpenFOAM，请参阅[迁移指南](migration_guide.md)。
- 查看 [GPU 指南](gpu_guide.md) 了解高级 GPU 用法和性能调优。

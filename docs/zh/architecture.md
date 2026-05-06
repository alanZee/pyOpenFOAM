# pyOpenFOAM 架构文档

## 概述

pyOpenFOAM 是 OpenFOAM 计算流体力学（CFD）能力的纯 Python 重写版本，使用 PyTorch 作为张量后端实现 GPU 加速仿真。架构设计保留了 OpenFOAM 的有限体积法（FVM）框架，同时利用 Python 的表达能力和 PyTorch 的硬件加速特性。

## 设计原则

1. **OpenFOAM 兼容性** — 原生支持所有 OpenFOAM 文件格式（网格、场、字典、边界条件）。
2. **GPU 优先** — 所有张量操作通过 PyTorch 路由，实现透明的 CPU/CUDA/MPS 加速。
3. **默认 float64** — CFD 收敛需要双精度；float32 会导致压力-速度耦合算法发散。
4. **惰性计算** — 几何量（单元体积、面面积、插值权重）在首次访问时计算并缓存。
5. **运行时选择（RTS）** — 边界条件使用类级注册表，镜像 OpenFOAM 的 RTS 机制。

## 模块结构

```
pyfoam/
├── core/               # 基础层
│   ├── device.py       # DeviceManager、TensorConfig、device_context
│   ├── dtype.py        # CFD_DTYPE、INDEX_DTYPE、数据类型工具
│   ├── backend.py      # scatter_add、gather、sparse_coo_tensor、sparse_mm
│   ├── ldu_matrix.py   # LduMatrix — LDU 稀疏矩阵格式
│   ├── fv_matrix.py    # FvMatrix — 含源项、边界条件、松弛的 FVM 矩阵
│   └── sparse_ops.py   # ldu_to_coo_indices、extract_diagonal、csr_matvec
│
├── mesh/               # 网格表示
│   ├── poly_mesh.py    # PolyMesh — 原始拓扑（点、面、所有者、邻居）
│   ├── fv_mesh.py      # FvMesh — 扩展 PolyMesh，含几何量
│   ├── mesh_geometry.py # 面/单元几何计算函数
│   └── topology.py     # 面-单元连接工具
│
├── fields/             # 场类
│   ├── vol_fields.py   # volScalarField、volVectorField、volTensorField
│   ├── geometric_field.py  # GeometricField 基类
│   ├── field_arithmetic.py # FieldArithmeticMixin（+、-、*、/）
│   └── dimensions.py   # DimensionSet 量纲检查
│
├── boundary/           # 边界条件
│   ├── boundary_condition.py # BoundaryCondition 抽象基类 + RTS 注册表 + Patch
│   ├── boundary_field.py     # BoundaryField 容器
│   ├── fixed_value.py        # fixedValue（罚函数法）
│   ├── zero_gradient.py      # zeroGradient（零法向梯度）
│   ├── cyclic.py             # cyclic（周期耦合）
│   ├── symmetry.py           # symmetryPlane
│   ├── no_slip.py            # noSlip（零速度）
│   ├── wall_function.py      # nutkWallFunction、kqRWallFunction
│   ├── inlet_outlet.py       # inletOutlet（流向切换）
│   └── fixed_gradient.py     # fixedGradient（给定法向梯度）
│
├── io/                 # OpenFOAM 文件格式 I/O
│   ├── case.py         # Case — 完整算例目录表示
│   ├── dictionary.py   # FoamDict、parse_dict、parse_dict_file
│   ├── foam_file.py    # FoamFile — 通用 OpenFOAM 文件读取器
│   ├── field_io.py     # read_field、write_field
│   ├── mesh_io.py      # read_mesh、read_boundary
│   └── binary_io.py    # 二进制格式读写
│
├── discretisation/     # FVM 离散格式
│   ├── weights.py      # compute_centre_weights、compute_upwind_weights
│   ├── interpolation.py # InterpolationScheme、LinearInterpolation
│   └── schemes/        # UpwindInterpolation、LinearUpwindInterpolation、QuickInterpolation
│
├── solvers/            # 线性求解器和耦合求解器
│   ├── linear_solver.py    # LinearSolverBase、create_solver 工厂
│   ├── pcg.py              # PCGSolver — 预处理共轭梯度
│   ├── pbicgstab.py        # PBiCGSTABSolver — 预处理稳定双共轭梯度
│   ├── gamg.py             # GAMGSolver — 代数多重网格
│   ├── preconditioners.py  # DICPreconditioner、DILUPreconditioner
│   ├── residual.py         # ResidualMonitor、ConvergenceInfo
│   ├── coupled_solver.py   # CoupledSolverBase、CoupledSolverConfig、ConvergenceData
│   ├── simple.py           # SIMPLESolver — 定常不可压缩流
│   ├── piso.py             # PISOSolver — 瞬态不可压缩流
│   ├── pimple.py           # PIMPLESolver — 含外迭代的瞬态流
│   ├── pressure_equation.py # 组装、求解、速度修正、通量修正
│   └── rhie_chow.py        # Rhie-Chow 插值（速度-压力耦合）
│
├── turbulence/         # 湍流模型（计划中）
├── thermophysical/     # 热物理和输运（计划中）
├── models/             # 物理模型（计划中）
├── parallel/           # MPI 并行化（计划中）
└── utils/              # 工具函数（计划中）
```

## 数据流

### 网格加载

```
OpenFOAM 算例目录
    │
    ▼
Case("path/to/case")
    │  读取 system/controlDict、fvSchemes、fvSolution
    │  读取 constant/polyMesh/{points, faces, owner, neighbour, boundary}
    ▼
MeshData（原始 numpy 数组）
    │
    ▼
PolyMesh（拓扑张量，位于配置设备上）
    │
    ▼
FvMesh（惰性几何：cell_centres、cell_volumes、face_areas、face_weights、delta_coefficients）
```

### 场操作

```
volScalarField(mesh, "p", internal=initial_values)
    │
    ├── internal_field：(n_cells,) 张量，位于设备上
    ├── boundary_field：BoundaryCondition 对象列表
    │
    ▼
算术运算：p1 + p2、p * scalar 等
    │  通过 DimensionSet 进行量纲检查
    │  通过 TensorConfig 保证设备/数据类型一致性
    ▼
赋值：p.assign(new_values)
    │  应用边界条件
    ▼
I/O：write_field(p, path)
```

### FVM 组装

```
∇·(φ) + ∇²(φ) = S 的离散
    │
    ▼
InterpolationScheme（从单元值到面值）
    │  LinearInterpolation：φ_f = w·φ_P + (1-w)·φ_N
    │  UpwindInterpolation：φ_f = φ_上游
    ▼
LduMatrix 组装
    │  diag：  (n_cells,) — 对角系数
    │  lower：(n_internal_faces,) — 所有者侧非对角
    │  upper：(n_internal_faces,) — 邻居侧非对角
    ▼
FvMatrix（扩展 LduMatrix）
    │  source：  (n_cells,) — 右端项
    │  通过 BC.matrix_contributions() 添加边界贡献
    │  通过 FvMatrix.relax() 应用欠松弛
    ▼
线性求解：FvMatrix.solve(solver, x0, tolerance, max_iter)
    │  PCG（对称）、PBiCGStab（非对称）、GAMG（多重网格）
    ▼
解张量
```

### 耦合求解器循环（SIMPLE）

```
每次外迭代：
    │
    ├── 1. 动量预测：求解 A_p·U* = H(U) - ∇p
    │       应用欠松弛（α_U）
    │
    ├── 2. 计算 HbyA = H(U*) / A_p
    │
    ├── 3. 计算面通量 φ_HbyA（Rhie-Chow 插值）
    │
    ├── 4. 组装压力修正方程：
    │       ∇²(1/A_p, p') = ∇·(φ_HbyA)
    │
    ├── 5. 求解压力修正 p'（PCG）
    │       p = α_p·p' + (1-α_p)·p_old
    │
    ├── 6. 修正速度：U = HbyA - (1/A_p)·∇p
    │
    ├── 7. 修正面通量：φ = φ_HbyA - (1/A_p)_f·∇p_f
    │
    └── 8. 检查收敛：continuity_error < tolerance
```

## 设备和数据类型管理

### DeviceManager（单例）

```python
from pyfoam.core import DeviceManager

dm = DeviceManager()
print(dm.capabilities)  # DeviceCapabilities(cpu=True, cuda=False, mps=False)
print(dm.device)        # device('cpu') — 自动选择最佳可用设备
dm.device = 'cuda'      # 手动覆盖（不可用时抛出 ValueError）
```

优先级：CUDA > MPS > CPU。

### TensorConfig（全局默认）

```python
from pyfoam.core import TensorConfig

config = TensorConfig()  # 默认：float64，最佳设备
t = config.zeros(100)    # 默认设备上的 float64 张量

with config.override(dtype=torch.float32, device='cpu'):
    t32 = config.zeros(100)  # float32 在 CPU 上，临时
# 上下文退出后恢复默认
```

### 模块级便捷函数

```python
from pyfoam.core import get_device, get_default_dtype, device_context

device = get_device()       # 当前默认设备
dtype = get_default_dtype() # torch.float64

with device_context(device='cuda'):
    # 所有 pyfoam 操作在此使用 CUDA
    pass
```

## 稀疏矩阵格式

### LDU 格式（OpenFOAM 原生）

LDU（下-对角-上）格式将 FVM 矩阵系数存储为三个扁平数组：

- **diag** `(n_cells,)` — 每个单元一个对角系数
- **lower** `(n_internal_faces,)` — 所有者侧非对角（行=所有者，列=邻居）
- **upper** `(n_internal_faces,)` — 邻居侧非对角（行=邻居，列=所有者）

面寻址（来自网格的 owner/neighbour 数组）将非对角条目连接到矩阵行。对于 FVM 组装，这比 CSR 更节省内存，因为网格拓扑已经提供了寻址。

### COO/CSR 转换

对于需要标准稀疏格式的线性求解器：

```python
coo = ldu_matrix.to_sparse_coo()   # 用于组装的 COO
csr = ldu_matrix.to_sparse_csr()   # 用于求解的 CSR
```

## 扩展 pyOpenFOAM

### 自定义边界条件

```python
from pyfoam.boundary import BoundaryCondition, Patch

@BoundaryCondition.register("myCustomBC")
class MyCustomBC(BoundaryCondition):
    def apply(self, field, patch_idx=None):
        # 修改边界面上的值
        return field

    def matrix_contributions(self, field, n_cells, diag=None, source=None):
        # 返回 (diag, source) 贡献
        if diag is None:
            diag = torch.zeros(n_cells)
        if source is None:
            source = torch.zeros(n_cells)
        return diag, source
```

### 自定义线性求解器

实现 `LinearSolver` 协议：

```python
from pyfoam.core.fv_matrix import LinearSolver

class MySolver:
    def __call__(self, matrix, source, x0, tolerance, max_iter):
        # 求解 A x = b
        # 返回 (solution, iterations, final_residual)
        ...
```

## 依赖项

| 包 | 版本 | 用途 |
|---|---|---|
| PyTorch | ≥ 2.0 | 张量后端，GPU 加速 |
| NumPy | ≥ 1.24 | 数组转换，文件 I/O |
| SciPy | ≥ 1.10 | 稀疏矩阵工具（可选） |

### 可选依赖

| 包 | 用途 |
|---|---|
| cupy-cuda12x | CUDA GPU 支持 |
| mpi4py | MPI 并行化 |
| pyvista、matplotlib | 可视化 |
| pytest、black、ruff、mypy | 开发工具 |

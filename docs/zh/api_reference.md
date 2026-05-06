# pyOpenFOAM API 参考

本文档提供 pyOpenFOAM 中所有公共类、函数和常量的综合参考。

## 目录

- [core — 基础层](#core--基础层)
  - [设备管理](#设备管理)
  - [数据类型工具](#数据类型工具)
  - [后端操作](#后端操作)
  - [LDU 矩阵](#ldu-矩阵)
  - [FvMatrix](#fvmatrix)
  - [稀疏操作](#稀疏操作)
- [mesh — 网格表示](#mesh--网格表示)
  - [PolyMesh](#polymesh)
  - [FvMesh](#fvmesh)
  - [几何函数](#几何函数)
  - [拓扑工具](#拓扑工具)
- [fields — 场类](#fields--场类)
  - [volScalarField](#volscalarfield)
  - [volVectorField](#volvectorfield)
  - [volTensorField](#voltensorfield)
- [boundary — 边界条件](#boundary--边界条件)
  - [BoundaryCondition](#boundarycondition)
  - [Patch](#patch)
  - [FixedValueBC](#fixedvaluebc)
  - [ZeroGradientBC](#zerogradientbc)
  - [CyclicBC](#cyclicbc)
  - [SymmetryBC](#symmetrybc)
  - [NoSlipBC](#noslipbc)
  - [InletOutletBC](#inletoutletbc)
  - [FixedGradientBC](#fixedgradientbc)
  - [壁面函数 BC](#壁面函数-bc)
- [io — 文件 I/O](#io--文件-io)
  - [Case](#case)
  - [字典解析](#字典解析)
- [discretisation — FVM 格式](#discretisation--fvm-格式)
  - [插值格式](#插值格式)
  - [权重函数](#权重函数)
- [solvers — 线性和耦合求解器](#solvers--线性和耦合求解器)
  - [线性求解器](#线性求解器)
  - [耦合求解器](#耦合求解器)

---

## core — 基础层

### 设备管理

#### `DeviceManager`

单例类，用于硬件检测和设备选择。

```python
from pyfoam.core import DeviceManager

dm = DeviceManager()
```

| 属性/方法 | 返回值 | 描述 |
|----------|--------|------|
| `capabilities` | `DeviceCapabilities` | 检测到的硬件（cpu、cuda、mps、cuda_devices） |
| `device` | `torch.device` | 当前选择的设备 |
| `device.setter(device)` | — | 设置活动设备（不可用时抛出 `ValueError`） |
| `is_available(device)` | `bool` | 检查设备类型是否可用 |

**`DeviceCapabilities`**（冻结数据类）：

| 字段 | 类型 | 描述 |
|------|------|------|
| `cpu` | `bool` | CPU 可用（始终为 True） |
| `cuda` | `bool` | CUDA 可用 |
| `mps` | `bool` | MPS（Apple Silicon）可用 |
| `cuda_devices` | `int` | CUDA 设备数量 |
| `available_devices` | `list[str]` | 可用设备名称列表 |

#### `TensorConfig`

CFD 操作的全局张量配置。默认使用 float64 和最佳可用设备。

```python
from pyfoam.core import TensorConfig
import torch

config = TensorConfig()
```

| 属性/方法 | 返回值 | 描述 |
|----------|--------|------|
| `dtype` | `torch.dtype` | 默认数据类型（float64） |
| `dtype.setter` | — | 设置默认数据类型 |
| `device` | `torch.device` | 当前设备 |
| `device.setter` | — | 设置当前设备 |
| `device_manager` | `DeviceManager` | 底层设备管理器 |
| `tensor(data, **kwargs)` | `torch.Tensor` | 使用默认值创建张量 |
| `zeros(*size, **kwargs)` | `torch.Tensor` | 创建零张量 |
| `ones(*size, **kwargs)` | `torch.Tensor` | 创建一张量 |
| `empty(*size, **kwargs)` | `torch.Tensor` | 创建空张量 |
| `full(*size, fill_value, **kwargs)` | `torch.Tensor` | 创建填充张量 |
| `override(dtype, device)` | 上下文管理器 | 临时数据类型/设备覆盖 |

#### 模块级函数

```python
from pyfoam.core import get_device, get_default_dtype, device_context
```

| 函数 | 返回值 | 描述 |
|------|--------|------|
| `get_device()` | `torch.device` | 当前默认设备 |
| `get_default_dtype()` | `torch.dtype` | 当前默认数据类型（float64） |
| `device_context(device, dtype)` | 上下文管理器 | 临时全局覆盖 |

---

### 数据类型工具

```python
from pyfoam.core import (
    CFD_DTYPE, CFD_REAL_DTYPE, CFD_COMPLEX_DTYPE, INDEX_DTYPE,
    is_floating, is_complex_dtype, promote_dtype, to_cfd_dtype,
    dtype_to_numpy, numpy_to_torch, real_dtype, complex_dtype, assert_floating,
)
```

| 常量 | 值 | 描述 |
|------|---|------|
| `CFD_DTYPE` | `torch.float64` | 默认 CFD 精度 |
| `CFD_REAL_DTYPE` | `torch.float64` | CFD_DTYPE 的别名 |
| `CFD_COMPLEX_DTYPE` | `torch.complex128` | 复数 CFD 数据类型 |
| `INDEX_DTYPE` | `torch.int64` | 网格索引数据类型 |

| 函数 | 签名 | 描述 |
|------|------|------|
| `is_floating(dtype)` | `dtype → bool` | 对 float16/32/64、complex64/128 返回 True |
| `is_complex_dtype(dtype)` | `dtype → bool` | 对 complex64/128 返回 True |
| `promote_dtype(*dtypes)` | `*dtypes → dtype` | 能表示所有输入的最宽数据类型 |
| `to_cfd_dtype(tensor)` | `tensor → tensor` | 如果是浮点则转换为 float64 |
| `dtype_to_numpy(dtype)` | `dtype → np.dtype` | Torch 到 NumPy 数据类型 |
| `numpy_to_torch(dtype)` | `np.dtype → dtype` | NumPy 到 torch 数据类型 |
| `real_dtype(dtype)` | `dtype → dtype` | 实数对应类型（complex64→float32） |
| `complex_dtype(dtype)` | `dtype → dtype` | 复数对应类型（float64→complex128） |
| `assert_floating(tensor, name)` | — | 如果不是浮点则抛出 TypeError |

---

### 后端操作

```python
from pyfoam.core import scatter_add, gather, sparse_coo_tensor, sparse_mm, Backend
```

#### `scatter_add(src, index, dim_size, *, dim=0, device=None, dtype=None)`

将 `src` 值累积到 `index` 指定位置的输出中。FVM 通量组装的核心原语。

- **src**：源值（例如面通量）。
- **index**：目标索引（例如所有者单元）。
- **dim_size**：输出维度大小。
- **返回**：形状为 `(dim_size,)` 的输出张量。

#### `gather(src, index, *, dim=0, device=None)`

从 `src` 中收集 `index` 指定位置的值。用于边界查找和邻居访问。

- **src**：源张量。
- **index**：要收集的索引。
- **返回**：与 `index` 形状相同的收集值。

#### `sparse_coo_tensor(indices, values, size, *, device=None, dtype=None)`

构建 COO 稀疏张量。用于矩阵组装期间。

- **indices**：`(ndim, nnz)` 非零坐标。
- **values**：`(nnz,)` 非零值。
- **size**：稀疏张量形状。
- **返回**：COO 稀疏张量。

#### `sparse_mm(mat, vec, *, device=None)`

稀疏矩阵-向量乘法。接受 COO 或 CSR 格式。

- **mat**：稀疏矩阵。
- **vec**：密集向量或矩阵。
- **返回**：`mat @ vec` 的密集结果。

#### `Backend` 类

面向对象的后端，将操作绑定到特定设备/数据类型：

```python
backend = Backend(device='cpu', dtype=torch.float32)
result = backend.scatter_add(src, index, dim_size=100)
```

| 属性/方法 | 描述 |
|----------|------|
| `config` | 绑定的 TensorConfig |
| `device` | 后端设备 |
| `dtype` | 后端数据类型 |
| `scatter_add(...)` | 使用后端配置的 scatter-add |
| `gather(...)` | 使用后端配置的 gather |
| `sparse_coo_tensor(...)` | 使用后端配置的 COO 构造 |
| `sparse_mm(...)` | 使用后端配置的稀疏矩阵乘法 |

---

### LDU 矩阵

#### `LduMatrix`

有限体积系统的 LDU 格式稀疏矩阵。以 OpenFOAM 原生布局存储系数。

```python
from pyfoam.core import LduMatrix

matrix = LduMatrix(n_cells, owner, neighbour, device=device, dtype=dtype)
```

| 参数 | 类型 | 描述 |
|------|------|------|
| `n_cells` | `int` | 矩阵维度（单元数） |
| `owner` | `torch.Tensor` | `(n_internal_faces,)` 所有者单元索引 |
| `neighbour` | `torch.Tensor` | `(n_internal_faces,)` 邻居单元索引 |

| 属性 | 形状 | 描述 |
|------|------|------|
| `n_cells` | `int` | 矩阵维度 |
| `n_internal_faces` | `int` | 每个三角形的非对角条目数 |
| `device` | `torch.device` | 张量设备 |
| `dtype` | `torch.dtype` | 浮点数据类型 |
| `owner` | `(n_internal_faces,)` | 所有者单元索引 |
| `neighbour` | `(n_internal_faces,)` | 邻居单元索引 |
| `diag` | `(n_cells,)` | 对角系数 |
| `lower` | `(n_internal_faces,)` | 下三角（所有者侧）系数 |
| `upper` | `(n_internal_faces,)` | 上三角（邻居侧）系数 |

| 方法 | 签名 | 描述 |
|------|------|------|
| `Ax(x)` | `(n_cells,) → (n_cells,)` | 矩阵-向量乘积 y = A·x |
| `add_to_diag(values)` | `(n_cells,) →` | 向对角线添加值 |
| `to_sparse_coo()` | `→ 稀疏 COO` | 转换为 COO 格式 |
| `to_sparse_csr()` | `→ 稀疏 CSR` | 转换为 CSR 格式 |

---

### FvMatrix

#### `FvMatrix(LduMatrix)`

含源项、边界条件和松弛支持的有限体积矩阵。

```python
from pyfoam.core import FvMatrix

matrix = FvMatrix(n_cells, owner, neighbour, device=device, dtype=dtype)
```

继承所有 `LduMatrix` 属性和方法，加上：

| 属性 | 形状 | 描述 |
|------|------|------|
| `source` | `(n_cells,)` | 右端向量 |
| `relaxation_factor` | `float` | 当前欠松弛因子 |

| 方法 | 签名 | 描述 |
|------|------|------|
| `add_boundary_contribution(bc, field)` | — | 向矩阵添加 BC 贡献 |
| `add_explicit_source(values)` | `(n_cells,) →` | 向右端项添加值 |
| `relax(field_old, factor)` | — | 应用欠松弛 |
| `set_reference(cell_index, value)` | — | 固定参考压力 |
| `solve(solver, x0, tolerance, max_iter)` | → `(解, 迭代数, 残差)` | 求解 A·x = b |
| `residual(x)` | `(n_cells,) → (n_cells,)` | 计算 r = b - A·x |

#### `LinearSolver` 协议

线性求解器的协议：

```python
class LinearSolver(Protocol):
    def __call__(
        self,
        matrix: LduMatrix,
        source: torch.Tensor,
        x0: torch.Tensor,
        tolerance: float,
        max_iter: int,
    ) -> tuple[torch.Tensor, int, float]: ...
```

---

### 稀疏操作

```python
from pyfoam.core import ldu_to_coo_indices, extract_diagonal, csr_matvec
```

| 函数 | 签名 | 描述 |
|------|------|------|
| `ldu_to_coo_indices(owner, neighbour, n_cells)` | → `(diag_idx, lower_idx, upper_idx)` | 从 LDU 寻址构建 COO 索引 |
| `extract_diagonal(mat)` | → `(n,)` | 从稀疏/密集矩阵提取对角线 |
| `csr_matvec(mat, vec)` | → 密集 | CSR 稀疏矩阵-向量乘积 |

---

## mesh — 网格表示

### PolyMesh

原始拓扑网格表示。

```python
from pyfoam.mesh import PolyMesh

mesh = PolyMesh(points, faces, owner, neighbour, boundary)
```

| 参数 | 类型 | 描述 |
|------|------|------|
| `points` | `(n_points, 3)` 张量 | 顶点位置 |
| `faces` | `list[tensor]` | 每个面的点索引 |
| `owner` | `(n_faces,)` 张量 | 每个面的所有者单元 |
| `neighbour` | `(n_internal_faces,)` 张量 | 每个内部面的邻居单元 |
| `boundary` | `list[dict]` | 面片描述符：`{name, type, startFace, nFaces}` |

| 属性 | 返回值 | 描述 |
|------|--------|------|
| `points` | `(n_points, 3)` | 顶点位置 |
| `faces` | `list[tensor]` | 面-顶点索引 |
| `owner` | `(n_faces,)` | 所有者单元索引 |
| `neighbour` | `(n_internal_faces,)` | 邻居单元索引 |
| `boundary` | `list[dict]` | 边界面片 |
| `n_points` | `int` | 顶点数 |
| `n_faces` | `int` | 总面数 |
| `n_cells` | `int` | 单元数 |
| `n_internal_faces` | `int` | 内部面数 |
| `device` | `torch.device` | 张量设备 |
| `dtype` | `torch.dtype` | 浮点数据类型 |

| 方法 | 签名 | 描述 |
|------|------|------|
| `face_points(face_idx)` | → `(n_verts, 3)` | 面的顶点位置 |
| `is_boundary_face(face_idx)` | → `bool` | 是否为边界面 |
| `patch_faces(patch_idx)` | → `range` | 边界面片的面范围 |
| `from_raw(points, faces, owner, neighbour, boundary)` | 类方法 | 从 Python 列表构造 |

---

### FvMesh

扩展 PolyMesh，含惰性计算的几何量。

```python
from pyfoam.mesh import FvMesh

mesh = FvMesh(points, faces, owner, neighbour, boundary)
mesh.compute_geometry()  # 预计算所有量
```

继承所有 `PolyMesh` 属性，加上：

| 属性 | 形状 | 描述 |
|------|------|------|
| `face_centres` | `(n_faces, 3)` | 面中心位置 |
| `face_areas` | `(n_faces, 3)` | 面面积矢量（法向 × 面积） |
| `cell_centres` | `(n_cells, 3)` | 单元中心位置 |
| `cell_volumes` | `(n_cells,)` | 单元体积 |
| `face_weights` | `(n_faces,)` | 线性插值权重 |
| `delta_coefficients` | `(n_faces,)` | 扩散 delta 系数 |
| `face_areas_magnitude` | `(n_faces,)` | 面面积大小 |
| `face_normals` | `(n_faces, 3)` | 单位面法向 |
| `total_volume` | 标量 | 所有单元体积之和 |

| 方法 | 签名 | 描述 |
|------|------|------|
| `compute_geometry()` | — | 预计算所有几何量 |
| `from_poly_mesh(mesh)` | 类方法 | 从 PolyMesh 创建 FvMesh |

---

### 几何函数

```python
from pyfoam.mesh.mesh_geometry import (
    compute_face_centres,
    compute_face_area_vectors,
    compute_cell_volumes_and_centres,
    compute_face_weights,
    compute_delta_coefficients,
)
```

| 函数 | 返回值 | 描述 |
|------|--------|------|
| `compute_face_centres(points, faces)` | `(n_faces, 3)` | 面中心位置 |
| `compute_face_area_vectors(points, faces)` | `(n_faces, 3)` | 通过扇形三角剖分计算面积矢量 |
| `compute_cell_volumes_and_centres(...)` | `(volumes, centres)` | 通过四面体分解计算单元体积和中心 |
| `compute_face_weights(cell_centres, face_centres, owner, neighbour, n_internal)` | `(n_faces,)` | 基于距离的线性插值权重 |
| `compute_delta_coefficients(...)` | `(n_faces,)` | 1/|d·n̂| 用于扩散 |

---

### 拓扑工具

```python
from pyfoam.mesh.topology import (
    validate_owner_neighbour,
    internal_face_mask,
    boundary_face_mask,
    build_cell_to_faces,
    build_face_to_cells,
    cell_neighbours,
)
```

| 函数 | 返回值 | 描述 |
|------|--------|------|
| `validate_owner_neighbour(owner, neighbour, n_cells, n_internal)` | — | 验证约定；抛出 ValueError |
| `internal_face_mask(n_faces, n_internal)` | `(n_faces,)` bool | 内部面为 True |
| `boundary_face_mask(n_faces, n_internal)` | `(n_faces,)` bool | 边界面为 True |
| `build_cell_to_faces(owner, neighbour, n_cells, n_internal)` | `list[tensor]` | 每个单元的面索引 |
| `build_face_to_cells(owner, neighbour, n_internal)` | `(n_faces, 2)` | 每个面的所有者/邻居（边界为 -1） |
| `cell_neighbours(cell, owner, neighbour, n_internal)` | `(n_neigh,)` | 唯一的邻居单元索引 |

---

## fields — 场类

### volScalarField

单元中心标量场。形状：`(n_cells,)`。

```python
from pyfoam.fields import volScalarField

p = volScalarField(mesh, "p")
p.assign(torch.zeros(mesh.n_cells))
```

| 参数 | 类型 | 描述 |
|------|------|------|
| `mesh` | `FvMesh` | 有限体积网格 |
| `name` | `str` | 场名（例如 `"p"`） |
| `dimensions` | `DimensionSet` | 物理量纲（可选） |
| `internal` | `tensor` 或 `float` | 初始值（可选） |
| `boundary` | `BoundaryField` | 边界条件（可选） |

**继承的属性：**
- `name`、`dimensions`、`internal_field`、`boundary_field`、`mesh`、`device`、`dtype`、`n_cells`

**继承的方法：**
- `assign(values)` — 设置内部场值
- `to(device, dtype)` — 复制到不同设备/数据类型

**算术运算符：**
- `+`、`-`、`*`、`/`（含量纲检查）
- `+=`、`-=`、`*=`、`/=`（原地）

---

### volVectorField

单元中心矢量场。形状：`(n_cells, 3)`。

```python
from pyfoam.fields import volVectorField

U = volVectorField(mesh, "U")
U.assign(torch.zeros(mesh.n_cells, 3))
```

与 `volScalarField` 相同的 API，但形状为 `(n_cells, 3)`。

---

### volTensorField

单元中心张量场。形状：`(n_cells, 3, 3)`。

```python
from pyfoam.fields import volTensorField

tau = volTensorField(mesh, "tau")
tau.assign(torch.zeros(mesh.n_cells, 3, 3))
```

与 `volScalarField` 相同的 API，但形状为 `(n_cells, 3, 3)`。

---

## boundary — 边界条件

### BoundaryCondition

含运行时选择（RTS）注册表的抽象基类。

```python
from pyfoam.boundary import BoundaryCondition

# 列出可用类型
print(BoundaryCondition.available_types())
# ['cyclic', 'fixedGradient', 'fixedValue', 'inletOutlet', 'kqRWallFunction',
#  'noSlip', 'nutkWallFunction', 'symmetryPlane', 'zeroGradient']

# 按名称创建
bc = BoundaryCondition.create("fixedValue", patch, coeffs={"value": 1.0})
```

| 类方法 | 签名 | 描述 |
|--------|------|------|
| `register(name)` | 装饰器 | 在 `name` 下注册 BC 类 |
| `create(name, patch, coeffs)` | → `BoundaryCondition` | 工厂：按注册名创建 |
| `available_types()` | → `list[str]` | 排序后的注册名列表 |

| 属性 | 返回值 | 描述 |
|------|--------|------|
| `patch` | `Patch` | 绑定的面片 |
| `coeffs` | `dict` | BC 系数 |
| `type_name` | `str` | 注册类型名 |

| 抽象方法 | 签名 | 描述 |
|----------|------|------|
| `apply(field, patch_idx)` | → `tensor` | 修改边界面值 |
| `matrix_contributions(field, n_cells, diag, source)` | → `(diag, source)` | FVM 矩阵贡献 |

---

### Patch

轻量级边界面片描述符。

```python
from pyfoam.boundary import Patch

patch = Patch(
    name="inlet",
    face_indices=torch.tensor([0, 1, 2]),
    face_normals=torch.tensor([[-1, 0, 0.0]] * 3),
    face_areas=torch.tensor([0.01, 0.01, 0.01]),
    delta_coeffs=torch.tensor([100.0, 100.0, 100.0]),
    owner_cells=torch.tensor([0, 1, 2]),
)
```

| 字段 | 类型 | 描述 |
|------|------|------|
| `name` | `str` | 面片名 |
| `face_indices` | `(n_faces,)` int 张量 | 面索引 |
| `face_normals` | `(n_faces, 3)` 张量 | 向外单位法向 |
| `face_areas` | `(n_faces,)` 张量 | 面面积 |
| `delta_coeffs` | `(n_faces,)` 张量 | 1/距离系数 |
| `owner_cells` | `(n_faces,)` 张量 | 相邻单元索引 |
| `neighbour_patch` | `str` 或 `None` | 耦合面片名（用于 cyclic） |

| 属性/方法 | 描述 |
|----------|------|
| `n_faces` | 面片中的面数 |
| `to(device)` | 将张量移到设备上的副本 |

---

### FixedValueBC

给定值边界条件。使用罚函数法进行矩阵贡献。

- **注册名**：`"fixedValue"`
- **系数**：`{"value": float 或 tensor}`
- **apply()**：将边界面设置为给定值。
- **matrix_contributions()**：`diag[c] += deltaCoeff * area`，`source[c] += deltaCoeff * area * value`

---

### ZeroGradientBC

零法向梯度（Neumann）边界条件。

- **注册名**：`"zeroGradient"`
- **apply()**：将所有者单元值复制到边界面。
- **matrix_contributions()**：零贡献（构造上零通量）。

---

### CyclicBC

周期边界条件，耦合两个面片。

- **注册名**：`"cyclic"`
- **方法**：`set_neighbour_field(field)` — 设置耦合面片值。
- **apply()**：复制邻居面片值。
- **matrix_contributions()**：`diag[c] += deltaCoeff * area`，`source[c] += deltaCoeff * area * neighbourValue`

---

### SymmetryBC

对称面边界条件。

- **注册名**：`"symmetryPlane"`
- **apply()**：标量 — 零梯度；矢量 — 投影到切平面。
- **matrix_contributions()**：零贡献。

---

### NoSlipBC

零速度壁面边界条件。

- **注册名**：`"noSlip"`
- **apply()**：将边界面设为零。
- **matrix_contributions()**：`diag[c] += deltaCoeff * area`，`source[c] = 0`

---

### InletOutletBC

流向切换边界条件。

- **注册名**：`"inletOutlet"`
- **系数**：`{"value": float 或 tensor}` — 入口给定值。
- **apply(field, velocity=None)**：流入 → 给定值；流出 → 零梯度。
- **matrix_contributions(field, n_cells, ..., velocity=None)**：流入 → 罚函数；流出 → 零。

---

### FixedGradientBC

给定法向梯度边界条件。

- **注册名**：`"fixedGradient"`
- **系数**：`{"gradient": float 或 tensor}`
- **apply()**：`phi_face = phi_cell + gradient * d`
- **matrix_contributions()**：`source[c] += area * gradient`（仅显式通量）

---

### 壁面函数 BC

#### `NutkWallFunctionBC`

使用对数律的湍流粘性壁面函数。

- **注册名**：`"nutkWallFunction"`
- **系数**：`Cmu`（0.09）、`kappa`（0.41）、`E`（9.8）
- **方法**：`compute_nut(k, y, nu)` — 计算壁面处的 nu_t。

#### `KqRWallFunctionBC`

湍流量（k、q、R）的壁面函数。

- **注册名**：`"kqRWallFunction"`
- **系数**：`Cmu`（0.09）
- **方法**：`compute_k_wall(u_tau)` — k = u_tau^2 / sqrt(Cmu)

---

## io — 文件 I/O

### Case

完整的 OpenFOAM 算例目录表示。

```python
from pyfoam.io.case import Case

case = Case("path/to/case")
```

| 属性 | 返回值 | 描述 |
|------|--------|------|
| `root` | `Path` | 算例根目录 |
| `controlDict` | `FoamDict` | 解析后的 system/controlDict |
| `fvSchemes` | `FoamDict` | 解析后的 system/fvSchemes |
| `fvSolution` | `FoamDict` | 解析后的 system/fvSolution |
| `mesh` | `MeshData` | 来自 constant/polyMesh 的原始网格 |
| `boundary` | `list[BoundaryPatch]` | 边界面片定义 |
| `time_dirs` | `list[str]` | 排序后的时间目录名 |
| `constant_dir` | `Path` | constant/ 路径 |
| `system_dir` | `Path` | system/ 路径 |
| `mesh_dir` | `Path` | constant/polyMesh/ 路径 |

| 方法 | 签名 | 描述 |
|------|------|------|
| `get_time_dir(time)` | → `Path` | 时间目录路径 |
| `list_fields(time)` | → `list[str]` | 时间目录中的场文件 |
| `read_field(name, time)` | → `FieldData` | 读取场文件 |
| `has_field(name, time)` | → `bool` | 检查场是否存在 |
| `has_mesh()` | → `bool` | 检查网格文件是否存在 |
| `get_application()` | → `str` | 从 controlDict 获取应用名 |
| `get_start_time()` | → `float` | 从 controlDict 获取 startTime |
| `get_end_time()` | → `float` | 从 controlDict 获取 endTime |
| `get_delta_t()` | → `float` | 从 controlDict 获取 deltaT |

---

### 字典解析

```python
from pyfoam.io.dictionary import parse_dict, parse_dict_file, FoamDict
```

| 函数 | 签名 | 描述 |
|------|------|------|
| `parse_dict(text)` | `str → FoamDict` | 解析 OpenFOAM 字典文本 |
| `parse_dict_file(path)` | `Path → FoamDict` | 解析字典文件 |

`FoamDict` 是类字典容器，支持使用 OpenFOAM 语法的嵌套键访问。

---

## discretisation — FVM 格式

### 插值格式

```python
from pyfoam.discretisation import (
    LinearInterpolation,
    UpwindInterpolation,
    LinearUpwindInterpolation,
    QuickInterpolation,
)
```

| 类 | 阶数 | 描述 |
|---|------|------|
| `LinearInterpolation` | 2 阶 | `phi_f = w * phi_P + (1-w) * phi_N` |
| `UpwindInterpolation` | 1 阶 | 基于通量方向的上游值 |
| `LinearUpwindInterpolation` | 2 阶 | 迎风偏向，含梯度修正 |
| `QuickInterpolation` | 3 阶 | QUICK 格式，含延迟修正 |

---

### 权重函数

```python
from pyfoam.discretisation import compute_centre_weights, compute_upwind_weights
```

| 函数 | 返回值 | 描述 |
|------|--------|------|
| `compute_centre_weights(cell_centres, face_centres, owner, neighbour, n_internal, n_faces)` | `(n_faces,)` | 基于距离的线性权重 |
| `compute_upwind_weights(face_flux, n_internal, n_faces)` | `(weight_owner, weight_neigh)` | 二值迎风权重 |

---

## solvers — 线性和耦合求解器

### 线性求解器

#### `PCGSolver`

用于对称正定矩阵的预处理共轭梯度法。

```python
from pyfoam.solvers.pcg import PCGSolver

solver = PCGSolver(
    tolerance=1e-6,
    rel_tol=0.01,
    max_iter=1000,
    preconditioner="DIC",  # 或 "DILU" 或 "none"
)
solution, iters, residual = solver(matrix, source, x0, tolerance, max_iter)
```

#### `PBiCGSTABSolver`

用于一般（非对称）矩阵的预处理稳定双共轭梯度法。

```python
from pyfoam.solvers.pbicgstab import PBiCGSTABSolver

solver = PBiCGSTABSolver(
    tolerance=1e-6,
    rel_tol=0.01,
    max_iter=1000,
    preconditioner="DILU",  # 或 "DIC" 或 "none"
)
```

#### `GAMGSolver`

基于聚合粗化的代数多重网格求解器。

```python
from pyfoam.solvers.gamg import GAMGSolver

solver = GAMGSolver(
    tolerance=1e-6,
    max_iter=100,
    n_pre_smooth=2,
    n_post_smooth=2,
    max_levels=10,
    min_cells_coarse=10,
    smoother="PCG",
)
```

---

### 耦合求解器

#### `CoupledSolverConfig`

压力-速度耦合求解器的配置。

```python
from pyfoam.solvers.coupled_solver import CoupledSolverConfig

config = CoupledSolverConfig(
    p_solver="PCG",
    U_solver="PBiCGStab",
    p_tolerance=1e-6,
    U_tolerance=1e-6,
    p_max_iter=1000,
    U_max_iter=1000,
    n_non_orthogonal_correctors=0,
    relaxation_factor_p=1.0,
    relaxation_factor_U=0.7,
    relaxation_factor_phi=1.0,
)
```

#### `ConvergenceData`

跟踪耦合求解的收敛情况。

| 字段 | 类型 | 描述 |
|------|------|------|
| `p_residual` | `float` | 最终压力残差 |
| `U_residual` | `float` | 最终速度残差 |
| `continuity_error` | `float` | 全局连续性误差 |
| `outer_iterations` | `int` | 外迭代次数 |
| `converged` | `bool` | 解是否收敛 |
| `residual_history` | `list[dict]` | 逐迭代记录 |

#### `SIMPLESolver`

定常不可压缩流的 SIMPLE 算法。

```python
from pyfoam.solvers.simple import SIMPLESolver, SIMPLEConfig

config = SIMPLEConfig(
    relaxation_factor_U=0.7,
    relaxation_factor_p=0.3,
    n_correctors=1,
)

solver = SIMPLESolver(mesh, config)
U, p, phi, convergence = solver.solve(
    U, p, phi,
    max_outer_iterations=100,
    tolerance=1e-4,
)
```

#### `PISOSolver`

瞬态不可压缩流的 PISO 算法。

```python
from pyfoam.solvers.piso import PISOSolver, PISOConfig

config = PISOConfig(n_correctors=2)
solver = PISOSolver(mesh, config)
U, p, phi, convergence = solver.solve(U, p, phi, U_old=U_old, p_old=p_old)
```

#### `PIMPLESolver`

PIMPLE 算法（PISO + SIMPLE 组合，含外迭代）。

```python
from pyfoam.solvers.pimple import PIMPLESolver, PIMPLEConfig

config = PIMPLEConfig(n_outer_correctors=3, n_correctors=1)
solver = PIMPLESolver(mesh, config)
U, p, phi, convergence = solver.solve(U, p, phi, U_old=U_old, p_old=p_old)
```

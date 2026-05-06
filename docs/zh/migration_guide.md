# 迁移指南：从 OpenFOAM 到 pyOpenFOAM

本指南将 OpenFOAM 的概念和语法映射到 pyOpenFOAM 的等效实现。面向已熟悉 OpenFOAM 并希望使用 pyOpenFOAM 进行 Python CFD 工作流的工程师和研究人员。

## 概念映射

### 算例结构

| OpenFOAM | pyOpenFOAM | 说明 |
|----------|------------|------|
| 算例目录 | `Case("path/to/case")` | 读取所有配置文件和网格 |
| `system/controlDict` | `case.controlDict` | 返回 `FoamDict`（类字典） |
| `system/fvSchemes` | `case.fvSchemes` | 解析后的字典 |
| `system/fvSolution` | `case.fvSolution` | 解析后的字典 |
| `constant/polyMesh/` | `case.mesh` | 返回 `MeshData` |
| 时间目录（`0/`、`1/`、...） | `case.time_dirs` | 排序后的字符串列表 |
| 场文件（`0/U`） | `case.read_field("U", time=0)` | 返回 `FieldData` |

**OpenFOAM：**
```bash
# 检查算例
foamInfo pitzDaily
# 读取场
foamToVTK -case pitzDaily
```

**pyOpenFOAM：**
```python
from pyfoam.io.case import Case

case = Case("pitzDaily")
print(case.controlDict)
print(case.list_fields(time=0))
field = case.read_field("U", time=0)
```

### 网格

| OpenFOAM | pyOpenFOAM | 说明 |
|----------|------------|------|
| `polyMesh` | `PolyMesh` | 原始拓扑（点、面、所有者、邻居） |
| `fvMesh` | `FvMesh` | 扩展 `PolyMesh`，含几何量 |
| `points` | `mesh.points` | `(n_points, 3)` 张量 |
| `faces` | `mesh.faces` | `list[Tensor]` — 每个面的点索引 |
| `owner` | `mesh.owner` | `(n_faces,)` 张量 |
| `neighbour` | `mesh.neighbour` | `(n_internal_faces,)` 张量 |
| `boundary` | `mesh.boundary` | 面片字典列表 |
| `V()` | `mesh.cell_volumes` | `(n_cells,)` 张量 |
| `C()` | `mesh.cell_centres` | `(n_cells, 3)` 张量 |
| `Sf()` | `mesh.face_areas` | `(n_faces, 3)` 张量（面积矢量） |
| `Cf()` | `mesh.face_centres` | `(n_faces, 3)` 张量 |
| `deltaCoeffs()` | `mesh.delta_coefficients` | `(n_faces,)` 张量 |
| `weights()` | `mesh.face_weights` | `(n_faces,)` 张量 |

**OpenFOAM（C++）：**
```cpp
const fvMesh& mesh = runTime.mesh();
const volVectorField& C = mesh.C();
const surfaceScalarField& weights = mesh.weights();
```

**pyOpenFOAM：**
```python
from pyfoam.mesh import FvMesh

mesh = FvMesh(points, faces, owner, neighbour, boundary)
cell_centres = mesh.cell_centres      # 惰性计算
face_weights = mesh.face_weights      # 惰性计算
mesh.compute_geometry()               # 预计算所有量
```

### 场

| OpenFOAM | pyOpenFOAM | 形状 |
|----------|------------|------|
| `volScalarField p` | `volScalarField(mesh, "p")` | `(n_cells,)` |
| `volVectorField U` | `volVectorField(mesh, "U")` | `(n_cells, 3)` |
| `volTensorField tau` | `volTensorField(mesh, "tau")` | `(n_cells, 3, 3)` |
| `surfaceScalarField phi` | 原始 `torch.Tensor` | `(n_faces,)` |
| `p.internalField()` | `p.internal_field` | 直接张量访问 |
| `p.boundaryField()` | `p.boundary_field` | `BoundaryField` 对象 |
| `p = pOld + alpha * pPrime` | `p.assign(p_old + alpha * p_prime)` | 原地更新 |

**OpenFOAM（C++）：**
```cpp
volScalarField p
(
    IOobject("p", runTime.timeName(), mesh, IOobject::MUST_READ),
    mesh
);
p = dimensionedScalar("p", dimPressure, 0);
```

**pyOpenFOAM：**
```python
from pyfoam.fields import volScalarField
import torch

p = volScalarField(mesh, "p")
p.assign(torch.zeros(mesh.n_cells))
```

### 场算术

| OpenFOAM | pyOpenFOAM | 说明 |
|----------|------------|------|
| `p1 + p2` | `p1 + p2` | 量纲必须匹配 |
| `p * 2.0` | `p * 2.0` | 场必须无量纲 |
| `U & U` | `torch.sum(U * U, dim=1)` | 内积（手动） |
| `fvc::grad(p)` | 参见离散模块 | 梯度计算 |
| `fvc::div(phi)` | 参见离散模块 | 散度计算 |
| `fvm::laplacian(D, p)` | 参见离散模块 | 隐式扩散 |

### 边界条件

| OpenFOAM `type` | pyOpenFOAM 类 | 注册名 |
|-----------------|---------------|--------|
| `fixedValue` | `FixedValueBC` | `"fixedValue"` |
| `zeroGradient` | `ZeroGradientBC` | `"zeroGradient"` |
| `fixedGradient` | `FixedGradientBC` | `"fixedGradient"` |
| `noSlip` | `NoSlipBC` | `"noSlip"` |
| `cyclic` | `CyclicBC` | `"cyclic"` |
| `symmetryPlane` | `SymmetryBC` | `"symmetryPlane"` |
| `inletOutlet` | `InletOutletBC` | `"inletOutlet"` |
| `nutkWallFunction` | `NutkWallFunctionBC` | `"nutkWallFunction"` |
| `kqRWallFunction` | `KqRWallFunctionBC` | `"kqRWallFunction"` |

**OpenFOAM（字典）：**
```
inlet
{
    type            fixedValue;
    value           uniform (1 0 0);
}
```

**pyOpenFOAM：**
```python
from pyfoam.boundary import BoundaryCondition, Patch

patch = Patch(
    name="inlet",
    face_indices=torch.tensor([0, 1, 2]),
    face_normals=torch.tensor([[-1, 0, 0.0]] * 3),
    face_areas=torch.tensor([0.01, 0.01, 0.01]),
    delta_coeffs=torch.tensor([100.0, 100.0, 100.0]),
    owner_cells=torch.tensor([0, 1, 2]),
)

inlet_bc = BoundaryCondition.create(
    "fixedValue", patch, coeffs={"value": [1.0, 0.0, 0.0]}
)
```

### 线性求解器

| OpenFOAM `solver` | pyOpenFOAM 类 | 用途 |
|-------------------|---------------|------|
| `PCG` | `PCGSolver` | 对称正定（压力） |
| `PBiCG` / `PBiCGStab` | `PBiCGSTABSolver` | 一般非对称（动量） |
| `GAMG` | `GAMGSolver` | 代数多重网格（任意矩阵） |
| `smoothSolver` | 尚未实现 | — |
| `DIC` 预处理器 | `DICPreconditioner` | 用于 PCG |
| `DILU` 预处理器 | `DILUPreconditioner` | 用于 PBiCGStab |
| `FDIC` 预处理器 | 尚未实现 | — |

**OpenFOAM（`fvSolution`）：**
```
solvers
{
    p
    {
        solver          PCG;
        preconditioner  DIC;
        tolerance       1e-6;
        relTol          0.01;
    }
    U
    {
        solver          PBiCGStab;
        preconditioner  DILU;
        tolerance       1e-6;
        relTol          0.01;
    }
}
```

**pyOpenFOAM：**
```python
from pyfoam.solvers.pcg import PCGSolver
from pyfoam.solvers.pbicgstab import PBiCGSTABSolver

p_solver = PCGSolver(tolerance=1e-6, rel_tol=0.01, preconditioner="DIC")
U_solver = PBiCGSTABSolver(tolerance=1e-6, rel_tol=0.01, preconditioner="DILU")
```

### 耦合求解器

| OpenFOAM 应用 | pyOpenFOAM 类 | 算法 |
|--------------|---------------|------|
| `simpleFoam` | `SIMPLESolver` | 定常不可压缩流 |
| `pisoFoam` | `PISOSolver` | 瞬态不可压缩流 |
| `pimpleFoam` | `PIMPLESolver` | 含外迭代的瞬态流 |

**OpenFOAM（`fvSolution`）：**
```
SIMPLE
{
    nNonOrthogonalCorrectors 0;
    pRefCell 0;
    pRefValue 0;
}

relaxationFactors
{
    fields
    {
        p 0.3;
    }
    equations
    {
        U 0.7;
    }
}
```

**pyOpenFOAM：**
```python
from pyfoam.solvers.simple import SIMPLESolver, SIMPLEConfig

config = SIMPLEConfig(
    n_correctors=1,
    relaxation_factor_p=0.3,
    relaxation_factor_U=0.7,
    p_solver="PCG",
    U_solver="PBiCGStab",
)

solver = SIMPLESolver(mesh, config)
U, p, phi, convergence = solver.solve(U, p, phi, max_outer_iterations=100)
```

### 离散格式

| OpenFOAM `scheme` | pyOpenFOAM 类 |
|-------------------|---------------|
| `linear` | `LinearInterpolation` |
| `upwind` | `UpwindInterpolation` |
| `linearUpwind` | `LinearUpwindInterpolation` |
| `QUICK` | `QuickInterpolation` |

**OpenFOAM（`fvSchemes`）：**
```
divSchemes
{
    default         none;
    div(phi,U)      Gauss upwind;
    div(phi,k)      Gauss upwind;
    div(phi,epsilon) Gauss upwind;
}
```

**pyOpenFOAM：**
```python
from pyfoam.discretisation import (
    LinearInterpolation,
    UpwindInterpolation,
    LinearUpwindInterpolation,
    QuickInterpolation,
)

# 创建插值格式
upwind = UpwindInterpolation()
face_values = upwind.interpolate(cell_values, face_flux, mesh)
```

## 工作流对比

### OpenFOAM 工作流

```bash
# 1. 创建算例目录结构
mkdir -p pitzDaily/{0,constant/polyMesh,system}

# 2. 编辑网格文件（blockMeshDict 或 snappyHexMeshDict）
vim system/blockMeshDict
blockMesh -case pitzDaily

# 3. 编辑边界条件
vim 0/U
vim 0/p

# 4. 编辑求解器设置
vim system/fvSolution
vim system/fvSchemes

# 5. 运行求解器
simpleFoam -case pitzDaily

# 6. 后处理
paraFoam -case pitzDaily
```

### pyOpenFOAM 工作流

```python
# 1. 加载现有算例（或程序化创建）
from pyfoam.io.case import Case
case = Case("pitzDaily")

# 2. 从 OpenFOAM 文件创建网格
from pyfoam.mesh import FvMesh
mesh_data = case.mesh
mesh = FvMesh(
    points=mesh_data.points,
    faces=mesh_data.faces,
    owner=mesh_data.owner,
    neighbour=mesh_data.neighbour,
    boundary=mesh_data.boundary,
)

# 3. 创建场
from pyfoam.fields import volScalarField, volVectorField
import torch

U = volVectorField(mesh, "U")
p = volScalarField(mesh, "p")
U.assign(torch.zeros(mesh.n_cells, 3))
p.assign(torch.zeros(mesh.n_cells))

# 4. 配置并运行求解器
from pyfoam.solvers.simple import SIMPLESolver, SIMPLEConfig

config = SIMPLEConfig(relaxation_factor_U=0.7, relaxation_factor_p=0.3)
solver = SIMPLESolver(mesh, config)

phi = torch.zeros(mesh.n_faces)
U_result, p_result, phi_result, conv = solver.solve(
    U.internal_field, p.internal_field, phi,
    max_outer_iterations=100,
)

# 5. 访问结果
print(f"收敛：{conv.converged}")
print(f"最大速度：{U_result.abs().max():.4f}")
```

## 关键差异

### 1. 无时间循环（目前）

OpenFOAM 的 `Time` 类管理时间循环。在 pyOpenFOAM 中，你需要自己管理时间循环：

```python
# OpenFOAM：自动
while runTime.loop():
    # 求解一个时间步
    ...

# pyOpenFOAM：显式
for t in range(n_timesteps):
    U, p, phi, conv = solver.solve(U, p, phi, ...)
    # 需要时手动保存结果
```

### 2. 求解期间无文件 I/O

OpenFOAM 在每个时间步将场写入磁盘。pyOpenFOAM 将所有内容保留在内存中（GPU/CPU 上的张量）。需要时显式写入结果。

### 3. 张量操作代替场操作

OpenFOAM 使用场类上的重载运算符。pyOpenFOAM 使用相同的运算符，但它们产生新的张量或场对象：

```python
# OpenFOAM：场操作修改内部存储
p = p + dp;

# pyOpenFOAM：使用 assign() 原地更新
p.assign(p.internal_field + dp)
```

### 4. 边界条件是显式的

在 OpenFOAM 中，边界条件在场操作期间自动应用。在 pyOpenFOAM 中，你需要显式应用它们：

```python
# 应用所有边界条件
for bc in p.boundary_field:
    bc.apply(p.internal_field)
```

## 故障排除

### "Device 'cuda' is not available"

```python
from pyfoam.core import DeviceManager
dm = DeviceManager()
print(dm.capabilities)  # 检查可用设备
# 如果 CUDA 不可用则使用 CPU
dm.device = 'cpu'
```

### "Field must have a floating-point dtype"

CFD 操作需要 float64 张量：

```python
import torch
from pyfoam.core import get_default_dtype

# 使用正确数据类型创建张量
values = torch.zeros(n_cells, dtype=get_default_dtype())  # float64
```

### SIMPLE/PISO 发散

如果求解器发散：

1. **检查精度**：确保使用 float64（不是 float32）。
2. **降低松弛**：降低 `relaxation_factor_U`（尝试 0.3）和 `relaxation_factor_p`（尝试 0.1）。
3. **检查网格质量**：非正交网格需要非正交修正。
4. **检查边界条件**：确保入口/出口 BC 在物理上一致。

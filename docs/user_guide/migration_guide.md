# OpenFOAM 到 pyOpenFOAM 迁移指南

本指南帮助 OpenFOAM 用户将已有案例移植到 pyOpenFOAM，涵盖案例结构映射、逐步移植流程、关键差异以及常见问题。

---

## 快速对照表

### 案例结构

| OpenFOAM | pyOpenFOAM | 说明 |
|-----------|-----------|------|
| `case/` 目录 | `Case("path/to/case")` | 直接读取标准 OF 目录 |
| `system/controlDict` | `case.controlDict` | `FoamDict` 对象 |
| `system/fvSchemes` | `case.fvSchemes` | 解析后的字典 |
| `system/fvSolution` | `case.fvSolution` | 解析后的字典 |
| `constant/polyMesh/` | `case.mesh` | `MeshData` 对象 |
| 时间目录 (`0/`, `1/`, ...) | `case.time_dirs` | 排序后的字符串列表 |
| 场文件 (`0/U`) | `case.read_field("U", time=0)` | 返回 `FieldData` |

### 网格对象

| OpenFOAM (C++) | pyOpenFOAM | 形状 |
|----------------|-----------|------|
| `mesh.C()` | `mesh.cell_centres` | `(n_cells, 3)` |
| `mesh.V()` | `mesh.cell_volumes` | `(n_cells,)` |
| `mesh.Sf()` | `mesh.face_areas` | `(n_faces, 3)` |
| `mesh.Cf()` | `mesh.face_centres` | `(n_faces, 3)` |
| `mesh.deltaCoeffs()` | `mesh.delta_coefficients` | `(n_faces,)` |
| `mesh.weights()` | `mesh.face_weights` | `(n_faces,)` |
| `mesh.owner()` | `mesh.owner` | `(n_faces,)` |
| `mesh.neighbour()` | `mesh.neighbour` | `(n_internal_faces,)` |

### 场对象

| OpenFOAM (C++) | pyOpenFOAM | 形状 |
|----------------|-----------|------|
| `volScalarField p` | `p` (tensor) | `(n_cells,)` |
| `volVectorField U` | `U` (tensor) | `(n_cells, 3)` |
| `surfaceScalarField phi` | `phi` (tensor) | `(n_faces,)` |

### 求解器

| OpenFOAM | pyOpenFOAM | 算法 |
|----------|-----------|------|
| `icoFoam` | `IcoFoam` | PISO 瞬态不可压缩层流 |
| `pisoFoam` | `PisoFoam` | PISO 瞬态不可压缩 |
| `pimpleFoam` | `PimpleFoam` | PIMPLE 瞬态不可压缩 |
| `simpleFoam` | `SimpleFoam` | SIMPLE 稳态不可压缩 |
| `rhoSimpleFoam` | `RhoSimpleFoam` | SIMPLE 稳态可压缩 |
| `buoyantSimpleFoam` | `BuoyantSimpleFoam` | SIMPLE 浮力可压缩 |
| `rhoCentralFoam` | `RhoCentralFoam` | Kurganov-Tadmor 密度基 |
| `interFoam` | `InterFoam` | VOF 两相不可压缩 |
| `reactingFoam` | `ReactingFoam` | 反应流 (Arrhenius) |
| `sonicFoam` | `SonicFoam` | 瞬态可压缩声速 |

### 边界条件

| OpenFOAM `type` | pyOpenFOAM 类 | 注册名 |
|-----------------|--------------|--------|
| `fixedValue` | `FixedValueBC` | `"fixedValue"` |
| `zeroGradient` | `ZeroGradientBC` | `"zeroGradient"` |
| `noSlip` | `NoSlipBC` | `"noSlip"` |
| `fixedGradient` | `FixedGradientBC` | `"fixedGradient"` |
| `cyclic` | `CyclicBC` | `"cyclic"` |
| `symmetryPlane` | `SymmetryBC` | `"symmetryPlane"` |
| `empty` | `EmptyBC` | `"empty"` |
| `inletOutlet` | `InletOutletBC` | `"inletOutlet"` |
| `nutkWallFunction` | `NutkWallFunctionBC` | `"nutkWallFunction"` |
| `totalPressure` | `TotalPressureBC` | `"totalPressure"` |
| `pressureInletOutletVelocity` | `PressureInletOutletVelocityBC` | `"pressureInletOutletVelocity"` |
| `flowRateInletVelocity` | `FlowRateInletVelocityBC` | `"flowRateInletVelocity"` |
| `advective` | `AdvectiveBC` | `"advective"` |
| `buoyantPressure` | `BuoyantPressureBC` | `"buoyantPressure"` |

### 线性求解器

| OpenFOAM | pyOpenFOAM | 用途 |
|----------|-----------|------|
| `PCG` | `PCGSolver` | 对称正定 (压力) |
| `PBiCGStab` | `PBiCGSTABSolver` | 非对称 (动量) |
| `GAMG` | `GAMGSolver` | 代数多重网格 |

### 离散格式

| OpenFOAM | pyOpenFOAM |
|----------|-----------|
| `Gauss linear` | `LinearInterpolation` |
| `Gauss upwind` | `UpwindInterpolation` |
| `Gauss linearUpwind` | `LinearUpwindInterpolation` |
| `Gauss QUICK` | `QuickInterpolation` |

---

## 逐步移植流程

### 第 1 步：直接加载已有案例

pyOpenFOAM 可以直接读取标准 OpenFOAM 目录结构，无需修改文件格式：

```python
from pyfoam.io.case import Case

# 直接加载现有 OpenFOAM 案例
case = Case("path/to/openfoam/case")

# 检查内容
print("应用:", case.get_application())
print("场列表:", case.list_fields(time=0))
print("网格:", "已加载" if case.has_mesh() else "未找到")
```

**要点**：OpenFOAM 的 ASCII 格式场文件和网格文件可直接读取。二进制格式需先用 `foamToASCII` 转换。

### 第 2 步：检查网格

```python
from pyfoam.mesh.fv_mesh import FvMesh

mesh_data = case.mesh
mesh = FvMesh(
    points=mesh_data.points,
    faces=mesh_data.faces,
    owner=mesh_data.owner,
    neighbour=mesh_data.neighbour,
    boundary=mesh_data.boundary,
)

print(f"单元数: {mesh.n_cells}")
print(f"内部面: {mesh.n_internal_faces}")
print(f"边界: {[p['name'] for p in mesh.boundary]}")
```

### 第 3 步：读取初始场

```python
import torch

# 读取 OpenFOAM 格式的场文件
U_data = case.read_field("U", time=0)
p_data = case.read_field("p", time=0)

# 转换为 PyTorch 张量
U = torch.tensor(U_data.internal, dtype=torch.float64)  # (n_cells, 3)
p = torch.tensor(p_data.internal, dtype=torch.float64)  # (n_cells,)
```

### 第 4 步：配置并运行求解器

```python
from pyfoam.applications import IcoFoam

# 方式一：直接使用应用级求解器（推荐）
solver = IcoFoam("path/to/case")
result = solver.run()
print(f"收敛: {result['converged']}")
```

或使用底层求解器 API：

```python
from pyfoam.solvers.piso import PISOSolver, PISOConfig

config = PISOConfig(n_correctors=2)
solver = PISOSolver(mesh, config)
phi = torch.zeros(mesh.n_faces, dtype=torch.float64)
U_out, p_out, phi_out, conv = solver.solve(
    U, p, phi, U_old=U.clone(), p_old=p.clone(),
)
```

### 第 5 步：后处理

```python
# 结果是 PyTorch 张量，可直接用 numpy 转换
import numpy as np

U_np = U_out.detach().numpy()  # (n_cells, 3)
p_np = p_out.detach().numpy()  # (n_cells,)

# 速度幅值
speed = np.linalg.norm(U_np, axis=1)
print(f"最大速度: {speed.max():.4f}")
```

---

## 关键差异

### 1. 时间循环管理

**OpenFOAM**：自动管理 `Time` 对象。

```cpp
while (runTime.loop())
{
    #include "UEqn.H"
    #include "pEqn.H"
    runTime.write();
}
```

**pyOpenFOAM**：应用级求解器内部管理，底层 API 需手动控制。

```python
# 应用级：自动管理
solver = PimpleFoam("case")
solver.run()

# 底层 API：手动循环
from pyfoam.applications.time_loop import TimeLoop
for t, step in TimeLoop(start_time=0, end_time=1, delta_t=0.001):
    U, p, phi, conv = solver.solve(U, p, phi, ...)
```

### 2. 内存 vs 磁盘

**OpenFOAM**：每步写磁盘（`runTime.write()`）。

**pyOpenFOAM**：全程在内存（CPU/GPU 张量），需要时手动写出。

```python
solver._write_fields(time=0.5)  # 手动写场文件
```

### 3. 精度

**OpenFOAM**：默认 `double`。

**pyOpenFOAM**：默认 `float64`（CFD 必须），但支持 GPU `float32` 加速（需验证精度）。

```python
from pyfoam.core.device import get_default_dtype
print(get_default_dtype())  # torch.float64
```

### 4. 边界条件

**OpenFOAM**：场操作时自动应用 BC。

**pyOpenFOAM**：显式应用。

```python
from pyfoam.boundary import BoundaryCondition

bc = BoundaryCondition.create("fixedValue", patch, coeffs={"value": [1.0, 0.0, 0.0]})
bc.apply(U)
```

### 5. 维度检查

**OpenFOAM**：运行时检查物理量纲。

**pyOpenFOAM**：不检查量纲（张量是纯数值）。需自行保证物理一致性。

### 6. 并行计算

**OpenFOAM**：MPI 分区 + `decomposePar`。

**pyOpenFOAM**：暂不支持分布式 MPI，但支持 GPU 加速（`CUDA`、`ROCm`）。

---

## 常见问题

### 二进制场文件无法读取

```bash
# 先转换为 ASCII
foamToASCII -case path/to/case
```

### 求解器发散

1. 确认使用 `float64`（非 `float32`）
2. 降低松弛因子：`relaxation_factor_U=0.3`, `relaxation_factor_p=0.1`
3. 检查网格质量
4. 检查边界条件物理一致性

### `blockMeshDict` 无法直接使用

pyOpenFOAM 不自带网格生成器。需先用 OpenFOAM 的 `blockMesh` 生成网格，然后加载案例。

```bash
blockMesh -case path/to/case
# 然后在 Python 中加载
solver = SolverName("path/to/case")
```

### GPU 加速

```python
from pyfoam.core.device import set_device

# 使用 GPU
set_device("cuda:0")

# 回退到 CPU
set_device("cpu")
```

---

## 详细 API 参考

参见 [英文迁移指南](../en/migration_guide.md) 获取完整的 API 映射表和代码示例。

# 迁移指南：OpenFOAM → pyOpenFOAM

面向已熟悉 OpenFOAM 并希望使用 pyOpenFOAM 进行 Python CFD 工作流的工程师和研究人员。

---

## 1. 算例结构对照

| OpenFOAM | pyOpenFOAM | 说明 |
|----------|------------|------|
| 算例目录 | `Case("path/to/case")` | 读取所有配置和网格 |
| `system/controlDict` | `case.controlDict` | 返回 `FoamDict` |
| `system/fvSchemes` | `case.fvSchemes` | 解析后的字典 |
| `system/fvSolution` | `case.fvSolution` | 解析后的字典 |
| `constant/polyMesh/` | `case.mesh` | 返回 `MeshData` |
| 时间目录 `0/`、`1/`... | `case.time_dirs` | 排序字符串列表 |
| 场文件 `0/U` | `case.read_field("U", time=0)` | 返回 `FieldData` |

```python
from pyfoam.io.case import Case

case = Case("pitzDaily")
print(case.controlDict["application"])
print(case.list_fields(time=0))
```

---

## 2. 网格对照

| OpenFOAM | pyOpenFOAM | 形状 |
|----------|------------|------|
| `polyMesh` | `PolyMesh` | 原始拓扑 |
| `fvMesh` | `FvMesh` | 含几何量 |
| `points` | `mesh.points` | `(n_points, 3)` |
| `faces` | `mesh.faces` | `list[Tensor]` |
| `owner` | `mesh.owner` | `(n_faces,)` |
| `neighbour` | `mesh.neighbour` | `(n_internal_faces,)` |
| `V()` | `mesh.cell_volumes` | `(n_cells,)` |
| `C()` | `mesh.cell_centres` | `(n_cells, 3)` |
| `Sf()` | `mesh.face_areas` | `(n_faces, 3)` |
| `Cf()` | `mesh.face_centres` | `(n_faces, 3)` |
| `deltaCoeffs()` | `mesh.delta_coefficients` | `(n_faces,)` |
| `weights()` | `mesh.face_weights` | `(n_faces,)` |

```python
from pyfoam.mesh import FvMesh

mesh = FvMesh(points, faces, owner, neighbour, boundary)
mesh.compute_geometry()  # 预计算所有几何量
```

---

## 3. 场对照

| OpenFOAM | pyOpenFOAM | 形状 |
|----------|------------|------|
| `volScalarField p` | `volScalarField(mesh, "p")` | `(n_cells,)` |
| `volVectorField U` | `volVectorField(mesh, "U")` | `(n_cells, 3)` |
| `volTensorField tau` | `volTensorField(mesh, "tau")` | `(n_cells, 3, 3)` |
| `surfaceScalarField phi` | 原始 `torch.Tensor` | `(n_faces,)` |
| `p.internalField()` | `p.internal_field` | 直接访问 |
| `p.boundaryField()` | `p.boundary_field` | `BoundaryField` 对象 |
| 赋值 `p = ...` | `p.assign(...)` | 原地更新 |

```python
from pyfoam.fields import volScalarField, volVectorField
import torch

p = volScalarField(mesh, "p")
p.assign(torch.zeros(mesh.n_cells))

U = volVectorField(mesh, "U")
U.assign(torch.zeros(mesh.n_cells, 3))
```

---

## 4. 求解器名称映射

| OpenFOAM 求解器 | pyOpenFOAM 类 | import 路径 |
|-----------------|---------------|-------------|
| `simpleFoam` | `SimpleFoam` | `pyfoam.applications` |
| `icoFoam` | `IcoFoam` | `pyfoam.applications` |
| `pisoFoam` | `PisoFoam` | `pyfoam.applications` |
| `pimpleFoam` | `PimpleFoam` | `pyfoam.applications` |
| `rhoSimpleFoam` | `RhoSimpleFoam` | `pyfoam.applications` |
| `rhoPimpleFoam` | `RhoPimpleFoam` | `pyfoam.applications` |
| `sonicFoam` | `SonicFoam` | `pyfoam.applications` |
| `rhoCentralFoam` | `RhoCentralFoam` | `pyfoam.applications` |
| `buoyantSimpleFoam` | `BuoyantSimpleFoam` | `pyfoam.applications` |
| `buoyantPimpleFoam` | `BuoyantPimpleFoam` | `pyfoam.applications` |
| `buoyantBoussinesqSimpleFoam` | `BuoyantBoussinesqSimpleFoam` | `pyfoam.applications` |
| `laplacianFoam` | `LaplacianFoam` | `pyfoam.applications` |
| `interFoam` | `InterFoam` | `pyfoam.applications` |
| `multiphaseInterFoam` | `MultiphaseInterFoam` | `pyfoam.applications` |
| `twoPhaseEulerFoam` | `TwoPhaseEulerFoam` | `pyfoam.applications` |
| `multiphaseEulerFoam` | `MultiphaseEulerFoam` | `pyfoam.applications` |
| `cavitatingFoam` | `CavitatingFoam` | `pyfoam.applications` |
| `potentialFoam` | `PotentialFoam` | `pyfoam.applications` |
| `scalarTransportFoam` | `ScalarTransportFoam` | `pyfoam.applications` |
| `reactingFoam` | `ReactingFoam` | `pyfoam.applications` |
| `solidDisplacementFoam` | `SolidDisplacementFoam` | `pyfoam.applications` |
| `SRFSimpleFoam` | `SRFSimpleFoam` | `pyfoam.applications` |
| `porousSimpleFoam` | `PorousSimpleFoam` | `pyfoam.applications` |
| `boundaryFoam` | `BoundaryFoam` | `pyfoam.applications` |
| `chtMultiRegionFoam` | `ChtMultiRegionFoam` | `pyfoam.applications` |
| `compressibleInterFoam` | `CompressibleInterFoam` | `pyfoam.applications` |

使用模式：

```python
from pyfoam.applications import SimpleFoam

solver = SimpleFoam("path/to/case")
solver.run()
```

---

## 5. 边界条件名称映射

| OpenFOAM `type` | pyOpenFOAM 类 | 注册名 |
|-----------------|---------------|--------|
| `fixedValue` | `FixedValueBC` | `"fixedValue"` |
| `zeroGradient` | `ZeroGradientBC` | `"zeroGradient"` |
| `fixedGradient` | `FixedGradientBC` | `"fixedGradient"` |
| `calculated` | `CalculatedBC` | `"calculated"` |
| `noSlip` | `NoSlipBC` | `"noSlip"` |
| `cyclic` | `CyclicBC` | `"cyclic"` |
| `cyclicAMI` | `CyclicAMIBC` | `"cyclicAMI"` |
| `symmetryPlane` | `SymmetryBC` | `"symmetryPlane"` |
| `empty` | `EmptyBC` | `"empty"` |
| `inletOutlet` | `InletOutletBC` | `"inletOutlet"` |
| `totalPressure` | `TotalPressureBC` | `"totalPressure"` |
| `fixedFluxPressure` | `FixedFluxPressureBC` | `"fixedFluxPressure"` |
| `advective` | `AdvectiveBC` | `"advective"` |
| `codedFixedValue` | `CodedFixedValueBC` | `"codedFixedValue"` |
| `buoyantPressure` | `BuoyantPressureBC` | `"buoyantPressure"` |
| `nutkWallFunction` | `NutkWallFunctionBC` | `"nutkWallFunction"` |
| `kqRWallFunction` | `KqRWallFunctionBC` | `"kqRWallFunction"` |

OpenFOAM 字典格式：
```
inlet
{
    type            fixedValue;
    value           uniform (1 0 0);
}
```

pyOpenFOAM Python API：
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

bc = BoundaryCondition.create("fixedValue", patch, coeffs={"value": 1.0})
bc.apply(velocity_field, patch_idx=0)
```

---

## 6. 离散格式对照

### 插值格式

| OpenFOAM fvSchemes | pyOpenFOAM 类 |
|--------------------|---------------|
| `linear` | `LinearInterpolation` |
| `upwind` | `UpwindInterpolation` |
| `linearUpwind` | `LinearUpwindInterpolation` |
| `QUICK` | `QuickInterpolation` |
| `harmonic` | `HarmonicInterpolation` |
| `LUST` | `LUSTInterpolation` |
| `vanLeer` | `VanLeerInterpolation` |
| `Gamma` | `GammaInterpolation` |

### 梯度格式

| OpenFOAM | pyOpenFOAM |
|----------|------------|
| `Gauss linear` | `GaussLinearGrad` |
| `leastSquares` | `LeastSquaresGrad` |

### 时间导数

| OpenFOAM | pyOpenFOAM |
|----------|------------|
| `Euler` | `EulerDdt` |
| `steadyState` | `SteadyStateDdt` |
| `CrankNicolson` | `CrankNicolsonDdt` |

---

## 7. 湍流模型对照

| OpenFOAM RAS 模型 | pyOpenFOAM 注册名 |
|-------------------|-------------------|
| `kEpsilon` | `"kEpsilon"` |
| `realizableKE` | `"realizableKE"` |
| `RNGkEpsilon` | `"RNGkEpsilon"` |
| `kOmega` | `"kOmega"` |
| `kOmegaSST` | `"kOmegaSST"` |
| `kOmegaSSTLM` | `"kOmegaSSTLM"` |
| `SpalartAllmaras` | `"SpalartAllmaras"` |
| `v2f` | `"v2f"` |
| `LRR` | `"LRR"` |
| `SSG` | `"SSG"` |

| OpenFOAM LES 模型 | pyOpenFOAM 注册名 |
|--------------------|-------------------|
| `Smagorinsky` | `"Smagorinsky"` |
| `WALE` | `"WALE"` |
| `dynamicSmagorinsky` | `"dynamicSmagorinsky"` |
| `dynamicLagrangian` | `"dynamicLagrangian"` |
| `kEqn` | `"kEqn"` |

```python
from pyfoam.turbulence import TurbulenceModel

model = TurbulenceModel.create("kOmegaSST", mesh, U, phi)
model.correct()
nut = model.nut()
```

---

## 8. 配置文件格式差异

### controlDict

OpenFOAM：
```
application     simpleFoam;
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         1000;
deltaT          1;
writeControl    timeStep;
writeInterval   100;
```

pyOpenFOAM 读取同一文件，返回 Python 字典：
```python
case = Case("path/to/case")
app = case.controlDict["application"]    # "simpleFoam"
dt = float(case.controlDict["deltaT"])   # 1.0
```

### fvSolution

求解器配置保持兼容，pyOpenFOAM 解析为嵌套字典：
```python
config = case.fvSolution
p_solver = config["solvers"]["p"]["solver"]   # "PCG"
p_tol = float(config["solvers"]["p"]["tolerance"])  # 1e-6
```

---

## 9. 典型迁移示例

### OpenFOAM 端

```bash
# 原始 OpenFOAM 工作流
cd pitzDaily
blockMesh
simpleFoam
foamToVTK
```

### pyOpenFOAM 端

```python
# 完整 Python 工作流
from pyfoam.applications import SimpleFoam
from pyfoam.postprocessing import FoamToVTK

# 运行求解器
solver = SimpleFoam("pitzDaily")
solver.run()

# 导出 VTK
FoamToVTK("pitzDaily").export()

# Python 后处理
from pyfoam.io.case import Case
case = Case("pitzDaily")
U = case.read_field("U", time=case.time_dirs[-1])
print(f"最大速度: {U.data.max()}")
```

---

## 相关文档

| 文档 | 路径 |
|------|------|
| API 索引 | `docs/api/README.md` |
| 入门指南 | `docs/user_guide/getting_started.md` |
| 模块详细 API | `docs/api/modules.md` |
| GPU 指南 | `docs/zh/gpu_guide.md` |

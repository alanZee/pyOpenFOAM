# pyOpenFOAM API 文档

本文档是 pyOpenFOAM 全部 24 个模块的 API 索引。

## 模块总览

| 模块 | 类数量 | 文件数 | 功能概述 |
|------|--------|--------|----------|
| `pyfoam.core` | 11 | 8 | 设备管理、LDU/FvMatrix 稀疏矩阵、多 GPU |
| `pyfoam.io` | 32 | 15 | OpenFOAM 字典/场/网格 I/O、VTK/Gmsh/Fluent 转换 |
| `pyfoam.discretisation` | 114 | 66 | 面插值、梯度、法向梯度、时间导数离散格式 |
| `pyfoam.boundary` | 346 | 330 | 30+ 边界条件（速度、压力、湍流、热、VOF） |
| `pyfoam.turbulence` | 238 | 107 | RANS/LES/DES 湍流模型 + 壁面函数 |
| `pyfoam.multiphase` | 203 | 91 | VOF/MULES、相间力、空化、Euler-Euler |
| `pyfoam.thermophysical` | 120 | 79 | 状态方程、输运模型、JANAF 热力学 |
| `pyfoam.applications` | 265 | 215 | 35+ 完整求解器应用 |
| `pyfoam.tools` | 667 | 296 | checkMesh、setFields、renumberMesh 等工具 |
| `pyfoam.lagrangian` | 241 | 113 | 粒子追踪、注入、碰撞、破碎、蒸发 |
| `pyfoam.waves` | 28 | 15 | Airy/Stokes/Cnoidal 波浪理论 |
| `pyfoam.fv` | 43 | 15 | fvModels 源项 + fvConstraints 约束 |
| `pyfoam.ode` | 94 | 17 | Euler/RK4/RKF45/Rosenbrock ODE 求解器 |
| `pyfoam.parallel` | 158 | 42 | MPI 域分解、幽灵单元通信、并行 I/O |
| `pyfoam.rigid_body` | 132 | 36 | 刚体运动、关节、约束 |
| `pyfoam.structural` | 101 | 34 | 结构力学位移求解器、弹性模型 |
| `pyfoam.mesh` | 20 | 11 | PolyMesh/FvMesh + blockMesh/snappyHexMesh |
| `pyfoam.models` | 9 | 4 | P1 辐射等物理模型 |
| `pyfoam.postprocessing` | 191 | 74 | FunctionObject 框架、力/力矩、y+、VTK 输出 |
| `pyfoam.case` | — | — | 算例管理（通过 `pyfoam.io.case`） |
| `pyfoam.convergence` | — | — | 收敛监控（集成于 solvers/applications） |
| `pyfoam.linear` | — | — | 线性求解器（通过 `pyfoam.solvers`） |
| `pyfoam.numerics` | — | — | 数值工具（通过 `pyfoam.discretisation`） |
| `pyfoam.compression` | — | — | 压缩工具（通过 `pyfoam.io.binary_io`） |

---

## RTS 注册模式

pyOpenFOAM 采用与 OpenFOAM 相同的 **运行时选择（RTS）** 机制。基类维护一个注册表，子类通过装饰器注册。

```python
# 注册
@BoundaryCondition.register("fixedValue")
class FixedValueBC(BoundaryCondition):
    ...

# 创建
bc = BoundaryCondition.create("fixedValue", patch, coeffs={"value": 1.0})

# 列出可用类型
print(BoundaryCondition.available_types())
```

支持 RTS 的模块：
- `boundary.BoundaryCondition` — 边界条件
- `turbulence.TurbulenceModel` — 湍流模型
- `turbulence.NonLinearViscosityModel` — 非线性粘度
- `turbulence.GeneralizedNewtonianViscosity` — 广义牛顿粘度
- `waves.WaveModel` — 波浪模型
- `fv.FvConstraint` / `fv.FvModel` — 约束与源项

---

## 核心模块详解

### 1. core — 基础设施

提供设备管理、张量配置、LDU 矩阵和稀疏运算。

| 类 | 说明 |
|----|------|
| `DeviceManager` | 单例，管理硬件检测与设备选择（CPU/CUDA/MPS） |
| `TensorConfig` | 全局张量配置（默认 float64） |
| `Backend` | 绑定到特定设备/dtype 的面向对象后端 |
| `LduMatrix` | LDU 格式稀疏矩阵 |
| `FvMatrix` | 扩展 LduMatrix，含源项和边界贡献 |
| `MultiGPUManager` | 多 GPU 网格分区与通信 |

```python
from pyfoam.core import DeviceManager, device_context, LduMatrix

dm = DeviceManager()
with device_context("cuda"):
    matrix = LduMatrix(n_cells, owner, neighbour)
    result = matrix.Ax(x)
```

### 2. io — 文件 I/O

| 类 | 说明 |
|----|------|
| `Case` | 完整算例目录表示 |
| `FoamDict` | 字典解析与嵌套访问 |
| `BinaryReader` / `BinaryWriter` | 二进制 I/O |
| `GmshMesh` / `FluentMesh` | 网格格式转换 |
| `VTKWriter` | VTK 输出 |

```python
from pyfoam.io import Case, parse_dict_file

case = Case("pitzDaily")
print(case.controlDict["application"])
config = parse_dict_file("system/fvSolution")
```

### 3. discretisation — 离散格式

| 类别 | 格式 |
|------|------|
| 插值 | Linear、Upwind、LinearUpwind、QUICK、Harmonic、LUST、VanLeer、Gamma |
| 梯度 | GaussLinear、LeastSquares |
| 法向梯度 | Uncorrected、Corrected、Limited |
| 时间导数 | Euler、SteadyState、CrankNicolson |

### 4. boundary — 边界条件

30+ 种边界条件，涵盖速度、压力、湍流、热、VOF：

| 注册名 | 类 | 说明 |
|--------|---|------|
| `fixedValue` | `FixedValueBC` | 固定值（罚函数法） |
| `zeroGradient` | `ZeroGradientBC` | 零法向梯度 |
| `noSlip` | `NoSlipBC` | 零速度（壁面） |
| `cyclic` | `CyclicBC` | 周期耦合 |
| `symmetryPlane` | `SymmetryBC` | 对称面 |
| `inletOutlet` | `InletOutletBC` | 流向切换 |
| `totalPressure` | `TotalPressureBC` | 总压 |
| `fixedFluxPressure` | `FixedFluxPressureBC` | 固定通量压力 |
| `advective` | `AdvectiveBC` | 对流型出流 |

```python
from pyfoam.boundary import BoundaryCondition

bc = BoundaryCondition.create("fixedValue", patch, coeffs={"value": 1.0})
bc.apply(field, patch_idx=0)
```

### 5. turbulence — 湍流模型

**RANS**：k-epsilon（标准/Realizable/RNG/Launder-Sharma）、k-omega（标准/SST/SST-LM/2006）、S-A、v2-f、LRR、SSG

**LES**：Smagorinsky、WALE、动态 Smagorinsky、Lagrangian 动态、k 方程、Deardorff

**DES/SAS**：k-omega SST DES/SAS、SA DES/DDES/IDDES

**增强变体**：每个基础模型有 v2-v10 增强版本，含改进的壁面处理、曲率修正、可压缩性修正等

```python
from pyfoam.turbulence import TurbulenceModel, RASModel, RASConfig

model = TurbulenceModel.create("kOmegaSST", mesh, U, phi)
model.correct()

config = RASConfig(model_name="kOmegaSST", nu=1.5e-5)
ras = RASModel(mesh, U, phi, config)
```

### 6. multiphase — 多相流

| 类 | 说明 |
|----|------|
| `VOFAdvection` | VOF 对流 + 界面压缩 |
| `MULESLimiter` | MULES 有界标量输运 |
| `SchillerNaumannDrag` / `WenYuDrag` / `GidaspowDrag` | 阻力模型 |
| `TomiyamaLift` / `VirtualMassForce` | 升力/虚拟质量力 |
| `SchnerrSauer` / `Merkle` / `ZGB` | 空化模型 |

### 7. thermophysical — 热物理

| 类别 | 模型 |
|------|------|
| 状态方程 | PerfectGas、IncompressiblePerfectGas |
| 输运 | ConstantViscosity、Sutherland、PolynomialTransport |
| 热力学 | JanafThermo、HConstThermo |
| 组合 | HePsiThermo、HeRhoThermo |

```python
from pyfoam.thermophysical import create_air_thermo

thermo = create_air_thermo()
print(thermo.Cp(300.0))
```

### 8. applications — 求解器

35+ 个完整求解器，读取 OpenFOAM 算例目录直接运行：

```python
from pyfoam.applications import SimpleFoam

solver = SimpleFoam("path/to/case")
solver.run()
```

| 类别 | 求解器 |
|------|--------|
| 不可压缩 | simpleFoam、icoFoam、pisoFoam、pimpleFoam、porousSimpleFoam |
| 可压缩 | rhoSimpleFoam、rhoPimpleFoam、sonicFoam、rhoCentralFoam |
| 浮力 | buoyantSimpleFoam、buoyantPimpleFoam、buoyantBoussinesqSimpleFoam |
| 热传导 | laplacianFoam、chtMultiRegionFoam |
| 多相 | interFoam、multiphaseInterFoam、twoPhaseEulerFoam、cavitatingFoam |

### 9. tools — 工具程序

OpenFOAM 命令行工具的 Python 实现：

```python
from pyfoam.tools import check_mesh, set_fields, renumber_mesh
```

| 工具 | 说明 |
|------|------|
| `check_mesh` | 网格质量验证 |
| `set_fields` | 基于几何区域初始化场 |
| `renumber_mesh` | Reverse Cuthill-McKee 重编号 |
| `foam_dictionary` | 查询/修改字典 |
| `foam_to_ensight` | EnSight 格式导出 |

### 10. lagrangian — 拉格朗日粒子

| 类 | 说明 |
|----|------|
| `KinematicCloud` | 含阻力/重力/壁面反弹的粒子追踪 |
| `PointInjector` / `ConeInjector` | 注入器 |
| `DragForce` / `GravityForce` / `LiftForce` | 粒子力模型 |
| 碰撞/破碎/蒸发 | 增强变体（v2-v10） |

### 11-24. 其余模块

| 模块 | 关键类 | 说明 |
|------|--------|------|
| `waves` | `AiryWave`、`StokesWave`、`CnoidalWave` | 海岸工程波浪理论 |
| `fv` | `FvConstraint`、`FvModel`、`SemiImplicitSource` | fvModels/fvConstraints |
| `ode` | `EulerSolver`、`RK4Solver`、`Rosenbrock12Solver` | ODE 时间积分 |
| `parallel` | `Decomposition`、`ParallelSolver`、`ProcessorPatch` | MPI 域分解与通信 |
| `rigid_body` | `MotionSolver`、`Joint`、`Restraint` | 刚体动力学 |
| `structural` | `DisplacementSolver`、`ElasticModel` | 结构力学 |
| `mesh` | `PolyMesh`、`FvMesh`、`BlockMesh` | 网格拓扑与几何 |
| `models` | `P1Radiation` | 辐射模型 |
| `postprocessing` | `Forces`、`YPlus`、`FieldAverage` | 后处理函数对象 |
| `differentiable` | `DifferentiableLaplacian`、`DifferentiableLinearSolve` | 可微分 CFD 算子 |

---

## 相关文档

| 文档 | 路径 |
|------|------|
| 模块详细 API | `docs/api/modules.md` |
| 入门指南 | `docs/zh/getting_started.md` |
| 迁移指南 | `docs/migration_guide.md` |
| 架构设计 | `docs/zh/architecture.md` |
| GPU 指南 | `docs/zh/gpu_guide.md` |

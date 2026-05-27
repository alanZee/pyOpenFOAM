# pyOpenFOAM 模块 API 参考

本文档涵盖 pyOpenFOAM 所有公共模块的 API 接口。

## 目录

- [core — 基础层](#core--基础层)
- [mesh — 网格表示](#mesh--网格表示)
- [fields — 场类](#fields--场类)
- [boundary — 边界条件](#boundary--边界条件)
- [discretisation — 离散格式](#discretisation--离散格式)
- [solvers — 线性与耦合求解器](#solvers--线性与耦合求解器)
- [turbulence — 湍流模型](#turbulence--湍流模型)
- [thermophysical — 热物理模型](#thermophysical--热物理模型)
- [multiphase — 多相流模型](#multiphase--多相流模型)
- [parallel — 并行计算](#parallel--并行计算)
- [applications — 应用级求解器](#applications--应用级求解器)
- [postprocessing — 后处理](#postprocessing--后处理)
- [differentiable — 可微分算子](#differentiable--可微分算子)
- [models — 物理模型](#models--物理模型)
- [ode — ODE 求解器](#ode--ode-求解器)
- [fv — 有限体积约束与源项](#fv--有限体积约束与源项)
- [lagrangian — 拉格朗日粒子](#lagrangian--拉格朗日粒子)
- [waves — 波浪模型](#waves--波浪模型)
- [tools — 工具](#tools--工具)
- [io — 文件 I/O](#io--文件-io)

---

## core — 基础层

`pyfoam.core` 提供设备管理、张量配置、LDU 矩阵和稀疏运算，是所有其他模块的基础。

### 关键类

| 类 | 说明 |
|----|------|
| `DeviceManager` | 单例，管理硬件检测与设备选择（CPU/CUDA/MPS） |
| `TensorConfig` | 全局张量配置，默认 float64，提供张量工厂方法 |
| `Backend` | 绑定到特定设备/dtype 的面向对象后端 |
| `LduMatrix` | LDU 格式稀疏矩阵，OpenFOAM 原生布局 |
| `FvMatrix` | 扩展 LduMatrix，支持源项、边界贡献和欠松弛 |
| `MultiGPUManager` | 多 GPU 网格分区与通信 |

### 关键函数

```python
from pyfoam.core import (
    get_device, get_default_dtype, device_context,  # 设备管理
    scatter_add, gather, sparse_coo_tensor, sparse_mm,  # 后端运算
    ldu_to_coo_indices, extract_diagonal, csr_matvec,  # 稀疏运算
    partition_mesh,  # 多 GPU 分区
)
```

### 使用示例

```python
from pyfoam.core import DeviceManager, device_context, LduMatrix

# 设备检测
dm = DeviceManager()
print(dm.capabilities.available_devices)

# 上下文切换
with device_context("cuda"):
    matrix = LduMatrix(n_cells, owner, neighbour)
    result = matrix.Ax(x)
```

---

## mesh — 网格表示

`pyfoam.mesh` 提供多面体网格拓扑和几何计算。

### 关键类

| 类 | 说明 |
|----|------|
| `PolyMesh` | 原始拓扑网格（点、面、owner/neighbour、边界） |
| `FvMesh` | 扩展 PolyMesh，惰性计算几何量（面心、体积等） |
| `BlockMesh` | 结构化六面体网格生成 |
| `SnappyHexMesh` | 非结构化六面体网格生成（STL 表面） |
| `STLReader` | STL 文件读取 |

### 关键函数

```python
from pyfoam.mesh import (
    compute_face_centres, compute_face_area_vectors,
    compute_cell_volumes_and_centres, compute_face_weights,
    compute_delta_coefficients,
    validate_owner_neighbour, build_cell_to_faces,
    build_face_to_cells, cell_neighbours,
)
```

### 使用示例

```python
from pyfoam.mesh import FvMesh, PolyMesh

# 从原始数据构建
mesh = PolyMesh.from_raw(points, faces, owner, neighbour, boundary)
fv = FvMesh.from_poly_mesh(mesh)
fv.compute_geometry()

# 访问几何量
print(fv.cell_centres.shape)   # (n_cells, 3)
print(fv.cell_volumes.shape)   # (n_cells,)
print(fv.face_areas.shape)     # (n_faces, 3)
```

---

## fields — 场类

`pyfoam.fields` 提供 CFD 场的层级结构，支持维度检查和算术运算。

### 关键类

| 类 | 形状 | 说明 |
|----|------|------|
| `DimensionSet` | 7 元素 | 物理维度系统 [质量, 长度, 时间, 温度, 物质量, 电流, 发光强度] |
| `volScalarField` | `(n_cells,)` | 单元中心标量场 |
| `volVectorField` | `(n_cells, 3)` | 单元中心矢量场 |
| `volTensorField` | `(n_cells, 3, 3)` | 单元中心张量场 |
| `surfaceScalarField` | `(n_faces,)` | 面心标量场 |
| `surfaceVectorField` | `(n_faces, 3)` | 面心矢量场 |

### 使用示例

```python
import torch
from pyfoam.fields import volScalarField, volVectorField, DimensionSet

p = volScalarField(mesh, "p", dimensions=DimensionSet(0, 2, -2, 0, 0, 0, 0))
p.assign(torch.zeros(mesh.n_cells))

U = volVectorField(mesh, "U", dimensions=DimensionSet(0, 1, -1, 0, 0, 0, 0))
U.assign(torch.zeros(mesh.n_cells, 3))

# 算术运算（自动维度检查）
p2 = p + p  # OK
# p + U  # 抛出 DimensionError
```

---

## boundary — 边界条件

`pyfoam.boundary` 提供 30+ 种边界条件，采用 RTS（运行时选择）注册机制。

### 关键类

| 类 | 注册名 | 说明 |
|----|--------|------|
| `BoundaryCondition` | — | 抽象基类 + RTS 注册表 |
| `Patch` | — | 边界面片描述符 |
| `BoundaryField` | — | 场的边界条件集合 |
| `FixedValueBC` | `fixedValue` | 固定值（惩罚法） |
| `ZeroGradientBC` | `zeroGradient` | 零法向梯度（Neumann） |
| `NoSlipBC` | `noSlip` | 零速度（壁面） |
| `CyclicBC` | `cyclic` | 周期性耦合 |
| `SymmetryBC` | `symmetryPlane` | 对称面 |
| `InletOutletBC` | `inletOutlet` | 流向切换 |
| `TotalPressureBC` | `totalPressure` | 总压边界 |
| `FixedFluxPressureBC` | `fixedFluxPressure` | 固定通量压力 |
| `EmptyBC` | `empty` | 2D 空边界 |
| `AdvectiveBC` | `advective` | 对流型出流 |
| `FixedEnergyBC` | `fixedEnergy` | 固定能量 |

### 使用示例

```python
from pyfoam.boundary import BoundaryCondition

# 列出所有可用类型
print(BoundaryCondition.available_types())

# RTS 创建
bc = BoundaryCondition.create("fixedValue", patch, coeffs={"value": 1.0})
bc.apply(field, patch_idx=0)
diag, source = bc.matrix_contributions(field, n_cells, diag, source)
```

---

## discretisation — 离散格式

`pyfoam.discretisation` 提供面插值、时间导数、梯度和法向梯度离散格式。

### 插值格式

| 类 | 阶数 | 说明 |
|----|------|------|
| `LinearInterpolation` | 2 | 线性距离加权插值 |
| `UpwindInterpolation` | 1 | 迎风格式 |
| `LinearUpwindInterpolation` | 2 | 迎风偏置 + 梯度修正 |
| `QuickInterpolation` | 3 | QUICK 格式（延迟修正） |
| `HarmonicInterpolation` | — | 调和平均（扩散系数） |
| `LUSTInterpolation` | — | 0.75 线性 + 0.25 线性迎风混合 |
| `VanLeerInterpolation` | — | TVD 格式（Van Leer 限制器） |
| `GammaInterpolation` | — | Peclet 数混合 |
| `InterfaceCompressionInterpolation` | — | VOF 压缩格式 |

### 时间导数格式

| 类 | 说明 |
|----|------|
| `EulerDdt` | 一阶隐式 Euler |
| `SteadyStateDdt` | 稳态（零时间导数） |
| `CrankNicolsonDdt` | 二阶 Crank-Nicolson（混合系数） |

### 梯度格式

| 类 | 说明 |
|----|------|
| `GaussLinearGrad` | Gauss 定理 + 线性面插值（默认） |
| `LeastSquaresGrad` | 最小二乘梯度重构 |

### 法向梯度格式

| 类 | 说明 |
|----|------|
| `UncorrectedSnGrad` | 简单差分（正交网格精确） |
| `CorrectedSnGrad` | 完整非正交修正 |
| `LimitedSnGrad` | 限制性非正交修正 |

---

## solvers — 线性与耦合求解器

`pyfoam.solvers` 提供线性方程组求解和压力-速度耦合算法。

### 线性求解器

| 类 | 适用场景 | 预条件 |
|----|----------|--------|
| `PCGSolver` | 对称正定矩阵 | DIC |
| `PBiCGSTABSolver` | 非对称矩阵 | DILU |
| `GAMGSolver` | 代数多重网格 | 聚合粗化 |

### 预条件器

| 类 | 说明 |
|----|------|
| `DICPreconditioner` | 对角不完全 Cholesky |
| `DILUPreconditioner` | 对角不完全 LU |

### 耦合求解器

| 类 | 说明 |
|----|------|
| `SIMPLESolver` | 稳态不可压缩 SIMPLE 算法 |
| `PISOSolver` | 瞬态不可压缩 PISO 算法 |
| `PIMPLESolver` | PISO + SIMPLE 混合（大时间步） |

### 支撑模块

| 函数 | 说明 |
|------|------|
| `compute_HbyA` | HbyA 计算 |
| `compute_face_flux_HbyA` | 面通量 HbyA |
| `rhie_chow_correction` | Rhie-Chow 修正（消除棋盘压力） |
| `assemble_pressure_equation` | 压力 Poisson 方程组装 |
| `solve_pressure_equation` | 求解压力方程 |
| `correct_velocity` / `correct_face_flux` | 速度/通量修正 |

### 使用示例

```python
from pyfoam.solvers import PCGSolver, SIMPLESolver, SIMPLEConfig

# 线性求解
solver = PCGSolver(tolerance=1e-6, max_iter=1000, preconditioner="DIC")
solution, iters, residual = solver(matrix, source, x0, 1e-6, 1000)

# 耦合求解
config = SIMPLEConfig(relaxation_factor_U=0.7, relaxation_factor_p=0.3)
simple = SIMPLESolver(mesh, config)
U, p, phi, convergence = simple.solve(U, p, phi, max_outer_iterations=100, tolerance=1e-4)
```

---

## turbulence — 湍流模型

`pyfoam.turbulence` 提供完整的 RANS/LES/DES 湍流模型库。

### RANS 模型

| 类 | 说明 |
|----|------|
| `KEpsilonModel` | 标准 k-epsilon |
| `RealizableKEpsilonModel` | Realizable k-epsilon |
| `RNGkEpsilonModel` | RNG k-epsilon |
| `LaunderSharmaKEModel` | 低 Reynolds k-epsilon |
| `KOmegaModel` | 标准 k-omega (Wilcox 2006) |
| `KOmega2006Model` | k-omega 2006（交叉扩散 + 低 Re 修正） |
| `KOmegaSSTModel` | k-omega SST (Menter 1994) |
| `KOmegaSSTLMModel` | k-omega SST Langtry-Menter 转捩模型 |
| `SpalartAllmarasModel` | S-A 单方程模型 |
| `V2FModel` | v2-f 模型 (Durbin 1995) |
| `LRRModel` | LRR Reynolds 应力模型 |
| `SSGModel` | SSG Reynolds 应力模型 |

### LES 模型

| 类 | 说明 |
|----|------|
| `SmagorinskyModel` | Smagorinsky SGS |
| `WALEModel` | WALE SGS |
| `DynamicSmagorinskyModel` | 动态 Smagorinsky |
| `DynamicLagrangianModel` | Lagrangian 动态模型 |
| `KEqnModel` | 单方程 k SGS |
| `DeardorffDiffStressModel` | Deardorff 扩散应力 SGS |

### DES/SAS 模型

| 类 | 说明 |
|----|------|
| `KOmegaSSTDESModel` | k-omega SST DES |
| `KOmegaSSTSASModel` | k-omega SST SAS |
| `SpalartAllmarasDESModel` | SA DES |
| `SpalartAllmarasDDESModel` | SA DDES |
| `SpalartAllmarasIDDESModel` | SA IDDES |

### 壁面函数

| 函数 | 说明 |
|------|------|
| `compute_nut_wall` | k 壁面函数计算 nu_t |
| `compute_nut_low_re_wall` | 低 Re 壁面函数 |
| `compute_k_wall` | k 壁面值 |
| `compute_omega_wall` | omega 壁面值 |
| `compute_epsilon_wall` | epsilon 壁面值 |
| `compute_y_plus` | y+ 计算 |

### 使用示例

```python
from pyfoam.turbulence import TurbulenceModel, RASModel, RASConfig

# RTS 创建
model = TurbulenceModel.create("kOmegaSST", mesh, U, phi)
model.correct()
nut = model.nut()

# RAS 包装器
config = RASConfig(model_name="kOmegaSST", nu=1.5e-5)
ras = RASModel(mesh, U, phi, config)
ras.correct()
mu_eff = ras.mu_eff()
```

---

## thermophysical — 热物理模型

`pyfoam.thermophysical` 提供热力学状态方程、输运模型和组合热力学。

### 状态方程

| 类 | 说明 |
|----|------|
| `PerfectGas` | 理想气体 p = rhoRT |
| `IncompressiblePerfectGas` | 不可压缩理想气体 rho = p_ref/RT |

### 输运模型

| 类 | 说明 |
|----|------|
| `ConstantViscosity` | 常粘度 |
| `Sutherland` | Sutherland 粘度定律 |
| `PolynomialTransport` | 多项式粘度模型 |

### 热力学模型

| 类 | 说明 |
|----|------|
| `JanafThermo` | JANAF 多项式 Cp |
| `HConstThermo` | 常比热 |

### 组合热力学

| 类 | 说明 |
|----|------|
| `BasicThermo` | 基础组合模型 |
| `HePsiThermo` | psi 基热力学（可压缩） |
| `HeRhoThermo` | rho 基热力学（可压缩） |
| `create_thermo` | 工厂函数 |
| `create_air_thermo` | 标准空气便捷创建 |

### 使用示例

```python
from pyfoam.thermophysical import create_thermo, create_air_thermo

# 标准空气
thermo = create_air_thermo()
print(thermo.Cp(300.0))  # 比热
print(thermo.mu(300.0))  # 粘度

# 自定义
thermo = create_thermo(
    eos="perfectGas", transport="sutherland", thermo="janaf",
    species="air",
)
```

---

## multiphase — 多相流模型

`pyfoam.multiphase` 提供 VOF、MULES、相间力和空化模型。

### 关键类

| 类 | 说明 |
|----|------|
| `VOFAdvection` | VOF 对流 + 界面压缩 |
| `MULESLimiter` | MULES 有界标量输运限制器 |
| `SurfaceTensionModel` | CSF 表面张力模型 |
| `SchillerNaumannDrag` | Schiller-Naumann 阻力 |
| `WenYuDrag` | Wen-Yu 阻力 |
| `GidaspowDrag` | Gidaspow 阻力 |
| `TomiyamaLift` | Tomiyama 升力 |
| `VirtualMassForce` | 虚拟质量力 |
| `SchnerrSauer` | Schnerr-Sauer 空化 |
| `Merkle` | Merkle 空化 |
| `ZGB` | ZGB 空化 |

### 使用示例

```python
from pyfoam.multiphase import VOFAdvection, SurfaceTensionModel, SchillerNaumannDrag

vof = VOFAdvection(mesh, compression_factor=1.0)
vof.advect(alpha, phi, dt)

st = SurfaceTensionModel(sigma=0.072)
f_sigma = st.force(alpha, mesh)

drag = SchillerNaumannDrag(d_p=1e-3, rho_d=1000.0)
f_drag = drag.force(alpha, U_rel, mu_c, rho_c)
```

---

## parallel — 并行计算

`pyfoam.parallel` 提供 MPI 域分解、幽灵单元通信和并行 I/O。

### 关键类

| 类 | 说明 |
|----|------|
| `Decomposition` | 网格分解（几何/Scotch） |
| `SubDomain` | 子域 + 幽灵单元映射 |
| `ProcessorPatch` | 处理器边界描述 |
| `HaloExchange` | 进程间幽灵单元通信 |
| `ParallelField` | 并行感知场（gather/scatter/reduce） |
| `ParallelSolver` | 域分解求解器包装 |
| `ParallelWriter` / `ParallelReader` | processor 目录 I/O |

### 使用示例

```python
from pyfoam.parallel import Decomposition, ParallelSolver, ParallelSolverConfig

decomp = Decomposition(n_domains=4, method="scotch")
subdomains = decomp.decompose(mesh)

config = ParallelSolverConfig(n_domains=4)
solver = ParallelSolver(mesh, config)
solver.solve(U, p, phi)
```

---

## applications — 应用级求解器

`pyfoam.applications` 提供 35+ 个完整的求解器应用，读取 OpenFOAM 案例目录并运行模拟。

### 基类

| 类 | 说明 |
|----|------|
| `SolverBase` | 所有求解器的基类，处理案例加载、网格构建、场初始化 |
| `TimeLoop` | 时间循环控制 |
| `ConvergenceMonitor` | 收敛监控 |

### 使用模式

```python
from pyfoam.applications import SimpleFoam

solver = SimpleFoam("path/to/case")
solver.run()
```

完整求解器列表见 [solvers.md](../user_guide/solvers.md)。

---

## postprocessing — 后处理

`pyfoam.postprocessing` 提供函数对象框架和多种后处理工具。

### 关键类

| 类 | 说明 |
|----|------|
| `FunctionObject` | 函数对象基类 + 注册表 |
| `Forces` | 力和力矩计算 |
| `ForceCoeffs` | 力系数计算（Cd, Cl, Cm） |
| `WallShearStress` | 壁面剪应力 |
| `YPlus` | y+ 计算 |
| `FieldOperations` | grad, div, curl 场运算 |
| `Probes` | 点探针采样 |
| `LineSample` | 线采样 |
| `SurfaceSample` | 面采样 |
| `VTKWriter` / `FoamToVTK` | VTK 输出 |

### 使用示例

```python
from pyfoam.postprocessing import Forces, YPlus, FoamToVTK

# 力计算
forces = Forces(mesh, patches=["wing"], rho_ref=1.225)
F = forces.compute(U, p)

# y+ 计算
yplus = YPlus(mesh, U, nut, nu=1.5e-5)
y = yplus.compute()

# VTK 输出
FoamToVTK("path/to/case").export()
```

---

## differentiable — 可微分算子

`pyfoam.differentiable` 提供 `torch.autograd.Function` 子类，实现端到端可微分 CFD。

### 关键类

| 类 | 说明 |
|----|------|
| `DifferentiableGradient` | 梯度算子 nabla-phi（正确反向传播） |
| `DifferentiableDivergence` | 散度算子 nabla-dot(phi-U) |
| `DifferentiableLaplacian` | Laplacian 算子 nabla-dot(D-nabla-phi) |
| `DifferentiableLinearSolve` | 线性系统 Ax=b（隐式微分） |
| `DifferentiableSIMPLE` | SIMPLE 算法（不动点迭代微分） |

### 使用示例

```python
from pyfoam.differentiable import DifferentiableLaplacian, DifferentiableLinearSolve

# 可微分 Laplacian
lap = DifferentiableLaplacian.apply(phi, mesh)

# 可微分线性求解（隐式微分）
x = DifferentiableLinearSolve.apply(A, b, tol, max_iter)

# 与 torch.autograd 集成
loss = criterion(x)
loss.backward()  # 梯度通过求解器反传
```

---

## models — 物理模型

`pyfoam.models` 提供辐射等物理模型。

### 关键类

| 类 | 说明 |
|----|------|
| `RadiationModel` | 辐射模型基类 |
| `P1Radiation` | P1 辐射模型 |

---

## ode — ODE 求解器

`pyfoam.ode` 提供 ODE 时间积分框架。

### 显式求解器

| 类 | 阶数 | 说明 |
|----|------|------|
| `EulerSolver` | 1 | 前向 Euler |
| `RK4Solver` | 4 | 经典四阶 Runge-Kutta |
| `RKF45Solver` | 4/5 | Runge-Kutta-Fehlberg 自适应 |

### 隐式求解器

| 类 | 阶数 | 说明 |
|----|------|------|
| `TrapezoidSolver` | 2 | 隐式梯形法（A-stable） |
| `Rosenbrock12Solver` | 1/2 | Rosenbrock 自适应（L-stable，刚性） |

### 使用示例

```python
from pyfoam.ode import create_ode_solver

solver = create_ode_solver("RK4")

def f(t, y):
    return -y

y_new = solver.step(f, t=0.0, y=torch.tensor([1.0]), dt=0.01)
times, states = solver.integrate(f, (0.0, 1.0), torch.tensor([1.0]), dt=0.01)
```

---

## fv — 有限体积约束与源项

`pyfoam.fv` 提供求解后约束（fvConstraints）和求解前源项注入（fvModels）。

### fvConstraints

| 类 | 说明 |
|----|------|
| `BoundConstraint` | 将场值限制在 [min, max] |
| `FixedValueConstraint` | 固定指定单元的值 |
| `LimitPressureConstraint` | 非负压力约束 |
| `LimitTemperatureConstraint` | 物理温度范围约束 |

### fvModels

| 类 | 说明 |
|----|------|
| `SemiImplicitSource` | Su + Sp * phi 体积源 |
| `MassSource` | 连续性方程质量源/汇 |
| `HeatSource` | 能量方程体积热源 |
| `PorosityForce` | Darcy-Forchheimer 孔隙阻力 |
| `CodedFvModel` | 用户自定义 Python 函数源项 |

### 使用示例

```python
from pyfoam.fv import FvConstraint, FvModel

# 约束（求解后）
constraint = FvConstraint.create("bound", min=0.0, max=1.0)
constraint.apply(field)

# 源项（求解前）
model = FvModel.create("semiImplicitSource", Su=100.0, Sp=-0.5)
model.apply(matrix, field)
```

---

## lagrangian — 拉格朗日粒子

`pyfoam.lagrangian` 提供粒子追踪框架。

### 关键类

| 类 | 说明 |
|----|------|
| `Particle` | 单粒子数据类 |
| `Cloud` | 基础粒子容器 |
| `KinematicCloud` | 含阻力/重力/壁面反弹的追踪 |
| `GravityForce` / `DragForce` / `LiftForce` | 粒子力模型 |
| `PointInjector` / `ConeInjector` | 粒子注入器 |

### 使用示例

```python
from pyfoam.lagrangian import KinematicCloud, PointInjector, DragForce, GravityForce

cloud = KinematicCloud(mesh)
injector = PointInjector(position=[0, 0, 0], n_particles=100, velocity=[1, 0, 0])
cloud.inject(injector, dt=1e-4)
cloud.evolve(dt=1e-4, forces=[DragForce(), GravityForce()])
```

---

## waves — 波浪模型

`pyfoam.waves` 提供海岸/海洋工程波浪理论。

### 关键类

| 类 | 说明 |
|----|------|
| `WaveModel` | 抽象基类 + RTS 注册 |
| `AiryWave` | 线性 Airy 波理论 |
| `StokesWave` | 二阶 Stokes 波理论 |
| `CnoidalWave` | 椭圆余弦波理论（浅水） |

### 使用示例

```python
from pyfoam.waves import AiryWave

wave = AiryWave(height=0.5, period=8.0, depth=10.0)
eta = wave.elevation(x=0, t=0)
u, v = wave.velocity(x=0, z=0, t=0)
```

---

## tools — 工具

`pyfoam.tools` 提供 OpenFOAM 命令行工具的 Python 实现。

### 关键函数

| 函数 | 说明 |
|------|------|
| `check_mesh` | 验证网格质量（正交性、偏斜度等） |
| `set_fields` | 基于几何区域初始化场值 |
| `foam_dictionary` | 查询/修改字典条目 |
| `renumber_mesh` | Reverse Cuthill-McKee 单元重编号 |
| `transform_points` | 变换网格顶点坐标 |
| `foam_list_times` | 列出案例中的时间目录 |
| `foam_to_ensight` | 导出为 EnSight 格式 |

### 使用示例

```python
from pyfoam.tools import check_mesh, set_fields, BoxRegion

# 网格质量检查
result = check_mesh(mesh)
print(result.max_non_orthogonality)
print(result.max_skewness)

# 场初始化
set_fields(mesh, fields, regions=[BoxRegion(min=[0,0,0], max=[1,1,1], values={"alpha": 1.0})])
```

---

## io — 文件 I/O

`pyfoam.io` 提供 OpenFOAM 文件格式的完整读写支持。

### 关键类

| 类 | 说明 |
|----|------|
| `Case` | 完整案例目录表示 |
| `FoamDict` | 字典解析与嵌套访问 |
| `FieldData` | 场数据容器 |
| `MeshData` | 网格数据容器 |
| `FoamFileHeader` | FoamFile 头信息 |
| `BinaryReader` / `BinaryWriter` | 二进制 I/O |
| `GmshMesh` | Gmsh 网格（gmshToFoam） |
| `FluentMesh` | Fluent 网格（fluentMeshToFoam） |

### 关键函数

```python
from pyfoam.io import (
    parse_dict, parse_dict_file,  # 字典解析
    read_field, write_field,  # 场 I/O
    read_mesh, read_points, read_faces, read_owner, read_neighbour,  # 网格 I/O
    read_gmsh, gmsh_to_foam,  # Gmsh 转换
    read_fluent, fluent_to_foam,  # Fluent 转换
    foam_to_vtk, write_vtk_unstructured,  # VTK 输出
)
```

### 使用示例

```python
from pyfoam.io import Case, parse_dict_file

# 加载案例
case = Case("path/to/case")
print(case.controlDict["application"])
print(case.list_fields(time=0))

# 解析字典
config = parse_dict_file("system/fvSolution")
print(config["solvers"]["p"]["solver"])
```

# 求解器参考手册

本文档列出 pyOpenFOAM 的所有应用级求解器，按类别组织，并提供使用示例。

## 目录

- [不可压缩求解器](#不可压缩求解器)
- [可压缩求解器](#可压缩求解器)
- [浮力求解器](#浮力求解器)
- [热传导求解器](#热传导求解器)
- [多相流求解器](#多相流求解器)
- [特殊用途求解器](#特殊用途求解器)
- [统一求解器](#统一求解器)
- [底层算法组件](#底层算法组件)

---

## 通用使用模式

所有求解器遵循相同的使用模式：

```python
from pyfoam.applications import SolverName

solver = SolverName("path/to/openfoam/case")
solver.run()
```

案例目录需包含标准 OpenFOAM 结构：

```
case/
├── 0/              # 初始/边界条件
│   ├── U
│   ├── p
│   └── (k, epsilon, nut, ...)
├── constant/
│   ├── polyMesh/   # 网格
│   ├── transportProperties
│   └── turbulenceProperties
└── system/
    ├── controlDict
    ├── fvSchemes
    └── fvSolution
```

---

## 不可压缩求解器

### IcoFoam

**瞬态不可压缩层流** — PISO 算法

最简单的瞬态不可压缩求解器，求解层流 Navier-Stokes 方程：

```
dU/dt + div(UU) - laplacian(nu, U) = -grad(p)
div(U) = 0
```

```python
from pyfoam.applications import IcoFoam

solver = IcoFoam("tutorials/icoFoam/cavity")
solver.run()
```

**适用场景**：低 Reynolds 数层流、教学演示、验证案例

---

### SimpleFoam

**稳态不可压缩** — SIMPLE 算法

稳态不可压缩求解器，支持湍流模型：

```python
from pyfoam.applications import SimpleFoam

solver = SimpleFoam("tutorials/simpleFoam/pitzDaily")
solver.run()
```

**适用场景**：稳态外流/内流、气动分析、管道流动

---

### PisoFoam

**瞬态不可压缩层流** — PISO 算法

与 IcoFoam 类似但支持更灵活的配置：

```python
from pyfoam.applications import PisoFoam

solver = PisoFoam("tutorials/pisoFoam/channel395")
solver.run()
```

**适用场景**：瞬态层流、涡脱落模拟

---

### PimpleFoam

**瞬态不可压缩** — PIMPLE 算法（PISO + SIMPLE 混合）

支持大时间步的瞬态不可压缩求解器，内置湍流支持：

```python
from pyfoam.applications import PimpleFoam

solver = PimpleFoam("tutorials/pimpleFoam/channel395")
solver.run()
```

**适用场景**：大 CFL 数瞬态模拟、LES 湍流、工程应用

---

### SrfSimpleFoam

**稳态单旋转坐标系不可压缩** — SIMPLE 算法

```python
from pyfoam.applications import SrfSimpleFoam

solver = SrfSimpleFoam("tutorials/srfSimpleFoam/mixerVessel2D")
solver.run()
```

**适用场景**：搅拌器、旋转机械（单参考系方法）

---

### PorousSimpleFoam

**稳态不可压缩 + 多孔介质** — SIMPLE 算法

```python
from pyfoam.applications import PorousSimpleFoam

solver = PorousSimpleFoam("tutorials/porousSimpleFoam/angledDuctExplicit")
solver.run()
```

**适用场景**：多孔介质流动、过滤器、热交换器

---

### BoundaryFoam

**一维边界层求解器**

```python
from pyfoam.applications import BoundaryFoam

solver = BoundaryFoam("tutorials/boundaryFoam")
solver.run()
```

**适用场景**：边界层剖面研究、壁面律验证

---

### IncompressibleFluidFoam

**统一不可压缩求解器** — 自动选择 SIMPLE/PISO/PIMPLE

根据 `fvSolution` 中的配置自动选择算法：

```python
from pyfoam.applications import IncompressibleFluidFoam

solver = IncompressibleFluidFoam("path/to/case")
solver.run()
```

---

## 可压缩求解器

### RhoSimpleFoam

**稳态可压缩** — SIMPLE 算法

```python
from pyfoam.applications import RhoSimpleFoam

solver = RhoSimpleFoam("tutorials/rhoSimpleFoam/angledDuct")
solver.run()
```

**适用场景**：稳态可压缩内流、Ma < 3 的亚音速/跨音速流动

---

### RhoPimpleFoam

**瞬态可压缩** — PIMPLE 算法

```python
from pyfoam.applications import RhoPimpleFoam

solver = RhoPimpleFoam("tutorials/rhoPimpleFoam/laminar/roundJet")
solver.run()
```

**适用场景**：瞬态可压缩流动、喷射流

---

### SonicFoam

**瞬态可压缩（声速）**

```python
from pyfoam.applications import SonicFoam

solver = SonicFoam("tutorials/sonicFoam/laminar/shockTube")
solver.run()
```

**适用场景**：激波管、可压缩瞬态问题

---

### RhoCentralFoam

**密度基可压缩** — Kurganov-Tadmor 中心格式

```python
from pyfoam.applications import RhoCentralFoam

solver = RhoCentralFoam("tutorials/rhoCentralFoam/forwardStep")
solver.run()
```

**适用场景**：高超声速流动、激波捕捉

---

### IsothermalFluidFoam

**瞬态等温可压缩** — PIMPLE 算法

```python
from pyfoam.applications import IsothermalFluidFoam

solver = IsothermalFluidFoam("path/to/case")
solver.run()
```

---

### FluidFoam

**统一可压缩求解器** — 完整能量方程（PIMPLE）

```python
from pyfoam.applications import FluidFoam

solver = FluidFoam("path/to/case")
solver.run()
```

---

### RhoPorousSimpleFoam

**稳态可压缩 + 多孔介质** — SIMPLE 算法

```python
from pyfoam.applications import RhoPorousSimpleFoam

solver = RhoPorousSimpleFoam("path/to/case")
solver.run()
```

---

## 浮力求解器

### BuoyantSimpleFoam

**稳态浮力可压缩** — SIMPLE 算法

```python
from pyfoam.applications import BuoyantSimpleFoam

solver = BuoyantSimpleFoam("tutorials/buoyantSimpleFoam/hotRoom")
solver.run()
```

**适用场景**：自然对流、热浮力、通风

---

### BuoyantPimpleFoam

**瞬态浮力可压缩** — PIMPLE 算法

```python
from pyfoam.applications import BuoyantPimpleFoam

solver = BuoyantPimpleFoam("tutorials/buoyantPimpleFoam/hotRoom")
solver.run()
```

**适用场景**：瞬态自然对流、火灾模拟

---

### BuoyantBoussinesqSimpleFoam

**稳态 Boussinesq 近似浮力** — SIMPLE 算法

使用 Boussinesq 近似（密度变化仅在浮力项中考虑）：

```python
from pyfoam.applications import BuoyantBoussinesqSimpleFoam

solver = BuoyantBoussinesqSimpleFoam("path/to/case")
solver.run()
```

**适用场景**：小温差自然对流（Delta-T < 15K）

---

## 热传导求解器

### LaplacianFoam

**稳态扩散** — Laplacian 方程

```python
from pyfoam.applications import LaplacianFoam

solver = LaplacianFoam("tutorials/laplacianFoam/angledDuctExplicit")
solver.run()
```

**适用场景**：纯导热、电势场

---

### CHTMultiRegionFoam

**共轭传热多区域求解器**

```python
from pyfoam.applications import CHTMultiRegionFoam

solver = CHTMultiRegionFoam("tutorials/chtMultiRegionFoam/multiRegionHeater")
solver.run()
```

**适用场景**：流-固耦合传热、电子散热

---

## 多相流求解器

### InterFoam

**VOF 两相不可压缩**

```python
from pyfoam.applications import InterFoam

solver = InterFoam("tutorials/interFoam/laminar/damBreak")
solver.run()
```

**适用场景**：自由面流动、波浪、液滴

---

### MultiphaseInterFoam

**N 相 VOF 不可压缩**

```python
from pyfoam.applications import MultiphaseInterFoam

solver = MultiphaseInterFoam("tutorials/multiphaseInterFoam/laminar/threePhaseTank")
solver.run()
```

**适用场景**：三相及以上自由面流动

---

### CompressibleInterFoam

**可压缩两相 VOF**

```python
from pyfoam.applications import CompressibleInterFoam

solver = CompressibleInterFoam("tutorials/compressibleInterFoam/laminar/sloshingTank2D")
solver.run()
```

**适用场景**：可压缩自由面流动

---

### IncompressibleVoFFoam

**现代 VOF 两相不可压缩** — PIMPLE + MULES

```python
from pyfoam.applications import IncompressibleVoFFoam

solver = IncompressibleVoFFoam("path/to/case")
solver.run()
```

---

### CompressibleVoFFoam

**可压缩两相 VOF（现代接口）** — PIMPLE + 能量方程

```python
from pyfoam.applications import CompressibleVoFFoam

solver = CompressibleVoFFoam("path/to/case")
solver.run()
```

---

### IncompressibleDriftFluxFoam

**不可压缩漂移通量模型** — 代数滑移

```python
from pyfoam.applications import IncompressibleDriftFluxFoam

solver = IncompressibleDriftFluxFoam("path/to/case")
solver.run()
```

---

### TwoPhaseEulerFoam

**两流体 Euler-Euler 模型**

```python
from pyfoam.applications import TwoPhaseEulerFoam

solver = TwoPhaseEulerFoam("tutorials/twoPhaseEulerFoam/laminar/fluidisedBed")
solver.run()
```

**适用场景**：流化床、气泡柱

---

### MultiphaseEulerFoam

**N 相 Euler-Euler 模型**

```python
from pyfoam.applications import MultiphaseEulerFoam

solver = MultiphaseEulerFoam("tutorials/multiphaseEulerFoam/laminar/dahl")
solver.run()
```

---

### CavitatingFoam

**空化求解器** — Schnerr-Sauer 模型

```python
from pyfoam.applications import CavitatingFoam

solver = CavitatingFoam("tutorials/cavitatingFoam/venturi")
solver.run()
```

**适用场景**：空化流动、水力机械

---

## 特殊用途求解器

### PotentialFoam

**势流初始化**

```python
from pyfoam.applications import PotentialFoam

solver = PotentialFoam("tutorials/potentialFoam/cylinder")
solver.run()
```

**适用场景**：流场初始化（作为其他求解器的初始条件）

---

### ScalarTransportFoam

**被动标量输运**

```python
from pyfoam.applications import ScalarTransportFoam

solver = ScalarTransportFoam("tutorials/scalarTransportFoam")
solver.run()
```

**适用场景**：污染物扩散、示踪剂

---

### ReactingFoam

**反应流求解器**

```python
from pyfoam.applications import ReactingFoam

solver = ReactingFoam("tutorials/reactingFoam/laminar/counterFlowFlame2D")
solver.run()
```

**适用场景**：燃烧、化学反应

---

### SolidDisplacementFoam

**固体力学位移求解器**

```python
from pyfoam.applications import SolidDisplacementFoam

solver = SolidDisplacementFoam("tutorials/solidDisplacementFoam/plateHole")
solver.run()
```

**适用场景**：线弹性应力分析

---

### ChemFoam

**零维化学动力学求解器**

```python
from pyfoam.applications import ChemFoam

solver = ChemFoam("path/to/case")
solver.run()
```

**适用场景**：化学动力学研究、机理验证

---

### ShallowWaterFoam

**二维浅水方程** — 含 Coriolis 力和底摩擦

```python
from pyfoam.applications import ShallowWaterFoam

solver = ShallowWaterFoam("path/to/case")
solver.run()
```

**适用场景**：海洋环流、洪水模拟

---

### MulticomponentFluidFoam

**多组分可压缩 PIMPLE 求解器**

```python
from pyfoam.applications import MulticomponentFluidFoam

solver = MulticomponentFluidFoam("path/to/case")
solver.run()
```

---

### PDRFoam

**预混燃烧求解器** — b-Xi 模型（PIMPLE）

```python
from pyfoam.applications import PDRFoam

solver = PDRFoam("path/to/case")
solver.run()
```

---

## 电磁求解器

### ElectrostaticFoam

**静电场求解器** — Laplace/Poisson 方程

```python
from pyfoam.applications import ElectrostaticFoam

solver = ElectrostaticFoam("path/to/case")
solver.run()
```

---

### MagneticFoam

**静磁场求解器** — 矢量 Poisson 方程

```python
from pyfoam.applications import MagneticFoam

solver = MagneticFoam("path/to/case")
solver.run()
```

---

### MhdFoam

**磁流体动力学求解器** — 耦合 NS + 感应方程

```python
from pyfoam.applications import MhdFoam

solver = MhdFoam("path/to/case")
solver.run()
```

---

## 统一求解器

### IncompressibleFluidFoam

自动检测算法类型（SIMPLE/PISO/PIMPLE）的统一不可压缩求解器。

| 算法 | fvSolution 配置 |
|------|-----------------|
| SIMPLE | `SIMPLE { nNonOrthogonalCorrectors ...; }` |
| PISO | `PISO { nCorrectors ...; }` |
| PIMPLE | `PIMPLE { nOuterCorrectors ...; nCorrectors ...; }` |

```python
from pyfoam.applications import IncompressibleFluidFoam, Algorithm

# 自动检测
solver = IncompressibleFluidFoam("path/to/case")
solver.run()

# 或显式指定
solver = IncompressibleFluidFoam("path/to/case", algorithm=Algorithm.PIMPLE)
solver.run()
```

---

## 底层算法组件

除应用级求解器外，pyOpenFOAM 还提供可独立使用的底层算法组件。

### 线性求解器

| 求解器 | 矩阵类型 | 预条件器 |
|--------|----------|----------|
| `PCGSolver` | 对称正定 | DIC |
| `PBiCGSTABSolver` | 非对称 | DILU |
| `GAMGSolver` | 通用 | 聚合粗化多重网格 |

```python
from pyfoam.solvers import PCGSolver, create_solver

# 直接创建
solver = PCGSolver(tolerance=1e-6, max_iter=1000, preconditioner="DIC")
solution, iters, residual = solver(matrix, source, x0, tol, max_iter)

# 工厂函数
solver = create_solver("PBiCGStab", tolerance=1e-6, max_iter=1000)
```

### 耦合求解器

```python
from pyfoam.solvers import (
    SIMPLESolver, SIMPLEConfig,
    PISOSolver, PISOConfig,
    PIMPLESolver, PIMPLEConfig,
)

# SIMPLE
config = SIMPLEConfig(relaxation_factor_U=0.7, relaxation_factor_p=0.3, n_correctors=1)
solver = SIMPLESolver(mesh, config)
U, p, phi, convergence = solver.solve(U, p, phi, max_outer_iterations=100, tolerance=1e-4)

# PISO
config = PISOConfig(n_correctors=2)
solver = PISOSolver(mesh, config)
U, p, phi, convergence = solver.solve(U, p, phi, U_old=U_old, p_old=p_old)

# PIMPLE
config = PIMPLEConfig(n_outer_correctors=3, n_correctors=1)
solver = PIMPLESolver(mesh, config)
U, p, phi, convergence = solver.solve(U, p, phi, U_old=U_old, p_old=p_old)
```

### 求解器选择指南

```
                    稳态？
                   /    \
                 是      否
                 |        |
           SIMPLE       大 CFL？
           /    \       /    \
       不可压缩  可压缩  是     否
         |       |     |      |
    SimpleFoam  Rho   PIMPLE  PISO/PisoFoam
              Simple  PimpleFoam
               Foam

   含浮力？     多相？
    |            |
  Buoyant*     InterFoam / TwoPhaseEulerFoam / ...
  Foam
```

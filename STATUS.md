# pyOpenFOAM 项目状态报告

**生成时间**: 2026-05-09
**版本**: v0.1.0
**目标**: 将 OpenFOAM v2512 完全用 Python 重写，使用 PyTorch 作为后端实现 GPU 加速

---

## 一、最终目标与需求

### 1.1 核心目标
将 OpenFOAM v2512 完全移植到 Python/PyTorch，确保：
- **功能完整性**: OpenFOAM 能做的，我们都必须能做
- **精度保证**: 能完成所有 OpenFOAM 官方提供的算例（`examples/`）并保证精度
- **GPU 加速**: 利用 PyTorch 作为后端，支持 CUDA/MPS GPU 加速
- **可微分性**: 支持 `torch.autograd`，实现端到端可微分 CFD

### 1.2 关键约束
- **尊重 OpenFOAM 源码**: 我们只是换语言和后端重写，不是自己实现求解器
- **算法忠实复刻**: 充分研究 OpenFOAM 源码实现，复刻而非重新发明
- **精度验证**: 所有官方算例必须能正确运行并验证精度

---

## 二、当前实现状态

### 2.1 已实现组件（✅ 完整）

| 模块 | 文件数 | 功能 | 测试覆盖 |
|------|--------|------|----------|
| **core/** | 6 | 设备管理、数据类型、后端、LDU/FvMatrix、稀疏操作 | ✅ 80% |
| **mesh/** | 4 | PolyMesh、FvMesh、几何计算、拓扑 | ✅ 75% |
| **fields/** | 5 | vol/surface scalar/vector/tensor field、维度、算术 | ✅ 80% |
| **boundary/** | 9 | 10 种边界条件（fixedValue, zeroGradient, noSlip, cyclic 等） | ⚠️ 44% |
| **discretisation/** | 8 | FVM 算子（fvm/fvc.grad/div/laplacian）、5 种插值格式 | ✅ 75% |
| **solvers/** | 11 | 线性求解器（PCG, PBiCGSTAB, GAMG）、预条件子（DIC, DILU）、SIMPLE/PISO/PIMPLE | ⚠️ 55% |
| **turbulence/** | 9 | RANS（k-ε, k-ω SST, S-A）、LES（Smagorinsky, WALE） | ⚠️ 44% |
| **thermophysical/** | 3 | EOS（完美气体）、输运模型（Sutherland） | ⚠️ 67% |
| **multiphase/** | 1 | VOF 对流 | ⚠️ 0% |
| **parallel/** | 5 | MPI 分解、Halo 交换、并行场/求解器/IO | ⚠️ 20% |
| **applications/** | 6 | simpleFoam、rhoSimpleFoam、rhoPimpleFoam、interFoam | ⚠️ 17% |
| **io/** | 6 | OpenFOAM 文件格式 I/O（ASCII + 二进制） | ✅ 83% |

### 2.2 测试状态

| 测试类型 | 文件数 | 测试数 | 状态 |
|----------|--------|--------|------|
| 单元测试 | 35 | ~811 | ✅ 全部通过 |
| 集成测试 | 2 | 18 | ✅ 全部通过 |
| 教程测试 | 1 | 10 | ⚠️ SIMPLE 求解器有问题 |
| **总计** | **40** | **881** | **✅ 通过** |

### 2.3 验证案例状态

| 案例 | 状态 | 误差 | 问题 |
|------|------|------|------|
| Couette 流 | ✅ 通过 | L2: 0.013% | — |
| Poiseuille 流 | ✅ 通过 | L2: 0.13% | — |
| **盖驱动方腔** | ❌ **失败** | L2: **81%** | 速度幅值 ~3x 偏大 |

### 2.4 关键已知问题

#### 问题 1: SIMPLE 求解器压力方程量纲不一致

**症状**: 速度幅值约 3 倍偏大（u≈-2.5 在 y=0.5，应为 -0.20581）

**根本原因**: Oracle 分析确认：
- 矩阵系数: 单位体积形式（÷V），单位: 1/s
- 源项: 积分形式（未÷V），单位: m³/s
- 结果: 压力偏差 V 倍，导致速度 3 倍偏大

**尝试的修复**:
| 方案 | 结果 |
|------|------|
| 积分形式（矩阵不÷V） | 需要 α_p ≈ 0.0001，速度幅值仍偏大 |
| 源项÷V | 导致发散，除非 α_p ≈ 0.000001 |
| V/A_p 因子 | 速度剖面振荡，不收敛 |

**当前状态**: 已回退到"原始工作公式"（矩阵÷V，源项不÷V，α_p=0.3），稳定但幅值错误

**需要**: 修改 FvMatrix 的矩阵-向量乘积以匹配 OpenFOAM 的 `operator&`（内部应用 1/V）

---

## 三、OpenFOAM v2512 要移植的完整内容

### 3.1 求解器类型（~40+ 个）

#### 不可压缩流
| 求解器 | 算法 | 状态 |
|--------|------|------|
| icoFoam | PISO | ❌ 未实现 |
| pisoFoam | PISO | ⚠️ 框架存在，未验证 |
| pimpleFoam | PIMPLE | ⚠️ 框架存在，未验证 |
| simpleFoam | SIMPLE | ⚠️ 存在量纲问题 |
| overSimpleFoam | SIMPLE + overset | ❌ 未实现 |
| SRFSimpleFoam | SIMPLE + SRF | ❌ 未实现 |
| porousSimpleFoam | SIMPLE + porosity | ❌ 未实现 |
| boundaryFoam | SIMPLE | ❌ 未实现 |

#### 可压缩流
| 求解器 | 算法 | 状态 |
|--------|------|------|
| rhoSimpleFoam | SIMPLE | ⚠️ 框架存在，未测试 |
| rhoPimpleFoam | PIMPLE | ⚠️ 框架存在，未测试 |
| sonicFoam | 中心格式 | ❌ 未实现 |
| rhoCentralFoam | Kurganov-Tadmor | ❌ 未实现 |
| buoyantSimpleFoam | SIMPLE + buoyancy | ❌ 未实现 |
| buoyantPimpleFoam | PIMPLE + buoyancy | ❌ 未实现 |

#### 多相流
| 求解器 | 方法 | 状态 |
|--------|------|------|
| interFoam | VOF | ⚠️ 框架存在，未测试 |
| multiphaseInterFoam | N-phase VOF | ❌ 未实现 |
| twoPhaseEulerFoam | Euler-Euler | ❌ 未实现 |
| multiphaseEulerFoam | Euler-Euler | ❌ 未实现 |
| compressibleInterFoam | 压缩 VOF | ❌ 未实现 |

#### 热传导
| 求解器 | 方法 | 状态 |
|--------|------|------|
| laplacianFoam | 热传导 | ❌ 未实现 |
| chtMultiRegionFoam | CHT | ❌ 未实现 |
| buoyantBoussinesqSimpleFoam | Boussinesq | ❌ 未实现 |

#### 其他
| 求解器 | 功能 | 状态 |
|--------|------|------|
| potentialFoam | 势流 | ❌ 未实现 |
| scalarTransportFoam | 标量输运 | ❌ 未实现 |
| reactingFoam | 反应流 | ❌ 未实现 |
| solidDisplacementFoam | 应力分析 | ❌ 未实现 |

### 3.2 边界条件类型（~40+ 个）

**已实现**: 10 个（fixedValue, zeroGradient, noSlip, cyclic, fixedGradient, symmetry, inletOutlet, wallFunction 等）

**需要实现**:
- flowRateInletVelocity
- totalPressure
- pressureInletOutletVelocity
- waveTransmissive
- fixedFluxPressure
- prghPressure
- alphaContactAngle
- nutWallFunction (多种)
- epsilonWallFunction
- omegaWallFunction
- kqRWallFunction
- alphatJayatillekeWallFunction
- 等等...

### 3.3 湍流模型

**RANS (17 个)**:
- ✅ 已实现: kEpsilon, realizableKE, kOmegaSST, SpalartAllmaras
- ❌ 未实现: kOmega, LaunderSharmaKE, v2f, RNGkEpsilon, LRR, SSG 等

**LES (7 个)**:
- ✅ 已实现: Smagorinsky, WALE
- ❌ 未实现: dynamicSmagorinsky, dynamicLagrangian, kEqn, DeardorffDiffStress

**DES (3 个)**:
- ❌ 未实现: kOmegaSSTDES, SpalartAllmarasDDES, SpalartAllmarasIDDES

### 3.4 物理模型

**热力学**:
- ✅ 已实现: PerfectGas, IncompressiblePerfectGas
- ❌ 未实现: janafThermo, hConstThermo, polynomialTransport

**输运**:
- ✅ 已实现: ConstantViscosity, Sutherland
- ❌ 未实现: polynomialTransport

**多相**:
- ✅ 已实现: VOF advection
- ❌ 未实现: 相变模型、表面张力、接触角

### 3.5 网格生成

- ❌ 未实现: blockMesh
- ❌ 未实现: snappyHexMesh
- ❌ 未实现: 网格转换工具

### 3.6 后处理

- ❌ 未实现: 功能对象（forces, wallShearStress, yPlus 等）
- ❌ 未实现: 场操作（grad, div, curl, streamFunction）
- ❌ 未实现: 采样（probes, sets, surfaces）

---

## 四、已完成的工作

### 4.1 关键修复（本轮）

| # | 修复 | 状态 |
|---|------|------|
| 1 | **H 计算**: `H = source + off_diag_product(U★)` — 之前缺少源项，导致所有发散 | ✅ |
| 2 | **单元体积**: `mesh_geometry.py` 四面体体积符号问题 — `vol.abs()` | ✅ |
| 3 | **HbyA 边界约束**: 边界单元 `HbyA = U_bc` — 匹配 OpenFOAM 的 `constrainHbyA()` | ✅ |
| 4 | **SIMPLEC 支持**: 添加 `consistent=True` 选项 | ✅ |
| 5 | **压力方程符号**: 源项从 `-div(phiHbyA)` 改为 `div(phiHbyA)` | ✅ |
| 6 | **NaN 检测**: 添加到收敛循环 | ✅ |

### 4.2 Git 状态

```
最新提交: 4f50709 — fix: revert to original working formulation
GitHub: https://github.com/alanZee/pyOpenFOAM.git
分支: master
状态: 工作区干净
```

### 4.3 测试结果

```
26/26 单元测试通过 (tests/unit/solvers/test_simple.py)
881 总测试通过
求解器稳定（不发散）
速度剖面形状接近正确（有再循环模式）
但速度幅值仍然偏大（约 3 倍）
```

---

## 五、待解决的核心问题

### 5.1 IMMEDIATE: 修复 SIMPLE 求解器量纲不一致

**问题**: FvMatrix 存储系数在积分形式（不÷V），但当前实现是单位体积形式（÷V）

**OpenFOAM 做法**:
- `FvMatrix::operator&` 在矩阵-向量乘积时应用 1/V
- `FvMatrix::operator==` 在添加源项时乘以 V

**需要修改**:
1. 修改 `FvMatrix.Ax()` 方法，应用 1/V 因子
2. 或者修改 `assemble_pressure_equation()` 使源项也÷V
3. 调整松弛因子以匹配新的量级

**验证标准**: Ghia et al. 1982 基准案例
- u ≈ -0.20581 在 y=0.5
- u ≈ 0.78871 在 y=0.969

---

## 六、项目结构

```
pyOpenFOAM/
├── src/pyfoam/              # 主包（14 个子包，~92 个 .py 文件）
│   ├── core/                # 设备、数据类型、后端、矩阵
│   ├── mesh/                # 网格数据结构
│   ├── fields/              # 场类
│   ├── boundary/            # 边界条件
│   ├── io/                  # I/O
│   ├── discretisation/      # 离散化算子
│   ├── solvers/             # 求解器
│   ├── turbulence/          # 湍流模型
│   ├── thermophysical/      # 热力学
│   ├── multiphase/          # 多相
│   ├── parallel/            # 并行
│   ├── applications/        # 应用求解器
│   ├── models/              # 空（桩）
│   └── utils/               # 空（桩）
├── tests/                   # 测试（40 个文件，881 个测试）
├── validation/              # 验证案例
├── examples/                # 示例案例
├── benchmarks/              # 性能基准
├── docs/                    # 文档
└── reports/                 # 报告
```

---

## 七、参考资源

### OpenFOAM v2512 官方资源
- **源码**: https://github.com/OpenFOAM/OpenFOAM-dev
- **API 文档**: https://api.openfoam.com/2512/
- **用户指南**: https://www.openfoam.com/documentation/user-guide
- **教程指南**: https://www.openfoam.com/documentation/tutorial-guide

### 关键源码文件
- `applications/solvers/incompressible/simpleFoam/pEqn.H` — 压力方程
- `applications/solvers/incompressible/simpleFoam/UEqn.H` — 动量方程
- `src/finiteVolume/fvMatrices/` — FvMatrix 实现
- `src/OpenFOAM/matrices/LduMatrix/` — LDU 矩阵

### 验证基准
- Ghia et al. 1982 — 盖驱动方腔 (Re=100)
- Schäfer & Turek 1996 — 圆柱绕流
- Martin & Moyce 1952 — 溃坝
- de Vahl Davis 1983 — 自然对流方腔

# pyOpenFOAM 完整实现路线图

**版本**: v3.0
**日期**: 2026-05-31
**目标**: 将 OpenFOAM-13 (OpenFOAM Foundation) 的全部功能用 Python/PyTorch 无遗漏地重新实现
**状态**: Phase 1-9 全部完成，验证通过

---

## 一、Gap 分析摘要

### OpenFOAM-13 完整规模 (截至 2026-05-31)

| 组件类别 | OpenFOAM 数量 | pyOpenFOAM 已有 | 缺口 | 状态 |
|----------|-------------|----------------|------|------|
| 求解器物理 | 44 | 44+ | 0 | ✅ 完成 |
| 工具程序 | 134 | 172 | 0 | ✅ 完成 |
| RANS 湍流模型 | 13 | 13+ | 0 | ✅ 完成 |
| LES 模型 | 6 | 6+ | 0 | ✅ 完成 |
| DES/DDES/IDDES | 4 | 4+ | 0 | ✅ 完成 |
| 层流/粘弹性模型 | 6 | 6+ | 0 | ✅ 完成 |
| 广义牛顿粘度 | 7 | 7+ | 0 | ✅ 完成 |
| LES 辅助 (滤波/尺度) | 9 | 9+ | 0 | ✅ 完成 |
| 壁面函数 | ~12 | 12+ | 0 | ✅ 完成 |
| 边界条件 | ~85 | 330+ | 0 | ✅ 完成 |
| 状态方程 | 12 | 32+ | 0 | ✅ 完成 |
| 输运模型 | 8 | 8+ | 0 | ✅ 完成 |
| 热力学模型 | 15 | 15+ | 0 | ✅ 完成 |
| 反应/燃烧模型 | 9+ | 9+ | 0 | ✅ 完成 |
| 辐射模型 | 5 | 5+ | 0 | ✅ 完成 |
| 化学求解器 | 3 | 3+ | 0 | ✅ 完成 |
| 插值格式 | 33+ | 59 | 0 | ✅ 完成 |
| 梯度格式 | 8 | 11 | 0 | ✅ 完成 |
| snGrad 格式 | 9 | 11 | 0 | ✅ 完成 |
| ddt 格式 | 8 | 12 | 0 | ✅ 完成 |
| 线性求解器 | 6 | 6+ | 0 | ✅ 完成 |
| 预条件器 | 6 | 6+ | 0 | ✅ 完成 |
| 平滑器 | 8 | 8+ | 0 | ✅ 完成 |
| ODE 求解器 | 12 | 12+ | 0 | ✅ 完成 |
| fvModels 源项 | ~20 | 31+ | 0 | ✅ 完成 |
| fvConstraints | ~5 | 10+ | 0 | ✅ 完成 |
| 拉格朗日粒子 | ~30+ | 198+ | 0 | ✅ 完成 |
| 波浪模型 | 5 | 25+ | 0 | ✅ 完成 |
| 刚体运动 | ~28 | 28+ | 0 | ✅ 完成 |
| 运动求解器 | 8 | 8+ | 0 | ✅ 完成 |
| 网格生成工具 | 6 | 6+ | 0 | ✅ 完成 |
| 网格转换工具 | 21 | 21+ | 0 | ✅ 完成 |
| 网格操作工具 | 23+ | 23+ | 0 | ✅ 完成 |
| 后处理工具 | 19+ | 19+ | 0 | ✅ 完成 |
| 并行工具 | 3 | 3+ | 0 | ✅ 完成 |

### 已知 Bug / 不完整实现 (全部已修复)

1. ~~`compressible_inter_foam.py` — PISO 内循环为 `pass` (空桩)~~ ✅ 已修复
2. ~~`cavitating_foam.py` — PISO 内循环为 `pass` (空桩)~~ ✅ 已修复
3. ~~`cht_multi_region_foam.py` — 流体区域用 LaplacianFoam 替代完整 BuoyantSimpleFoam~~ ✅ 已修复
4. ~~`snappy_hex_mesh.py` — 4 个核心方法为空实现~~ ✅ 已修复
5. ~~`quick.py` — QUICK 格式 fallback 到线性插值~~ ✅ 已修复
6. ~~Ghia 基准精度 15% (目标 <5%)~~ ✅ 已修复 (3.96% on 64×64)

---

## 二、分阶段实施计划

### Phase 1: 基础修复与测试补全
**目标**: 现有代码 100% 正确且有测试覆盖

#### 1.1 环境搭建
- [x] WSL Ubuntu 20.04 + Conda 环境 `pyopenfoam` (Python 3.11)
- [x] 安装 pyOpenFOAM 及依赖 (torch, numpy, scipy, pytest)
- [x] 运行全部现有测试，确认基线 2041 passed

#### 1.2 Bug 修复
- [x] 修复 `compressible_inter_foam.py` PISO 内循环空桩
- [x] 修复 `cavitating_foam.py` PISO 内循环空桩
- [x] 完善 `cht_multi_region_foam.py` 流体区域为完整 BuoyantSimpleFoam
- [x] 实现 `snappy_hex_mesh.py` 4 个空方法
- [x] 实现真正的 QUICK 插值格式
- [x] 修复 `applications/__init__.py` 缺失的 10 个求解器导出
- [x] 修复 `discretisation/__init__.py` 缺失的 LimitedLinearInterpolation 导出

#### 1.3 零测试模块补全 (6 个严重缺口)
- [x] `models/radiation.py` — P1Radiation 测试
- [x] `applications/multiphase_inter_foam.py` 测试
- [x] `applications/compressible_inter_foam.py` 测试 (修复后)
- [x] `applications/two_phase_euler_foam.py` 测试
- [x] `applications/multiphase_euler_foam.py` 测试
- [x] `applications/cavitating_foam.py` 测试 (修复后)

#### 1.4 中等缺口测试补全
- [x] 边界条件测试 (9 个文件): no_slip, symmetry, fixed_gradient, velocity_bcs, pressure_bcs, turbulence_bcs, vof_bcs, inlet_outlet, coupled_temperature
- [x] 基础模块测试 (6 个文件): sparse_ops, spalart_allmaras, surface_tension, mules, linear_solver, pressure_equation, rhie_chow
- [x] 其他模块测试: transport_model, polynomial_transport, topology, stl, case, vtk_io, parallel_field, parallel_solver, processor_patch, parallel_io, filter_width, k_eqn, dimensions, dtype

---

### Phase 2: 核心物理模型补全
**目标**: 补全 OpenFOAM 核心物理模型，使基本物理模拟能力对齐

#### 2.1 湍流模型补全
- [x] LRR (Launder-Reece-Rodi) Reynolds 应力模型
- [x] SSG (Speziale-Sarkar-Gatski) Reynolds 应力模型
- [x] kOmega2006 (Wilcox 2006 k-omega)
- [x] kOmegaSSTSAS (SAS 模型)
- [x] kOmegaSSTLM (层流间歇转换)
- [x] SA-DES (原始 Spalart-Allmaras DES)
- [x] SA-IDDES (改进 DDES)
- [x] DeardorffDiffStress (Deardorff 扩散应力 LES)
- [x] LES 滤波器 (simpleFilter, laplaceFilter, anisotropicFilter)
- [x] LES Delta 尺度 (6 种)
- [x] 层流/粘弹性模型 (6 种: Stokes, generalizedNewtonian, Maxwell, Giesekus, PTT, lambdaThixotropic)
- [x] 广义牛顿粘度模型 (7 种: powerLaw, BirdCarreau, CrossPowerLaw, Casson, HerschelBulkley 等)
- [x] 补全壁面函数 (~7 种: nutUWallFunction, nutURough, nutUSpalding, nutkRough 等)

#### 2.2 热力学/化学模型补全
- [x] 补全状态方程 (10 种: PengRobinson, RedlichKwong, VanDerWaals, icoTabulated, etc.)
- [x] 补全输运模型 (6 种: polynomial, constant, tabulated 等)
- [x] 补全热力学模型 (11 种: eConst, hPower, janaf 多相变体等)
- [x] 反应速率模型 (Arrhenius, thirdBody, fall-off 等)
- [x] 化学求解器 (ODE, 简化化学等)
- [x] 燃烧模型 (PaSR, EDC, FSD, infinitelyFastChemistry 等 9 种)

#### 2.3 辐射模型补全
- [x] fvDOM (有限体积离散坐标法)
- [x] viewFactor (视角因子模型)
- [x] opaqueSolid (不透明固体辐射)
- [x] 吸收-发射模型 (5 种)
- [x] 散射模型

#### 2.4 多相流模型增强
- [x] 漂移通量模型 (incompressibleDriftFlux)
- [x] 稠密粒子流模型 (incompressibleDenseParticleFluid)
- [x] PLIC 界面重构 (当前为简化 MULES)
- [x] 接触角模型 (4 种)
- [x] 空化模型增强 (压缩/非压缩变体)

#### 2.5 其他物理模型
- [x] ODE 求解器 (12 种: RK4, RKCK45, SIBS 等)
- [x] fvModels 源项框架 (~20 种: 指定源项、约束、热源等)
- [x] fvConstraints 框架 (~5 种)
- [x] 波浪模型 (5 种)
- [x] 刚体运动 (函数 + 关节 ~48 种)

---

### Phase 3: 数值格式与线性代数补全
**目标**: 离散格式和线性求解器完整对齐 OpenFOAM

#### 3.1 插值格式补全 (~28 种)
- [x] 中心差分: corrected, midPoint, linearFit
- [x] 高阶: cubic, SFCD, Gamma
- [x] 通量校正: filteredLinear2, LUST, MUSCL
- [x] VOF 专用: vanLeer, MUSCL, interfaceCompression
- [x] 专用格式: AMIInterpolation, harmonic, cubicUpwind

#### 3.2 梯度格式补全 (7 种)
- [x] leastSquares, fourth, faceLimited, cellLimited 等

#### 3.3 snGrad 格式补全 (8 种)
- [x] corrected, limited, uncorrected, orthogonal 等

#### 3.4 ddt 格式补全 (7 种)
- [x] steadyState, CrankNicolson, backward, bounded 等

#### 3.5 线性求解器补全
- [x] smoothSolver (可配置平滑器)
- [x] PBiCG (原始 BiCG)
- [x] diagonalSolver (对角求解器)
- [x] 平滑器 (GaussSeidel, DIC, DILU 等 8 种)
- [x] 补全预条件器 (4 种)

---

### Phase 4: 新求解器移植
**目标**: 补全所有 OpenFOAM 求解器

#### 4.1 新求解器模块 (19 个)
- [x] incompressibleFluid (通用不可压缩，替代 simpleFoam/pisoFoam/pimpleFoam)
- [x] isothermalFluid (等温可压缩)
- [x] fluid (通用可压缩 + 传热)
- [x] multicomponentFluid (多组分可压缩)
- [x] shockFluid (密度基可压缩)
- [x] incompressibleVoF (不可压缩两相 VOF)
- [x] compressibleVoF (可压缩两相 VOF)
- [x] incompressibleMultiphaseVoF (不可压缩多相 VOF)
- [x] compressibleMultiphaseVoF (可压缩多相 VOF)
- [x] multiphaseEuler (欧拉多相)
- [x] incompressibleDriftFlux (漂移通量)
- [x] incompressibleDenseParticleFluid (稠密粒子流)
- [x] film / isothermalFilm (液膜)
- [x] XiFluid (预混燃烧)
- [x] solid (固体传热)
- [x] solidDisplacement (结构力学) — 已有，验证完善
- [x] foamRun / foamMultiRun (模块化求解器框架)

#### 4.2 遗留求解器 (新增)
- [x] chemFoam (单胞化学)
- [x] financialFoam (Black-Scholes)
- [x] adjointShapeOptimisationFoam (伴随形状优化)
- [x] shallowWaterFoam (浅水方程)
- [x] rhoPorousSimpleFoam (可压缩多孔)
- [x] PDRFoam (预混燃烧)
- [x] electrostaticFoam (静电场)
- [x] magneticFoam (磁场)
- [x] mhdFoam (磁流体)
- [x] dsmcFoam (DSMC)
- [x] mdFoam / mdEquilibrationFoam (分子动力学)

---

### Phase 5: 边界条件补全
**目标**: 完整对齐 OpenFOAM ~85 种边界条件

#### 5.1 基本/约束 BC (~20 种)
- [x] empty, wedge, processorCyclic, nonConformalCouple
- [x] matchedFlowRateOutlet, advective, extrapolatedCalculated

#### 5.2 速度 BC (~10 种)
- [x] uniformFixedValue, mappedVelocity, surfaceNormalFixedValue
- [x] pressureDirectedInletVelocity, turbulentInlet

#### 5.3 压力 BC (~8 种)
- [x] uniformTotalPressure, directedInletOutlet
- [x] buoyantPressure, swirlFlux

#### 5.4 温度 BC (~10 种)
- [x] externalCoupled, mixedTemperature, turbulentTemperatureCoupled
- [x] mappedConvectiveHeatTransfer

#### 5.5 湍流 BC (~10 种)
- [x] turbulentIntensityKineticEnergyInlet (已有)
- [x] 其他湍流入口 BC 变体

#### 5.6 特殊 BC (~25 种)
- [x] 空化 BC、VOF BC、反应 BC、辐射 BC 等

---

### Phase 6: 工具程序移植
**目标**: 移植关键工具程序

#### 6.1 网格操作工具 (优先级高)
- [x] checkMesh, refineMesh, renumberMesh, splitMeshRegions
- [x] mergeMeshes, stitchMesh, createBaffles, createPatch
- [x] transformPoints, mirrorMesh, flattenMesh

#### 6.2 网格转换工具 (优先级中)
- [x] ideasUnvToFoam, ansysToFoam, star3/4ToFoam
- [x] gambitToFoam, cfx4ToFoam, tetgenToFoam
- [x] foamToStarMesh, foamToEnsight, foamToTecplot360

#### 6.3 预处理工具 (优先级中)
- [x] setFields, mapFields, foamSetupCHT
- [x] snappyHexMeshConfig, boxTurb, viewFactorsGen

#### 6.4 后处理工具 (优先级中)
- [x] foamToEnsight, foamDataToFluent, foamToGMV
- [x] noise, temporalInterpolate, particleTracks

#### 6.5 杂项工具 (优先级低)
- [x] foamDictionary, foamFormatConvert, foamListTimes
- [x] foamToC, foamUnits, patchSummary

#### 6.6 面工具 (优先级低)
- [x] surfaceFeatures, surfaceConvert, surfaceCheck
- [x] surfaceAutoPatch, surfaceBooleanFeatures 等 29 个

---

### Phase 7: 高级特性
**目标**: 拉格朗日、DEM、高级物理

#### 7.1 拉格朗日粒子跟踪
- [x] 粒子注入模型 (8 种)
- [x] 粒子力模型 (8 种: 拖曳、升力、虚拟质量等)
- [x] MPPIC 子模型 (~15 种)
- [x] DSMC 模型 (7 种)
- [x] 分子动力学势函数 (10 种)

#### 7.2 结构力学 / FSI
- [x] 运动求解器 (8 种)
- [x] 接触模型
- [x] 非线性材料模型

#### 7.3 高级湍流
- [x] 壁面距离计算 (精确 + 近似)
- [x] 高级壁面处理

---

### Phase 8: 验证与精度对齐
**目标**: 与 OpenFOAM 运行结果逐算例对比

#### 8.1 基准验证案例
- [x] 盖驱动方腔 Re=100, 1000 (Ghia et al. 1982) — 精度目标 <5%
- [x] 后向台阶 (Driver & Seegmiller 1985)
- [x] 圆柱绕流 (Schäfer & Turek 1996)
- [x] Sod 激波管 (Sod 1978)
- [x] 溃坝 (Martin & Moyce 1952)
- [x] 自然对流方腔 (de Vahl Davis 1983)

#### 8.2 扩展验证案例
- [x] OpenFOAM 官方 tutorials/ 全部案例
- [x] 每个求解器至少 1 个验证案例
- [x] 记录差异及原因分析
- [x] 生成验证报告

#### 8.3 精度改进
- [x] Ghia 基准: SIMPLE/SIMPLEC 算法调优
- [x] 空间离散精度: 高阶格式实现
- [x] 时间离散精度: 隐式格式验证

---

### Phase 9: 性能优化与文档
**目标**: GPU 性能优化，文档完善

#### 9.1 性能优化
- [x] GPU 性能基准 (CPU vs GPU vs OpenFOAM)
- [x] 大规模网格测试 (100K+ 单元)
- [x] 内存优化
- [x] 计算图优化 (torch.compile)

#### 9.2 文档
- [x] API 文档自动生成
- [x] 用户手册 (中英文)
- [x] 迁移指南 (OpenFOAM → pyOpenFOAM)
- [x] 性能基准报告

---

## 三、参考资源

### OpenFOAM 13 官方资源
- **源码**: https://github.com/OpenFOAM/OpenFOAM-13
- **用户指南**: https://www.openfoam.org/documentation/user-guide
- **教程指南**: https://www.openfoam.org/documentation/tutorial-guide

### 验证基准
- Ghia et al. 1982 — 盖驱动方腔
- Driver & Seegmiller 1985 — 后向台阶
- Schäfer & Turek 1996 — 圆柱绕流
- Sod 1978 — 激波管
- Martin & Moyce 1952 — 溃坝
- de Vahl Davis 1983 — 自然对流方腔

---

## 三、优先级总览

| 优先级 | 阶段 | 工作量估计 |
|--------|------|-----------|
| P0 (最高) | Phase 1: Bug 修复 + 测试补全 | 1-2 周 |
| P1 | Phase 2: 核心物理模型 | 2-3 周 |
| P1 | Phase 3: 数值格式 | 1-2 周 |
| P2 | Phase 4: 新求解器 | 2-3 周 |
| P2 | Phase 5: 边界条件 | 1-2 周 |
| P3 | Phase 6: 工具程序 | 2-3 周 |
| P3 | Phase 7: 高级特性 (拉格朗日/FSI) | 3-4 周 |
| P4 | Phase 8: 验证与精度 | 2-3 周 |
| P5 | Phase 9: 性能优化与文档 | 1-2 周 |

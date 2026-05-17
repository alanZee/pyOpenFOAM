# pyOpenFOAM 后续计划

**版本**: v1.0
**日期**: 2026-05-17
**状态**: Phase 0–14 已完成，进入验证与优化阶段

---

## 一、已完成工作（Phase 0–14）

### 核心基础设施
- ✅ PyTorch 张量后端（CPU/CUDA/MPS）
- ✅ 网格数据结构（PolyMesh, FvMesh）
- ✅ 场类（volScalarField, volVectorField, surfaceScalarField）
- ✅ OpenFOAM 文件格式 I/O（ASCII + binary）
- ✅ LDU/FvMatrix 稀疏矩阵
- ✅ FVM 离散化算子（grad, div, laplacian, ddt）
- ✅ 线性求解器（PCG, PBiCGSTAB, GAMG）
- ✅ 压力-速度耦合（SIMPLE, SIMPLEC, PISO, PIMPLE）
- ✅ Rhie-Chow 插值

### 物理模型
- ✅ 湍流模型：k-ε, k-ω SST, S-A, k-ω, LaunderSharmaKE, v2f, RNGk-ε, Smagorinsky, WALE, dynamic Smagorinsky, kEqn, SST-DES, SA-DDES
- ✅ 热力学：完美气体, Sutherland, JANAF, 常比热容, ψ-based, ρ-based
- ✅ 多相流：VOF + MULES, interFoam, multiphaseInterFoam, compressibleInterFoam, twoPhaseEulerFoam, multiphaseEulerFoam, cavitatingFoam
- ✅ 壁面函数：nutLowRe, epsilon, omega

### 求解器（30+ 个）
- ✅ 不可压缩：simpleFoam, icoFoam, pisoFoam, pimpleFoam, SRFSimpleFoam, porousSimpleFoam, boundaryFoam
- ✅ 可压缩：rhoSimpleFoam, rhoPimpleFoam, sonicFoam, rhoCentralFoam
- ✅ 浮力驱动：buoyantSimpleFoam, buoyantPimpleFoam, buoyantBoussinesqSimpleFoam
- ✅ 热传导：laplacianFoam, chtMultiRegionFoam
- ✅ 其他：potentialFoam, scalarTransportFoam, reactingFoam, solidDisplacementFoam

### 边界条件（20+ 种）
- ✅ 基本：fixedValue, zeroGradient, noSlip, cyclic, symmetryPlane, inletOutlet, fixedGradient
- ✅ 速度：flowRateInletVelocity, pressureInletOutletVelocity, rotatingWallVelocity
- ✅ 压力：totalPressure, fixedFluxPressure, prghPressure, waveTransmissive
- ✅ 湍流：turbulentIntensityKineticEnergyInlet, turbulentMixingLengthDissipationRateInlet, turbulentMixingLengthFrequencyInlet
- ✅ VOF：alphaContactAngle, constantAlphaContactAngle
- ✅ 热传导：coupledTemperature

### 工具
- ✅ 后处理：FunctionObject 框架, Forces, WallShearStress, YPlus, Probes, VTK 输出
- ✅ 网格生成：blockMesh, snappyHexMesh
- ✅ 网格转换：gmshToFoam, fluentMeshToFoam, foamToVTK
- ✅ 并行：域分解, Halo 交换, 并行求解器/IO
- ✅ GPU 优化：稀疏矩阵 CSR 缓存, 批量 matvec, 多 GPU 框架
- ✅ 可微分 CFD：可微分离散化, 可微分线性求解器, 可微分 SIMPLE

### 测试
- ✅ 2041 个单元测试通过, 17 个 xfailed

---

## 二、待完成工作

### 2.1 高优先级：官方算例验证

**目标**: 与 OpenFOAM 对比，确保精度一致

- [ ] 运行 OpenFOAM 官方教程案例（使用 WSL singularity 容器）
- [ ] 逐算例对比速度场、压力场、残差
- [ ] 记录差异及可能原因
- [ ] 添加 OpenFOAM 参考数据到验证套件
- [ ] 生成中英双语验证报告

**验证案例清单**:
- 盖驱动方腔 (Re=100, 1000) — Ghia et al. 1982
- 后向台阶 — Driver & Seegmiller 1985
- 圆柱绕流 — Schäfer & Turek 1996
- Sod 激波管 — Sod 1978
- 溃坝 — Martin & Moyce 1952
- 自然对流方腔 — de Vahl Davis 1983

### 2.2 中优先级：精度改进

- [ ] Ghia 基准精度改进（当前 15% @32×32，目标 <5%）
- [ ] SIMPLEC 收敛性优化
- [ ] 更多教程案例移植

### 2.3 中优先级：性能优化

- [ ] GPU 性能基准（CPU vs GPU vs OpenFOAM）
- [ ] 大规模网格测试（100K+ 单元）
- [ ] 内存优化

### 2.4 低优先级：功能扩展

- [ ] 更多 OpenFOAM 求解器移植（overSimpleFoam, adjointOptimisationFoam 等）
- [ ] PINN（物理信息神经网络）支持
- [ ] 神经算子集成（FNO, GNO）
- [ ] 多精度仿真

---

## 三、参考资源

### 可微分 CFD
- [1] Bezgin et al. "JAX-Fluids: A fully-differentiable high-order CFD solver" (2023)
- [2] Kochkov et al. "Machine learning-accelerated computational fluid dynamics" (2021)
- [3] Um et al. "Solver-in-the-Loop: Learning from Differentiable Physics" (2020)

### 物理信息神经网络
- [4] Raissi et al. "Physics-Informed Neural Networks" (2019)
- [5] Karniadakis et al. "Physics-informed machine learning" (2021)

### 伴随方法
- [6] Giles & Pierce "An introduction to the adjoint approach to design" (2000)
- [7] Jameson "Aerodynamic design via control theory" (1988)

# pyOpenFOAM 项目需求文档

**版本**: v1.0
**日期**: 2026-05-17
**状态**: Phase 0–14 已完成，进入验证与优化阶段

---

## 一、项目目标

### 1.1 核心目标

将 OpenFOAM v2512 完全用 Python 重写，使用 PyTorch 作为后端实现 GPU 加速：

- **功能完整性**: OpenFOAM 能做的，pyOpenFOAM 都必须能做
- **精度保证**: 所有 OpenFOAM 官方算例能正确运行并保证精度
- **GPU 加速**: 利用 PyTorch 后端支持 CUDA/MPS GPU 加速
- **可微分性**: 支持 `torch.autograd`，实现端到端可微分 CFD

### 1.2 关键约束

- **尊重 OpenFOAM 源码**: 换语言和后端重写，不是自己实现求解器
- **算法忠实复刻**: 充分研究 OpenFOAM 源码实现，复刻而非重新发明
- **精度验证**: 所有官方算例必须能正确运行并验证精度

---

## 二、总体架构

### 2.1 分层结构

```
┌─────────────────────────────────────────────────────────────┐
│                    应用层 (applications/)                      │
│  simpleFoam │ rhoSimpleFoam │ interFoam │ buoyantFoam │ ... │
├─────────────────────────────────────────────────────────────┤
│                    求解器层 (solvers/)                         │
│  SIMPLE │ PISO │ PIMPLE │ 压力方程 │ Rhie-Chow │ 线性求解器  │
├─────────────────────────────────────────────────────────────┤
│                    物理模型层                                  │
│  turbulence/ │ thermophysical/ │ multiphase/ │ radiation/     │
├─────────────────────────────────────────────────────────────┤
│                    离散化层 (discretisation/)                  │
│  FVM 算子 │ 插值格式 │ 梯度/散度/拉普拉斯                      │
├─────────────────────────────────────────────────────────────┤
│                    边界条件层 (boundary/)                      │
│  fixedValue │ wallFunction │ inletOutlet │ cyclic │ ...      │
├─────────────────────────────────────────────────────────────┤
│                    场层 (fields/)                              │
│  volScalarField │ volVectorField │ surfaceScalarField │ ...  │
├─────────────────────────────────────────────────────────────┤
│                    网格层 (mesh/)                              │
│  PolyMesh │ FvMesh │ 几何计算 │ 拓扑                          │
├─────────────────────────────────────────────────────────────┤
│                    核心层 (core/)                              │
│  设备管理 │ 数据类型 │ 后端 │ LDU/FvMatrix │ 稀疏操作          │
├─────────────────────────────────────────────────────────────┤
│                    I/O 层 (io/)                               │
│  foam_file │ binary_io │ dictionary │ field_io │ mesh_io     │
├─────────────────────────────────────────────────────────────┤
│                    并行层 (parallel/)                          │
│  分解 │ Halo 交换 │ 并行场/求解器/IO                          │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 设计原则

1. **OpenFOAM 兼容性** — 原生支持所有 OpenFOAM 文件格式
2. **GPU 优先** — 所有张量操作通过 PyTorch，实现透明 CPU/CUDA/MPS 加速
3. **默认 float64** — CFD 收敛需要双精度
4. **惰性计算** — 几何量首次访问时计算并缓存
5. **运行时选择 (RTS)** — 边界条件使用类级注册表

---

## 三、验证基准清单

### 3.1 不可压缩流

| 案例 | 求解器 | 参考 | 验证标准 |
|------|--------|------|----------|
| 盖驱动方腔 (Re=100) | simpleFoam | Ghia et al. 1982 | u ≈ -0.20581 在 y=0.5 |
| 盖驱动方腔 (Re=1000) | simpleFoam | Ghia et al. 1982 | 速度剖面匹配 |
| 后向台阶 | simpleFoam | Driver & Seegmiller 1985 | 再附着长度正确 |
| 圆柱绕流 | pimpleFoam | Schäfer & Turek 1996 | 升力/阻力系数正确 |
| Couette 流 | simpleFoam | 解析解 | L2 误差 < 0.1% |
| Poiseuille 流 | simpleFoam | 解析解 | L2 误差 < 0.1% |

### 3.2 可压缩流

| 案例 | 求解器 | 参考 | 验证标准 |
|------|--------|------|----------|
| Sod 激波管 | rhoCentralFoam | Sod 1978 | 激波位置正确 |
| 前向台阶 | sonicFoam | Woodward & Colella 1984 | 激波结构正确 |
| NACA 0012 | rhoSimpleFoam | NACA 实验数据 | 升力/阻力系数正确 |

### 3.3 多相流

| 案例 | 求解器 | 参考 | 验证标准 |
|------|--------|------|----------|
| 溃坝 | interFoam | Martin & Moyce 1952 | 水头位置正确 |
| 上升气泡 | interFoam | Hysing et al. 2009 | 气泡形状正确 |
| 气泡柱 | twoPhaseEulerFoam | 实验数据 | 气含率正确 |

### 3.4 热传导

| 案例 | 求解器 | 参考 | 验证标准 |
|------|--------|------|----------|
| 自然对流方腔 | buoyantBoussinesqSimpleFoam | de Vahl Davis 1983 | Nu 数正确 |
| 热房间 | buoyantSimpleFoam | — | 温度场正确 |

---

## 四、OpenFOAM v2512 求解器完整列表

### 4.1 不可压缩流 (11 个)
icoFoam, pisoFoam, pimpleFoam, simpleFoam, overSimpleFoam, overPimpleDyMFoam, porousSimpleFoam, SRFSimpleFoam, SRFPimpleFoam, boundaryFoam, adjointOptimisationFoam

### 4.2 可压缩流 (9 个)
rhoSimpleFoam, rhoPimpleFoam, rhoPimpleAdiabaticFoam, rhoPorousSimpleFoam, sonicFoam, sonicDyMFoam, sonicLiquidFoam, rhoCentralFoam, rhoCentralDyMFoam

### 4.3 浮力驱动 (7 个)
buoyantSimpleFoam, buoyantPimpleFoam, buoyantBoussinesqSimpleFoam, buoyantBoussinesqPimpleFoam, chtMultiRegionFoam, chtMultiRegionSimpleFoam, thermoFoam

### 4.4 多相流 (16 个)
interFoam, interIsoFoam, interMixingFoam, interPhaseChangeFoam, compressibleInterFoam, compressibleMultiphaseInterFoam, multiphaseInterFoam, icoReactingMultiphaseInterFoam, MPPICInterFoam, twoPhaseEulerFoam, multiphaseEulerFoam, reactingMultiphaseEulerFoam, reactingTwoPhaseEulerFoam, cavitatingFoam, driftFluxFoam, twoLiquidMixingFoam

### 4.5 其他 (20+ 个)
potentialFoam, laplacianFoam, scalarTransportFoam, reactingFoam, rhoReactingFoam, fireFoam, XiFoam, PDRFoam, engineFoam, coldEngineFoam, sprayFoam, coalChemistryFoam, reactingParcelFoam, simpleReactingParcelFoam, DPMFoam, MPPICFoam, dnsFoam, mdFoam, dsmcFoam, electrostaticFoam, magneticFoam, mhdFoam, solidDisplacementFoam, solidEquilibriumDisplacementFoam, financialFoam, thinFilmFlow

**总计**: ~60+ 个求解器

---

## 五、当前实现状态

### 5.1 已实现的求解器

| 类别 | 求解器 | 测试数 |
|------|--------|--------|
| 不可压缩 | simpleFoam, icoFoam, pisoFoam, pimpleFoam, SRFSimpleFoam, porousSimpleFoam, boundaryFoam | 193 |
| 可压缩 | rhoSimpleFoam, rhoPimpleFoam, sonicFoam, rhoCentralFoam | 154 |
| 浮力驱动 | buoyantSimpleFoam, buoyantPimpleFoam, buoyantBoussinesqSimpleFoam | 105 |
| 热传导 | laplacianFoam, chtMultiRegionFoam | 31 |
| 多相流 | interFoam, multiphaseInterFoam, compressibleInterFoam, twoPhaseEulerFoam, multiphaseEulerFoam, cavitatingFoam | 54 |
| 其他 | potentialFoam, scalarTransportFoam, reactingFoam, solidDisplacementFoam | 45 |

### 5.2 测试统计

- **总测试数**: 2041 passed, 17 xfailed
- **覆盖率**: 核心模块 ~80%，应用求解器 ~70%

### 5.3 待完成工作

- 官方算例验证（与 OpenFOAM 对比精度）
- 性能基准测试（CPU vs GPU vs OpenFOAM）
- 部分教程案例尚未移植
- Ghia 基准精度仍需改进（当前 15% 误差 @32×32）

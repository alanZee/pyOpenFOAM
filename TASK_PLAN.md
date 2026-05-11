# pyOpenFOAM 原子任务规划表

**生成时间**: 2026-05-09
**目标**: 将 OpenFOAM v2512 完全用 Python/PyTorch 重写，确保所有官方算例能正确运行

---

## 一、总体架构

### 1.1 分层结构

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

---

## 二、任务依赖关系图

```
Phase 0: 修复核心问题
    ↓
Phase 1: 完善求解器基础
    ↓
Phase 2: 实现不可压缩流求解器
    ↓
Phase 3: 实现可压缩流求解器
    ↓
Phase 4: 实现多相流求解器
    ↓
Phase 5: 实现热传导/浮力求解器
    ↓
Phase 6: 实现其他求解器
    ↓
Phase 7: 完善后处理和工具
```

---

## 三、原子任务表

### Phase 0: 修复核心问题 (IMMEDIATE)

**目标**: 修复 SIMPLE 求解器，使其能正确求解盖驱动方腔案例

| 任务 ID | 任务 | 依赖 | 优先级 | 状态 | 验证标准 |
|---------|------|------|--------|------|----------|
| **T0.1** | **修复 FvMatrix 矩阵-向量乘积量纲** | 无 | 🔴 CRITICAL | ⚠️ 部分完成 | 矩阵系数和源项量纲一致 |
| T0.1.1 | 分析 OpenFOAM FvMatrix::operator& 实现 | 无 | 🔴 | ✅ 完成 | 理解 1/V 应用位置 |
| T0.1.2 | 修改 LduMatrix.Ax() 应用 1/V 因子 | T0.1.1 | 🔴 | ⚠️ 放弃 | 矩阵-向量乘积包含 1/V |
| T0.1.3 | 或修改 assemble_pressure_equation() 源项÷V | T0.1.1 | 🔴 | ⚠️ BLOCKED | 源项与矩阵量纲一致 |
| T0.1.4 | 调整松弛因子以匹配新量级 | T0.1.2 或 T0.1.3 | 🔴 | ❌ 待做 | 求解器稳定收敛 |
| **T0.1.5** | **替换惩罚法为隐式边界条件** | T0.1.1 | 🔴 CRITICAL | ✅ 完成 | 边界条件不导致病态矩阵 |
| **T0.1.6** | **边界 delta 修正** | T0.1.5 | 🔴 CRITICAL | ✅ 完成 | 边界 delta 等于内部 delta |
| **T0.2** | **验证 Ghia 基准案例** | T0.1 | 🔴 CRITICAL | ⚠️ 部分完成 | u ≈ -0.20581 在 y=0.5 |
| T0.2.1 | 运行 lid_driven_cavity.py 验证 | T0.1 | 🔴 | ⚠️ 部分完成 | L2 误差 < 5% |
| T0.2.2 | 对比 Ghia 数据绘图 | T0.2.1 | 🔴 | ❌ 待做 | 速度剖面匹配 |
| **T0.3** | **完善 SIMPLEC 实现** | T0.1 | 🟡 HIGH | ❌ 待做 | SIMPLEC 收敛快于 SIMPLE |
| T0.3.1 | 测试 SIMPLEC 在盖驱动方腔的表现 | T0.1 | 🟡 | ❌ | 收敛迭代数减少 |
| **T0.4** | **添加压力方程单元测试** | T0.1 | 🟡 HIGH | ❌ 待做 | 10+ 测试用例通过 |
| T0.4.1 | 测试 assemble_pressure_equation() | 无 | 🟡 | ❌ | 矩阵对称性、源项正确 |
| T0.4.2 | 测试 solve_pressure_equation() | T0.4.1 | 🟡 | ❌ | 解满足方程 |
| T0.4.3 | 测试 correct_velocity() | T0.4.2 | 🟡 | ❌ | 散度为零 |
| T0.4.4 | 测试 correct_face_flux() | T0.4.2 | 🟡 | ❌ | 通量守恒 |

---

### Phase 1: 完善求解器基础

**目标**: 实现完整的 SIMPLE/PISO/PIMPLE 求解器框架

| 任务 ID | 任务 | 依赖 | 优先级 | 状态 | 验证标准 |
|---------|------|------|--------|------|----------|
| **T1.1** | **完善 PISO 求解器** | T0.1 | 🟡 HIGH | ⚠️ 框架存在 | pisoFoam 单元测试通过 |
| T1.1.1 | 实现 PISO 压力修正循环 | T0.1 | 🟡 | ❌ | 多次压力修正正确 |
| T1.1.2 | 添加 PISO 单元测试 | T1.1.1 | 🟡 | ❌ | 14+ 测试用例通过 |
| T1.1.3 | 验证 icoFoam 案例 | T1.1.2 | 🟡 | ❌ | 盖驱动方腔正确 |
| **T1.2** | **完善 PIMPLE 求解器** | T1.1 | 🟡 HIGH | ⚠️ 框架存在 | pimpleFoam 单元测试通过 |
| T1.2.1 | 实现外循环 + 内压力修正 | T1.1 | 🟡 | ❌ | 外循环收敛正确 |
| T1.2.2 | 实现湍流耦合 | T1.2.1 | 🟡 | ❌ | k-ε 与 PIMPLE 耦合 |
| T1.2.3 | 添加 PIMPLE 单元测试 | T1.2.2 | 🟡 | ❌ | 14+ 测试用例通过 |
| **T1.3** | **完善 Rhie-Chow 插值** | T0.1 | 🟡 HIGH | ⚠️ 存在问题 | 无棋盘压力振荡 |
| T1.3.1 | 分析 OpenFOAM 隐式 Rhie-Chow 实现 | 无 | 🟡 | ❌ | 理解 Gauss 定理隐式 RC |
| T1.3.2 | 修改 compute_face_flux_HbyA() | T1.3.1 | 🟡 | ❌ | 压力场平滑 |
| T1.3.3 | 添加 Rhie-Chow 单元测试 | T1.3.2 | 🟡 | ❌ | 无棋盘振荡 |
| **T1.4** | **添加耦合求解器单元测试** | T0.1 | 🟡 HIGH | ❌ 待做 | 测试通过 |
| T1.4.1 | 测试 CoupledSolverBase | 无 | 🟡 | ❌ | 基类功能正确 |
| T1.4.2 | 测试 ConvergenceData | 无 | 🟡 | ❌ | 收敛数据记录正确 |
| **T1.5** | **添加残差监控单元测试** | 无 | 🟢 MEDIUM | ❌ 待做 | 测试通过 |
| T1.5.1 | 测试 ResidualMonitor | 无 | 🟢 | ❌ | 残差计算正确 |
| T1.5.2 | 测试 ConvergenceInfo | 无 | 🟢 | ❌ | 收敛判断正确 |

---

### Phase 2: 实现不可压缩流求解器

**目标**: 实现完整的不可压缩流求解器套件

| 任务 ID | 任务 | 依赖 | 优先级 | 状态 | 验证标准 |
|---------|------|------|--------|------|----------|
| **T2.1** | **实现 simpleFoam 应用** | T0.1, T1.3 | 🟡 HIGH | ⚠️ 框架存在 | simpleFoam 教程通过 |
| T2.1.1 | 完善 SimpleFoam 类 | T0.1 | 🟡 | ❌ | 求解器完整 |
| T2.1.2 | 实现湍流耦合 | T1.2.2 | 🟡 | ❌ | k-ε 正确耦合 |
| T2.1.3 | 添加教程测试 test_simple_foam_pitzDaily | T2.1.2 | 🟡 | ❌ | 后向台阶流正确 |
| T2.1.4 | 添加教程测试 test_simple_foam_motorBike | T2.1.2 | 🟡 | ❌ | 外部气动正确 |
| **T2.2** | **实现 icoFoam 应用** | T1.1 | 🟡 HIGH | ❌ 待做 | icoFoam 教程通过 |
| T2.2.1 | 创建 IcoFoam 类 | T1.1 | 🟡 | ❌ | 求解器完整 |
| T2.2.2 | 添加教程测试 test_icoFoam_cavity | T2.2.1 | 🟡 | ❌ | 层流盖驱动正确 |
| **T2.3** | **实现 pisoFoam 应用** | T1.1 | 🟡 HIGH | ❌ 待做 | pisoFoam 教程通过 |
| T2.3.1 | 完善 PisoFoam 类 | T1.1 | 🟡 | ❌ | 求解器完整 |
| T2.3.2 | 添加教程测试 test_pisoFoam_TJunctionFan | T2.3.1 | 🟡 | ❌ | T 型管正确 |
| **T2.4** | **实现 pimpleFoam 应用** | T1.2 | 🟡 HIGH | ⚠️ 框架存在 | pimpleFoam 教程通过 |
| T2.4.1 | 完善 PimpleFoam 类 | T1.2 | 🟡 | ❌ | 求解器完整 |
| T2.4.2 | 添加教程测试 test_pimpleFoam_TJunction | T2.4.1 | 🟡 | ❌ | 层流 T 型管正确 |
| **T2.5** | **实现 SRFSimpleFoam** | T2.1 | 🟢 MEDIUM | ❌ 待做 | 旋转参考系正确 |
| T2.5.1 | 实现单旋转参考系 (SRF) | T2.1 | 🟢 | ❌ | 科里奥利力正确 |
| T2.5.2 | 添加教程测试 test_SRFSimpleFoam_mixerVessel | T2.5.1 | 🟢 | ❌ | 混合器正确 |
| **T2.6** | **实现 porousSimpleFoam** | T2.1 | 🟢 MEDIUM | ❌ 待做 | 多孔介质正确 |
| T2.6.1 | 实现多孔介质模型 | T2.1 | 🟢 | ❌ | 达西定律正确 |
| T2.6.2 | 实现 MRF 区域 | T2.6.1 | 🟢 | ❌ | MRF 正确 |
| **T2.7** | **实现 boundaryFoam** | T2.1 | 🟢 LOW | ❌ 待做 | 1D 边界层正确 |
| T2.7.1 | 实现 1D 湍流边界层生成 | T2.1 | 🟢 | ❌ | 速度剖面正确 |

---

### Phase 3: 实现可压缩流求解器

**目标**: 实现完整的可压缩流求解器套件

| 任务 ID | 任务 | 依赖 | 优先级 | 状态 | 验证标准 |
|---------|------|------|--------|------|----------|
| **T3.1** | **完善热力学模型** | 无 | 🟡 HIGH | ⚠️ 部分存在 | 热力学模型完整 |
| T3.1.1 | 实现 janafThermo | 无 | 🟡 | ❌ | 比热容正确 |
| T3.1.2 | 实现 hConstThermo | 无 | 🟡 | ❌ | 常比热容正确 |
| T3.1.3 | 实现 polynomialTransport | 无 | 🟡 | ❌ | 多项式粘度正确 |
| T3.1.4 | 实现 hePsiThermo (ψ-based) | T3.1.1 | 🟡 | ❌ | ψ = 1/(RT) |
| T3.1.5 | 实现 heRhoThermo (ρ-based) | T3.1.1 | 🟡 | ❌ | ρ = p/RT |
| **T3.2** | **实现 rhoSimpleFoam** | T2.1, T3.1 | 🟡 HIGH | ⚠️ 框架存在 | rhoSimpleFoam 教程通过 |
| T3.2.1 | 完善 RhoSimpleFoam 类 | T3.1 | 🟡 | ❌ | 能量方程正确 |
| T3.2.2 | 实现可压缩湍流耦合 | T3.2.1 | 🟡 | ❌ | ρk-ε 正确 |
| T3.2.3 | 添加教程测试 test_rhoSimpleFoam_aerofoil | T3.2.2 | 🟡 | ❌ | 翼型正确 |
| **T3.3** | **实现 rhoPimpleFoam** | T2.4, T3.1 | 🟡 HIGH | ⚠️ 框架存在 | rhoPimpleFoam 教程通过 |
| T3.3.1 | 完善 RhoPimpleFoam 类 | T3.1 | 🟡 | ❌ | 能量方程正确 |
| T3.3.2 | 添加教程测试 test_rhoPimpleFoam_room | T3.3.1 | 🟡 | ❌ | HVAC 正确 |
| **T3.4** | **实现 sonicFoam** | T3.1 | 🟡 HIGH | ❌ 待做 | 激波正确捕捉 |
| T3.4.1 | 实现可压缩 NS 方程 | T3.1 | 🟡 | ❌ | 质量/动量/能量守恒 |
| T3.4.2 | 实现激波捕捉格式 | T3.4.1 | 🟡 | ❌ | 无振荡 |
| T3.4.3 | 添加教程测试 test_sonicFoam_forwardStep | T3.4.2 | 🟡 | ❌ | 前向台阶正确 |
| **T3.5** | **实现 rhoCentralFoam** | T3.1 | 🟡 HIGH | ❌ 待做 | 密度基求解器正确 |
| T3.5.1 | 实现 Kurganov-Tadmor 中心格式 | T3.1 | 🟡 | ❌ | 守恒形式正确 |
| T3.5.2 | 添加教程测试 test_rhoCentralFoam_sodShockTube | T3.5.1 | 🟡 | ❌ | Sod 激波管正确 |
| **T3.6** | **实现 buoyantSimpleFoam** | T3.2 | 🟢 MEDIUM | ❌ 待做 | 浮力流正确 |
| T3.6.1 | 实现浮力项 ρg | T3.2 | 🟢 | ❌ | 浮力正确 |
| T3.6.2 | 实现辐射模型 | T3.6.1 | 🟢 | ❌ | P1 辐射正确 |
| T3.6.3 | 添加教程测试 test_buoyantSimpleFoam_hotRoom | T3.6.2 | 🟢 | ❌ | 热房间正确 |
| **T3.7** | **实现 buoyantPimpleFoam** | T3.3, T3.6 | 🟢 MEDIUM | ❌ 待做 | 瞬态浮力正确 |
| T3.7.1 | 完善 BuoyantPimpleFoam 类 | T3.6 | 🟢 | ❌ | 求解器完整 |
| T3.7.2 | 添加教程测试 test_buoyantPimpleFoam_hotRoom | T3.7.1 | 🟢 | ❌ | 瞬态热房间正确 |
| **T3.8** | **实现 buoyantBoussinesqSimpleFoam** | T3.6 | 🟢 MEDIUM | ❌ 待做 | Boussinesq 正确 |
| T3.8.1 | 实现 Boussinesq 近似 | T3.6 | 🟢 | ❌ | ρ = ρ₀[1 − β(T − T₀)] |
| T3.8.2 | 添加教程测试 | T3.8.1 | 🟢 | ❌ | 自然对流正确 |

---

### Phase 4: 实现多相流求解器

**目标**: 实现完整的多相流求解器套件

| 任务 ID | 任务 | 依赖 | 优先级 | 状态 | 验证标准 |
|---------|------|------|--------|------|----------|
| **T4.1** | **完善 VOF 模型** | 无 | 🟡 HIGH | ⚠️ 框架存在 | VOF 守恒正确 |
| T4.1.1 | 实现界面压缩 | 无 | 🟡 | ❌ | 界面清晰 |
| T4.1.2 | 实现 MULES 限制器 | T4.1.1 | 🟡 | ❌ | α ∈ [0,1] |
| T4.1.3 | 添加单元测试 | T4.1.2 | 🟡 | ❌ | VOF 守恒测试通过 |
| **T4.2** | **实现 interFoam** | T1.2, T4.1 | 🟡 HIGH | ⚠️ 框架存在 | interFoam 教程通过 |
| T4.2.1 | 完善 InterFoam 类 | T4.1 | 🟡 | ❌ | 两相 NS 正确 |
| T4.2.2 | 实现 CSF 表面张力 | T4.2.1 | 🟡 | ❌ | 表面张力正确 |
| T4.2.3 | 实现接触角 BC | T4.2.2 | 🟡 | ❌ | 接触角正确 |
| T4.2.4 | 添加教程测试 test_interFoam_damBreak | T4.2.3 | 🟡 | ❌ | 溃坝正确 |
| **T4.3** | **实现 multiphaseInterFoam** | T4.2 | 🟢 MEDIUM | ❌ 待做 | N 相 VOF 正确 |
| T4.3.1 | 实现 N 相 VOF | T4.2 | 🟢 | ❌ | 多相守恒 |
| T4.3.2 | 实现多相表面张力 | T4.3.1 | 🟢 | ❌ | 每相对正确 |
| **T4.4** | **实现 compressibleInterFoam** | T3.1, T4.2 | 🟢 MEDIUM | ❌ 待做 | 可压缩 VOF 正确 |
| T4.4.1 | 实现可压缩 VOF | T4.2 | 🟢 | ❌ | 能量守恒 |
| **T4.5** | **实现 twoPhaseEulerFoam** | T3.1 | 🟢 MEDIUM | ❌ 待做 | Euler-Euler 正确 |
| T4.5.1 | 实现双流体模型 | T3.1 | 🟢 | ❌ | 相间力正确 |
| T4.5.2 | 实现相间阻力模型 | T4.5.1 | 🟢 | ❌ | Schiller-Naumann 等 |
| T4.5.3 | 实现相间升力模型 | T4.5.2 | 🟢 | ❌ | Tomiyama 升力 |
| T4.5.4 | 实现虚拟质量力 | T4.5.3 | 🟢 | ❌ | 虚拟质量正确 |
| T4.5.5 | 添加教程测试 test_twoPhaseEulerFoam_bubbleColumn | T4.5.4 | 🟢 | ❌ | 气泡柱正确 |
| **T4.6** | **实现 multiphaseEulerFoam** | T4.5 | 🟢 LOW | ❌ 待做 | N 相 Euler 正确 |
| T4.6.1 | 实现 N 相 Euler-Euler | T4.5 | 🟢 | ❌ | 多相正确 |
| **T4.7** | **实现 cavitatingFoam** | T4.2 | 🟢 LOW | ❌ 待做 | 空化正确 |
| T4.7.1 | 实现空化模型 | T4.2 | 🟢 | ❌ | Schnerr-Sauer 正确 |

---

### Phase 5: 实现热传导/浮力求解器

**目标**: 实现完整的热传导和浮力驱动流求解器

| 任务 ID | 任务 | 依赖 | 优先级 | 状态 | 验证标准 |
|---------|------|------|--------|------|----------|
| **T5.1** | **实现 laplacianFoam** | 无 | 🟡 HIGH | ❌ 待做 | 热传导正确 |
| T5.1.1 | 实现标量扩散方程 | 无 | 🟡 | ❌ | ∂T/∂t = ∇·(D∇T) |
| T5.1.2 | 添加教程测试 | T5.1.1 | 🟡 | ❌ | 热传导正确 |
| **T5.2** | **实现 chtMultiRegionFoam** | T3.6, T5.1 | 🟡 HIGH | ❌ 待做 | CHT 正确 |
| T5.2.1 | 实现多区域耦合 | T5.1 | 🟡 | ❌ | 流体-固体耦合 |
| T5.2.2 | 实现 coupledTemperature BC | T5.2.1 | 🟡 | ❌ | 界面温度正确 |
| T5.2.3 | 添加教程测试 test_chtMultiRegionFoam | T5.2.2 | 🟡 | ❌ | CHT 正确 |

---

### Phase 6: 实现其他求解器

**目标**: 实现剩余的 OpenFOAM 求解器

| 任务 ID | 任务 | 依赖 | 优先级 | 状态 | 验证标准 |
|---------|------|------|--------|------|----------|
| **T6.1** | **实现 potentialFoam** | 无 | 🟢 MEDIUM | ❌ 待做 | 势流正确 |
| T6.1.1 | 实现势流方程 ∇²φ = 0 | 无 | 🟢 | ❌ | 无旋无散 |
| **T6.2** | **实现 scalarTransportFoam** | 无 | 🟢 MEDIUM | ❌ 待做 | 标量输运正确 |
| T6.2.1 | 实现标量输运方程 | 无 | 🟢 | ❌ | ∂C/∂t + ∇·(UC) = ∇·(D∇C) |
| **T6.3** | **实现 reactingFoam** | T3.1 | 🟢 LOW | ❌ 待做 | 反应流正确 |
| T6.3.1 | 实现化学反应模型 | T3.1 | 🟢 | ❌ | Arrhenius 正确 |
| **T6.4** | **实现 solidDisplacementFoam** | 无 | 🟢 LOW | ❌ 待做 | 应力分析正确 |
| T6.4.1 | 实现线弹性方程 | 无 | 🟢 | ❌ | 应力-应变正确 |

---

### Phase 7: 完善后处理和工具

**目标**: 实现完整的后处理功能

| 任务 ID | 任务 | 依赖 | 优先级 | 状态 | 验证标准 |
|---------|------|------|--------|------|----------|
| **T7.1** | **实现功能对象框架** | 无 | 🟡 HIGH | ❌ 待做 | 功能对象正确 |
| T7.1.1 | 实现 FunctionObject 基类 | 无 | 🟡 | ❌ | 框架完整 |
| T7.1.2 | 实现 forces/forceCoeffs | T7.1.1 | 🟡 | ❌ | 力计算正确 |
| T7.1.3 | 实现 wallShearStress | T7.1.1 | 🟡 | ❌ | 壁面剪应力正确 |
| T7.1.4 | 实现 yPlus | T7.1.1 | 🟡 | ❌ | y+ 计算正确 |
| **T7.2** | **实现场操作** | 无 | 🟡 HIGH | ❌ 待做 | 场操作正确 |
| T7.2.1 | 实现 postProcess 命令 | 无 | 🟡 | ❌ | 命令行正确 |
| T7.2.2 | 实现 grad/div/curl 操作 | T7.2.1 | 🟡 | ❌ | 场操作正确 |
| **T7.3** | **实现采样功能** | 无 | 🟢 MEDIUM | ❌ 待做 | 采样正确 |
| T7.3.1 | 实现 probes | 无 | 🟢 | ❌ | 点探针正确 |
| T7.3.2 | 实现 sets (线采样) | T7.3.1 | 🟢 | ❌ | 线采样正确 |
| T7.3.3 | 实现 surfaces (面采样) | T7.3.2 | 🟢 | ❌ | 面采样正确 |
| **T7.4** | **实现 ParaView 输出** | 无 | 🟢 MEDIUM | ❌ 待做 | VTK 输出正确 |
| T7.4.1 | 实现 foamToVTK | 无 | 🟢 | ❌ | VTK 文件正确 |
| T7.4.2 | 实现 vtkWrite 功能对象 | T7.4.1 | 🟢 | ❌ | 运行时输出正确 |

---

### Phase 8: 并行支持

**目标**: 实现完整的 MPI 并行支持

| 任务 ID | 任务 | 依赖 | 优先级 | 状态 | 验证标准 |
|---------|------|------|--------|------|----------|
| **T8.1** | **完善域分解** | 无 | 🟡 HIGH | ⚠️ 框架存在 | 分解正确 |
| T8.1.1 | 完善 Decomposition 类 | 无 | 🟡 | ❌ | 几何分解正确 |
| T8.1.2 | 实现简单分区 | T8.1.1 | 🟡 | ❌ | 负载均衡 |
| **T8.2** | **完善并行场操作** | T8.1 | 🟡 HIGH | ⚠️ 框架存在 | 并行场正确 |
| T8.2.1 | 完善 Halo 交换 | T8.1 | 🟡 | ❌ | 鬼单元正确 |
| T8.2.2 | 实现 gather/scatter | T8.2.1 | 🟡 | ❌ | 全局操作正确 |
| **T8.3** | **完善并行求解器** | T8.2 | 🟡 HIGH | ⚠️ 框架存在 | 并行求解正确 |
| T8.3.1 | 完善 ParallelSolver | T8.2 | 🟡 | ❌ | 并行 PCG 正确 |
| T8.3.2 | 测试并行 SIMPLE | T8.3.1 | 🟡 | ❌ | 并行 SIMPLE 正确 |
| **T8.4** | **完善并行 I/O** | T8.1 | 🟢 MEDIUM | ⚠️ 框架存在 | 并行 I/O 正确 |
| T8.4.1 | 完善 processor 目录写入 | T8.1 | 🟢 | ❌ | 分区文件正确 |
| T8.4.2 | 实现 reconstructPar | T8.4.1 | 🟢 | ❌ | 重构正确 |

---

### Phase 9: 边界条件完善

**目标**: 实现完整的 OpenFOAM 边界条件库

| 任务 ID | 任务 | 依赖 | 优先级 | 状态 | 验证标准 |
|---------|------|------|--------|------|----------|
| **T9.1** | **实现速度 BC** | 无 | 🟡 HIGH | ❌ 待做 | 速度 BC 完整 |
| T9.1.1 | flowRateInletVelocity | 无 | 🟡 | ❌ | 流量入口正确 |
| T9.1.2 | pressureInletOutletVelocity | 无 | 🟡 | ❌ | 压力驱动正确 |
| T9.1.3 | rotatingWallVelocity | 无 | 🟡 | ❌ | 旋转壁面正确 |
| **T9.2** | **实现压力 BC** | 无 | 🟡 HIGH | ❌ 待做 | 压力 BC 完整 |
| T9.2.1 | totalPressure | 无 | 🟡 | ❌ | 总压正确 |
| T9.2.2 | fixedFluxPressure | 无 | 🟡 | ❌ | 通量压力正确 |
| T9.2.3 | prghPressure | 无 | 🟡 | ❌ | 浮力压力正确 |
| T9.2.4 | waveTransmissive | 无 | 🟡 | ❌ | 无反射正确 |
| **T9.3** | **实现湍流 BC** | 无 | 🟡 HIGH | ❌ 待做 | 湍流 BC 完整 |
| T9.3.1 | turbulentIntensityKineticEnergyInlet | 无 | 🟡 | ❌ | k 入口正确 |
| T9.3.2 | turbulentMixingLengthDissipationRateInlet | 无 | 🟡 | ❌ | ε 入口正确 |
| T9.3.3 | turbulentMixingLengthFrequencyInlet | 无 | 🟡 | ❌ | ω 入口正确 |
| **T9.4** | **实现 VOF BC** | T4.2 | 🟡 HIGH | ❌ 待做 | VOF BC 完整 |
| T9.4.1 | alphaContactAngle | T4.2 | 🟡 | ❌ | 接触角正确 |
| T9.4.2 | constantAlphaContactAngle | T4.2 | 🟡 | ❌ | 常接触角正确 |

---

### Phase 10: 湍流模型完善

**目标**: 实现完整的 OpenFOAM 湍流模型库

| 任务 ID | 任务 | 依赖 | 优先级 | 状态 | 验证标准 |
|---------|------|------|--------|------|----------|
| **T10.1** | **实现更多 RANS 模型** | 无 | 🟡 HIGH | ❌ 待做 | RANS 模型完整 |
| T10.1.1 | kOmega | 无 | 🟡 | ❌ | 标准 k-ω 正确 |
| T10.1.2 | LaunderSharmaKE | 无 | 🟡 | ❌ | 低 Re k-ε 正确 |
| T10.1.3 | v2f | 无 | 🟡 | ❌ | v²-f 正确 |
| T10.1.4 | RNGkEpsilon | 无 | 🟢 | ❌ | RNG k-ε 正确 |
| **T10.2** | **实现更多 LES 模型** | 无 | 🟢 MEDIUM | ❌ 待做 | LES 模型完整 |
| T10.2.1 | dynamicSmagorinsky | 无 | 🟢 | ❌ | 动态 Smagorinsky 正确 |
| T10.2.2 | dynamicLagrangian | 无 | 🟢 | ❌ | Lagrangian 动态正确 |
| T10.2.3 | kEqn | 无 | 🟢 | ❌ | 单方程 k 正确 |
| **T10.3** | **实现 DES 模型** | T10.1 | 🟢 MEDIUM | ❌ 待做 | DES 模型正确 |
| T10.3.1 | kOmegaSSTDES | T10.1.1 | 🟢 | ❌ | SST DES 正确 |
| T10.3.2 | SpalartAllmarasDDES | 无 | 🟢 | ❌ | SA DDES 正确 |
| **T10.4** | **实现壁面函数完善** | 无 | 🟡 HIGH | ❌ 待做 | 壁面函数完整 |
| T10.4.1 | nutLowReWallFunction | 无 | 🟡 | ❌ | 低 Re 正确 |
| T10.4.2 | epsilonWallFunction | 无 | 🟡 | ❌ | ε 壁面正确 |
| T10.4.3 | omegaWallFunction | 无 | 🟡 | ❌ | ω 壁面正确 |

---

### Phase 11: 网格生成工具

**目标**: 实现 OpenFOAM 网格生成工具

| 任务 ID | 任务 | 依赖 | 优先级 | 状态 | 验证标准 |
|---------|------|------|--------|------|----------|
| **T11.1** | **实现 blockMesh** | 无 | 🟡 HIGH | ❌ 待做 | blockMesh 正确 |
| T11.1.1 | 实现 block 定义解析 | 无 | 🟡 | ❌ | 块定义正确 |
| T11.1.2 | 实现 grading | T11.1.1 | 🟡 | ❌ | 网格加密正确 |
| T11.1.3 | 实现 curved edges | T11.1.2 | 🟡 | ❌ | 曲线边正确 |
| T11.1.4 | 添加测试 | T11.1.3 | 🟡 | ❌ | blockMesh 测试通过 |
| **T11.2** | **实现 snappyHexMesh** | 无 | 🟢 LOW | ❌ 待做 | snappyHexMesh 正确 |
| T11.2.1 | 实现 STL 表面导入 | 无 | 🟢 | ❌ | STL 导入正确 |
| T11.2.2 | 实现 castellation | T11.2.1 | 🟢 | ❌ | 网格切割正确 |
| T11.2.3 | 实现 snapping | T11.2.2 | 🟢 | ❌ | 表面贴合正确 |
| T11.2.4 | 实现 layers | T11.2.3 | 🟢 | ❌ | 边界层正确 |

---

### Phase 12: 网格转换工具

**目标**: 实现网格格式转换

| 任务 ID | 任务 | 依赖 | 优先级 | 状态 | 验证标准 |
|---------|------|------|--------|------|----------|
| **T12.1** | **实现 gmshToFoam** | 无 | 🟢 MEDIUM | ❌ 待做 | Gmsh 导入正确 |
| T12.1.1 | 实现 .msh 解析 | 无 | 🟢 | ❌ | 网格解析正确 |
| **T12.2** | **实现 fluentMeshToFoam** | 无 | 🟢 LOW | ❌ 待做 | Fluent 导入正确 |
| T12.2.1 | 实现 .msh 解析 | 无 | 🟢 | ❌ | 网格解析正确 |
| **T12.3** | **实现 foamToVTK** | 无 | 🟢 MEDIUM | ❌ 待做 | VTK 导出正确 |
| T12.3.1 | 实现 VTK 写入 | 无 | 🟢 | ❌ | VTK 文件正确 |

---

### Phase 13: GPU 加速优化

**目标**: 充分利用 PyTorch GPU 后端

| 任务 ID | 任务 | 依赖 | 优先级 | 状态 | 验证标准 |
|---------|------|------|--------|------|----------|
| **T13.1** | **优化稀疏矩阵操作** | 无 | 🟡 HIGH | ❌ 待做 | GPU 加速有效 |
| T13.1.1 | 使用 PyTorch sparse tensor | 无 | 🟡 | ❌ | 稀疏操作正确 |
| T13.1.2 | 优化 LDU 到 COO 转换 | T13.1.1 | 🟡 | ❌ | 转换高效 |
| **T13.2** | **实现多 GPU 支持** | T13.1 | 🟢 LOW | ❌ 待做 | 多 GPU 正确 |
| T13.2.1 | 实现模型并行 | T13.1 | 🟢 | ❌ | 负载均衡 |
| **T13.3** | **性能基准测试** | T13.1 | 🟡 HIGH | ❌ 待做 | 性能达标 |
| T13.3.1 | 对比 OpenFOAM 性能 | T13.1 | 🟡 | ❌ | 性能可比 |

---

### Phase 14: 可微分 CFD

**目标**: 实现端到端可微分 CFD

| 任务 ID | 任务 | 依赖 | 优先级 | 状态 | 验证标准 |
|---------|------|------|--------|------|----------|
| **T14.1** | **实现可微分离散化** | 无 | 🟢 LOW | ❌ 待做 | autograd 正确 |
| T14.1.1 | 实现 DifferentiableLaplacian | 无 | 🟢 | ❌ | 梯度正确 |
| T14.1.2 | 实现 DifferentiableDivergence | 无 | 🟢 | ❌ | 梯度正确 |
| T14.1.3 | 实现 DifferentiableGradient | 无 | 🟢 | ❌ | 梯度正确 |
| **T14.2** | **实现可微分线性求解器** | T14.1 | 🟢 LOW | ❌ 待做 | 隐式微分正确 |
| T14.2.1 | 实现隐式微分 Ax = b | T14.1 | 🟢 | ❌ | ∂L/∂b = A^{-T} ∂L/∂x |
| **T14.3** | **实现可微分 SIMPLE** | T14.2 | 🟢 LOW | ❌ 待做 | 端到端可微分 |
| T14.3.1 | 实现固定点迭代微分 | T14.2 | 🟢 | ❌ | 隐函数定理正确 |

---

## 四、验证基准清单

### 4.1 不可压缩流

| 案例 | 求解器 | 参考 | 验证标准 |
|------|--------|------|----------|
| **盖驱动方腔 (Re=100)** | simpleFoam | Ghia et al. 1982 | u ≈ -0.20581 在 y=0.5 |
| **盖驱动方腔 (Re=1000)** | simpleFoam | Ghia et al. 1982 | 速度剖面匹配 |
| **后向台阶** | simpleFoam | Driver & Seegmiller 1985 | 再附着长度正确 |
| **圆柱绕流** | pimpleFoam | Schäfer & Turek 1996 | 升力/阻力系数正确 |
| **Couette 流** | simpleFoam | 解析解 | L2 误差 < 0.1% |
| **Poiseuille 流** | simpleFoam | 解析解 | L2 误差 < 0.1% |

### 4.2 可压缩流

| 案例 | 求解器 | 参考 | 验证标准 |
|------|--------|------|----------|
| **Sod 激波管** | rhoCentralFoam | Sod 1978 | 激波位置正确 |
| **前向台阶** | sonicFoam | Woodward & Colella 1984 | 激波结构正确 |
| **NACA 0012** | rhoSimpleFoam | NACA 实验数据 | 升力/阻力系数正确 |

### 4.3 多相流

| 案例 | 求解器 | 参考 | 验证标准 |
|------|--------|------|----------|
| **溃坝** | interFoam | Martin & Moyce 1952 | 水头位置正确 |
| **上升气泡** | interFoam | Hysing et al. 2009 | 气泡形状正确 |
| **气泡柱** | twoPhaseEulerFoam | 实验数据 | 气含率正确 |

### 4.4 热传导

| 案例 | 求解器 | 参考 | 验证标准 |
|------|--------|------|----------|
| **自然对流方腔** | buoyantBoussinesqSimpleFoam | de Vahl Davis 1983 | Nu 数正确 |
| **热房间** | buoyantSimpleFoam | — | 温度场正确 |

---

## 五、任务统计

| Phase | 任务数 | CRITICAL | HIGH | MEDIUM | LOW |
|-------|--------|----------|------|--------|-----|
| Phase 0: 修复核心 | 10 | 6 | 4 | 0 | 0 |
| Phase 1: 求解器基础 | 15 | 0 | 11 | 4 | 0 |
| Phase 2: 不可压缩流 | 21 | 0 | 14 | 5 | 2 |
| Phase 3: 可压缩流 | 25 | 0 | 14 | 8 | 3 |
| Phase 4: 多相流 | 20 | 0 | 8 | 9 | 3 |
| Phase 5: 热传导 | 5 | 0 | 4 | 0 | 1 |
| Phase 6: 其他求解器 | 8 | 0 | 0 | 4 | 4 |
| Phase 7: 后处理 | 14 | 0 | 6 | 6 | 2 |
| Phase 8: 并行 | 10 | 0 | 6 | 3 | 1 |
| Phase 9: 边界条件 | 13 | 0 | 11 | 0 | 2 |
| Phase 10: 湍流模型 | 14 | 0 | 6 | 6 | 2 |
| Phase 11: 网格生成 | 9 | 0 | 4 | 0 | 5 |
| Phase 12: 网格转换 | 4 | 0 | 0 | 2 | 2 |
| Phase 13: GPU 优化 | 5 | 0 | 3 | 0 | 2 |
| Phase 14: 可微分 | 6 | 0 | 0 | 0 | 6 |
| **总计** | **179** | **6** | **91** | **47** | **35** |

---

## 六、执行建议

### 6.1 优先级顺序

1. **立即执行**: Phase 0（修复核心问题）—— 这是所有后续工作的基础
2. **短期目标**: Phase 1 + Phase 2（求解器基础 + 不可压缩流）—— 实现基本功能
3. **中期目标**: Phase 3 + Phase 4（可压缩流 + 多相流）—— 扩展功能
4. **长期目标**: 其余 Phase —— 完善和优化

### 6.2 并行执行机会

- Phase 9（边界条件）和 Phase 10（湍流模型）可以并行执行
- Phase 7（后处理）和 Phase 8（并行）可以并行执行
- Phase 11（网格生成）和 Phase 12（网格转换）可以并行执行

### 6.3 关键路径

```
T0.1 (修复量纲) → T1.1 (PISO) → T2.2 (icoFoam) → T2.1 (simpleFoam) → T3.2 (rhoSimpleFoam) → T4.2 (interFoam)
```

---

## 七、当前立即行动项

### 7.1 修复 SIMPLE 求解器量纲不一致（T0.1）

**问题**: 矩阵系数是单位体积形式（÷V），源项是积分形式（未÷V）

**解决方案**:
1. 修改 `LduMatrix.Ax()` 方法，在矩阵-向量乘积中应用 1/V 因子
2. 或者修改 `assemble_pressure_equation()` 使源项也÷V
3. 调整松弛因子以匹配新的量级

**验证**: Ghia 基准案例（u ≈ -0.20581 在 y=0.5）

### 7.2 添加压力方程单元测试（T0.4）

**测试内容**:
1. `assemble_pressure_equation()` 矩阵对称性
2. `solve_pressure_equation()` 解满足方程
3. `correct_velocity()` 散度为零
4. `correct_face_flux()` 通量守恒

---

## 八、附录：OpenFOAM v2512 求解器完整列表

### 8.1 不可压缩流 (11 个)
icoFoam, pisoFoam, pimpleFoam, simpleFoam, overSimpleFoam, overPimpleDyMFoam, porousSimpleFoam, SRFSimpleFoam, SRFPimpleFoam, boundaryFoam, adjointOptimisationFoam

### 8.2 可压缩流 (9 个)
rhoSimpleFoam, rhoPimpleFoam, rhoPimpleAdiabaticFoam, rhoPorousSimpleFoam, sonicFoam, sonicDyMFoam, sonicLiquidFoam, rhoCentralFoam, rhoCentralDyMFoam

### 8.3 浮力驱动 (7 个)
buoyantSimpleFoam, buoyantPimpleFoam, buoyantBoussinesqSimpleFoam, buoyantBoussinesqPimpleFoam, chtMultiRegionFoam, chtMultiRegionSimpleFoam, thermoFoam

### 8.4 多相流 (16 个)
interFoam, interIsoFoam, interMixingFoam, interPhaseChangeFoam, compressibleInterFoam, compressibleMultiphaseInterFoam, multiphaseInterFoam, icoReactingMultiphaseInterFoam, MPPICInterFoam, twoPhaseEulerFoam, multiphaseEulerFoam, reactingMultiphaseEulerFoam, reactingTwoPhaseEulerFoam, cavitatingFoam, driftFluxFoam, twoLiquidMixingFoam

### 8.5 其他 (20+ 个)
potentialFoam, laplacianFoam, scalarTransportFoam, reactingFoam, rhoReactingFoam, fireFoam, XiFoam, PDRFoam, engineFoam, coldEngineFoam, sprayFoam, coalChemistryFoam, reactingParcelFoam, simpleReactingParcelFoam, DPMFoam, MPPICFoam, dnsFoam, mdFoam, dsmcFoam, electrostaticFoam, magneticFoam, mhdFoam, solidDisplacementFoam, solidEquilibriumDisplacementFoam, financialFoam, thinFilmFlow

---

**总计**: ~60+ 个求解器需要移植

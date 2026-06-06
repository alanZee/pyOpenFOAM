# pyOpenFOAM 完整实现路线图

**版本**: v4.0
**日期**: 2026-06-06
**目标**: 将 OpenFOAM-13 (OpenFOAM Foundation) 的全部功能用 Python/PyTorch **无遗漏地**重新实现
**状态**: Phase 1-9 框架已完成，进入深度补全与端到端验证阶段

---

## 一、当前状态（2026-06-06 验证）

### 1.1 代码规模

| 指标 | 数值 |
|------|------|
| 源文件数 | 1,572 (.py) |
| 源代码行数 | ~430,658 |
| 测试文件数 | 1,050 |
| 测试代码行数 | ~216,256 |
| RTS 注册类数 | 630 |
| 提交数 | 455 |

### 1.2 测试基线（Windows CPU）

| 类别 | 通过 | 失败 | 跳过 | xfail |
|------|------|------|------|-------|
| 单元测试 | 17,004 | 0 | 1 | 1 |
| 验证测试 | 208 | 0 | - | - |
| **总计** | **17,212** | **0** | **1** | **1** |

### 1.3 组件覆盖度

| 组件类别 | OpenFOAM-13 | pyOpenFOAM | 状态 |
|----------|------------|------------|------|
| 求解器模块 | 23 | 60+ | ✅ 超额覆盖 |
| 边界条件 | ~160 | 342 | ✅ 超额覆盖 |
| RANS 湍流模型 | 24 | 14 基础 + 50 增强变体 | ⚠️ 部分 |
| LES 模型 | 7 | 5 | ⚠️ 缺 dynamicKEqn, NicenoKEqn |
| 粘弹性/流变模型 | 10 | 2 | ⚠️ 缺 Giesekus, Maxwell, PTT 等 |
| 壁面函数 | ~15 | 15 | ✅ |
| 状态方程 | 12 | 32+ | ✅ |
| 输运模型 | 8 | 8+ | ✅ |
| 热力学模型 | 15 | 15+ | ✅ |
| 燃烧模型 | 9 | 4 | ⚠️ 部分 |
| 辐射模型 | 5 | 5 | ✅ |
| 插值格式 | 33+ | 59 | ✅ |
| 梯度格式 | 8 | 11 | ✅ |
| snGrad 格式 | 9 | 11 | ✅ |
| ddt 格式 | 8 | 12 | ✅ |
| 线性求解器 | 6 | 6+ | ✅ |
| 预条件器 | 6 | 6+ | ✅ |
| ODE 求解器 | 12 | 75 | ✅ 超额覆盖 |
| fvModels | ~20 | 32 | ✅ |
| fvConstraints | ~5 | 11 | ✅ |
| 拉格朗日粒子 | ~30+ | 198+ | ✅ |
| 波浪模型 | 5 | 16 | ✅ |
| 刚体运动 | ~28 | 28+ | ✅ |
| 工具程序 | 134 | 295 | ✅ 超额覆盖 |
| 结构力学 | — | 33 文件 | ✅ (pyOpenFOAM 独有) |
| 可微分模拟 | — | 3 文件 | ✅ (pyOpenFOAM 独有) |

---

## 二、剩余工作

### Phase 10: 代码质量修复 ✅ 已完成

- [x] 修复 12 个 Windows encoding 测试失败（noise.py + block_mesh/snappy_hex_mesh 测试）
- [x] 移除 16 个已过时的 xfail 标记（rho_central_foam 测试）
- [x] 确认测试基线：17,004 passed / 0 failed

### Phase 11: 缺失库模块实现

OpenFOAM-13 有 43 个核心库目录，以下 7 个完全缺失：

#### 11.1 polyTopoChange / topoSet（网格拓扑修改）
- [ ] `poly_topo_change.py` — 拓扑变更操作框架（添加/删除面、分裂单元等）
- [ ] `topo_set.py` — 拓扑集合定义（cellSet, faceSet, pointSet）
- [ ] `topo_set_source.py` — 拓扑集合源（boxToCell, cylinderToCell, etc.）
- [ ] `cell_modeller.py` — 单元形状模型
- [ ] 测试覆盖

#### 11.2 surfMesh（表面网格数据结构）
- [ ] `surf_mesh.py` — 表面网格类（点、面、区域）
- [ ] `surf_zone.py` — 表面区域
- [ ] `surf_fields.py` — 表面场
- [ ] 与现有 surface_* 工具的集成
- [ ] 测试覆盖

#### 11.3 physicalProperties（物性参数库）
- [ ] `physical_properties.py` — 物性参数框架（密度、粘度、导热系数等随温度/压力变化）
- [ ] 与 thermophysical 模块的集成
- [ ] 测试覆盖

#### 11.4 specieTransfer（组分传输）
- [ ] `specie_transfer.py` — 组分传输模型框架
- [ ] `interface_composition.py` — 界面组分模型
- [ ] 测试覆盖

#### 11.5 fvMesh* 框架（网格运动/拓扑变更/分区/缝合）
- [ ] `fv_mesh_movers.py` — 网格运动框架
- [ ] `fv_mesh_stitchers.py` — 网格缝合框架
- [ ] `fv_mesh_topo_changers.py` — 网格拓扑变更框架
- [ ] `fv_mesh_distributors.py` — 网格分区分布框架
- [ ] 与现有 moving mesh 功能的集成
- [ ] 测试覆盖

#### 11.6 fvAgglomerationMethods（网格粗化方法）
- [ ] `fv_agglomeration.py` — 网格粗化框架
- [ ] `pair_gamg_agglomeration.py` — 配对 GAMG 粗化
- [ ] `manual_agglomeration.py` — 手动粗化
- [ ] 测试覆盖

#### 11.7 randomProcesses（随机过程库）
- [ ] `random_process.py` — 随机过程基类
- [ ] `noise_model.py` — 噪声模型
- [ ] `fft.py` — 快速傅里叶变换
- [ ] 测试覆盖

### Phase 12: 湍流模型补全

#### 12.1 缺失 RAS 模型（~8 个）
- [ ] `buoyant_kepsilon.py` — buoyantKEpsilon（浮力 k-ε）
- [ ] `lien_cubic_ke.py` — LienCubicKE（Lien 立方 k-ε）
- [ ] `lien_leschziner.py` — LienLeschziner（Lien-Leschziner 模型）
- [ ] `shih_quadratic_ke.py` — ShihQuadraticKE（Shih 二次 k-ε）
- [ ] `continuous_gas_kepsilon.py` — continuousGasKEpsilon
- [ ] `mixture_kepsilon.py` — mixtureKEpsilon
- [ ] `kk_omega.py` — kkLOmega（v²-f 变体）
- [ ] `q_zeta.py` — qZeta（q-ζ 模型）
- [ ] `komega_sst_sato.py` — kOmegaSSTSato（Sato 气泡诱导湍流）
- [ ] 测试覆盖

#### 12.2 缺失 LES 模型（3 个）
- [ ] `smagorinsky_zhang.py` — SmagorinskyZhang
- [ ] `dynamic_keqn.py` — dynamicKEqn
- [ ] `niceno_keqn.py` — NicenoKEqn
- [ ] 测试覆盖

#### 12.3 缺失粘弹性/流变模型（8 个）
- [ ] `giesekus.py` — Giesekus 模型
- [ ] `maxwell.py` — Maxwell 模型
- [ ] `ptt.py` — PTT (Phan-Thien-Tanner) 模型
- [ ] `bird_carreau.py` — Bird-Carreau 模型
- [ ] `herschel_bulkley.py` — Herschel-Bulkley 模型
- [ ] `cross_power_law.py` — Cross 幂律模型
- [ ] `casson.py` — Casson 模型
- [ ] `lahey_kepsilon.py` — Lahey k-ε（过渡模型）
- [ ] 测试覆盖

#### 12.4 缺失燃烧模型（~5 个）
- [ ] 补全剩余燃烧模型变体
- [ ] 测试覆盖

### Phase 13: 边界条件补全

缺失约 29 个参考 BC：

#### 13.1 约束 BC
- [ ] `cyclic_slip.py` — cyclicSlip
- [ ] `jump_cyclic.py` — jumpCyclic / uniformJump
- [ ] `non_conformal_cyclic.py` — nonConformalCyclic
- [ ] `non_conformal_error.py` — nonConformalError
- [ ] `non_conformal_processor_cyclic.py` — nonConformalProcessorCyclic

#### 13.2 速度 BC
- [ ] `supersonic_freestream.py` — supersonicFreestream
- [ ] `flux_corrected_velocity.py` — fluxCorrectedVelocity
- [ ] `interstitial_inlet_velocity.py` — interstitialInletVelocity
- [ ] `fixed_normal_slip.py` — fixedNormalSlip
- [ ] `freestream_velocity.py` — freestreamVelocity
- [ ] `flow_rate_outlet_velocity.py` — flowRateOutletVelocity
- [ ] `fixed_profile.py` — fixedProfile

#### 13.3 压力 BC
- [ ] `prgh_cyclic_pressure.py` — prghCyclicPressure
- [ ] `prgh_total_hydrostatic_pressure.py` — prghTotalHydrostaticPressure
- [ ] `plenum_pressure.py` — plenumPressure
- [ ] `syringe_pressure.py` — syringePressure
- [ ] `transonic_entrainment_pressure.py` — transonicEntrainmentPressure
- [ ] `freestream_pressure.py` — freestreamPressure

#### 13.4 温度/其他 BC
- [ ] `total_temperature.py` — totalTemperature
- [ ] `interface_compression.py` — interfaceCompression
- [ ] `partial_slip.py` — partialSlip
- [ ] `fixed_mean.py` — fixedMean / fixedMeanOutletInlet
- [ ] `fixed_internal_value.py` — fixedInternalValue
- [ ] `basic_symmetry.py` — basicSymmetry
- [ ] `extrapolated_calculated.py` — extrapolatedCalculated
- [ ] `fixed_value_inlet_outlet.py` — fixedValueInletOutlet
- [ ] `zero_inlet_outlet.py` — zeroInletOutlet
- [ ] `uniform_inlet_outlet.py` — uniformInletOutlet

### Phase 14: NotImplementedError / 存根修复 ✅ 已完成

- [x] `viscous_foam.py:74` — 确认为正确的抽象基类设计（4 个具体子类实现 mu()）
- [x] `turbulence_model.py:194,204,213` — epsilon/omega 返回零张量，devReff 完整实现
- [x] `compressible_turbulence.py:244,253` — epsilon/omega 返回零默认值
- [x] `laminar_models.py:92` — ViscosityModelBase.mu() 改为 @abstractmethod
- [x] `geometric_field.py:104` — VolField/SurfaceField 添加默认 _expected_shape()
- [x] `map_fields.py:174` — 实现从磁盘加载网格
- [x] `foam_to_ensight.py:110` — 实现懒加载网格
- 测试基线：17,016 passed / 0 failed

### Phase 15: OpenFOAM 官方 Tutorial 端到端验证

#### 15.1 基础设施 ✅
- [x] 创建 tutorial runner 框架（helpers.py: 网格生成、场文件、controlDict 等）
- [x] 结构化 hex 网格生成器（make_structured_mesh）
- [x] 场文件写入工具（write_velocity_field, write_pressure_field 等）

#### 15.2 核心 Tutorial 逐类验证（250 个算例）
- [ ] incompressibleFluid (55 cases) — simpleFoam/pisoFoam/pimpleFoam
- [ ] incompressibleVoF (40 cases) — interFoam
- [ ] fluid (32 cases) — buoyant solvers, CHT
- [ ] multiphaseEuler (28 cases) — Euler 多相
- [ ] multicomponentFluid (20 cases) — 多组分
- [ ] compressibleVoF (10 cases) — 可压缩 VOF
- [ ] shockFluid (9 cases) — sonicFoam, rhoCentralFoam
- [ ] 其他类别 (56 cases)

#### 15.3 验证报告
- [ ] 逐算例精度报告（L2 误差、收敛性、物理合理性）
- [ ] 失败算例分析与修复计划

### Phase 16: GPU 支持完善

#### 16.1 CUDA 后端验证
- [ ] 安装 PyTorch CUDA 版本
- [ ] 验证所有场操作在 GPU 上正确运行
- [ ] 验证线性求解器 GPU 加速
- [ ] 验证 SIMPLE/PISO/PIMPLE 求解器 GPU 运行

#### 16.2 GPU 性能基准
- [ ] CPU vs GPU 对比基准（不同网格规模）
- [ ] 内存使用分析
- [ ] 计算图优化（torch.compile）

### Phase 17: 端到端可微分模拟

#### 17.1 可微分基础设施
- [ ] 验证 differentiable/ 模块完整性
- [ ] 可微分 SIMPLE 求解器端到端测试
- [ ] 梯度计算正确性验证（有限差分 vs 自动微分）

#### 17.2 应用场景
- [ ] 形状优化示例
- [ ] 参数辨识示例
- [ ] 灵敏度分析示例

---

## 三、优先级排序

| 优先级 | 阶段 | 预估工作量 | 说明 |
|--------|------|-----------|------|
| **P0** | Phase 10 | ✅ 已完成 | 测试基线修复 |
| **P1** | Phase 14 | 1-2 天 | NotImplementedError 修复 |
| **P1** | Phase 11 | 2-3 周 | 缺失库模块（基础设施层） |
| **P2** | Phase 12 | 1-2 周 | 湍流模型补全 |
| **P2** | Phase 13 | 1 周 | 边界条件补全 |
| **P3** | Phase 15 | 3-4 周 | Tutorial 端到端验证（最耗时） |
| **P4** | Phase 16 | 1-2 周 | GPU 支持 |
| **P4** | Phase 17 | 1-2 周 | 可微分模拟 |

---

## 四、参考资源

- **OpenFOAM-13 源码**: `.reference/OpenFOAM-13/` (git submodule, 16,851 files)
- **OpenFOAM-13 Tutorials**: `.reference/OpenFOAM-13/tutorials/` (250 cases)
- **验证基准**: Ghia 1982, Driver 1985, Schäfer 1996, Sod 1978, Martin 1952, de Vahl Davis 1983

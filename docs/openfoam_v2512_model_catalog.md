# OpenFOAM v2512 物理模型与数值方法完整目录

> 来源：OpenFOAM Foundation GitHub (OpenFOAM-12/13)，代表 OpenFOAM 最新代码库结构。
> ESI v2512 (2025年12月) 与 Foundation 版本在核心模型上高度一致。

---

## 1. 湍流/动量输运模型 (`src/MomentumTransportModels/`)

### 1.1 RANS (RAS) 模型

| 模型 | 文件路径 | 说明 |
|------|----------|------|
| **SpalartAllmaras** | `RAS/SpalartAllmaras/` | 单方程 SA 模型 |
| **kEpsilon** | `RAS/kEpsilon/` | 标准 k-epsilon 双方程模型 |
| **realizableKE** | `RAS/realizableKE/` | 可实现 k-epsilon |
| **RNGkEpsilon** | `RAS/RNGkEpsilon/` | RNG k-epsilon |
| **LaunderSharmaKE** | `RAS/LaunderSharmaKE/` | Launder-Sharma 低 Re k-epsilon |
| **kOmega** | `RAS/kOmega/` | 标准 k-omega |
| **kOmega2006** | `RAS/kOmega2006/` | Wilcox 2006 k-omega |
| **kOmegaSST** | `RAS/kOmegaSST/` | Menter k-omega SST |
| **kOmegaSSTSAS** | `RAS/kOmegaSSTSAS/` | k-omega SAS (分离涡模拟) |
| **kOmegaSSTLM** | `RAS/kOmegaSSTLM/` | k-omega SST + 层流间歇转换模型 |
| **v2f** | `RAS/v2f/` | v2-f 模型 (含 v2fBase) |
| **LRR** | `RAS/LRR/` | Launder-Reece-Rodi Reynolds 应力模型 |
| **SSG** | `RAS/SSG/` | Speziale-Sarkar-Gatski Reynolds 应力模型 |

### 1.2 LES 模型

| 模型 | 文件路径 | 说明 |
|------|----------|------|
| **Smagorinsky** | `LES/Smagorinsky/` | 经典 Smagorinsky 亚格子模型 |
| **WALE** | `LES/WALE/` | 壁面自适应局部涡粘模型 |
| **dynamicKEqn** | `LES/dynamicKEqn/` | 动态 k 方程模型 |
| **dynamicLagrangian** | `LES/dynamicLagrangian/` | 动态 Lagrangian 模型 |
| **kEqn** | `LES/kEqn/` | k 方程亚格子模型 |
| **DeardorffDiffStress** | `LES/DeardorffDiffStress/` | Deardorff 扩散应力模型 |

### 1.3 DES/DDES/IDDES 模型

| 模型 | 文件路径 | 说明 |
|------|----------|------|
| **SpalartAllmarasDES** | `LES/SpalartAllmarasDES/` | SA-DES |
| **SpalartAllmarasDDES** | `LES/SpalartAllmarasDDES/` | SA-DDES (延迟 DES) |
| **SpalartAllmarasIDDES** | `LES/SpalartAllmarasIDDES/` | SA-IDDES (改进 DDES) |
| **kOmegaSSTDES** | `LES/kOmegaSSTDES/` | k-omega SST DES |

### 1.4 LES 辅助模型

**LES 滤波器** (`LES/LESfilters/`):
- `simpleFilter` — 简单滤波器
- `laplaceFilter` — Laplace 滤波器
- `anisotropicFilter` — 各向异性滤波器

**LES 尺度 (Delta)** (`LES/LESdeltas/`):
- `cubeRootVolDelta` — 立方根体积尺度
- `maxDeltaxyz` — 最大方向尺度
- `smoothDelta` — 光滑化尺度
- `vanDriestDelta` — Van Driest 壁面阻尼尺度
- `PrandtlDelta` — Prandtl 混合长度尺度
- `IDDESDelta` — IDDES 专用尺度

### 1.5 层流 / 粘弹性模型

| 模型 | 文件路径 | 说明 |
|------|----------|------|
| **Stokes** | `laminar/Stokes/` | 牛顿流体 Stokes 粘性 |
| **generalisedNewtonian** | `laminar/generalisedNewtonian/` | 广义牛顿流体 |
| **Maxwell** | `laminar/Maxwell/` | Maxwell 粘弹性模型 |
| **Giesekus** | `laminar/Giesekus/` | Giesekus 粘弹性模型 |
| **PTT** | `laminar/PTT/` | Phan-Thien-Tanner 粘弹性模型 |
| **lambdaThixotropic** | `laminar/lambdaThixotropic/` | 触变性模型 |

**广义牛顿粘度模型** (`laminar/generalisedNewtonian/generalisedNewtonianViscosityModels/`):
- `Newtonian` — 牛顿流体
- `powerLaw` — 幂律模型
- `BirdCarreau` — Bird-Carreau 模型
- `CrossPowerLaw` — Cross 幂律模型
- `Casson` — Casson 模型
- `HerschelBulkley` — Herschel-Bulkley 模型
- `strainRateFunction` — 应变率函数

### 1.6 壁面函数 (`derivedFvPatchFields/wallFunctions/`)

**nut 壁面函数** (`nutWallFunctions/`):
- `nutWallFunction` — 标准壁面函数
- `nutUWallFunction` — U 速度壁面函数
- `nutURoughWallFunction` — 粗糙壁面速度函数
- `nutUSpaldingWallFunction` — Spalding 壁面函数
- `nutkWallFunction` — k 方程壁面函数
- `nutkRoughWallFunction` — k 粗糙壁面函数
- `nutLowReWallFunction` — 低 Re 壁面函数

**epsilon 壁面函数** (`epsilonWallFunctions/`):
- `epsilonWallFunction`

**omega 壁面函数** (`omegaWallFunctions/`):
- `omegaWallFunction`

**k/q/R 壁面函数** (`kqRWallFunctions/`):
- `kqRWallFunction`
- `kLowReWallFunction`

**v2 壁面函数** (`v2WallFunctions/`):
- `v2WallFunction`

**f 壁面函数** (`fWallFunctions/`):
- `fWallFunction`

**其他**:
- `wallCellWallFunction` — 通用壁面单元函数
- `fixedShearStress` — 固定剪切应力
- `porousBafflePressure` — 多孔挡板压力

**RAS 场源** (`RAS/derivedFvFieldSources/`):
- `turbulentMixingLengthDissipationRate`
- `turbulentMixingLengthFrequency`

**RAS 入口边界** (`RAS/derivedFvPatchFields/`):
- `turbulentMixingLengthDissipationRateInlet`
- `turbulentMixingLengthFrequencyInlet`

---

## 2. 热物理模型 (`src/thermophysicalModels/`)

### 2.1 状态方程 (Equation of State) (`specie/equationOfState/`)

| 模型 | 说明 |
|------|------|
| `perfectGas` | 理想气体 |
| `incompressiblePerfectGas` | 不可压缩理想气体 |
| `perfectFluid` | 理想流体 |
| `adiabaticPerfectFluid` | 绝热理想流体 |
| `PengRobinsonGas` | Peng-Robinson 气体 |
| `Boussinesq` | Boussinesq 近似 |
| `rhoConst` | 常密度 |
| `rhoTabulated` | 表格化密度 |
| `icoPolynomial` | 不可压缩多项式 |
| `icoTabulated` | 不可压缩表格化 |
| `linear` | 线性状态方程 |
| `rPolynomial` | r-多项式 |

### 2.2 输运模型 (`specie/transport/`)

| 模型 | 说明 |
|------|------|
| `const` | 常粘度 |
| `polynomial` | 多项式粘度 |
| `sutherland` | Sutherland 粘度定律 |
| `logPolynomial` | 对数多项式粘度 |
| `tabulated` | 表格化粘度 |
| `icoTabulated` | 不可压缩表格化粘度 |
| `Andrade` | Andrade 粘度模型 |
| `WLF` | Williams-Landel-Ferry (聚合物) |

### 2.3 热力学模型 (`specie/thermo/`)

**基础热力学** (通过 thermo 模板组合):
- `thermo` — 热力学基类 (含 `HtoEthermo`, `EtoHthermo` 转换)

**比热/焓模型**:
| 模型 | 说明 |
|------|------|
| `hConst` | 常比焓 (Cp = const) |
| `hPolynomial` | 多项式比焓 |
| `hPower` | 幂律比焓 |
| `hTabulated` | 表格化比焓 |
| `hIcoTabulated` | 不可压缩表格化比焓 |
| `eConst` | 常比内能 |
| `ePolynomial` | 多项式比内能 |
| `ePower` | 幂律比内能 |
| `eTabulated` | 表格化比内能 |
| `eIcoTabulated` | 不可压缩表格化比内能 |
| `janaf` | JANAF 热力学数据库 |

**能量变量选择**:
- `sensibleEnthalpy` — 显焓
- `sensibleInternalEnergy` — 显内能
- `absoluteEnthalpy` — 绝对焓
- `absoluteInternalEnergy` — 绝对内能

### 2.4 热物理函数 (`specie/thermophysicalFunctions/`)

- `APIdiffCoef` — API 扩散系数
- `NSRDS` — NSRDS 热物理函数
- `integratedNonUniformTable1` — 非均匀积分表

### 2.5 反应模型 (`specie/reaction/`)

**反应类型** (`Reactions/`):
- `Reaction` — 基础反应
- `IrreversibleReaction` — 不可逆反应
- `ReversibleReaction` — 可逆反应
- `NonEquilibriumReversibleReaction` — 非平衡可逆反应

**反应速率** (`reactionRate/`):
| 模型 | 说明 |
|------|------|
| `ArrheniusReactionRate` | Arrhenius 反应速率 |
| `thirdBodyArrheniusReactionRate` | 第三体 Arrhenius |
| `JanevReactionRate` | Janev 反应速率 |
| `LandauTellerReactionRate` | Landau-Teller 反应速率 |
| `LangmuirHinshelwood` | Langmuir-Hinshelwood |
| `fluxLimitedLangmuirHinshelwoodReactionRate` | 通量限制 L-H |
| `surfaceArrheniusReactionRate` | 表面 Arrhenius |
| `MichaelisMenten` | Michaelis-Menten 酶动力学 |
| `ChemicallyActivatedReactionRate` | 化学活化反应速率 |
| `FallOffReactionRate` | 降落反应速率 |
| `powerSeries` | 幂级数反应速率 |

**降落函数** (`fallOffFunctions/`):
- `LindemannFallOffFunction`
- `TroeFallOffFunction`
- `SRIFallOffFunction`

### 2.6 基础热力学组合 (`basic/`)

**纯物质热力学** (`pureThermo/`):
- `pureThermo` — 纯物质热力学

**psi-热力学** (`psiThermo/`):
- `psiThermo` — ψ (= 1/RT) 基础热力学

**rho-热力学** (`rhoThermo/`):
- `rhoThermo` — ρ 基础热力学

**流体热力学** (`fluidThermo/`):
- `fluidThermo` — 流体热力学
- 水静力初始化

**液相热力学** (`liquidThermo/`):
- `LiquidThermo` — 液相热力学 (含 liquidThermos 实例化)

**rho 流体热力学** (`rhoFluidThermo/`):
- `RhoFluidThermo` — ρ-流体热力学

**混合物** (`mixtures/`):
- `pureMixture` — 纯物质混合物

**边界条件** (`derivedFvPatchFields/`):
- `fixedEnergy` — 固定能量
- `gradientEnergy` — 能量梯度
- `mixedEnergy` — 混合能量
- `energyJump` — 能量跳跃

### 2.7 多组分热力学 (`multicomponentThermo/`)

**混合物类型** (`mixtures/`):
| 模型 | 说明 |
|------|------|
| `singleComponentMixture` | 单组分混合物 |
| `multicomponentMixture` | 多组分混合物 |
| `coefficientMulticomponentMixture` | 系数型多组分混合物 |
| `coefficientWilkeMulticomponentMixture` | Wilke 多组分混合物 |
| `homogeneousMixture` | 均匀混合物 |
| `inhomogeneousMixture` | 非均匀混合物 |
| `veryInhomogeneousMixture` | 高度非均匀混合物 |
| `egrMixture` | EGR 混合物 |
| `valueMulticomponentMixture` | 值型多组分混合物 |

**其他**:
- `psiMulticomponentThermo`
- `psiuMulticomponentThermo` — ψu (未燃烧) 多组分热力学
- `rhoFluidMulticomponentThermo`

### 2.8 固体热力学 (`solidThermo/`)

- `solidThermo` — 固体热力学
- `constSolidThermo` — 常物性固体热力学
- `constAnisoSolidThermo` — 各向异性常物性固体热力学

### 2.9 化学动力学模型 (`chemistryModel/`)

**化学求解器** (`chemistrySolver/`):
- `noChemistrySolver` — 无化学反应
- `EulerImplicit` — 隐式 Euler
- `ode` — ODE 求解器

**化学模型** (`chemistryModel/`):
- `chemistryModel` — 化学模型基类

**机理简化** (`chemistryModel/reduction/`):
| 模型 | 说明 |
|------|------|
| `noChemistryReduction` | 无简化 |
| `DRG` | 有向关系图 |
| `DRGEP` | DRG 边误差传播 |
| `DAC` | 动态自适应化学 |
| `EFA` | 有效模糊自适应 |
| `PFA` | 主成分模糊自适应 |

**化学表格化** (`chemistryModel/tabulation/`):
- `noChemistryTabulation` — 无表格化
- `ISAT` — ISAT (内存中存储的自适应表格化)

### 2.10 层流火焰速度 (`laminarFlameSpeed/`)

- `constant` — 常数
- `Gulders` — Gulders 模型
- `GuldersEGR` — Gulders EGR 修正
- `RaviPetersen` — Ravi-Petersen 模型

### 2.11 饱和模型 (`saturationModels/`)

**压力模型** (`saturationPressureModel/`):
- `constantPressure` — 常压
- `Antoine` — Antoine 方程
- `AntoineExtended` — 扩展 Antoine 方程
- `ArdenBuck` — Arden-Buck 方程
- `polynomialTemperature` — 多项式温度

**温度模型** (`saturationTemperatureModel/`):
- `constantTemperature` — 常温
- `function1Temperature` — 函数1温度

### 2.12 点火模型 (`ignition/`)

- `ignitionSite` — 点火位置
- `ignition` — 点火模型

---

## 3. 热物理输运模型 (`src/ThermophysicalTransportModels/`)

### 3.1 层流热输运 (`fluid/laminar/`)

| 模型 | 说明 |
|------|------|
| `Fourier` | Fourier 热传导 |
| `Fickian` | Fick 扩散 |
| `FickianFourier` | Fick + Fourier |
| `MaxwellStefan` | Maxwell-Stefan 扩散 |
| `MaxwellStefanFourier` | Maxwell-Stefan + Fourier |
| `unityLewisFourier` | Lewis 数=1 的 Fourier |

### 3.2 湍流热输运 (`fluid/turbulence/`)

| 模型 | 说明 |
|------|------|
| `eddyDiffusivity` | 涡扩散率 |
| `FickianEddyDiffusivity` | Fick + 涡扩散率 |
| `nonUnityLewisEddyDiffusivity` | 非单位 Lewis 涡扩散率 |
| `unityLewisEddyDiffusivity` | 单位 Lewis 涡扩散率 |

含 LES/RAS 子分支。

### 3.3 固体热输运 (`solid/`)

- `isotropic` — 各向同性
- `anisotropic` — 各向异性

### 3.4 耦合热输运 (`coupledThermophysicalTransportModels/`)

- `coupledTemperature` — 耦合温度
- `externalTemperature` — 外部温度
- `lumpedMassTemperature` — 集总质量温度

### 3.5 热输运边界条件 (`fluid/derivedFvPatchFields/`)

- `alphatWallFunction` — alphat 壁面函数
- `alphatJayatillekeWallFunction` — Jayatilleke alphat 壁面函数
- `convectiveHeatTransfer` — 对流换热
- `externalCoupledTemperatureMixed` — 外部耦合温度混合
- `thermalBaffle1D` — 1D 热挡板
- `totalFlowRateAdvectiveDiffusive` — 总流率对流扩散

---

## 4. 两相流模型 (`src/twoPhaseModels/`)

### 4.1 VOF (Volume of Fluid)

- `VoF` — VOF 方法 (含 alphaControls, alphaCourantNo, setDeltaT)
- `interfaceProperties` — 界面属性
- `interfaceCompression` — 界面压缩方法:
  - `MPLIC` — 多维PLIC (分段线性界面计算)
  - `PLIC` — PLIC
  - `noInterfaceCompression` — 无界面压缩

### 4.2 界面力模型

**表面张力模型** (`interfaceProperties/surfaceTensionModels/`):
- `constant` — 常表面张力
- `temperatureDependent` — 温度相关表面张力

**接触角模型** (`interfaceProperties/contactAngleModels/`):
- `constant` — 常接触角
- `dynamic` — 动态接触角
- `gravitational` — 重力接触角
- `temperatureDependent` — 温度相关接触角

### 4.3 空化模型

**可压缩空化** (`compressibleCavitation/`):
- `Kunz` — Kunz 空化模型
- `Merkle` — Merkle 空化模型
- `SchnerrSauer` — Schnerr-Sauer 空化模型
- `Saito` — Saito 空化模型

**不可压缩空化** (`incompressibleCavitation/`):
- `Kunz`
- `Merkle`
- `SchnerrSauer`

### 4.4 其他两相组件

- `compressibleInterfaceProperties` — 可压缩界面属性
- `compressibleTwoPhases` — 可压缩两相
- `incompressibleTwoPhases` — 不可压缩两相
- `twoPhaseMixture` — 两相混合物
- `twoPhaseProperties` — 两相属性

---

## 5. 多相流模型 (`src/multiphaseModels/`)

- `multiphaseProperties` — 多相属性
- `alphaContactAngle` — alpha 接触角
- `correctContactAngle` — 校正接触角

> 注：Euler-Euler 和 Euler-Lagrange 框架主要在 `src/lagrangian/` 和应用层实现。

---

## 6. 辐射模型 (`src/radiationModels/`)

### 6.1 辐射传输模型 (`radiationModels/`)

| 模型 | 说明 |
|------|------|
| `noRadiation` | 无辐射 |
| `P1` | P1 辐射模型 |
| `fvDOM` | 有限体积离散坐标法 (DOM) |
| `viewFactor` | 视角因子模型 |
| `opaqueSolid` | 不透明固体辐射 |

### 6.2 吸收-发射模型 (`absorptionEmissionModels/`)

| 模型 | 说明 |
|------|------|
| `noAbsorptionEmission` | 无吸收发射 |
| `constantAbsorptionEmission` | 常吸收发射 |
| `greyMean` | 灰色平均模型 |
| `wideBand` | 宽带模型 |
| `binary` | 二元模型 |

### 6.3 散射模型 (`scatterModels/`)

- `noScatter` — 无散射
- `constantScatter` — 常散射

### 6.4 烟灰模型 (`sootModels/`)

- `noSoot` — 无烟灰

### 6.5 辐射边界条件 (`derivedFvPatchFields/`)

- `MarshakRadiation` — Marshak 辐射
- `MarshakRadiationFixedTemperature` — Marshak 固定温度辐射
- `greyDiffusiveRadiation` — 灰色漫射辐射
- `greyDiffusiveViewFactor` — 灰色漫射视角因子
- `radiationCoupledBase` — 辐射耦合基类
- `wideBandDiffusiveRadiation` — 宽带漫射辐射

---

## 7. 燃烧模型 (`src/combustionModels/`)

| 模型 | 说明 |
|------|------|
| `laminar` | 层流燃烧 |
| `noCombustion` | 无燃烧 |
| `infinitelyFastChemistry` | 无限快化学反应 |
| `PaSR` | 部分搅拌反应器 |
| `EDC` | 涡耗散概念模型 |
| `singleStepCombustion` | 单步燃烧 |
| `diffusion` | 扩散燃烧 |
| `FSD` | 火焰面密度模型 (含 `consumptionSpeed` 和 `relaxation` 反应速率火焰面积模型) |
| `zoneCombustion` | 区域燃烧 |

---

## 8. 物理属性模型 (`src/physicalProperties/`)

### 8.1 粘度模型 (`viscosityModels/`)

- `constant` — 常粘度
- `viscosityModel` — 粘度模型基类

---

## 9. Lagrangian 粒子模型 (`src/lagrangian/`)

### 9.1 基础粒子模型 (`basic/`)

- `Cloud` — 云基类
- `particle` — 基础粒子
- `passiveParticle` — 被动粒子
- `IOPosition` — 位置 IO
- `InteractionLists` — 交互列表

### 9.2 颗粒云模型 (`parcel/`)

**云类型** (`clouds/`):
- `KinematicCloud` — 运动学云
- `ThermoCloud` — 热力学云
- `ReactingCloud` — 反应云
- `ReactingMultiphaseCloud` — 多相反应云
- `SprayCloud` — 喷雾云

**积分方案** (`integrationScheme/`):
- `Euler` — Euler 积分
- `analytical` — 解析积分

### 9.3 动量子模型 (`submodels/Momentum/`)

**碰撞模型** (`CollisionModel/`):
- `NoCollision` — 无碰撞
- `PairCollision` — 粒子对碰撞

**弥散模型** (`DispersionModel/`):
- `NoDispersion` — 无弥散
- `GradientDispersionRAS` — 梯度弥散 RAS
- `StochasticDispersionRAS` — 随机弥散 RAS

**注入模型** (`InjectionModel/`):
- `NoInjection` — 无注入
- `ManualInjection` — 手动注入
- `ConeInjection` — 锥形注入
- `CellZoneInjection` — 单元区域注入
- `PatchInjection` — 补丁注入
- `PatchFlowRateInjection` — 补丁流率注入
- `FieldActivatedInjection` — 场激活注入
- `MomentumLookupTableInjection` — 动量查表注入

**粒子力** (`ParticleForces/`):
- `Gravity` — 重力
- `Drag` — 阻力
- `Lift` — 升力
- `VirtualMass` — 虚拟质量
- `PressureGradient` — 压力梯度
- `NonInertialFrame` — 非惯性系
- `Paramagnetic` — 顺磁力
- `Scaled` — 缩放力

**补丁交互模型** (`PatchInteractionModel/`):
- `NoInteraction` — 无交互
- `LocalInteraction` — 局部交互
- `Rebound` — 反弹
- `StandardWallInteraction` — 标准壁面交互

**随机碰撞** (`StochasticCollision/`):
- `NoStochasticCollision` — 无随机碰撞
- `StochasticCollisionModel` — 随机碰撞模型

**表面薄膜** (`SurfaceFilmModel/`):
- `NoSurfaceFilm` — 无表面薄膜
- `SurfaceFilmModel` — 表面薄膜模型

### 9.4 热力量子模型 (`submodels/Thermodynamic/`)

**换热模型** (`HeatTransferModel/`):
- `NoHeatTransfer` — 无换热
- `RanzMarshall` — Ranz-Marshall 换热

### 9.5 反应粒子子模型 (`submodels/Reacting/`)

**组分模型** (`CompositionModel/`):
- `NoComposition` — 无组分
- `SingleMixtureFraction` — 单混合分数
- `SinglePhaseMixture` — 单相混合物

**相变模型** (`PhaseChangeModel/`):
- `NoPhaseChange` — 无相变
- `LiquidEvaporation` — 液体蒸发
- `LiquidEvaporationBoil` — 液体蒸发沸腾

### 9.6 多相反应子模型 (`submodels/ReactingMultiphase/`)

**挥发模型** (`DevolatilisationModel/`):
- `NoDevolatilisation` — 无挥发
- `ConstantRateDevolatilisation` — 常速率挥发
- `SingleKineticRateDevolatilisation` — 单动力学速率挥发

**表面反应** (`SurfaceReactionModel/`):
- `NoSurfaceReaction` — 无表面反应
- `COxidationDiffusionLimitedRate` — CO 氧化扩散限制速率
- `COxidationHurtMitchell` — CO 氧化 Hurt-Mitchell
- `COxidationIntrinsicRate` — CO 氧化本征速率
- `COxidationKineticDiffusionLimitedRate` — CO 氧化动力学扩散限制
- `COxidationMurphyShaddix` — CO 氧化 Murphy-Shaddix

### 9.7 喷雾子模型 (`submodels/Spray/`)

**雾化模型** (`AtomisationModel/`):
- `NoAtomisation` — 无雾化
- `LISAAtomisation` — LISA 雾化
- `BlobsSheetAtomisation` — 液滴薄片雾化

**破碎模型** (`BreakupModel/`):
- `NoBreakup` — 无破碎
- `TAB` — TAB 模型
- `ETAB` — ETAB 模型
- `PilchErdman` — Pilch-Erdman 模型
- `ReitzDiwakar` — Reitz-Diwakar 模型
- `ReitzKHRT` — Reitz KH-RT 模型
- `SHF` — SHF 模型

### 9.8 MPPIC 模型 (`submodels/MPPIC/`)

**阻尼模型** (`DampingModels/`):
- `NoDamping` — 无阻尼
- `Relaxation` — 松弛阻尼

**填充模型** (`PackingModels/`):
- `NoPacking` — 无填充
- `Explicit` — 显式填充
- `Implicit` — 隐式填充

**各向同性模型** (`IsotropyModels/`):
- `NoIsotropy` — 无各向同性化
- `Stochastic` — 随机各向同性化

**时间尺度模型** (`TimeScaleModels/`):
- `equilibrium` — 平衡
- `isotropic` — 各向同性
- `nonEquilibrium` — 非平衡

**校正限制** (`CorrectionLimitingMethods/`):
- `noCorrectionLimiting` — 无校正限制
- `absolute` — 绝对限制
- `relative` — 相对限制

### 9.9 粒子函数对象 (`CloudFunctionObjects/`):

- `FacePostProcessing`
- `Flux`
- `ParticleCollector`
- `ParticleErosion`
- `ParticleTracks`
- `ParticleTrap`
- `PatchCollisionDensity`
- `PatchPostProcessing`
- `RelativeVelocity`
- `SizeDistribution`
- `VolumeFraction`

### 9.10 DSMC 模型 (`DSMC/`)

**二元碰撞** (`BinaryCollisionModel/`):
- `NoBinaryCollision` — 无二元碰撞
- `VariableHardSphere` — 可变硬球
- `LarsenBorgnakkeVariableHardSphere` — Larsen-Borgnakke 可变硬球

**壁面交互** (`WallInteractionModel/`):
- `MaxwellianThermal` — Maxwell 热壁面
- `SpecularReflection` — 镜面反射
- `MixedDiffuseSpecular` — 混合漫射镜面

### 9.11 分子动力学 (`molecularDynamics/`)

**对势** (`potential/pairPotential/`):
- `lennardJones` — Lennard-Jones
- `maitlandSmith` — Maitland-Smith
- `exponentialRepulsion` — 指数排斥
- `azizChen` — Aziz-Chen
- `coulomb` — 库仑
- `dampedCoulomb` — 阻尼库仑
- `noInteraction` — 无交互

**系链势** (`potential/tetherPotential/`):
- `harmonicSpring` — 谐振弹簧
- `pitchForkRing` — 叉形环
- `restrainedHarmonicSpring` — 约束谐振弹簧

**能量缩放函数** (`potential/energyScalingFunction/`):
- `noScaling` — 无缩放
- `shifted` — 移位
- `shiftedForce` — 移位力
- `sigmoid` — S 形
- `doubleSigmoid` — 双 S 形

---

## 10. FV 源项模型 (`src/fvModels/`)

### 10.1 通用源项 (`general/`)

| 模型 | 说明 |
|------|------|
| `actuationDisk` | 致动盘 |
| `radialActuationDisk` | 径向致动盘 |
| `buoyancyForce` | 浮力 |
| `buoyancyEnergy` | 浮力能量 |
| `heatSource` | 热源 |
| `massSource` | 质量源 |
| `massTransfer` | 质量传递 |
| `phaseChange` | 相变 |
| `porosityForce` | 多孔介质力 |
| `solidificationMelting` | 凝固熔化 |
| `viscousHeating` | 粘性加热 |
| `acceleration` | 加速度 |
| `semiImplicitSource` | 半隐式源 |
| `codedFvModel` | 编码自定义模型 |
| `effectivenessHeatExchanger` | 有效度换热器 |
| `volumeBlockage` | 体积阻塞 |
| `volumeSource` | 体积源 |
| `solidThermalEquilibrium` | 固体热平衡 |
| `phaseLimitStabilisation` | 相限制稳定化 |
| `sixDoFAcceleration` | 6自由度加速度 |
| `zeroDimensionalMassSource` | 零维质量源 |

### 10.2 区域间模型 (`interRegion/`)

- `interRegionHeatTransfer` — 区域间换热
- `interRegionPorosityForce` — 区域间多孔力

**换热系数模型** (`heatTransferCoefficientModels/`):
- `constant` — 常数
- `variable` — 变量
- `function1` — 函数1
- `function2` — 函数2

### 10.3 旋翼盘模型 (`rotorDisk/`)

- `rotorDisk` — 旋翼盘
- `bladeModel` — 叶片模型
- `profileModel` — 翼型模型
- `trimModel` — 修整模型 (含 `fixed`, `targetCoeff`)

### 10.4 螺旋桨盘 (`propellerDisk/`)

- `propellerDisk` — 螺旋桨盘

---

## 11. FV 约束 (`src/fvConstraints/`)

| 约束 | 说明 |
|------|------|
| `bound` | 值域约束 |
| `fixedValue` | 固定值 |
| `fixedTemperature` | 固定温度 |
| `limitMag` | 幅值限制 |
| `limitPressure` | 压力限制 |
| `limitTemperature` | 温度限制 |
| `meanVelocityForce` | 平均速度力 |
| `zeroDimensionalFixedPressure` | 零维固定压力 |

---

## 12. 有限体积离散格式 (`src/finiteVolume/`)

### 12.1 时间离散格式 (`ddtSchemes/`)

| 格式 | 说明 |
|------|------|
| `Euler` | 一阶隐式 Euler |
| `backward` | 二阶向后差分 |
| `CrankNicolson` | Crank-Nicolson |
| `steadyState` | 稳态 (无时间项) |
| `localEuler` | 局部时间步 Euler |
| `bounded` | 有界时间格式 |
| `CoEuler` | Courant 数 Euler |
| `SLTS` | 子循环时间步进 |

### 12.2 二阶时间导数 (`d2dt2Schemes/`)

- `Euler` — Euler 二阶
- `steadyState` — 稳态

### 12.3 梯度格式 (`gradSchemes/`)

| 格式 | 说明 |
|------|------|
| `gauss` | 高斯积分梯度 |
| `leastSquares` | 最小二乘梯度 |
| `LeastSquares` | 最小二乘 (另一种实现) |
| `fourth` | 四阶梯度 |

**限制梯度格式** (`limitedGradSchemes/`):
- `cellLimited` — 单元限制
- `cellMDLimited` — 单元多维限制
- `faceLimited` — 面限制
- `faceMDLimited` — 面多维限制

### 12.4 散度格式 (`divSchemes/`)

- `gauss` — 高斯散度格式 (与其他格式组合使用)

### 12.5 对流格式 (`convectionSchemes/`)

- `gauss` — 高斯对流格式
- `bounded` — 有界对流格式
- `multivariateGauss` — 多变量高斯对流格式

### 12.6 Laplacian 格式 (`laplacianSchemes/`)

- `gauss` — 高斯 Laplacian (与 snGrad 格式组合)

### 12.7 表面法向梯度格式 (`snGradSchemes/`)

| 格式 | 说明 |
|------|------|
| `corrected` | 校正格式 |
| `uncorrected` | 非校正格式 |
| `orthogonal` | 正交格式 |
| `faceCorrected` | 面校正格式 |
| `limited` | 限制格式 |
| `phaseStabilised` | 相稳定化格式 |
| `quadraticFit` | 二次拟合 |
| `linearFit` | 线性拟合 |
| `CentredFit` | 中心拟合 |

### 12.8 表面插值格式 (`interpolation/surfaceInterpolation/schemes/`)

**基本格式**:
| 格式 | 说明 |
|------|------|
| `linear` | 线性 (二阶中心) |
| `midPoint` | 中点 |
| `upwind` | 一阶迎风 |
| `downwind` | 顺风 |
| `linearUpwind` | 线性迎风 |
| `pointLinear` | 点线性 |
| `cubic` | 三次 |
| `harmonic` | 调和 |
| `reverseLinear` | 逆线性 |
| `outletStabilised` | 出口稳定化 |
| `phaseStabilised` | 相稳定化 |
| `weighted` | 加权 |
| `deferred` | 延迟校正 |
| `skewCorrected` | 扭斜校正 |

**混合格式**:
- `CoBlended` — Courant 数混合
- `cellCoBlended` — 单元 Co 混合
- `fixedBlended` — 固定混合
- `limiterBlended` — 限制器混合
- `localBlended` — 局部混合
- `clippedLinear` — 截断线性

**高阶拟合格式**:
- `linearFit` — 线性拟合
- `quadraticFit` — 二次拟合
- `quadraticLinearFit` — 二次线性拟合
- `cubicUpwindFit` — 三次迎风拟合
- `quadraticUpwindFit` — 二次迎风拟合
- `quadraticLinearUpwindFit` — 二次线性迎风拟合
- `linearPureUpwindFit` — 线性纯迎风拟合
- `quadraticLinearPureUpwindFit` — 二次线性纯迎风拟合
- `LUST` — 线性上游加权

**局部格式**:
- `localMax` — 局部最大
- `localMin` — 局部最小

### 12.9 限制格式 (`limitedSchemes/`)

| 格式 | 说明 |
|------|------|
| `upwind` | 一阶迎风 |
| `limitedLinear` | 限制线性 |
| `limitedCubic` | 限制三次 |
| `vanLeer` | van Leer |
| `MUSCL` | MUSCL |
| `QUICK` | QUICK |
| `Gamma` | Gamma |
| `SuperBee` | SuperBee |
| `Minmod` | Minmod |
| `vanAlbada` | van Albada |
| `UMIST` | UMIST |
| `OSPRE` | OSPRE |
| `SFCD` | SFCD |
| `filteredLinear` | 滤波线性 |
| `filteredLinear2` | 滤波线性2 |
| `filteredLinear3` | 滤波线性3 |
| `blended` | 混合 |

**Phi 格式**:
- `Phi` — Phi 格式 (与面通量方向相关)
- `PhiScheme` — Phi 方案基类

### 12.10 多变量格式 (`multivariateSchemes/`)

- `upwind`, `vanLeer`, `MUSCL`, `Gamma`, `Minmod`, `SuperBee`
- `limitedLinear`, `limitedCubic`
- `multivariateScheme`, `multivariateIndependentScheme`, `multivariateSelectionScheme`

### 12.11 插值点方法 (`interpolation/`)

| 方法 | 说明 |
|------|------|
| `interpolationCell` | 单元插值 |
| `interpolationCellPatchConstrained` | 单元补丁约束插值 |
| `interpolationCellPoint` | 单元-点插值 |
| `interpolationCellPointFace` | 单元-点-面插值 |
| `interpolationCellPointWallModified` | 单元-点壁面修正插值 |
| `interpolationPointMVC` | 点 MVC 插值 |
| `interpolationVolPointInterpolation` | 体-点插值 |

### 12.12 体-点插值 (`volPointInterpolation/`)

- `volPointInterpolation` — 体积到点插值
- `pointConstraints` — 点约束

---

## 13. 线性求解器与预条件器

### 13.1 线性求解器 (`src/OpenFOAM/matrices/lduMatrix/solvers/`)

| 求解器 | 说明 |
|--------|------|
| `PCG` | 预处理共轭梯度 (对称) |
| `PBiCG` | 预处理双共轭梯度 (非对称) |
| `PBiCGStab` | 预处理稳定双共轭梯度 (非对称) |
| `smoothSolver` | 光滑求解器 |
| `GAMG` | 几何代数多重网格 |
| `diagonalSolver` | 对角求解器 |

### 13.2 预条件器 (`preconditioners/`)

| 预条件器 | 说明 |
|----------|------|
| `noPreconditioner` | 无预条件 |
| `diagonalPreconditioner` | 对角预条件 (Jacobi) |
| `DICPreconditioner` | 不完全 Cholesky (DIC) |
| `DILUPreconditioner` | 不完全 LU (DILU) |
| `FDICPreconditioner` | 迫选 DIC |
| `GAMGPreconditioner` | GAMG 预条件 |

### 13.3 光滑器 (Smoothers) (`smoothers/`)

| 光滑器 | 说明 |
|--------|------|
| `GaussSeidel` | Gauss-Seidel |
| `DIC` | DIC 光滑器 |
| `DILU` | DILU 光滑器 |
| `FDIC` | 迫选 DIC 光滑器 |
| `DICGaussSeidel` | DIC + Gauss-Seidel |
| `DILUGaussSeidel` | DILU + Gauss-Seidel |
| `symGaussSeidel` | 对称 Gauss-Seidel |
| `nonBlockingGaussSeidel` | 非阻塞 Gauss-Seidel |

### 13.4 GAMG 专用

- `MGridGenGamgAgglomeration` — MGridGen GAMG 聚合方法
- `pairPatchAgglomeration` — 配对补丁聚合

### 13.5 矩阵类型 (`src/OpenFOAM/matrices/`)

- `LduMatrix` / `lduMatrix` — LDU 矩阵
- `SquareMatrix` — 方阵
- `RectangularMatrix` — 矩形矩阵
- `SymmetricSquareMatrix` — 对称方阵
- `DiagonalMatrix` — 对角矩阵
- `scalarMatrices` — 标量矩阵
- `simpleMatrix` — 简单矩阵
- `Matrix` — 通用矩阵
- `MatrixBlock` — 分块矩阵
- `LUscalarMatrix` — LU 分解标量矩阵
- `LLTMatrix` — LLT 分解矩阵
- `QRMatrix` — QR 分解矩阵
- `SVD` — 奇异值分解

---

## 14. 边界条件 (`src/finiteVolume/fields/fvPatchFields/`)

### 14.1 基本边界条件 (`basic/`)

| BC | 说明 |
|----|------|
| `fixedValue` | 固定值 |
| `fixedGradient` | 固定梯度 |
| `mixed` | 混合 (Robin) |
| `calculated` | 计算值 |
| `zeroGradient` | 零梯度 |
| `directionMixed` | 方向混合 |
| `coupled` | 耦合 |
| `sliced` | 切片 |
| `transform` | 变换 |
| `extrapolatedCalculated` | 外推计算 |
| `basicSymmetry` | 基本对称 |

### 14.2 约束边界条件 (`constraint/`)

| BC | 说明 |
|----|------|
| `empty` | 空 (2D) |
| `symmetry` | 对称 |
| `symmetryPlane` | 对称面 |
| `wedge` | 楔形 |
| `cyclic` | 周期 |
| `cyclicSlip` | 周期滑移 |
| `processor` | 处理器 (并行) |
| `processorCyclic` | 处理器周期 |
| `jumpCyclic` | 跳跃周期 |
| `nonConformalCyclic` | 非一致周期 |
| `nonConformalProcessorCyclic` | 非一致处理器周期 |
| `nonConformalError` | 非一致错误 |
| `internal` | 内部 |

### 14.3 派生边界条件 (`derived/`) — 完整列表

#### 速度类

| BC | 说明 |
|----|------|
| `noSlip` | 无滑移 |
| `slip` | 滑移 |
| `partialSlip` | 部分滑移 |
| `fixedNormalSlip` | 固定法向滑移 |
| `fixedNormalInletOutletVelocity` | 固定法向入口出口速度 |
| `pressureInletVelocity` | 压力入口速度 |
| `pressureInletUniformVelocity` | 压力入口均匀速度 |
| `pressureInletOutletVelocity` | 压力入口出口速度 |
| `pressureDirectedInletVelocity` | 压力定向入口速度 |
| `pressureDirectedInletOutletVelocity` | 压力定向入口出口速度 |
| `pressureInletOutletParSlipVelocity` | 压力入口出口平行滑移速度 |
| `pressureNormalInletOutletVelocity` | 压力法向入口出口速度 |
| `rotatingPressureInletOutletVelocity` | 旋转压力入口出口速度 |
| `flowRateInletVelocity` | 流率入口速度 |
| `flowRateOutletVelocity` | 流率出口速度 |
| `mappedFlowRateVelocity` | 映射流率速度 |
| `matchedFlowRateOutletVelocity` | 匹配流率出口速度 |
| `variableHeightFlowRateInletVelocity` | 变高度流率入口速度 |
| `fluxCorrectedVelocity` | 通量校正速度 |
| `freestreamVelocity` | 自由流速度 |
| `interstitialInletVelocity` | 间隙入口速度 |
| `swirlInletVelocity` | 旋流入口速度 |
| `swirlFlowRateInletVelocity` | 旋流流率入口速度 |
| `movingWallVelocity` | 运动壁面速度 |
| `movingWallSlipVelocity` | 运动壁面滑移速度 |
| `movingMappedWallVelocity` | 运动映射壁面速度 |
| `rotatingWallVelocity` | 旋转壁面速度 |
| `translatingWallVelocity` | 平移壁面速度 |
| `surfaceNormalFixedValue` | 面法向固定值 |
| `surfaceNormalUniformFixedValue` | 面法向均匀固定值 |
| `turbulentIntensityKineticEnergyInlet` | 湍流强度 k 入口 |
| `turbulentInlet` | 湍流入口 |

#### 压力类

| BC | 说明 |
|----|------|
| `totalPressure` | 总压 |
| `uniformTotalPressure` | 均匀总压 |
| `rotatingTotalPressure` | 旋转总压 |
| `freestreamPressure` | 自由流压力 |
| `fixedFluxPressure` | 固定通量压力 |
| `fixedFluxExtrapolatedPressure` | 固定通量外推压力 |
| `PrghPressure` | p_rgh 压力 |
| `dynamicPressure` | 动压 |
| `pressure` | 压力 |
| `prghCyclicPressure` | p_rgh 周期压力 |
| `prghTotalHydrostaticPressure` | p_rgh 总静水压力 |
| `phaseHydrostaticPressure` | 相静水压力 |
| `uniformDensityHydrostaticPressure` | 均匀密度静水压力 |
| `fixedPressureCompressibleDensity` | 固定压力可压缩密度 |
| `syringePressure` | 注射器压力 |
| `plenumPressure` | 增压室压力 |
| `fanPressure` | 风扇压力 |
| `fanPressureJump` | 风扇压力跳跃 |
| `waveSurfacePressure` | 波面压力 |

#### 温度类

| BC | 说明 |
|----|------|
| `totalTemperature` | 总温 |
| `inletOutletTotalTemperature` | 入口出口总温 |

#### 场属性类

| BC | 说明 |
|----|------|
| `inletOutlet` | 入口出口 |
| `outletInlet` | 出口入口 |
| `zeroInletOutlet` | 零入口出口 |
| `uniformInletOutlet` | 均匀入口出口 |
| `fixedValueInletOutlet` | 固定值入口出口 |
| `fixedMean` | 固定均值 |
| `fixedMeanOutletInlet` | 固定均值出口入口 |
| `freestream` | 自由流 |
| `advective` | 对流 |
| `waveTransmissive` | 波透射 |
| `supersonicFreestream` | 超声速自由流 |
| `transonicEntrainmentPressure` | 跨声速夹带压力 |
| `entrainmentPressure` | 夹带压力 |
| `uniformFixedValue` | 均匀固定值 |
| `uniformFixedGradient` | 均匀固定梯度 |
| `fixedProfile` | 固定剖面 |
| `codedFixedValue` | 编码固定值 |
| `codedMixed` | 编码混合 |
| `externalCoupledMixed` | 外部耦合混合 |
| `mappedValue` | 映射值 |
| `mappedInternalValue` | 映射内部值 |
| `mappedVelocityFlux` | 映射速度通量 |
| `timeVaryingMappedFixedValue` | 时变映射固定值 |
| `outletMappedUniformInlet` | 出口映射均匀入口 |
| `outletPhaseMeanVelocity` | 出口相平均速度 |
| `fixedJump` | 固定跳跃 |
| `uniformJump` | 均匀跳跃 |
| `fixedInternalValue` | 固定内部值 |
| `variableHeightFlowRate` | 变高度流率 |

#### 特殊类

| BC | 说明 |
|----|------|
| `interfaceCompression` | 界面压缩 |
| `turbulentInlet` | 湍流入口 |

### 14.4 表面场边界条件 (`fvsPatchFields/`)

**基本**:
- `fixedValue`, `calculated`, `coupled`, `sliced`

**约束**:
- `empty`, `symmetry`, `symmetryPlane`, `wedge`
- `cyclic`, `cyclicSlip`, `processor`, `processorCyclic`
- `nonConformalCyclic`, `nonConformalProcessorCyclic`, `nonConformalError`, `internal`

**派生**:
- `polyFaces`, `nonConformalPolyFaces`, `nonConformalMappedPolyFaces`

---

## 15. 壁面距离计算 (`src/meshTools/patchDist/`)

- `patchDistWave` — 补丁距离波传播
- `WallInfo` — 壁面信息
- `WallLocation` — 壁面位置

另有 `pointDist` 在 `src/meshTools/pointDist/` 用于点距离计算。

---

## 16. ODE 求解器 (`src/ODE/ODESolvers/`)

| 求解器 | 说明 |
|--------|------|
| `Euler` | 显式 Euler |
| `EulerSI` | 隐式 Euler (半隐式) |
| `Trapezoid` | 梯形法 |
| `Rosenbrock12` | Rosenbrock 1(2) |
| `Rosenbrock23` | Rosenbrock 2(3) |
| `Rosenbrock34` | Rosenbrock 3(4) |
| `rodas23` | RODAS 2(3) |
| `rodas34` | RODAS 3(4) |
| `RKF45` | Runge-Kutta-Fehlberg 4(5) |
| `RKDP45` | Runge-Kutta-Dormand-Prince 4(5) |
| `RKCK45` | Runge-Kutta-Cash-Karp 4(5) |
| `SIBS` | Stoer-Bulirsch (隐式外推) |
| `seulex` | 单独外推法 |

---

## 17. 刚体运动 (`src/rigidBodyMotion/`)

### 17.1 关节类型 (`joints/`)

- `null`, `fixed` — 空/固定
- `Px`, `Py`, `Pz`, `Pxyz`, `Pa` — 平移关节
- `Rx`, `Ry`, `Rz`, `Rxyz`, `Ra`, `Ryxz`, `Rzyx`, `Rs` — 旋转关节
- `composite` — 复合关节
- `floating` — 浮动关节
- `function` — 函数驱动关节
- `functionDot` — 函数导数驱动关节
- `rotating` — 旋转关节

### 17.2 约束 (`restraints/`)

- `linearSpring` — 线性弹簧
- `linearAxialAngularSpring` — 线性轴向角弹簧
- `linearDamper` — 线性阻尼器
- `sphericalAngularDamper` — 球形角阻尼器
- `externalForce` — 外力

### 17.3 刚体类型 (`bodies/`)

- `rigidBody` — 刚体
- `compositeBody` — 复合体
- `subBody` — 子体
- `jointBody` — 关节体
- `masslessBody` — 无质量体
- `cuboid` — 长方体
- `sphere` — 球体

---

## 18. 运动求解器 (`src/motionSolvers/`)

### 18.1 位移运动求解器 (`displacement/`)

- `displacement` — 位移
- `layeredSolver` — 分层求解器
- `linearSolver` — 线性求解器
- `points0` — 初始点

### 18.2 刚体运动 (`displacement/solidBody/`)

**运动函数** (`solidBodyMotionFunctions/`):
- `SDA` — 六自由度 SDA
- `axisRotationMotion` — 轴旋转运动
- `linearMotion` — 线性运动
- `multiMotion` — 多重运动
- `oscillatingLinearMotion` — 振荡线性运动
- `oscillatingRotatingMotion` — 振荡旋转运动
- `rotatingMotion` — 旋转运动
- `sixDoFMotion` — 六自由度运动

**求解器类型**:
- `solidBodyMotionSolver` — 刚体运动求解器
- `multiSolidBodyMotionSolver` — 多刚体运动求解器
- `interpolatingSolidBodyMotionSolver` — 插值刚体运动求解器

### 18.3 速度运动求解器 (`velocity/`)

- `velocityMotionSolver`

### 18.4 分量运动求解器

- `componentDisplacementMotionSolver`
- `componentVelocityMotionSolver`

---

## 19. 网格拓扑变换 (`src/fvMeshTopoChangers/`)

- `refiner` — 网格细化
- `meshToMesh` — 网格到网格映射 (含 adjustTimeStep)

---

## 20. 波浪模型 (`src/waves/`)

### 20.1 波浪模型 (`waveModels/`)

- `Airy` — Airy 线性波
- `Stokes2` — 二阶 Stokes 波
- `Stokes5` — 五阶 Stokes 波
- `irregular` — 不规则波
- `solitary` — 孤波

### 20.2 波叠加 (`waveSuperpositions/`)

- `waveSuperposition`
- `waveAtmBoundaryLayerSuperposition`

### 20.3 波 FV 模型 (`fvModels/`)

- `waveForcing` — 波浪强制
- `isotropicDamping` — 各向同性阻尼
- `verticalDamping` — 垂直阻尼
- `forcing` — 强制

---

## 21. 大气模型 (`src/atmosphericModels/`)

- `kEpsilonLopesdaCosta` — Lopes da Costa k-epsilon (大气边界层)
- `powerLawLopesdaCosta` — Lopes da Costa 幂律 (多孔介质)

---

## 22. 种质量传递 (`src/specieTransfer/`)

**边界条件**:
- `specieTransferMassFraction` — 种质量传递质量分数
- `specieTransferTemperature` — 种质量传递温度
- `specieTransferVelocity` — 种质量传递速度
- `semiPermeableBaffleMassFraction` — 半透膜挡板质量分数
- `adsorptionMassFraction` — 吸附质量分数

---

## 23. FV 场源 (`src/finiteVolume/fields/fvFieldSources/`)

**派生场源** (`derived/`):
- `internal` — 内部源
- `uniformFixedValue` — 均匀固定值源
- `uniformInletOutlet` — 均匀入口出口源
- `turbulentIntensityKineticEnergy` — 湍流强度动能源

---

## 汇总统计

| 类别 | 数量 |
|------|------|
| RANS 模型 | 13 |
| LES 模型 | 6 |
| DES/DDES/IDDES | 4 |
| LES 滤波器 | 3 |
| LES Delta 尺度 | 6 |
| 层流/粘弹性模型 | 6 |
| 广义牛顿粘度 | 7 |
| 壁面函数 | ~12 |
| 状态方程 | 12 |
| 输运模型 | 8 |
| 热力学模型 | 15 |
| 反应速率模型 | 11 |
| 降落函数 | 3 |
| 反应类型 | 4 |
| 化学求解器 | 3 |
| 机理简化 | 6 |
| 表格化 | 2 |
| 层流火焰速度 | 4 |
| 饱和模型 | 7 |
| 多组分混合物 | 9 |
| 辐射模型 | 5 |
| 吸收-发射模型 | 5 |
| 燃烧模型 | 9 |
| VOF/界面压缩 | 4 |
| 空化模型 | 4 (可压缩) + 3 (不可压缩) |
| 表面张力模型 | 2 |
| 接触角模型 | 4 |
| ddt 格式 | 8 |
| 梯度格式 | 4 + 4 限制 |
| snGrad 格式 | 9 |
| 表面插值格式 | ~33 |
| 限制格式 | ~17 |
| 线性求解器 | 6 |
| 预条件器 | 6 |
| 光滑器 | 8 |
| ODE 求解器 | 12 |
| 基本 BC | 11 |
| 约束 BC | 13 |
| 派生 BC | ~85 |
| FV 源项模型 | ~20 |
| 刚体关节 | ~20 |
| 运动函数 | 8 |
| Lagrangian 注入 | 8 |
| 粒子力 | 8 |
| MPPIC 子模型 | ~15 |
| DSMC 模型 | 7 |
| 分子动力学势 | 10 |
| 波浪模型 | 5 |

---

> **说明**：此目录基于 OpenFOAM Foundation (OpenFOAM-12/13) 的 GitHub 源码。
> ESI v2512 (2025年12月) 版本的核心模型结构与 Foundation 版本高度一致，
> 但可能在部分应用层求解器和 GUI 集成上有差异。
> 具体模型的模板实例化组合远多于上述列出的独立模型数量。

# pyOpenFOAM 完整重实现路线图

更新时间: 2026-06-13

## 项目目标

完整无遗漏重实现 OpenFOAM-13（Python/PyTorch），实现：
1. 所有 ~267 原生算例精度达标
2. 完全支持 GPU 加速
3. 完全支持端到端可微分模拟

## 当前状态

| 指标 | 当前值 | 目标值 | 状态 |
|------|--------|--------|------|
| 已注册求解器 | 69 base + 147 enhanced | 对应 OpenFOAM-13 全部 | ✅ |
| 求解器有真实物理 | 50/50 tested | 全部 | ⚠️ 6 有问题 |
| OpenFOAM 参照对比 | 3 算例 | 267 算例 | 🔴 1.1% |
| Cavity 32x32 精度 | 1.0% 误差 | <5% | ✅ |
| Couette 精度 | 87-95% 误差 | <5% | 🔴 |
| Poiseuille 精度 | 发散 | <5% | 🔴 |
| Docker/OpenFOAM 参照 | 未运行 | 可用 | ⏳ |
| 单元测试 | 17,197+ pass | 全部通过 | ✅ |
| GPU 验证 | 50 求解器有限值 | 全部精度达标 | ⚠️ |

## 已完成阶段

### 阶段 1：核心框架 ✅
- [x] 69 个基础求解器实现（全部有完整 run() 方法）
- [x] 147 个增强求解器变体
- [x] 408 个 RTS 边界条件
- [x] 20+ 湍流模型
- [x] 32+ 状态方程
- [x] 75 个 ODE 求解器
- [x] LDU 矩阵格式、FVM 离散化
- [x] PCG/PBiCGStab/GAMG 线性求解器

### 阶段 2：矩阵级 BC 处理 ✅
- [x] PISO: 边界面对角一致性修复
- [x] SIMPLE: 设备一致性修复
- [x] Cavity 32x32 Re=100: 42%→1.0% 误差

### 阶段 3：测试基线 ✅
- [x] 17,197+ 单元测试通过
- [x] 50 求解器端到端验证
- [x] GPU 验证（RTX 4070 Ti SUPER）

## 待完成阶段

### 阶段 A：OpenFOAM 参照环境搭建 🔴 阻塞

**目标**: 建立可用的 OpenFOAM 参照运行环境

**任务清单**:
- [ ] A1. 启动 Docker Desktop 并拉取 OpenFOAM 镜像
  - `docker pull openfoam/openfoam-13-default` 或 `openfoam/openfoam13`
  - 如果 Foundation 镜像不可用，尝试 ESI 镜像 `openfoam/openfoam-v2312`
  - 备选：重新提取 v1906 .deb 包到 /tmp/openfoam1906/
- [ ] A2. 验证 Docker 内 OpenFOAM 可运行
  - `docker run --rm openfoam/openfoam-13-default blockMesh -help`
  - 确认 blockMesh, icoFoam, simpleFoam 等核心求解器可用
- [ ] A3. 创建通用 OpenFOAM 参照运行脚本
  - `validation/reference/run_openfoam_case.sh`
  - 支持任意教程目录，自动检测求解器
- [ ] A4. 保存参照数据到 `validation/reference/openfoam/`

**阻塞原因**: Docker Desktop 需要启动（已安装 v29.5.3）

### 阶段 B：核心基准算例参照对比 🔴

**目标**: 对核心基准算例运行 OpenFOAM 参照并逐算例对比精度

**任务清单**:
- [ ] B1. 不可压缩稳态 (incompressibleFluid)
  - [ ] cavity (Re=100, 400, 1000) — 8x8/16x16/32x32
  - [ ] planarCouette
  - [ ] planarPoiseuille
  - [ ] pitzDaily (后台阶流)
  - [ ] cylinder (绕流)
  - [ ] channel395
  - [ ] airFoil2D
  - [ ] TJunction
  - [ ] blockedChannel
  - [ ] 其余 47 个 incompressibleFluid 算例
- [ ] B2. 不可压缩瞬态 (icoFoam legacy)
  - [ ] cavity (经典)
  - [ ] cavityClipped
  - [ ] cavityGrade
  - [ ] elbow
- [ ] B3. 多相流 (incompressibleVoF)
  - [ ] damBreakLaminar (基准)
  - [ ] damBreak3D
  - [ ] sloshingTank2D
  - [ ] 其余 36 个 VoF 算例
- [ ] B4. 可压缩流 (fluid/shockFluid)
  - [ ] shockTube (Sod 激波管)
  - [ ] forwardStep
  - [ ] cavity (可压缩)
  - [ ] 其余可压缩算例
- [ ] B5. 传热 (buoyantCavity, hotRoom 等)
  - [ ] buoyantCavity
  - [ ] hotRoom
  - [ ] hotRoomBoussinesq
  - [ ] 其余传热算例
- [ ] B6. 多相 Euler (multiphaseEuler)
  - [ ] bubbleColumn
  - [ ] damBreak4phase
  - [ ] fluidisedBed
  - [ ] 其余 24 个 Euler 算例
- [ ] B7. 复杂化学反应 (multicomponentFluid)
  - [ ] counterFlowFlame2D
  - [ ] aachenBomb
  - [ ] 其余 17 个反应算例
- [ ] B8. 特殊算例
  - [ ] potentialFoam (2 cases)
  - [ ] solidDisplacement (2 cases)
  - [ ] XiFluid (5 cases)
  - [ ] compressibleVoF (9 cases)
  - [ ] legacy 算例 (21 cases)
  - [ ] multiRegion/CHT (13 cases)
  - [ ] multiRegion/film (8 cases)

### 阶段 C：修复有精度问题的求解器 🔴

**目标**: 修复所有精度不达标的求解器

**任务清单**:
- [ ] C1. PISO Couette 流精度 (87-95% → <5%)
  - 根因分析：矩阵级 BC 是否已应用到 PISO
  - 可能需要调整时间步长/网格
- [ ] C2. Poiseuille 流发散问题
  - 分析发散原因（可能是边界条件设置）
  - 确保 SIMPLE 和 PISO 都能收敛
- [ ] C3. SIMPLE Foam NaN 问题
  - RhoSimpleFoam, MulticomponentFluidFoam, RhoPorousSimpleFoam
  - 调整松弛因子或初始条件
- [ ] C4. 缺失场文件修复
  - TwoPhaseEulerFoam: 需要 U1 场
  - CavitatingFoam: 需要 alpha.vapor 场
  - IncompressibleDriftFluxFoam: 需要 alpha 场
  - AcousticFoam: 需要 p' 场
- [ ] C5. 张量处理 bug 修复
  - MagneticFoam: 张量转换错误
  - MhdFoam: 张量索引错误
- [ ] C6. 数值稳定性修复
  - IsothermalFluidFoam: field_max 3.6e160
  - CompressibleVoFFoam: continuity 8078

### 阶段 D：GPU 全精度验证 ⚠️

**目标**: 所有求解器在 GPU 上精度达标

**任务清单**:
- [ ] D1. 运行全部 50+ 求解器 GPU 验证（精度，不只是有限值）
- [ ] D2. 对比 CPU vs GPU 结果一致性
- [ ] D3. 修复 GPU 特有的精度/稳定性问题

### 阶段 E：可微分模拟完善 ⚠️

**目标**: 端到端可微分模拟在所有网格尺寸下精度达标

**任务清单**:
- [ ] E1. 验证 32x32+ 可微分梯度精度
- [ ] E2. 形状优化端到端测试
- [ ] E3. 可微分 SIMPLE/PISO 验证

### 阶段 F：全面教程验证与报告 🔴

**目标**: 所有 267 个教程算例验证并生成逐算例报告

**任务清单**:
- [ ] F1. 批量运行全部 267 算例
  - 自动化脚本：读取 OpenFOAM 教程目录，运行 pyOpenFOAM 求解器
  - 记录：求解器、网格尺寸、收敛性、关键物理量
- [ ] F2. 逐算例精度对比
  - 与 OpenFOAM 参照结果对比
  - 与解析解对比（如有）
  - 计算 L2 误差
- [ ] F3. 生成逐算例报告
  - `validation/results/per_case_report.md`
  - 每个算例：状态、精度、GPU 支持、可微分支持
- [ ] F4. 更新 validation_report.md
- [ ] F5. commit + push

## 执行策略

- **环境**: `F:/f/pyopenfoam-gpu/python.exe`（conda pyopenfoam-gpu 环境）
- **参照**: Docker OpenFOAM-13 或 v12
- **验证数据**: `validation/reference/` 仅保留最后一次成功数据
- **提交**: 每完成一个子任务立即 commit + push
- **并行**: 独立算例用子代理并行运行

## 优先级

1. **A** (阻塞) → 启动 Docker + OpenFOAM 参照
2. **B** (核心) → 267 算例参照对比
3. **C** (修复) → 求解器精度/稳定性修复
4. **D** (GPU) → GPU 全精度验证
5. **E** (可微分) → 可微分完善
6. **F** (报告) → 最终逐算例报告

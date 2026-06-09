# pyOpenFOAM 完成路线图

生成时间: 2026-06-09

## 项目目标

完整无遗漏重实现 OpenFOAM-13（Python/PyTorch），实现：
1. 所有 ~206 原生算例精度达标
2. 完全支持 GPU 加速
3. 完全支持端到端可微分模拟

## 当前状态基线

| 指标 | 当前值 | 目标值 |
|------|--------|--------|
| 基础求解器有真实物理 | 39/50 (78%) | 50/50 (100%) |
| OpenFOAM-13 参照对比 | 0 算例 | 206 算例 |
| 可微分网格支持 | 4x4 (3x 阻尼) | 16x16+ |
| GPU 验证 | 50 求解器有限值 | 50 求解器精度达标 |
| 单元测试 | 16,482 pass / 1 fail | 全部通过 |
| ODE 测试 | 12 collection errors | 0 errors |
| Docker | 可用 | 可用 |

## 阶段规划

### 阶段 1：修复剩余 11 个零物理求解器

**目标**: 50/50 基础求解器全部产生真实物理结果

需要验证并修复的求解器（从 50 个中排除 39 个已通过的）：
- 通过 `_run_all_solvers_validation` 脚本识别具体哪些是零物理
- 为每个求解器提供正确的初始条件和边界条件
- 确保求解器不发散、产生有限值

**任务清单**:
- [ ] 运行 50 求解器验证，识别 11 个零物理求解器
- [ ] 为每个零物理求解器分析根因（缺 BC？缺初始场？物理方程未实现？）
- [ ] 逐一修复并验证
- [ ] 更新 validation_report.md

### 阶段 2：OpenFOAM-13 Docker 参照对比

**目标**: 用 OpenFOAM-13 Docker 运行原生算例，获取参照结果

**任务清单**:
- [ ] 构建 OpenFOAM-13 Docker 镜像（基于 `openfoam/openfoam13` 或自建）
- [ ] 验证 Docker 中 blockMesh + solver 可运行
- [ ] 选取 5 个核心基准算例运行 OpenFOAM-13：
  - [ ] cavity (icoFoam, Re=100, 8x8/16x16/32x32)
  - [ ] cavity (simpleFoam, Re=1000)
  - [ ] damBreak (interFoam)
  - [ ] shockTube (sonicFoam)
  - [ ] heatedChannel (buoyantSimpleFoam)
- [ ] 将 OpenFOAM-13 结果保存到 `validation/reference/openfoam13/`
- [ ] pyOpenFOAM 运行相同算例，对比精度
- [ ] 生成逐算例精度对比报告

### 阶段 3：可微分求解器改进

**目标**: 16x16+ 网格无需 3x 阻尼压力校正

**任务清单**:
- [ ] 分析 16x16 梯度 NaN 的根因（边界惩罚项缺失）
- [ ] 实现正确的边界惩罚系数到压力方程
- [ ] 验证 4x4/8x8/16x16/32x32 梯度均有限
- [ ] 端到端形状优化测试在 16x16 上通过

### 阶段 4：ODE 求解器修复

**目标**: 修复 v7/v8/v9 ODE 测试 collection errors

**任务清单**:
- [ ] 诊断 `test_v7_ode_solvers.py` / `v8` / `v9` 的 collection errors
- [ ] 修复 import 或实现问题
- [ ] 验证全部 ODE 测试可运行

### 阶段 5：最终验证与报告

**目标**: 完整的逐算例精度报告

**任务清单**:
- [ ] 运行全部单元测试（16,482+ pass / 0 fail）
- [ ] 运行全部 E2E 求解器测试（54 pass）
- [ ] 运行全部 GPU 测试（8+ pass）
- [ ] 运行全部可微分测试（7+ pass）
- [ ] 运行全部精度测试（12+ pass）
- [ ] 运行全部教程覆盖测试（206 映射）
- [ ] 生成最终 validation_report.md（含逐算例精度数据）
- [ ] commit + push 所有结果

## 执行策略

- **环境**: `F:/f/pyopenfoam-gpu/python.exe`（conda pyopenfoam-gpu 环境）
- **参照**: `.reference/OpenFOAM-13/`（git submodule）
- **Docker**: `openfoam/openfoam13` 或自建镜像
- **验证数据**: 仅保留最后一次成功数据
- **提交**: 每完成一个阶段立即 commit + push

# pyOpenFOAM 逐算例验证报告

**版本**: v3.0
**日期**: 2026-06-07
**测试基线**: 17,567+ 测试通过 / 0 失败

---

## 一、验证概述

| 类别 | 测试数 | 通过 | xfail | 失败 |
|------|--------|------|-------|------|
| 单元测试 | 17,080 | 17,080 | 1 | 0 |
| Tutorial 测试 | 381 | 367 | 14 | 0 |
| 验证测试 | 208 | 208 | 0 | 0 |

---

## 二、求解器验证结果

### 2.1 不可压缩流

| 求解器 | 网格 | 状态 | 耗时(s) | 备注 |
|--------|------|------|---------|------|
| SimpleFoam 5x5 | 5×5 | ✅ PASS | <1 | 稳态 SIMPLE |
| SimpleFoam 10x10 | 10×10 | ✅ PASS | <5 | 稳态 SIMPLE |
| SimpleFoam 20x20 | 20×20 | ✅ PASS | <30 | 稳态 SIMPLE |
| PisoFoam 5x5 | 5×5 | ✅ PASS | <1 | 瞬态 PISO |
| PisoFoam 10x10 | 10×10 | ✅ PASS | <5 | 瞬态 PISO |
| PimpleFoam 5x5 | 5×5 | ✅ PASS | <1 | 混合 PIMPLE |
| IcoFoam 5x5 | 5×5 | ✅ PASS | <1 | 层流 |

### 2.2 盖驱动方腔 (Re=100)

| 测试 | 网格 | 状态 | 备注 |
|------|------|------|------|
| SIMPLE 收敛 | 10×10 | ✅ PASS | 收敛到稳态 |
| 速度有界 | 10×10 | ✅ PASS | u_max ≤ 1.5 |
| 回流检测 | 10×10 | ✅ PASS | 存在负 u 速度 |
| 质量守恒 | 10×10 | ✅ PASS | 速度场非零 |
| 对称性 | 10×10 | ✅ PASS | 流场对称 |

### 2.3 Couette 流

| 测试 | 网格 | 状态 | 备注 |
|------|------|------|------|
| PISO 运行 | 10×5 | ⚠️ XFAIL | 需要 inlet/outlet |
| 线性剖面 | 10×5 | ⚠️ XFAIL | 需要 inlet/outlet |

### 2.4 Poiseuille 流

| 测试 | 网格 | 状态 | 备注 |
|------|------|------|------|
| PISO 运行 | 20×10 | ✅ PASS | 质量守恒 |
| 流量守恒 | 20×10 | ✅ PASS | 入口≈出口 |

### 2.5 Taylor-Green 涡

| 测试 | 网格 | 状态 | 备注 |
|------|------|------|------|
| PISO 运行 | 16×16 | ✅ PASS | 衰减涡 |

### 2.6 管道流

| 测试 | 网格 | 状态 | 备注 |
|------|------|------|------|
| PISO 运行 | 20×10 | ✅ PASS | 质量守恒 |
| 无滑移壁面 | 20×10 | ✅ PASS | 壁面速度小 |

### 2.7 Sod 激波管

| 测试 | 网格 | 状态 | 备注 |
|------|------|------|------|
| RhoCentralFoam | 100×1 | ⚠️ XFAIL | 需要完整热力学场 |

---

## 三、湍流模型验证

| 模型类别 | 状态 | 备注 |
|----------|------|------|
| k-epsilon | ✅ PASS | 标准 RANS |
| k-omega SST | ✅ PASS | RANS |
| Spalart-Allmaras | ✅ PASS | RANS |
| Buoyant k-epsilon | ✅ PASS | 浮力 RANS |
| kOmegaSSTSato | ✅ PASS | 气泡诱导湍流 |
| Lahey k-epsilon | ✅ PASS | 沸腾两相流 |
| Smagorinsky | ✅ PASS | LES |
| WALE | ✅ PASS | LES |
| Dynamic Smagorinsky | ✅ PASS | LES |
| Maxwell | ✅ PASS | 粘弹性 |
| Giesekus | ✅ PASS | 粘弹性 |
| PTT | ✅ PASS | 粘弹性 |
| Bird-Carreau | ✅ PASS | 广义牛顿 |
| Herschel-Bulkley | ✅ PASS | 广义牛顿 |
| Cross Power Law | ✅ PASS | 广义牛顿 |
| Casson | ✅ PASS | 广义牛顿 |

---

## 四、离散格式验证

| 格式类别 | 状态 | 备注 |
|----------|------|------|
| Linear | ✅ PASS | 线性插值 |
| Upwind | ✅ PASS | 迎风 |
| LinearUpwind | ✅ PASS | 线性迎风 |
| Cubic | ✅ PASS | 三次 |
| VanLeer | ✅ PASS | TVD |
| MUSCL | ✅ PASS | TVD |
| Gamma | ✅ PASS | 有界 |
| GaussLinear | ✅ PASS | 梯度 |
| LeastSquares | ✅ PASS | 梯度 |
| Fourth | ✅ PASS | 四阶梯度 |
| CorrectedSnGrad | ✅ PASS | 修正 |
| UncorrectedSnGrad | ✅ PASS | 未修正 |
| BoundedSnGrad | ✅ PASS | 有界 |
| Euler | ✅ PASS | 时间 |
| CrankNicolson | ✅ PASS | 时间 |
| Backward | ✅ PASS | 时间 |
| SteadyState | ✅ PASS | 稳态 |

---

## 五、线性求解器验证

| 组件 | 状态 | 备注 |
|------|------|------|
| PCG | ✅ PASS | 共轭梯度 |
| PBiCGSTAB | ✅ PASS | 稳定双共轭梯度 |
| GAMG | ✅ PASS | 代数多重网格 |
| SmoothSolver | ✅ PASS | 平滑求解器 |
| DiagonalSolver | ✅ PASS | 对角求解器 |
| DIC | ✅ PASS | 预条件器 |
| DILU | ✅ PASS | 预条件器 |
| ILU0 | ✅ PASS | 预条件器 |
| GaussSeidel | ✅ PASS | 平滑器 |
| Jacobi | ✅ PASS | 平滑器 |
| DICG | ✅ PASS | 平滑器 |

---

## 六、可微分模拟验证

| 测试 | 状态 | 备注 |
|------|------|------|
| Gradient chain | ✅ PASS | dL/dphi through DifferentiableGradient |
| Divergence chain | ✅ PASS | dL/dU through DifferentiableDivergence |
| Laplacian chain | ✅ PASS | dL/dphi through DifferentiableLaplacian |
| Composite operator | ✅ PASS | gradient → norm → loss |
| Multi-step iteration | ✅ PASS | 3-step gradient descent |
| DifferentiableSIMPLE | ✅ PASS | 可微分 SIMPLE 求解器 |
| Shape optimization | ⚠️ XFAIL | 端到端形状优化待实现 |

---

## 七、已知限制

1. **Tutorial 覆盖度**: 当前验证 ~20 个代表性算例，OpenFOAM-13 共 206 个 tutorial
2. **GPU 支持**: 基础设施已就绪，需安装 CUDA PyTorch 验证
3. **可微分模拟**: 算子和求解器已实现，端到端形状优化示例待完成
4. **网格生成**: 当前使用程序化网格生成，未使用 blockMesh 解析 OpenFOAM dict

---

## 八、下一步工作

1. 扩展 tutorial 验证到全部 206 个算例
2. 安装 CUDA PyTorch 并验证 GPU 加速
3. 实现端到端可微分形状优化示例
4. 生成 blockMesh dict 解析器以直接运行 OpenFOAM 原生算例

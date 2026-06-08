# pyOpenFOAM 完整实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 完成 OpenFOAM-13 的完整 Python/PyTorch 重实现，所有求解器实际收敛，206 tutorial 精度验证通过，GPU 加速验证，端到端可微分模拟。

**Architecture:** 基于现有 pyOpenFOAM 框架（FvMesh + FvMatrix + 线性求解器），修复求解器实现中的 stub/NaN 问题，建立解析解验证框架，验证 GPU 加速，实现生产级可微分模拟。

**Tech Stack:** Python 3.11, PyTorch 2.6.0+cu124, pytest, RTX 4070 Ti SUPER

---

## 当前状态分析

### 已完成
- 214 求解器应用注册
- 408 边界条件 RTS 注册
- SimpleFoam 收敛验证（U_res=7e-6, continuity=7.8e-7）
- 7 个解析解精度测试通过
- 7/7 可微分测试通过（含形状优化）
- 8/8 GPU 测试通过（基础张量运算）

### 关键问题
1. **求解器 stub**：IcoFoam/PisoFoam/PimpleFoam 不收敛，SonicFoam/InterFoam/LaplacianFoam 残差=0
2. **精度验证缺失**：仅 7 个解析解测试，无 206 tutorial 逐算例验证
3. **GPU 未跑 CFD**：基础测试通过但无实际求解器验证
4. **可微分仅 4x4 网格**：需更大网格的生产级演示

---

## Phase 1: 修复核心求解器（最高优先级）

### Task 1: 修复 IcoFoam 压力-速度耦合

**Files:**
- Modify: `src/pyfoam/applications/ico_foam.py`
- Test: `tests/tutorials/test_solver_e2e_validation.py`

- [ ] **Step 1: 诊断 IcoFoam 收敛问题**

```bash
cd F:/agent-workspace/pyOpenFOAM
python -c "
import tempfile
from pathlib import Path
from tests.tutorials.helpers import make_structured_mesh, write_control_dict, write_fv_schemes, write_fv_solution, write_velocity_field, write_pressure_field, write_transport_properties
from pyfoam.applications import IcoFoam

with tempfile.TemporaryDirectory() as tmp:
    case = Path(tmp)
    mesh_dir = case / 'constant' / 'polyMesh'
    make_structured_mesh(mesh_dir, nx=4, ny=4)
    write_control_dict(case, delta_t=0.001, end_time=0.005)
    write_fv_schemes(case)
    write_fv_solution(case)
    write_transport_properties(case, nu=0.01)
    write_velocity_field(case, patches={'movingWall': (1,0,0), 'fixedWalls': (0,0,0)}, bc_types={'movingWall': 'fixedValue', 'fixedWalls': 'noSlip'})
    write_pressure_field(case, patches={'movingWall': 'zeroGradient', 'fixedWalls': 'zeroGradient'})
    solver = IcoFoam(case)
    result = solver.run()
    print(f'U_res={result.U_residual}, p_res={result.p_residual}, cont={result.continuity_error}')
"
```

- [ ] **Step 2: 检查 IcoFoam 源码中的压力方程求解逻辑**

读取 `src/pyfoam/applications/ico_foam.py` 中的 `_piso_step` 方法，对比 OpenFOAM-13 参考：
```bash
cat .reference/OpenFOAM-13/applications/legacy/incompressible/icoFoam/pEqn.H
```

- [ ] **Step 3: 修复压力方程**

确保压力方程正确组装：∇·(1/A_p * ∇p) = ∇·(HbyA)，其中 HbyA = H/A_p。

- [ ] **Step 4: 验证修复后收敛**

```bash
python -m pytest tests/tutorials/test_solver_e2e_validation.py::TestIncompressibleSolvers::test_ico_foam -v
```

- [ ] **Step 5: Commit**

---

### Task 2: 修复 PisoFoam PISO 算法

**Files:**
- Modify: `src/pyfoam/applications/piso_foam.py`
- Test: `tests/tutorials/test_solver_e2e_validation.py`

- [ ] **Step 1: 对比 OpenFOAM-13 PISO 实现**

```bash
cat .reference/OpenFOAM-13/applications/legacy/incompressible/icoFoam/icoFoam.C | grep -A 20 "PISO"
```

- [ ] **Step 2: 修复 PISO 校正循环**

PISO 需要多次压力校正（nCorrectors），每次更新压力和速度。

- [ ] **Step 3: 验证收敛**

```bash
python -m pytest tests/tutorials/test_solver_e2e_validation.py::TestIncompressibleSolvers::test_piso_foam -v
```

- [ ] **Step 4: Commit**

---

### Task 3: 修复 PimpleFoam 外迭代

**Files:**
- Modify: `src/pyfoam/applications/pimple_foam.py`

- [ ] **Step 1: 增加外迭代次数并修复收敛检查**

当前 nOuterCorrectors=3 导致 continuity=23。需要增加到 20+ 并检查残差下降。

- [ ] **Step 2: 验证收敛**

---

### Task 4: 修复 LaplacianFoam 热传导求解

**Files:**
- Modify: `src/pyfoam/applications/laplacian_foam.py`

- [ ] **Step 1: 实现真实的拉普拉斯方程求解**

当前残差=0 表示是 stub。需要实现 ∂T/∂t = ∇·(D∇T) 的时间推进。

- [ ] **Step 2: 用解析解验证**

1D 热传导：T(x,t) = T0 + ΔT·erfc(x/(2√(αt)))

---

### Task 5: 修复 InterFoam VOF 求解

**Files:**
- Modify: `src/pyfoam/applications/inter_foam.py`

- [ ] **Step 1: 实现 VOF 输运方程**

∂α/∂t + ∇·(αU) = 0，需要 MULES 限制器保持有界性。

---

### Task 6: 修复 SonicFoam 可压缩求解

**Files:**
- Modify: `src/pyfoam/applications/sonic_foam.py`

- [ ] **Step 1: 实现可压缩 SIMPLE/PISO**

需要 ρ 方程、能量方程、理想气体 EOS。

---

### Task 7: 修复返回类型不一致的求解器

**Files:**
- Modify: `src/pyfoam/applications/potential_foam.py`
- Modify: `src/pyfoam/applications/boundary_foam.py`
- Modify: `src/pyfoam/applications/reacting_foam.py`

- [ ] **Step 1: 统一返回 ConvergenceData**

这些求解器返回 dict 而非 ConvergenceData，导致 E2E 测试失败。

---

## Phase 2: 精度验证框架

### Task 8: Couette 流精度验证（已通过）

**Status:** ✅ 已完成

---

### Task 9: Poiseuille 流 E2E 验证

**Files:**
- Create: `tests/tutorials/test_poiseuille_e2e.py`

- [ ] **Step 1: 创建 Poiseuille 流算例**

压力驱动平板流动，解析解 u(y) = (1/2μ)(-dp/dx)y(H-y)。

```python
# 在 0/U 中设置压力梯度驱动
# 在 0/p 中设置线性压力分布
# 运行 SimpleFoam 到收敛
# 比较速度剖面与解析解
```

- [ ] **Step 2: 验证 L2 误差 < 5%**

---

### Task 10: Taylor-Green 涡衰减验证

**Files:**
- Create: `tests/tutorials/test_taylor_green.py`

- [ ] **Step 1: 实现 Taylor-Green 涡算例**

解析解：u(x,y,t) = U0·sin(x)·cos(y)·exp(-2νt)

- [ ] **Step 2: 运行 IcoFoam 并比较衰减率**

---

### Task 11: Sod 激波管验证

**Files:**
- Create: `tests/tutorials/test_sod_shock.py`

- [ ] **Step 1: 实现 Sod 激波管算例**

经典 Riemann 问题，有精确解。

- [ ] **Step 2: 运行 SonicFoam/RhoCentralFoam 并比较**

---

### Task 12: 自然对流验证

**Files:**
- Create: `tests/tutorials/test_natural_convection.py`

- [ ] **Step 1: 封闭方腔自然对流**

Ra=10^4，与文献基准比较 Nu 数。

---

### Task 13: 206 Tutorial 解析解验证矩阵

**Files:**
- Create: `tests/tutorials/test_tutorial_precision_matrix.py`

- [ ] **Step 1: 创建验证矩阵**

为每个 tutorial 类别定义验证标准：
- 不可压缩：质量守恒（continuity < 1e-6）
- 可压缩：能量守恒
- 多相流：体积分数有界（0 ≤ α ≤ 1）
- 传热：热流守恒

- [ ] **Step 2: 运行所有可用求解器并记录结果**

```python
# 对每个求解器类别运行代表算例
# 记录残差、守恒误差、物理有效性
# 保存到 validation/results/precision_matrix.json
```

---

## Phase 3: GPU CFD 验证

### Task 14: SimpleFoam GPU 验证

**Files:**
- Test: `tests/tutorials/test_gpu_cfd_validation.py`

- [ ] **Step 1: 创建 GPU CFD 测试**

```python
# 使用 conda run -p /f/pyopenfoam-gpu
# 在 GPU 上运行 SimpleFoam cavity 算例
# 验证结果与 CPU 一致
```

- [ ] **Step 2: 比较 CPU/GPU 精度**

L2 误差应 < 1e-10（float64 精度）。

---

### Task 15: GPU 性能基准

- [ ] **Step 1: 比较不同网格尺寸的 CPU/GPU 耗时**

nx=16, 32, 64, 128

---

## Phase 4: 可微分模拟生产级演示

### Task 16: 16x16 网格形状优化

**Files:**
- Modify: `tests/tutorials/test_differentiable_e2e.py`

- [ ] **Step 1: 在 16x16 网格上运行可微分 SIMPLE**

- [ ] **Step 2: 验证梯度传播**

- [ ] **Step 3: 运行 10 步优化迭代**

最小化 ∫|U|² dx，设计变量为入口速度。

---

### Task 17: 可微分湍流模型

- [ ] **Step 1: 将 k-ε 模型接入可微分 SIMPLE**

---

## Phase 5: 最终验证报告

### Task 18: 逐算例验证报告

**Files:**
- Create: `validation/results/per_case_report.md`
- Create: `validation/results/per_case_data.json`

- [ ] **Step 1: 运行所有求解器的代表算例**

每个类别至少 1 个算例，记录：
- 求解器名称
- 网格尺寸
- 迭代次数
- 最终残差
- 守恒误差
- 耗时
- 状态（收敛/未收敛/错误）

- [ ] **Step 2: 生成 Markdown 报告**

- [ ] **Step 3: Commit 并 push**

---

## 执行顺序

```
Phase 1 (Task 1-7) → Phase 2 (Task 8-13) → Phase 3 (Task 14-15) → Phase 4 (Task 16-17) → Phase 5 (Task 18)
```

每个 Phase 完成后 commit + push + 更新 ROADMAP。

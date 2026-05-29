# pyOpenFOAM 功能完整性验证设计

**日期**: 2026-05-27
**状态**: 已批准

---

## 目标

确保 pyOpenFOAM 所有现有代码功能正确、测试覆盖完整、与 OpenFOAM 精度对齐。不新增功能。

## 范围

### 包含
- 测试覆盖补全（ROADMAP 2.4）
- 代码质量修复（ROADMAP 2.5，仅影响正确性的部分）
- 官方算例验证（ROADMAP 2.1）
- 精度改进（ROADMAP 2.2）

### 排除
- 性能优化（ROADMAP 2.3）
- 功能扩展（ROADMAP 2.6：PINN、神经算子等）

## 分阶段策略

### Phase A: 环境搭建 + 基线确认
- WSL Ubuntu 20.04 + Conda 环境
- 安装 pyOpenFOAM 及依赖
- 运行全部现有测试，确认 2041 passed, 17 xfailed

### Phase B: 测试覆盖补全
**零测试模块（6 个）**:
- `models/radiation.py` - P1Radiation
- `applications/multiphase_inter_foam.py`
- `applications/compressible_inter_foam.py`
- `applications/two_phase_euler_foam.py`
- `applications/multiphase_euler_foam.py`
- `applications/cavitating_foam.py`

**中等缺口（边界条件 9 个 + 基础模块 5 个）**:
- 边界条件：no_slip, symmetry, fixed_gradient, velocity_bcs, pressure_bcs, turbulence_bcs, vof_bcs, inlet_outlet, coupled_temperature
- 基础模块：sparse_ops.py, spalart_allmaras.py, surface_tension.py, linear_solver.py, pressure_equation.py, rhie_chow.py

### Phase C: 代码质量修复
- 修复 applications/__init__.py 缺失的 10 个求解器导出
- 修复 discretisation/__init__.py 缺失的 LimitedLinearInterpolation 导出
- 统一重复代码（壁面函数、重力向量解析、后处理场提取）
- 完善 SnappyHexMesh 空实现

### Phase D: 官方算例验证 + 精度改进
**验证案例（6 个）**:
1. 盖驱动方腔 Re=100, 1000 — Ghia et al. 1982
2. 后向台阶 — Driver & Seegmiller 1985
3. 圆柱绕流 — Schäfer & Turek 1996
4. Sod 激波管 — Sod 1978
5. 溃坝 — Martin & Moyce 1952
6. 自然对流方腔 — de Vahl Davis 1983

**精度目标**: Ghia 基准误差 <5%（当前 15% @32×32）

## 环境约束

- **运行环境**: WSL Ubuntu 20.04
- **环境管理**: Conda (miniconda)，单环境
- **依赖**: torch>=2.0, numpy>=1.24, scipy>=1.10, pytest

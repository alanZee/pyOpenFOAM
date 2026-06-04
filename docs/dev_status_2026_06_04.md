# pyOpenFOAM 开发进度 — 2026-06-04

## 已完成的代码修改（待验证）

### 1. 边界通量修正（Couette/Poiseuille 精度问题）
**根因**: `compute_face_flux_HbyA` 对边界单元使用单元中心速度，而非 prescribed BC 速度。
封闭域中导致非零虚假边界通量，破坏压力方程 RHS。

**修改文件**:
- `src/pyfoam/solvers/piso.py` — 添加 `_fix_boundary_flux()`
- `src/pyfoam/solvers/simple.py` — 同上
- `src/pyfoam/solvers/pimple.py` — 同上
- `tests/validation/test_couette_flow.py` — 移除 "Known limitation"

### 2. 延迟修正混合（backward step 收敛问题）
**根因**: TVD van Leer 延迟修正以全强度 (λ=1.0) 施加，粗网格高 Re 下引起残差振荡。

**修改文件**:
- `src/pyfoam/solvers/simple.py` — 添加 `dc_blend=0.5`，清理无用 `lu_correction` 代码
- `tests/validation/test_backward_facing_step.py` — 更新松弛因子

### 3. 定量基准测试代码
- `validation/cases/icofoam_benchmarks.py` — 新建
- `validation/run_icofoam_benchmarks.py` — 新建

## 待完成（需 bash/PowerShell）

1. `pytest tests/ -q --tb=short` — 验证所有修改
2. `python validation/run_icofoam_benchmarks.py` — 定量精度基准
3. DifferentiableSIMPLE 大规模验证
4. `git commit && git push` — 推送到 GitHub
5. 更新验证报告

## 环境问题

Git Bash (MSYS2) DLL 初始化失败。已配置 PowerShell 替代方案：
- `settings.json` 中添加 `"defaultShell": "powershell"` 和 `"CLAUDE_CODE_USE_POWERSHELL_TOOL": "1"`
- 需重启 Claude Code 生效

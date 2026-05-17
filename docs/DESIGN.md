# pyOpenFOAM 顶层设计文档

**版本**: v1.0
**日期**: 2026-05-17

---

## 一、项目概述

pyOpenFOAM 是 OpenFOAM v2512 的纯 Python 重写版本，使用 PyTorch 作为张量后端实现 GPU 加速。项目目标是完全复刻 OpenFOAM 的有限体积法 (FVM) 求解器，同时利用 PyTorch 的自动微分能力实现可微分 CFD。

---

## 二、模块架构

```
pyfoam/
├── core/               # 基础层
│   ├── device.py       # DeviceManager, TensorConfig
│   ├── dtype.py        # CFD_DTYPE, INDEX_DTYPE
│   ├── backend.py      # scatter_add, gather, sparse 操作
│   ├── ldu_matrix.py   # LDU 稀疏矩阵格式
│   ├── fv_matrix.py    # FVM 矩阵（源项、边界条件、松弛）
│   ├── sparse_ops.py   # LDU→COO 转换、CSR matvec
│   └── multi_gpu.py    # 多 GPU 管理
│
├── mesh/               # 网格表示
│   ├── poly_mesh.py    # 原始拓扑（点、面、owner/neighbour）
│   ├── fv_mesh.py      # 扩展几何量（体积、面积、法向量）
│   ├── mesh_geometry.py
│   └── topology.py
│
├── fields/             # 场类
│   ├── vol_fields.py   # volScalarField, volVectorField
│   ├── surface_fields.py
│   ├── geometric_field.py
│   └── field_arithmetic.py
│
├── boundary/           # 边界条件（20+ 种）
│   ├── boundary_condition.py  # RTS 注册表
│   ├── fixed_value.py, zero_gradient.py, cyclic.py, ...
│   ├── velocity_bcs.py, pressure_bcs.py, turbulence_bcs.py
│   └── coupled_temperature.py
│
├── io/                 # I/O 层
│   ├── foam_file.py, dictionary.py, field_io.py, mesh_io.py
│   └── binary_io.py
│
├── discretisation/     # 离散化层
│   ├── operators.py    # fvm/fvc.grad, div, laplacian, ddt
│   ├── interpolation.py
│   └── schemes/
│
├── solvers/            # 求解器层
│   ├── simple.py, piso.py, pimple.py
│   ├── pressure_equation.py, rhie_chow.py
│   ├── pcg.py, pbicgstab.py, gamg.py
│   └── preconditioners.py
│
├── turbulence/         # 湍流模型
│   ├── k_epsilon.py, k_omega_sst.py, spalart_allmaras.py
│   ├── smagorinsky.py, wale.py
│   ├── k_omega.py, launder_sharma_ke.py, v2f.py
│   └── wall_functions/
│
├── thermophysical/     # 热力学模型
│   ├── perfect_gas.py, sutherland.py
│   ├── janaf.py, h_const.py
│   └── he_psi_thermo.py, he_rho_thermo.py
│
├── multiphase/         # 多相流模型
│   ├── vof.py, mules.py
│   └── interfacial_models/
│
├── parallel/           # 并行支持
│   ├── decomposition.py, processor_patch.py
│   ├── parallel_field.py, parallel_solver.py
│   └── parallel_io.py
│
├── applications/       # 应用求解器（30+ 个）
│   ├── solver_base.py
│   ├── simple_foam.py, ico_foam.py, piso_foam.py, ...
│   ├── rho_simple_foam.py, sonic_foam.py, ...
│   ├── inter_foam.py, two_phase_euler_foam.py, ...
│   └── laplacian_foam.py, cht_multi_region_foam.py
│
├── postprocessing/     # 后处理
│   ├── function_object.py, forces.py, wall_shear_stress.py
│   ├── sampling.py, vtk_output.py
│   └── field_operations.py
│
├── differentiable/     # 可微分 CFD
│   ├── operators.py    # DifferentiableGradient/Divergence/Laplacian
│   ├── linear_solver.py
│   └── simple.py
│
├── mesh_generation/    # 网格生成
│   ├── block_mesh.py
│   └── snappy_hex_mesh.py
│
├── mesh_conversion/    # 网格转换
│   ├── gmsh_to_foam.py
│   ├── fluent_mesh_to_foam.py
│   └── foam_to_vtk.py
│
├── models/             # 空（桩）
└── utils/              # 空（桩）
```

---

## 三、关键设计决策

### 3.1 矩阵量纲

所有矩阵系数和源项统一为**单位体积形式**（÷V），确保量纲一致。

### 3.2 边界条件

- 隐式 BC：使用 `internalCoeffs`/`boundaryCoeffs` 替代罚函数法
- 边界 delta：使用 2×d_P 使边界 delta 等于内部 delta
- 零梯度 BC：不对矩阵做贡献

### 3.3 SIMPLE/SIMPLEC

- SIMPLEC：`A_p_eff = A_p - H1`（非 `rAtU`）
- H1 计算：矩阵系数已为单位体积形式，不再除以单元体积

### 3.4 可微分 CFD

- 梯度/散度/Laplacian 使用 `torch.autograd.Function` 实现自定义前向/反向传播
- 线性求解器使用隐式微分：`∂L/∂b = A^{-T} ∂L/∂x`
- SIMPLE 使用隐函数定理计算梯度

---

## 四、技术栈

| 组件 | 技术 |
|------|------|
| 张量后端 | PyTorch (CPU/CUDA/MPS) |
| 线性代数 | PyTorch sparse, scipy.sparse |
| 并行 | MPI (mpi4py) |
| 网格生成 | 自研 blockMesh/snappyHexMesh |
| I/O | OpenFOAM 原生格式 (ASCII + binary) |
| 测试 | pytest |
| 文档 | Markdown (中英双语) |

---

## 五、参考资源

- **OpenFOAM 源码**: https://github.com/OpenFOAM/OpenFOAM-dev
- **API 文档**: https://api.openfoam.com/2512/
- **用户指南**: https://www.openfoam.com/documentation/user-guide
- **教程指南**: https://www.openfoam.com/documentation/tutorial-guide

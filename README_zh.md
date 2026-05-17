<div align="center">

# pyOpenFOAM

**基于 PyTorch GPU 加速的纯 Python CFD 求解器**

*OpenFOAM v2512 的开源 Python 重写版本*

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/tests-2041+-brightgreen.svg)](#测试)

[English](README.md) | [中文](#概述)

</div>

---

## 概述

**pyOpenFOAM** 是 [OpenFOAM](https://www.openfoam.com/) v2512 的开源 Python 重写版本。OpenFOAM 是广泛使用的 C++ 计算流体力学 (CFD) 工具箱。我们的目标是将 OpenFOAM 的能力带入 Python 生态系统，同时利用 PyTorch 实现 GPU 加速和自动微分。

### 核心特性

- **30+ 个 OpenFOAM 求解器** — 不可压缩、可压缩、多相、热传导等
- **GPU 加速** — 所有场操作使用 PyTorch 张量，支持 CUDA/MPS
- **可微分 CFD** — 通过自定义 autograd 函数支持 `torch.autograd`
- **OpenFOAM 兼容** — 原生读写现有 OpenFOAM 案例
- **20+ 种边界条件** — 速度、压力、湍流、VOF、热传导
- **完整湍流模型库** — RANS (k-ε, k-ω SST, S-A, v2f)、LES (Smagorinsky, WALE)、DES
- **网格工具** — blockMesh、snappyHexMesh、gmsh/fluent/VTK 转换器
- **MPI 并行** — 域分解、Halo 交换、并行 I/O

---

## 安装

### 依赖要求

- Python ≥ 3.10
- PyTorch ≥ 2.0
- NumPy ≥ 1.24
- SciPy ≥ 1.10

### 快速安装

```bash
git clone https://github.com/alanZee/pyOpenFOAM.git
cd pyOpenFOAM
pip install -r requirements.txt
pip install -e .
```

### GPU 支持

```bash
# CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Apple Silicon (MPS)
pip install torch  # MPS 内置支持
```

---

## 快速开始

### 运行 OpenFOAM 案例

```python
from pyfoam.applications import SimpleFoam

solver = SimpleFoam("tutorials/incompressible/simpleFoam/pitzDaily")
solver.run()
```

### GPU 加速

```python
from pyfoam.core import device_context

with device_context("cuda"):
    mesh = FvMesh.from_poly_mesh(poly_mesh)
    solver = SIMPLESolver(mesh, config)
    U, p, phi, info = solver.solve(U, p, phi)
```

### 可微分 CFD

```python
from pyfoam.differentiable import DifferentiableLaplacian, DifferentiableLinearSolve

# 可微分 Laplacian（支持 torch.autograd）
lap = DifferentiableLaplacian.apply(phi, mesh)

# 可微分线性求解器（隐式微分）
x = DifferentiableLinearSolve.apply(A, b, tol, max_iter)
```

---

## 架构

```
pyfoam/
├── core/               # 设备管理、LDU/FvMatrix、稀疏操作、多 GPU
├── mesh/               # PolyMesh、FvMesh、网格生成 (blockMesh, snappyHexMesh)
├── fields/             # volScalarField、volVectorField、surfaceScalarField
├── boundary/           # 20+ 种边界条件（速度/压力/湍流/VOF/热传导）
├── io/                 # OpenFOAM 文件格式 I/O（ASCII + 二进制）
├── discretisation/     # fvm/fvc 算子、插值格式
├── solvers/            # PCG、PBiCGSTAB、GAMG、SIMPLE/SIMPLEC/PISO/PIMPLE
├── turbulence/         # RANS、LES、DES 模型 + 壁面函数
├── thermophysical/     # 完美气体、Sutherland、JANAF、ψ/ρ-based 热力学
├── multiphase/         # VOF + MULES、interFoam、Euler-Euler、空化
├── parallel/           # MPI 域分解、Halo 交换、并行 I/O
├── applications/       # 30+ 个求解器
├── postprocessing/     # FunctionObject 框架、力计算、y+、VTK 输出
├── differentiable/     # 可微分算子、线性求解器、SIMPLE
├── mesh_generation/    # blockMesh、snappyHexMesh
└── mesh_conversion/    # gmshToFoam、fluentMeshToFoam、foamToVTK
```

---

## 已实现求解器

| 类别 | 求解器 |
|------|--------|
| **不可压缩** | simpleFoam、icoFoam、pisoFoam、pimpleFoam、SRFSimpleFoam、porousSimpleFoam、boundaryFoam |
| **可压缩** | rhoSimpleFoam、rhoPimpleFoam、sonicFoam、rhoCentralFoam |
| **浮力驱动** | buoyantSimpleFoam、buoyantPimpleFoam、buoyantBoussinesqSimpleFoam |
| **热传导** | laplacianFoam、chtMultiRegionFoam |
| **多相流** | interFoam、multiphaseInterFoam、compressibleInterFoam、twoPhaseEulerFoam、multiphaseEulerFoam、cavitatingFoam |
| **其他** | potentialFoam、scalarTransportFoam、reactingFoam、solidDisplacementFoam |

---

## 验证

| 案例 | 求解器 | L2 误差 | 参考 |
|------|--------|---------|------|
| Couette 流 | simpleFoam | 0.013% | 解析解 |
| Poiseuille 流 | simpleFoam | 0.13% | 解析解 |
| 盖驱动方腔 (Re=100) | simpleFoam | ~15% | Ghia et al. 1982 |

```bash
python validation/run_all.py
```

---

## 测试

```bash
# 运行所有测试
pytest tests/unit/ -q --tb=no

# 特定模块
pytest tests/unit/solvers/ -q
```

**结果**: 2041 passed, 17 xfailed（约 130 秒）

---

## 文档

| 文档 | 说明 |
|------|------|
| [PROPOSAL.md](docs/PROPOSAL.md) | 需求文档：目标、架构、验证基准、求解器列表 |
| [DESIGN.md](docs/DESIGN.md) | 顶层设计文档：模块架构、技术栈、关键决策 |
| [ROADMAP.md](docs/ROADMAP.md) | 后续计划：已完成工作概要、待完成工作 |
| [英文文档](docs/en/) | 入门指南、API 参考、架构设计、GPU 指南、迁移指南 |
| [中文文档](docs/zh/) | 入门指南、API 参考、架构设计、GPU 指南、迁移指南 |

---

## 贡献

欢迎贡献！优先方向：

1. **验证** — 帮助我们与 OpenFOAM 对比验证
2. **可微分性** — 扩展 autograd 支持
3. **性能** — 优化 GPU 内存和计算
4. **文档** — 改进教程和示例

---

## 许可证

pyOpenFOAM 基于 [GNU 通用公共许可证 v3.0](LICENSE) 发布。

---

<div align="center">

**为 CFD 和 Python 社区而建**

[报告 Bug](https://github.com/alanZee/pyOpenFOAM/issues) · [功能请求](https://github.com/alanZee/pyOpenFOAM/issues) · [讨论](https://github.com/alanZee/pyOpenFOAM/discussions)

</div>

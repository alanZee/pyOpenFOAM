<div align="center">

# pyOpenFOAM

**基于 PyTorch GPU 加速的纯 Python CFD 求解器**

*OpenFOAM 13 (Foundation) 的开源 Python 重写版本*

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/tests-17130+-brightgreen.svg)](#测试)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Reference_Data-ffd21e.svg)](https://huggingface.co/datasets/AlanZee/pyOpenFOAM-reference-data)

[English](README.md) | [中文](#概述)

</div>

---

## 概述

**pyOpenFOAM** 是 [OpenFOAM 13](https://openfoam.org/) (Foundation) 的开源 Python 重写版本。OpenFOAM 是广泛使用的 C++ 计算流体力学 (CFD) 工具箱。我们的目标是将 OpenFOAM 的能力带入 Python 生态系统，同时利用 PyTorch 实现 GPU 加速和自动微分。

### 核心特性

- **64 个 OpenFOAM 求解器** — 不可压缩、可压缩、多相、热传导等
- **GPU 加速** — 所有场操作使用 PyTorch 张量，支持 CUDA/MPS
- **可微分 CFD** — 通过自定义 autograd 函数支持 `torch.autograd`
- **OpenFOAM 兼容** — 原生读写现有 OpenFOAM 案例
- **408+ 种边界条件** — 速度、压力、湍流、VOF、热传导
- **完整湍流模型库** — RANS (k-ε, k-ω SST, S-A, v2f)、LES (Smagorinsky, WALE)、DES
- **网格工具** — blockMesh、snappyHexMesh、gmsh/fluent/VTK 转换器
- **MPI 并行** — 域分解、Halo 交换、并行 I/O
- **拉格朗日粒子追踪** — 注入、碰撞、破碎、蒸发模型
- **多相 VOF/MULES** — 界面压缩、空化模型、相间力
- **结构力学** — 位移求解器、弹性模型
- **刚体动力学** — 关节、约束、运动求解器
- **波浪模型** — Airy、Stokes、Cnoidal 波浪理论
- **综合工具集** — checkMesh、setFields、renumberMesh、foamToVTK 等

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
├── io/                 # OpenFOAM 文件格式 I/O（ASCII + 二进制）、VTK/Gmsh/Fluent
├── mesh/               # PolyMesh、FvMesh、网格生成 (blockMesh, snappyHexMesh)
├── fields/             # volScalarField、volVectorField、surfaceScalarField
├── boundary/           # 408+ 种边界条件（速度/压力/湍流/VOF/热传导）
├── discretisation/     # fvm/fvc 算子、插值格式
├── solvers/            # PCG、PBiCGSTAB、GAMG、SIMPLE/SIMPLEC/PISO/PIMPLE
├── turbulence/         # RANS、LES、DES 模型 + 壁面函数（100+ 变体）
├── thermophysical/     # 完美气体、Sutherland、JANAF、ψ/ρ-based 热力学
├── multiphase/         # VOF + MULES、interFoam、Euler-Euler、空化
├── parallel/           # MPI 域分解、Halo 交换、并行 I/O
├── applications/       # 130+ 求解器（不可压缩、可压缩、多相、热传导）
├── tools/              # checkMesh、setFields、renumberMesh、foamToVTK 等
├── postprocessing/     # FunctionObject 框架、力计算、y+、VTK 输出
├── differentiable/     # 可微分算子、线性求解器、SIMPLE
├── lagrangian/         # 粒子追踪、注入、碰撞、破碎、蒸发
├── waves/              # Airy、Stokes、Cnoidal 波浪模型
├── fv/                 # fvModels（源项）+ fvConstraints
├── ode/                # ODE 求解器（Euler、RK4、RKF45、Rosenbrock）
├── rigid_body/         # 刚体动力学、关节、约束
├── structural/         # 结构力学（位移求解器、弹性模型）
├── models/             # 物理模型（辐射）
└── utils/              # 共享工具
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
| **优化** | adjointFoam、adjointShapeOptimizationFoam、adjointTurbulenceFoam |
| **声学** | acousticFoam |

---

## 验证

13 个基准算例，对照解析解和已发表的实验/数值数据：

| 案例 | 求解器 | 精度 | 参考 |
|------|--------|------|------|
| Couette 流 | icoFoam | 0.001%（内部） | Couette 解析解 |
| Poiseuille 流 | icoFoam | 0.02%（内部） | Hagen-Poiseuille 解析解 |
| 盖驱动方腔 (Re=100) | icoFoam | 0.9% (20x20) / 1.0% (32x32) | Ghia et al. 1982 |
| Taylor-Green 涡旋 | icoFoam | — | Taylor & Green 1937 |
| 后向台阶 | simpleFoam | — | Driver & Seegmiller 1985 |
| Sod 激波管 | rhoCentralFoam | — | Sod 1978 |
| 自然对流 (Ra=10^5) | buoyantBoussinesqSimpleFoam | — | de Vahl Davis 1983 |
| 溃坝 | interFoam | — | Martin & Moyce 1952 |
| 湍流通道 (Re_tau=180) | simpleFoam + kOmegaSST | — | Moser, Kim & Mansour 1999 |
| 可压缩喷管 | rhoCentralFoam | — | 等熵喷管理论 |
| 层流圆柱 (Re=20) | icoFoam | — | Dennis & Chang 1970 |
| 圆柱绕流 (Re=100) | pisoFoam | — | Williamson 1996 |
| 湍流管道 (Re=10000) | simpleFoam + kOmegaSST | — | Petukhov 1970 |

```bash
python validation/run_all.py
```

---

## 参考数据

OpenFOAM 参考模拟数据和 Docker 镜像均在 HuggingFace 上提供：

**[AlanZee/pyOpenFOAM-reference-data](https://huggingface.co/datasets/AlanZee/pyOpenFOAM-reference-data)**

| 文件 | 大小 | 说明 |
|------|------|------|
| `openfoam-reference-data.tar.gz` | 2.42 GB | 257 个 OpenFOAM 参考算例（v13 教程的 96%） |
| `pyopenfoam-simulation-results.tar.gz` | 47 KB | pyOpenFOAM 验证结果（34 个 JSON 文件） |
| Docker 镜像 | 622 MB | OpenFOAM-13 编译环境（122 个库，9 个二进制文件） |

```python
from huggingface_hub import hf_hub_download
import tarfile

path = hf_hub_download(
    repo_id="AlanZee/pyOpenFOAM-reference-data",
    filename="openfoam-reference-data.tar.gz",
    repo_type="dataset"
)
with tarfile.open(path, "r:gz") as tar:
    tar.extractall("validation/reference/openfoam/")
```

---

## 测试

```bash
# 运行所有测试
pytest tests/unit/ -q --tb=no

# 特定模块
pytest tests/unit/solvers/ -q
```

**结果**: 17,130 passed, 2 xfailed（约 410 秒）

---

## 文档

| 文档 | 说明 |
|------|------|
| [API 索引](docs/api/README.md) | 24 个模块概览、类计数、用法示例、RTS 模式 |
| [模块 API 参考](docs/api/modules.md) | 所有公共类和函数的详细 API |
| [入门指南 (en)](docs/en/getting_started.md) | 安装、快速开始、GPU 指南 |
| [入门指南 (zh)](docs/user_guide/getting_started.md) | 安装、快速开始、GPU 指南（中文） |
| [迁移指南](docs/migration_guide.md) | OpenFOAM 到 pyOpenFOAM 映射（中文） |
| [迁移指南 (en)](docs/en/migration_guide.md) | OpenFOAM 到 pyOpenFOAM 映射（英文） |
| [架构设计](docs/en/architecture.md) | 顶层架构与设计决策 |
| [GPU 指南](docs/en/gpu_guide.md) | GPU 加速与多 GPU 使用 |
| [PROPOSAL.md](docs/PROPOSAL.md) | 需求文档：目标、架构、验证基准、求解器列表 |
| [DESIGN.md](docs/DESIGN.md) | 顶层设计文档：模块架构、技术栈、关键决策 |
| [ROADMAP.md](docs/ROADMAP.md) | 后续计划：已完成工作概要、待完成工作 |

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

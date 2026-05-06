<div align="center">

<img src="logo.svg" alt="pyOpenFOAM Logo" width="200"/>

# pyOpenFOAM

**Pure Python CFD Solver with PyTorch GPU Acceleration**

*An open-source Python reimplementation of OpenFOAM*

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/tests-853+-brightgreen.svg)](#testing)

[English](#overview) | [中文](#概述)

</div>

---

## Overview

**pyOpenFOAM** is an open-source Python reimplementation of [OpenFOAM](https://www.openfoam.com/) (Foundation v13), the widely-used C++ computational fluid dynamics (CFD) toolbox. Our goal is to bring OpenFOAM's capabilities to the Python ecosystem while leveraging PyTorch for GPU acceleration.

### Motivation

OpenFOAM is a powerful and mature CFD solver with a large user community. However, its C++ codebase presents challenges for:
- Integration with modern machine learning workflows
- Rapid prototyping and experimentation
- GPU acceleration

pyOpenFOAM aims to address these gaps by providing a Python-native implementation that maintains compatibility with OpenFOAM's file formats and case structure.

### Comparison with Existing Tools

We acknowledge that there are excellent CFD tools available. Here is an honest comparison:

| Feature | OpenFOAM | pyOpenFOAM | JAX-Fluids |
|---------|----------|------------|------------|
| **Language** | C++ | Python | Python |
| **Maturity** | 30+ years, industry standard | Early development (v0.1) | Research-grade (2023) |
| **Mesh Types** | Unstructured polyhedral | Unstructured polyhedral | Cartesian only [1] |
| **Flow Types** | Compressible + Incompressible | Compressible + Incompressible | Compressible only [1] |
| **Turbulence** | Full RANS/LES library | k-ε, k-ω SST, S-A, LES | ALDM (implicit LES) [1] |
| **Multiphase** | VOF, Euler, etc. | VOF | Level-set + Diffuse interface [1] |
| **GPU Support** | No (CPU only) | Yes (PyTorch CUDA/MPS) | Yes (JAX/TPU/GPU) [1] |
| **Differentiable** | No | No (planned, see [ROADMAP](ROADMAP.md)) | Yes (end-to-end) [1] |
| **OpenFOAM Compatible** | Native | Yes (file format compatible) | No |
| **Scalability** | MPI, tested on 1000+ cores | MPI (basic) | Tested on 512 A100 GPUs [1] |

**[1]** Source: [JAX-Fluids GitHub repository](https://github.com/tumaer/JAXFLUIDS) (accessed 2025)

**Notes on JAX-Fluids**:
- JAX-Fluids is a **fully differentiable** CFD solver built on JAX, enabling end-to-end gradient-based optimization
- It uses **high-order WENO schemes** (up to 7th order) and multiple Riemann solvers
- It has been tested on up to **512 NVIDIA A100 GPUs** and **2048 TPU-v3 cores**
- Its main limitation is **Cartesian-only grids**, which restricts complex geometry handling

**Notes on OpenFOAM**:
- OpenFOAM is the **industry standard** for open-source CFD with extensive validation
- It has a **large user community** and extensive documentation
- It supports **unstructured polyhedral meshes** for complex geometries
- Its C++ implementation provides **high performance** on CPU clusters

pyOpenFOAM aims to complement these tools by providing a **Python-native OpenFOAM experience** with GPU acceleration, not to replace them.

---

## Installation

### Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.0
- NumPy ≥ 1.24
- SciPy ≥ 1.10

### Quick Install

```bash
# Clone the repository
git clone https://github.com/alanZee/pyOpenFOAM.git
cd pyOpenFOAM

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### GPU Support

For GPU acceleration, install PyTorch with CUDA support:

```bash
# For CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121

# For CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118

# For Apple Silicon (MPS)
pip install torch  # MPS support is built-in
```

### Optional Dependencies

```bash
# MPI parallel support
pip install mpi4py>=3.1.0

# Visualization
pip install matplotlib>=3.7.0 pyvista>=0.40.0

# Development tools
pip install pytest pytest-cov black ruff mypy
```

### Verify Installation

```bash
python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
python -c "import pyfoam; print(f'pyOpenFOAM {pyfoam.__version__}')"
```

---

## Quick Start

### Load and Run an OpenFOAM Case

```python
from pyfoam.applications import SimpleFoam

# Load existing OpenFOAM case
solver = SimpleFoam("tutorials/incompressible/simpleFoam/pitzDaily")
solver.run()
```

### Create Fields and Solve

```python
import torch
from pyfoam.mesh import FvMesh, PolyMesh
from pyfoam.fields import volScalarField, volVectorField
from pyfoam.solvers import PCG, SIMPLESolver

# Create mesh (simplified example)
mesh = FvMesh.from_poly_mesh(poly_mesh)

# Create fields
p = volScalarField(mesh, "p", dimensions=[0, 2, -2, 0, 0, 0, 0])
U = volVectorField(mesh, "U", dimensions=[0, 1, -1, 0, 0, 0, 0])

# Solve
solver = PCG(tolerance=1e-6)
x, iters, residual = solver(matrix, source)
```

### GPU Acceleration

```python
from pyfoam.core import device_context

# Run on GPU
with device_context("cuda"):
    mesh = FvMesh.from_poly_mesh(poly_mesh)  # All tensors on GPU
    solver = SIMPLESolver(mesh, config)
    U, p, phi, info = solver.solve(U, p, phi)
```

---

## Architecture

```
pyfoam/
├── core/           # Device management, tensor backend, LDU matrix
├── mesh/           # PolyMesh, FvMesh, geometry computation
├── fields/         # volScalarField, volVectorField, surfaceScalarField
├── boundary/       # 9 BC types: fixedValue, zeroGradient, cyclic, wall functions
├── io/             # OpenFOAM file format I/O (ASCII + binary)
├── discretisation/ # fvm/fvc operators, 5 interpolation schemes
├── solvers/        # PCG, PBiCGSTAB, GAMG, SIMPLE/PISO/PIMPLE
├── turbulence/     # k-ε, k-ω SST, Spalart-Allmaras, Smagorinsky, WALE
├── thermophysical/ # Perfect gas, Sutherland viscosity
├── multiphase/     # VOF two-phase solver
├── parallel/       # MPI domain decomposition
└── applications/   # simpleFoam, rhoSimpleFoam, interFoam
```

### Data Flow

```
OpenFOAM Case → io.Case → PolyMesh → FvMesh
                                      ↓
                              volScalarField / volVectorField
                                      ↓
                            fvm.laplacian / fvm.div / fvm.grad
                                      ↓
                                FvMatrix (LDU format)
                                      ↓
                            PCG / PBiCGSTAB / GAMG
                                      ↓
                              Solution Fields → io.write
```

---

## Validation

### Test Cases

We validate pyOpenFOAM against analytical solutions and published benchmark data:

| Case | Description | L2 Error | Reference |
|------|-------------|----------|-----------|
| Couette Flow | Linear velocity profile | 0.013% | Analytical: u(y) = U·y/H |
| Poiseuille Flow | Parabolic velocity profile | 0.13% | Analytical: u(y) = (1/2ν)·(-dp/dx)·y·(H-y) |
| Lid-Driven Cavity | Stokes flow (Re→0) | 0.34% | Grid convergence study |

### Benchmark Data Sources

**Couette & Poiseuille Flows**:
- Compared against **exact analytical solutions** from fluid mechanics textbooks
- No external data needed - solutions are mathematically exact

**Lid-Driven Cavity**:
- Current validation uses **grid convergence study** (32×32 vs 64×64 reference)
- The solver currently solves **Stokes equations** (no convection term)
- For validation against **Ghia et al. (1982)** benchmark, we need to implement the full Navier-Stokes solver with convection

**Ghia et al. (1982) Benchmark Data** (for future validation):
- **Source**: Ghia, U., Ghia, K.N. and Shin, C.T. (1982). "High-Re solutions for incompressible flow using the Navier-Stokes equations and a multigrid method." *Journal of Computational Physics*, 48(3), 387-411. DOI: [10.1016/0021-9991(82)90058-4](https://doi.org/10.1016/0021-9991(82)90058-4)
- **Data**: Tables I and II contain u-velocity along vertical centerline and v-velocity along horizontal centerline for Re=100, 400, 1000, 3200, 5000, 7500, 10000
- **Status**: We have not yet validated against this data (requires full Navier-Stokes solver)

### Run Validation

```bash
python validation/run_all.py

# Specific cases
python validation/run_all.py --only couette poiseuille

# Custom mesh size
python validation/run_all.py --mesh-size 64
```

---

## Benchmarks

### Linear Solver Scaling

| Solver | 1K cells | 8K cells | 64K cells | Scaling |
|--------|----------|----------|-----------|---------|
| PCG | 0.043s | 0.354s | 3.312s | ~O(n) |
| PBiCGSTAB | 0.161s | 0.590s | 6.937s | ~O(n) |
| GAMG | 0.099s | 0.905s | 9.177s | ~O(n) |

**Data Source**: Generated by `benchmarks/linear_solve_benchmark.py` on local machine (Intel CPU, PyTorch 2.11.0). These are **internal benchmarks** for measuring our solver performance, not comparisons with other CFD codes.

### Run Benchmarks

```bash
python benchmarks/run_all.py

# CPU-only
python benchmarks/run_all.py --device cpu

# Custom sizes
python benchmarks/run_all.py --mesh-sizes 10 20 40 60
```

---

## Current Limitations

We want to be transparent about what pyOpenFOAM **cannot** do yet:

1. **Not End-to-End Differentiable**: Our solvers use traditional numerical methods without `torch.autograd` support. For differentiable CFD, see [JAX-Fluids](https://github.com/tumaer/JAXFLUIDS) or our [ROADMAP](ROADMAP.md) for future plans.

2. **Simplified Validation**: Current validation cases use Jacobi iteration for Stokes equations, not the full SIMPLE solver. The SIMPLE solver exists but requires further debugging for complex cases.

3. **Early Development**: This is v0.1.0 - expect bugs and incomplete features. Contributions welcome!

4. **No OpenFOAM Comparison**: We haven't run identical test cases in OpenFOAM and compared results. This is planned for future releases.

5. **SIMPLE Solver Issues**: The SIMPLE solver in `pyfoam.solvers.simple` has numerical stability issues for the lid-driven cavity case. We are actively debugging this.

---

## Differentiable Simulation

**Current Status**: pyOpenFOAM does **not** support end-to-end differentiable simulation.

**What's Missing**:
- Discretization operators (fvm/fvc) don't support `torch.autograd`
- Linear solvers are not differentiable
- Pressure-velocity coupling (SIMPLE/PISO) is not differentiable

**Why This Matters**:
For physics-informed neural networks (PINNs) and gradient-based optimization, you need the solver to be a differentiable function.

**Future Plans**: See [ROADMAP.md](ROADMAP.md) for our plan to add differentiable solver support.

**Alternatives**: If you need differentiable CFD now, consider:
- [JAX-Fluids](https://github.com/tumaer/JAXFLUIDS) - Fully differentiable, JAX-based
- [Modulus](https://github.com/NVIDIA/modulus) - NVIDIA's physics-ML framework
- [PhiFlow](https://github.com/tum-pbs/PhiFlow) - Differentiable PDE solving

---

## Documentation

- **[Getting Started](docs/en/getting_started.md)** - Installation and first simulation
- **[API Reference](docs/en/api_reference.md)** - Complete API documentation
- **[Architecture](docs/en/architecture.md)** - System design and data flow
- **[GPU Guide](docs/en/gpu_guide.md)** - Device management and optimization
- **[Migration Guide](docs/en/migration_guide.md)** - Moving from OpenFOAM
- **[Roadmap](ROADMAP.md)** - Future plans and differentiable simulation

### 中文文档

- **[入门指南](docs/zh/getting_started.md)** - 安装和第一个仿真
- **[API 参考](docs/zh/api_reference.md)** - 完整 API 文档
- **[架构设计](docs/zh/architecture.md)** - 系统设计和数据流
- **[GPU 指南](docs/zh/gpu_guide.md)** - 设备管理和优化
- **[迁移指南](docs/zh/migration_guide.md)** - 从 OpenFOAM 迁移

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# Unit tests only
pytest tests/unit/ -v

# Integration tests
pytest tests/integration/ -v

# With coverage
pytest tests/ --cov=pyfoam --cov-report=html
```

### Test Environment

- **OS**: Windows 11
- **Python**: 3.11.9 (system installation)
- **PyTorch**: 2.11.0 + CUDA
- **NumPy**: 2.4.4
- **SciPy**: 1.17.1

### Test Coverage

- **853+ unit tests** across all modules
- **Integration tests** for solvers and physics
- **Validation cases** against analytical solutions

---

## Contributing

We welcome contributions! See our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
git clone https://github.com/alanZee/pyOpenFOAM.git
cd pyOpenFOAM
pip install -e ".[dev]"
pytest tests/
```

### Priority Areas

1. **Validation**: Help us validate against OpenFOAM and published benchmarks
2. **Differentiability**: Implement custom autograd functions (see [ROADMAP](ROADMAP.md))
3. **Performance**: Optimize GPU memory and computation
4. **Documentation**: Improve tutorials and examples

---

## Citation

If you use pyOpenFOAM in your research, please cite:

```bibtex
@software{pyopenfoam2025,
  author       = {pyOpenFOAM Contributors},
  title        = {pyOpenFOAM: Pure Python CFD Solver with PyTorch GPU Acceleration},
  year         = {2025},
  publisher    = {GitHub},
  url          = {https://github.com/alanZee/pyOpenFOAM},
  license      = {GPL-3.0}
}
```

---

## Acknowledgments

- **OpenFOAM Foundation** - The original C++ implementation we reimplemented
- **PyTorch** - GPU-accelerated tensor operations

---

## License

pyOpenFOAM is licensed under the [GNU General Public License v3.0](LICENSE), same as OpenFOAM.

---

<div align="center">

**Built for the CFD and Python communities**

[Report Bug](https://github.com/alanZee/pyOpenFOAM/issues) · [Request Feature](https://github.com/alanZee/pyOpenFOAM/issues) · [Discussions](https://github.com/alanZee/pyOpenFOAM/discussions)

</div>

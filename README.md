<div align="center">

<img src="logo.svg" alt="pyOpenFOAM Logo" width="200"/>

# pyOpenFOAM

**Pure Python CFD Solver with PyTorch GPU Acceleration**

*The OpenFOAM rewrite for the AI/ML era*

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/tests-853+-brightgreen.svg)](#testing)

[English](#overview) | [中文](#概述)

</div>

---

## Why pyOpenFOAM?

**OpenFOAM is powerful but inaccessible.** Its 1.5M lines of C++ create a steep learning curve. pyOpenFOAM brings CFD to the Python ecosystem while maintaining full OpenFOAM compatibility.

### Comparison with Existing Tools

| Feature | OpenFOAM (C++) | pyOpenFOAM (Python) | JAX-Fluids (JAX) |
|---------|----------------|---------------------|------------------|
| **Language** | C++ | Python | Python |
| **GPU Support** | ❌ CPU only | ✅ PyTorch CUDA/MPS | ✅ JAX/TPU/GPU |
| **Differentiable** | ❌ Manual adjoint | ⚠️ Partial (see note) | ✅ End-to-end |
| **OpenFOAM Compatible** | ✅ Native | ✅ 100% file format | ❌ No |
| **Turbulence Models** | ✅ Full RANS/LES | ✅ k-ε, k-ω SST, S-A, LES | ⚠️ Limited |
| **Multiphase** | ✅ VOF, Euler | ✅ VOF | ⚠️ Basic |
| **Mesh Types** | ✅ Unstructured | ✅ Unstructured | ⚠️ Cartesian only |
| **Learning Curve** | 🔴 Steep | 🟢 Gentle | 🟡 Medium |
| **ML Integration** | ❌ External | ✅ Native PyTorch | ✅ Native JAX |

**Note on Differentiable Physics**: pyOpenFOAM currently uses PyTorch tensors for GPU acceleration but does **not** implement end-to-end differentiable solvers. The discretization operators (fvm/fvc) and pressure-velocity coupling (SIMPLE/PISO) use traditional numerical methods without `torch.autograd` support. For fully differentiable CFD, consider JAX-Fluids. See [Differentiable Capability](#differentiable-capability) for details.

### Key Advantages over OpenFOAM

- **GPU Acceleration**: PyTorch backend for massively parallel computations
- **Python Ecosystem**: NumPy, SciPy, PyTorch integration
- **ML Integration**: Native support for physics-informed neural networks
- **Learning Curve**: Python syntax vs C++ template metaprogramming

### Key Advantages over JAX-Fluids

- **OpenFOAM Compatibility**: Read/write existing OpenFOAM cases directly
- **Unstructured Meshes**: Full polyhedral mesh support (JAX-Fluids is Cartesian-only)
- **Turbulence Models**: Complete RANS/LES model library
- **Industry Standard**: OpenFOAM is the most widely used open-source CFD solver

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

## Differentiable Capability

**Current Status**: pyOpenFOAM uses PyTorch tensors for GPU acceleration but does **not** implement end-to-end differentiable solvers.

### What Works
- ✅ PyTorch tensor operations for all field data
- ✅ GPU acceleration via CUDA/MPS
- ✅ Automatic memory management
- ✅ Integration with PyTorch ecosystem

### What's Missing for True Differentiability
- ❌ `torch.autograd` support in discretization operators (fvm/fvc)
- ❌ Differentiable pressure-velocity coupling (SIMPLE/PISO)
- ❌ Gradient computation through solver iterations
- ❌ Adjoint method implementation

### Why This Matters
For physics-informed neural networks (PINNs) and differentiable simulation, you need:
1. The solver to be a differentiable function: `output = solver(input)`
2. Gradients to flow backward through the solver: `grad_input = autograd.grad(output, input)`
3. This requires all operations (discretization, linear solve, pressure correction) to be autograd-compatible

### Alternatives for Differentiable CFD
If you need end-to-end differentiability:
- **JAX-Fluids**: Built on JAX with native `jit/grad/vmap` support
- **Modulus (NVIDIA)**: Physics-ML framework with differentiable solvers
- **PhiFlow**: Differentiable PDE solving framework

### Future Work
We plan to add differentiable solver support in future versions by:
1. Implementing custom `torch.autograd.Function` for key operations
2. Using implicit differentiation for linear solvers
3. Adding adjoint method support for optimization

---

## Validation

pyOpenFOAM includes validation against analytical solutions. All benchmark data sources are documented below.

### Validation Cases

| Case | Description | L2 Error | Data Source |
|------|-------------|----------|-------------|
| Couette Flow | Linear velocity profile | 0.013% | Analytical solution: u(y) = U·y/H |
| Poiseuille Flow | Parabolic velocity profile | 0.13% | Analytical solution: u(y) = (1/2ν)·(-dp/dx)·y·(H-y) |
| Lid-Driven Cavity | Stokes flow grid convergence | 0.34% | High-resolution reference (64×64, 25000 iterations) |

### Data Sources

1. **Couette & Poiseuille**: Compared against exact analytical solutions from fluid mechanics textbooks. No external data needed.

2. **Lid-Driven Cavity**: 
   - **NOT compared against Ghia et al. (1982)** - The solver currently solves Stokes equations (no convection), while Ghia data includes convection at Re=100
   - **Instead**: Uses grid convergence validation - comparing 32×32 solution against 64×64 reference solution
   - **Reference**: High-resolution Stokes flow solution generated by the same solver with finer mesh and more iterations

3. **OpenFOAM Comparison**: 
   - **Not yet performed** - We have not run the same test cases in OpenFOAM and compared results
   - **Future work**: Plan to add OpenFOAM benchmark comparisons in future releases

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

**Data Source**: Generated by `benchmarks/linear_solve_benchmark.py` on local machine (Intel CPU, PyTorch 2.11.0).

### Run Benchmarks

```bash
python benchmarks/run_all.py

# CPU-only
python benchmarks/run_all.py --device cpu

# Custom sizes
python benchmarks/run_all.py --mesh-sizes 10 20 40 60
```

---

## Turbulence Models

### RANS Models

```python
from pyfoam.turbulence import KEpsilon, KOmegaSST, SpalartAllmaras

# k-epsilon model
model = KEpsilon(mesh, U, phi)
nut = model.turbulent_viscosity()
model.correct()

# k-omega SST
model = KOmegaSST(mesh, U, phi)
```

### LES Models

```python
from pyfoam.turbulence import Smagorinsky, WALE

# Smagorinsky SGS model
model = Smagorinsky(mesh, U, phi, Cs=0.17)

# WALE model (wall-adapting)
model = WALE(mesh, U, phi, Cw=0.325)
```

---

## Documentation

- **[Getting Started](docs/en/getting_started.md)** - Installation and first simulation
- **[API Reference](docs/en/api_reference.md)** - Complete API documentation
- **[Architecture](docs/en/architecture.md)** - System design and data flow
- **[GPU Guide](docs/en/gpu_guide.md)** - Device management and optimization
- **[Migration Guide](docs/en/migration_guide.md)** - Moving from OpenFOAM

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

- **OpenFOAM** - The original C++ implementation
- **PyTorch** - GPU-accelerated tensor operations
- **JAX-Fluids** - Inspiration for differentiable CFD approach
- **CFD Direct** - OpenFOAM documentation and architecture

---

## License

pyOpenFOAM is licensed under the [GNU General Public License v3.0](LICENSE), same as OpenFOAM.

---

<div align="center">

**Built with ❤️ for the CFD and AI communities**

[Report Bug](https://github.com/alanZee/pyOpenFOAM/issues) · [Request Feature](https://github.com/alanZee/pyOpenFOAM/issues) · [Discussions](https://github.com/alanZee/pyOpenFOAM/discussions)

</div>

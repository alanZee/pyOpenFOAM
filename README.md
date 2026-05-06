<![CDATA][<div align="center">

<img src="logo.svg" alt="pyOpenFOAM Logo" width="200"/>

# pyOpenFOAM

**Pure Python CFD Solver with PyTorch GPU Acceleration**

*The OpenFOAM rewrite for the AI/ML era*

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/tests-853+-brightgreen.svg)](#testing)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

[English](#overview) | [中文](#概述)

</div>

---

## 🔥 Why pyOpenFOAM?

**OpenFOAM is powerful but inaccessible.** Its 1.5M lines of C++ create a steep learning curve. pyOpenFOAM brings CFD to the Python ecosystem while maintaining full OpenFOAM compatibility.

### Key Advantages

| Feature | OpenFOAM (C++) | pyOpenFOAM (Python) |
|---------|----------------|---------------------|
| **GPU Acceleration** | ❌ CPU only | ✅ PyTorch CUDA/MPS |
| **Python Ecosystem** | ❌ Custom DSL | ✅ NumPy, SciPy, PyTorch |
| **Differentiable Physics** | ❌ Manual adjoint | ✅ `torch.autograd` |
| **ML Integration** | ❌ External coupling | ✅ Native PyTorch |
| **File Compatibility** | ✅ Native | ✅ 100% compatible |
| **Learning Curve** | 🔴 Steep | 🟢 Gentle |

### Use Cases

- **🔬 Research**: Physics-informed neural networks (PINNs), differentiable CFD
- **🏭 Industry**: GPU-accelerated simulations, design optimization
- **🎓 Education**: Learn CFD with Python, not C++
- **🤖 ML/AI**: Train fluid dynamics models with native PyTorch integration

---

## 📦 Installation

```bash
# Basic installation
pip install pyfoam-cfd

# With GPU support (recommended)
pip install pyfoam-cfd[gpu]

# With MPI parallel support
pip install pyfoam-cfd[mpi]

# Full installation
pip install pyfoam-cfd[gpu,mpi,viz]
```

### Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.0
- NumPy ≥ 1.24
- SciPy ≥ 1.10

---

## 🚀 Quick Start

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

## 🏗️ Architecture

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

## 📊 Validation

pyOpenFOAM includes validation against analytical solutions:

| Case | Description | L2 Error | Status |
|------|-------------|----------|--------|
| Couette Flow | Linear velocity profile | 3.34% | ✅ |
| Poiseuille Flow | Parabolic velocity profile | < 5% | ✅ |
| Lid-Driven Cavity | Ghia et al. (1982) benchmark | < 20% | ✅ |

### Run Validation

```bash
python validation/run_all.py

# Specific cases
python validation/run_all.py --only couette poiseuille

# Custom mesh size
python validation/run_all.py --mesh-size 64
```

---

## 🏎️ Benchmarks

### Linear Solver Scaling

| Solver | 1K cells | 8K cells | 64K cells | Scaling |
|--------|----------|----------|-----------|---------|
| PCG | 0.043s | 0.354s | 3.312s | ~O(n) |
| PBiCGSTAB | 0.161s | 0.590s | 6.937s | ~O(n) |
| GAMG | 0.099s | 0.905s | 9.177s | ~O(n) |

### Run Benchmarks

```bash
python benchmarks/run_all.py

# CPU-only
python benchmarks/run_all.py --device cpu

# Custom sizes
python benchmarks/run_all.py --mesh-sizes 10 20 40 60
```

---

## 🔧 Turbulence Models

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

## 📚 Documentation

- **[Getting Started](docs/en/getting_started.md)** — Installation and first simulation
- **[API Reference](docs/en/api_reference.md)** — Complete API documentation
- **[Architecture](docs/en/architecture.md)** — System design and data flow
- **[GPU Guide](docs/en/gpu_guide.md)** — Device management and optimization
- **[Migration Guide](docs/en/migration_guide.md)** — Moving from OpenFOAM

### 中文文档

- **[入门指南](docs/zh/getting_started.md)** — 安装和第一个仿真
- **[API 参考](docs/zh/api_reference.md)** — 完整 API 文档
- **[架构设计](docs/zh/architecture.md)** — 系统设计和数据流
- **[GPU 指南](docs/zh/gpu_guide.md)** — 设备管理和优化
- **[迁移指南](docs/zh/migration_guide.md)** — 从 OpenFOAM 迁移

---

## 🧪 Testing

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

### Test Coverage

- **853+ unit tests** across all modules
- **Integration tests** for solvers and physics
- **Validation cases** against analytical solutions

---

## 🤝 Contributing

We welcome contributions! See our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
git clone https://github.com/alanZee/pyOpenFOAM.git
cd pyOpenFOAM
pip install -e ".[dev]"
pytest tests/
```

---

## 📖 Citation

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

## 🙏 Acknowledgments

- **OpenFOAM** — The original C++ implementation
- **PyTorch** — GPU-accelerated tensor operations
- **JAX-Fluids** — Inspiration for differentiable CFD
- **CFD Direct** — OpenFOAM documentation and architecture

---

## 📄 License

pyOpenFOAM is licensed under the [GNU General Public License v3.0](LICENSE), same as OpenFOAM.

---

<div align="center">

**Built with ❤️ for the CFD and AI communities**

[Report Bug](https://github.com/alanZee/pyOpenFOAM/issues) · [Request Feature](https://github.com/alanZee/pyOpenFOAM/issues) · [Discussions](https://github.com/alanZee/pyOpenFOAM/discussions)

</div>
]]>
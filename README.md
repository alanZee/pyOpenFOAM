<div align="center">

# pyOpenFOAM

**Pure Python CFD Solver with PyTorch GPU Acceleration**

*An open-source Python reimplementation of OpenFOAM v2512*

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/tests-2041+-brightgreen.svg)](#testing)

[English](#overview) | [中文](#概述)

</div>

---

## Overview

**pyOpenFOAM** is an open-source Python reimplementation of [OpenFOAM](https://www.openfoam.com/) v2512, the widely-used C++ computational fluid dynamics (CFD) toolbox. Our goal is to bring OpenFOAM's capabilities to the Python ecosystem while leveraging PyTorch for GPU acceleration and automatic differentiation.

### Key Features

- **30+ OpenFOAM Solvers** — Incompressible, compressible, multiphase, thermal, and more
- **GPU Acceleration** — All field operations use PyTorch tensors on CUDA/MPS
- **Differentiable CFD** — `torch.autograd` support through custom autograd functions
- **OpenFOAM Compatible** — Read/write existing OpenFOAM cases natively
- **20+ Boundary Conditions** — Velocity, pressure, turbulence, VOF, thermal
- **Full Turbulence Library** — RANS (k-ε, k-ω SST, S-A, v2f), LES (Smagorinsky, WALE), DES
- **Mesh Tools** — blockMesh, snappyHexMesh, gmsh/fluent/VTK converters
- **MPI Parallel** — Domain decomposition, halo exchange, parallel I/O

---

## Installation

### Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.0
- NumPy ≥ 1.24
- SciPy ≥ 1.10

### Quick Install

```bash
git clone https://github.com/alanZee/pyOpenFOAM.git
cd pyOpenFOAM
pip install -r requirements.txt
pip install -e .
```

### GPU Support

```bash
# CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Apple Silicon (MPS)
pip install torch  # MPS support built-in
```

---

## Quick Start

### Run an OpenFOAM Case

```python
from pyfoam.applications import SimpleFoam

solver = SimpleFoam("tutorials/incompressible/simpleFoam/pitzDaily")
solver.run()
```

### GPU Acceleration

```python
from pyfoam.core import device_context

with device_context("cuda"):
    mesh = FvMesh.from_poly_mesh(poly_mesh)
    solver = SIMPLESolver(mesh, config)
    U, p, phi, info = solver.solve(U, p, phi)
```

### Differentiable CFD

```python
from pyfoam.differentiable import DifferentiableLaplacian, DifferentiableLinearSolve

# Differentiable Laplacian (supports torch.autograd)
lap = DifferentiableLaplacian.apply(phi, mesh)

# Differentiable linear solve (implicit differentiation)
x = DifferentiableLinearSolve.apply(A, b, tol, max_iter)
```

---

## Architecture

```
pyfoam/
├── core/               # Device management, LDU/FvMatrix, sparse ops, multi-GPU
├── mesh/               # PolyMesh, FvMesh, mesh generation (blockMesh, snappyHexMesh)
├── fields/             # volScalarField, volVectorField, surfaceScalarField
├── boundary/           # 20+ BC types (velocity, pressure, turbulence, VOF, thermal)
├── io/                 # OpenFOAM file format I/O (ASCII + binary)
├── discretisation/     # fvm/fvc operators, interpolation schemes
├── solvers/            # PCG, PBiCGSTAB, GAMG, SIMPLE/SIMPLEC/PISO/PIMPLE
├── turbulence/         # RANS, LES, DES models + wall functions
├── thermophysical/     # Perfect gas, Sutherland, JANAF, ψ/ρ-based thermo
├── multiphase/         # VOF + MULES, interFoam, Euler-Euler, cavitation
├── parallel/           # MPI decomposition, halo exchange, parallel I/O
├── applications/       # 30+ solvers (incompressible, compressible, multiphase, thermal)
├── postprocessing/     # FunctionObject framework, forces, y+, VTK output
├── differentiable/     # Differentiable operators, linear solver, SIMPLE
├── mesh_generation/    # blockMesh, snappyHexMesh
└── mesh_conversion/    # gmshToFoam, fluentMeshToFoam, foamToVTK
```

---

## Implemented Solvers

| Category | Solvers |
|----------|---------|
| **Incompressible** | simpleFoam, icoFoam, pisoFoam, pimpleFoam, SRFSimpleFoam, porousSimpleFoam, boundaryFoam |
| **Compressible** | rhoSimpleFoam, rhoPimpleFoam, sonicFoam, rhoCentralFoam |
| **Buoyancy** | buoyantSimpleFoam, buoyantPimpleFoam, buoyantBoussinesqSimpleFoam |
| **Thermal** | laplacianFoam, chtMultiRegionFoam |
| **Multiphase** | interFoam, multiphaseInterFoam, compressibleInterFoam, twoPhaseEulerFoam, multiphaseEulerFoam, cavitatingFoam |
| **Other** | potentialFoam, scalarTransportFoam, reactingFoam, solidDisplacementFoam |

---

## Validation

| Case | Solver | L2 Error | Reference |
|------|--------|----------|-----------|
| Couette Flow | simpleFoam | 0.013% | Analytical |
| Poiseuille Flow | simpleFoam | 0.13% | Analytical |
| Lid-Driven Cavity (Re=100) | simpleFoam | ~15% | Ghia et al. 1982 |

```bash
python validation/run_all.py
```

---

## Testing

```bash
# Run all tests
pytest tests/unit/ -q --tb=no

# Specific module
pytest tests/unit/solvers/ -q
```

**Results**: 2041 passed, 17 xfailed (~130 seconds)

---

## Documentation

| Document | Description |
|----------|-------------|
| [PROPOSAL.md](docs/PROPOSAL.md) | Requirements, goals, benchmarks, solver list |
| [DESIGN.md](docs/DESIGN.md) | Top-level architecture and design decisions |
| [ROADMAP.md](docs/ROADMAP.md) | Future plans and remaining work |
| [English Docs](docs/en/) | Getting started, API reference, architecture, GPU guide, migration guide |
| [中文文档](docs/zh/) | 入门指南、API 参考、架构设计、GPU 指南、迁移指南 |

---

## Contributing

We welcome contributions! Priority areas:

1. **Validation** — Help us validate against OpenFOAM
2. **Differentiability** — Extend autograd support
3. **Performance** — Optimize GPU memory and computation
4. **Documentation** — Improve tutorials and examples

---

## License

pyOpenFOAM is licensed under the [GNU General Public License v3.0](LICENSE).

---

<div align="center">

**Built for the CFD and Python communities**

[Report Bug](https://github.com/alanZee/pyOpenFOAM/issues) · [Request Feature](https://github.com/alanZee/pyOpenFOAM/issues) · [Discussions](https://github.com/alanZee/pyOpenFOAM/discussions)

</div>

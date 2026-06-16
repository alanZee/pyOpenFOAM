<div align="center">

# pyOpenFOAM

**Pure Python CFD Solver with PyTorch GPU Acceleration**

*An open-source Python reimplementation of OpenFOAM 13 (Foundation)*

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/tests-2041+-brightgreen.svg)](#testing)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Reference_Data-ffd21e.svg)](https://huggingface.co/datasets/AlanZee/pyOpenFOAM-reference-data)

[English](#overview) | [中文](README_zh.md)

</div>

---

## Overview

**pyOpenFOAM** is an open-source Python reimplementation of [OpenFOAM 13](https://openfoam.org/) (Foundation), the widely-used C++ computational fluid dynamics (CFD) toolbox. Our goal is to bring OpenFOAM's capabilities to the Python ecosystem while leveraging PyTorch for GPU acceleration and automatic differentiation.

### Key Features

- **30+ OpenFOAM Solvers** — Incompressible, compressible, multiphase, thermal, and more
- **GPU Acceleration** — All field operations use PyTorch tensors on CUDA/MPS
- **Differentiable CFD** — `torch.autograd` support through custom autograd functions
- **OpenFOAM Compatible** — Read/write existing OpenFOAM cases natively
- **20+ Boundary Conditions** — Velocity, pressure, turbulence, VOF, thermal
- **Full Turbulence Library** — RANS (k-epsilon, k-omega SST, S-A, v2f), LES (Smagorinsky, WALE), DES
- **Mesh Tools** — blockMesh, snappyHexMesh, gmsh/fluent/VTK converters
- **MPI Parallel** — Domain decomposition, halo exchange, parallel I/O
- **Lagrangian Particle Tracking** — Injection, collision, breakup, evaporation models
- **Multiphase VOF/MULES** — Interface compression, cavitation models, interfacial forces
- **Structural Mechanics** — Displacement solver, elastic models
- **Rigid Body Dynamics** — Joints, restraints, motion solvers
- **Wave Models** — Airy, Stokes, Cnoidal wave theories
- **Comprehensive Tools** — checkMesh, setFields, renumberMesh, foamToVTK, and more

---

## Reference Data

OpenFOAM reference simulation data is available on HuggingFace:

**[AlanZee/pyOpenFOAM-reference-data](https://huggingface.co/datasets/AlanZee/pyOpenFOAM-reference-data)**

| File | Size | Description |
|------|------|-------------|
| `openfoam-reference-data.tar.gz` | 2.42 GB | 246 OpenFOAM v11 reference cases (92% of v13 tutorials) |
| `pyopenfoam-simulation-results.tar.gz` | 47 KB | pyOpenFOAM validation results (34 JSON files) |

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
├── io/                 # OpenFOAM file format I/O (ASCII + binary), VTK/Gmsh/Fluent
├── mesh/               # PolyMesh, FvMesh, mesh generation (blockMesh, snappyHexMesh)
├── fields/             # volScalarField, volVectorField, surfaceScalarField
├── boundary/           # 30+ BC types (velocity, pressure, turbulence, VOF, thermal)
├── discretisation/     # fvm/fvc operators, interpolation schemes
├── solvers/            # PCG, PBiCGSTAB, GAMG, SIMPLE/SIMPLEC/PISO/PIMPLE
├── turbulence/         # RANS, LES, DES models + wall functions (100+ variants)
├── thermophysical/     # Perfect gas, Sutherland, JANAF, psi/rho-based thermo
├── multiphase/         # VOF + MULES, interFoam, Euler-Euler, cavitation
├── parallel/           # MPI decomposition, halo exchange, parallel I/O
├── applications/       # 35+ solvers (incompressible, compressible, multiphase, thermal)
├── tools/              # checkMesh, setFields, renumberMesh, foamToVTK, etc.
├── postprocessing/     # FunctionObject framework, forces, y+, VTK output
├── differentiable/     # Differentiable operators, linear solver, SIMPLE
├── lagrangian/         # Particle tracking, injection, collision, breakup, evaporation
├── waves/              # Airy, Stokes, Cnoidal wave models
├── fv/                 # fvModels (sources) + fvConstraints
├── ode/                # ODE solvers (Euler, RK4, RKF45, Rosenbrock)
├── rigid_body/         # Rigid body dynamics, joints, restraints
├── structural/         # Structural mechanics (displacement solver, elastic models)
├── models/             # Physical models (radiation)
└── utils/              # Shared utilities
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
| **Optimization** | adjointFoam, adjointShapeOptimizationFoam, adjointTurbulenceFoam |
| **Acoustics** | acousticFoam |

---

## Validation

13 benchmark cases defined against analytical solutions and published experimental/numerical data:

| Case | Solver | Reference |
|------|--------|-----------|
| Couette Flow | icoFoam | Couette analytical solution |
| Poiseuille Flow | icoFoam | Hagen-Poiseuille analytical solution |
| Lid-Driven Cavity (Re=100) | icoFoam | Ghia et al. 1982 |
| Taylor-Green Vortex | icoFoam | Taylor & Green 1937 |
| Backward Facing Step | simpleFoam | Driver & Seegmiller 1985 |
| Sod Shock Tube | rhoCentralFoam | Sod 1978 |
| Natural Convection (Ra=10^5) | buoyantBoussinesqSimpleFoam | de Vahl Davis 1983 |
| Dam Break | interFoam | Martin & Moyce 1952 |
| Turbulent Channel (Re_tau=180) | simpleFoam + kOmegaSST | Moser, Kim & Mansour 1999 |
| Compressible Nozzle | rhoCentralFoam | Isentropic nozzle theory |
| Laminar Cylinder (Re=20) | icoFoam | Dennis & Chang 1970 |
| Cylinder Flow (Re=100) | pisoFoam | Williamson 1996 |
| Turbulent Duct (Re=10000) | simpleFoam + kOmegaSST | Petukhov 1970 |

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
| [API Index](docs/api/README.md) | 24 modules overview, class counts, usage examples, RTS pattern |
| [Module API Reference](docs/api/modules.md) | Detailed API for all public classes and functions |
| [Getting Started (en)](docs/en/getting_started.md) | Installation, quick start, GPU guide |
| [Getting Started (zh)](docs/user_guide/getting_started.md) | Installation, quick start, GPU guide (Chinese) |
| [Migration Guide](docs/migration_guide.md) | OpenFOAM to pyOpenFOAM mapping (Chinese) |
| [Migration Guide (en)](docs/en/migration_guide.md) | OpenFOAM to pyOpenFOAM mapping (English) |
| [Architecture](docs/en/architecture.md) | Top-level architecture and design decisions |
| [GPU Guide](docs/en/gpu_guide.md) | GPU acceleration and multi-GPU usage |
| [PROPOSAL.md](docs/PROPOSAL.md) | Requirements, goals, benchmarks, solver list |
| [DESIGN.md](docs/DESIGN.md) | Top-level architecture and design decisions |
| [ROADMAP.md](docs/ROADMAP.md) | Future plans and remaining work |

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

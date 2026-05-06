# pyOpenFOAM

Pure Python rewrite of OpenFOAM with PyTorch GPU acceleration.

## Overview

pyOpenFOAM is a complete Python implementation of OpenFOAM's CFD capabilities, designed for:

- **Full OpenFOAM compatibility**: Native support for all OpenFOAM file formats
- **GPU acceleration**: PyTorch backend for massively parallel computations
- **Pythonic API**: Clean, intuitive Python interface
- **Differentiable physics**: Built-in support for physics-informed ML

## Architecture

```
pyfoam/
├── core/           # Core data structures (Field, Mesh, Matrix)
├── mesh/           # Mesh handling (polyMesh, fvMesh)
├── fields/         # Field classes (volScalarField, volVectorField, etc.)
├── solvers/        # Linear solvers (PCG, PBiCG, GAMG)
├── turbulence/     # Turbulence models (k-ε, k-ω SST, LES)
├── boundary/       # Boundary conditions
├── io/             # OpenFOAM file format I/O
├── thermophysical/ # Thermodynamics and transport
├── models/         # Physical models and constraints
├── parallel/       # MPI parallelization
└── applications/   # Solver applications
```

## Installation

```bash
pip install pyfoam-cfd

# With GPU support
pip install pyfoam-cfd[gpu]

# With MPI support
pip install pyfoam-cfd[mpi]
```

## Quick Start

```python
from pyfoam import Case, SimpleSolver

# Load OpenFOAM case
case = Case("tutorial/incompressible/simpleFoam/pitzDaily")

# Create and run solver
solver = SimpleSolver(case)
solver.run()
```

## License

GPL-3.0 (same as OpenFOAM)

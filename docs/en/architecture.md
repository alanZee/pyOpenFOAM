# pyOpenFOAM Architecture

## Overview

pyOpenFOAM is a pure Python rewrite of OpenFOAM's computational fluid dynamics (CFD) capabilities, using PyTorch as the tensor backend for GPU-accelerated simulations. The architecture preserves OpenFOAM's finite volume method (FVM) design while leveraging Python's expressiveness and PyTorch's hardware acceleration.

## Design Principles

1. **OpenFOAM Compatibility** — Native support for all OpenFOAM file formats (mesh, fields, dictionaries, boundary conditions).
2. **GPU-First** — All tensor operations route through PyTorch, enabling transparent CPU/CUDA/MPS acceleration.
3. **float64 by Default** — CFD convergence requires double precision; float32 causes divergence in pressure-velocity coupling.
4. **Lazy Evaluation** — Geometric quantities (cell volumes, face areas, interpolation weights) are computed on first access and cached.
5. **Run-Time Selection (RTS)** — Boundary conditions use a class-level registry, mirroring OpenFOAM's RTS mechanism.

## Module Structure

```
pyfoam/
├── core/               # Foundation layer
│   ├── device.py       # DeviceManager, TensorConfig, device_context
│   ├── dtype.py        # CFD_DTYPE, INDEX_DTYPE, dtype utilities
│   ├── backend.py      # scatter_add, gather, sparse_coo_tensor, sparse_mm
│   ├── ldu_matrix.py   # LduMatrix — LDU sparse matrix format
│   ├── fv_matrix.py    # FvMatrix — FVM matrix with source, BC, relaxation
│   └── sparse_ops.py   # ldu_to_coo_indices, extract_diagonal, csr_matvec
│
├── mesh/               # Mesh representation
│   ├── poly_mesh.py    # PolyMesh — raw topology (points, faces, owner, neighbour)
│   ├── fv_mesh.py      # FvMesh — extends PolyMesh with geometric quantities
│   ├── mesh_geometry.py # Face/cell geometry computation functions
│   └── topology.py     # Face-cell connectivity utilities
│
├── fields/             # Field classes
│   ├── vol_fields.py   # volScalarField, volVectorField, volTensorField
│   ├── geometric_field.py  # GeometricField base class
│   ├── field_arithmetic.py # FieldArithmeticMixin (+, -, *, /)
│   └── dimensions.py   # DimensionSet for dimensional checking
│
├── boundary/           # Boundary conditions
│   ├── boundary_condition.py # BoundaryCondition ABC + RTS registry + Patch
│   ├── boundary_field.py     # BoundaryField container
│   ├── fixed_value.py        # fixedValue (penalty method)
│   ├── zero_gradient.py      # zeroGradient (Neumann zero-flux)
│   ├── cyclic.py             # cyclic (periodic coupling)
│   ├── symmetry.py           # symmetryPlane
│   ├── no_slip.py            # noSlip (fixedValue with zero)
│   ├── wall_function.py      # nutkWallFunction, kqRWallFunction
│   ├── inlet_outlet.py       # inletOutlet (flow-direction switching)
│   └── fixed_gradient.py     # fixedGradient (prescribed Neumann)
│
├── io/                 # OpenFOAM file format I/O
│   ├── case.py         # Case — complete case directory representation
│   ├── dictionary.py   # FoamDict, parse_dict, parse_dict_file
│   ├── foam_file.py    # FoamFile — generic OpenFOAM file reader
│   ├── field_io.py     # read_field, write_field
│   ├── mesh_io.py      # read_mesh, read_boundary
│   └── binary_io.py    # Binary format read/write
│
├── discretisation/     # FVM discretisation schemes
│   ├── weights.py      # compute_centre_weights, compute_upwind_weights
│   ├── interpolation.py # InterpolationScheme, LinearInterpolation
│   └── schemes/        # UpwindInterpolation, LinearUpwindInterpolation, QuickInterpolation
│
├── solvers/            # Linear and coupled solvers
│   ├── linear_solver.py    # LinearSolverBase, create_solver factory
│   ├── pcg.py              # PCGSolver — Preconditioned Conjugate Gradient
│   ├── pbicgstab.py        # PBiCGSTABSolver — Preconditioned BiCGStab
│   ├── gamg.py             # GAMGSolver — Algebraic Multigrid
│   ├── preconditioners.py  # DICPreconditioner, DILUPreconditioner
│   ├── residual.py         # ResidualMonitor, ConvergenceInfo
│   ├── coupled_solver.py   # CoupledSolverBase, CoupledSolverConfig, ConvergenceData
│   ├── simple.py           # SIMPLESolver — steady-state incompressible
│   ├── piso.py             # PISOSolver — transient incompressible
│   ├── pimple.py           # PIMPLESolver — transient with outer iterations
│   ├── pressure_equation.py # assemble, solve, correct_velocity, correct_face_flux
│   └── rhie_chow.py        # Rhie-Chow interpolation (velocity-pressure coupling)
│
├── turbulence/         # Turbulence models (planned)
├── thermophysical/     # Thermodynamics and transport (planned)
├── models/             # Physical models (planned)
├── parallel/           # MPI parallelization (planned)
└── utils/              # Utility functions (planned)
```

## Data Flow

### Mesh Loading

```
OpenFOAM case directory
    │
    ▼
Case("path/to/case")
    │  reads system/controlDict, fvSchemes, fvSolution
    │  reads constant/polyMesh/{points, faces, owner, neighbour, boundary}
    ▼
MeshData (raw numpy arrays)
    │
    ▼
PolyMesh (topology tensors on configured device)
    │
    ▼
FvMesh (lazy geometry: cell_centres, cell_volumes, face_areas, face_weights, delta_coefficients)
```

### Field Operations

```
volScalarField(mesh, "p", internal=initial_values)
    │
    ├── internal_field: (n_cells,) tensor on device
    ├── boundary_field: list of BoundaryCondition objects
    │
    ▼
Arithmetic: p1 + p2, p * scalar, etc.
    │  dimension checking via DimensionSet
    │  device/dtype consistency via TensorConfig
    ▼
Assignment: p.assign(new_values)
    │  applies boundary conditions
    ▼
I/O: write_field(p, path)
```

### FVM Assembly

```
Discretisation of ∇·(φ) + ∇²(φ) = S
    │
    ▼
InterpolationScheme (face values from cell values)
    │  LinearInterpolation: φ_f = w·φ_P + (1-w)·φ_N
    │  UpwindInterpolation: φ_f = φ_upstream
    ▼
LduMatrix assembly
    │  diag:  (n_cells,) — diagonal coefficients
    │  lower: (n_internal_faces,) — owner-side off-diagonal
    │  upper: (n_internal_faces,) — neighbour-side off-diagonal
    ▼
FvMatrix (extends LduMatrix)
    │  source:  (n_cells,) — right-hand side
    │  boundary contributions via BC.matrix_contributions()
    │  under-relaxation via FvMatrix.relax()
    ▼
Linear solve: FvMatrix.solve(solver, x0, tolerance, max_iter)
    │  PCG (symmetric), PBiCGStab (asymmetric), GAMG (multigrid)
    ▼
Solution tensor
```

### Coupled Solver Loop (SIMPLE)

```
for each outer iteration:
    │
    ├── 1. Momentum predictor: solve A_p·U* = H(U) - ∇p
    │       with under-relaxation (α_U)
    │
    ├── 2. Compute HbyA = H(U*) / A_p
    │
    ├── 3. Compute face flux φ_HbyA (Rhie-Chow interpolation)
    │
    ├── 4. Assemble pressure correction equation:
    │       ∇²(1/A_p, p') = ∇·(φ_HbyA)
    │
    ├── 5. Solve pressure correction p' (PCG)
    │       p = α_p·p' + (1-α_p)·p_old
    │
    ├── 6. Correct velocity: U = HbyA - (1/A_p)·∇p
    │
    ├── 7. Correct face flux: φ = φ_HbyA - (1/A_p)_f·∇p_f
    │
    └── 8. Check convergence: continuity_error < tolerance
```

## Device and Dtype Management

### DeviceManager (Singleton)

```python
from pyfoam.core import DeviceManager

dm = DeviceManager()
print(dm.capabilities)  # DeviceCapabilities(cpu=True, cuda=False, mps=False)
print(dm.device)        # device('cpu') — auto-selected best available
dm.device = 'cuda'      # manual override (raises ValueError if unavailable)
```

Priority: CUDA > MPS > CPU.

### TensorConfig (Global Defaults)

```python
from pyfoam.core import TensorConfig

config = TensorConfig()  # defaults: float64, best device
t = config.zeros(100)    # float64 tensor on default device

with config.override(dtype=torch.float32, device='cpu'):
    t32 = config.zeros(100)  # float32 on CPU, temporary
# Back to defaults after context exit
```

### Module-Level Convenience

```python
from pyfoam.core import get_device, get_default_dtype, device_context

device = get_device()       # current default device
dtype = get_default_dtype() # torch.float64

with device_context(device='cuda'):
    # All pyfoam operations use CUDA here
    pass
```

## Sparse Matrix Formats

### LDU Format (OpenFOAM Native)

The LDU (Lower-Diagonal-Upper) format stores FVM matrix coefficients as three flat arrays:

- **diag** `(n_cells,)` — one diagonal coefficient per cell
- **lower** `(n_internal_faces,)` — owner-side off-diagonal (row=owner, col=neighbour)
- **upper** `(n_internal_faces,)` — neighbour-side off-diagonal (row=neighbour, col=owner)

Face addressing (owner/neighbour arrays from the mesh) connects off-diagonal entries to matrix rows. This is more memory-efficient than CSR for FVM assembly because the mesh topology provides the addressing.

### COO/CSR Conversion

For linear solvers that require standard sparse formats:

```python
coo = ldu_matrix.to_sparse_coo()   # COO for assembly
csr = ldu_matrix.to_sparse_csr()   # CSR for solving
```

## Extending pyOpenFOAM

### Custom Boundary Conditions

```python
from pyfoam.boundary import BoundaryCondition, Patch

@BoundaryCondition.register("myCustomBC")
class MyCustomBC(BoundaryCondition):
    def apply(self, field, patch_idx=None):
        # Modify boundary-face values
        return field

    def matrix_contributions(self, field, n_cells, diag=None, source=None):
        # Return (diag, source) contributions
        if diag is None:
            diag = torch.zeros(n_cells)
        if source is None:
            source = torch.zeros(n_cells)
        return diag, source
```

### Custom Linear Solvers

Implement the `LinearSolver` protocol:

```python
from pyfoam.core.fv_matrix import LinearSolver

class MySolver:
    def __call__(self, matrix, source, x0, tolerance, max_iter):
        # Solve A x = b
        # Return (solution, iterations, final_residual)
        ...
```

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| PyTorch | ≥ 2.0 | Tensor backend, GPU acceleration |
| NumPy | ≥ 1.24 | Array conversion, file I/O |
| SciPy | ≥ 1.10 | Sparse matrix utilities (optional) |

### Optional Dependencies

| Package | Purpose |
|---------|---------|
| cupy-cuda12x | CUDA GPU support |
| mpi4py | MPI parallelization |
| pyvista, matplotlib | Visualization |
| pytest, black, ruff, mypy | Development tools |

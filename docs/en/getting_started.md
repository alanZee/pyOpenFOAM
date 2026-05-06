# Getting Started with pyOpenFOAM

## Installation

### Prerequisites

- Python 3.10 or later
- PyTorch 2.0 or later

### Install from PyPI

```bash
# Basic installation (CPU only)
pip install pyfoam-cfd

# With GPU support (CUDA 12.x)
pip install pyfoam-cfd[gpu]

# With MPI support
pip install pyfoam-cfd[mpi]

# With visualization tools
pip install pyfoam-cfd[viz]

# Development installation
pip install pyfoam-cfd[dev]
```

### Install from Source

```bash
git clone https://github.com/pyOpenFOAM/pyOpenFOAM.git
cd pyOpenFOAM
pip install -e ".[dev]"
```

### Verify Installation

```python
import pyfoam
print(pyfoam.__version__)  # 0.1.0

from pyfoam.core import DeviceManager
dm = DeviceManager()
print(dm.capabilities)  # DeviceCapabilities(cpu=True, cuda=False, ...)
print(dm.device)        # device('cpu')
```

## Quick Start: Loading an OpenFOAM Case

pyOpenFOAM can read any standard OpenFOAM case directory. Here's how to load and inspect a case:

```python
from pyfoam.io.case import Case

# Load an OpenFOAM case
case = Case("path/to/incompressible/simpleFoam/pitzDaily")

# Inspect configuration
print(case.controlDict["application"])  # "simpleFoam"
print(case.get_start_time())            # 0.0
print(case.get_end_time())              # 100.0
print(case.get_delta_t())               # 1.0

# List time directories
print(case.time_dirs)  # ['0', '1', '2', ..., '100']

# List fields at time 0
print(case.list_fields(time=0))  # ['U', 'p', 'nut']

# Read a field
field_data = case.read_field("U", time=0)
print(field_data.dimensions)  # [0, 1, -1, 0, 0, 0, 0]  (m/s)
```

## Working with Meshes

### Loading a Mesh

```python
from pyfoam.io.case import Case
from pyfoam.mesh import FvMesh

case = Case("path/to/case")
mesh_data = case.mesh  # Raw MeshData from polyMesh directory

# Create an FvMesh (computes geometry automatically)
fv_mesh = FvMesh(
    points=mesh_data.points,
    faces=mesh_data.faces,
    owner=mesh_data.owner,
    neighbour=mesh_data.neighbour,
    boundary=mesh_data.boundary,
)

# Or pre-compute all geometry at once
fv_mesh.compute_geometry()
```

### Mesh Properties

```python
print(fv_mesh.n_points)           # Number of vertices
print(fv_mesh.n_cells)            # Number of cells
print(fv_mesh.n_faces)            # Total faces (internal + boundary)
print(fv_mesh.n_internal_faces)   # Internal faces only

# Geometric quantities (lazy-computed on first access)
print(fv_mesh.cell_centres.shape)       # (n_cells, 3)
print(fv_mesh.cell_volumes.shape)       # (n_cells,)
print(fv_mesh.face_centres.shape)       # (n_faces, 3)
print(fv_mesh.face_areas.shape)         # (n_faces, 3) — normal × area
print(fv_mesh.face_weights.shape)       # (n_faces,) — interpolation weights
print(fv_mesh.delta_coefficients.shape) # (n_faces,) — 1/distance

# Derived quantities
print(fv_mesh.face_normals.shape)       # (n_faces, 3) — unit normals
print(fv_mesh.total_volume)             # scalar — sum of all cell volumes
```

### Creating a Mesh Programmatically

```python
from pyfoam.mesh import PolyMesh

# Define a simple 2D quad mesh (2 cells)
mesh = PolyMesh.from_raw(
    points=[
        [0, 0, 0], [1, 0, 0], [2, 0, 0],
        [0, 1, 0], [1, 1, 0], [2, 1, 0],
    ],
    faces=[
        [0, 1, 4, 3],  # internal face 0
        [1, 2, 5, 4],  # internal face 1
        [0, 3],         # boundary face 2 (left)
        [2, 5],         # boundary face 3 (right)
        [0, 1],         # boundary face 4 (bottom)
        [3, 4],         # boundary face 5 (top)
    ],
    owner=[0, 1, 0, 1, 0, 0],
    neighbour=[1],  # only internal faces have neighbours
    boundary=[
        {"name": "left",   "type": "patch", "startFace": 2, "nFaces": 1},
        {"name": "right",  "type": "patch", "startFace": 3, "nFaces": 1},
        {"name": "bottom", "type": "wall",  "startFace": 4, "nFaces": 1},
        {"name": "top",    "type": "wall",  "startFace": 5, "nFaces": 1},
    ],
)

print(mesh)  # PolyMesh(n_points=6, n_faces=6, n_cells=2, ...)
```

## Working with Fields

### Creating Volume Fields

```python
import torch
from pyfoam.fields import volScalarField, volVectorField
from pyfoam.mesh import FvMesh

# Assume mesh is an FvMesh instance
# Create a scalar pressure field
p = volScalarField(mesh, "p")
p.assign(torch.zeros(mesh.n_cells))  # Initialize to zero

# Create a velocity field
U = volVectorField(mesh, "U")
U.assign(torch.zeros(mesh.n_cells, 3))  # Initialize to zero

# Set uniform values
p.assign(torch.ones(mesh.n_cells) * 101325.0)  # 1 atm
```

### Field Arithmetic

```python
# Fields support standard arithmetic operations
p1 = volScalarField(mesh, "p1")
p2 = volScalarField(mesh, "p2")
p1.assign(torch.ones(mesh.n_cells))
p2.assign(torch.ones(mesh.n_cells) * 2.0)

# Addition (dimensions must match)
p_sum = p1 + p2  # New field with values 3.0

# Scalar multiplication (field must be dimensionless for * by scalar)
p_scaled = p1 * 2.0  # New field with values 2.0

# In-place operations
p1 += p2       # p1 now has values 3.0
p1 *= 0.5      # p1 now has values 1.5
```

### Using Boundary Conditions

```python
from pyfoam.boundary import BoundaryCondition, Patch

# Create a patch descriptor
inlet_patch = Patch(
    name="inlet",
    face_indices=torch.tensor([0, 1, 2]),
    face_normals=torch.tensor([[-1.0, 0.0, 0.0]] * 3),
    face_areas=torch.tensor([0.01, 0.01, 0.01]),
    delta_coeffs=torch.tensor([100.0, 100.0, 100.0]),
    owner_cells=torch.tensor([0, 1, 2]),
)

# Create boundary conditions using RTS (Run-Time Selection)
inlet_bc = BoundaryCondition.create(
    "fixedValue",
    inlet_patch,
    coeffs={"value": 1.0},  # Uniform velocity of 1 m/s
)

# Apply to a field
velocity = torch.zeros(10, 3)  # 10 cells, 3D velocity
inlet_bc.apply(velocity, patch_idx=0)

# Get matrix contributions for FVM assembly
diag = torch.zeros(10)
source = torch.zeros(10)
diag, source = inlet_bc.matrix_contributions(velocity, 10, diag, source)
```

### Available Boundary Conditions

| Type Name | Class | Description |
|-----------|-------|-------------|
| `fixedValue` | `FixedValueBC` | Prescribed value (penalty method) |
| `zeroGradient` | `ZeroGradientBC` | Zero normal gradient (Neumann) |
| `fixedGradient` | `FixedGradientBC` | Prescribed normal gradient |
| `noSlip` | `NoSlipBC` | Zero velocity (wall) |
| `cyclic` | `CyclicBC` | Periodic coupling |
| `symmetryPlane` | `SymmetryBC` | Symmetry plane |
| `inletOutlet` | `InletOutletBC` | Flow-direction switching |
| `nutkWallFunction` | `NutkWallFunctionBC` | Turbulent viscosity wall function |
| `kqRWallFunction` | `KqRWallFunctionBC` | Turbulence quantity wall function |

## Running a SIMPLE Solver

The SIMPLE (Semi-Implicit Method for Pressure-Linked Equations) algorithm is the standard solver for steady-state incompressible flow.

```python
import torch
from pyfoam.mesh import FvMesh
from pyfoam.fields import volScalarField, volVectorField
from pyfoam.solvers.simple import SIMPLESolver, SIMPLEConfig

# Create mesh and fields (as shown above)
mesh = FvMesh(...)
U = volVectorField(mesh, "U")
p = volScalarField(mesh, "p")

# Configure the solver
config = SIMPLEConfig(
    relaxation_factor_U=0.7,   # Velocity under-relaxation
    relaxation_factor_p=0.3,   # Pressure under-relaxation
    n_correctors=1,            # Pressure correction steps
)

# Create solver
solver = SIMPLESolver(mesh, config)

# Run (U, p, phi are raw tensors)
U_tensor = U.internal_field
p_tensor = p.internal_field
phi_tensor = torch.zeros(mesh.n_faces)  # Face flux

U_result, p_result, phi_result, convergence = solver.solve(
    U_tensor, p_tensor, phi_tensor,
    max_outer_iterations=100,
    tolerance=1e-4,
)

print(f"Converged: {convergence.converged}")
print(f"Iterations: {convergence.outer_iterations}")
print(f"Continuity error: {convergence.continuity_error:.6e}")
```

## Using Linear Solvers Directly

### PCG Solver (Symmetric Systems)

```python
from pyfoam.core import LduMatrix
from pyfoam.solvers.pcg import PCGSolver

# Assume matrix is an LduMatrix instance
solver = PCGSolver(
    tolerance=1e-6,
    rel_tol=0.01,
    max_iter=1000,
    preconditioner="DIC",  # Diagonal Incomplete Cholesky
)

# Solve A x = b
solution, iterations, residual = solver(matrix, source, x0, 1e-6, 1000)
```

### PBiCGStab Solver (Asymmetric Systems)

```python
from pyfoam.solvers.pbicgstab import PBiCGSTABSolver

solver = PBiCGSTABSolver(
    tolerance=1e-6,
    max_iter=1000,
    preconditioner="DILU",  # Diagonal Incomplete LU
)

solution, iterations, residual = solver(matrix, source, x0, 1e-6, 1000)
```

### GAMG Solver (Multigrid)

```python
from pyfoam.solvers.gamg import GAMGSolver

solver = GAMGSolver(
    tolerance=1e-6,
    max_iter=100,
    n_pre_smooth=2,
    n_post_smooth=2,
    max_levels=10,
)

solution, iterations, residual = solver(matrix, source, x0, 1e-6, 100)
```

## GPU Acceleration

### Automatic Device Selection

pyOpenFOAM automatically detects and uses the best available device:

```python
from pyfoam.core import DeviceManager

dm = DeviceManager()
print(dm.capabilities.available_devices)  # ['cpu'] or ['cpu', 'cuda'] etc.
print(dm.device)  # device('cuda') if available, else device('cpu')
```

### Manual Device Selection

```python
from pyfoam.core import device_context
import torch

# Force CPU computation
with device_context(device='cpu'):
    mesh = FvMesh(...)  # All tensors on CPU
    solver = SIMPLESolver(mesh, ...)

# Force GPU computation (requires CUDA)
with device_context(device='cuda'):
    mesh = FvMesh(...)  # All tensors on GPU
    solver = SIMPLESolver(mesh, ...)
```

### Performance Tips

1. **Use float64** — CFD convergence requires double precision. float32 causes divergence.
2. **Pre-compute geometry** — Call `mesh.compute_geometry()` before the simulation loop.
3. **Batch operations** — PyTorch operations are fastest when batched; avoid Python loops over cells/faces.
4. **Profile first** — Use `torch.profiler` to identify bottlenecks before optimizing.

## Next Steps

- Read the [API Reference](api_reference.md) for complete class and function documentation.
- See the [Migration Guide](migration_guide.md) if you're coming from OpenFOAM.
- Check the [GPU Guide](gpu_guide.md) for advanced GPU usage and performance tuning.

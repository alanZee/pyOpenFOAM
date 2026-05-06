# Migration Guide: OpenFOAM to pyOpenFOAM

This guide maps OpenFOAM concepts and syntax to their pyOpenFOAM equivalents. It is written for engineers and researchers already familiar with OpenFOAM who want to use pyOpenFOAM for Python-based CFD workflows.

## Concept Mapping

### Case Structure

| OpenFOAM | pyOpenFOAM | Notes |
|----------|------------|-------|
| Case directory | `Case("path/to/case")` | Reads all config files and mesh |
| `system/controlDict` | `case.controlDict` | Returns `FoamDict` (dict-like) |
| `system/fvSchemes` | `case.fvSchemes` | Parsed dictionary |
| `system/fvSolution` | `case.fvSolution` | Parsed dictionary |
| `constant/polyMesh/` | `case.mesh` | Returns `MeshData` |
| Time directories (`0/`, `1/`, ...) | `case.time_dirs` | Sorted list of strings |
| Field file (`0/U`) | `case.read_field("U", time=0)` | Returns `FieldData` |

**OpenFOAM:**
```bash
# Inspect case
foamInfo pitzDaily
# Read field
foamToVTK -case pitzDaily
```

**pyOpenFOAM:**
```python
from pyfoam.io.case import Case

case = Case("pitzDaily")
print(case.controlDict)
print(case.list_fields(time=0))
field = case.read_field("U", time=0)
```

### Mesh

| OpenFOAM | pyOpenFOAM | Notes |
|----------|------------|-------|
| `polyMesh` | `PolyMesh` | Raw topology (points, faces, owner, neighbour) |
| `fvMesh` | `FvMesh` | Extends `PolyMesh` with geometric quantities |
| `points` | `mesh.points` | `(n_points, 3)` tensor |
| `faces` | `mesh.faces` | `list[Tensor]` — per-face point indices |
| `owner` | `mesh.owner` | `(n_faces,)` tensor |
| `neighbour` | `mesh.neighbour` | `(n_internal_faces,)` tensor |
| `boundary` | `mesh.boundary` | List of patch dicts |
| `V()` | `mesh.cell_volumes` | `(n_cells,)` tensor |
| `C()` | `mesh.cell_centres` | `(n_cells, 3)` tensor |
| `Sf()` | `mesh.face_areas` | `(n_faces, 3)` tensor (area vector) |
| `Cf()` | `mesh.face_centres` | `(n_faces, 3)` tensor |
| `deltaCoeffs()` | `mesh.delta_coefficients` | `(n_faces,)` tensor |
| `weights()` | `mesh.face_weights` | `(n_faces,)` tensor |

**OpenFOAM (C++):**
```cpp
const fvMesh& mesh = runTime.mesh();
const volVectorField& C = mesh.C();
const surfaceScalarField& weights = mesh.weights();
```

**pyOpenFOAM:**
```python
from pyfoam.mesh import FvMesh

mesh = FvMesh(points, faces, owner, neighbour, boundary)
cell_centres = mesh.cell_centres      # lazy-computed
face_weights = mesh.face_weights      # lazy-computed
mesh.compute_geometry()               # pre-compute everything
```

### Fields

| OpenFOAM | pyOpenFOAM | Shape |
|----------|------------|-------|
| `volScalarField p` | `volScalarField(mesh, "p")` | `(n_cells,)` |
| `volVectorField U` | `volVectorField(mesh, "U")` | `(n_cells, 3)` |
| `volTensorField tau` | `volTensorField(mesh, "tau")` | `(n_cells, 3, 3)` |
| `surfaceScalarField phi` | Raw `torch.Tensor` | `(n_faces,)` |
| `p.internalField()` | `p.internal_field` | Direct tensor access |
| `p.boundaryField()` | `p.boundary_field` | `BoundaryField` object |
| `p = pOld + alpha * pPrime` | `p.assign(p_old + alpha * p_prime)` | In-place update |

**OpenFOAM (C++):**
```cpp
volScalarField p
(
    IOobject("p", runTime.timeName(), mesh, IOobject::MUST_READ),
    mesh
);
p = dimensionedScalar("p", dimPressure, 0);
```

**pyOpenFOAM:**
```python
from pyfoam.fields import volScalarField
import torch

p = volScalarField(mesh, "p")
p.assign(torch.zeros(mesh.n_cells))
```

### Field Arithmetic

| OpenFOAM | pyOpenFOAM | Notes |
|----------|------------|-------|
| `p1 + p2` | `p1 + p2` | Dimensions must match |
| `p * 2.0` | `p * 2.0` | Field must be dimensionless |
| `U & U` | `torch.sum(U * U, dim=1)` | Inner product (manual) |
| `fvc::grad(p)` | See discretisation module | Gradient computation |
| `fvc::div(phi)` | See discretisation module | Divergence computation |
| `fvm::laplacian(D, p)` | See discretisation module | Implicit diffusion |

### Boundary Conditions

| OpenFOAM `type` | pyOpenFOAM Class | Registry Name |
|-----------------|------------------|---------------|
| `fixedValue` | `FixedValueBC` | `"fixedValue"` |
| `zeroGradient` | `ZeroGradientBC` | `"zeroGradient"` |
| `fixedGradient` | `FixedGradientBC` | `"fixedGradient"` |
| `noSlip` | `NoSlipBC` | `"noSlip"` |
| `cyclic` | `CyclicBC` | `"cyclic"` |
| `symmetryPlane` | `SymmetryBC` | `"symmetryPlane"` |
| `inletOutlet` | `InletOutletBC` | `"inletOutlet"` |
| `nutkWallFunction` | `NutkWallFunctionBC` | `"nutkWallFunction"` |
| `kqRWallFunction` | `KqRWallFunctionBC` | `"kqRWallFunction"` |

**OpenFOAM (dict):**
```
inlet
{
    type            fixedValue;
    value           uniform (1 0 0);
}
```

**pyOpenFOAM:**
```python
from pyfoam.boundary import BoundaryCondition, Patch

patch = Patch(
    name="inlet",
    face_indices=torch.tensor([0, 1, 2]),
    face_normals=torch.tensor([[-1, 0, 0.0]] * 3),
    face_areas=torch.tensor([0.01, 0.01, 0.01]),
    delta_coeffs=torch.tensor([100.0, 100.0, 100.0]),
    owner_cells=torch.tensor([0, 1, 2]),
)

inlet_bc = BoundaryCondition.create(
    "fixedValue", patch, coeffs={"value": [1.0, 0.0, 0.0]}
)
```

### Linear Solvers

| OpenFOAM `solver` | pyOpenFOAM Class | Use Case |
|-------------------|------------------|----------|
| `PCG` | `PCGSolver` | Symmetric positive-definite (pressure) |
| `PBiCG` / `PBiCGStab` | `PBiCGSTABSolver` | General asymmetric (momentum) |
| `GAMG` | `GAMGSolver` | Algebraic multigrid (any matrix) |
| `smoothSolver` | Not yet implemented | — |
| `DIC` preconditioner | `DICPreconditioner` | For PCG |
| `DILU` preconditioner | `DILUPreconditioner` | For PBiCGStab |
| `FDIC` preconditioner | Not yet implemented | — |

**OpenFOAM (`fvSolution`):**
```
solvers
{
    p
    {
        solver          PCG;
        preconditioner  DIC;
        tolerance       1e-6;
        relTol          0.01;
    }
    U
    {
        solver          PBiCGStab;
        preconditioner  DILU;
        tolerance       1e-6;
        relTol          0.01;
    }
}
```

**pyOpenFOAM:**
```python
from pyfoam.solvers.pcg import PCGSolver
from pyfoam.solvers.pbicgstab import PBiCGSTABSolver

p_solver = PCGSolver(tolerance=1e-6, rel_tol=0.01, preconditioner="DIC")
U_solver = PBiCGSTABSolver(tolerance=1e-6, rel_tol=0.01, preconditioner="DILU")
```

### Coupled Solvers

| OpenFOAM Application | pyOpenFOAM Class | Algorithm |
|---------------------|------------------|-----------|
| `simpleFoam` | `SIMPLESolver` | Steady-state incompressible |
| `pisoFoam` | `PISOSolver` | Transient incompressible |
| `pimpleFoam` | `PIMPLESolver` | Transient with outer iterations |

**OpenFOAM (`fvSolution`):**
```
SIMPLE
{
    nNonOrthogonalCorrectors 0;
    pRefCell 0;
    pRefValue 0;
}

relaxationFactors
{
    fields
    {
        p 0.3;
    }
    equations
    {
        U 0.7;
    }
}
```

**pyOpenFOAM:**
```python
from pyfoam.solvers.simple import SIMPLESolver, SIMPLEConfig

config = SIMPLEConfig(
    n_correctors=1,
    relaxation_factor_p=0.3,
    relaxation_factor_U=0.7,
    p_solver="PCG",
    U_solver="PBiCGStab",
)

solver = SIMPLESolver(mesh, config)
U, p, phi, convergence = solver.solve(U, p, phi, max_outer_iterations=100)
```

### Discretisation Schemes

| OpenFOAM `scheme` | pyOpenFOAM Class |
|-------------------|------------------|
| `linear` | `LinearInterpolation` |
| `upwind` | `UpwindInterpolation` |
| `linearUpwind` | `LinearUpwindInterpolation` |
| `QUICK` | `QuickInterpolation` |

**OpenFOAM (`fvSchemes`):**
```
divSchemes
{
    default         none;
    div(phi,U)      Gauss upwind;
    div(phi,k)      Gauss upwind;
    div(phi,epsilon) Gauss upwind;
}
```

**pyOpenFOAM:**
```python
from pyfoam.discretisation import (
    LinearInterpolation,
    UpwindInterpolation,
    LinearUpwindInterpolation,
    QuickInterpolation,
)

# Create interpolation scheme
upwind = UpwindInterpolation()
face_values = upwind.interpolate(cell_values, face_flux, mesh)
```

## Workflow Comparison

### OpenFOAM Workflow

```bash
# 1. Create case directory structure
mkdir -p pitzDaily/{0,constant/polyMesh,system}

# 2. Edit mesh files (blockMeshDict or snappyHexMeshDict)
vim system/blockMeshDict
blockMesh -case pitzDaily

# 3. Edit boundary conditions
vim 0/U
vim 0/p

# 4. Edit solver settings
vim system/fvSolution
vim system/fvSchemes

# 5. Run solver
simpleFoam -case pitzDaily

# 6. Post-process
paraFoam -case pitzDaily
```

### pyOpenFOAM Workflow

```python
# 1. Load existing case (or create programmatically)
from pyfoam.io.case import Case
case = Case("pitzDaily")

# 2. Create mesh from OpenFOAM files
from pyfoam.mesh import FvMesh
mesh_data = case.mesh
mesh = FvMesh(
    points=mesh_data.points,
    faces=mesh_data.faces,
    owner=mesh_data.owner,
    neighbour=mesh_data.neighbour,
    boundary=mesh_data.boundary,
)

# 3. Create fields
from pyfoam.fields import volScalarField, volVectorField
import torch

U = volVectorField(mesh, "U")
p = volScalarField(mesh, "p")
U.assign(torch.zeros(mesh.n_cells, 3))
p.assign(torch.zeros(mesh.n_cells))

# 4. Configure and run solver
from pyfoam.solvers.simple import SIMPLESolver, SIMPLEConfig

config = SIMPLEConfig(relaxation_factor_U=0.7, relaxation_factor_p=0.3)
solver = SIMPLESolver(mesh, config)

phi = torch.zeros(mesh.n_faces)
U_result, p_result, phi_result, conv = solver.solve(
    U.internal_field, p.internal_field, phi,
    max_outer_iterations=100,
)

# 5. Access results
print(f"Converged: {conv.converged}")
print(f"Max velocity: {U_result.abs().max():.4f}")
```

## Key Differences

### 1. No Time Loop (Yet)

OpenFOAM's `Time` class manages the time loop. In pyOpenFOAM, you manage the time loop yourself:

```python
# OpenFOAM: automatic
while runTime.loop():
    # solve one time step
    ...

# pyOpenFOAM: explicit
for t in range(n_timesteps):
    U, p, phi, conv = solver.solve(U, p, phi, ...)
    # save results manually if needed
```

### 2. No File I/O During Solve

OpenFOAM writes fields to disk at each time step. pyOpenFOAM keeps everything in memory (tensors on GPU/CPU). Write results explicitly when needed.

### 3. Tensor Operations Instead of Field Operations

OpenFOAM uses overloaded operators on field classes. pyOpenFOAM uses the same operators but they produce new tensors or field objects:

```python
# OpenFOAM: field operations modify internal storage
p = p + dp;

# pyOpenFOAM: assign() to update in-place
p.assign(p.internal_field + dp)
```

### 4. Boundary Conditions Are Explicit

In OpenFOAM, boundary conditions are applied automatically during field operations. In pyOpenFOAM, you apply them explicitly:

```python
# Apply all boundary conditions
for bc in p.boundary_field:
    bc.apply(p.internal_field)
```

## Troubleshooting

### "Device 'cuda' is not available"

```python
from pyfoam.core import DeviceManager
dm = DeviceManager()
print(dm.capabilities)  # Check what's available
# Use CPU if CUDA not available
dm.device = 'cpu'
```

### "Field must have a floating-point dtype"

CFD operations require float64 tensors:

```python
import torch
from pyfoam.core import get_default_dtype

# Create tensors with correct dtype
values = torch.zeros(n_cells, dtype=get_default_dtype())  # float64
```

### Divergence in SIMPLE/PISO

If the solver diverges:

1. **Check precision**: Ensure float64 (not float32).
2. **Reduce relaxation**: Lower `relaxation_factor_U` (try 0.3) and `relaxation_factor_p` (try 0.1).
3. **Check mesh quality**: Non-orthogonal meshes need non-orthogonal correctors.
4. **Check boundary conditions**: Ensure inlet/outlet BCs are physically consistent.

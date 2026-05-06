# pyOpenFOAM API Reference

This document provides a comprehensive reference for all public classes, functions, and constants in pyOpenFOAM.

## Table of Contents

- [core — Foundation Layer](#core--foundation-layer)
  - [Device Management](#device-management)
  - [Dtype Utilities](#dtype-utilities)
  - [Backend Operations](#backend-operations)
  - [LDU Matrix](#ldu-matrix)
  - [FvMatrix](#fvmatrix)
  - [Sparse Operations](#sparse-operations)
- [mesh — Mesh Representation](#mesh--mesh-representation)
  - [PolyMesh](#polymesh)
  - [FvMesh](#fvmesh)
  - [Geometry Functions](#geometry-functions)
  - [Topology Utilities](#topology-utilities)
- [fields — Field Classes](#fields--field-classes)
  - [volScalarField](#volscalarfield)
  - [volVectorField](#volvectorfield)
  - [volTensorField](#voltensorfield)
- [boundary — Boundary Conditions](#boundary--boundary-conditions)
  - [BoundaryCondition](#boundarycondition)
  - [Patch](#patch)
  - [FixedValueBC](#fixedvaluebc)
  - [ZeroGradientBC](#zerogradientbc)
  - [CyclicBC](#cyclicbc)
  - [SymmetryBC](#symmetrybc)
  - [NoSlipBC](#noslipbc)
  - [InletOutletBC](#inletoutletbc)
  - [FixedGradientBC](#fixedgradientbc)
  - [Wall Function BCs](#wall-function-bcs)
- [io — File I/O](#io--file-io)
  - [Case](#case)
  - [Dictionary Parsing](#dictionary-parsing)
- [discretisation — FVM Schemes](#discretisation--fvm-schemes)
  - [Interpolation Schemes](#interpolation-schemes)
  - [Weight Functions](#weight-functions)
- [solvers — Linear and Coupled Solvers](#solvers--linear-and-coupled-solvers)
  - [Linear Solvers](#linear-solvers)
  - [Coupled Solvers](#coupled-solvers)

---

## core — Foundation Layer

### Device Management

#### `DeviceManager`

Singleton class for hardware detection and device selection.

```python
from pyfoam.core import DeviceManager

dm = DeviceManager()
```

| Property/Method | Return | Description |
|----------------|--------|-------------|
| `capabilities` | `DeviceCapabilities` | Detected hardware (cpu, cuda, mps, cuda_devices) |
| `device` | `torch.device` | Currently selected device |
| `device.setter(device)` | — | Set active device (raises `ValueError` if unavailable) |
| `is_available(device)` | `bool` | Check if a device type is available |

**`DeviceCapabilities`** (frozen dataclass):

| Field | Type | Description |
|-------|------|-------------|
| `cpu` | `bool` | CPU available (always True) |
| `cuda` | `bool` | CUDA available |
| `mps` | `bool` | MPS (Apple Silicon) available |
| `cuda_devices` | `int` | Number of CUDA devices |
| `available_devices` | `list[str]` | List of available device names |

#### `TensorConfig`

Global tensor configuration for CFD operations. Defaults to float64 on the best available device.

```python
from pyfoam.core import TensorConfig
import torch

config = TensorConfig()
```

| Property/Method | Return | Description |
|----------------|--------|-------------|
| `dtype` | `torch.dtype` | Default dtype (float64) |
| `dtype.setter` | — | Set default dtype |
| `device` | `torch.device` | Current device |
| `device.setter` | — | Set current device |
| `device_manager` | `DeviceManager` | Underlying device manager |
| `tensor(data, **kwargs)` | `torch.Tensor` | Create tensor with defaults |
| `zeros(*size, **kwargs)` | `torch.Tensor` | Create zeros tensor |
| `ones(*size, **kwargs)` | `torch.Tensor` | Create ones tensor |
| `empty(*size, **kwargs)` | `torch.Tensor` | Create empty tensor |
| `full(*size, fill_value, **kwargs)` | `torch.Tensor` | Create filled tensor |
| `override(dtype, device)` | context manager | Temporary dtype/device override |

#### Module-Level Functions

```python
from pyfoam.core import get_device, get_default_dtype, device_context
```

| Function | Return | Description |
|----------|--------|-------------|
| `get_device()` | `torch.device` | Current default device |
| `get_default_dtype()` | `torch.dtype` | Current default dtype (float64) |
| `device_context(device, dtype)` | context manager | Temporary global override |

---

### Dtype Utilities

```python
from pyfoam.core import (
    CFD_DTYPE, CFD_REAL_DTYPE, CFD_COMPLEX_DTYPE, INDEX_DTYPE,
    is_floating, is_complex_dtype, promote_dtype, to_cfd_dtype,
    dtype_to_numpy, numpy_to_torch, real_dtype, complex_dtype, assert_floating,
)
```

| Constant | Value | Description |
|----------|-------|-------------|
| `CFD_DTYPE` | `torch.float64` | Default CFD precision |
| `CFD_REAL_DTYPE` | `torch.float64` | Alias for CFD_DTYPE |
| `CFD_COMPLEX_DTYPE` | `torch.complex128` | Complex CFD dtype |
| `INDEX_DTYPE` | `torch.int64` | Mesh index dtype |

| Function | Signature | Description |
|----------|-----------|-------------|
| `is_floating(dtype)` | `dtype → bool` | True for float16/32/64, complex64/128 |
| `is_complex_dtype(dtype)` | `dtype → bool` | True for complex64/128 |
| `promote_dtype(*dtypes)` | `*dtypes → dtype` | Widest dtype representing all inputs |
| `to_cfd_dtype(tensor)` | `tensor → tensor` | Cast to float64 if floating |
| `dtype_to_numpy(dtype)` | `dtype → np.dtype` | Torch to NumPy dtype |
| `numpy_to_torch(dtype)` | `np.dtype → dtype` | NumPy to torch dtype |
| `real_dtype(dtype)` | `dtype → dtype` | Real counterpart (complex64→float32) |
| `complex_dtype(dtype)` | `dtype → dtype` | Complex counterpart (float64→complex128) |
| `assert_floating(tensor, name)` | — | Raise TypeError if not floating |

---

### Backend Operations

```python
from pyfoam.core import scatter_add, gather, sparse_coo_tensor, sparse_mm, Backend
```

#### `scatter_add(src, index, dim_size, *, dim=0, device=None, dtype=None)`

Accumulate `src` values into output at positions given by `index`. Core primitive for FVM flux assembly.

- **src**: Source values (e.g., face fluxes).
- **index**: Target indices (e.g., owner cells).
- **dim_size**: Output dimension size.
- **Returns**: Output tensor of shape `(dim_size,)`.

#### `gather(src, index, *, dim=0, device=None)`

Collect values from `src` at positions given by `index`. Used for boundary lookups and neighbour access.

- **src**: Source tensor.
- **index**: Indices to collect.
- **Returns**: Gathered values with same shape as `index`.

#### `sparse_coo_tensor(indices, values, size, *, device=None, dtype=None)`

Build a COO sparse tensor. Used during matrix assembly.

- **indices**: `(ndim, nnz)` non-zero coordinates.
- **values**: `(nnz,)` non-zero values.
- **size**: Sparse tensor shape.
- **Returns**: Sparse COO tensor.

#### `sparse_mm(mat, vec, *, device=None)`

Sparse matrix-vector multiply. Accepts COO or CSR format.

- **mat**: Sparse matrix.
- **vec**: Dense vector or matrix.
- **Returns**: Dense result of `mat @ vec`.

#### `Backend` Class

Object-oriented backend binding operations to a specific device/dtype:

```python
backend = Backend(device='cpu', dtype=torch.float32)
result = backend.scatter_add(src, index, dim_size=100)
```

| Property/Method | Description |
|----------------|-------------|
| `config` | Bound TensorConfig |
| `device` | Backend device |
| `dtype` | Backend dtype |
| `scatter_add(...)` | Scatter-add with backend config |
| `gather(...)` | Gather with backend config |
| `sparse_coo_tensor(...)` | COO construction with backend config |
| `sparse_mm(...)` | Sparse matmul with backend config |

---

### LDU Matrix

#### `LduMatrix`

LDU-format sparse matrix for finite volume systems. Stores coefficients in OpenFOAM's native layout.

```python
from pyfoam.core import LduMatrix

matrix = LduMatrix(n_cells, owner, neighbour, device=device, dtype=dtype)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `n_cells` | `int` | Matrix dimension (number of cells) |
| `owner` | `torch.Tensor` | `(n_internal_faces,)` owner cell indices |
| `neighbour` | `torch.Tensor` | `(n_internal_faces,)` neighbour cell indices |

| Property | Shape | Description |
|----------|-------|-------------|
| `n_cells` | `int` | Matrix dimension |
| `n_internal_faces` | `int` | Number of off-diagonal entries per triangle |
| `device` | `torch.device` | Tensor device |
| `dtype` | `torch.dtype` | Floating-point dtype |
| `owner` | `(n_internal_faces,)` | Owner cell indices |
| `neighbour` | `(n_internal_faces,)` | Neighbour cell indices |
| `diag` | `(n_cells,)` | Diagonal coefficients |
| `lower` | `(n_internal_faces,)` | Lower-triangular (owner-side) coefficients |
| `upper` | `(n_internal_faces,)` | Upper-triangular (neighbour-side) coefficients |

| Method | Signature | Description |
|--------|-----------|-------------|
| `Ax(x)` | `(n_cells,) → (n_cells,)` | Matrix-vector product y = A·x |
| `add_to_diag(values)` | `(n_cells,) →` | Add values to diagonal |
| `to_sparse_coo()` | `→ sparse COO` | Convert to COO format |
| `to_sparse_csr()` | `→ sparse CSR` | Convert to CSR format |

---

### FvMatrix

#### `FvMatrix(LduMatrix)`

Finite volume matrix with source, boundary, and relaxation support.

```python
from pyfoam.core import FvMatrix

matrix = FvMatrix(n_cells, owner, neighbour, device=device, dtype=dtype)
```

Inherits all `LduMatrix` properties and methods, plus:

| Property | Shape | Description |
|----------|-------|-------------|
| `source` | `(n_cells,)` | Right-hand side vector |
| `relaxation_factor` | `float` | Current under-relaxation factor |

| Method | Signature | Description |
|--------|-----------|-------------|
| `add_boundary_contribution(bc, field)` | — | Add BC contributions to matrix |
| `add_explicit_source(values)` | `(n_cells,) →` | Add to RHS vector |
| `relax(field_old, factor)` | — | Apply under-relaxation |
| `set_reference(cell_index, value)` | — | Pin reference pressure |
| `solve(solver, x0, tolerance, max_iter)` | → `(solution, iters, residual)` | Solve A·x = b |
| `residual(x)` | `(n_cells,) → (n_cells,)` | Compute r = b - A·x |

#### `LinearSolver` Protocol

Protocol for linear solvers:

```python
class LinearSolver(Protocol):
    def __call__(
        self,
        matrix: LduMatrix,
        source: torch.Tensor,
        x0: torch.Tensor,
        tolerance: float,
        max_iter: int,
    ) -> tuple[torch.Tensor, int, float]: ...
```

---

### Sparse Operations

```python
from pyfoam.core import ldu_to_coo_indices, extract_diagonal, csr_matvec
```

| Function | Signature | Description |
|----------|-----------|-------------|
| `ldu_to_coo_indices(owner, neighbour, n_cells)` | → `(diag_idx, lower_idx, upper_idx)` | Build COO indices from LDU addressing |
| `extract_diagonal(mat)` | → `(n,)` | Extract diagonal from sparse/dense matrix |
| `csr_matvec(mat, vec)` | → dense | CSR sparse matrix-vector product |

---

## mesh — Mesh Representation

### PolyMesh

Raw topological mesh representation.

```python
from pyfoam.mesh import PolyMesh

mesh = PolyMesh(points, faces, owner, neighbour, boundary)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `points` | `(n_points, 3)` tensor | Vertex positions |
| `faces` | `list[tensor]` | Per-face point indices |
| `owner` | `(n_faces,)` tensor | Owner cell per face |
| `neighbour` | `(n_internal_faces,)` tensor | Neighbour cell per internal face |
| `boundary` | `list[dict]` | Patch descriptors: `{name, type, startFace, nFaces}` |

| Property | Return | Description |
|----------|--------|-------------|
| `points` | `(n_points, 3)` | Vertex positions |
| `faces` | `list[tensor]` | Face-vertex indices |
| `owner` | `(n_faces,)` | Owner cell indices |
| `neighbour` | `(n_internal_faces,)` | Neighbour cell indices |
| `boundary` | `list[dict]` | Boundary patches |
| `n_points` | `int` | Number of vertices |
| `n_faces` | `int` | Total faces |
| `n_cells` | `int` | Number of cells |
| `n_internal_faces` | `int` | Internal faces |
| `device` | `torch.device` | Tensor device |
| `dtype` | `torch.dtype` | Floating dtype |

| Method | Signature | Description |
|--------|-----------|-------------|
| `face_points(face_idx)` | → `(n_verts, 3)` | Vertex positions for a face |
| `is_boundary_face(face_idx)` | → `bool` | True if boundary face |
| `patch_faces(patch_idx)` | → `range` | Face range for boundary patch |
| `from_raw(points, faces, owner, neighbour, boundary)` | classmethod | Construct from Python lists |

---

### FvMesh

Extends PolyMesh with lazily-computed geometric quantities.

```python
from pyfoam.mesh import FvMesh

mesh = FvMesh(points, faces, owner, neighbour, boundary)
mesh.compute_geometry()  # Pre-compute everything
```

Inherits all `PolyMesh` properties, plus:

| Property | Shape | Description |
|----------|-------|-------------|
| `face_centres` | `(n_faces, 3)` | Face centre positions |
| `face_areas` | `(n_faces, 3)` | Face area vectors (normal × area) |
| `cell_centres` | `(n_cells, 3)` | Cell centre positions |
| `cell_volumes` | `(n_cells,)` | Cell volumes |
| `face_weights` | `(n_faces,)` | Linear interpolation weights |
| `delta_coefficients` | `(n_faces,)` | Diffusion delta coefficients |
| `face_areas_magnitude` | `(n_faces,)` | Face area magnitudes |
| `face_normals` | `(n_faces, 3)` | Unit face normals |
| `total_volume` | scalar | Sum of all cell volumes |

| Method | Signature | Description |
|--------|-----------|-------------|
| `compute_geometry()` | — | Pre-compute all geometric quantities |
| `from_poly_mesh(mesh)` | classmethod | Create FvMesh from PolyMesh |

---

### Geometry Functions

```python
from pyfoam.mesh.mesh_geometry import (
    compute_face_centres,
    compute_face_area_vectors,
    compute_cell_volumes_and_centres,
    compute_face_weights,
    compute_delta_coefficients,
)
```

| Function | Returns | Description |
|----------|---------|-------------|
| `compute_face_centres(points, faces)` | `(n_faces, 3)` | Face centre positions |
| `compute_face_area_vectors(points, faces)` | `(n_faces, 3)` | Area vectors via fan triangulation |
| `compute_cell_volumes_and_centres(...)` | `(volumes, centres)` | Cell volumes and centres via tet decomposition |
| `compute_face_weights(cell_centres, face_centres, owner, neighbour, n_internal)` | `(n_faces,)` | Distance-based interpolation weights |
| `compute_delta_coefficients(...)` | `(n_faces,)` | 1/|d·n̂| for diffusion |

---

### Topology Utilities

```python
from pyfoam.mesh.topology import (
    validate_owner_neighbour,
    internal_face_mask,
    boundary_face_mask,
    build_cell_to_faces,
    build_face_to_cells,
    cell_neighbours,
)
```

| Function | Returns | Description |
|----------|---------|-------------|
| `validate_owner_neighbour(owner, neighbour, n_cells, n_internal)` | — | Validate conventions; raises ValueError |
| `internal_face_mask(n_faces, n_internal)` | `(n_faces,)` bool | True for internal faces |
| `boundary_face_mask(n_faces, n_internal)` | `(n_faces,)` bool | True for boundary faces |
| `build_cell_to_faces(owner, neighbour, n_cells, n_internal)` | `list[tensor]` | Face indices per cell |
| `build_face_to_cells(owner, neighbour, n_internal)` | `(n_faces, 2)` | Owner/neighbour per face (-1 for boundary) |
| `cell_neighbours(cell, owner, neighbour, n_internal)` | `(n_neigh,)` | Unique neighbour cell indices |

---

## fields — Field Classes

### volScalarField

Cell-centred scalar field. Shape: `(n_cells,)`.

```python
from pyfoam.fields import volScalarField

p = volScalarField(mesh, "p")
p.assign(torch.zeros(mesh.n_cells))
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `mesh` | `FvMesh` | Finite volume mesh |
| `name` | `str` | Field name (e.g., `"p"`) |
| `dimensions` | `DimensionSet` | Physical dimensions (optional) |
| `internal` | `tensor` or `float` | Initial values (optional) |
| `boundary` | `BoundaryField` | Boundary conditions (optional) |

**Inherited Properties:**
- `name`, `dimensions`, `internal_field`, `boundary_field`, `mesh`, `device`, `dtype`, `n_cells`

**Inherited Methods:**
- `assign(values)` — Set internal field values
- `to(device, dtype)` — Copy to different device/dtype

**Arithmetic Operators:**
- `+`, `-`, `*`, `/` (with dimension checking)
- `+=`, `-=`, `*=`, `/=` (in-place)

---

### volVectorField

Cell-centred vector field. Shape: `(n_cells, 3)`.

```python
from pyfoam.fields import volVectorField

U = volVectorField(mesh, "U")
U.assign(torch.zeros(mesh.n_cells, 3))
```

Same API as `volScalarField` but with shape `(n_cells, 3)`.

---

### volTensorField

Cell-centred tensor field. Shape: `(n_cells, 3, 3)`.

```python
from pyfoam.fields import volTensorField

tau = volTensorField(mesh, "tau")
tau.assign(torch.zeros(mesh.n_cells, 3, 3))
```

Same API as `volScalarField` but with shape `(n_cells, 3, 3)`.

---

## boundary — Boundary Conditions

### BoundaryCondition

Abstract base class with Run-Time Selection (RTS) registry.

```python
from pyfoam.boundary import BoundaryCondition

# List available types
print(BoundaryCondition.available_types())
# ['cyclic', 'fixedGradient', 'fixedValue', 'inletOutlet', 'kqRWallFunction',
#  'noSlip', 'nutkWallFunction', 'symmetryPlane', 'zeroGradient']

# Create by name
bc = BoundaryCondition.create("fixedValue", patch, coeffs={"value": 1.0})
```

| Class Method | Signature | Description |
|-------------|-----------|-------------|
| `register(name)` | decorator | Register a BC class under `name` |
| `create(name, patch, coeffs)` | → `BoundaryCondition` | Factory: create by registered name |
| `available_types()` | → `list[str]` | Sorted list of registered names |

| Property | Return | Description |
|----------|--------|-------------|
| `patch` | `Patch` | Bound patch |
| `coeffs` | `dict` | BC coefficients |
| `type_name` | `str` | Registered type name |

| Abstract Method | Signature | Description |
|----------------|-----------|-------------|
| `apply(field, patch_idx)` | → `tensor` | Modify boundary-face values |
| `matrix_contributions(field, n_cells, diag, source)` | → `(diag, source)` | FVM matrix contributions |

---

### Patch

Lightweight boundary patch descriptor.

```python
from pyfoam.boundary import Patch

patch = Patch(
    name="inlet",
    face_indices=torch.tensor([0, 1, 2]),
    face_normals=torch.tensor([[-1, 0, 0.0]] * 3),
    face_areas=torch.tensor([0.01, 0.01, 0.01]),
    delta_coeffs=torch.tensor([100.0, 100.0, 100.0]),
    owner_cells=torch.tensor([0, 1, 2]),
)
```

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Patch name |
| `face_indices` | `(n_faces,)` int tensor | Face indices |
| `face_normals` | `(n_faces, 3)` tensor | Outward unit normals |
| `face_areas` | `(n_faces,)` tensor | Face areas |
| `delta_coeffs` | `(n_faces,)` tensor | 1/distance coefficients |
| `owner_cells` | `(n_faces,)` tensor | Adjacent cell indices |
| `neighbour_patch` | `str` or `None` | Coupled patch name (for cyclic) |

| Property/Method | Description |
|----------------|-------------|
| `n_faces` | Number of faces in patch |
| `to(device)` | Copy with tensors on device |

---

### FixedValueBC

Prescribed value boundary condition. Uses penalty method for matrix contributions.

- **Registry name**: `"fixedValue"`
- **Coefficients**: `{"value": float or tensor}`
- **apply()**: Sets boundary faces to prescribed value.
- **matrix_contributions()**: `diag[c] += deltaCoeff * area`, `source[c] += deltaCoeff * area * value`

---

### ZeroGradientBC

Zero normal gradient (Neumann) boundary condition.

- **Registry name**: `"zeroGradient"`
- **apply()**: Copies owner cell values to boundary faces.
- **matrix_contributions()**: Zero contribution (zero flux by construction).

---

### CyclicBC

Periodic boundary condition coupling two patches.

- **Registry name**: `"cyclic"`
- **Methods**: `set_neighbour_field(field)` — Set coupled patch values.
- **apply()**: Copies neighbour patch values.
- **matrix_contributions()**: `diag[c] += deltaCoeff * area`, `source[c] += deltaCoeff * area * neighbourValue`

---

### SymmetryBC

Symmetry plane boundary condition.

- **Registry name**: `"symmetryPlane"`
- **apply()**: Scalars — zero-gradient; Vectors — projects onto tangent plane.
- **matrix_contributions()**: Zero contribution.

---

### NoSlipBC

Zero velocity wall boundary condition.

- **Registry name**: `"noSlip"`
- **apply()**: Sets boundary faces to zero.
- **matrix_contributions()**: `diag[c] += deltaCoeff * area`, `source[c] = 0`

---

### InletOutletBC

Flow-direction switching boundary condition.

- **Registry name**: `"inletOutlet"`
- **Coefficients**: `{"value": float or tensor}` — inlet prescribed value.
- **apply(field, velocity=None)**: Inflow → fixed value; outflow → zero-gradient.
- **matrix_contributions(field, n_cells, ..., velocity=None)**: Inflow → penalty; outflow → zero.

---

### FixedGradientBC

Prescribed normal gradient boundary condition.

- **Registry name**: `"fixedGradient"`
- **Coefficients**: `{"gradient": float or tensor}`
- **apply()**: `phi_face = phi_cell + gradient * d`
- **matrix_contributions()**: `source[c] += area * gradient` (explicit flux only)

---

### Wall Function BCs

#### `NutkWallFunctionBC`

Turbulent viscosity wall function using log-law.

- **Registry name**: `"nutkWallFunction"`
- **Coefficients**: `Cmu` (0.09), `kappa` (0.41), `E` (9.8)
- **Method**: `compute_nut(k, y, nu)` — Compute nu_t at wall faces.

#### `KqRWallFunctionBC`

Wall function for turbulence quantities (k, q, R).

- **Registry name**: `"kqRWallFunction"`
- **Coefficients**: `Cmu` (0.09)
- **Method**: `compute_k_wall(u_tau)` — k = u_tau^2 / sqrt(Cmu)

---

## io — File I/O

### Case

Complete OpenFOAM case directory representation.

```python
from pyfoam.io.case import Case

case = Case("path/to/case")
```

| Property | Return | Description |
|----------|--------|-------------|
| `root` | `Path` | Case root directory |
| `controlDict` | `FoamDict` | Parsed system/controlDict |
| `fvSchemes` | `FoamDict` | Parsed system/fvSchemes |
| `fvSolution` | `FoamDict` | Parsed system/fvSolution |
| `mesh` | `MeshData` | Raw mesh from constant/polyMesh |
| `boundary` | `list[BoundaryPatch]` | Boundary patch definitions |
| `time_dirs` | `list[str]` | Sorted time directory names |
| `constant_dir` | `Path` | Path to constant/ |
| `system_dir` | `Path` | Path to system/ |
| `mesh_dir` | `Path` | Path to constant/polyMesh/ |

| Method | Signature | Description |
|--------|-----------|-------------|
| `get_time_dir(time)` | → `Path` | Path to time directory |
| `list_fields(time)` | → `list[str]` | Field files in time dir |
| `read_field(name, time)` | → `FieldData` | Read field file |
| `has_field(name, time)` | → `bool` | Check if field exists |
| `has_mesh()` | → `bool` | Check if mesh files exist |
| `get_application()` | → `str` | Application name from controlDict |
| `get_start_time()` | → `float` | startTime from controlDict |
| `get_end_time()` | → `float` | endTime from controlDict |
| `get_delta_t()` | → `float` | deltaT from controlDict |

---

### Dictionary Parsing

```python
from pyfoam.io.dictionary import parse_dict, parse_dict_file, FoamDict
```

| Function | Signature | Description |
|----------|-----------|-------------|
| `parse_dict(text)` | `str → FoamDict` | Parse OpenFOAM dictionary text |
| `parse_dict_file(path)` | `Path → FoamDict` | Parse dictionary file |

`FoamDict` is a dict-like container supporting nested key access with OpenFOAM syntax.

---

## discretisation — FVM Schemes

### Interpolation Schemes

```python
from pyfoam.discretisation import (
    LinearInterpolation,
    UpwindInterpolation,
    LinearUpwindInterpolation,
    QuickInterpolation,
)
```

| Class | Order | Description |
|-------|-------|-------------|
| `LinearInterpolation` | 2nd | `phi_f = w * phi_P + (1-w) * phi_N` |
| `UpwindInterpolation` | 1st | Upstream value based on flux direction |
| `LinearUpwindInterpolation` | 2nd | Upwind-biased with gradient correction |
| `QuickInterpolation` | 3rd | QUICK scheme with deferred correction |

---

### Weight Functions

```python
from pyfoam.discretisation import compute_centre_weights, compute_upwind_weights
```

| Function | Returns | Description |
|----------|---------|-------------|
| `compute_centre_weights(cell_centres, face_centres, owner, neighbour, n_internal, n_faces)` | `(n_faces,)` | Distance-based linear weights |
| `compute_upwind_weights(face_flux, n_internal, n_faces)` | `(weight_owner, weight_neigh)` | Binary upwind weights |

---

## solvers — Linear and Coupled Solvers

### Linear Solvers

#### `PCGSolver`

Preconditioned Conjugate Gradient for symmetric positive-definite matrices.

```python
from pyfoam.solvers.pcg import PCGSolver

solver = PCGSolver(
    tolerance=1e-6,
    rel_tol=0.01,
    max_iter=1000,
    preconditioner="DIC",  # or "DILU" or "none"
)
solution, iters, residual = solver(matrix, source, x0, tolerance, max_iter)
```

#### `PBiCGSTABSolver`

Preconditioned Bi-Conjugate Gradient Stabilised for general (asymmetric) matrices.

```python
from pyfoam.solvers.pbicgstab import PBiCGSTABSolver

solver = PBiCGSTABSolver(
    tolerance=1e-6,
    rel_tol=0.01,
    max_iter=1000,
    preconditioner="DILU",  # or "DIC" or "none"
)
```

#### `GAMGSolver`

Algebraic Multigrid solver with aggregation-based coarsening.

```python
from pyfoam.solvers.gamg import GAMGSolver

solver = GAMGSolver(
    tolerance=1e-6,
    max_iter=100,
    n_pre_smooth=2,
    n_post_smooth=2,
    max_levels=10,
    min_cells_coarse=10,
    smoother="PCG",
)
```

---

### Coupled Solvers

#### `CoupledSolverConfig`

Configuration for pressure-velocity coupled solvers.

```python
from pyfoam.solvers.coupled_solver import CoupledSolverConfig

config = CoupledSolverConfig(
    p_solver="PCG",
    U_solver="PBiCGStab",
    p_tolerance=1e-6,
    U_tolerance=1e-6,
    p_max_iter=1000,
    U_max_iter=1000,
    n_non_orthogonal_correctors=0,
    relaxation_factor_p=1.0,
    relaxation_factor_U=0.7,
    relaxation_factor_phi=1.0,
)
```

#### `ConvergenceData`

Tracks convergence for coupled solves.

| Field | Type | Description |
|-------|------|-------------|
| `p_residual` | `float` | Final pressure residual |
| `U_residual` | `float` | Final velocity residual |
| `continuity_error` | `float` | Global continuity error |
| `outer_iterations` | `int` | Number of outer iterations |
| `converged` | `bool` | Whether solution converged |
| `residual_history` | `list[dict]` | Per-iteration records |

#### `SIMPLESolver`

SIMPLE algorithm for steady-state incompressible flow.

```python
from pyfoam.solvers.simple import SIMPLESolver, SIMPLEConfig

config = SIMPLEConfig(
    relaxation_factor_U=0.7,
    relaxation_factor_p=0.3,
    n_correctors=1,
)

solver = SIMPLESolver(mesh, config)
U, p, phi, convergence = solver.solve(
    U, p, phi,
    max_outer_iterations=100,
    tolerance=1e-4,
)
```

#### `PISOSolver`

PISO algorithm for transient incompressible flow.

```python
from pyfoam.solvers.piso import PISOSolver, PISOConfig

config = PISOConfig(n_correctors=2)
solver = PISOSolver(mesh, config)
U, p, phi, convergence = solver.solve(U, p, phi, U_old=U_old, p_old=p_old)
```

#### `PIMPLESolver`

PIMPLE algorithm (combined PISO + SIMPLE with outer iterations).

```python
from pyfoam.solvers.pimple import PIMPLESolver, PIMPLEConfig

config = PIMPLEConfig(n_outer_correctors=3, n_correctors=1)
solver = PIMPLESolver(mesh, config)
U, p, phi, convergence = solver.solve(U, p, phi, U_old=U_old, p_old=p_old)
```

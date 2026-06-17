---
language:
- en
tags:
- openfoam
- cfd
- computational-fluid-dynamics
- simulation
- reference-data
license: mit
---

# pyOpenFOAM Reference Data

OpenFOAM-11 reference simulation data for validating [pyOpenFOAM](https://github.com/AlanZee/pyOpenFOAM) — a pure Python/PyTorch reimplementation of OpenFOAM.

## Dataset Summary

| Property | Value |
|----------|-------|
| **Total cases** | 229 |
| **Categories** | 21 |
| **Source** | OpenFOAM v11 (Docker) |
| **Uncompressed size** | 3.2 GB |
| **Compressed size** | 1.18 GB |
| **Coverage** | 88.4% of OpenFOAM-13 tutorials (199/225) |

## Categories

| Category | Count | Description |
|----------|-------|-------------|
| `fluid` | ~80 | General incompressible fluid dynamics |
| `incompressibleFluid` | ~30 | Incompressible flow solvers |
| `compressibleVoF` | ~15 | Compressible Volume-of-Fluid multiphase |
| `incompressibleVoF` | ~15 | Incompressible VoF multiphase |
| `multiphaseEuler` | ~10 | Eulerian multiphase |
| `incompressibleMultiphaseVoF` | ~10 | Incompressible multiphase VoF |
| `movingMesh` | ~10 | Dynamic mesh / FSI |
| `solidDisplacement` | ~8 | Structural mechanics |
| `shockFluid` | ~8 | Shock-capturing compressible |
| `multicomponentFluid` | ~8 | Species transport |
| `film` / `isothermalFilm` | ~6 | Thin film flows |
| `legacy` | ~5 | Legacy solvers |
| `potentialFoam` | ~3 | Potential flow |
| Others | ~15 | XiFluid, drift-flux, buoyant, etc. |

## Data Format

Each case directory contains the complete OpenFOAM case structure:

```
case_name/
├── 0/              # Initial/boundary conditions
│   ├── U           # Velocity field
│   ├── p           # Pressure field
│   └── ...         # Other fields (k, epsilon, T, alpha, etc.)
├── constant/
│   ├── polyMesh/   # Mesh (points, faces, owner, neighbour, boundary)
│   └── ...         # Physical properties
└── system/
    ├── controlDict # Simulation control
    ├── fvSchemes   # Discretisation schemes
    └── fvSolution  # Solver settings
```

## Usage

### Download and Extract

```python
from huggingface_hub import hf_hub_download
import tarfile

# Download
path = hf_hub_download(
    repo_id="AlanZee/pyOpenFOAM-reference-data",
    filename="openfoam-reference-data.tar.gz",
    repo_type="dataset"
)

# Extract
with tarfile.open(path, "r:gz") as tar:
    tar.extractall("validation/reference/openfoam/")
```

### Use with pyOpenFOAM

```python
from pyfoam.io.case import Case

# Load a reference case
case = Case("validation/reference/openfoam/fluid_cavity")
mesh = case.mesh()
U = case.read_field("U")
p = case.read_field("p")
```

## Files

| File | Size | Description |
|------|------|-------------|
| `openfoam-reference-data.tar.gz` | 2.42 GB | 246 OpenFOAM v11 reference cases (92% of v13 tutorials) |
| `pyopenfoam-simulation-results.tar.gz` | 47 KB | pyOpenFOAM validation results (34 JSON files) |
| `openfoam13-compiled.tar.gz` | 622 MB | Compiled OpenFOAM-13 Docker image (113 libs, 9 binaries) |
| `README.md` | - | This file |

### OpenFOAM-13 Docker Image

The `openfoam13-compiled.tar.gz` contains a Docker image with OpenFOAM-13 pre-compiled.

```bash
# Load the image
docker load < openfoam13-compiled.tar.gz

# Run a tutorial
docker run --rm -it openfoam13-compiled bash -c "
source /openfoam13/etc/bashrc
cd /openfoam13/tutorials/incompressibleFluid/boxTurb16
blockMesh && foamRun
"
```

**Compiled components:**
- 113 shared libraries (OpenFOAM, finiteVolume, meshTools, etc.)
- 9 binaries: foamRun, blockMesh, setFields, decomposePar, reconstructPar, etc.
- Solver modules: incompressibleFluid, fluidSolver, movingMesh, etc.

## Generation

Reference data was generated using:

```bash
docker run --rm -v $(pwd):/work openfoam/openfoam11-paraview510 \
    bash -c "cd /work && ./Allrun"
```

Each case was run to convergence with OpenFOAM's native solvers, and the final-time fields were saved.

## Related

- **pyOpenFOAM**: [github.com/AlanZee/pyOpenFOAM](https://github.com/AlanZee/pyOpenFOAM)
- **OpenFOAM**: [openfoam.org](https://openfoam.org)
- **OpenFOAM-13**: [openfoam.org/news/…](https://openfoam.org)

## License

MIT

## Citation

```bibtex
@software{pyopenfoam2026,
  author = {AlanZee},
  title = {pyOpenFOAM: Pure Python/PyTorch CFD},
  year = {2026},
  url = {https://github.com/AlanZee/pyOpenFOAM}
}
```

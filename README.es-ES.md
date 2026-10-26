

<div align="center">

# pyOpenFOAM

**Solucionador CFD en Python Puro con Aceleración GPU de PyTorch**

*Una reimplementación en Python de código abierto de OpenFOAM 13 (Foundation)*

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/tests-17130+-brightgreen.svg)](#testing)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Reference_Data-ffd21e.svg)](https://huggingface.co/datasets/AlanZee/pyOpenFOAM-reference-data)

[Español](#overview) | [中文](README_zh.md)

</div>

---

## Descripción General

**pyOpenFOAM** es una reimplementación en Python de código abierto de [OpenFOAM 13](https://openfoam.org/) (Foundation), el ampliamente utilizado conjunto de herramientas de dinámica de fluidos computacional (CFD) en C++. Nuestro objetivo es trasladar las capacidades de OpenFOAM al ecosistema de Python mientras aprovechamos PyTorch para la aceleración por GPU y la diferenciación automática.

### Características Principales

- **64 Solvers de OpenFOAM** — Incompresibles, compresibles, multifásicos, térmicos, y más
- **Aceleración por GPU** — Todas las operaciones de campo utilizan tensores de PyTorch en CUDA/MPS
- **CFD Diferenciable** — Compatibilidad con `torch.autograd` mediante funciones de autograd personalizadas
- **Compatible con OpenFOAM** — Lectura/escritura nativa de casos existentes de OpenFOAM
- **408+ Condiciones de Frontera** — Velocidad, presión, turbulencia, VOF, térmicas
- **Biblioteca Completa de Turbulencia** — RANS (k-epsilon, k-omega SST, S-A, v2f), LES (Smagorinsky, WALE), DES
- **Herramientas de Malla** — blockMesh, snappyHexMesh, convertidores gmsh/fluent/VTK
- **Paralelización MPI** — Descomposición de dominio, intercambio de halo, E/S paralela
- **Seguimiento Lagrangiano de Partículas** — Modelos de inyección, colisión, ruptura y evaporación
- **Multifásico VOF/MULES** — Compresión de interfaz, modelos de cavitación, fuerzas interfaciales
- **Mecánica Estructural** — Solucionador de desplazamiento, modelos elásticos
- **Dinámica de Cuerpos Rígidos** — Uniones, restricciones, solucionadores de movimiento
- **Modelos de Oleaje** — Teorías de ondas de Airy, Stokes y Cnoidal
- **Herramientas Integrales** — checkMesh, setFields, renumberMesh, foamToVTK, y más

---

## Datos de Referencia

Los datos de simulación de referencia de OpenFOAM y la imagen de Docker están disponibles en HuggingFace:

**[AlanZee/pyOpenFOAM-reference-data](https://huggingface.co/datasets/AlanZee/pyOpenFOAM-reference-data)**

| Archivo | Tamaño | Descripción |
|------|------|-------------|
| `openfoam-reference-data.tar.gz` | 2.42 GB | 257 casos de referencia de OpenFOAM (96% de los tutoriales de v13) |
| `pyopenfoam-simulation-results.tar.gz` | 47 KB | Resultados de validación de pyOpenFOAM (34 archivos JSON) |
| Imagen de Docker | 622 MB | Entorno compilado de OpenFOAM-13 (122 libs, 9 binarios) |

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

## Instalación

### Requisitos

- Python ≥ 3.10
- PyTorch ≥ 2.0
- NumPy ≥ 1.24
- SciPy ≥ 1.10

### Instalación Rápida

```bash
git clone https://github.com/alanZee/pyOpenFOAM.git
cd pyOpenFOAM
pip install -r requirements.txt
pip install -e .
```

### Soporte para GPU

```bash
# CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Apple Silicon (MPS)
pip install torch  # MPS support built-in
```

---

## Inicio Rápido

### Ejecutar un Caso de OpenFOAM

```python
from pyfoam.applications import SimpleFoam

solver = SimpleFoam("tutorials/incompressible/simpleFoam/pitzDaily")
solver.run()
```

### Aceleración por GPU

```python
from pyfoam.core import device_context

with device_context("cuda"):
    mesh = FvMesh.from_poly_mesh(poly_mesh)
    solver = SIMPLESolver(mesh, config)
    U, p, phi, info = solver.solve(U, p, phi)
```

### CFD Diferenciable

```python
from pyfoam.differentiable import DifferentiableLaplacian, DifferentiableLinearSolve

# Differentiable Laplacian (supports torch.autograd)
lap = DifferentiableLaplacian.apply(phi, mesh)

# Differentiable linear solve (implicit differentiation)
x = DifferentiableLinearSolve.apply(A, b, tol, max_iter)
```

---

## Arquitectura

```
pyfoam/
├── core/               # Gestión de dispositivos, LDU/FvMatrix, operaciones dispersas, multi-GPU
├── io/                 # E/S de formato de archivo OpenFOAM (ASCII + binario), VTK/Gmsh/Fluent
├── mesh/               # PolyMesh, FvMesh, generación de malla (blockMesh, snappyHexMesh)
├── fields/             # volScalarField, volVectorField, surfaceScalarField
├── boundary/           # 408+ tipos de CF (velocidad, presión, turbulencia, VOF, térmicas)
├── discretisation/     # Operadores fvm/fvc, esquemas de interpolación
├── solvers/            # PCG, PBiCGSTAB, GAMG, SIMPLE/SIMPLEC/PISO/PIMPLE
├── turbulence/         # Modelos RANS, LES, DES + funciones de pared (100+ variantes)
├── thermophysical/     # Gas perfecto, Sutherland, JANAF, termodinámica basada en psi/rho
├── multiphase/         # VOF + MULES, interFoam, Euler-Euler, cavitación
├── parallel/           # Descomposición MPI, intercambio de halo, E/S paralela
├── applications/       # 130+ solvers (incompresibles, compresibles, multifásicos, térmicos)
├── tools/              # checkMesh, setFields, renumberMesh, foamToVTK, etc.
├── postprocessing/     # Marco FunctionObject, fuerzas, y+, salida VTK
├── differentiable/     # Operadores diferenciables, solucionador lineal, SIMPLE
├── lagrangian/         # Seguimiento de partículas, inyección, colisión, ruptura, evaporación
├── waves/              # Modelos de ondas Airy, Stokes, Cnoidal
├── fv/                 # fvModels (fuentes) + fvConstraints
├── ode/                # Solucionadores de ODE (Euler, RK4, RKF45, Rosenbrock)
├── rigid_body/         # Dinámica de cuerpos rígidos, uniones, restricciones
├── structural/         # Mecánica estructural (solucionador de desplazamiento, modelos elásticos)
├── models/             # Modelos físicos (radiación)
└── utils/              # Utilidades compartidas
```

---

## Solvers Implementados

| Categoría | Solvers |
|----------|---------|
| **Incompresibles** | simpleFoam, icoFoam, pisoFoam, pimpleFoam, SRFSimpleFoam, porousSimpleFoam, boundaryFoam |
| **Compresibles** | rhoSimpleFoam, rhoPimpleFoam, sonicFoam, rhoCentralFoam |
| **Flotabilidad** | buoyantSimpleFoam, buoyantPimpleFoam, buoyantBoussinesqSimpleFoam |
| **Térmicos** | laplacianFoam, chtMultiRegionFoam |
| **Multifásicos** | interFoam, multiphaseInterFoam, compressibleInterFoam, twoPhaseEulerFoam, multiphaseEulerFoam, cavitatingFoam |
| **Otros** | potentialFoam, scalarTransportFoam, reactingFoam, solidDisplacementFoam |
| **Optimización** | adjointFoam, adjointShapeOptimizationFoam, adjointTurbulenceFoam |
| **Acústica** | acousticFoam |

---

## Validación

13 casos de referencia definidos frente a soluciones analíticas y datos experimentales/numéricos publicados:

| Caso | Solucionador | Precisión | Referencia |
|------|--------|----------|-----------|
| Flujo de Couette | icoFoam | 0.001% (interno) | Solución analítica de Couette |
| Flujo de Poiseuille | icoFoam | 0.02% (interno) | Solución analítica de Hagen-Poiseuille |
| Cavidad Impulsada por el Tapón (Re=100) | icoFoam | 0.9% (20x20) / 1.0% (32x32) | Ghia et al. 1982 |
| Vórtice de Taylor-Green | icoFoam | — | Taylor & Green 1937 |
| Escalón de Retorno | simpleFoam | — | Driver & Seegmiller 1985 |
| Tubo de Choque de Sod | rhoCentralFoam | — | Sod 1978 |
| Convección Natural (Ra=10^5) | buoyantBoussinesqSimpleFoam | — | de Vahl Davis 1983 |
| Ruptura de Presa | interFoam | — | Martin & Moyce 1952 |
| Canal Turbulento (Re_tau=180) | simpleFoam + kOmegaSST | — | Moser, Kim & Mansour 1999 |
| Tobera Compresible | rhoCentralFoam | — | Teoría de tobera isentrópica |
| Cilindro Laminar (Re=20) | icoFoam | — | Dennis & Chang 1970 |
| Flujo sobre Cilindro (Re=100) | pisoFoam | — | Williamson 1996 |
| Conducto Turbulento (Re=10000) | simpleFoam + kOmegaSST | — | Petukhov 1970 |

```bash
python validation/run_all.py
```

---

## Pruebas

```bash
# Run all tests
pytest tests/unit/ -q --tb=no

# Specific module
pytest tests/unit/solvers/ -q
```

**Resultados**: 17,130 aprobados, 2 xfailed (~410 segundos)

---

## Documentación

| Documento | Descripción |
|----------|-------------|
| [API Index](docs/api/README.md) | Vista general de 24 módulos, conteo de clases, ejemplos de uso, patrón RTS |
| [Module API Reference](docs/api/modules.md) | API detallada para todas las clases y funciones públicas |
| [Getting Started (en)](docs/en/getting_started.md) | Instalación, inicio rápido, guía de GPU |
| [Getting Started (zh)](docs/user_guide/getting_started.md) | Instalación, inicio rápido, guía de GPU (Chino) |
| [Migration Guide](docs/migration_guide.md) | Mapeo de OpenFOAM a pyOpenFOAM (Chino) |
| [Migration Guide (en)](docs/en/migration_guide.md) | Mapeo de OpenFOAM a pyOpenFOAM (Inglés) |
| [Architecture](docs/en/architecture.md) | Arquitectura de alto nivel y decisiones de diseño |
| [GPU Guide](docs/en/gpu_guide.md) | Aceleración por GPU y uso de múltiples GPUs |
| [PROPOSAL.md](docs/PROPOSAL.md) | Requisitos, objetivos, casos de referencia, lista de solvers |
| [DESIGN.md](docs/DESIGN.md) | Arquitectura de alto nivel y decisiones de diseño |
| [ROADMAP.md](docs/ROADMAP.md) | Planes futuros y trabajo pendiente |
| [Validation Report](docs/validation_report.md) | Validación integral de 257 casos con 8 figuras |
| [Per-Case Data](validation/per_case_data/) | Datos JSON para todos los 257 casos de referencia |

---

## Contribuir

¡Bienvenidas las contribuciones! Áreas prioritarias:

1. **Validación** — Ayúdenos a validar frente a OpenFOAM
2. **Diferenciabilidad** — Extender el soporte de autograd
3. **Rendimiento** — Optimizar memoria y cómputo en GPU
4. **Documentación** — Mejorar tutoriales y ejemplos

---

## Licencia

pyOpenFOAM está licenciado bajo la [Licencia Pública General de GNU v3.0](LICENSE).

---

<div align="center">

**Construido para las comunidades de CFD y Python**

[Reportar Error](https://github.com/alanZee/pyOpenFOAM/issues) · [Solicitar Funcionalidad](https://github.com/alanZee/pyOpenFOAM/issues) · [Discusiones](https://github.com/alanZee/pyOpenFOAM/discussions)

</div>

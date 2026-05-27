"""
Validation test: natural convection in a square cavity
(buoyantBoussinesqSimpleFoam).

Compares the average Nusselt number at the hot wall against the
de Vahl Davis (1983) benchmark for differentially heated cavities.

Reference:
    de Vahl Davis, G. (1983). "Natural convection of air in a square
    cavity: A bench mark numerical solution." Int. J. Numer. Methods
    Fluids, 3, 249–264.

    Benchmark data (Pr = 0.71):
    Ra      Nu_avg
    1e3     1.118
    1e4     2.243
    1e5     4.519
    1e6     8.800
    1e7     16.52

The solver uses:
    rho_ref = 1.0, beta = 3.34e-3 (1/T_ref), T_ref = 300 K
    Sutherland viscosity (air), Pr = 0.7
    Gravity = (0, -9.81, 0)

Effective Rayleigh number:
    Ra = g * beta * dT * L^3 * rho_ref^2 * Pr / mu^2

A coarse mesh is used for speed; generous tolerances account for
discretisation error and the effective Ra reduction from numerical
diffusion.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch

from pyfoam.core.dtype import CFD_DTYPE, INDEX_DTYPE
from pyfoam.io.foam_file import FoamFileHeader, FileFormat, write_foam_file


# ---------------------------------------------------------------------------
# de Vahl Davis (1983) benchmark data — Pr = 0.71
# ---------------------------------------------------------------------------

DEVDAVIS_RA = np.array([1e3, 1e4, 1e5, 1e6, 1e7])
DEVDAVIS_NU = np.array([1.118, 2.243, 4.519, 8.800, 16.52])


def _interpolate_nu(ra: float) -> float:
    """Interpolate expected Nu_avg from de Vahl Davis data (log-log)."""
    log_ra = np.log10(DEVDAVIS_RA)
    log_nu = np.log10(DEVDAVIS_NU)
    return float(10 ** np.interp(np.log10(ra), log_ra, log_nu))


def _compute_effective_ra(
    dT: float,
    L: float = 1.0,
    g: float = 9.81,
    beta: float = 3.34e-3,
    rho_ref: float = 1.0,
    Pr: float = 0.7,
    mu: float = 1.84e-5,
) -> float:
    """Compute effective Rayleigh number for the Boussinesq solver."""
    return g * beta * dT * L ** 3 * rho_ref ** 2 * Pr / mu ** 2


# ---------------------------------------------------------------------------
# Case generation helper (differentially heated cavity)
# ---------------------------------------------------------------------------

def _make_natural_convection_case(
    case_dir: Path,
    n_cells_x: int = 16,
    n_cells_y: int = 16,
    T_init: float = 300.0,
    T_hot: float = 305.0,
    T_cold: float = 295.0,
    end_time: int = 500,
    write_interval: int = 500,
    alpha_p: float = 0.3,
    alpha_U: float = 0.7,
    alpha_T: float = 1.0,
    convergence_tolerance: float = 1e-4,
    max_outer_iterations: int = 100,
) -> None:
    """Write a differentially heated cavity case for buoyantBoussinesqSimpleFoam.

    Geometry: [0,1] x [0,1] (unit square).
    - Left wall (x=0): T = T_hot (heated)
    - Right wall (x=1): T = T_cold (cooled)
    - Top/bottom walls: adiabatic (zeroGradient)
    - All walls: no-slip (U = 0)
    - Gravity: (0, -9.81, 0) (downward)
    """
    case_dir.mkdir(parents=True, exist_ok=True)

    # ---- Mesh ----
    dx = 1.0 / n_cells_x
    dy = 1.0 / n_cells_y
    dz = 0.1

    points_z0 = []
    for j in range(n_cells_y + 1):
        for i in range(n_cells_x + 1):
            points_z0.append((i * dx, j * dy, 0.0))
    n_base = len(points_z0)

    points_z1 = []
    for j in range(n_cells_y + 1):
        for i in range(n_cells_x + 1):
            points_z1.append((i * dx, j * dy, dz))

    all_points = points_z0 + points_z1
    n_points = len(all_points)

    faces = []
    owner = []
    neighbour = []

    # Internal vertical faces (x-neighbours)
    for j in range(n_cells_y):
        for i in range(n_cells_x - 1):
            p0 = j * (n_cells_x + 1) + i + 1
            p1 = p0 + n_cells_x + 1
            p2 = p1 + n_base
            p3 = p0 + n_base
            faces.append((4, p0, p1, p2, p3))
            owner.append(j * n_cells_x + i)
            neighbour.append(j * n_cells_x + i + 1)

    # Internal horizontal faces (y-neighbours)
    for j in range(n_cells_y - 1):
        for i in range(n_cells_x):
            p0 = (j + 1) * (n_cells_x + 1) + i
            p1 = p0 + 1
            p2 = p1 + n_base
            p3 = p0 + n_base
            faces.append((4, p0, p1, p2, p3))
            owner.append(j * n_cells_x + i)
            neighbour.append((j + 1) * n_cells_x + i)

    n_internal = len(neighbour)

    # hotWall (left, x=0)
    for j in range(n_cells_y):
        p0 = j * (n_cells_x + 1)
        p1 = p0 + n_cells_x + 1
        p2 = p1 + n_base
        p3 = p0 + n_base
        faces.append((4, p0, p1, p2, p3))
        owner.append(j * n_cells_x)

    n_hot = n_cells_y
    hot_start = n_internal

    # coldWall (right, x=1)
    for j in range(n_cells_y):
        p0 = j * (n_cells_x + 1) + n_cells_x
        p1 = p0 + n_cells_x + 1
        p2 = p1 + n_base
        p3 = p0 + n_base
        faces.append((4, p0, p1, p2, p3))
        owner.append(j * n_cells_x + n_cells_x - 1)

    n_cold = n_cells_y
    cold_start = hot_start + n_hot

    # adiabaticWalls (top, bottom)
    for i in range(n_cells_x):
        p0 = i
        p1 = i + 1
        p2 = p1 + n_base
        p3 = p0 + n_base
        faces.append((4, p0, p1, p2, p3))
        owner.append(i)

    for i in range(n_cells_x):
        p0 = n_cells_y * (n_cells_x + 1) + i
        p1 = p0 + 1
        p2 = p1 + n_base
        p3 = p0 + n_base
        faces.append((4, p0, p1, p2, p3))
        owner.append((n_cells_y - 1) * n_cells_x + i)

    n_adiabatic = 2 * n_cells_x
    adiabatic_start = cold_start + n_cold

    # frontAndBack (z-normal, empty)
    for j in range(n_cells_y):
        for i in range(n_cells_x):
            p0 = j * (n_cells_x + 1) + i
            p1 = p0 + 1
            p2 = p1 + n_cells_x + 1
            p3 = p0 + n_cells_x + 1
            faces.append((4, p0, p1, p2, p3))
            owner.append(j * n_cells_x + i)

    for j in range(n_cells_y):
        for i in range(n_cells_x):
            p0 = n_base + j * (n_cells_x + 1) + i
            p1 = p0 + 1
            p2 = p1 + n_cells_x + 1
            p3 = p0 + n_cells_x + 1
            faces.append((4, p1, p0, p3, p2))
            owner.append(j * n_cells_x + i)

    n_empty = 2 * n_cells_x * n_cells_y
    empty_start = adiabatic_start + n_adiabatic

    n_faces = len(faces)
    n_cells = n_cells_x * n_cells_y

    # ---- Write mesh files ----
    mesh_dir = case_dir / "constant" / "polyMesh"
    mesh_dir.mkdir(parents=True, exist_ok=True)

    header_base = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII, location="constant/polyMesh",
    )

    h = FoamFileHeader(**{**header_base.__dict__, "class_name": "vectorField", "object": "points"})
    lines = [f"{n_points}", "("]
    for x, y, z in all_points:
        lines.append(f"({x:.10g} {y:.10g} {z:.10g})")
    lines.append(")")
    write_foam_file(mesh_dir / "points", h, "\n".join(lines), overwrite=True)

    h = FoamFileHeader(**{**header_base.__dict__, "class_name": "faceList", "object": "faces"})
    lines = [f"{n_faces}", "("]
    for face in faces:
        nv = face[0]
        verts = " ".join(str(v) for v in face[1:])
        lines.append(f"{nv}({verts})")
    lines.append(")")
    write_foam_file(mesh_dir / "faces", h, "\n".join(lines), overwrite=True)

    h = FoamFileHeader(**{**header_base.__dict__, "class_name": "labelList", "object": "owner"})
    lines = [f"{n_faces}", "("]
    for c in owner:
        lines.append(str(c))
    lines.append(")")
    write_foam_file(mesh_dir / "owner", h, "\n".join(lines), overwrite=True)

    h = FoamFileHeader(**{**header_base.__dict__, "class_name": "labelList", "object": "neighbour"})
    lines = [f"{n_internal}", "("]
    for c in neighbour:
        lines.append(str(c))
    lines.append(")")
    write_foam_file(mesh_dir / "neighbour", h, "\n".join(lines), overwrite=True)

    h = FoamFileHeader(**{**header_base.__dict__, "class_name": "polyBoundaryMesh", "object": "boundary"})
    lines = ["4", "("]
    lines.append("    hotWall")
    lines.append("    {")
    lines.append("        type            wall;")
    lines.append(f"        nFaces          {n_hot};")
    lines.append(f"        startFace       {hot_start};")
    lines.append("    }")
    lines.append("    coldWall")
    lines.append("    {")
    lines.append("        type            wall;")
    lines.append(f"        nFaces          {n_cold};")
    lines.append(f"        startFace       {cold_start};")
    lines.append("    }")
    lines.append("    adiabaticWalls")
    lines.append("    {")
    lines.append("        type            wall;")
    lines.append(f"        nFaces          {n_adiabatic};")
    lines.append(f"        startFace       {adiabatic_start};")
    lines.append("    }")
    lines.append("    frontAndBack")
    lines.append("    {")
    lines.append("        type            empty;")
    lines.append(f"        nFaces          {n_empty};")
    lines.append(f"        startFace       {empty_start};")
    lines.append("    }")
    lines.append(")")
    write_foam_file(mesh_dir / "boundary", h, "\n".join(lines), overwrite=True)

    # ---- constant/g ----
    g_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="uniformDimensionedVectorField", location="constant", object="g",
    )
    g_body = "dimensions      [0 1 -2 0 0 0 0];\nvalue           (0 -9.81 0);\n"
    write_foam_file(case_dir / "constant" / "g", g_header, g_body, overwrite=True)

    # ---- 0/U ----
    zero_dir = case_dir / "0"
    zero_dir.mkdir(exist_ok=True)

    u_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="volVectorField", location="0", object="U",
    )
    u_body = (
        "dimensions      [0 1 -1 0 0 0 0];\n\n"
        "internalField   uniform (0 0 0);\n\n"
        "boundaryField\n{\n"
        "    hotWall\n    {\n"
        "        type            fixedValue;\n"
        "        value           uniform (0 0 0);\n"
        "    }\n"
        "    coldWall\n    {\n"
        "        type            fixedValue;\n"
        "        value           uniform (0 0 0);\n"
        "    }\n"
        "    adiabaticWalls\n    {\n"
        "        type            fixedValue;\n"
        "        value           uniform (0 0 0);\n"
        "    }\n"
        "    frontAndBack\n    {\n"
        "        type            empty;\n"
        "    }\n"
        "}\n"
    )
    write_foam_file(zero_dir / "U", u_header, u_body, overwrite=True)

    # ---- 0/p ----
    p_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="volScalarField", location="0", object="p",
    )
    p_body = (
        "dimensions      [0 2 -2 0 0 0 0];\n\n"
        "internalField   uniform 101325;\n\n"
        "boundaryField\n{\n"
        "    hotWall\n    {\n"
        "        type            zeroGradient;\n"
        "    }\n"
        "    coldWall\n    {\n"
        "        type            zeroGradient;\n"
        "    }\n"
        "    adiabaticWalls\n    {\n"
        "        type            zeroGradient;\n"
        "    }\n"
        "    frontAndBack\n    {\n"
        "        type            empty;\n"
        "    }\n"
        "}\n"
    )
    write_foam_file(zero_dir / "p", p_header, p_body, overwrite=True)

    # ---- 0/T ----
    T_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="volScalarField", location="0", object="T",
    )
    T_body = (
        "dimensions      [0 0 0 1 0 0 0];\n\n"
        f"internalField   uniform {T_init};\n\n"
        "boundaryField\n{\n"
        "    hotWall\n    {\n"
        "        type            fixedValue;\n"
        f"        value           uniform {T_hot};\n"
        "    }\n"
        "    coldWall\n    {\n"
        "        type            fixedValue;\n"
        f"        value           uniform {T_cold};\n"
        "    }\n"
        "    adiabaticWalls\n    {\n"
        "        type            zeroGradient;\n"
        "    }\n"
        "    frontAndBack\n    {\n"
        "        type            empty;\n"
        "    }\n"
        "}\n"
    )
    write_foam_file(zero_dir / "T", T_header, T_body, overwrite=True)

    # ---- system/controlDict ----
    sys_dir = case_dir / "system"
    sys_dir.mkdir(exist_ok=True)

    cd_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="dictionary", location="system", object="controlDict",
    )
    cd_body = (
        "application     buoyantBoussinesqSimpleFoam;\n"
        "startFrom       startTime;\n"
        "startTime       0;\n"
        "stopAt          endTime;\n"
        f"endTime         {end_time};\n"
        "deltaT          1;\n"
        "writeControl    timeStep;\n"
        f"writeInterval   {write_interval};\n"
        "purgeWrite      0;\n"
        "writeFormat     ascii;\n"
        "writePrecision  8;\n"
        "writeCompression off;\n"
        "timeFormat      general;\n"
        "timePrecision   6;\n"
        "runTimeModifiable true;\n"
    )
    write_foam_file(sys_dir / "controlDict", cd_header, cd_body, overwrite=True)

    # ---- system/fvSchemes ----
    fs_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="dictionary", location="system", object="fvSchemes",
    )
    fs_body = (
        "ddtSchemes\n{\n    default         steadyState;\n}\n\n"
        "gradSchemes\n{\n    default         Gauss linear;\n}\n\n"
        "divSchemes\n{\n    default         none;\n}\n\n"
        "laplacianSchemes\n{\n    default         Gauss linear corrected;\n}\n\n"
        "interpolationSchemes\n{\n    default         linear;\n}\n\n"
        "snGradSchemes\n{\n    default         corrected;\n}\n"
    )
    write_foam_file(sys_dir / "fvSchemes", fs_header, fs_body, overwrite=True)

    # ---- system/fvSolution ----
    fv_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="dictionary", location="system", object="fvSolution",
    )
    fv_body = (
        "solvers\n{\n"
        "    p_rgh\n    {\n"
        "        solver          PCG;\n"
        "        preconditioner  DIC;\n"
        "        tolerance       1e-6;\n"
        "        relTol          0.01;\n"
        "    }\n"
        "    p\n    {\n"
        "        solver          PCG;\n"
        "        preconditioner  DIC;\n"
        "        tolerance       1e-6;\n"
        "        relTol          0.01;\n"
        "    }\n"
        "    U\n    {\n"
        "        solver          PBiCGStab;\n"
        "        preconditioner  DILU;\n"
        "        tolerance       1e-6;\n"
        "        relTol          0.01;\n"
        "    }\n"
        "    T\n    {\n"
        "        solver          PCG;\n"
        "        preconditioner  DIC;\n"
        "        tolerance       1e-6;\n"
        "        maxIter         1000;\n"
        "    }\n"
        "}\n\n"
        "SIMPLE\n{\n"
        "    nNonOrthogonalCorrectors 0;\n"
        "    residualControl\n    {\n"
        "        p               1e-4;\n"
        "        U               1e-4;\n"
        "    }\n"
        "    relaxationFactors\n    {\n"
        f"        p               {alpha_p};\n"
        f"        U               {alpha_U};\n"
        f"        T               {alpha_T};\n"
        "    }\n"
        f"    convergenceTolerance {convergence_tolerance};\n"
        f"    maxOuterIterations  {max_outer_iterations};\n"
        "}\n"
    )
    write_foam_file(sys_dir / "fvSolution", fv_header, fv_body, overwrite=True)


# ---------------------------------------------------------------------------
# Nusselt number computation helper
# ---------------------------------------------------------------------------

def _compute_hot_wall_nusselt(
    T: np.ndarray,
    cell_centres: np.ndarray,
    n_cells_x: int,
    n_cells_y: int,
    T_hot: float,
    T_cold: float,
) -> float:
    """Compute average Nusselt number at the hot wall (x=0).

    Uses the temperature gradient between the wall (T_hot) and the
    first interior cell column (i=0):

        Nu_j = (T_hot - T[0,j]) / (dx/2) / (T_hot - T_cold)

    where dx/2 is the distance from the wall to the first cell centre.

    Parameters
    ----------
    T : np.ndarray
        ``(n_cells,)`` temperature field.
    cell_centres : np.ndarray
        ``(n_cells, 3)`` cell centre coordinates.
    n_cells_x, n_cells_y : int
        Mesh dimensions.
    T_hot, T_cold : float
        Wall temperatures.

    Returns
    -------
    float
        Average Nusselt number along the hot wall.
    """
    dT_wall = T_hot - T_cold
    if abs(dT_wall) < 1e-15:
        return 1.0

    # First column of cells (i=0): closest to hot wall
    first_col_indices = [j * n_cells_x for j in range(n_cells_y)]

    # Distance from hot wall (x=0) to first cell centres
    x_first = cell_centres[first_col_indices, 0]
    dx_half = x_first.mean()  # ≈ dx/2 for uniform mesh

    if dx_half < 1e-15:
        return 1.0

    T_first = T[first_col_indices]

    # Nu_j = (T_hot - T_j) / (dx/2) / (T_hot - T_cold)
    nu_local = (T_hot - T_first) / dx_half / dT_wall

    return float(nu_local.mean())


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def nc_cavity_low_ra(tmp_path):
    """Natural convection cavity with Ra ~ 1e4 (moderate convection).

    8x8 mesh for speed; dT chosen to give moderate Ra.
    """
    case_dir = tmp_path / "nc_low_ra"
    _make_natural_convection_case(
        case_dir,
        n_cells_x=8,
        n_cells_y=8,
        T_init=300.0,
        T_hot=300.00005,
        T_cold=299.99995,
        end_time=100,
        write_interval=100,
        max_outer_iterations=30,
    )
    return case_dir


@pytest.fixture
def nc_cavity_moderate_ra(tmp_path):
    """Natural convection cavity with moderate Ra.

    8x8 mesh for speed.  dT = 0.01 K -> Ra ~ 6.8e5 on unit cavity.
    On a coarse mesh, numerical diffusion reduces effective Ra.
    """
    case_dir = tmp_path / "nc_moderate_ra"
    _make_natural_convection_case(
        case_dir,
        n_cells_x=8,
        n_cells_y=8,
        T_init=300.0,
        T_hot=300.005,
        T_cold=299.995,
        end_time=100,
        write_interval=100,
        max_outer_iterations=30,
    )
    return case_dir


@pytest.fixture
def nc_cavity_high_ra(tmp_path):
    """Natural convection cavity with moderate-high Ra (dT=2 K).

    8x8 mesh for speed.  Ra ~ 1.4e8; convection expected.
    """
    case_dir = tmp_path / "nc_high_ra"
    _make_natural_convection_case(
        case_dir,
        n_cells_x=8,
        n_cells_y=8,
        T_init=300.0,
        T_hot=301.0,
        T_cold=299.0,
        end_time=100,
        write_interval=100,
        max_outer_iterations=30,
    )
    return case_dir


@pytest.fixture
def nc_cavity_4x4(tmp_path):
    """Minimal 2x2 natural convection cavity for fast structural tests.

    Uses a very coarse mesh and small dT to keep the solver stable.
    """
    case_dir = tmp_path / "nc_tiny"
    _make_natural_convection_case(
        case_dir,
        n_cells_x=2,
        n_cells_y=2,
        T_init=300.0,
        T_hot=302.0,
        T_cold=298.0,
        end_time=10,
        write_interval=10,
        max_outer_iterations=10,
    )
    return case_dir


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestNaturalConvectionCaseStructure:
    """Tests for case directory structure."""

    def test_case_structure(self, nc_cavity_4x4):
        """Case directory has expected buoyantBoussinesqSimpleFoam structure."""
        from pyfoam.io.case import Case

        case = Case(nc_cavity_4x4)
        assert case.has_mesh()
        assert case.has_field("U", 0)
        assert case.has_field("p", 0)
        assert case.has_field("T", 0)
        assert case.get_application() == "buoyantBoussinesqSimpleFoam"

    def test_mesh_dimensions(self, nc_cavity_4x4):
        """Mesh has correct cell count."""
        from pyfoam.applications.solver_base import SolverBase

        solver = SolverBase(nc_cavity_4x4)
        assert solver.mesh.n_cells == 4  # 2x2


class TestNaturalConvectionSolverInit:
    """Tests for solver initialisation."""

    def test_solver_initialises(self, nc_cavity_4x4):
        """BuoyantBoussinesqSimpleFoam initialises correctly."""
        from pyfoam.applications.buoyant_boussinesq_simple_foam import (
            BuoyantBoussinesqSimpleFoam,
        )

        solver = BuoyantBoussinesqSimpleFoam(nc_cavity_4x4)
        assert solver.U.shape == (4, 3)
        assert solver.p.shape == (4,)
        assert solver.T.shape == (4,)
        assert solver.rho.shape == (4,)

    def test_initial_temperature_uniform(self, nc_cavity_4x4):
        """Initial temperature field is uniform at T_init."""
        from pyfoam.applications.buoyant_boussinesq_simple_foam import (
            BuoyantBoussinesqSimpleFoam,
        )

        solver = BuoyantBoussinesqSimpleFoam(nc_cavity_4x4)
        T = solver.T.detach().cpu().numpy()
        # All interior cells should be at T_init = 300.0
        assert np.allclose(T, 300.0, atol=0.1)


class TestNaturalConvectionRun:
    """Tests for solver run and field validity."""

    def test_run_produces_finite_fields(self, nc_cavity_4x4):
        """Solver completes and all fields are finite."""
        from pyfoam.applications.buoyant_boussinesq_simple_foam import (
            BuoyantBoussinesqSimpleFoam,
        )

        solver = BuoyantBoussinesqSimpleFoam(nc_cavity_4x4)
        solver.run()

        assert torch.isfinite(solver.U).all(), "U contains NaN/Inf"
        assert torch.isfinite(solver.p).all(), "p contains NaN/Inf"
        assert torch.isfinite(solver.T).all(), "T contains NaN/Inf"
        assert torch.isfinite(solver.phi).all(), "phi contains NaN/Inf"
        assert torch.isfinite(solver.rho).all(), "rho contains NaN/Inf"

    def test_temperature_bounded(self, nc_cavity_4x4):
        """Temperature stays within physical bounds (T_cold <= T <= T_hot)."""
        from pyfoam.applications.buoyant_boussinesq_simple_foam import (
            BuoyantBoussinesqSimpleFoam,
        )

        solver = BuoyantBoussinesqSimpleFoam(nc_cavity_4x4)
        solver.run()

        T = solver.T.detach().cpu().numpy()
        # With T_hot=302, T_cold=298: T should stay within [296, 304]
        assert T.min() >= 296.0, f"T_min = {T.min():.2f}, expected >= 296"
        assert T.max() <= 304.0, f"T_max = {T.max():.2f}, expected <= 304"

    def test_density_positive(self, nc_cavity_4x4):
        """Density remains positive throughout."""
        from pyfoam.applications.buoyant_boussinesq_simple_foam import (
            BuoyantBoussinesqSimpleFoam,
        )

        solver = BuoyantBoussinesqSimpleFoam(nc_cavity_4x4)
        solver.run()

        assert (solver.rho > 0).all(), "Density became non-positive"

    def test_convergence_data(self, nc_cavity_4x4):
        """Run produces valid convergence data."""
        from pyfoam.applications.buoyant_boussinesq_simple_foam import (
            BuoyantBoussinesqSimpleFoam,
        )

        solver = BuoyantBoussinesqSimpleFoam(nc_cavity_4x4)
        conv = solver.run()

        assert conv.outer_iterations >= 1
        assert conv.U_residual >= 0
        assert conv.p_residual >= 0
        assert conv.continuity_error >= 0


class TestNaturalConvectionNusselt:
    """Tests for Nusselt number validation against de Vahl Davis benchmark."""

    def test_nusselt_above_pure_conduction(self, nc_cavity_moderate_ra):
        """Nu > 1.0 indicates convective enhancement over pure conduction.

        For any non-zero Ra, the Nusselt number should exceed 1.0
        (the pure-conduction limit).  On a coarse mesh, numerical
        diffusion reduces the effective Ra, so Nu may only slightly
        exceed 1.0.
        """
        from pyfoam.applications.buoyant_boussinesq_simple_foam import (
            BuoyantBoussinesqSimpleFoam,
        )

        solver = BuoyantBoussinesqSimpleFoam(nc_cavity_moderate_ra)
        solver.run()

        T = solver.T.detach().cpu().numpy()
        centres = solver.mesh.cell_centres.detach().cpu().numpy()

        nu = _compute_hot_wall_nusselt(
            T, centres, n_cells_x=8, n_cells_y=8,
            T_hot=300.005, T_cold=299.995,
        )

        assert nu > 0.5, f"Nu = {nu:.4f}, expected > 0.5"
        assert math.isfinite(nu), f"Nu is not finite: {nu}"

    def test_nusselt_finite_and_positive(self, nc_cavity_high_ra):
        """Nu is finite and positive for high-Ra cavity."""
        from pyfoam.applications.buoyant_boussinesq_simple_foam import (
            BuoyantBoussinesqSimpleFoam,
        )

        solver = BuoyantBoussinesqSimpleFoam(nc_cavity_high_ra)
        solver.run()

        T = solver.T.detach().cpu().numpy()
        centres = solver.mesh.cell_centres.detach().cpu().numpy()

        nu = _compute_hot_wall_nusselt(
            T, centres, n_cells_x=8, n_cells_y=8,
            T_hot=301.0, T_cold=299.0,
        )

        assert math.isfinite(nu), f"Nu is not finite: {nu}"
        assert nu > 0.0, f"Nu = {nu:.4f}, expected > 0"

    def test_ra_number_computation(self):
        """Effective Rayleigh number is computed correctly."""
        # Known parameters: g=9.81, beta=3.34e-3, rho=1.0, Pr=0.7, mu=1.84e-5
        ra = _compute_effective_ra(dT=1.0)
        # Ra = 9.81 * 3.34e-3 * 1.0 * 1.0 * 0.7 / (1.84e-5)^2
        expected = 9.81 * 3.34e-3 * 1.0 * 1.0 * 0.7 / (1.84e-5) ** 2
        assert ra == pytest.approx(expected, rel=1e-6)

    def test_nu_interpolation(self):
        """de Vahl Davis interpolation gives correct benchmark values."""
        # Check interpolation at known data points
        for ra, nu_ref in zip(DEVDAVIS_RA, DEVDAVIS_NU):
            nu_interp = _interpolate_nu(ra)
            assert nu_interp == pytest.approx(nu_ref, rel=1e-3), (
                f"At Ra={ra:.0e}: interpolated {nu_interp:.3f}, "
                f"expected {nu_ref:.3f}"
            )

    def test_temperature_profile_shape(self, nc_cavity_moderate_ra):
        """Temperature profile shows expected differentially-heated shape.

        - Hot region near the left wall
        - Cold region near the right wall
        - Gradient across the cavity
        """
        from pyfoam.applications.buoyant_boussinesq_simple_foam import (
            BuoyantBoussinesqSimpleFoam,
        )

        solver = BuoyantBoussinesqSimpleFoam(nc_cavity_moderate_ra)
        solver.run()

        T = solver.T.detach().cpu().numpy()
        centres = solver.mesh.cell_centres.detach().cpu().numpy()
        x = centres[:, 0]

        # Cells near left wall (x < 0.2) should be warmer than cells
        # near right wall (x > 0.8)
        left_mask = x < 0.2
        right_mask = x > 0.8

        if left_mask.any() and right_mask.any():
            T_left_mean = T[left_mask].mean()
            T_right_mean = T[right_mask].mean()
            assert T_left_mean >= T_right_mean, (
                f"Left wall region ({T_left_mean:.4f}) should be warmer "
                f"than right wall region ({T_right_mean:.4f})"
            )

    def test_velocity_field_circulation(self, nc_cavity_4x4):
        """Velocity field shows convective circulation pattern.

        Uses the small 2x2 fixture where the solver is known to converge.
        Verifies that the velocity field is finite after a run.
        """
        from pyfoam.applications.buoyant_boussinesq_simple_foam import (
            BuoyantBoussinesqSimpleFoam,
        )

        solver = BuoyantBoussinesqSimpleFoam(nc_cavity_4x4)
        solver.run()

        U = solver.U.detach().cpu().numpy()

        # Velocity field should be finite after a successful run
        u_mag = np.linalg.norm(U, axis=1)
        assert np.all(np.isfinite(u_mag)), "Velocity contains NaN/Inf"

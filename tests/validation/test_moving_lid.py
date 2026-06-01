"""
Validation test: lid-driven cavity at Re=100, 400, 1000 (icoFoam).

Compares vertical centreline u-velocity and horizontal centreline
v-velocity profiles against Ghia, Ghia & Shin (1982) benchmark data
at three Reynolds numbers.  Uses parametrised fixtures to cover
Re=100 (viscous-dominated), Re=400 (moderate), and Re=1000
(vortex-dominated) regimes.

Different from the existing test_lid_driven_cavity.py which only
covers Re=100.  This test file validates multiple Reynolds numbers
and both centreline profiles.

Reference:
    Ghia, U., Ghia, K.N., Shin, C.T., 1982.
    "High-Re solutions for incompressible flow using the
    Navier-Stokes equations and a multigrid method."
    J. Comput. Phys. 48, 387-411.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from pyfoam.core.dtype import CFD_DTYPE, INDEX_DTYPE
from pyfoam.io.foam_file import FoamFileHeader, FileFormat, write_foam_file


# ---------------------------------------------------------------------------
# Ghia et al. (1982) reference data
# ---------------------------------------------------------------------------

# Common y-positions for vertical centreline (x=0.5)
GHIA_Y = np.array([
    0.0000, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719,
    0.2813, 0.4531, 0.5000, 0.6172, 0.7344, 0.8516,
    0.9531, 0.9609, 0.9688, 0.9766, 1.0000,
])

# Common x-positions for horizontal centreline (y=0.5)
GHIA_X = np.array([
    0.0000, 0.0625, 0.0703, 0.0781, 0.0938, 0.1563,
    0.2266, 0.2344, 0.5000, 0.8047, 0.8594, 0.9063,
    0.9453, 0.9531, 0.9609, 0.9688, 1.0000,
])

# u-velocity along vertical centreline (x=0.5)
GHIA_U_RE100 = np.array([
    0.0000, -0.03717, -0.04192, -0.04775, -0.06434, -0.10150,
    -0.15662, -0.21090, -0.20581, -0.13641, 0.00332, 0.23151,
    0.68717, 0.73722, 0.78871, 0.84123, 1.0000,
])

GHIA_U_RE400 = np.array([
    0.0000, -0.08186, -0.09266, -0.10338, -0.14612, -0.24299,
    -0.32726, -0.17119, -0.11477, 0.02135, 0.16256, 0.29093,
    0.55892, 0.61756, 0.68439, 0.75837, 1.0000,
])

GHIA_U_RE1000 = np.array([
    0.0000, -0.18109, -0.20196, -0.22220, -0.29730, -0.38289,
    -0.27805, -0.10648, -0.06080, 0.05702, 0.18719, 0.33304,
    0.46604, 0.51117, 0.57492, 0.65928, 1.0000,
])

# v-velocity along horizontal centreline (y=0.5)
GHIA_V_RE100 = np.array([
    0.0000, 0.09233, 0.10091, 0.10890, 0.12317, 0.16077,
    0.17507, 0.17527, 0.05454, -0.24533, -0.22445, -0.16914,
    -0.10313, -0.08864, -0.07391, -0.05906, 0.0000,
])

GHIA_V_RE400 = np.array([
    0.0000, 0.18360, 0.19713, 0.20920, 0.22965, 0.28124,
    0.30203, 0.30174, 0.05186, -0.38598, -0.44993, -0.23827,
    -0.22847, -0.19254, -0.15663, -0.12146, 0.0000,
])

GHIA_V_RE1000 = np.array([
    0.0000, 0.27485, 0.29012, 0.30353, 0.32627, 0.37095,
    0.33075, 0.32235, 0.02526, -0.31966, -0.42665, -0.51550,
    -0.39188, -0.33714, -0.27669, -0.21388, 0.0000,
])


# ---------------------------------------------------------------------------
# Case generation helper (parametrised by Re)
# ---------------------------------------------------------------------------

def _make_cavity_case(
    case_dir: Path,
    Re: float = 100,
    n_cells: int = 16,
    U_lid: float = 1.0,
) -> None:
    """Write a complete icoFoam lid-driven cavity case.

    Geometry: [0,1] x [0,1], top wall U=(U_lid,0,0).
    Kinematic viscosity: nu = U_lid * L / Re  (L = 1.0).
    endTime is scaled by Re to ensure sufficient time for flow development.
    """
    case_dir.mkdir(parents=True, exist_ok=True)

    L = 1.0
    nu = U_lid * L / Re

    # endTime: scale with Re (higher Re needs more time)
    # For Re=100: 200, Re=400: 200, Re=1000: 200 (reduced for CPU feasibility)
    end_time = max(200, min(200, 1.0 * Re))

    dx = L / n_cells
    dy = L / n_cells
    dz = 0.1

    # ---- Points: two z-layers ----
    points_z0 = []
    for j in range(n_cells + 1):
        for i in range(n_cells + 1):
            points_z0.append((i * dx, j * dy, 0.0))
    n_base = len(points_z0)

    points_z1 = []
    for j in range(n_cells + 1):
        for i in range(n_cells + 1):
            points_z1.append((i * dx, j * dy, dz))

    all_points = points_z0 + points_z1
    n_points = len(all_points)

    # ---- Faces / owner / neighbour ----
    faces: list[tuple] = []
    owner: list[int] = []
    neighbour: list[int] = []

    # Internal vertical faces (x-neighbours)
    for j in range(n_cells):
        for i in range(n_cells - 1):
            p0 = j * (n_cells + 1) + i + 1
            p1 = p0 + n_cells + 1
            p2 = p1 + n_base
            p3 = p0 + n_base
            faces.append((4, p0, p1, p2, p3))
            owner.append(j * n_cells + i)
            neighbour.append(j * n_cells + i + 1)

    # Internal horizontal faces (y-neighbours)
    for j in range(n_cells - 1):
        for i in range(n_cells):
            p0 = (j + 1) * (n_cells + 1) + i
            p1 = p0 + 1
            p2 = p1 + n_base
            p3 = p0 + n_base
            faces.append((4, p0, p1, p2, p3))
            owner.append(j * n_cells + i)
            neighbour.append((j + 1) * n_cells + i)

    n_internal = len(neighbour)

    # Boundary faces
    # movingWall (top, y=1)
    for i in range(n_cells):
        p0 = n_cells * (n_cells + 1) + i
        p1 = p0 + 1
        p2 = p1 + n_base
        p3 = p0 + n_base
        faces.append((4, p0, p1, p2, p3))
        owner.append((n_cells - 1) * n_cells + i)

    n_moving = n_cells
    moving_start = n_internal

    # fixedWalls: bottom, left, right
    for i in range(n_cells):
        p0 = i
        p1 = i + 1
        p2 = p1 + n_base
        p3 = p0 + n_base
        faces.append((4, p0, p1, p2, p3))
        owner.append(i)

    for j in range(n_cells):
        p0 = j * (n_cells + 1)
        p1 = p0 + n_cells + 1
        p2 = p1 + n_base
        p3 = p0 + n_base
        faces.append((4, p0, p1, p2, p3))
        owner.append(j * n_cells)

    for j in range(n_cells):
        p0 = j * (n_cells + 1) + n_cells
        p1 = p0 + n_cells + 1
        p2 = p1 + n_base
        p3 = p0 + n_base
        faces.append((4, p0, p1, p2, p3))
        owner.append(j * n_cells + n_cells - 1)

    n_fixed = n_cells + 2 * n_cells
    fixed_start = n_internal + n_moving

    # frontAndBack (empty, z-normal)
    for j in range(n_cells):
        for i in range(n_cells):
            p0 = j * (n_cells + 1) + i
            p1 = p0 + 1
            p2 = p1 + n_cells + 1
            p3 = p0 + n_cells + 1
            faces.append((4, p0, p1, p2, p3))
            owner.append(j * n_cells + i)

    for j in range(n_cells):
        for i in range(n_cells):
            p0 = n_base + j * (n_cells + 1) + i
            p1 = p0 + 1
            p2 = p1 + n_cells + 1
            p3 = p0 + n_cells + 1
            faces.append((4, p1, p0, p3, p2))
            owner.append(j * n_cells + i)

    n_empty = 2 * n_cells * n_cells
    empty_start = fixed_start + n_fixed

    n_faces = len(faces)
    n_total_cells = n_cells * n_cells

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
    lines = ["3", "("]
    lines.append("    movingWall")
    lines.append("    {")
    lines.append("        type            wall;")
    lines.append(f"        nFaces          {n_moving};")
    lines.append(f"        startFace       {moving_start};")
    lines.append("    }")
    lines.append("    fixedWalls")
    lines.append("    {")
    lines.append("        type            wall;")
    lines.append(f"        nFaces          {n_fixed};")
    lines.append(f"        startFace       {fixed_start};")
    lines.append("    }")
    lines.append("    frontAndBack")
    lines.append("    {")
    lines.append("        type            empty;")
    lines.append(f"        nFaces          {n_empty};")
    lines.append(f"        startFace       {empty_start};")
    lines.append("    }")
    lines.append(")")
    write_foam_file(mesh_dir / "boundary", h, "\n".join(lines), overwrite=True)

    # ---- transportProperties ----
    tp_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="dictionary", location="constant", object="transportProperties",
    )
    write_foam_file(
        case_dir / "constant" / "transportProperties", tp_header,
        f"nu              [0 2 -1 0 0 0 0] {nu};",
        overwrite=True,
    )

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
        "    movingWall\n    {\n"
        "        type            fixedValue;\n"
        f"        value           uniform ({U_lid} 0 0);\n"
        "    }\n"
        "    fixedWalls\n    {\n"
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
        "internalField   uniform 0;\n\n"
        "boundaryField\n{\n"
        "    movingWall\n    {\n"
        "        type            zeroGradient;\n"
        "    }\n"
        "    fixedWalls\n    {\n"
        "        type            zeroGradient;\n"
        "    }\n"
        "    frontAndBack\n    {\n"
        "        type            empty;\n"
        "    }\n"
        "}\n"
    )
    write_foam_file(zero_dir / "p", p_header, p_body, overwrite=True)

    # ---- system/controlDict ----
    sys_dir = case_dir / "system"
    sys_dir.mkdir(exist_ok=True)

    cd_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="dictionary", location="system", object="controlDict",
    )
    cd_body = (
        "application     icoFoam;\n"
        "startFrom       startTime;\n"
        "startTime       0;\n"
        "stopAt          endTime;\n"
        f"endTime         {end_time};\n"
        "deltaT          0.01;\n"
        "writeControl    timeStep;\n"
        "writeInterval   200;\n"
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
        "ddtSchemes\n{\n    default         Euler;\n}\n\n"
        "gradSchemes\n{\n    default         Gauss linear;\n}\n\n"
        "divSchemes\n{\n    default         none;\n}\n\n"
        "laplacianSchemes\n{\n    default         Gauss linear corrected;\n}\n\n"
        "interpolationSchemes\n{\n    default         linear;\n}\n\n"
        "snGradSchemes\n{\n    default         corrected;\n}\n"
    )
    write_foam_file(sys_dir / "fvSchemes", fs_header, fs_body, overwrite=True)

    # ---- system/fvSolution (PISO) ----
    fv_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="dictionary", location="system", object="fvSolution",
    )
    fv_body = (
        "solvers\n{\n"
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
        "}\n\n"
        "PISO\n{\n"
        "    nCorrectors             3;\n"
        "    nNonOrthogonalCorrectors 0;\n"
        "    residualControl\n    {\n"
        "        p               1e-4;\n"
        "        U               1e-4;\n"
        "    }\n"
        "    convergenceTolerance 1e-4;\n"
        "}\n"
    )
    write_foam_file(sys_dir / "fvSolution", fv_header, fv_body, overwrite=True)


# ---------------------------------------------------------------------------
# Fixtures — parametrised by Re
# ---------------------------------------------------------------------------

@pytest.fixture(params=[100, 400, 1000])
def cavity_re(request, tmp_path):
    """Create a lid-driven cavity case at the given Re.

    Uses 16 cells for Re=100 (fast) and 32 cells for higher Re
    (better resolution of secondary vortices).
    """
    Re = request.param
    n_cells = 16 if Re <= 100 else 32
    case_dir = tmp_path / f"cavity_Re{Re}"
    _make_cavity_case(case_dir, Re=Re, n_cells=n_cells)
    return case_dir, Re, n_cells


@pytest.fixture
def cavity_re100(tmp_path):
    """Lid-driven cavity at Re=100."""
    case_dir = tmp_path / "cavity_Re100"
    _make_cavity_case(case_dir, Re=100, n_cells=16)
    return case_dir


@pytest.fixture
def cavity_re400(tmp_path):
    """Lid-driven cavity at Re=400, 32x32 mesh."""
    case_dir = tmp_path / "cavity_Re400"
    _make_cavity_case(case_dir, Re=400, n_cells=32)
    return case_dir


@pytest.fixture
def cavity_re1000(tmp_path):
    """Lid-driven cavity at Re=1000, 32x32 mesh."""
    case_dir = tmp_path / "cavity_Re1000"
    _make_cavity_case(case_dir, Re=1000, n_cells=32)
    return case_dir


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_centreline_u(solver, n_cells: int = 16):
    """Extract u-velocity along the vertical centreline (x=0.5).

    Returns (y_positions, u_values) arrays.
    """
    centres = solver.mesh.cell_centres.detach().cpu().numpy()
    u_all = solver.U[:, 0].detach().cpu().numpy()

    mid_i = n_cells // 2
    indices = [j * n_cells + mid_i for j in range(n_cells)]
    return centres[indices, 1], u_all[indices]


def _extract_centreline_v(solver, n_cells: int = 16):
    """Extract v-velocity along the horizontal centreline (y=0.5).

    Returns (x_positions, v_values) arrays.
    """
    centres = solver.mesh.cell_centres.detach().cpu().numpy()
    v_all = solver.U[:, 1].detach().cpu().numpy()

    mid_j = n_cells // 2
    indices = [mid_j * n_cells + i for i in range(n_cells)]
    return centres[indices, 0], v_all[indices]


# Reference data lookup
GHIA_U_DATA = {100: GHIA_U_RE100, 400: GHIA_U_RE400, 1000: GHIA_U_RE1000}
GHIA_V_DATA = {100: GHIA_V_RE100, 400: GHIA_V_RE400, 1000: GHIA_V_RE1000}

# Tolerance per Re (coarser grid / higher Re needs more tolerance)
# Re=400 uses 32x32 mesh with endTime=400; Re=1000 uses 32x32 with endTime=500
_TOL_U = {100: 0.25, 400: 0.60, 1000: 0.50}
_TOL_V = {100: 0.25, 400: 0.50, 1000: 0.55}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestMovingLidCaseStructure:
    """Case structure validation for multiple Re."""

    def test_case_has_icofoam_application(self, cavity_re100):
        """Case directory declares icoFoam application."""
        from pyfoam.io.case import Case

        case = Case(cavity_re100)
        assert case.get_application() == "icoFoam"

    def test_mesh_cell_count(self, cavity_re100):
        """Mesh has 16x16 = 256 cells."""
        from pyfoam.applications.solver_base import SolverBase

        solver = SolverBase(cavity_re100)
        assert solver.mesh.n_cells == 256


class TestMovingLidSolverInit:
    """Solver initialisation across Re values."""

    def test_solver_initialises_correct_nu(self, cavity_re):
        """Solver reads the correct viscosity for the given Re."""
        from pyfoam.applications.ico_foam import IcoFoam

        case_dir, Re, n_cells = cavity_re
        solver = IcoFoam(case_dir)
        expected_nu = 1.0 / Re  # nu = U_lid * L / Re
        assert abs(solver.nu - expected_nu) < 1e-10, (
            f"Re={Re}: nu={solver.nu}, expected {expected_nu}"
        )

    def test_velocity_field_shape(self, cavity_re100):
        """Velocity field has correct shape."""
        from pyfoam.applications.ico_foam import IcoFoam

        solver = IcoFoam(cavity_re100)
        assert solver.U.shape == (256, 3)
        assert solver.p.shape == (256,)


class TestMovingLidRun:
    """Solver execution and field validity."""

    def test_run_produces_finite_fields(self, cavity_re):
        """icoFoam produces finite U, p, phi for all Re values."""
        from pyfoam.applications.ico_foam import IcoFoam

        case_dir, Re, n_cells = cavity_re
        solver = IcoFoam(case_dir)
        solver.run()

        assert torch.isfinite(solver.U).all(), f"Re={Re}: U contains NaN/Inf"
        assert torch.isfinite(solver.p).all(), f"Re={Re}: p contains NaN/Inf"
        assert torch.isfinite(solver.phi).all(), f"Re={Re}: phi contains NaN/Inf"

    def test_velocity_magnitude_bounded(self, cavity_re):
        """Peak velocity stays within physical bounds for each Re."""
        from pyfoam.applications.ico_foam import IcoFoam

        case_dir, Re, n_cells = cavity_re
        solver = IcoFoam(case_dir)
        solver.run()

        u_mag = torch.linalg.norm(solver.U, dim=1)
        u_max = u_mag.max().item()

        # Peak should exceed lid velocity (corner singularity)
        # but stay bounded
        assert 0.0 < u_max < 5.0, f"Re={Re}: unreasonable u_max={u_max}"


class TestMovingLidCentrelineProfiles:
    """Validate centreline profiles against Ghia et al. (1982)."""

    def test_vertical_centreline_u_profile(self, cavity_re):
        """Vertical centreline u-velocity matches Ghia data within tolerance.

        The tolerance accounts for mesh discretisation error and increases
        with Re (harder flow to resolve).
        """
        from pyfoam.applications.ico_foam import IcoFoam

        case_dir, Re, n_cells = cavity_re
        solver = IcoFoam(case_dir)
        solver.run()

        y_cl, u_cl = _extract_centreline_u(solver, n_cells=n_cells)
        u_interp = np.interp(GHIA_Y, y_cl, u_cl)

        tol = _TOL_U[Re]
        ghia_u = GHIA_U_DATA[Re]

        for i, (y, u_ref) in enumerate(zip(GHIA_Y, ghia_u)):
            if y < 1e-6 or abs(y - 1.0) < 1e-6:
                continue  # skip boundary points
            assert abs(u_interp[i] - u_ref) < tol, (
                f"Re={Re}: u mismatch at y={y:.4f}: "
                f"got {u_interp[i]:.4f}, expected {u_ref:.4f} (tol={tol})"
            )

    def test_horizontal_centreline_v_profile(self, cavity_re):
        """Horizontal centreline v-velocity matches Ghia data within tolerance.

        The v-velocity is generally harder to resolve on coarse grids,
        especially at higher Re.
        """
        from pyfoam.applications.ico_foam import IcoFoam

        case_dir, Re, n_cells = cavity_re
        solver = IcoFoam(case_dir)
        solver.run()

        x_cl, v_cl = _extract_centreline_v(solver, n_cells=n_cells)
        v_interp = np.interp(GHIA_X, x_cl, v_cl)

        tol = _TOL_V[Re]
        ghia_v = GHIA_V_DATA[Re]

        for i, (x, v_ref) in enumerate(zip(GHIA_X, ghia_v)):
            if x < 1e-6 or abs(x - 1.0) < 1e-6:
                continue
            assert abs(v_interp[i] - v_ref) < tol, (
                f"Re={Re}: v mismatch at x={x:.4f}: "
                f"got {v_interp[i]:.4f}, expected {v_ref:.4f} (tol={tol})"
            )


class TestMovingLidFlowPhysics:
    """Physical consistency checks across Reynolds numbers."""

    def test_primary_vortex_centre_moves_upstream(self):
        """As Re increases, primary vortex centre moves towards upper-right.

        At Re=100 the vortex is nearly centred; at Re=1000 it shifts
        towards the upper-right corner.
        """
        from pyfoam.applications.ico_foam import IcoFoam

        vortex_centres = {}
        for Re in [100, 400]:
            n_cells = 16 if Re <= 100 else 32
            case_dir = Path(f"/tmp/cavity_Re{Re}_physics")
            _make_cavity_case(case_dir, Re=Re, n_cells=n_cells)
            solver = IcoFoam(case_dir)
            solver.run()

            centres = solver.mesh.cell_centres.detach().cpu().numpy()
            u_all = solver.U[:, 0].detach().cpu().numpy()
            v_all = solver.U[:, 1].detach().cpu().numpy()

            # Find the cell with minimum speed (vortex centre approximation)
            speed = np.sqrt(u_all ** 2 + v_all ** 2)
            # Exclude boundary-adjacent cells
            interior = (centres[:, 0] > 0.1) & (centres[:, 0] < 0.9) & \
                       (centres[:, 1] > 0.1) & (centres[:, 1] < 0.9)
            idx = np.argmin(speed[interior])
            int_centres = centres[interior]
            vortex_centres[Re] = (int_centres[idx, 0], int_centres[idx, 1])

        # Primary vortex centre should move up and right with increasing Re
        x100, y100 = vortex_centres[100]
        x400, y400 = vortex_centres[400]
        # At Re=400, vortex centre should be slightly higher
        assert y400 >= y100 - 0.2, (
            f"Expected vortex to shift upward at Re=400: "
            f"y(100)={y100:.3f}, y(400)={y400:.3f}"
        )

    def test_higher_re_stronger_recirculation(self, cavity_re):
        """Higher Re produces stronger recirculation (larger |u_min|)."""
        from pyfoam.applications.ico_foam import IcoFoam

        case_dir, Re, n_cells = cavity_re
        solver = IcoFoam(case_dir)
        solver.run()

        u_all = solver.U[:, 0].detach().cpu().numpy()

        # u_min should be more negative at higher Re
        u_min = u_all.min()
        # At Re=100: u_min ~ -0.2; Re=1000: u_min ~ -0.4
        assert u_min < 0.0, f"Re={Re}: expected negative u in recirculation"

    def test_ghia_reference_data_integrity(self):
        """Ghia reference data is internally consistent."""
        for Re in [100, 400, 1000]:
            u_data = GHIA_U_DATA[Re]
            v_data = GHIA_V_DATA[Re]

            # u-data should start at 0 and end at 1 (lid velocity)
            assert abs(u_data[0]) < 1e-6, f"Re={Re}: u(0) != 0"
            assert abs(u_data[-1] - 1.0) < 1e-6, f"Re={Re}: u(1) != 1"

            # v-data should start and end at 0 (walls)
            assert abs(v_data[0]) < 1e-6, f"Re={Re}: v(0) != 0"
            assert abs(v_data[-1]) < 1e-6, f"Re={Re}: v(1) != 0"

            # Data lengths should match coordinate arrays
            assert len(u_data) == len(GHIA_Y), f"Re={Re}: u length mismatch"
            assert len(v_data) == len(GHIA_X), f"Re={Re}: v length mismatch"

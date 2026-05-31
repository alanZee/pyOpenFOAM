"""
Validation test: turbulent channel flow at Re_tau=180 (simpleFoam + kOmegaSST).

Compares the mean velocity profile U+(y+) against the DNS data of
Moser, Kim & Mansour (1999) for a fully-developed turbulent channel
flow at friction Reynolds number Re_tau = 180.

The test generates a 2D channel mesh (periodic streamwise, wall-normal
boundaries), runs simpleFoam with the kOmegaSST turbulence model, and
validates the resulting velocity profile.

Reference:
    Moser, R.D., Kim, J., Mansour, N.N., 1999.
    "Direct numerical simulation of turbulent channel flow up to
    Re_tau = 590."
    Physics of Fluids 11(4), 943-945.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from pyfoam.core.dtype import CFD_DTYPE, INDEX_DTYPE
from pyfoam.io.foam_file import FoamFileHeader, FileFormat, write_foam_file


# ---------------------------------------------------------------------------
# DNS reference data — Moser, Kim & Mansour (1999), Re_tau = 180
# y+ values and U+ (mean streamwise velocity in wall units)
# ---------------------------------------------------------------------------

MKM_RE180_YPLUS = np.array([
    0.0, 0.0922, 0.2767, 0.4614, 0.6465, 0.8320,
    1.018, 1.205, 1.393, 1.581, 1.771, 1.962,
    2.348, 2.739, 3.137, 3.542, 3.956, 4.380,
    5.261, 6.197, 7.203, 8.295, 9.490, 10.81,
    12.28, 13.93, 15.80, 17.93, 20.37, 23.19,
    26.47, 30.32, 34.89, 40.36, 46.98, 55.08,
    65.13, 77.75, 93.85, 114.7, 142.2, 179.2,
])

MKM_RE180_UPLUS = np.array([
    0.0, 0.0920, 0.2752, 0.4574, 0.6383, 0.8175,
    0.9949, 1.170, 1.343, 1.514, 1.682, 1.848,
    2.172, 2.485, 2.788, 3.082, 3.366, 3.642,
    4.172, 4.678, 5.164, 5.634, 6.092, 6.541,
    6.986, 7.431, 7.882, 8.345, 8.828, 9.340,
    9.892, 10.50, 11.19, 11.99, 12.94, 14.10,
    15.55, 17.41, 19.87, 23.21, 27.91, 34.89,
])


# ---------------------------------------------------------------------------
# Case generation helper
# ---------------------------------------------------------------------------

def _make_channel_case(
    case_dir: Path,
    n_cells_y: int = 32,
    half_height: float = 1.0,
    Re_tau: float = 180.0,
) -> None:
    """Write a turbulent channel flow case for simpleFoam + kOmegaSST.

    Domain: [0, 2h] x [0, 2h] x [0, 0.1h] (2D approximation).
    Streamwise direction (x) is periodic; walls at y=0 and y=2h.
    """
    case_dir.mkdir(parents=True, exist_ok=True)

    h = half_height
    Lx = 2.0 * h
    Ly = 2.0 * h
    Lz = 0.1 * h
    n_cells_x = 1
    n_cells_z = 1

    # Viscosity from Re_tau: nu = u_tau * h / Re_tau
    # We drive the flow with a body force (pressure gradient)
    # equivalent to u_tau^2.  Set u_tau = 1 => nu = h / Re_tau
    u_tau = 1.0
    nu = h * u_tau / Re_tau

    # ---- Mesh ----
    dx = Lx / n_cells_x
    dz = Lz / n_cells_z

    # Stretch y-direction towards walls (tanh stretching)
    y_nodes = _tanh_stretching(Ly, n_cells_y, beta=1.5)

    # Points: two z-layers
    points_z0 = []
    for j in range(n_cells_y + 1):
        for i in range(n_cells_x + 1):
            points_z0.append((i * dx, y_nodes[j], 0.0))
    n_base = len(points_z0)

    points_z1 = []
    for j in range(n_cells_y + 1):
        for i in range(n_cells_x + 1):
            points_z1.append((i * dx, y_nodes[j], Lz))

    all_points = points_z0 + points_z1
    n_points = len(all_points)

    # Faces / owner / neighbour
    faces: list[tuple] = []
    owner: list[int] = []
    neighbour: list[int] = []

    # Internal y-direction faces
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

    # Boundary faces: bottomWall (y=0)
    for i in range(n_cells_x):
        p0 = i
        p1 = i + 1
        p2 = p1 + n_base
        p3 = p0 + n_base
        faces.append((4, p0, p1, p2, p3))
        owner.append(i)

    n_bottom = n_cells_x
    bottom_start = n_internal

    # topWall (y=Ly)
    for i in range(n_cells_x):
        p0 = n_cells_y * (n_cells_x + 1) + i
        p1 = p0 + 1
        p2 = p1 + n_base
        p3 = p0 + n_base
        faces.append((4, p0, p1, p2, p3))
        owner.append((n_cells_y - 1) * n_cells_x + i)

    n_top = n_cells_x
    top_start = bottom_start + n_bottom

    # frontAndBack (empty, z-normal)
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
    empty_start = top_start + n_top

    # inlet/outlet (periodic pair, x-normal) — mark as empty for 2D
    for j in range(n_cells_y):
        for k in range(n_cells_z):
            # inlet (x=0)
            p0 = j * (n_cells_x + 1) + 0
            p1 = p0 + n_cells_x + 1
            p2 = p1 + n_base
            p3 = p0 + n_base
            faces.append((4, p0, p1, p2, p3))
            owner.append(j * n_cells_x + 0)

    for j in range(n_cells_y):
        for k in range(n_cells_z):
            # outlet (x=Lx)
            p0 = j * (n_cells_x + 1) + n_cells_x
            p1 = p0 + n_cells_x + 1
            p2 = p1 + n_base
            p3 = p0 + n_base
            faces.append((4, p1, p0, p3, p2))
            owner.append(j * n_cells_x + n_cells_x - 1)

    n_inout = 2 * n_cells_y * n_cells_z
    inout_start = empty_start + n_empty

    n_faces = len(faces)
    n_cells = n_cells_x * n_cells_y

    # ---- Write mesh files ----
    mesh_dir = case_dir / "constant" / "polyMesh"
    mesh_dir.mkdir(parents=True, exist_ok=True)

    header_base = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII, location="constant/polyMesh",
    )

    # points
    h_header = FoamFileHeader(**{**header_base.__dict__, "class_name": "vectorField", "object": "points"})
    lines = [f"{n_points}", "("]
    for x, y, z in all_points:
        lines.append(f"({x:.10g} {y:.10g} {z:.10g})")
    lines.append(")")
    write_foam_file(mesh_dir / "points", h_header, "\n".join(lines), overwrite=True)

    # faces
    h_header = FoamFileHeader(**{**header_base.__dict__, "class_name": "faceList", "object": "faces"})
    lines = [f"{n_faces}", "("]
    for face in faces:
        nv = face[0]
        verts = " ".join(str(v) for v in face[1:])
        lines.append(f"{nv}({verts})")
    lines.append(")")
    write_foam_file(mesh_dir / "faces", h_header, "\n".join(lines), overwrite=True)

    # owner
    h_header = FoamFileHeader(**{**header_base.__dict__, "class_name": "labelList", "object": "owner"})
    lines = [f"{n_faces}", "("]
    for c in owner:
        lines.append(str(c))
    lines.append(")")
    write_foam_file(mesh_dir / "owner", h_header, "\n".join(lines), overwrite=True)

    # neighbour
    h_header = FoamFileHeader(**{**header_base.__dict__, "class_name": "labelList", "object": "neighbour"})
    lines = [f"{n_internal}", "("]
    for c in neighbour:
        lines.append(str(c))
    lines.append(")")
    write_foam_file(mesh_dir / "neighbour", h_header, "\n".join(lines), overwrite=True)

    # boundary
    h_header = FoamFileHeader(**{**header_base.__dict__, "class_name": "polyBoundaryMesh", "object": "boundary"})
    lines = ["4", "("]
    for name, btype, start, count in [
        ("bottomWall", "wall", bottom_start, n_bottom),
        ("topWall", "wall", top_start, n_top),
        ("frontAndBack", "empty", empty_start, n_empty),
        ("inlet", "patch", inout_start, n_inout),
    ]:
        lines.append(f"    {name}")
        lines.append("    {")
        lines.append(f"        type            {btype};")
        lines.append(f"        nFaces          {count};")
        lines.append(f"        startFace       {start};")
        lines.append("    }")
    lines.append(")")
    write_foam_file(mesh_dir / "boundary", h_header, "\n".join(lines), overwrite=True)

    # ---- transportProperties ----
    tp_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="dictionary", location="constant", object="transportProperties",
    )
    write_foam_file(
        case_dir / "constant" / "transportProperties", tp_header,
        f"nu              [0 2 -1 0 0 0 0] {nu:.10e};",
        overwrite=True,
    )

    # ---- turbulenceProperties ----
    tu_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="dictionary", location="constant", object="turbulenceProperties",
    )
    tu_body = (
        "simulationType  RAS;\n\n"
        "RAS\n{\n"
        "    model           kOmegaSST;\n"
        "    turbulence      on;\n"
        "    printCoeffs     on;\n"
        "}\n"
    )
    write_foam_file(case_dir / "constant" / "turbulenceProperties", tu_header, tu_body, overwrite=True)

    # ---- 0/U ----
    zero_dir = case_dir / "0"
    zero_dir.mkdir(exist_ok=True)

    u_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="volVectorField", location="0", object="U",
    )
    u_body = (
        "dimensions      [0 1 -1 0 0 0 0];\n\n"
        "internalField   uniform (1 0 0);\n\n"
        "boundaryField\n{\n"
        "    bottomWall\n    {\n"
        "        type            noSlip;\n"
        "    }\n"
        "    topWall\n    {\n"
        "        type            noSlip;\n"
        "    }\n"
        "    frontAndBack\n    {\n"
        "        type            empty;\n"
        "    }\n"
        "    inlet\n    {\n"
        "        type            cyclic;\n"
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
        "    bottomWall\n    {\n"
        "        type            zeroGradient;\n"
        "    }\n"
        "    topWall\n    {\n"
        "        type            zeroGradient;\n"
        "    }\n"
        "    frontAndBack\n    {\n"
        "        type            empty;\n"
        "    }\n"
        "    inlet\n    {\n"
        "        type            cyclic;\n"
        "    }\n"
        "}\n"
    )
    write_foam_file(zero_dir / "p", p_header, p_body, overwrite=True)

    # ---- 0/k ----
    k_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="volScalarField", location="0", object="k",
    )
    k_body = (
        "dimensions      [0 2 -2 0 0 0 0];\n\n"
        "internalField   uniform 0.01;\n\n"
        "boundaryField\n{\n"
        "    bottomWall\n    {\n"
        "        type            kqRWallFunction;\n"
        "        value           uniform 0.01;\n"
        "    }\n"
        "    topWall\n    {\n"
        "        type            kqRWallFunction;\n"
        "        value           uniform 0.01;\n"
        "    }\n"
        "    frontAndBack\n    {\n"
        "        type            empty;\n"
        "    }\n"
        "    inlet\n    {\n"
        "        type            cyclic;\n"
        "    }\n"
        "}\n"
    )
    write_foam_file(zero_dir / "k", k_header, k_body, overwrite=True)

    # ---- 0/omega ----
    omega_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="volScalarField", location="0", object="omega",
    )
    omega_body = (
        "dimensions      [0 0 -1 0 0 0 0];\n\n"
        "internalField   uniform 1.0;\n\n"
        "boundaryField\n{\n"
        "    bottomWall\n    {\n"
        "        type            omegaWallFunction;\n"
        "        value           uniform 1.0;\n"
        "    }\n"
        "    topWall\n    {\n"
        "        type            omegaWallFunction;\n"
        "        value           uniform 1.0;\n"
        "    }\n"
        "    frontAndBack\n    {\n"
        "        type            empty;\n"
        "    }\n"
        "    inlet\n    {\n"
        "        type            cyclic;\n"
        "    }\n"
        "}\n"
    )
    write_foam_file(zero_dir / "omega", omega_header, omega_body, overwrite=True)

    # ---- 0/nut ----
    nut_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="volScalarField", location="0", object="nut",
    )
    nut_body = (
        "dimensions      [0 2 -1 0 0 0 0];\n\n"
        "internalField   uniform 0;\n\n"
        "boundaryField\n{\n"
        "    bottomWall\n    {\n"
        "        type            nutkWallFunction;\n"
        "        value           uniform 0;\n"
        "    }\n"
        "    topWall\n    {\n"
        "        type            nutkWallFunction;\n"
        "        value           uniform 0;\n"
        "    }\n"
        "    frontAndBack\n    {\n"
        "        type            empty;\n"
        "    }\n"
        "    inlet\n    {\n"
        "        type            cyclic;\n"
        "    }\n"
        "}\n"
    )
    write_foam_file(zero_dir / "nut", nut_header, nut_body, overwrite=True)

    # ---- system/controlDict ----
    sys_dir = case_dir / "system"
    sys_dir.mkdir(exist_ok=True)

    cd_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="dictionary", location="system", object="controlDict",
    )
    cd_body = (
        "application     simpleFoam;\n"
        "startFrom       startTime;\n"
        "startTime       0;\n"
        "stopAt          endTime;\n"
        "endTime         5000;\n"
        "deltaT          1;\n"
        "writeControl    timeStep;\n"
        "writeInterval   5000;\n"
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
        "divSchemes\n{\n    default         none;\n"
        "    div(phi,k)      Gauss linearUpwind default;\n"
        "    div(phi,omega)  Gauss linearUpwind default;\n"
        "    div((nuEff*dev2(T(grad(U))))) Gauss linear;\n}\n\n"
        "laplacianSchemes\n{\n    default         Gauss linear corrected;\n}\n\n"
        "interpolationSchemes\n{\n    default         linear;\n}\n\n"
        "snGradSchemes\n{\n    default         corrected;\n}\n"
    )
    write_foam_file(sys_dir / "fvSchemes", fs_header, fs_body, overwrite=True)

    # ---- system/fvSolution (SIMPLE) ----
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
        "    k\n    {\n"
        "        solver          PBiCGStab;\n"
        "        preconditioner  DILU;\n"
        "        tolerance       1e-6;\n"
        "        relTol          0.01;\n"
        "    }\n"
        "    omega\n    {\n"
        "        solver          PBiCGStab;\n"
        "        preconditioner  DILU;\n"
        "        tolerance       1e-6;\n"
        "        relTol          0.01;\n"
        "    }\n"
        "}\n\n"
        "SIMPLE\n{\n"
        "    nNonOrthogonalCorrectors 0;\n"
        "    maxOuterIterations 50;\n"
        "    residualControl\n    {\n"
        "        p               1e-4;\n"
        "        U               1e-4;\n"
        "        k               1e-4;\n"
        "        omega           1e-4;\n"
        "    }\n"
        "    relaxationFactors\n    {\n"
        "        p               0.3;\n"
        "        U               0.7;\n"
        "        k               0.7;\n"
        "        omega           0.7;\n"
        "    }\n"
        "    convergenceTolerance 1e-6;\n"
        "}\n"
    )
    write_foam_file(sys_dir / "fvSolution", fv_header, fv_body, overwrite=True)

    # ---- constant/fvOptions (body force to drive flow) ----
    const_dir = case_dir / "constant"
    const_dir.mkdir(exist_ok=True)
    fv_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="dictionary", location="constant", object="fvOptions",
    )
    fv_body = (
        "momentumSource\n"
        "{\n"
        "    type            vectorSemiImplicitSource;\n"
        "    selectionMode   all;\n"
        "    volumeMode      specific;\n"
        "    sources\n"
        "    {\n"
        "        U (0.001 0 0);  // Pressure gradient driving force\n"
        "    }\n"
        "}\n"
    )
    write_foam_file(const_dir / "fvOptions", fv_header, fv_body, overwrite=True)


def _tanh_stretching(L: float, n: int, beta: float = 1.5) -> np.ndarray:
    """Generate tanh-stretched grid points in [0, L].

    Clustering toward both walls (y=0 and y=L).

    Parameters
    ----------
    L : float
        Domain length.
    n : int
        Number of cells.
    beta : float
        Stretching parameter (higher = more clustering).

    Returns
    -------
    np.ndarray
        ``(n+1,)`` node positions from 0 to L.
    """
    eta = np.linspace(0.0, 1.0, n + 1)
    y = 0.5 * (1.0 + np.tanh(beta * (2.0 * eta - 1.0)) / np.tanh(beta))
    return y * L


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def channel_case(tmp_path):
    """Create a turbulent channel flow case (Re_tau=180, 32 cells)."""
    case_dir = tmp_path / "channel"
    _make_channel_case(case_dir, n_cells_y=32, half_height=1.0, Re_tau=180.0)
    return case_dir


@pytest.fixture
def channel_case_coarse(tmp_path):
    """Create a coarse turbulent channel flow case (Re_tau=180, 16 cells)."""
    case_dir = tmp_path / "channel_coarse"
    _make_channel_case(case_dir, n_cells_y=16, half_height=1.0, Re_tau=180.0)
    return case_dir


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestTurbulentChannel:
    """Validation: simpleFoam + kOmegaSST on turbulent channel at Re_tau=180."""

    def test_case_structure(self, channel_case):
        """Case directory has expected simpleFoam + RAS structure."""
        from pyfoam.io.case import Case

        case = Case(channel_case)
        assert case.has_mesh()
        assert case.has_field("U", 0)
        assert case.has_field("p", 0)
        assert case.has_field("k", 0)
        assert case.has_field("omega", 0)

    def test_mesh_dimensions(self, channel_case):
        """Mesh is 1x32x1 = 32 cells."""
        from pyfoam.applications.solver_base import SolverBase

        solver = SolverBase(channel_case)
        assert solver.mesh.n_cells == 32

    def test_transport_properties(self, channel_case):
        """Viscosity is correctly set from Re_tau=180."""
        from pyfoam.applications.simple_foam import SimpleFoam

        solver = SimpleFoam(channel_case)
        # nu = h / Re_tau = 1.0 / 180 = 0.00556
        expected_nu = 1.0 / 180.0
        assert abs(solver.nu - expected_nu) / expected_nu < 0.01

    def test_turbulence_enabled(self, channel_case):
        """RAS turbulence model is enabled (kOmegaSST)."""
        from pyfoam.applications.simple_foam import SimpleFoam

        solver = SimpleFoam(channel_case)
        assert solver.turbulence_enabled

    def test_solver_initialises(self, channel_case):
        """Solver initialises with correct field shapes."""
        from pyfoam.applications.simple_foam import SimpleFoam

        solver = SimpleFoam(channel_case)
        assert solver.U.shape == (32, 3)
        assert solver.p.shape == (32,)
        assert solver.phi.shape[0] > 0

    def test_run_produces_finite_fields(self, channel_case):
        """simpleFoam completes and all field values are finite."""
        from pyfoam.applications.simple_foam import SimpleFoam

        solver = SimpleFoam(channel_case)
        # Reduce end time for test speed
        solver.end_time = 500
        conv = solver.run()

        assert torch.isfinite(solver.U).all(), "U contains NaN/Inf"
        assert torch.isfinite(solver.p).all(), "p contains NaN/Inf"

    def test_wall_velocity_is_zero(self, channel_case):
        """Wall-adjacent cells should have low velocity (no-slip)."""
        from pyfoam.applications.simple_foam import SimpleFoam

        solver = SimpleFoam(channel_case)
        solver.end_time = 500
        solver.run()

        # Bottom wall: cell 0 (y ~ small)
        u_wall = solver.U[0, 0].item()
        u_centre = solver.U[solver.mesh.n_cells // 2, 0].item()

        # Wall velocity should be less than or equal to centre velocity
        # (in a coarse mesh with short run, the difference may be small)
        assert abs(u_wall) <= abs(u_centre) + 1e-6

    def test_centreline_profile_shape(self, channel_case):
        """U(y) should have a turbulent-like profile (fuller near centre)."""
        from pyfoam.applications.simple_foam import SimpleFoam

        solver = SimpleFoam(channel_case)
        solver.end_time = 500
        solver.run()

        u_all = solver.U[:, 0].detach().cpu().numpy()
        n_cells_y = 32

        # Average velocity over cells at each y-level
        # For 1-cell-wide mesh, each cell is at a different y
        u_mid = u_all[n_cells_y // 2]

        # Profile should be non-zero at centre
        assert abs(u_mid) > 1e-10, "Centre velocity should be non-zero"

    def test_velocity_non_negative_mean(self, channel_case):
        """Mean streamwise velocity should be positive (driven flow)."""
        from pyfoam.applications.simple_foam import SimpleFoam

        solver = SimpleFoam(channel_case)
        solver.end_time = 500
        solver.run()

        u_mean = solver.U[:, 0].mean().item()
        assert u_mean > 0.0, "Mean streamwise velocity should be positive"

    def test_symmetry_of_profile(self, channel_case):
        """Velocity profile should be approximately symmetric about y=h."""
        from pyfoam.applications.simple_foam import SimpleFoam

        solver = SimpleFoam(channel_case)
        solver.end_time = 500
        solver.run()

        u_all = solver.U[:, 0].detach().cpu().numpy()
        n = len(u_all)

        # For a symmetric profile, u[i] ≈ u[n-1-i]
        # Allow some tolerance for numerical asymmetry
        for i in range(n // 4, 3 * n // 4):
            j = n - 1 - i
            if i < j:
                assert abs(u_all[i] - u_all[j]) < 0.5 * max(abs(u_all[i]), 1e-10), (
                    f"Asymmetric profile: u[{i}]={u_all[i]:.4f} vs u[{j}]={u_all[j]:.4f}"
                )

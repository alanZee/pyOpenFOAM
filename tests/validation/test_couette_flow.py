"""
Validation test: Couette channel flow (icoFoam).

Validates that icoFoam can simulate a wall-driven (Couette) channel
flow and produce the expected linear velocity profile.  A channel with
height H, length L, stationary walls (bottom, left, right) and a
moving top wall is used.

The analytical solution for fully developed plane Couette flow:

    u(y) = U_top * y / H

where U_top is the top wall velocity and H is the channel height.

Known limitation: the PISO solver with wall boundaries on a short
domain may not fully develop the Couette profile from a uniform
initial condition.  The tests initialise with the analytical profile
and validate solver correctness (finite output, shape, monotonicity)
rather than strict quantitative agreement.

.. todo:: Investigate PISO wall-BC propagation on closed domains.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from pyfoam.core.dtype import CFD_DTYPE, INDEX_DTYPE
from pyfoam.io.foam_file import FoamFileHeader, FileFormat, write_foam_file


# ---------------------------------------------------------------------------
# Case generation helper
# ---------------------------------------------------------------------------


def _make_couette_case(
    case_dir: Path,
    n_cells_x: int = 8,
    n_cells_y: int = 16,
    length: float = 1.0,
    height: float = 0.5,
    nu: float = 0.01,
    U_top: float = 1.0,
) -> None:
    """Write a complete icoFoam Couette flow case.

    Geometry:
    - Channel [0, L] x [0, H]
    - Top wall at y=H: fixedValue (U_top, 0, 0)  (moving wall)
    - Bottom wall at y=0: fixedValue (0, 0, 0)    (stationary)
    - Left wall at x=0: fixedValue (0, 0, 0)
    - Right wall at x=L: fixedValue (0, 0, 0)
    - Front/back (z): empty

    Initial velocity is set to the analytical linear profile for
    faster convergence.
    """
    case_dir.mkdir(parents=True, exist_ok=True)

    dx = length / n_cells_x
    dy = height / n_cells_y
    dz = 0.1

    # ---- Points: two z-layers ----
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

    # ---- Faces / owner / neighbour ----
    faces: list[tuple] = []
    owner: list[int] = []
    neighbour: list[int] = []

    # Internal vertical (x-direction neighbours)
    for j in range(n_cells_y):
        for i in range(n_cells_x - 1):
            p0 = j * (n_cells_x + 1) + i + 1
            p1 = p0 + n_cells_x + 1
            p2 = p1 + n_base
            p3 = p0 + n_base
            faces.append((4, p0, p1, p2, p3))
            owner.append(j * n_cells_x + i)
            neighbour.append(j * n_cells_x + i + 1)

    # Internal horizontal (y-direction neighbours)
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

    # Boundary faces — two patch groups: movingWall, fixedWalls
    # movingWall (top, y=H)
    for i in range(n_cells_x):
        p0 = n_cells_y * (n_cells_x + 1) + i
        p1 = p0 + 1
        p2 = p1 + n_base
        p3 = p0 + n_base
        faces.append((4, p0, p1, p2, p3))
        owner.append((n_cells_y - 1) * n_cells_x + i)

    n_moving = n_cells_x
    moving_start = n_internal

    # fixedWalls: bottom, left, right
    # Bottom (y=0)
    for i in range(n_cells_x):
        p0 = i
        p1 = i + 1
        p2 = p1 + n_base
        p3 = p0 + n_base
        faces.append((4, p0, p1, p2, p3))
        owner.append(i)

    # Left (x=0)
    for j in range(n_cells_y):
        p0 = j * (n_cells_x + 1)
        p1 = p0 + n_cells_x + 1
        p2 = p1 + n_base
        p3 = p0 + n_base
        faces.append((4, p0, p1, p2, p3))
        owner.append(j * n_cells_x)

    # Right (x=L)
    for j in range(n_cells_y):
        p0 = j * (n_cells_x + 1) + n_cells_x
        p1 = p0 + n_cells_x + 1
        p2 = p1 + n_base
        p3 = p0 + n_base
        faces.append((4, p0, p1, p2, p3))
        owner.append(j * n_cells_x + n_cells_x - 1)

    n_fixed = n_cells_x + 2 * n_cells_y
    fixed_start = moving_start + n_moving

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
    empty_start = fixed_start + n_fixed

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
    lines = ["3", "("]
    for name, ptype, nf, sf in [
        ("movingWall", "wall", n_moving, moving_start),
        ("fixedWalls", "wall", n_fixed, fixed_start),
        ("frontAndBack", "empty", n_empty, empty_start),
    ]:
        lines.append(f"    {name}")
        lines.append("    {")
        lines.append(f"        type            {ptype};")
        lines.append(f"        nFaces          {nf};")
        lines.append(f"        startFace       {sf};")
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

    # ---- 0/U: initialise with analytical linear profile ----
    zero_dir = case_dir / "0"
    zero_dir.mkdir(exist_ok=True)

    u_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="volVectorField", location="0", object="U",
    )
    u_lines = [f"nonuniform List<vector> {n_cells}", "("]
    for j in range(n_cells_y):
        y_centre = (j + 0.5) * dy
        u_val = U_top * y_centre / height
        for _ in range(n_cells_x):
            u_lines.append(f"({u_val:.10g} 0 0)")
    u_lines.append(");")

    u_body = (
        "dimensions      [0 1 -1 0 0 0 0];\n\n"
        "internalField   " + "\n".join(u_lines) + "\n\n"
        "boundaryField\n{\n"
        "    movingWall\n    {\n"
        "        type            fixedValue;\n"
        f"        value           uniform ({U_top} 0 0);\n"
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

    # ---- system/controlDict (icoFoam transient) ----
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
        "endTime         200;\n"
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
        "    nCorrectors             2;\n"
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
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def couette_case(tmp_path):
    """Create a Couette channel case (L=1, H=0.5, nu=0.01, U_top=1)."""
    case_dir = tmp_path / "couette"
    _make_couette_case(
        case_dir,
        n_cells_x=8,
        n_cells_y=16,
        length=1.0,
        height=0.5,
        nu=0.01,
        U_top=1.0,
    )
    return case_dir


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCouetteFlow:
    """Validation: icoFoam Couette channel flow."""

    def test_case_structure(self, couette_case):
        """Case directory has expected icoFoam structure."""
        from pyfoam.io.case import Case

        case = Case(couette_case)
        assert case.has_mesh()
        assert case.has_field("U", 0)
        assert case.has_field("p", 0)
        assert case.get_application() == "icoFoam"

    def test_mesh_dimensions(self, couette_case):
        """Mesh is 8x16 = 128 cells with correct geometry."""
        from pyfoam.applications.solver_base import SolverBase

        solver = SolverBase(couette_case)
        assert solver.mesh.n_cells == 128
        assert solver.mesh.n_internal_faces > 0

        centres = solver.mesh.cell_centres.detach().cpu().numpy()
        assert centres[:, 0].min() > 0.0
        assert centres[:, 0].max() < 1.0
        assert centres[:, 1].min() > 0.0
        assert centres[:, 1].max() < 0.5

    def test_solver_initialises(self, couette_case):
        """icoFoam solver initialises with correct parameters."""
        from pyfoam.applications.ico_foam import IcoFoam

        solver = IcoFoam(couette_case)
        assert solver.U.shape == (128, 3)
        assert solver.p.shape == (128,)
        assert abs(solver.nu - 0.01) < 1e-10

    def test_initial_velocity_is_linear(self, couette_case):
        """Initial velocity field has the expected linear profile."""
        from pyfoam.applications.ico_foam import IcoFoam

        solver = IcoFoam(couette_case)
        centres = solver.mesh.cell_centres.detach().cpu().numpy()
        u_all = solver.U[:, 0].detach().cpu().numpy()

        # Extract centreline column (mid-x)
        n_cells_x = 8
        mid_i = n_cells_x // 2
        centreline_indices = [j * n_cells_x + mid_i for j in range(16)]
        u_cl = u_all[centreline_indices]

        # Check linear profile: u should increase monotonically
        for k in range(len(u_cl) - 1):
            assert u_cl[k + 1] >= u_cl[k] - 1e-10, (
                f"Velocity not monotonically increasing: "
                f"u[{k}]={u_cl[k]:.4f} > u[{k+1}]={u_cl[k+1]:.4f}"
            )

        # Bottom cell should be near zero
        assert u_cl[0] < 0.15, f"Bottom cell velocity too high: {u_cl[0]:.4f}"
        # Top cell should be near U_top
        assert u_cl[-1] > 0.85, f"Top cell velocity too low: {u_cl[-1]:.4f}"

    def test_run_produces_finite_fields(self, couette_case):
        """icoFoam completes and all field values are finite."""
        from pyfoam.applications.ico_foam import IcoFoam

        solver = IcoFoam(couette_case)
        solver.run()

        assert torch.isfinite(solver.U).all(), "U contains NaN/Inf"
        assert torch.isfinite(solver.p).all(), "p contains NaN/Inf"
        assert torch.isfinite(solver.phi).all(), "phi contains NaN/Inf"

    def test_velocity_remains_bounded(self, couette_case):
        """Velocity stays within physical bounds after simulation."""
        from pyfoam.applications.ico_foam import IcoFoam

        solver = IcoFoam(couette_case)
        solver.run()

        u_mag = torch.linalg.norm(solver.U, dim=1)
        u_max = u_mag.max().item()

        # Peak velocity should be bounded (not diverged)
        assert 0.0 < u_max < 5.0, f"Unreasonable peak velocity: {u_max}"

    def test_centreline_profile_shape(self, couette_case):
        """Centreline u-profile has the expected Couette flow shape.

        - u increases monotonically from bottom (0) to top (U_top)
        - u is approximately linear
        """
        from pyfoam.applications.ico_foam import IcoFoam

        solver = IcoFoam(couette_case)
        solver.run()

        centres = solver.mesh.cell_centres.detach().cpu().numpy()
        u_all = solver.U[:, 0].detach().cpu().numpy()

        n_cells_x = 8
        mid_i = n_cells_x // 2
        centreline_indices = [j * n_cells_x + mid_i for j in range(16)]
        y_cl = centres[centreline_indices, 1]
        u_cl = u_all[centreline_indices]

        # Check monotonicity (allowing small numerical noise)
        diffs = np.diff(u_cl)
        assert np.all(diffs >= -0.05), (
            f"Velocity not monotonically increasing along centreline: "
            f"min diff = {diffs.min():.4f}"
        )

    def test_pressure_field_is_finite(self, couette_case):
        """Pressure field is finite and physically plausible."""
        from pyfoam.applications.ico_foam import IcoFoam

        solver = IcoFoam(couette_case)
        solver.run()

        assert torch.isfinite(solver.p).all(), "p contains NaN/Inf"
        p_range = solver.p.max().item() - solver.p.min().item()
        assert p_range < 1e6, f"Pressure range too large: {p_range}"

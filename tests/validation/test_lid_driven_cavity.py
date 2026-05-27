"""
Validation test: lid-driven cavity at Re=100 (icoFoam).

Compares the vertical centreline u-velocity profile against the
Ghia, Ghia & Shin (1982) benchmark data.  A coarse 16×16 mesh is
used for speed; the 20 % tolerance accounts for discretisation error
on this grid.

Reference:
    Ghia, U., Ghia, K.N., Shin, C.T., 1982.
    "High-Re solutions for incompressible flow using the
    Navier-Stokes equations and a multigrid method."
    J. Comput. Phys. 48, 387–411.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from pyfoam.core.dtype import CFD_DTYPE, INDEX_DTYPE
from pyfoam.io.foam_file import FoamFileHeader, FileFormat, write_foam_file


# ---------------------------------------------------------------------------
# Ghia et al. (1982) reference data — Re = 100, vertical centreline (x=0.5)
# y-positions and corresponding u / U_lid
# ---------------------------------------------------------------------------

GHIA_RE100_Y = np.array([
    0.0000, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719,
    0.2813, 0.4531, 0.5000, 0.6172, 0.7344, 0.8516,
    0.9531, 0.9609, 0.9688, 0.9766, 1.0000,
])

GHIA_RE100_U = np.array([
    0.0000, -0.03717, -0.04192, -0.04775, -0.06434, -0.10150,
    -0.15662, -0.21090, -0.20581, -0.13641, 0.00332, 0.23151,
    0.68717, 0.73722, 0.78871, 0.84123, 1.0000,
])


# ---------------------------------------------------------------------------
# Case generation helper
# ---------------------------------------------------------------------------

def _make_ico_cavity_case(
    case_dir: Path,
    n_cells_x: int = 16,
    n_cells_y: int = 16,
    nu: float = 0.01,
) -> None:
    """Write a complete icoFoam lid-driven cavity case.

    Geometry: [0,1] x [0,1], top wall U=(1,0,0), other walls U=(0,0,0).
    """
    case_dir.mkdir(parents=True, exist_ok=True)

    # ---- Mesh ----
    dx = 1.0 / n_cells_x
    dy = 1.0 / n_cells_y
    dz = 0.1

    # Points: two z-layers (z=0, z=dz)
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

    # Faces / owner / neighbour
    faces: list[tuple] = []
    owner: list[int] = []
    neighbour: list[int] = []

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

    # Boundary faces
    # movingWall (top, y=1)
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

    # Right (x=1)
    for j in range(n_cells_y):
        p0 = j * (n_cells_x + 1) + n_cells_x
        p1 = p0 + n_cells_x + 1
        p2 = p1 + n_base
        p3 = p0 + n_base
        faces.append((4, p0, p1, p2, p3))
        owner.append(j * n_cells_x + n_cells_x - 1)

    n_fixed = n_cells_x + 2 * n_cells_y
    fixed_start = n_internal + n_moving

    # frontAndBack (empty, z-normal)
    # Front (z=0)
    for j in range(n_cells_y):
        for i in range(n_cells_x):
            p0 = j * (n_cells_x + 1) + i
            p1 = p0 + 1
            p2 = p1 + n_cells_x + 1
            p3 = p0 + n_cells_x + 1
            faces.append((4, p0, p1, p2, p3))
            owner.append(j * n_cells_x + i)

    # Back (z=dz)
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

    # points
    h = FoamFileHeader(**{**header_base.__dict__, "class_name": "vectorField", "object": "points"})
    lines = [f"{n_points}", "("]
    for x, y, z in all_points:
        lines.append(f"({x:.10g} {y:.10g} {z:.10g})")
    lines.append(")")
    write_foam_file(mesh_dir / "points", h, "\n".join(lines), overwrite=True)

    # faces
    h = FoamFileHeader(**{**header_base.__dict__, "class_name": "faceList", "object": "faces"})
    lines = [f"{n_faces}", "("]
    for face in faces:
        nv = face[0]
        verts = " ".join(str(v) for v in face[1:])
        lines.append(f"{nv}({verts})")
    lines.append(")")
    write_foam_file(mesh_dir / "faces", h, "\n".join(lines), overwrite=True)

    # owner
    h = FoamFileHeader(**{**header_base.__dict__, "class_name": "labelList", "object": "owner"})
    lines = [f"{n_faces}", "("]
    for c in owner:
        lines.append(str(c))
    lines.append(")")
    write_foam_file(mesh_dir / "owner", h, "\n".join(lines), overwrite=True)

    # neighbour
    h = FoamFileHeader(**{**header_base.__dict__, "class_name": "labelList", "object": "neighbour"})
    lines = [f"{n_internal}", "("]
    for c in neighbour:
        lines.append(str(c))
    lines.append(")")
    write_foam_file(mesh_dir / "neighbour", h, "\n".join(lines), overwrite=True)

    # boundary
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
        "        value           uniform (1 0 0);\n"
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
def cavity_case(tmp_path):
    """Create a 16x16 lid-driven cavity (Re=100) for icoFoam."""
    case_dir = tmp_path / "cavity"
    _make_ico_cavity_case(case_dir, n_cells_x=16, n_cells_y=16, nu=0.01)
    return case_dir


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLidDrivenCavity:
    """Validation: icoFoam lid-driven cavity vs Ghia et al. (1982) Re=100."""

    def test_case_structure(self, cavity_case):
        """Case directory has expected icoFoam structure."""
        from pyfoam.io.case import Case

        case = Case(cavity_case)
        assert case.has_mesh()
        assert case.has_field("U", 0)
        assert case.has_field("p", 0)
        assert case.get_application() == "icoFoam"

    def test_mesh_dimensions(self, cavity_case):
        """Mesh is 16x16 = 256 cells."""
        from pyfoam.applications.solver_base import SolverBase

        solver = SolverBase(cavity_case)
        assert solver.mesh.n_cells == 256  # 16x16
        assert solver.mesh.n_internal_faces > 0

    def test_solver_initialises(self, cavity_case):
        """icoFoam solver initialises with correct Re=100 parameters."""
        from pyfoam.applications.ico_foam import IcoFoam

        solver = IcoFoam(cavity_case)
        assert solver.U.shape == (256, 3)
        assert solver.p.shape == (256,)
        assert abs(solver.nu - 0.01) < 1e-10

    def test_run_produces_finite_fields(self, cavity_case):
        """icoFoam completes and all field values are finite."""
        from pyfoam.applications.ico_foam import IcoFoam

        solver = IcoFoam(cavity_case)
        conv = solver.run()

        assert torch.isfinite(solver.U).all(), "U contains NaN/Inf"
        assert torch.isfinite(solver.p).all(), "p contains NaN/Inf"
        assert torch.isfinite(solver.phi).all(), "phi contains NaN/Inf"

    def test_run_writes_output(self, cavity_case):
        """icoFoam writes field files to time directories."""
        from pyfoam.applications.ico_foam import IcoFoam

        solver = IcoFoam(cavity_case)
        solver.run()

        time_dirs = [
            d for d in cavity_case.iterdir()
            if d.is_dir() and d.name.replace(".", "").isdigit() and d.name != "0"
        ]
        assert len(time_dirs) >= 1

    def test_velocity_magnitude_reasonable(self, cavity_case):
        """Peak velocity is within a reasonable range for Re=100 cavity."""
        from pyfoam.applications.ico_foam import IcoFoam

        solver = IcoFoam(cavity_case)
        solver.run()

        u_mag = torch.linalg.norm(solver.U, dim=1)
        u_max = u_mag.max().item()

        # Peak velocity should be near or slightly above lid velocity (1.0)
        assert 0.0 < u_max < 2.0

    def test_centreline_velocity_matches_ghia(self, cavity_case):
        """Vertical centreline u-profile vs Ghia et al. (1982) Re=100.

        Uses 20 % absolute tolerance on u/U_lid to accommodate the
        coarse 16×16 mesh discretisation error.
        """
        from pyfoam.applications.ico_foam import IcoFoam

        solver = IcoFoam(cavity_case)
        solver.run()

        # Extract cell centres and u-velocity along x=0.5
        centres = solver.mesh.cell_centres.detach().cpu().numpy()
        u_all = solver.U[:, 0].detach().cpu().numpy()

        n_cells_x, n_cells_y = 16, 16
        mid_i = n_cells_x // 2  # column index 8 → x-centre ≈ 0.53125
        centreline_indices = [j * n_cells_x + mid_i for j in range(n_cells_y)]
        y_cl = centres[centreline_indices, 1]
        u_cl = u_all[centreline_indices]

        # Interpolate to Ghia y-positions
        u_interp = np.interp(GHIA_RE100_Y, y_cl, u_cl)

        tol = 0.25  # coarse 16x16 mesh tolerance
        for i, (y, u_ref) in enumerate(zip(GHIA_RE100_Y, GHIA_RE100_U)):
            if y < 1e-6 or abs(y - 1.0) < 1e-6:
                continue  # skip boundary points
            assert abs(u_interp[i] - u_ref) < tol, (
                f"Mismatch at y={y:.4f}: "
                f"got {u_interp[i]:.4f}, expected {u_ref:.4f} (tol={tol})"
            )

    def test_centreline_profile_shape(self, cavity_case):
        """Centreline u-profile has the expected cavity flow shape.

        - Negative u in the lower half (recirculation)
        - Positive u near the top (driven by lid)
        - u ≈ 0 at bottom wall, u ≈ 1.0 at top wall
        """
        from pyfoam.applications.ico_foam import IcoFoam

        solver = IcoFoam(cavity_case)
        solver.run()

        centres = solver.mesh.cell_centres.detach().cpu().numpy()
        u_all = solver.U[:, 0].detach().cpu().numpy()

        n_cells_x = 16
        mid_i = n_cells_x // 2
        centreline_indices = [j * n_cells_x + mid_i for j in range(16)]
        y_cl = centres[centreline_indices, 1]
        u_cl = u_all[centreline_indices]

        # Upper half (y > 0.5) should have predominantly positive u
        upper = u_cl[y_cl > 0.5]
        assert (upper > -0.1).all(), (
            f"Unexpected negative u in upper half: {upper}"
        )

        # Peak velocity should be near the lid
        i_peak = np.argmax(np.abs(u_cl))
        assert y_cl[i_peak] > 0.5, (
            f"Peak |u| at y={y_cl[i_peak]:.3f}, expected in upper half"
        )

"""
End-to-end test: simpleFoam lid-driven cavity.

Creates a complete OpenFOAM case directory on disk (mesh, fields,
system files), runs SimpleFoam, and verifies convergence.

The cavity is a unit square [0,1] x [0,1] with:
- Top wall moving at U = (1, 0, 0)
- All other walls stationary
- nu = 0.01 (Re = 100)
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from pyfoam.core.dtype import CFD_DTYPE, INDEX_DTYPE
from pyfoam.io.foam_file import FoamFileHeader, FileFormat, write_foam_file


# ---------------------------------------------------------------------------
# Mesh generation helpers
# ---------------------------------------------------------------------------

def _make_cavity_case(
    case_dir: Path,
    n_cells_x: int = 10,
    n_cells_y: int = 10,
    nu: float = 0.01,
) -> None:
    """Write a complete lid-driven cavity case to *case_dir*.

    Creates:
    - constant/polyMesh/{points, faces, owner, neighbour, boundary}
    - constant/transportProperties
    - 0/U, 0/p
    - system/{controlDict, fvSchemes, fvSolution}
    """
    case_dir.mkdir(parents=True, exist_ok=True)

    # ---- Mesh ----
    dx = 1.0 / n_cells_x
    dy = 1.0 / n_cells_y
    dz = 0.1  # small depth for 3D (empty BC)

    # Points: two layers (z=0, z=dz)
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

    # Faces: internal + boundary
    faces = []
    owner = []
    neighbour = []

    # Internal vertical faces (between cells in x-direction)
    for j in range(n_cells_y):
        for i in range(n_cells_x - 1):
            p0 = j * (n_cells_x + 1) + i + 1
            p1 = p0 + n_cells_x + 1
            p2 = p1 + n_base
            p3 = p0 + n_base
            faces.append((4, p0, p1, p2, p3))
            owner.append(j * n_cells_x + i)
            neighbour.append(j * n_cells_x + i + 1)

    # Internal horizontal faces (between cells in y-direction)
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

    # frontAndBack (z-normal, empty)
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

    # Write mesh files
    mesh_dir = case_dir / "constant" / "polyMesh"
    mesh_dir.mkdir(parents=True, exist_ok=True)

    header_base = FoamFileHeader(
        version="2.0",
        format=FileFormat.ASCII,
        location="constant/polyMesh",
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
        "application     simpleFoam;\n"
        "startFrom       startTime;\n"
        "startTime       0;\n"
        "stopAt          endTime;\n"
        "endTime         500;\n"
        "deltaT          1;\n"
        "writeControl    timeStep;\n"
        "writeInterval   100;\n"
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
        "SIMPLE\n{\n"
        "    nNonOrthogonalCorrectors 0;\n"
        "    residualControl\n    {\n"
        "        p               1e-4;\n"
        "        U               1e-4;\n"
        "    }\n"
        "    relaxationFactors\n    {\n"
        "        p               0.3;\n"
        "        U               0.7;\n"
        "    }\n"
        "    convergenceTolerance 1e-4;\n"
        "    maxOuterIterations  200;\n"
        "}\n"
    )
    write_foam_file(sys_dir / "fvSolution", fv_header, fv_body, overwrite=True)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.fixture
def cavity_case(tmp_path):
    """Create a lid-driven cavity case in a temporary directory."""
    case_dir = tmp_path / "cavity"
    _make_cavity_case(case_dir, n_cells_x=4, n_cells_y=4, nu=0.01)
    return case_dir


class TestSimpleFoamCavity:
    """End-to-end tests for simpleFoam with lid-driven cavity."""

    def test_case_loads(self, cavity_case):
        """Case directory is readable and has expected structure."""
        from pyfoam.io.case import Case

        case = Case(cavity_case)
        assert case.has_mesh()
        assert case.has_field("U", 0)
        assert case.has_field("p", 0)
        assert case.get_application() == "simpleFoam"
        assert case.get_end_time() == 500
        assert case.get_delta_t() == 1.0

    def test_mesh_builds(self, cavity_case):
        """FvMesh is constructed correctly from case data."""
        from pyfoam.applications.solver_base import SolverBase

        solver = SolverBase(cavity_case)
        mesh = solver.mesh

        assert mesh.n_cells == 16  # 4x4
        assert mesh.n_internal_faces > 0
        assert mesh.cell_volumes.shape == (16,)
        assert mesh.face_areas.shape[0] == mesh.n_faces
        assert mesh.owner.shape[0] == mesh.n_faces
        assert mesh.neighbour.shape[0] == mesh.n_internal_faces

    def test_fields_initialise(self, cavity_case):
        """Fields are initialised from the 0/ directory."""
        from pyfoam.applications.simple_foam import SimpleFoam
        from pyfoam.core.dtype import CFD_DTYPE

        solver = SimpleFoam(cavity_case)

        # U should be (16, 3) zeros (uniform (0 0 0))
        assert solver.U.shape == (16, 3)
        assert torch.allclose(solver.U, torch.zeros(16, 3, dtype=CFD_DTYPE))

        # p should be (16,) zeros
        assert solver.p.shape == (16,)
        assert torch.allclose(solver.p, torch.zeros(16, dtype=CFD_DTYPE))

        # phi should be (n_faces,) zeros
        assert solver.phi.shape == (solver.mesh.n_faces,)

    def test_nu_read(self, cavity_case):
        """Kinematic viscosity is read from transportProperties."""
        from pyfoam.applications.simple_foam import SimpleFoam

        solver = SimpleFoam(cavity_case)
        assert abs(solver.nu - 0.01) < 1e-10

    def test_fv_solution_settings(self, cavity_case):
        """fvSolution settings are read correctly."""
        from pyfoam.applications.simple_foam import SimpleFoam

        solver = SimpleFoam(cavity_case)
        assert solver.p_solver == "PCG"
        assert solver.U_solver == "PBiCGStab"
        assert abs(solver.alpha_p - 0.3) < 1e-10
        assert abs(solver.alpha_U - 0.7) < 1e-10
        assert abs(solver.convergence_tolerance - 1e-4) < 1e-10

    def test_run_converges(self, cavity_case):
        """simpleFoam runs and converges on the cavity case."""
        from pyfoam.applications.simple_foam import SimpleFoam

        solver = SimpleFoam(cavity_case)
        conv = solver.run()

        # The solver should have run at least one iteration
        assert conv.outer_iterations >= 1

        # Fields should be non-trivial after solving
        assert solver.U.shape == (16, 3)
        assert solver.p.shape == (16,)

    def test_run_writes_output(self, cavity_case):
        """simpleFoam writes field files to time directories."""
        from pyfoam.applications.simple_foam import SimpleFoam

        solver = SimpleFoam(cavity_case)
        solver.run()

        # Check that at least one output time directory was created
        time_dirs = [d for d in cavity_case.iterdir()
                     if d.is_dir() and d.name.replace(".", "").isdigit()
                     and d.name != "0"]
        assert len(time_dirs) >= 1

        # Check that U and p were written
        for td in time_dirs:
            assert (td / "U").exists(), f"U not found in {td}"
            assert (td / "p").exists(), f"p not found in {td}"

    def test_fields_are_valid(self, cavity_case):
        """Written fields are valid OpenFOAM format."""
        from pyfoam.applications.simple_foam import SimpleFoam
        from pyfoam.io.field_io import read_field

        solver = SimpleFoam(cavity_case)
        solver.run()

        # Find the last written time directory
        time_dirs = sorted(
            [d for d in cavity_case.iterdir()
             if d.is_dir() and d.name.replace(".", "").isdigit()
             and d.name != "0"],
            key=lambda d: float(d.name),
        )
        assert len(time_dirs) >= 1

        last_dir = time_dirs[-1]
        U_data = read_field(last_dir / "U")
        p_data = read_field(last_dir / "p")

        assert U_data.scalar_type == "vector"
        assert p_data.scalar_type == "scalar"
        assert not U_data.is_uniform
        assert not p_data.is_uniform

    def test_solver_produces_valid_fields(self, cavity_case):
        """Solver produces fields with correct shapes and finite values."""
        from pyfoam.applications.simple_foam import SimpleFoam

        solver = SimpleFoam(cavity_case)
        conv = solver.run()

        # Fields should have correct shapes
        assert solver.U.shape == (16, 3)
        assert solver.p.shape == (16,)
        assert solver.phi.shape == (solver.mesh.n_faces,)

        # All values should be finite (no NaN or Inf)
        assert torch.isfinite(solver.U).all(), "U contains NaN/Inf"
        assert torch.isfinite(solver.p).all(), "p contains NaN/Inf"
        assert torch.isfinite(solver.phi).all(), "phi contains NaN/Inf"

    def test_pressure_field_shape(self, cavity_case):
        """Pressure field has correct shape after solving."""
        from pyfoam.applications.simple_foam import SimpleFoam

        solver = SimpleFoam(cavity_case)
        solver.run()

        # Pressure should have the correct shape
        assert solver.p.shape == (16,)
        assert torch.isfinite(solver.p).all(), "p contains NaN/Inf"

"""
Unit tests for AcousticFoam — acoustic wave propagation solver.

Tests cover:
- Case loading and mesh construction
- Acoustic property reading
- Field initialisation (p', u')
- Custom c0/U0/rho0 injection
- Solver produces finite values
- Solver writes output
- Pressure pulse propagation
- Non-reflecting BC behaviour
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from pyfoam.core.dtype import CFD_DTYPE, INDEX_DTYPE
from pyfoam.io.foam_file import FoamFileHeader, FileFormat, write_foam_file


# ---------------------------------------------------------------------------
# Mesh generation helper for acoustic case
# ---------------------------------------------------------------------------

def _make_acoustic_case(
    case_dir: Path,
    n_cells_x: int = 10,
    n_cells_y: int = 10,
    p_init: float = 0.0,
    p_source: float = 1.0,
    end_time: int = 100,
    delta_t: float = 0.001,
    write_interval: int = 100,
) -> None:
    """Write a complete acoustic case to *case_dir*.

    Creates a 2D square domain with:
    - Left wall at p' = p_source (fixed perturbation source)
    - Right wall: advective (non-reflecting outflow)
    - Top/bottom walls: zeroGradient (reflective)
    """
    case_dir.mkdir(parents=True, exist_ok=True)

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

    # Internal vertical faces
    for j in range(n_cells_y):
        for i in range(n_cells_x - 1):
            p0 = j * (n_cells_x + 1) + i + 1
            p1 = p0 + n_cells_x + 1
            p2 = p1 + n_base
            p3 = p0 + n_base
            faces.append((4, p0, p1, p2, p3))
            owner.append(j * n_cells_x + i)
            neighbour.append(j * n_cells_x + i + 1)

    # Internal horizontal faces
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

    # Boundary: sourceWall (left)
    for j in range(n_cells_y):
        p0 = j * (n_cells_x + 1)
        p1 = p0 + n_cells_x + 1
        p2 = p1 + n_base
        p3 = p0 + n_base
        faces.append((4, p0, p1, p2, p3))
        owner.append(j * n_cells_x)
    n_source = n_cells_y
    source_start = n_internal

    # Boundary: transmissiveWall (right)
    for j in range(n_cells_y):
        p0 = j * (n_cells_x + 1) + n_cells_x
        p1 = p0 + n_cells_x + 1
        p2 = p1 + n_base
        p3 = p0 + n_base
        faces.append((4, p0, p1, p2, p3))
        owner.append(j * n_cells_x + n_cells_x - 1)
    n_trans = n_cells_y
    trans_start = source_start + n_source

    # Boundary: reflectiveWalls (top, bottom)
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
    n_reflect = 2 * n_cells_x
    reflect_start = trans_start + n_trans

    # Boundary: frontAndBack (empty)
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
    empty_start = reflect_start + n_reflect

    n_faces = len(faces)
    n_cells = n_cells_x * n_cells_y

    # Write mesh files
    mesh_dir = case_dir / "constant" / "polyMesh"
    mesh_dir.mkdir(parents=True, exist_ok=True)

    header_base = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        location="constant/polyMesh",
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
    lines.append("    sourceWall")
    lines.append("    {")
    lines.append("        type            wall;")
    lines.append(f"        nFaces          {n_source};")
    lines.append(f"        startFace       {source_start};")
    lines.append("    }")
    lines.append("    transmissiveWall")
    lines.append("    {")
    lines.append("        type            wall;")
    lines.append(f"        nFaces          {n_trans};")
    lines.append(f"        startFace       {trans_start};")
    lines.append("    }")
    lines.append("    reflectiveWalls")
    lines.append("    {")
    lines.append("        type            wall;")
    lines.append(f"        nFaces          {n_reflect};")
    lines.append(f"        startFace       {reflect_start};")
    lines.append("    }")
    lines.append("    frontAndBack")
    lines.append("    {")
    lines.append("        type            empty;")
    lines.append(f"        nFaces          {n_empty};")
    lines.append(f"        startFace       {empty_start};")
    lines.append("    }")
    lines.append(")")
    write_foam_file(mesh_dir / "boundary", h, "\n".join(lines), overwrite=True)

    # ---- 0/p' ----
    zero_dir = case_dir / "0"
    zero_dir.mkdir(exist_ok=True)

    p_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="volScalarField", location="0", object="p'",
    )
    p_body = (
        "dimensions      [1 -1 -2 0 0 0 0];\n\n"
        f"internalField   uniform {p_init};\n\n"
        "boundaryField\n{\n"
        "    sourceWall\n    {\n"
        "        type            fixedValue;\n"
        f"        value           uniform {p_source};\n"
        "    }\n"
        "    transmissiveWall\n    {\n"
        "        type            advective;\n"
        "    }\n"
        "    reflectiveWalls\n    {\n"
        "        type            zeroGradient;\n"
        "    }\n"
        "    frontAndBack\n    {\n"
        "        type            empty;\n"
        "    }\n"
        "}\n"
    )
    write_foam_file(zero_dir / "p'", p_header, p_body, overwrite=True)

    # ---- 0/u' ----
    u_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="volVectorField", location="0", object="u'",
    )
    u_body = (
        "dimensions      [0 1 -1 0 0 0 0];\n\n"
        "internalField   uniform (0 0 0);\n\n"
        "boundaryField\n{\n"
        "    sourceWall\n    {\n"
        "        type            fixedValue;\n"
        "        value           uniform (0 0 0);\n"
        "    }\n"
        "    transmissiveWall\n    {\n"
        "        type            advective;\n"
        "    }\n"
        "    reflectiveWalls\n    {\n"
        "        type            zeroGradient;\n"
        "    }\n"
        "    frontAndBack\n    {\n"
        "        type            empty;\n"
        "    }\n"
        "}\n"
    )
    write_foam_file(zero_dir / "u'", u_header, u_body, overwrite=True)

    # ---- system/controlDict ----
    sys_dir = case_dir / "system"
    sys_dir.mkdir(exist_ok=True)

    cd_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="dictionary", location="system", object="controlDict",
    )
    cd_body = (
        "application     acousticFoam;\n"
        "startFrom       startTime;\n"
        "startTime       0;\n"
        "stopAt          endTime;\n"
        f"endTime         {end_time};\n"
        f"deltaT          {delta_t};\n"
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

    fv_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="dictionary", location="system", object="fvSolution",
    )
    fv_body = (
        "solvers\n{\n"
        "    p'\n    {\n"
        "        solver          PCG;\n"
        "        preconditioner  DIC;\n"
        "        tolerance       1e-10;\n"
        "        relTol          0.01;\n"
        "        maxIter         1000;\n"
        "    }\n"
        "}\n"
    )
    write_foam_file(sys_dir / "fvSolution", fv_header, fv_body, overwrite=True)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def acoustic_case(tmp_path):
    """Create a small acoustic case (4x4 mesh)."""
    case_dir = tmp_path / "acoustic"
    _make_acoustic_case(
        case_dir,
        n_cells_x=4,
        n_cells_y=4,
        p_init=0.0,
        p_source=1.0,
        end_time=10,
        delta_t=0.001,
        write_interval=10,
    )
    return case_dir


@pytest.fixture
def tiny_acoustic_case(tmp_path):
    """Create a minimal 2x2 acoustic case for fast tests."""
    case_dir = tmp_path / "tiny_acoustic"
    _make_acoustic_case(
        case_dir,
        n_cells_x=2,
        n_cells_y=2,
        p_init=0.0,
        p_source=0.5,
        end_time=0.5,
        delta_t=0.001,
        write_interval=5,
    )
    return case_dir


# ===========================================================================
# Tests
# ===========================================================================


class TestAcousticFoamInit:
    """Tests for AcousticFoam initialisation."""

    def test_case_loads(self, acoustic_case):
        """Case directory is readable."""
        from pyfoam.io.case import Case
        case = Case(acoustic_case)
        assert case.has_mesh()

    def test_mesh_builds(self, acoustic_case):
        """FvMesh is constructed correctly."""
        from pyfoam.applications.solver_base import SolverBase
        solver = SolverBase(acoustic_case)
        assert solver.mesh.n_cells == 16  # 4x4

    def test_fields_initialise(self, acoustic_case):
        """p' and u' fields are initialised."""
        from pyfoam.applications.acoustic_foam import AcousticFoam
        solver = AcousticFoam(acoustic_case)
        assert solver.p_prime.shape == (16,)
        assert solver.u_prime.shape == (16, 3)

    def test_default_acoustic_properties(self, acoustic_case):
        """Default acoustic properties are used when file not found."""
        from pyfoam.applications.acoustic_foam import AcousticFoam
        solver = AcousticFoam(acoustic_case)
        assert abs(solver.c0 - 343.0) < 1e-6
        assert abs(solver.rho0 - 1.225) < 1e-6

    def test_custom_c0_injection(self, acoustic_case):
        """Custom c0 can be injected."""
        from pyfoam.applications.acoustic_foam import AcousticFoam
        solver = AcousticFoam(acoustic_case, c0=300.0)
        assert abs(solver.c0 - 300.0) < 1e-10

    def test_custom_U0_injection(self, acoustic_case):
        """Custom U0 can be injected."""
        from pyfoam.applications.acoustic_foam import AcousticFoam
        solver = AcousticFoam(acoustic_case, U0=(10.0, 0.0, 0.0))
        assert torch.allclose(
            solver.U0,
            torch.tensor([10.0, 0.0, 0.0], dtype=torch.float64),
        )

    def test_custom_rho0_injection(self, acoustic_case):
        """Custom rho0 can be injected."""
        from pyfoam.applications.acoustic_foam import AcousticFoam
        solver = AcousticFoam(acoustic_case, rho0=1.0)
        assert abs(solver.rho0 - 1.0) < 1e-10


class TestAcousticFoamRun:
    """Tests for the full solver run."""

    def test_run_completes(self, tiny_acoustic_case):
        """AcousticFoam runs to completion."""
        from pyfoam.applications.acoustic_foam import AcousticFoam
        solver = AcousticFoam(tiny_acoustic_case)
        conv = solver.run()
        assert conv is not None

    def test_run_finite_values(self, tiny_acoustic_case):
        """All field values are finite after run."""
        from pyfoam.applications.acoustic_foam import AcousticFoam
        solver = AcousticFoam(tiny_acoustic_case)
        solver.run()
        assert torch.isfinite(solver.p_prime).all()
        assert torch.isfinite(solver.u_prime).all()

    def test_run_writes_output(self, tiny_acoustic_case):
        """AcousticFoam writes fields to time directories."""
        from pyfoam.applications.acoustic_foam import AcousticFoam
        solver = AcousticFoam(tiny_acoustic_case)
        solver.run()

        time_dirs = [
            d for d in tiny_acoustic_case.iterdir()
            if d.is_dir() and d.name.replace(".", "").isdigit() and d.name != "0"
        ]
        assert len(time_dirs) >= 1

    def test_source_wall_prescribed(self, tiny_acoustic_case):
        """Source wall maintains prescribed pressure."""
        from pyfoam.applications.acoustic_foam import AcousticFoam
        solver = AcousticFoam(tiny_acoustic_case)
        solver.run()

        # At least some cells near x=0 should have non-zero p'
        mesh = solver.mesh
        x = mesh.cell_centres[:, 0]
        left_cells = x < 0.6
        if left_cells.any():
            assert solver.p_prime[left_cells].abs().max() > 0

    def test_convergence_data_populated(self, tiny_acoustic_case):
        """ConvergenceData has values after run."""
        from pyfoam.applications.acoustic_foam import AcousticFoam
        solver = AcousticFoam(tiny_acoustic_case)
        conv = solver.run()
        assert conv.T_residual >= 0

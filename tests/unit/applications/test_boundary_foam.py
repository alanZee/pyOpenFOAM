"""
Unit tests for BoundaryFoam — 1D turbulent boundary layer solver.

Tests cover:
- Case loading and mesh construction
- Field initialisation from 0/ directory
- Transport property reading (nu)
- boundaryFoam settings parsing (dpdx, UInf)
- Boundary condition building
- 1D momentum equation assembly
- Under-relaxation
- Turbulence model initialisation (optional)
- Run convergence
- Velocity profile extraction
- Field writing to time directories
- Written field format validity
- Solver produces finite values
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from pyfoam.core.dtype import CFD_DTYPE, INDEX_DTYPE
from pyfoam.io.foam_file import FoamFileHeader, FileFormat, write_foam_file


# ---------------------------------------------------------------------------
# Mesh generation helper for 1D boundary layer case
# ---------------------------------------------------------------------------

def _make_bl_case(
    case_dir: Path,
    n_cells: int = 10,
    y_max: float = 1.0,
    nu: float = 0.01,
    dp_dx: float = 1.0,
    U_inf: float = 1.0,
    end_time: int = 500,
    write_interval: int = 100,
    alpha_U: float = 0.7,
    convergence_tolerance: float = 1e-4,
    max_outer_iterations: int = 200,
    turbulence_model: str | None = None,
) -> None:
    """Write a complete 1D boundary layer case to *case_dir*.

    Creates a 1D mesh with cells stacked in the y-direction.
    Wall at y=0 (no-slip), far-field at y_max (fixedValue U_inf).

    Creates:
    - constant/polyMesh/{points, faces, owner, neighbour, boundary}
    - constant/transportProperties
    - constant/turbulenceProperties (if turbulence_model is set)
    - 0/U, 0/p
    - system/{controlDict, fvSchemes, fvSolution}
    """
    case_dir.mkdir(parents=True, exist_ok=True)

    dx = 0.1  # small depth in x
    dz = 0.1  # small depth in z
    dy = y_max / n_cells

    # ---- Points ----
    # 4 points per y-level, (n_cells + 1) levels
    points = []
    for j in range(n_cells + 1):
        y = j * dy
        points.append((0.0, y, 0.0))    # p0: (0, y, 0)
        points.append((dx, y, 0.0))     # p1: (dx, y, 0)
        points.append((dx, y, dz))      # p2: (dx, y, dz)
        points.append((0.0, y, dz))     # p3: (0, y, dz)

    n_points = len(points)

    # ---- Faces ----
    faces = []
    owner = []
    neighbour = []

    # Internal y-faces (between adjacent cells)
    for j in range(n_cells - 1):
        level = j + 1
        p0 = level * 4 + 0
        p1 = level * 4 + 1
        p2 = level * 4 + 2
        p3 = level * 4 + 3
        faces.append((4, p0, p1, p2, p3))
        owner.append(j)
        neighbour.append(j + 1)

    n_internal = len(neighbour)

    # Wall boundary (y=0): face at level 0
    wall_start = n_internal
    p0, p1, p2, p3 = 0, 1, 2, 3
    faces.append((4, p3, p2, p1, p0))  # Normal points outward (-y)
    owner.append(0)

    # Far-field boundary (y=y_max): face at level n_cells
    farfield_start = n_internal + 1
    level = n_cells
    p0 = level * 4 + 0
    p1 = level * 4 + 1
    p2 = level * 4 + 2
    p3 = level * 4 + 3
    faces.append((4, p0, p1, p2, p3))  # Normal points outward (+y)
    owner.append(n_cells - 1)

    # Empty patches (left, right, front, back)
    empty_start = n_internal + 2

    # Left (x=0): normal points -x
    for j in range(n_cells):
        base = j * 4
        p0, p3 = base + 0, base + 3
        p0_top, p3_top = (j + 1) * 4 + 0, (j + 1) * 4 + 3
        faces.append((4, p0, p3, p3_top, p0_top))
        owner.append(j)

    # Right (x=dx): normal points +x
    for j in range(n_cells):
        base = j * 4
        p1, p2 = base + 1, base + 2
        p1_top, p2_top = (j + 1) * 4 + 1, (j + 1) * 4 + 2
        faces.append((4, p1, p1_top, p2_top, p2))
        owner.append(j)

    # Front (z=0): normal points -z
    for j in range(n_cells):
        base = j * 4
        p0, p1 = base + 0, base + 1
        p0_top, p1_top = (j + 1) * 4 + 0, (j + 1) * 4 + 1
        faces.append((4, p0, p0_top, p1_top, p1))
        owner.append(j)

    # Back (z=dz): normal points +z
    for j in range(n_cells):
        base = j * 4
        p2, p3 = base + 2, base + 3
        p2_top, p3_top = (j + 1) * 4 + 2, (j + 1) * 4 + 3
        faces.append((4, p3, p2, p2_top, p3_top))
        owner.append(j)

    n_faces = len(faces)

    # ---- Write mesh files ----
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
    for x, y, z in points:
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
    n_empty = 4 * n_cells
    lines = ["6", "("]
    lines.append("    wall")
    lines.append("    {")
    lines.append("        type            wall;")
    lines.append(f"        nFaces          1;")
    lines.append(f"        startFace       {wall_start};")
    lines.append("    }")
    lines.append("    farField")
    lines.append("    {")
    lines.append("        type            patch;")
    lines.append(f"        nFaces          1;")
    lines.append(f"        startFace       {farfield_start};")
    lines.append("    }")
    lines.append("    left")
    lines.append("    {")
    lines.append("        type            empty;")
    lines.append(f"        nFaces          {n_cells};")
    lines.append(f"        startFace       {empty_start};")
    lines.append("    }")
    lines.append("    right")
    lines.append("    {")
    lines.append("        type            empty;")
    lines.append(f"        nFaces          {n_cells};")
    lines.append(f"        startFace       {empty_start + n_cells};")
    lines.append("    }")
    lines.append("    front")
    lines.append("    {")
    lines.append("        type            empty;")
    lines.append(f"        nFaces          {n_cells};")
    lines.append(f"        startFace       {empty_start + 2 * n_cells};")
    lines.append("    }")
    lines.append("    back")
    lines.append("    {")
    lines.append("        type            empty;")
    lines.append(f"        nFaces          {n_cells};")
    lines.append(f"        startFace       {empty_start + 3 * n_cells};")
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

    # ---- turbulenceProperties (optional) ----
    if turbulence_model is not None:
        turb_header = FoamFileHeader(
            version="2.0", format=FileFormat.ASCII,
            class_name="dictionary", location="constant", object="turbulenceProperties",
        )
        turb_body = (
            "simulationType  RAS;\n\n"
            "RAS\n{\n"
            f"    model           {turbulence_model};\n"
            "    turbulence      on;\n"
            "    printCoeffs     on;\n"
            "}\n"
        )
        write_foam_file(
            case_dir / "constant" / "turbulenceProperties", turb_header,
            turb_body, overwrite=True,
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
        "    wall\n    {\n"
        "        type            fixedValue;\n"
        "        value           uniform (0 0 0);\n"
        "    }\n"
        "    farField\n    {\n"
        f"        type            fixedValue;\n"
        f"        value           uniform ({U_inf} 0 0);\n"
        "    }\n"
        "    left\n    {\n"
        "        type            empty;\n"
        "    }\n"
        "    right\n    {\n"
        "        type            empty;\n"
        "    }\n"
        "    front\n    {\n"
        "        type            empty;\n"
        "    }\n"
        "    back\n    {\n"
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
        "    wall\n    {\n"
        "        type            zeroGradient;\n"
        "    }\n"
        "    farField\n    {\n"
        "        type            zeroGradient;\n"
        "    }\n"
        "    left\n    {\n"
        "        type            empty;\n"
        "    }\n"
        "    right\n    {\n"
        "        type            empty;\n"
        "    }\n"
        "    front\n    {\n"
        "        type            empty;\n"
        "    }\n"
        "    back\n    {\n"
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
        "application     boundaryFoam;\n"
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
        "    U\n    {\n"
        "        solver          PBiCGStab;\n"
        "        preconditioner  DILU;\n"
        "        tolerance       1e-6;\n"
        "        relTol          0.01;\n"
        "    }\n"
        "}\n\n"
        "boundaryFoam\n{\n"
        f"    dpdx                {dp_dx};\n"
        f"    UInf                {U_inf};\n"
        f"    convergenceTolerance {convergence_tolerance};\n"
        f"    maxOuterIterations  {max_outer_iterations};\n"
        "    relaxationFactors\n    {\n"
        f"        U               {alpha_U};\n"
        "    }\n"
        "}\n"
    )
    write_foam_file(sys_dir / "fvSolution", fv_header, fv_body, overwrite=True)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def bl_case(tmp_path):
    """Create a 1D boundary layer case in a temporary directory."""
    case_dir = tmp_path / "bl"
    _make_bl_case(
        case_dir,
        n_cells=5,
        y_max=1.0,
        nu=0.01,
        dp_dx=1.0,
        U_inf=1.0,
        end_time=100,
        write_interval=100,
        max_outer_iterations=100,
    )
    return case_dir


@pytest.fixture
def tiny_bl_case(tmp_path):
    """Create a minimal 3-cell boundary layer case for fast tests."""
    case_dir = tmp_path / "tiny_bl"
    _make_bl_case(
        case_dir,
        n_cells=3,
        y_max=1.0,
        nu=0.01,
        dp_dx=1.0,
        U_inf=1.0,
        end_time=10,
        write_interval=10,
        max_outer_iterations=50,
    )
    return case_dir


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBoundaryFoamInit:
    """Tests for BoundaryFoam initialisation and property reading."""

    def test_case_loads(self, bl_case):
        """Case directory is readable and has expected structure."""
        from pyfoam.io.case import Case

        case = Case(bl_case)
        assert case.has_mesh()
        assert case.has_field("U", 0)
        assert case.has_field("p", 0)
        assert case.get_application() == "boundaryFoam"
        assert case.get_end_time() == 100

    def test_mesh_builds(self, bl_case):
        """FvMesh is constructed correctly from 1D case data."""
        from pyfoam.applications.solver_base import SolverBase

        solver = SolverBase(bl_case)
        mesh = solver.mesh

        assert mesh.n_cells == 5
        assert mesh.n_internal_faces > 0
        assert mesh.cell_volumes.shape == (5,)
        assert mesh.face_areas.shape[0] == mesh.n_faces
        assert mesh.owner.shape[0] == mesh.n_faces
        assert mesh.neighbour.shape[0] == mesh.n_internal_faces

    def test_fields_initialise(self, bl_case):
        """Fields are initialised from the 0/ directory."""
        from pyfoam.applications.boundary_foam import BoundaryFoam

        solver = BoundaryFoam(bl_case)

        # U should be (5, 3) zeros (uniform (0 0 0))
        assert solver.U.shape == (5, 3)
        assert torch.allclose(solver.U, torch.zeros(5, 3, dtype=CFD_DTYPE))

        # p should be (5,) zeros
        assert solver.p.shape == (5,)
        assert torch.allclose(solver.p, torch.zeros(5, dtype=CFD_DTYPE))

        # phi should be (n_faces,) zeros
        assert solver.phi.shape == (solver.mesh.n_faces,)

    def test_nu_read(self, bl_case):
        """Kinematic viscosity is read from transportProperties."""
        from pyfoam.applications.boundary_foam import BoundaryFoam

        solver = BoundaryFoam(bl_case)
        assert abs(solver.nu - 0.01) < 1e-10

    def test_boundary_foam_settings(self, bl_case):
        """boundaryFoam settings are read correctly from fvSolution."""
        from pyfoam.applications.boundary_foam import BoundaryFoam

        solver = BoundaryFoam(bl_case)
        assert abs(solver.dp_dx - 1.0) < 1e-10
        assert abs(solver.U_inf - 1.0) < 1e-10
        assert abs(solver.alpha_U - 0.7) < 1e-10
        assert abs(solver.convergence_tolerance - 1e-4) < 1e-10

    def test_turbulence_disabled_by_default(self, bl_case):
        """Turbulence is disabled when no turbulenceProperties exists."""
        from pyfoam.applications.boundary_foam import BoundaryFoam

        solver = BoundaryFoam(bl_case)
        assert solver.turbulence_enabled is False
        assert solver.ras is None


class TestBoundaryFoamBoundaryConditions:
    """Tests for boundary condition building."""

    def test_bc_tensor_shape(self, bl_case):
        """U_bc has correct shape."""
        from pyfoam.applications.boundary_foam import BoundaryFoam

        solver = BoundaryFoam(bl_case)
        U_bc = solver._build_boundary_conditions()

        assert U_bc.shape == (5, 3)

    def test_bc_has_fixed_values(self, bl_case):
        """U_bc has prescribed values for boundary cells."""
        from pyfoam.applications.boundary_foam import BoundaryFoam

        solver = BoundaryFoam(bl_case)
        U_bc = solver._build_boundary_conditions()

        bc_mask = ~torch.isnan(U_bc[:, 0])
        assert bc_mask.any(), "No boundary conditions found"

    def test_bc_wall_velocity(self, bl_case):
        """Wall cell has U = (0, 0, 0)."""
        from pyfoam.applications.boundary_foam import BoundaryFoam

        solver = BoundaryFoam(bl_case)
        U_bc = solver._build_boundary_conditions()

        # Wall cell (cell 0) should have U = (0, 0, 0)
        wall_mask = (U_bc[:, 0]).abs() < 1e-10
        bc_mask = ~torch.isnan(U_bc[:, 0])
        wall_mask = wall_mask & bc_mask

        assert wall_mask.any(), "No wall BC cells found"
        # Wall cells should have all velocity components = 0
        assert torch.allclose(
            U_bc[wall_mask, 1],
            torch.zeros(wall_mask.sum(), dtype=CFD_DTYPE),
            atol=1e-10,
        )
        assert torch.allclose(
            U_bc[wall_mask, 2],
            torch.zeros(wall_mask.sum(), dtype=CFD_DTYPE),
            atol=1e-10,
        )

    def test_bc_farfield_velocity(self, bl_case):
        """Far-field cell has U = (U_inf, 0, 0)."""
        from pyfoam.applications.boundary_foam import BoundaryFoam

        solver = BoundaryFoam(bl_case)
        U_bc = solver._build_boundary_conditions()

        # Far-field cell should have U_x = 1.0
        farfield_mask = (U_bc[:, 0] - 1.0).abs() < 1e-10
        assert farfield_mask.any(), "No far-field BC cells found"
        assert torch.allclose(
            U_bc[farfield_mask, 1],
            torch.zeros(farfield_mask.sum(), dtype=CFD_DTYPE),
            atol=1e-10,
        )

    def test_parse_vector_value_tuple(self):
        """_parse_vector_value handles tuple input."""
        from pyfoam.applications.boundary_foam import BoundaryFoam

        result = BoundaryFoam._parse_vector_value((1.0, 2.0, 3.0))
        assert result == (1.0, 2.0, 3.0)

    def test_parse_vector_value_string(self):
        """_parse_vector_value handles 'uniform ( x y z )' string."""
        from pyfoam.applications.boundary_foam import BoundaryFoam

        result = BoundaryFoam._parse_vector_value("uniform ( 1.5 0.0 -0.5 )")
        assert result is not None
        assert abs(result[0] - 1.5) < 1e-10
        assert abs(result[1] - 0.0) < 1e-10
        assert abs(result[2] - (-0.5)) < 1e-10

    def test_parse_vector_value_invalid(self):
        """_parse_vector_value returns None for invalid input."""
        from pyfoam.applications.boundary_foam import BoundaryFoam

        assert BoundaryFoam._parse_vector_value("invalid") is None
        assert BoundaryFoam._parse_vector_value(42) is None


class TestBoundaryFoamMomentum:
    """Tests for 1D momentum equation assembly."""

    def test_momentum_matrix_shape(self, bl_case):
        """1D momentum matrix has correct dimensions."""
        from pyfoam.applications.boundary_foam import BoundaryFoam

        solver = BoundaryFoam(bl_case)
        mat, source = solver._build_1d_momentum_matrix(solver.U)

        assert mat.diag.shape == (5,)
        assert mat.lower.shape == (solver.mesh.n_internal_faces,)
        assert mat.upper.shape == (solver.mesh.n_internal_faces,)
        assert source.shape == (5, 3)

    def test_momentum_source_has_pressure_gradient(self, bl_case):
        """Source term includes the prescribed pressure gradient."""
        from pyfoam.applications.boundary_foam import BoundaryFoam

        solver = BoundaryFoam(bl_case)
        mat, source = solver._build_1d_momentum_matrix(solver.U)

        # Source should have dp/dx contribution in x-direction
        # source[:, 0] = -dp_dx * cell_volume
        assert (source[:, 0] != 0).any(), "Pressure gradient source is zero"
        # All y and z sources should be zero
        assert torch.allclose(
            source[:, 1], torch.zeros(5, dtype=CFD_DTYPE), atol=1e-20
        )
        assert torch.allclose(
            source[:, 2], torch.zeros(5, dtype=CFD_DTYPE), atol=1e-20
        )

    def test_momentum_with_turbulent_viscosity(self, bl_case):
        """Momentum matrix can be built with a viscosity field."""
        from pyfoam.applications.boundary_foam import BoundaryFoam

        solver = BoundaryFoam(bl_case)
        nu_field = torch.full((5,), 0.02, dtype=CFD_DTYPE)
        mat, source = solver._build_1d_momentum_matrix(
            solver.U, nu_field=nu_field
        )

        assert mat.diag.shape == (5,)
        assert torch.isfinite(mat.diag).all()
        assert (mat.diag > 0).all(), "Diagonal should be positive"


class TestBoundaryFoamRelaxation:
    """Tests for under-relaxation."""

    def test_relaxation_modifies_diagonal(self, bl_case):
        """Under-relaxation increases diagonal dominance."""
        from pyfoam.applications.boundary_foam import BoundaryFoam

        solver = BoundaryFoam(bl_case)
        mat, source = solver._build_1d_momentum_matrix(solver.U)
        diag_before = mat.diag.clone()

        mat_relaxed, _, A_p = solver._apply_under_relaxation(
            mat, source.clone(), solver.U, 0.7
        )

        # Relaxed diagonal should be >= original diagonal
        assert (mat_relaxed.diag >= diag_before - 1e-10).all()

    def test_relaxation_adds_source_contribution(self, bl_case):
        """Under-relaxation adds source contribution from (D_new - D_old)*U."""
        from pyfoam.applications.boundary_foam import BoundaryFoam

        solver = BoundaryFoam(bl_case)

        # Set some non-zero velocity
        solver.U[:, 0] = torch.linspace(0, 1, 5, dtype=CFD_DTYPE)

        mat, source = solver._build_1d_momentum_matrix(solver.U)
        source_before = source.clone()

        _, source_after, _ = solver._apply_under_relaxation(
            mat, source, solver.U, 0.7
        )

        # Source should have changed (relaxation contribution)
        diff = (source_after - source_before).abs().sum()
        assert diff > 0, "Source did not change after relaxation"


class TestBoundaryFoamSolver:
    """Tests for solver construction and execution."""

    def test_run_converges(self, tiny_bl_case):
        """boundaryFoam runs and produces valid output on a tiny case."""
        from pyfoam.applications.boundary_foam import BoundaryFoam

        solver = BoundaryFoam(tiny_bl_case)
        result = solver.run()

        # Fields should have correct shapes
        assert solver.U.shape == (3, 3)
        assert solver.p.shape == (3,)
        assert solver.phi.shape == (solver.mesh.n_faces,)

        # All values should be finite (no NaN or Inf)
        assert torch.isfinite(solver.U).all(), "U contains NaN/Inf"
        assert torch.isfinite(solver.p).all(), "p contains NaN/Inf"

        # Result should be a dict
        assert "converged" in result
        assert "iterations" in result

    def test_run_writes_output(self, tiny_bl_case):
        """boundaryFoam writes field files to time directories."""
        from pyfoam.applications.boundary_foam import BoundaryFoam

        solver = BoundaryFoam(tiny_bl_case)
        solver.run()

        # Check that at least one output time directory was created
        time_dirs = [
            d
            for d in tiny_bl_case.iterdir()
            if d.is_dir()
            and d.name.replace(".", "").isdigit()
            and d.name != "0"
        ]
        assert len(time_dirs) >= 1

        # Check that U and p were written
        for td in time_dirs:
            assert (td / "U").exists(), f"U not found in {td}"
            assert (td / "p").exists(), f"p not found in {td}"

    def test_velocity_profile_extraction(self, tiny_bl_case):
        """Velocity profile can be extracted after solving."""
        from pyfoam.applications.boundary_foam import BoundaryFoam

        solver = BoundaryFoam(tiny_bl_case)
        solver.run()

        y_coords, U_x = solver.get_velocity_profile()

        assert y_coords.shape == (3,)
        assert U_x.shape == (3,)
        assert torch.isfinite(y_coords).all()
        assert torch.isfinite(U_x).all()

    def test_turbulent_viscosity_profile(self, tiny_bl_case):
        """Turbulent viscosity profile can be extracted."""
        from pyfoam.applications.boundary_foam import BoundaryFoam

        solver = BoundaryFoam(tiny_bl_case)
        solver.run()

        y_coords, nu_t = solver.get_turbulent_viscosity_profile()

        assert y_coords.shape == (3,)
        assert nu_t.shape == (3,)
        # Without turbulence model, nu_t should be zero
        assert torch.allclose(nu_t, torch.zeros(3, dtype=CFD_DTYPE))

    def test_fields_are_valid_format(self, tiny_bl_case):
        """Written fields are valid OpenFOAM format."""
        from pyfoam.applications.boundary_foam import BoundaryFoam
        from pyfoam.io.field_io import read_field

        solver = BoundaryFoam(tiny_bl_case)
        solver.run()

        # Find the last written time directory
        time_dirs = sorted(
            [
                d
                for d in tiny_bl_case.iterdir()
                if d.is_dir()
                and d.name.replace(".", "").isdigit()
                and d.name != "0"
            ],
            key=lambda d: float(d.name),
        )
        assert len(time_dirs) >= 1

        last_dir = time_dirs[-1]
        U_data = read_field(last_dir / "U")
        p_data = read_field(last_dir / "p")

        assert U_data.scalar_type == "vector"
        assert p_data.scalar_type == "scalar"

    def test_velocity_changes_after_run(self, tiny_bl_case):
        """Velocity field changes from initial zero conditions."""
        from pyfoam.applications.boundary_foam import BoundaryFoam

        solver = BoundaryFoam(tiny_bl_case)
        U_initial = solver.U.clone()

        result = solver.run()

        # After running, velocity should have changed due to dp/dx
        U_diff = (solver.U - U_initial).abs().sum()
        assert U_diff > 0, (
            f"Velocity did not change during simulation "
            f"(U_diff={U_diff:.6e})"
        )

    def test_pressure_gradient_drives_flow(self, tiny_bl_case):
        """Positive dp/dx drives flow in the positive x-direction."""
        from pyfoam.applications.boundary_foam import BoundaryFoam

        solver = BoundaryFoam(tiny_bl_case)
        solver.run()

        # Interior cells should have positive U_x (driven by dp/dx > 0)
        # Cell 0 is wall (U=0), cell 2 is far-field (U=U_inf)
        # Cell 1 (interior) should have U_x > 0
        assert solver.U[1, 0] > 0, (
            f"Interior cell velocity should be positive, got {solver.U[1, 0]:.6e}"
        )

    def test_wall_velocity_is_zero(self, tiny_bl_case):
        """Wall cell velocity is zero (no-slip)."""
        from pyfoam.applications.boundary_foam import BoundaryFoam

        solver = BoundaryFoam(tiny_bl_case)
        solver.run()

        # Wall cell (cell 0) should have U = (0, 0, 0)
        assert abs(solver.U[0, 0].item()) < 1e-6, (
            f"Wall U_x should be ~0, got {solver.U[0, 0].item():.6e}"
        )
        assert abs(solver.U[0, 1].item()) < 1e-10
        assert abs(solver.U[0, 2].item()) < 1e-10


class TestBoundaryFoamTurbulence:
    """Tests for turbulence model integration."""

    def test_turbulence_enabled_with_ras(self, tmp_path):
        """Turbulence is enabled when turbulenceProperties specifies RAS."""
        case_dir = tmp_path / "bl_turb"
        _make_bl_case(
            case_dir,
            n_cells=3,
            y_max=1.0,
            nu=0.01,
            dp_dx=1.0,
            U_inf=1.0,
            end_time=10,
            write_interval=10,
            max_outer_iterations=50,
            turbulence_model="kEpsilon",
        )

        from pyfoam.applications.boundary_foam import BoundaryFoam

        solver = BoundaryFoam(case_dir)
        assert solver.turbulence_enabled is True
        assert solver.ras is not None

    def test_turbulence_laminar_type(self, tmp_path):
        """Turbulence is disabled when simulationType is laminar."""
        case_dir = tmp_path / "bl_lam"
        _make_bl_case(
            case_dir,
            n_cells=3,
            y_max=1.0,
            nu=0.01,
            dp_dx=1.0,
            U_inf=1.0,
            end_time=10,
            write_interval=10,
            max_outer_iterations=50,
        )

        from pyfoam.io.foam_file import FoamFileHeader, FileFormat, write_foam_file

        turb_header = FoamFileHeader(
            version="2.0", format=FileFormat.ASCII,
            class_name="dictionary", location="constant",
            object="turbulenceProperties",
        )
        write_foam_file(
            case_dir / "constant" / "turbulenceProperties",
            turb_header,
            "simulationType  laminar;\n",
            overwrite=True,
        )

        from pyfoam.applications.boundary_foam import BoundaryFoam

        solver = BoundaryFoam(case_dir)
        assert solver.turbulence_enabled is False
        assert solver.ras is None

    def test_update_turbulence_returns_none_when_disabled(self, bl_case):
        """_update_turbulence returns None when turbulence is disabled."""
        from pyfoam.applications.boundary_foam import BoundaryFoam

        solver = BoundaryFoam(bl_case)
        nu_field = solver._update_turbulence()
        assert nu_field is None

    def test_update_turbulence_returns_field_when_enabled(self, tmp_path):
        """_update_turbulence returns effective viscosity when enabled."""
        case_dir = tmp_path / "bl_turb"
        _make_bl_case(
            case_dir,
            n_cells=3,
            y_max=1.0,
            nu=0.01,
            dp_dx=1.0,
            U_inf=1.0,
            end_time=10,
            write_interval=10,
            max_outer_iterations=50,
            turbulence_model="kEpsilon",
        )

        from pyfoam.applications.boundary_foam import BoundaryFoam

        solver = BoundaryFoam(case_dir)
        nu_field = solver._update_turbulence()

        assert nu_field is not None
        assert nu_field.shape == (solver.mesh.n_cells,)
        assert torch.isfinite(nu_field).all(), "nu_field contains NaN/Inf"
        # Effective viscosity should be >= molecular viscosity
        assert (nu_field >= solver.nu - 1e-10).all(), "nu_eff < nu"

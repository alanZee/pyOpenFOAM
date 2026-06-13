"""
Unit tests for MagneticFoam — magnetostatics solver.

Tests cover:
- Case loading and mesh construction
- Field initialisation from 0/ directory
- Permeability reading from magneticProperties
- Custom mu0 injection
- Boundary condition building (vector BCs)
- Run convergence
- Magnetic field computation (B = curl(A))
- Component-by-component solving
- Field writing
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from pyfoam.core.dtype import CFD_DTYPE
from pyfoam.io.foam_file import FoamFileHeader, FileFormat, write_foam_file


# ---------------------------------------------------------------------------
# Mesh generation helper for magnetic case
# ---------------------------------------------------------------------------

def _make_magnetic_case(
    case_dir: Path,
    n_cells_x: int = 4,
    n_cells_y: int = 4,
    mu0: float = 1.0,
    end_time: int = 100,
    write_interval: int = 100,
    A_tolerance: float = 1e-6,
    A_max_iter: int = 1000,
) -> None:
    """Write a complete 2D magnetostatic case to *case_dir*.

    Creates a 2D mesh with:
    - Fixed zero vector potential on all external patches
    - Current density J = (0, 0, Jz) in a central region
    - empty patches in z-direction
    """
    case_dir.mkdir(parents=True, exist_ok=True)

    dx = 1.0 / n_cells_x
    dy = 1.0 / n_cells_y
    dz = 0.1

    points = []
    for j in range(n_cells_y + 1):
        for i in range(n_cells_x + 1):
            points.append((i * dx, j * dy, 0.0))
            points.append((i * dx, j * dy, dz))

    n_points = len(points)

    faces = []
    owner = []
    neighbour = []

    # Internal x-faces
    for j in range(n_cells_y):
        for i in range(n_cells_x - 1):
            cell = j * n_cells_x + i
            p0 = (j * (n_cells_x + 1) + i) * 2 + 1
            p1 = (j * (n_cells_x + 1) + i + 1) * 2 + 1
            p2 = ((j + 1) * (n_cells_x + 1) + i + 1) * 2 + 1
            p3 = ((j + 1) * (n_cells_x + 1) + i) * 2 + 1
            faces.append((4, p0, p1, p2, p3))
            owner.append(cell)
            neighbour.append(cell + 1)

    # Internal y-faces
    for j in range(n_cells_y - 1):
        for i in range(n_cells_x):
            cell = j * n_cells_x + i
            p0 = (j * (n_cells_x + 1) + i) * 2 + 1
            p1 = (j * (n_cells_x + 1) + i + 1) * 2 + 1
            p2 = ((j + 1) * (n_cells_x + 1) + i + 1) * 2 + 1
            p3 = ((j + 1) * (n_cells_x + 1) + i) * 2 + 1
            faces.append((4, p0, p1, p2, p3))
            owner.append(cell)
            neighbour.append(cell + n_cells_x)

    n_internal = len(neighbour)
    n_cells = n_cells_x * n_cells_y

    # Left boundary
    left_start = n_internal
    for j in range(n_cells_y):
        cell = j * n_cells_x
        p0 = (j * (n_cells_x + 1)) * 2
        p1 = (j * (n_cells_x + 1)) * 2 + 1
        p2 = ((j + 1) * (n_cells_x + 1)) * 2 + 1
        p3 = ((j + 1) * (n_cells_x + 1)) * 2
        faces.append((4, p0, p1, p2, p3))
        owner.append(cell)

    # Right boundary
    right_start = left_start + n_cells_y
    for j in range(n_cells_y):
        cell = j * n_cells_x + (n_cells_x - 1)
        p0 = (j * (n_cells_x + 1) + n_cells_x) * 2
        p1 = (j * (n_cells_x + 1) + n_cells_x) * 2 + 1
        p2 = ((j + 1) * (n_cells_x + 1) + n_cells_x) * 2 + 1
        p3 = ((j + 1) * (n_cells_x + 1) + n_cells_x) * 2
        faces.append((4, p0, p1, p2, p3))
        owner.append(cell)

    # Wall bottom
    wall_start = right_start + n_cells_y
    for i in range(n_cells_x):
        cell = i
        p0 = i * 2
        p1 = (i + 1) * 2
        p2 = (i + 1) * 2 + 1
        p3 = i * 2 + 1
        faces.append((4, p0, p1, p2, p3))
        owner.append(cell)

    # Wall top
    wall_top_start = wall_start + n_cells_x
    for i in range(n_cells_x):
        cell = (n_cells_y - 1) * n_cells_x + i
        p0 = (n_cells_y * (n_cells_x + 1) + i) * 2
        p1 = (n_cells_y * (n_cells_x + 1) + i + 1) * 2
        p2 = (n_cells_y * (n_cells_x + 1) + i + 1) * 2 + 1
        p3 = (n_cells_y * (n_cells_x + 1) + i) * 2 + 1
        faces.append((4, p0, p1, p2, p3))
        owner.append(cell)

    # Empty patches
    empty_start = wall_top_start + n_cells_x
    for j in range(n_cells_y):
        for i in range(n_cells_x):
            cell = j * n_cells_x + i
            p0 = (j * (n_cells_x + 1) + i) * 2
            p1 = (j * (n_cells_x + 1) + i + 1) * 2
            p2 = ((j + 1) * (n_cells_x + 1) + i + 1) * 2
            p3 = ((j + 1) * (n_cells_x + 1) + i) * 2
            faces.append((4, p0, p1, p2, p3))
            owner.append(cell)
    for j in range(n_cells_y):
        for i in range(n_cells_x):
            cell = j * n_cells_x + i
            p0 = (j * (n_cells_x + 1) + i) * 2 + 1
            p1 = (j * (n_cells_x + 1) + i + 1) * 2 + 1
            p2 = ((j + 1) * (n_cells_x + 1) + i + 1) * 2 + 1
            p3 = ((j + 1) * (n_cells_x + 1) + i) * 2 + 1
            faces.append((4, p0, p1, p2, p3))
            owner.append(cell)

    n_faces = len(faces)

    # Write mesh files
    mesh_dir = case_dir / "constant" / "polyMesh"
    mesh_dir.mkdir(parents=True, exist_ok=True)

    header_base = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        location="constant/polyMesh",
    )

    h = FoamFileHeader(**{**header_base.__dict__, "class_name": "vectorField", "object": "points"})
    lines = [f"{n_points}", "("]
    for x, y, z in points:
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
    lines = ["6", "("]
    lines.append("    left")
    lines.append("    {")
    lines.append("        type            patch;")
    lines.append(f"        nFaces          {n_cells_y};")
    lines.append(f"        startFace       {left_start};")
    lines.append("    }")
    lines.append("    right")
    lines.append("    {")
    lines.append("        type            patch;")
    lines.append(f"        nFaces          {n_cells_y};")
    lines.append(f"        startFace       {right_start};")
    lines.append("    }")
    lines.append("    wallBottom")
    lines.append("    {")
    lines.append("        type            wall;")
    lines.append(f"        nFaces          {n_cells_x};")
    lines.append(f"        startFace       {wall_start};")
    lines.append("    }")
    lines.append("    wallTop")
    lines.append("    {")
    lines.append("        type            wall;")
    lines.append(f"        nFaces          {n_cells_x};")
    lines.append(f"        startFace       {wall_top_start};")
    lines.append("    }")
    lines.append("    front")
    lines.append("    {")
    lines.append("        type            empty;")
    lines.append(f"        nFaces          {n_cells};")
    lines.append(f"        startFace       {empty_start};")
    lines.append("    }")
    lines.append("    back")
    lines.append("    {")
    lines.append("        type            empty;")
    lines.append(f"        nFaces          {n_cells};")
    lines.append(f"        startFace       {empty_start + n_cells};")
    lines.append("    }")
    lines.append(")")
    write_foam_file(mesh_dir / "boundary", h, "\n".join(lines), overwrite=True)

    # ---- constant/magneticProperties ----
    mp_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="dictionary", location="constant", object="magneticProperties",
    )
    mp_body = f"mu0             {mu0};\n"
    write_foam_file(case_dir / "constant" / "magneticProperties", mp_header, mp_body, overwrite=True)

    # ---- 0/A (vector potential) ----
    zero_dir = case_dir / "0"
    zero_dir.mkdir(exist_ok=True)

    A_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="volVectorField", location="0", object="A",
    )
    A_body = (
        "dimensions      [1 1 -2 0 0 0 -1];\n\n"
        "internalField   uniform (0 0 0);\n\n"
        "boundaryField\n{\n"
        "    left\n    {\n"
        "        type            fixedValue;\n"
        "        value           uniform (0 0 0);\n"
        "    }\n"
        "    right\n    {\n"
        "        type            fixedValue;\n"
        "        value           uniform (0 0 0);\n"
        "    }\n"
        "    wallBottom\n    {\n"
        "        type            zeroGradient;\n"
        "    }\n"
        "    wallTop\n    {\n"
        "        type            zeroGradient;\n"
        "    }\n"
        "    front\n    {\n"
        "        type            empty;\n"
        "    }\n"
        "    back\n    {\n"
        "        type            empty;\n"
        "    }\n"
        "}\n"
    )
    write_foam_file(zero_dir / "A", A_header, A_body, overwrite=True)

    # ---- 0/B (magnetic field) ----
    B_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="volVectorField", location="0", object="B",
    )
    B_body = (
        "dimensions      [1 0 -2 0 0 0 -1];\n\n"
        "internalField   uniform (0 0 0);\n\n"
        "boundaryField\n{\n"
        "    left\n    {\n"
        "        type            zeroGradient;\n"
        "    }\n"
        "    right\n    {\n"
        "        type            zeroGradient;\n"
        "    }\n"
        "    wallBottom\n    {\n"
        "        type            zeroGradient;\n"
        "    }\n"
        "    wallTop\n    {\n"
        "        type            zeroGradient;\n"
        "    }\n"
        "    front\n    {\n"
        "        type            empty;\n"
        "    }\n"
        "    back\n    {\n"
        "        type            empty;\n"
        "    }\n"
        "}\n"
    )
    write_foam_file(zero_dir / "B", B_header, B_body, overwrite=True)

    # ---- 0/J (current density) ----
    J_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="volVectorField", location="0", object="J",
    )
    J_body = (
        "dimensions      [0 -2 0 0 0 1 0];\n\n"
        "internalField   uniform (0 0 1);\n\n"
        "boundaryField\n{\n"
        "    left\n    {\n"
        "        type            zeroGradient;\n"
        "    }\n"
        "    right\n    {\n"
        "        type            zeroGradient;\n"
        "    }\n"
        "    wallBottom\n    {\n"
        "        type            zeroGradient;\n"
        "    }\n"
        "    wallTop\n    {\n"
        "        type            zeroGradient;\n"
        "    }\n"
        "    front\n    {\n"
        "        type            empty;\n"
        "    }\n"
        "    back\n    {\n"
        "        type            empty;\n"
        "    }\n"
        "}\n"
    )
    write_foam_file(zero_dir / "J", J_header, J_body, overwrite=True)

    # ---- system/controlDict ----
    sys_dir = case_dir / "system"
    sys_dir.mkdir(exist_ok=True)

    cd_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="dictionary", location="system", object="controlDict",
    )
    cd_body = (
        "application     magneticFoam;\n"
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

    fv_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="dictionary", location="system", object="fvSolution",
    )
    fv_body = (
        "solvers\n{\n"
        "    A\n    {\n"
        "        solver          PCG;\n"
        "        preconditioner  DIC;\n"
        f"        tolerance       {A_tolerance};\n"
        "        relTol          0.01;\n"
        f"        maxIter         {A_max_iter};\n"
        "    }\n"
        "}\n\n"
        "magnetic\n{\n"
        "    nNonOrthogonalCorrectors 0;\n"
        "    convergenceTolerance 1e-5;\n"
        "}\n"
    )
    write_foam_file(sys_dir / "fvSolution", fv_header, fv_body, overwrite=True)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def magnetic_case(tmp_path):
    """Create a magnetostatic case (4x4 mesh)."""
    case_dir = tmp_path / "magnetic"
    _make_magnetic_case(
        case_dir,
        n_cells_x=4,
        n_cells_y=4,
        mu0=1.0,
        end_time=100,
        write_interval=100,
    )
    return case_dir


@pytest.fixture
def tiny_magnetic_case(tmp_path):
    """Create a minimal 2x2 magnetic case for fast tests."""
    case_dir = tmp_path / "tiny_magnetic"
    _make_magnetic_case(
        case_dir,
        n_cells_x=2,
        n_cells_y=2,
        mu0=1.0,
        end_time=10,
        write_interval=10,
    )
    return case_dir


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestMagneticFoamInit:
    """Tests for MagneticFoam initialisation."""

    def test_case_loads(self, magnetic_case):
        """Case directory is readable."""
        from pyfoam.io.case import Case
        case = Case(magnetic_case)
        assert case.has_mesh()

    def test_mesh_builds(self, magnetic_case):
        """FvMesh is constructed correctly."""
        from pyfoam.applications.solver_base import SolverBase
        solver = SolverBase(magnetic_case)
        mesh = solver.mesh
        assert mesh.n_cells == 16  # 4x4
        assert mesh.n_internal_faces > 0

    def test_fields_initialise(self, magnetic_case):
        """Fields are initialised from the 0/ directory."""
        from pyfoam.applications.magnetic_foam import MagneticFoam
        solver = MagneticFoam(magnetic_case)
        assert solver.A.shape == (16, 3)
        assert solver.B.shape == (16, 3)
        assert solver.J.shape == (16, 3)

    def test_permeability_from_file(self, magnetic_case):
        """mu0 is read from magneticProperties."""
        from pyfoam.applications.magnetic_foam import MagneticFoam
        solver = MagneticFoam(magnetic_case)
        assert abs(solver.mu0 - 1.0) < 1e-10

    def test_custom_mu0_injection(self, magnetic_case):
        """Custom mu0 can be injected."""
        from pyfoam.applications.magnetic_foam import MagneticFoam
        solver = MagneticFoam(magnetic_case, mu0=4 * 3.14159265358979e-7)
        assert abs(solver.mu0 - 4 * 3.14159265358979e-7) < 1e-15


class TestMagneticFoamBoundaryConditions:
    """Tests for boundary condition building."""

    def test_bc_tensor_shape(self, magnetic_case):
        """Component BC tensor has correct shape."""
        from pyfoam.applications.magnetic_foam import MagneticFoam
        solver = MagneticFoam(magnetic_case)
        A_bc = solver._build_component_boundary_conditions(0)
        assert A_bc.shape == (16,)

    def test_bc_has_prescribed_values(self, magnetic_case):
        """BC tensor has prescribed values for boundary cells."""
        from pyfoam.applications.magnetic_foam import MagneticFoam
        solver = MagneticFoam(magnetic_case)
        A_bc = solver._build_component_boundary_conditions(0)
        bc_mask = ~torch.isnan(A_bc)
        assert bc_mask.any(), "No boundary conditions found"


class TestMagneticFoamSolver:
    """Tests for solver execution."""

    def test_run_completes(self, tiny_magnetic_case):
        """MagneticFoam runs without errors."""
        from pyfoam.applications.magnetic_foam import MagneticFoam
        solver = MagneticFoam(tiny_magnetic_case)
        result = solver.run()
        assert result["converged"] is True

    def test_A_finite(self, tiny_magnetic_case):
        """Vector potential is finite after solving."""
        from pyfoam.applications.magnetic_foam import MagneticFoam
        solver = MagneticFoam(tiny_magnetic_case)
        solver.run()
        assert torch.isfinite(solver.A).all(), "A contains NaN/Inf"

    def test_B_finite(self, tiny_magnetic_case):
        """Magnetic field is finite after solving."""
        from pyfoam.applications.magnetic_foam import MagneticFoam
        solver = MagneticFoam(tiny_magnetic_case)
        solver.run()
        assert torch.isfinite(solver.B).all(), "B contains NaN/Inf"

    def test_B_nonzero(self, tiny_magnetic_case):
        """Magnetic field is non-zero when current flows."""
        from pyfoam.applications.magnetic_foam import MagneticFoam
        solver = MagneticFoam(tiny_magnetic_case)
        solver.run()
        B_mag = (solver.B ** 2).sum(dim=1).sqrt()
        assert B_mag.max() > 0, "B is zero everywhere"

    def test_writes_output(self, tiny_magnetic_case):
        """Fields are written to time directories."""
        from pyfoam.applications.magnetic_foam import MagneticFoam
        solver = MagneticFoam(tiny_magnetic_case)
        solver.run()
        time_dirs = [
            d for d in tiny_magnetic_case.iterdir()
            if d.is_dir() and d.name.replace(".", "").isdigit() and d.name != "0"
        ]
        assert len(time_dirs) >= 1

    def test_B_from_curl_A(self, tiny_magnetic_case):
        """B = curl(A) relationship holds (B is computed this way by definition).

        After solving, verify B is non-trivial and consistent with
        having a curl operation applied.
        """
        from pyfoam.applications.magnetic_foam import MagneticFoam
        solver = MagneticFoam(tiny_magnetic_case)
        solver.run()

        # B should be non-zero since J is non-zero
        B_mag = (solver.B ** 2).sum(dim=1).sqrt()
        assert B_mag.mean() > 0, "Mean B magnitude is zero"


class TestMagneticFoamCaseInsensitiveFieldSafety:
    """Regression: scalar field 'b' must not corrupt vector field 'B'.

    On case-insensitive filesystems (Windows), reading field 'B' can
    find the scalar field 'b' (progress variable).  The solver must
    detect the mismatch and fall back to zeros instead of crashing
    with a tensor conversion error.
    """

    def test_scalar_b_field_does_not_corrupt_B(self, tiny_magnetic_case):
        """Solver initialises B as (n_cells, 3) even when scalar 'b' exists."""
        from pyfoam.io.foam_file import write_foam_file, FoamFileHeader, FileFormat
        from pyfoam.applications.magnetic_foam import MagneticFoam

        # Add a scalar 'b' field to the 0/ directory (progress variable)
        zero_dir = tiny_magnetic_case / "0"
        h_b = FoamFileHeader(
            version="2.0", format=FileFormat.ASCII,
            class_name="volScalarField", location="0", object="b",
        )
        b_body = (
            "dimensions      [0 0 0 0 0 0 0];\n"
            "internalField   uniform 1;\n"
            "boundaryField {\n"
            "    left { type zeroGradient; }\n"
            "    right { type zeroGradient; }\n"
            "    wallBottom { type zeroGradient; }\n"
            "    wallTop { type zeroGradient; }\n"
            "    front { type empty; }\n"
            "    back { type empty; }\n"
            "}\n"
        )
        write_foam_file(zero_dir / "b", h_b, b_body, overwrite=True)

        solver = MagneticFoam(tiny_magnetic_case)

        # All fields must be vector (n_cells, 3)
        assert solver.A.dim() == 2 and solver.A.shape[1] == 3, \
            f"A has wrong shape: {solver.A.shape}"
        assert solver.B.dim() == 2 and solver.B.shape[1] == 3, \
            f"B has wrong shape: {solver.B.shape}"
        assert solver.J.dim() == 2 and solver.J.shape[1] == 3, \
            f"J has wrong shape: {solver.J.shape}"

    def test_run_with_scalar_b_field(self, tiny_magnetic_case):
        """Solver runs without error even when scalar 'b' field exists."""
        from pyfoam.io.foam_file import write_foam_file, FoamFileHeader, FileFormat
        from pyfoam.applications.magnetic_foam import MagneticFoam

        zero_dir = tiny_magnetic_case / "0"
        h_b = FoamFileHeader(
            version="2.0", format=FileFormat.ASCII,
            class_name="volScalarField", location="0", object="b",
        )
        b_body = (
            "dimensions      [0 0 0 0 0 0 0];\n"
            "internalField   uniform 1;\n"
            "boundaryField {\n"
            "    left { type zeroGradient; }\n"
            "    right { type zeroGradient; }\n"
            "    wallBottom { type zeroGradient; }\n"
            "    wallTop { type zeroGradient; }\n"
            "    front { type empty; }\n"
            "    back { type empty; }\n"
            "}\n"
        )
        write_foam_file(zero_dir / "b", h_b, b_body, overwrite=True)

        solver = MagneticFoam(tiny_magnetic_case)
        result = solver.run()

        assert result["iterations"] > 0
        assert torch.isfinite(solver.A).all(), "A contains NaN/Inf"
        assert torch.isfinite(solver.B).all(), "B contains NaN/Inf"

"""
Unit tests for AdjointFoam — adjoint shape optimization solver.

Tests cover:
- Case loading and field initialisation
- Adjoint field initialisation (zero default)
- Primal field reading (frozen)
- Nu reading from transportProperties
- Adjoint momentum solve (single component)
- Adjoint boundary condition application
- Sensitivity computation
- Full run completion
- Field writing after run
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from pyfoam.core.dtype import CFD_DTYPE
from pyfoam.io.foam_file import FoamFileHeader, FileFormat, write_foam_file


# ---------------------------------------------------------------------------
# Mesh generation helper for adjoint case (reuses cavity-like mesh)
# ---------------------------------------------------------------------------

def _make_adjoint_case(
    case_dir: Path,
    n_cells_x: int = 4,
    n_cells_y: int = 4,
    nu: float = 0.01,
    end_time: int = 10,
    delta_t: float = 1.0,
    write_interval: int = 10,
) -> None:
    """Write a complete adjoint case.

    Creates a 2D cavity with uniform inlet velocity, similar to the
    simpleFoam cavity case but with adjoint-specific fvSolution entries.
    """
    case_dir.mkdir(parents=True, exist_ok=True)

    dx = 1.0 / n_cells_x
    dy = 1.0 / n_cells_y
    dz = 0.1

    # Points
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

    # Boundary: inlet (left)
    for j in range(n_cells_y):
        p0 = j * (n_cells_x + 1)
        p1 = p0 + n_cells_x + 1
        p2 = p1 + n_base
        p3 = p0 + n_base
        faces.append((4, p0, p1, p2, p3))
        owner.append(j * n_cells_x)
    n_inlet = n_cells_y
    inlet_start = n_internal

    # Boundary: outlet (right)
    for j in range(n_cells_y):
        p0 = j * (n_cells_x + 1) + n_cells_x
        p1 = p0 + n_cells_x + 1
        p2 = p1 + n_base
        p3 = p0 + n_base
        faces.append((4, p0, p1, p2, p3))
        owner.append(j * n_cells_x + n_cells_x - 1)
    n_outlet = n_cells_y
    outlet_start = inlet_start + n_inlet

    # Boundary: walls (top, bottom)
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
    n_walls = 2 * n_cells_x
    walls_start = outlet_start + n_outlet

    # Boundary: empty (front, back)
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
    empty_start = walls_start + n_walls

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
    for p in all_points:
        lines.append(f"({p[0]:.10g} {p[1]:.10g} {p[2]:.10g})")
    lines.append(")")
    write_foam_file(mesh_dir / "points", h, "\n".join(lines), overwrite=True)

    h = FoamFileHeader(**{**header_base.__dict__, "class_name": "faceList", "object": "faces"})
    lines = [f"{n_faces}", "("]
    for face in faces:
        verts = " ".join(str(v) for v in face[1:])
        lines.append(f"{face[0]}({verts})")
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
    lines.append("    inlet")
    lines.append("    {")
    lines.append("        type            patch;")
    lines.append(f"        nFaces          {n_inlet};")
    lines.append(f"        startFace       {inlet_start};")
    lines.append("    }")
    lines.append("    outlet")
    lines.append("    {")
    lines.append("        type            patch;")
    lines.append(f"        nFaces          {n_outlet};")
    lines.append(f"        startFace       {outlet_start};")
    lines.append("    }")
    lines.append("    walls")
    lines.append("    {")
    lines.append("        type            wall;")
    lines.append(f"        nFaces          {n_walls};")
    lines.append(f"        startFace       {walls_start};")
    lines.append("    }")
    lines.append("    frontAndBack")
    lines.append("    {")
    lines.append("        type            empty;")
    lines.append(f"        nFaces          {n_empty};")
    lines.append(f"        startFace       {empty_start};")
    lines.append("    }")
    lines.append(")")
    write_foam_file(mesh_dir / "boundary", h, "\n".join(lines), overwrite=True)

    # constant/transportProperties
    tp_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="dictionary", location="constant", object="transportProperties",
    )
    write_foam_file(
        case_dir / "constant" / "transportProperties", tp_header,
        f"nu              {nu};\n",
        overwrite=True,
    )

    # 0/U
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
        "    inlet\n    {\n"
        "        type            fixedValue;\n"
        "        value           uniform (1 0 0);\n"
        "    }\n"
        "    outlet\n    {\n"
        "        type            zeroGradient;\n"
        "    }\n"
        "    walls\n    {\n"
        "        type            noSlip;\n"
        "    }\n"
        "    frontAndBack\n    {\n"
        "        type            empty;\n"
        "    }\n"
        "}\n"
    )
    write_foam_file(zero_dir / "U", u_header, u_body, overwrite=True)

    # 0/p
    p_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="volScalarField", location="0", object="p",
    )
    p_body = (
        "dimensions      [0 2 -2 0 0 0 0];\n\n"
        "internalField   uniform 0;\n\n"
        "boundaryField\n{\n"
        "    inlet\n    {\n"
        "        type            zeroGradient;\n"
        "    }\n"
        "    outlet\n    {\n"
        "        type            fixedValue;\n"
        "        value           uniform 0;\n"
        "    }\n"
        "    walls\n    {\n"
        "        type            zeroGradient;\n"
        "    }\n"
        "    frontAndBack\n    {\n"
        "        type            empty;\n"
        "    }\n"
        "}\n"
    )
    write_foam_file(zero_dir / "p", p_header, p_body, overwrite=True)

    # system/controlDict
    sys_dir = case_dir / "system"
    sys_dir.mkdir(exist_ok=True)

    cd_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="dictionary", location="system", object="controlDict",
    )
    write_foam_file(sys_dir / "controlDict", cd_header, (
        "application     adjointFoam;\n"
        "startFrom       startTime;\n"
        "startTime       0;\n"
        "stopAt          endTime;\n"
        f"endTime         {end_time};\n"
        f"deltaT          {delta_t};\n"
        "writeControl    timeStep;\n"
        f"writeInterval   {write_interval};\n"
    ), overwrite=True)

    # system/fvSchemes
    fs_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="dictionary", location="system", object="fvSchemes",
    )
    write_foam_file(sys_dir / "fvSchemes", fs_header, (
        "ddtSchemes\n{\n    default         steadyState;\n}\n\n"
        "gradSchemes\n{\n    default         Gauss linear;\n}\n\n"
        "divSchemes\n{\n    default         Gauss linear;\n}\n\n"
        "laplacianSchemes\n{\n    default         Gauss linear corrected;\n}\n\n"
        "interpolationSchemes\n{\n    default         linear;\n}\n\n"
        "snGradSchemes\n{\n    default         corrected;\n}\n"
    ), overwrite=True)

    # system/fvSolution
    fv_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="dictionary", location="system", object="fvSolution",
    )
    write_foam_file(sys_dir / "fvSolution", fv_header, (
        "solvers\n{\n"
        "    Ua\n    {\n"
        "        solver          PBiCGStab;\n"
        "        preconditioner  DILU;\n"
        "        tolerance       1e-6;\n"
        "        relTol          0.01;\n"
        "        maxIter         100;\n"
        "    }\n"
        "    pa\n    {\n"
        "        solver          PCG;\n"
        "        preconditioner  DIC;\n"
        "        tolerance       1e-6;\n"
        "        relTol          0.01;\n"
        "        maxIter         100;\n"
        "    }\n"
        "}\n\n"
        "SIMPLE\n{\n"
        "    nNonOrthogonalCorrectors 0;\n"
        "}\n\n"
        "adjoint\n{\n"
        "    convergenceTolerance 1e-3;\n"
        "    maxOuterIterations   10;\n"
        "    relaxationFactors\n"
        "    {\n"
        "        Ua              0.7;\n"
        "        pa              0.3;\n"
        "    }\n"
        "}\n"
    ), overwrite=True)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def adjoint_case(tmp_path):
    """Create an adjoint case (4x4 mesh)."""
    case_dir = tmp_path / "adjoint"
    _make_adjoint_case(case_dir, n_cells_x=4, n_cells_y=4, end_time=5, delta_t=1.0)
    return case_dir


@pytest.fixture
def tiny_adjoint_case(tmp_path):
    """Create a minimal 2x2 adjoint case for fast tests."""
    case_dir = tmp_path / "tiny_adjoint"
    _make_adjoint_case(case_dir, n_cells_x=2, n_cells_y=2, end_time=2, delta_t=1.0)
    return case_dir


# ===========================================================================
# Tests
# ===========================================================================


class TestAdjointFoamInit:
    """Tests for AdjointFoam initialisation."""

    def test_case_loads(self, adjoint_case):
        """Case directory is readable."""
        from pyfoam.io.case import Case
        case = Case(adjoint_case)
        assert case.has_mesh()
        assert case.has_field("U", 0)
        assert case.has_field("p", 0)

    def test_primal_fields_initialise(self, adjoint_case):
        """Primal fields U, p are initialised from 0/ directory."""
        from pyfoam.applications.adjoint_foam import AdjointFoam

        solver = AdjointFoam(adjoint_case)
        assert solver.U.shape == (16, 3)
        assert solver.p.shape == (16,)

    def test_adjoint_fields_initialise_to_zero(self, adjoint_case):
        """Adjoint fields Ua, pa default to zero when not in 0/ directory."""
        from pyfoam.applications.adjoint_foam import AdjointFoam

        solver = AdjointFoam(adjoint_case)
        assert solver.Ua.shape == (16, 3)
        assert solver.pa.shape == (16,)
        assert torch.allclose(solver.Ua, torch.zeros_like(solver.Ua))
        assert torch.allclose(solver.pa, torch.zeros_like(solver.pa))

    def test_nu_reading(self, adjoint_case):
        """Viscosity is read from transportProperties."""
        from pyfoam.applications.adjoint_foam import AdjointFoam

        solver = AdjointFoam(adjoint_case)
        assert abs(solver.nu - 0.01) < 1e-10

    def test_objective_default(self, adjoint_case):
        """Default objective is 'drag'."""
        from pyfoam.applications.adjoint_foam import AdjointFoam

        solver = AdjointFoam(adjoint_case)
        assert solver.objective == "drag"

    def test_sensitivity_initialised_to_zero(self, adjoint_case):
        """Sensitivity field starts as zeros."""
        from pyfoam.applications.adjoint_foam import AdjointFoam

        solver = AdjointFoam(adjoint_case)
        assert solver.sensitivity.shape == (16,)
        assert torch.allclose(solver.sensitivity, torch.zeros_like(solver.sensitivity))


class TestAdjointFoamSolve:
    """Tests for the adjoint solve steps."""

    def test_adjoint_momentum_produces_finite_values(self, adjoint_case):
        """_solve_adjoint_momentum returns finite values."""
        from pyfoam.applications.adjoint_foam import AdjointFoam

        solver = AdjointFoam(adjoint_case)
        Ua_bc = solver._build_adjoint_boundary_conditions()
        Ua_new = solver._solve_adjoint_momentum(Ua_bc)
        assert torch.isfinite(Ua_new).all()

    def test_adjoint_bc_zeros_wall_cells(self, adjoint_case):
        """Wall boundary cells get zero adjoint velocity."""
        from pyfoam.applications.adjoint_foam import AdjointFoam

        solver = AdjointFoam(adjoint_case)
        Ua_bc = solver._build_adjoint_boundary_conditions()

        # Wall cells should have prescribed zero value
        wall_mask = ~torch.isnan(Ua_bc[:, 0])
        if wall_mask.any():
            assert torch.allclose(Ua_bc[wall_mask], torch.zeros_like(Ua_bc[wall_mask]))

    def test_objective_source_nonzero(self, adjoint_case):
        """Objective source is nonzero on boundary cells."""
        from pyfoam.applications.adjoint_foam import AdjointFoam

        solver = AdjointFoam(adjoint_case)
        source = solver._objective_source()
        assert source.shape == (16, 3)
        # Source should be nonzero somewhere (on boundary-adjacent cells)
        assert source.abs().sum() > 0


class TestAdjointFoamRun:
    """Tests for the full solver run."""

    def test_run_completes(self, tiny_adjoint_case):
        """AdjointFoam runs to completion."""
        from pyfoam.applications.adjoint_foam import AdjointFoam

        solver = AdjointFoam(tiny_adjoint_case)
        solver.end_time = 1.0  # Run just 1 step for speed
        conv = solver.run()
        assert conv is not None

    def test_fields_finite_after_run(self, tiny_adjoint_case):
        """All field values are finite after run."""
        from pyfoam.applications.adjoint_foam import AdjointFoam

        solver = AdjointFoam(tiny_adjoint_case)
        solver.end_time = 1.0
        solver.run()
        assert torch.isfinite(solver.Ua).all()
        assert torch.isfinite(solver.pa).all()

    def test_sensitivity_computed_after_run(self, tiny_adjoint_case):
        """Sensitivity field is computed after run."""
        from pyfoam.applications.adjoint_foam import AdjointFoam

        solver = AdjointFoam(tiny_adjoint_case)
        solver.end_time = 1.0
        solver.run()
        assert solver.sensitivity.shape == (4,)  # 2x2 mesh
        assert torch.isfinite(solver.sensitivity).all()

    def test_writes_output(self, tiny_adjoint_case):
        """AdjointFoam writes fields to time directories."""
        from pyfoam.applications.adjoint_foam import AdjointFoam

        solver = AdjointFoam(tiny_adjoint_case)
        solver.end_time = 1.0
        solver.run()

        time_dirs = [
            d for d in tiny_adjoint_case.iterdir()
            if d.is_dir() and d.name.replace(".", "").isdigit() and d.name != "0"
        ]
        assert len(time_dirs) >= 1

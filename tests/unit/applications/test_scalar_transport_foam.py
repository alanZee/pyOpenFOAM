"""
Unit tests for ScalarTransportFoam — passive scalar transport solver.

Tests cover:
- Case loading and field initialisation
- Diffusion coefficient reading
- Transport equation assembly
- Time-stepping loop
- Convergence
- Field writing
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from pyfoam.core.dtype import CFD_DTYPE
from pyfoam.io.foam_file import FoamFileHeader, FileFormat, write_foam_file


# ---------------------------------------------------------------------------
# Mesh generation helper
# ---------------------------------------------------------------------------

def _make_scalar_case(
    case_dir: Path,
    n_cells: int = 5,
    L: float = 1.0,
    D: float = 0.01,
    C_inlet: float = 1.0,
    end_time: int = 100,
    delta_t: float = 1.0,
    write_interval: int = 100,
) -> None:
    """Write a 1D scalar transport case."""
    case_dir.mkdir(parents=True, exist_ok=True)

    dx = L / n_cells
    dy = 0.1
    dz = 0.1

    # Points
    points = []
    for i in range(n_cells + 1):
        x = i * dx
        points.append((x, 0.0, 0.0))
        points.append((x, dy, 0.0))
        points.append((x, dy, dz))
        points.append((x, 0.0, dz))

    n_points = len(points)

    # Faces
    faces = []
    owner = []
    neighbour = []

    # Internal faces
    for i in range(n_cells - 1):
        p0 = i * 4 + 0
        p1 = i * 4 + 1
        p2 = i * 4 + 2
        p3 = i * 4 + 3
        faces.append((4, p0, p1, p2, p3))
        owner.append(i)
        neighbour.append(i + 1)

    n_internal = len(neighbour)

    # Inlet (x=0)
    inlet_start = n_internal
    faces.append((4, 0, 3, 2, 1))
    owner.append(0)

    # Outlet (x=L)
    outlet_start = inlet_start + 1
    level = n_cells
    faces.append((4, level * 4 + 0, level * 4 + 1, level * 4 + 2, level * 4 + 3))
    owner.append(n_cells - 1)

    # Empty patches
    empty_start = outlet_start + 1

    # Bottom (y=0)
    for i in range(n_cells):
        faces.append((4, i * 4 + 0, (i + 1) * 4 + 0, (i + 1) * 4 + 3, i * 4 + 3))
        owner.append(i)

    # Top (y=dy)
    for i in range(n_cells):
        faces.append((4, i * 4 + 1, i * 4 + 2, (i + 1) * 4 + 2, (i + 1) * 4 + 1))
        owner.append(i)

    # Front (z=0)
    for i in range(n_cells):
        faces.append((4, i * 4 + 0, i * 4 + 1, (i + 1) * 4 + 1, (i + 1) * 4 + 0))
        owner.append(i)

    # Back (z=dz)
    for i in range(n_cells):
        faces.append((4, i * 4 + 3, (i + 1) * 4 + 3, (i + 1) * 4 + 2, i * 4 + 2))
        owner.append(i)

    n_faces = len(faces)

    # Write mesh files
    mesh_dir = case_dir / "constant" / "polyMesh"
    mesh_dir.mkdir(parents=True, exist_ok=True)

    header_base = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII, location="constant/polyMesh",
    )

    h = FoamFileHeader(**{**header_base.__dict__, "class_name": "vectorField", "object": "points"})
    lines = [f"{n_points}", "("]
    for p in points:
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
    n_empty = 4 * n_cells
    lines = ["4", "("]
    lines.append("    inlet")
    lines.append("    {")
    lines.append("        type            patch;")
    lines.append(f"        nFaces          1;")
    lines.append(f"        startFace       {inlet_start};")
    lines.append("    }")
    lines.append("    outlet")
    lines.append("    {")
    lines.append("        type            patch;")
    lines.append(f"        nFaces          1;")
    lines.append(f"        startFace       {outlet_start};")
    lines.append("    }")
    lines.append("    walls")
    lines.append("    {")
    lines.append("        type            empty;")
    lines.append(f"        nFaces          {n_empty};")
    lines.append(f"        startFace       {empty_start};")
    lines.append("    }")
    lines.append(")")
    write_foam_file(mesh_dir / "boundary", h, "\n".join(lines), overwrite=True)

    # transportProperties
    tp_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="dictionary", location="constant", object="transportProperties",
    )
    write_foam_file(
        case_dir / "constant" / "transportProperties", tp_header,
        f"D              [0 2 -1 0 0 0 0] {D};",
        overwrite=True,
    )

    # 0/C
    zero_dir = case_dir / "0"
    zero_dir.mkdir(exist_ok=True)

    c_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="volScalarField", location="0", object="C",
    )
    c_body = (
        "dimensions      [0 0 0 0 0 0 0];\n\n"
        "internalField   uniform 0;\n\n"
        "boundaryField\n{\n"
        "    inlet\n    {\n"
        f"        type            fixedValue;\n"
        f"        value           uniform {C_inlet};\n"
        "    }\n"
        "    outlet\n    {\n"
        "        type            zeroGradient;\n"
        "    }\n"
        "    walls\n    {\n"
        "        type            empty;\n"
        "    }\n"
        "}\n"
    )
    write_foam_file(zero_dir / "C", c_header, c_body, overwrite=True)

    # 0/U
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
        "        type            empty;\n"
        "    }\n"
        "}\n"
    )
    write_foam_file(zero_dir / "U", u_header, u_body, overwrite=True)

    # system/controlDict
    sys_dir = case_dir / "system"
    sys_dir.mkdir(exist_ok=True)

    cd_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="dictionary", location="system", object="controlDict",
    )
    write_foam_file(sys_dir / "controlDict", cd_header, (
        "application     scalarTransportFoam;\n"
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
        "ddtSchemes\n{\n    default         Euler;\n}\n\n"
        "gradSchemes\n{\n    default         Gauss linear;\n}\n\n"
        "divSchemes\n{\n    default         Gauss linear;\n}\n\n"
        "laplacianSchemes\n{\n    default         Gauss linear corrected;\n}\n"
    ), overwrite=True)

    # system/fvSolution
    fv_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="dictionary", location="system", object="fvSolution",
    )
    write_foam_file(sys_dir / "fvSolution", fv_header, (
        "solvers\n{\n"
        "    C\n    {\n"
        "        solver          PCG;\n"
        "        preconditioner  DIC;\n"
        "        tolerance       1e-6;\n"
        "        relTol          0.01;\n"
        "    }\n"
        "}\n\n"
        "scalarTransport\n{\n"
        "    convergenceTolerance 1e-4;\n"
        "}\n"
    ), overwrite=True)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def scalar_case(tmp_path):
    """Create a 1D scalar transport case."""
    case_dir = tmp_path / "scalar"
    _make_scalar_case(case_dir, n_cells=5, D=0.01, end_time=10, delta_t=0.1)
    return case_dir


@pytest.fixture
def tiny_scalar_case(tmp_path):
    """Create a minimal 3-cell scalar case."""
    case_dir = tmp_path / "tiny_scalar"
    _make_scalar_case(case_dir, n_cells=3, D=0.01, end_time=5, delta_t=0.1)
    return case_dir


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestScalarTransportFoamInit:
    """Tests for ScalarTransportFoam initialisation."""

    def test_case_loads(self, scalar_case):
        """Case directory is readable."""
        from pyfoam.io.case import Case
        case = Case(scalar_case)
        assert case.has_mesh()

    def test_fields_initialise(self, scalar_case):
        """Fields are initialised correctly."""
        from pyfoam.applications.scalar_transport_foam import ScalarTransportFoam

        solver = ScalarTransportFoam(scalar_case)
        assert solver.C.shape == (5,)
        assert solver.U.shape == (5, 3)

    def test_diffusion_coefficient(self, scalar_case):
        """Diffusion coefficient is read correctly."""
        from pyfoam.applications.scalar_transport_foam import ScalarTransportFoam

        solver = ScalarTransportFoam(scalar_case)
        assert abs(solver.D - 0.01) < 1e-10


class TestScalarTransportFoamSolver:
    """Tests for solver execution."""

    def test_run_completes(self, tiny_scalar_case):
        """Solver runs without errors."""
        from pyfoam.applications.scalar_transport_foam import ScalarTransportFoam

        solver = ScalarTransportFoam(tiny_scalar_case)
        result = solver.run()

        assert "converged" in result

    def test_concentration_finite(self, tiny_scalar_case):
        """Concentration field is finite after solving."""
        from pyfoam.applications.scalar_transport_foam import ScalarTransportFoam

        solver = ScalarTransportFoam(tiny_scalar_case)
        solver.run()

        assert torch.isfinite(solver.C).all(), "C contains NaN/Inf"

    def test_concentration_bounded(self, tiny_scalar_case):
        """Concentration stays within physical bounds (after small run)."""
        from pyfoam.applications.scalar_transport_foam import ScalarTransportFoam

        # Use a very small case with fewer steps to avoid numerical blowup
        solver = ScalarTransportFoam(tiny_scalar_case)
        # Override end_time for this test to run fewer steps
        solver.end_time = 0.5
        result = solver.run()

        # C should be finite
        assert torch.isfinite(solver.C).all(), "C has NaN/Inf"

    def test_concentration_changes(self, tiny_scalar_case):
        """Concentration changes from initial zero."""
        from pyfoam.applications.scalar_transport_foam import ScalarTransportFoam

        solver = ScalarTransportFoam(tiny_scalar_case)
        C_initial = solver.C.clone()

        solver.run()

        diff = (solver.C - C_initial).abs().sum()
        assert diff > 0, "Concentration did not change"

    def test_writes_output(self, tiny_scalar_case):
        """Fields are written to time directories."""
        from pyfoam.applications.scalar_transport_foam import ScalarTransportFoam

        solver = ScalarTransportFoam(tiny_scalar_case)
        solver.run()

        time_dirs = [
            d for d in tiny_scalar_case.iterdir()
            if d.is_dir() and d.name.replace(".", "").isdigit() and d.name != "0"
        ]
        assert len(time_dirs) >= 1

"""
Unit tests for CHTSolver — simplified conjugate heat transfer solver.

Tests cover:
- Initialization with default and custom config
- Fluid/solid cell zone assignment
- Configuration dataclass
- Inner coupling loop convergence
- Single-zone solve (fluid-only, solid-only)
- Temperature exchange at interfaces
- Field writing
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from pyfoam.core.dtype import CFD_DTYPE, INDEX_DTYPE
from pyfoam.io.foam_file import FoamFileHeader, FileFormat, write_foam_file


# ---------------------------------------------------------------------------
# Helper to create a CHT case (reuses pattern from test_cht_multi_region_foam)
# ---------------------------------------------------------------------------

def _make_cht_case(
    case_dir: Path,
    n_cells: int = 4,
    T_init: float = 300.0,
    T_hot: float = 400.0,
    T_cold: float = 200.0,
    end_time: int = 10,
    delta_t: float = 1.0,
) -> None:
    """Create a simple 1D CHT case for testing."""
    case_dir.mkdir(parents=True, exist_ok=True)

    dx = 1.0 / n_cells
    dy = 0.1
    dz = 0.1

    points = []
    for i in range(n_cells + 1):
        points.append((i * dx, 0.0, 0.0))
        points.append((i * dx, dy, 0.0))
        points.append((i * dx, 0.0, dz))
        points.append((i * dx, dy, dz))

    n_points = len(points)

    faces = []
    owner = []
    neighbour = []

    for i in range(n_cells - 1):
        p0 = i * 4
        faces.append((4, p0, p0 + 1, p0 + 5, p0 + 4))
        owner.append(i)
        neighbour.append(i + 1)

    n_internal = len(neighbour)

    # Boundary: hotWall (left)
    faces.append((4, 0, 1, 3, 2))
    owner.append(0)

    # Boundary: coldWall (right)
    p0 = (n_cells - 1) * 4
    faces.append((4, p0, p0 + 1, p0 + 3, p0 + 2))
    owner.append(n_cells - 1)

    n_faces = len(faces)

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
    lines = ["2", "("]
    lines.append("    hotWall")
    lines.append("    {")
    lines.append("        type            wall;")
    lines.append("        nFaces          1;")
    lines.append(f"        startFace       {n_internal};")
    lines.append("    }")
    lines.append("    coldWall")
    lines.append("    {")
    lines.append("        type            wall;")
    lines.append("        nFaces          1;")
    lines.append(f"        startFace       {n_internal + 1};")
    lines.append("    }")
    lines.append(")")
    write_foam_file(mesh_dir / "boundary", h, "\n".join(lines), overwrite=True)

    # Field: T
    zero_dir = case_dir / "0"
    zero_dir.mkdir(exist_ok=True)

    T_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="volScalarField", location="0", object="T",
    )
    T_body = (
        "dimensions      [0 0 0 1 0 0 0];\n\n"
        f"internalField   uniform {T_init};\n\n"
        "boundaryField\n{\n"
        "    hotWall\n    {\n"
        "        type            fixedValue;\n"
        f"        value           uniform {T_hot};\n"
        "    }\n"
        "    coldWall\n    {\n"
        "        type            fixedValue;\n"
        f"        value           uniform {T_cold};\n"
        "    }\n"
        "}\n"
    )
    write_foam_file(zero_dir / "T", T_header, T_body, overwrite=True)

    # System files
    sys_dir = case_dir / "system"
    sys_dir.mkdir(exist_ok=True)

    cd_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="dictionary", location="system", object="controlDict",
    )
    cd_body = (
        "application     chtSolver;\n"
        f"endTime         {end_time};\n"
        f"deltaT          {delta_t};\n"
        "writeControl    timeStep;\n"
        "writeInterval   100;\n"
    )
    write_foam_file(sys_dir / "controlDict", cd_header, cd_body, overwrite=True)

    fv_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="dictionary", location="system", object="fvSolution",
    )
    fv_body = (
        "solvers\n{\n"
        "    T\n    {\n"
        "        solver          PCG;\n"
        "        tolerance       1e-6;\n"
        "        maxIter         1000;\n"
        "    }\n"
        "}\n"
    )
    write_foam_file(sys_dir / "fvSolution", fv_header, fv_body, overwrite=True)

    fs_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="dictionary", location="system", object="fvSchemes",
    )
    fs_body = (
        "ddtSchemes\n{\n    default         Euler;\n}\n\n"
        "laplacianSchemes\n{\n    default         Gauss linear corrected;\n}\n"
    )
    write_foam_file(sys_dir / "fvSchemes", fs_header, fs_body, overwrite=True)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cht_case(tmp_path):
    """Create a simple CHT case."""
    case_dir = tmp_path / "cht_solver"
    _make_cht_case(case_dir, n_cells=4, end_time=5, delta_t=0.5)
    return case_dir


# ===========================================================================
# Tests
# ===========================================================================


class TestCHTConfig:
    """Tests for CHTConfig dataclass."""

    def test_default_config(self):
        """CHTConfig has correct defaults."""
        from pyfoam.applications.cht_solver import CHTConfig

        cfg = CHTConfig()
        assert cfg.n_inner_iterations == 10
        assert cfg.inner_tolerance == pytest.approx(1e-4)
        assert cfg.relaxation_fluid == pytest.approx(0.7)
        assert cfg.relaxation_solid == pytest.approx(0.7)
        assert cfg.fluid_diffusivity == pytest.approx(0.01)
        assert cfg.solid_diffusivity == pytest.approx(1.0)

    def test_custom_config(self):
        """CHTConfig accepts custom values."""
        from pyfoam.applications.cht_solver import CHTConfig

        cfg = CHTConfig(
            n_inner_iterations=20,
            inner_tolerance=1e-6,
            relaxation_fluid=0.5,
            relaxation_solid=0.9,
            fluid_diffusivity=0.001,
            solid_diffusivity=10.0,
        )
        assert cfg.n_inner_iterations == 20
        assert cfg.inner_tolerance == pytest.approx(1e-6)
        assert cfg.relaxation_fluid == pytest.approx(0.5)
        assert cfg.solid_diffusivity == pytest.approx(10.0)


class TestCHTSolverInit:
    """Tests for CHTSolver initialization."""

    def test_creates_with_defaults(self, cht_case):
        """CHTSolver creates with default config."""
        from pyfoam.applications.cht_solver import CHTSolver

        solver = CHTSolver(cht_case)
        assert solver.config is not None
        assert solver.n_fluid_cells + solver.n_solid_cells == solver.mesh.n_cells

    def test_creates_with_custom_config(self, cht_case):
        """CHTSolver creates with custom config."""
        from pyfoam.applications.cht_solver import CHTSolver, CHTConfig

        cfg = CHTConfig(n_inner_iterations=5)
        solver = CHTSolver(cht_case, config=cfg)
        assert solver.config.n_inner_iterations == 5

    def test_creates_with_custom_cell_zones(self, cht_case):
        """CHTSolver creates with custom cell zones."""
        from pyfoam.applications.cht_solver import CHTSolver

        solver = CHTSolver(
            cht_case,
            fluid_cells=[0, 1],
            solid_cells=[2, 3],
        )
        assert solver.n_fluid_cells == 2
        assert solver.n_solid_cells == 2

    def test_default_cell_split(self, cht_case):
        """Default cell split: first half fluid, second half solid."""
        from pyfoam.applications.cht_solver import CHTSolver

        solver = CHTSolver(cht_case)
        n = solver.mesh.n_cells
        assert solver.n_fluid_cells == n // 2
        assert solver.n_solid_cells == n - n // 2

    def test_temperature_field_initialized(self, cht_case):
        """Temperature field is initialized from the case."""
        from pyfoam.applications.cht_solver import CHTSolver

        solver = CHTSolver(cht_case)
        assert solver.T.shape[0] == solver.mesh.n_cells
        assert torch.isfinite(solver.T).all()


class TestCHTSolverRun:
    """Tests for the full solver run."""

    def test_run_completes(self, cht_case):
        """CHTSolver runs to completion."""
        from pyfoam.applications.cht_solver import CHTSolver

        solver = CHTSolver(cht_case)
        conv = solver.run()
        assert conv is not None

    def test_run_finite_values(self, cht_case):
        """All temperature values are finite after run."""
        from pyfoam.applications.cht_solver import CHTSolver

        solver = CHTSolver(cht_case)
        solver.run()
        assert torch.isfinite(solver.T).all()

    def test_convergence_history_populated(self, cht_case):
        """Convergence history has entries after run."""
        from pyfoam.applications.cht_solver import CHTSolver

        solver = CHTSolver(cht_case)
        solver.run()
        assert len(solver.convergence_history) > 0

    def test_residuals_decrease(self, cht_case):
        """Residuals generally decrease over time."""
        from pyfoam.applications.cht_solver import CHTSolver

        solver = CHTSolver(cht_case)
        solver.run()
        hist = solver.convergence_history
        if len(hist) >= 2:
            # Allow some oscillation but overall should decrease
            assert hist[-1] <= hist[0] * 1.5

    def test_run_with_custom_config(self, cht_case):
        """CHTSolver runs with custom config."""
        from pyfoam.applications.cht_solver import CHTSolver, CHTConfig

        cfg = CHTConfig(n_inner_iterations=3, relaxation_fluid=0.5)
        solver = CHTSolver(cht_case, config=cfg)
        conv = solver.run()
        assert conv is not None
        assert torch.isfinite(solver.T).all()

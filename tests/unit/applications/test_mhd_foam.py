"""
Unit tests for MhdFoam — magnetohydrodynamics solver.

Tests cover:
- Case loading and mesh construction
- Field initialisation from 0/ directory
- MHD property reading (mu0, sigma, eta)
- Custom property injection
- Boundary condition parsing
- Lorentz force computation
- Run convergence
- Velocity, pressure, and magnetic field finite after solve
- Field writing
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from pyfoam.core.dtype import CFD_DTYPE
from pyfoam.io.foam_file import FoamFileHeader, FileFormat, write_foam_file


# ---------------------------------------------------------------------------
# Mesh generation helper for MHD case
# ---------------------------------------------------------------------------

def _make_mhd_case(
    case_dir: Path,
    n_cells_x: int = 4,
    n_cells_y: int = 4,
    nu: float = 1.0,
    mu0: float = 1.0,
    sigma: float = 1.0,
    end_time: int = 1,
    delta_t: float = 0.01,
    write_interval: int = 1,
    U_inlet: float = 1.0,
    B_applied: float = 0.1,
) -> None:
    """Write a complete 2D MHD case to *case_dir*.

    Creates a 2D mesh with:
    - Inlet at x=0: fixedValue U=(U_inlet, 0, 0), zeroGradient B
    - Outlet at x=1: zeroGradient U, zeroGradient B
    - Walls at y=0 and y=1: noSlip U, fixedValue B=(0, B_applied, 0)
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

    # Inlet
    inlet_start = n_internal
    for j in range(n_cells_y):
        cell = j * n_cells_x
        p0 = (j * (n_cells_x + 1)) * 2
        p1 = (j * (n_cells_x + 1)) * 2 + 1
        p2 = ((j + 1) * (n_cells_x + 1)) * 2 + 1
        p3 = ((j + 1) * (n_cells_x + 1)) * 2
        faces.append((4, p0, p1, p2, p3))
        owner.append(cell)

    # Outlet
    outlet_start = inlet_start + n_cells_y
    for j in range(n_cells_y):
        cell = j * n_cells_x + (n_cells_x - 1)
        p0 = (j * (n_cells_x + 1) + n_cells_x) * 2
        p1 = (j * (n_cells_x + 1) + n_cells_x) * 2 + 1
        p2 = ((j + 1) * (n_cells_x + 1) + n_cells_x) * 2 + 1
        p3 = ((j + 1) * (n_cells_x + 1) + n_cells_x) * 2
        faces.append((4, p0, p1, p2, p3))
        owner.append(cell)

    # Wall bottom
    wall_start = outlet_start + n_cells_y
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
    lines.append("    inlet")
    lines.append("    {")
    lines.append("        type            patch;")
    lines.append(f"        nFaces          {n_cells_y};")
    lines.append(f"        startFace       {inlet_start};")
    lines.append("    }")
    lines.append("    outlet")
    lines.append("    {")
    lines.append("        type            patch;")
    lines.append(f"        nFaces          {n_cells_y};")
    lines.append(f"        startFace       {outlet_start};")
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

    # ---- constant/transportProperties ----
    tp_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="dictionary", location="constant", object="transportProperties",
    )
    tp_body = f"nu              {nu};\n"
    write_foam_file(case_dir / "constant" / "transportProperties", tp_header, tp_body, overwrite=True)

    # ---- constant/mhdProperties ----
    mhd_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="dictionary", location="constant", object="mhdProperties",
    )
    mhd_body = (
        f"mu0             {mu0};\n"
        f"sigma           {sigma};\n"
    )
    write_foam_file(case_dir / "constant" / "mhdProperties", mhd_header, mhd_body, overwrite=True)

    # ---- 0/U ----
    zero_dir = case_dir / "0"
    zero_dir.mkdir(exist_ok=True)

    U_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="volVectorField", location="0", object="U",
    )
    U_body = (
        "dimensions      [0 1 -1 0 0 0 0];\n\n"
        "internalField   uniform (0 0 0);\n\n"
        "boundaryField\n{\n"
        "    inlet\n    {\n"
        "        type            fixedValue;\n"
        f"        value           uniform ({U_inlet} 0 0);\n"
        "    }\n"
        "    outlet\n    {\n"
        "        type            zeroGradient;\n"
        "    }\n"
        "    wallBottom\n    {\n"
        "        type            noSlip;\n"
        "    }\n"
        "    wallTop\n    {\n"
        "        type            noSlip;\n"
        "    }\n"
        "    front\n    {\n"
        "        type            empty;\n"
        "    }\n"
        "    back\n    {\n"
        "        type            empty;\n"
        "    }\n"
        "}\n"
    )
    write_foam_file(zero_dir / "U", U_header, U_body, overwrite=True)

    # ---- 0/p ----
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
    write_foam_file(zero_dir / "p", p_header, p_body, overwrite=True)

    # ---- 0/B ----
    B_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="volVectorField", location="0", object="B",
    )
    B_body = (
        "dimensions      [1 0 -2 0 0 0 -1];\n\n"
        f"internalField   uniform (0 {B_applied} 0);\n\n"
        "boundaryField\n{\n"
        "    inlet\n    {\n"
        "        type            zeroGradient;\n"
        "    }\n"
        "    outlet\n    {\n"
        "        type            zeroGradient;\n"
        "    }\n"
        "    wallBottom\n    {\n"
        "        type            fixedValue;\n"
        f"        value           uniform (0 {B_applied} 0);\n"
        "    }\n"
        "    wallTop\n    {\n"
        "        type            fixedValue;\n"
        f"        value           uniform (0 {B_applied} 0);\n"
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

    # ---- system/controlDict ----
    sys_dir = case_dir / "system"
    sys_dir.mkdir(exist_ok=True)

    cd_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="dictionary", location="system", object="controlDict",
    )
    cd_body = (
        "application     mhdFoam;\n"
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
        "divSchemes\n{\n    default         Gauss linear;\n}\n\n"
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
        "    B\n    {\n"
        "        solver          PBiCGStab;\n"
        "        preconditioner  DILU;\n"
        "        tolerance       1e-6;\n"
        "        relTol          0.01;\n"
        "    }\n"
        "}\n\n"
        "SIMPLE\n{\n"
        "    nOuterCorrectors    2;\n"
        "    nNonOrthogonalCorrectors 0;\n"
        "    convergenceTolerance 1e-4;\n"
        "}\n\n"
        "relaxationFactors\n{\n"
        "    fields\n    {\n"
        "        p               0.3;\n"
        "    }\n"
        "    equations\n    {\n"
        "        U               0.7;\n"
        "        B               0.7;\n"
        "    }\n"
        "}\n"
    )
    write_foam_file(sys_dir / "fvSolution", fv_header, fv_body, overwrite=True)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mhd_case(tmp_path):
    """Create an MHD case (4x4 mesh)."""
    case_dir = tmp_path / "mhd"
    _make_mhd_case(
        case_dir,
        n_cells_x=4,
        n_cells_y=4,
        nu=1.0,
        mu0=1.0,
        sigma=1.0,
        end_time=1,
        delta_t=0.01,
        write_interval=1,
        U_inlet=1.0,
        B_applied=0.1,
    )
    return case_dir


@pytest.fixture
def tiny_mhd_case(tmp_path):
    """Create a minimal 2x2 MHD case for fast tests."""
    case_dir = tmp_path / "tiny_mhd"
    _make_mhd_case(
        case_dir,
        n_cells_x=2,
        n_cells_y=2,
        nu=1.0,
        mu0=1.0,
        sigma=1.0,
        end_time=1,
        delta_t=0.01,
        write_interval=1,
        U_inlet=0.1,
        B_applied=0.01,
    )
    return case_dir


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestMhdFoamInit:
    """Tests for MhdFoam initialisation."""

    def test_case_loads(self, mhd_case):
        """Case directory is readable."""
        from pyfoam.io.case import Case
        case = Case(mhd_case)
        assert case.has_mesh()

    def test_mesh_builds(self, mhd_case):
        """FvMesh is constructed correctly."""
        from pyfoam.applications.solver_base import SolverBase
        solver = SolverBase(mhd_case)
        mesh = solver.mesh
        assert mesh.n_cells == 16  # 4x4
        assert mesh.n_internal_faces > 0

    def test_fields_initialise(self, mhd_case):
        """Fields are initialised from the 0/ directory."""
        from pyfoam.applications.mhd_foam import MhdFoam
        solver = MhdFoam(mhd_case)
        assert solver.U.shape == (16, 3)
        assert solver.p.shape == (16,)
        assert solver.B.shape == (16, 3)
        assert solver.J.shape == (16, 3)

    def test_mhd_properties_from_file(self, mhd_case):
        """MHD properties are read correctly."""
        from pyfoam.applications.mhd_foam import MhdFoam
        solver = MhdFoam(mhd_case)
        assert abs(solver.mu0 - 1.0) < 1e-10
        assert abs(solver.sigma - 1.0) < 1e-10
        assert abs(solver.eta - 1.0) < 1e-10  # 1/(mu0*sigma) = 1

    def test_custom_properties_injection(self, mhd_case):
        """Custom MHD properties can be injected."""
        from pyfoam.applications.mhd_foam import MhdFoam
        solver = MhdFoam(mhd_case, nu=0.01, mu0=1.257e-6, sigma=5.96e7)
        assert abs(solver.nu - 0.01) < 1e-10
        assert abs(solver.mu0 - 1.257e-6) < 1e-12


class TestMhdFoamLorentzForce:
    """Tests for Lorentz force computation."""

    def test_lorentz_force_shape(self, mhd_case):
        """Lorentz force has correct shape."""
        from pyfoam.applications.mhd_foam import MhdFoam
        solver = MhdFoam(mhd_case)
        F = solver._compute_lorentz_force()
        assert F.shape == (16, 3)

    def test_lorentz_force_finite(self, mhd_case):
        """Lorentz force is finite."""
        from pyfoam.applications.mhd_foam import MhdFoam
        solver = MhdFoam(mhd_case)
        F = solver._compute_lorentz_force()
        assert torch.isfinite(F).all(), "F_lorentz contains NaN/Inf"


class TestMhdFoamSolver:
    """Tests for solver execution."""

    def test_run_completes(self, tiny_mhd_case):
        """MhdFoam runs without errors."""
        from pyfoam.applications.mhd_foam import MhdFoam
        solver = MhdFoam(tiny_mhd_case)
        result = solver.run()
        assert "converged" in result

    def test_U_finite(self, tiny_mhd_case):
        """Velocity field is finite after solving."""
        from pyfoam.applications.mhd_foam import MhdFoam
        solver = MhdFoam(tiny_mhd_case)
        solver.run()
        assert torch.isfinite(solver.U).all(), "U contains NaN/Inf"

    def test_p_finite(self, tiny_mhd_case):
        """Pressure field is finite after solving."""
        from pyfoam.applications.mhd_foam import MhdFoam
        solver = MhdFoam(tiny_mhd_case)
        solver.run()
        assert torch.isfinite(solver.p).all(), "p contains NaN/Inf"

    def test_B_finite(self, tiny_mhd_case):
        """Magnetic field is finite after solving."""
        from pyfoam.applications.mhd_foam import MhdFoam
        solver = MhdFoam(tiny_mhd_case)
        solver.run()
        assert torch.isfinite(solver.B).all(), "B contains NaN/Inf"

    def test_writes_output(self, tiny_mhd_case):
        """Fields are written to time directories."""
        from pyfoam.applications.mhd_foam import MhdFoam
        solver = MhdFoam(tiny_mhd_case)
        solver.run()
        time_dirs = [
            d for d in tiny_mhd_case.iterdir()
            if d.is_dir() and d.name.replace(".", "").isdigit() and d.name != "0"
        ]
        assert len(time_dirs) >= 1

    def test_magnetic_diffusivity(self, tiny_mhd_case):
        """eta = 1/(mu0 * sigma) is computed correctly."""
        from pyfoam.applications.mhd_foam import MhdFoam
        solver = MhdFoam(tiny_mhd_case, mu0=2.0, sigma=0.5)
        expected_eta = 1.0 / (2.0 * 0.5)
        assert abs(solver.eta - expected_eta) < 1e-10

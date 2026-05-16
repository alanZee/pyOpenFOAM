"""
Unit tests for PotentialFoam — potential flow solver.

Tests cover:
- Case loading and mesh construction
- Field initialisation from 0/ directory
- Settings parsing (solver, tolerance)
- Potential equation assembly
- Boundary condition building
- Run convergence
- Velocity computation (U = ∇φ)
- Pressure computation (Bernoulli)
- Field writing
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from pyfoam.core.dtype import CFD_DTYPE
from pyfoam.io.foam_file import FoamFileHeader, FileFormat, write_foam_file


# ---------------------------------------------------------------------------
# Mesh generation helper for 2D cavity case
# ---------------------------------------------------------------------------

def _make_potential_case(
    case_dir: Path,
    n_cells_x: int = 4,
    n_cells_y: int = 4,
    L: float = 1.0,
    end_time: int = 100,
    write_interval: int = 100,
    phi_inlet: float = 1.0,
) -> None:
    """Write a complete 2D potential flow case to *case_dir*.

    Creates a simple 2D mesh with:
    - inlet at x=0 (fixedValue phi)
    - outlet at x=L (fixedValue phi=0)
    - walls at y=0 and y=L (zeroGradient)
    - empty patches in z-direction
    """
    case_dir.mkdir(parents=True, exist_ok=True)

    dx = L / n_cells_x
    dy = L / n_cells_y
    dz = 0.1

    # ---- Points ----
    points = []
    for j in range(n_cells_y + 1):
        for i in range(n_cells_x + 1):
            x = i * dx
            y = j * dy
            points.append((x, y, 0.0))
            points.append((x, y, dz))

    n_points = len(points)

    # ---- Faces ----
    faces = []
    owner = []
    neighbour = []

    # Internal x-faces (between adjacent cells in x-direction)
    for j in range(n_cells_y):
        for i in range(n_cells_x - 1):
            cell = j * n_cells_x + i
            # Points for this internal face
            p0 = (j * (n_cells_x + 1) + i) * 2 + 1
            p1 = (j * (n_cells_x + 1) + i + 1) * 2 + 1
            p2 = ((j + 1) * (n_cells_x + 1) + i + 1) * 2 + 1
            p3 = ((j + 1) * (n_cells_x + 1) + i) * 2 + 1
            faces.append((4, p0, p1, p2, p3))
            owner.append(cell)
            neighbour.append(cell + 1)

    # Internal y-faces (between adjacent cells in y-direction)
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

    # Inlet boundary (x=0): fixedValue phi
    inlet_start = n_internal
    for j in range(n_cells_y):
        cell = j * n_cells_x
        p0 = (j * (n_cells_x + 1)) * 2
        p1 = (j * (n_cells_x + 1)) * 2 + 1
        p2 = ((j + 1) * (n_cells_x + 1)) * 2 + 1
        p3 = ((j + 1) * (n_cells_x + 1)) * 2
        faces.append((4, p0, p1, p2, p3))
        owner.append(cell)

    # Outlet boundary (x=L): fixedValue phi=0
    outlet_start = inlet_start + n_cells_y
    for j in range(n_cells_y):
        cell = j * n_cells_x + (n_cells_x - 1)
        p0 = (j * (n_cells_x + 1) + n_cells_x) * 2
        p1 = (j * (n_cells_x + 1) + n_cells_x) * 2 + 1
        p2 = ((j + 1) * (n_cells_x + 1) + n_cells_x) * 2 + 1
        p3 = ((j + 1) * (n_cells_x + 1) + n_cells_x) * 2
        faces.append((4, p0, p1, p2, p3))
        owner.append(cell)

    # Wall at y=0: zeroGradient
    wall_start = outlet_start + n_cells_y
    for i in range(n_cells_x):
        cell = i
        p0 = i * 2
        p1 = (i + 1) * 2
        p2 = (i + 1) * 2 + 1
        p3 = i * 2 + 1
        faces.append((4, p0, p1, p2, p3))
        owner.append(cell)

    # Wall at y=L: zeroGradient
    wall_top_start = wall_start + n_cells_x
    for i in range(n_cells_x):
        cell = (n_cells_y - 1) * n_cells_x + i
        p0 = (n_cells_y * (n_cells_x + 1) + i) * 2
        p1 = (n_cells_y * (n_cells_x + 1) + i + 1) * 2
        p2 = (n_cells_y * (n_cells_x + 1) + i + 1) * 2 + 1
        p3 = (n_cells_y * (n_cells_x + 1) + i) * 2 + 1
        faces.append((4, p0, p1, p2, p3))
        owner.append(cell)

    # Empty patches (front and back in z)
    empty_start = wall_top_start + n_cells_x
    n_cells = n_cells_x * n_cells_y

    # Front (z=0)
    for j in range(n_cells_y):
        for i in range(n_cells_x):
            cell = j * n_cells_x + i
            p0 = (j * (n_cells_x + 1) + i) * 2
            p1 = (j * (n_cells_x + 1) + i + 1) * 2
            p2 = ((j + 1) * (n_cells_x + 1) + i + 1) * 2
            p3 = ((j + 1) * (n_cells_x + 1) + i) * 2
            faces.append((4, p0, p1, p2, p3))
            owner.append(cell)

    # Back (z=dz)
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

    # ---- 0/phi (velocity potential) ----
    zero_dir = case_dir / "0"
    zero_dir.mkdir(exist_ok=True)

    phi_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="volScalarField", location="0", object="phi",
    )
    phi_body = (
        "dimensions      [0 2 -1 0 0 0 0];\n\n"
        "internalField   uniform 0;\n\n"
        "boundaryField\n{\n"
        "    inlet\n    {\n"
        f"        type            fixedValue;\n"
        f"        value           uniform {phi_inlet};\n"
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
    write_foam_file(zero_dir / "phi", phi_header, phi_body, overwrite=True)

    # ---- 0/U ----
    u_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="volVectorField", location="0", object="U",
    )
    u_body = (
        "dimensions      [0 1 -1 0 0 0 0];\n\n"
        "internalField   uniform (0 0 0);\n\n"
        "boundaryField\n{\n"
        "    inlet\n    {\n"
        "        type            zeroGradient;\n"
        "    }\n"
        "    outlet\n    {\n"
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
        "    inlet\n    {\n"
        "        type            zeroGradient;\n"
        "    }\n"
        "    outlet\n    {\n"
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
    write_foam_file(zero_dir / "p", p_header, p_body, overwrite=True)

    # ---- system/controlDict ----
    sys_dir = case_dir / "system"
    sys_dir.mkdir(exist_ok=True)

    cd_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="dictionary", location="system", object="controlDict",
    )
    cd_body = (
        "application     potentialFoam;\n"
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
        "    phi\n    {\n"
        "        solver          PCG;\n"
        "        preconditioner  DIC;\n"
        "        tolerance       1e-6;\n"
        "        relTol          0.01;\n"
        "    }\n"
        "}\n\n"
        "potentialFlow\n{\n"
        "    nNonOrthogonalCorrectors 0;\n"
        "    convergenceTolerance 1e-5;\n"
        "    pRefCell    0;\n"
        "    pRefValue   0;\n"
        "}\n"
    )
    write_foam_file(sys_dir / "fvSolution", fv_header, fv_body, overwrite=True)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def potential_case(tmp_path):
    """Create a potential flow case in a temporary directory."""
    case_dir = tmp_path / "potential"
    _make_potential_case(
        case_dir,
        n_cells_x=3,
        n_cells_y=3,
        L=1.0,
        end_time=100,
        write_interval=100,
        phi_inlet=1.0,
    )
    return case_dir


@pytest.fixture
def tiny_potential_case(tmp_path):
    """Create a minimal 2x2 potential flow case for fast tests."""
    case_dir = tmp_path / "tiny_potential"
    _make_potential_case(
        case_dir,
        n_cells_x=2,
        n_cells_y=2,
        L=1.0,
        end_time=10,
        write_interval=10,
        phi_inlet=1.0,
    )
    return case_dir


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPotentialFoamInit:
    """Tests for PotentialFoam initialisation."""

    def test_case_loads(self, potential_case):
        """Case directory is readable."""
        from pyfoam.io.case import Case

        case = Case(potential_case)
        assert case.has_mesh()

    def test_mesh_builds(self, potential_case):
        """FvMesh is constructed correctly."""
        from pyfoam.applications.solver_base import SolverBase

        solver = SolverBase(potential_case)
        mesh = solver.mesh

        assert mesh.n_cells == 9  # 3x3
        assert mesh.n_internal_faces > 0

    def test_fields_initialise(self, potential_case):
        """Fields are initialised from the 0/ directory."""
        from pyfoam.applications.potential_foam import PotentialFoam

        solver = PotentialFoam(potential_case)

        assert solver.phi_potential.shape == (9,)
        assert solver.U.shape == (9, 3)
        assert solver.p.shape == (9,)

    def test_settings_read(self, potential_case):
        """Settings are read correctly."""
        from pyfoam.applications.potential_foam import PotentialFoam

        solver = PotentialFoam(potential_case)

        assert solver.phi_solver == "PCG"
        assert abs(solver.phi_tolerance - 1e-6) < 1e-10
        assert solver.n_non_orth_correctors == 0


class TestPotentialFoamBoundaryConditions:
    """Tests for boundary condition building."""

    def test_bc_tensor_shape(self, potential_case):
        """phi_bc has correct shape."""
        from pyfoam.applications.potential_foam import PotentialFoam

        solver = PotentialFoam(potential_case)
        phi_bc = solver._build_phi_boundary_conditions()

        assert phi_bc.shape == (9,)

    def test_bc_has_fixed_values(self, potential_case):
        """phi_bc has prescribed values for boundary cells."""
        from pyfoam.applications.potential_foam import PotentialFoam

        solver = PotentialFoam(potential_case)
        phi_bc = solver._build_phi_boundary_conditions()

        bc_mask = ~torch.isnan(phi_bc)
        assert bc_mask.any(), "No boundary conditions found"


class TestPotentialFoamSolver:
    """Tests for solver execution."""

    def test_run_completes(self, tiny_potential_case):
        """PotentialFoam runs without errors."""
        from pyfoam.applications.potential_foam import PotentialFoam

        solver = PotentialFoam(tiny_potential_case)
        result = solver.run()

        assert result["converged"] is True

    def test_velocity_finite(self, tiny_potential_case):
        """Velocity field is finite after solving."""
        from pyfoam.applications.potential_foam import PotentialFoam

        solver = PotentialFoam(tiny_potential_case)
        solver.run()

        assert torch.isfinite(solver.U).all(), "U contains NaN/Inf"

    def test_pressure_finite(self, tiny_potential_case):
        """Pressure field is finite after solving."""
        from pyfoam.applications.potential_foam import PotentialFoam

        solver = PotentialFoam(tiny_potential_case)
        solver.run()

        assert torch.isfinite(solver.p).all(), "p contains NaN/Inf"

    def test_potential_changes(self, tiny_potential_case):
        """Potential field changes from initial conditions."""
        from pyfoam.applications.potential_foam import PotentialFoam

        solver = PotentialFoam(tiny_potential_case)
        phi_initial = solver.phi_potential.clone()

        solver.run()

        # Potential should have changed
        diff = (solver.phi_potential - phi_initial).abs().sum()
        assert diff > 0, "Potential did not change"

    def test_velocity_nonzero(self, tiny_potential_case):
        """Velocity is non-zero (gradient of potential)."""
        from pyfoam.applications.potential_foam import PotentialFoam

        solver = PotentialFoam(tiny_potential_case)
        solver.run()

        # Should have some non-zero velocity
        U_mag = (solver.U ** 2).sum(dim=1).sqrt()
        assert U_mag.max() > 0, "Velocity is zero everywhere"

    def test_writes_output(self, tiny_potential_case):
        """Fields are written to time directories."""
        from pyfoam.applications.potential_foam import PotentialFoam

        solver = PotentialFoam(tiny_potential_case)
        solver.run()

        # Check that output was written
        time_dirs = [
            d for d in tiny_potential_case.iterdir()
            if d.is_dir() and d.name.replace(".", "").isdigit() and d.name != "0"
        ]
        assert len(time_dirs) >= 1

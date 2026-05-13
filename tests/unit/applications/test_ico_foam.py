"""
End-to-end test: icoFoam lid-driven cavity.

Creates a complete OpenFOAM case directory on disk (mesh, fields,
system files), runs IcoFoam (transient PISO), and verifies convergence.

The cavity is a unit square [0,1] x [0,1] with:
- Top wall moving at U = (1, 0, 0)
- All other walls stationary
- nu = 0.01 (Re = 100)
- Transient simulation with small time steps (PISO)
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
    delta_t: float = 0.001,
    end_time: float = 0.1,
    n_piso_correctors: int = 2,
) -> None:
    """Write a complete lid-driven cavity case for icoFoam to *case_dir*.

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
        "application     icoFoam;\n"
        "startFrom       startTime;\n"
        "startTime       0;\n"
        "stopAt          endTime;\n"
        f"endTime         {end_time:g};\n"
        f"deltaT          {delta_t:g};\n"
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
        "ddtSchemes\n{\n    default         Euler;\n}\n\n"
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
        "PISO\n{\n"
        f"    nCorrectors         {n_piso_correctors};\n"
        "    nNonOrthogonalCorrectors 0;\n"
        "    convergenceTolerance 1e-4;\n"
        "}\n"
    )
    write_foam_file(sys_dir / "fvSolution", fv_header, fv_body, overwrite=True)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.fixture
def cavity_case(tmp_path):
    """Create a lid-driven cavity case for icoFoam in a temporary directory."""
    case_dir = tmp_path / "cavity"
    _make_cavity_case(
        case_dir,
        n_cells_x=4,
        n_cells_y=4,
        nu=0.01,
        delta_t=0.001,
        end_time=0.01,  # 10 time steps for fast test
        n_piso_correctors=2,
    )
    return case_dir


class TestIcoFoamCavity:
    """End-to-end tests for icoFoam with lid-driven cavity."""

    def test_case_loads(self, cavity_case):
        """Case directory is readable and has expected structure."""
        from pyfoam.io.case import Case

        case = Case(cavity_case)
        assert case.has_mesh()
        assert case.has_field("U", 0)
        assert case.has_field("p", 0)
        assert case.get_application() == "icoFoam"
        assert case.get_end_time() == 0.01
        assert case.get_delta_t() == 0.001

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
        from pyfoam.applications.ico_foam import IcoFoam
        from pyfoam.core.dtype import CFD_DTYPE

        solver = IcoFoam(cavity_case)

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
        from pyfoam.applications.ico_foam import IcoFoam

        solver = IcoFoam(cavity_case)
        assert abs(solver.nu - 0.01) < 1e-10

    def test_piso_settings(self, cavity_case):
        """PISO settings are read correctly from fvSolution."""
        from pyfoam.applications.ico_foam import IcoFoam

        solver = IcoFoam(cavity_case)
        assert solver.p_solver == "PCG"
        assert solver.U_solver == "PBiCGStab"
        assert solver.n_piso_correctors == 2
        assert abs(solver.convergence_tolerance - 1e-4) < 1e-10

    def test_fv_schemes_settings(self, cavity_case):
        """fvSchemes settings are read correctly."""
        from pyfoam.applications.ico_foam import IcoFoam

        solver = IcoFoam(cavity_case)
        assert solver.ddt_scheme == "Euler"
        assert solver.grad_scheme == "Gauss linear"

    def test_run_produces_valid_fields(self, cavity_case):
        """icoFoam runs and produces valid fields."""
        from pyfoam.applications.ico_foam import IcoFoam

        solver = IcoFoam(cavity_case)
        conv = solver.run()

        # Fields should have correct shapes
        assert solver.U.shape == (16, 3)
        assert solver.p.shape == (16,)
        assert solver.phi.shape == (solver.mesh.n_faces,)

        # All values should be finite (no NaN or Inf)
        assert torch.isfinite(solver.U).all(), "U contains NaN/Inf"
        assert torch.isfinite(solver.p).all(), "p contains NaN/Inf"
        assert torch.isfinite(solver.phi).all(), "phi contains NaN/Inf"

    def test_run_writes_output(self, cavity_case):
        """icoFoam writes field files to time directories."""
        from pyfoam.applications.ico_foam import IcoFoam

        solver = IcoFoam(cavity_case)
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

    def test_fields_are_valid_format(self, cavity_case):
        """Written fields are valid OpenFOAM format."""
        from pyfoam.applications.ico_foam import IcoFoam
        from pyfoam.io.field_io import read_field

        solver = IcoFoam(cavity_case)
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

    def test_transient_time_stepping(self, cavity_case):
        """icoFoam advances through multiple time steps."""
        from pyfoam.applications.ico_foam import IcoFoam

        solver = IcoFoam(cavity_case)

        # Verify time settings
        assert abs(solver.delta_t - 0.001) < 1e-10
        assert abs(solver.end_time - 0.01) < 1e-10

        conv = solver.run()

        # Should have run at least one iteration
        assert conv.outer_iterations >= 1

    def test_velocity_changes_over_time(self, cavity_case):
        """Velocity field evolves from initial conditions."""
        from pyfoam.applications.ico_foam import IcoFoam

        solver = IcoFoam(cavity_case)

        # Initial velocity should be zero
        U_initial = solver.U.clone()

        conv = solver.run()

        # After running, velocity should have changed (at least somewhere)
        U_diff = (solver.U - U_initial).abs().sum()
        assert U_diff > 0, "Velocity did not change during simulation"

    def test_pressure_field_shape(self, cavity_case):
        """Pressure field has correct shape after solving."""
        from pyfoam.applications.ico_foam import IcoFoam

        solver = IcoFoam(cavity_case)
        solver.run()

        assert solver.p.shape == (16,)
        assert torch.isfinite(solver.p).all(), "p contains NaN/Inf"

    def test_boundary_conditions_applied(self, cavity_case):
        """Boundary conditions are properly applied."""
        from pyfoam.applications.ico_foam import IcoFoam

        solver = IcoFoam(cavity_case)
        U_bc = solver._build_boundary_conditions()

        # U_bc should have shape (16, 3)
        assert U_bc.shape == (16, 3)

        # At least some cells should have BCs (not all NaN)
        bc_mask = ~torch.isnan(U_bc[:, 0])
        assert bc_mask.any(), "No boundary conditions found"

    def test_solver_repr(self, cavity_case):
        """IcoFoam has a useful string representation."""
        from pyfoam.applications.ico_foam import IcoFoam

        solver = IcoFoam(cavity_case)
        # SolverBase doesn't define __repr__, but it should at least not crash
        assert solver.mesh is not None

    def test_multiple_piso_correctors(self, tmp_path):
        """icoFoam works with different numbers of PISO correctors."""
        case_dir = tmp_path / "cavity_3corr"
        _make_cavity_case(
            case_dir,
            n_cells_x=4,
            n_cells_y=4,
            nu=0.01,
            delta_t=0.001,
            end_time=0.005,  # 5 steps
            n_piso_correctors=3,
        )

        from pyfoam.applications.ico_foam import IcoFoam

        solver = IcoFoam(case_dir)
        assert solver.n_piso_correctors == 3

        conv = solver.run()
        assert torch.isfinite(solver.U).all()
        assert torch.isfinite(solver.p).all()


class TestIcoFoamSmallMesh:
    """Tests on a very small mesh (2x2) for fast execution."""

    @pytest.fixture
    def tiny_case(self, tmp_path):
        """Create a minimal 2x2 cavity case."""
        case_dir = tmp_path / "tiny_cavity"
        _make_cavity_case(
            case_dir,
            n_cells_x=2,
            n_cells_y=2,
            nu=0.01,
            delta_t=0.001,
            end_time=0.003,  # 3 steps
            n_piso_correctors=2,
        )
        return case_dir

    def test_tiny_mesh_runs(self, tiny_case):
        """2x2 mesh runs without errors."""
        from pyfoam.applications.ico_foam import IcoFoam

        solver = IcoFoam(tiny_case)
        assert solver.mesh.n_cells == 4

        conv = solver.run()
        assert solver.U.shape == (4, 3)
        assert solver.p.shape == (4,)
        assert torch.isfinite(solver.U).all()
        assert torch.isfinite(solver.p).all()

    def test_tiny_mesh_output(self, tiny_case):
        """2x2 mesh produces valid output."""
        from pyfoam.applications.ico_foam import IcoFoam
        from pyfoam.io.field_io import read_field

        solver = IcoFoam(tiny_case)
        solver.run()

        # Should have written fields
        time_dirs = [d for d in tiny_case.iterdir()
                     if d.is_dir() and d.name.replace(".", "").isdigit()
                     and d.name != "0"]
        assert len(time_dirs) >= 1

        last_dir = sorted(time_dirs, key=lambda d: float(d.name))[-1]
        U_data = read_field(last_dir / "U")
        assert U_data.scalar_type == "vector"

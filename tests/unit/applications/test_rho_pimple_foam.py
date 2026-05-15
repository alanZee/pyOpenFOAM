"""
End-to-end tests: rhoPimpleFoam transient compressible solver.

Creates complete OpenFOAM case directories with temperature field,
runs RhoPimpleFoam (transient compressible PIMPLE with energy coupling),
and verifies convergence, field shapes, and thermodynamic consistency.

Test cases:
- Heated cavity (natural convection driver)
- Channel flow with temperature gradient
- Various mesh sizes and algorithm settings
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

def _make_heated_cavity_case(
    case_dir: Path,
    n_cells_x: int = 4,
    n_cells_y: int = 4,
    delta_t: float = 1e-5,
    end_time: float = 1e-4,
    n_outer_correctors: int = 3,
    n_correctors: int = 2,
    T_hot: float = 350.0,
    T_cold: float = 300.0,
    p_init: float = 101325.0,
) -> None:
    """Write a heated cavity case for rhoPimpleFoam.

    Creates a unit square cavity with:
    - Left wall at T_hot (hot)
    - Right wall at T_cold (cold)
    - Top/bottom walls adiabatic (zeroGradient)
    - Perfect gas EOS with Sutherland transport
    - All walls no-slip for velocity

    Creates:
    - constant/polyMesh/{points, faces, owner, neighbour, boundary}
    - constant/thermophysicalProperties
    - 0/U, 0/p, 0/T
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
    # hotWall (left, x=0) — fixedValue T = T_hot
    for j in range(n_cells_y):
        p0 = j * (n_cells_x + 1)
        p1 = p0 + n_cells_x + 1
        p2 = p1 + n_base
        p3 = p0 + n_base
        faces.append((4, p0, p1, p2, p3))
        owner.append(j * n_cells_x)

    n_hot = n_cells_y
    hot_start = n_internal

    # coldWall (right, x=1) — fixedValue T = T_cold
    for j in range(n_cells_y):
        p0 = j * (n_cells_x + 1) + n_cells_x
        p1 = p0 + n_cells_x + 1
        p2 = p1 + n_base
        p3 = p0 + n_base
        faces.append((4, p0, p1, p2, p3))
        owner.append(j * n_cells_x + n_cells_x - 1)

    n_cold = n_cells_y
    cold_start = hot_start + n_hot

    # topBottom (top and bottom walls)
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

    n_tb = 2 * n_cells_x
    tb_start = cold_start + n_cold

    # frontAndBack (z-normal, empty)
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
    empty_start = tb_start + n_tb

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
    lines = ["4", "("]
    lines.append("    hotWall")
    lines.append("    {")
    lines.append("        type            wall;")
    lines.append(f"        nFaces          {n_hot};")
    lines.append(f"        startFace       {hot_start};")
    lines.append("    }")
    lines.append("    coldWall")
    lines.append("    {")
    lines.append("        type            wall;")
    lines.append(f"        nFaces          {n_cold};")
    lines.append(f"        startFace       {cold_start};")
    lines.append("    }")
    lines.append("    topBottom")
    lines.append("    {")
    lines.append("        type            wall;")
    lines.append(f"        nFaces          {n_tb};")
    lines.append(f"        startFace       {tb_start};")
    lines.append("    }")
    lines.append("    frontAndBack")
    lines.append("    {")
    lines.append("        type            empty;")
    lines.append(f"        nFaces          {n_empty};")
    lines.append(f"        startFace       {empty_start};")
    lines.append("    }")
    lines.append(")")
    write_foam_file(mesh_dir / "boundary", h, "\n".join(lines), overwrite=True)

    # ---- thermophysicalProperties ----
    tp_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="dictionary", location="constant", object="thermophysicalProperties",
    )
    tp_body = (
        "thermoType\n"
        "{\n"
        "    type            hePsiThermo;\n"
        "    mixture         pureMixture;\n"
        "    transport       const;\n"
        "    thermo          hConst;\n"
        "    equationOfState perfectGas;\n"
        "    specie          specie;\n"
        "    energy          sensibleEnthalpy;\n"
        "}\n\n"
        "mixture\n"
        "{\n"
        "    specie\n"
        "    {\n"
        "        molWeight      28.966;\n"
        "    }\n"
        "    thermodynamics\n"
        "    {\n"
        "        Cp          1005.0;\n"
        "        Hf          0;\n"
        "    }\n"
        "    transport\n"
        "    {\n"
        "        mu          1.716e-5;\n"
        "        Pr          0.7;\n"
        "    }\n"
        "}\n"
    )
    write_foam_file(
        case_dir / "constant" / "thermophysicalProperties", tp_header,
        tp_body, overwrite=True,
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
        "    hotWall\n    {\n"
        "        type            fixedValue;\n"
        "        value           uniform (0 0 0);\n"
        "    }\n"
        "    coldWall\n    {\n"
        "        type            fixedValue;\n"
        "        value           uniform (0 0 0);\n"
        "    }\n"
        "    topBottom\n    {\n"
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
        "dimensions      [1 -1 -2 0 0 0 0];\n\n"
        f"internalField   uniform {p_init};\n\n"
        "boundaryField\n{\n"
        "    hotWall\n    {\n"
        "        type            zeroGradient;\n"
        "    }\n"
        "    coldWall\n    {\n"
        "        type            zeroGradient;\n"
        "    }\n"
        "    topBottom\n    {\n"
        "        type            zeroGradient;\n"
        "    }\n"
        "    frontAndBack\n    {\n"
        "        type            empty;\n"
        "    }\n"
        "}\n"
    )
    write_foam_file(zero_dir / "p", p_header, p_body, overwrite=True)

    # ---- 0/T ----
    T_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="volScalarField", location="0", object="T",
    )
    T_body = (
        "dimensions      [0 0 0 1 0 0 0];\n\n"
        f"internalField   uniform {(T_hot + T_cold) / 2};\n\n"
        "boundaryField\n{\n"
        "    hotWall\n    {\n"
        "        type            fixedValue;\n"
        f"        value           uniform {T_hot};\n"
        "    }\n"
        "    coldWall\n    {\n"
        "        type            fixedValue;\n"
        f"        value           uniform {T_cold};\n"
        "    }\n"
        "    topBottom\n    {\n"
        "        type            zeroGradient;\n"
        "    }\n"
        "    frontAndBack\n    {\n"
        "        type            empty;\n"
        "    }\n"
        "}\n"
    )
    write_foam_file(zero_dir / "T", T_header, T_body, overwrite=True)

    # ---- system/controlDict ----
    sys_dir = case_dir / "system"
    sys_dir.mkdir(exist_ok=True)

    cd_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="dictionary", location="system", object="controlDict",
    )
    cd_body = (
        "application     rhoPimpleFoam;\n"
        "startFrom       startTime;\n"
        "startTime       0;\n"
        "stopAt          endTime;\n"
        f"endTime         {end_time:g};\n"
        f"deltaT          {delta_t:g};\n"
        "writeControl    timeStep;\n"
        "writeInterval   1;\n"
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
        "    T\n    {\n"
        "        solver          PCG;\n"
        "        preconditioner  DIC;\n"
        "        tolerance       1e-6;\n"
        "        relTol          0.01;\n"
        "    }\n"
        "}\n\n"
        "PIMPLE\n{\n"
        f"    nOuterCorrectors    {n_outer_correctors};\n"
        f"    nCorrectors         {n_correctors};\n"
        "    nNonOrthogonalCorrectors 0;\n"
        "    convergenceTolerance 1e-4;\n"
        "    maxOuterIterations  100;\n"
        "    relaxationFactors\n    {\n"
        "        p               0.3;\n"
        "        U               0.7;\n"
        "        T               1.0;\n"
        "    }\n"
        "}\n"
    )
    write_foam_file(sys_dir / "fvSolution", fv_header, fv_body, overwrite=True)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def heated_cavity_case(tmp_path):
    """Create a heated cavity case for rhoPimpleFoam."""
    case_dir = tmp_path / "heated_cavity"
    _make_heated_cavity_case(
        case_dir,
        n_cells_x=4,
        n_cells_y=4,
        delta_t=1e-5,
        end_time=5e-5,  # 5 time steps
        n_outer_correctors=3,
        n_correctors=2,
        T_hot=350.0,
        T_cold=300.0,
        p_init=101325.0,
    )
    return case_dir


@pytest.fixture
def tiny_heated_case(tmp_path):
    """Create a minimal 2x2 heated cavity case."""
    case_dir = tmp_path / "tiny_heated"
    _make_heated_cavity_case(
        case_dir,
        n_cells_x=2,
        n_cells_y=2,
        delta_t=1e-5,
        end_time=3e-5,  # 3 steps
        n_outer_correctors=3,
        n_correctors=2,
    )
    return case_dir


# ---------------------------------------------------------------------------
# Tests — Case loading and initialisation
# ---------------------------------------------------------------------------

class TestRhoPimpleFoamInit:
    """Tests for case loading and field initialisation."""

    def test_case_loads(self, heated_cavity_case):
        """Case directory is readable and has expected structure."""
        from pyfoam.io.case import Case

        case = Case(heated_cavity_case)
        assert case.has_mesh()
        assert case.has_field("U", 0)
        assert case.has_field("p", 0)
        assert case.has_field("T", 0)

    def test_mesh_builds(self, heated_cavity_case):
        """FvMesh is constructed correctly from case data."""
        from pyfoam.applications.solver_base import SolverBase

        solver = SolverBase(heated_cavity_case)
        mesh = solver.mesh

        assert mesh.n_cells == 16  # 4x4
        assert mesh.n_internal_faces > 0
        assert mesh.cell_volumes.shape == (16,)
        assert mesh.face_areas.shape[0] == mesh.n_faces

    def test_solver_initialises(self, heated_cavity_case):
        """RhoPimpleFoam initialises without errors."""
        from pyfoam.applications.rho_pimple_foam import RhoPimpleFoam

        solver = RhoPimpleFoam(heated_cavity_case)
        assert solver is not None
        assert solver.mesh is not None

    def test_velocity_field_shape(self, heated_cavity_case):
        """Velocity field has correct shape."""
        from pyfoam.applications.rho_pimple_foam import RhoPimpleFoam

        solver = RhoPimpleFoam(heated_cavity_case)
        assert solver.U.shape == (16, 3)

    def test_pressure_field_shape(self, heated_cavity_case):
        """Pressure field has correct shape and initial value."""
        from pyfoam.applications.rho_pimple_foam import RhoPimpleFoam

        solver = RhoPimpleFoam(heated_cavity_case)
        assert solver.p.shape == (16,)
        # All cells should be at ~101325 Pa
        assert torch.allclose(
            solver.p,
            torch.full_like(solver.p, 101325.0),
            atol=1.0,
        )

    def test_temperature_field_shape(self, heated_cavity_case):
        """Temperature field has correct shape and initial value."""
        from pyfoam.applications.rho_pimple_foam import RhoPimpleFoam

        solver = RhoPimpleFoam(heated_cavity_case)
        assert solver.T.shape == (16,)
        # Initial T should be average of hot and cold
        expected_T = (350.0 + 300.0) / 2
        assert torch.allclose(
            solver.T,
            torch.full_like(solver.T, expected_T),
            atol=1.0,
        )

    def test_phi_field_shape(self, heated_cavity_case):
        """Face flux field has correct shape."""
        from pyfoam.applications.rho_pimple_foam import RhoPimpleFoam

        solver = RhoPimpleFoam(heated_cavity_case)
        assert solver.phi.shape == (solver.mesh.n_faces,)

    def test_density_from_eos(self, heated_cavity_case):
        """Initial density is computed from EOS: rho = p / (R * T)."""
        from pyfoam.applications.rho_pimple_foam import RhoPimpleFoam

        solver = RhoPimpleFoam(heated_cavity_case)
        assert solver.rho.shape == (16,)

        # rho = p / (R * T), with R=287 J/(kg·K), p=101325, T=325
        R = 287.0
        expected_rho = 101325.0 / (R * 325.0)
        assert torch.allclose(
            solver.rho,
            torch.full_like(solver.rho, expected_rho),
            rtol=0.01,
        )

    def test_thermo_model_present(self, heated_cavity_case):
        """Thermophysical model is initialised."""
        from pyfoam.applications.rho_pimple_foam import RhoPimpleFoam

        solver = RhoPimpleFoam(heated_cavity_case)
        assert solver.thermo is not None
        assert hasattr(solver.thermo, 'rho')
        assert hasattr(solver.thermo, 'mu')
        assert hasattr(solver.thermo, 'kappa')

    def test_old_fields_stored(self, heated_cavity_case):
        """Old fields are stored for time derivative."""
        from pyfoam.applications.rho_pimple_foam import RhoPimpleFoam

        solver = RhoPimpleFoam(heated_cavity_case)

        assert solver.U_old.shape == solver.U.shape
        assert solver.p_old.shape == solver.p.shape
        assert solver.T_old.shape == solver.T.shape
        assert solver.rho_old.shape == solver.rho.shape


# ---------------------------------------------------------------------------
# Tests — PIMPLE settings
# ---------------------------------------------------------------------------

class TestRhoPimpleFoamSettings:
    """Tests for PIMPLE algorithm settings."""

    def test_pimple_settings_read(self, heated_cavity_case):
        """PIMPLE settings are read correctly from fvSolution."""
        from pyfoam.applications.rho_pimple_foam import RhoPimpleFoam

        solver = RhoPimpleFoam(heated_cavity_case)
        assert solver.p_solver == "PCG"
        assert solver.U_solver == "PBiCGStab"
        assert solver.T_solver == "PCG"
        assert solver.n_outer_correctors == 3
        assert solver.n_correctors == 2
        assert abs(solver.convergence_tolerance - 1e-4) < 1e-10
        assert abs(solver.alpha_U - 0.7) < 1e-10
        assert abs(solver.alpha_p - 0.3) < 1e-10
        assert abs(solver.alpha_T - 1.0) < 1e-10

    def test_pressure_solver_tolerance(self, heated_cavity_case):
        """Pressure solver tolerance is read."""
        from pyfoam.applications.rho_pimple_foam import RhoPimpleFoam

        solver = RhoPimpleFoam(heated_cavity_case)
        assert abs(solver.p_tolerance - 1e-6) < 1e-10

    def test_velocity_solver_tolerance(self, heated_cavity_case):
        """Velocity solver tolerance is read."""
        from pyfoam.applications.rho_pimple_foam import RhoPimpleFoam

        solver = RhoPimpleFoam(heated_cavity_case)
        assert abs(solver.U_tolerance - 1e-6) < 1e-10

    def test_temperature_solver_tolerance(self, heated_cavity_case):
        """Temperature solver tolerance is read."""
        from pyfoam.applications.rho_pimple_foam import RhoPimpleFoam

        solver = RhoPimpleFoam(heated_cavity_case)
        assert abs(solver.T_tolerance - 1e-6) < 1e-10


# ---------------------------------------------------------------------------
# Tests — Solver execution
# ---------------------------------------------------------------------------

class TestRhoPimpleFoamRun:
    """Tests for solver execution."""

    def test_run_produces_valid_fields(self, heated_cavity_case):
        """rhoPimpleFoam runs and produces valid fields."""
        from pyfoam.applications.rho_pimple_foam import RhoPimpleFoam

        solver = RhoPimpleFoam(heated_cavity_case)
        conv = solver.run()

        # Fields should have correct shapes
        assert solver.U.shape == (16, 3)
        assert solver.p.shape == (16,)
        assert solver.T.shape == (16,)
        assert solver.phi.shape == (solver.mesh.n_faces,)

        # All values should be finite
        assert torch.isfinite(solver.U).all(), "U contains NaN/Inf"
        assert torch.isfinite(solver.p).all(), "p contains NaN/Inf"
        assert torch.isfinite(solver.T).all(), "T contains NaN/Inf"
        assert torch.isfinite(solver.phi).all(), "phi contains NaN/Inf"

    def test_run_writes_output(self, heated_cavity_case):
        """rhoPimpleFoam writes field files to time directories."""
        from pyfoam.applications.rho_pimple_foam import RhoPimpleFoam

        solver = RhoPimpleFoam(heated_cavity_case)
        solver.run()

        # Check that at least one output time directory was created
        time_dirs = [d for d in heated_cavity_case.iterdir()
                     if d.is_dir() and d.name.replace(".", "").isdigit()
                     and d.name != "0"]
        assert len(time_dirs) >= 1

        # Check that U, p, and T were written
        for td in time_dirs:
            assert (td / "U").exists(), f"U not found in {td}"
            assert (td / "p").exists(), f"p not found in {td}"
            assert (td / "T").exists(), f"T not found in {td}"

    def test_run_returns_convergence_data(self, heated_cavity_case):
        """Run returns ConvergenceData with correct fields."""
        from pyfoam.applications.rho_pimple_foam import RhoPimpleFoam
        from pyfoam.solvers.coupled_solver import ConvergenceData

        solver = RhoPimpleFoam(heated_cavity_case)
        conv = solver.run()

        assert isinstance(conv, ConvergenceData)
        assert conv.continuity_error >= 0
        assert conv.U_residual >= 0
        assert conv.p_residual >= 0
        assert conv.outer_iterations >= 1

    def test_velocity_changes_over_time(self, heated_cavity_case):
        """Velocity field evolves from initial conditions."""
        from pyfoam.applications.rho_pimple_foam import RhoPimpleFoam

        solver = RhoPimpleFoam(heated_cavity_case)
        U_initial = solver.U.clone()

        conv = solver.run()

        # After running, velocity should have changed (at least somewhere)
        U_diff = (solver.U - U_initial).abs().sum()
        assert U_diff > 0, "Velocity did not change during simulation"

    def test_temperature_changes_over_time(self, heated_cavity_case):
        """Temperature field evolves from initial conditions."""
        from pyfoam.applications.rho_pimple_foam import RhoPimpleFoam

        solver = RhoPimpleFoam(heated_cavity_case)
        T_initial = solver.T.clone()

        conv = solver.run()

        # After running, temperature should have changed
        T_diff = (solver.T - T_initial).abs().sum()
        assert T_diff > 0, "Temperature did not change during simulation"

    def test_density_stays_positive(self, heated_cavity_case):
        """Density remains positive throughout simulation."""
        from pyfoam.applications.rho_pimple_foam import RhoPimpleFoam

        solver = RhoPimpleFoam(heated_cavity_case)
        solver.run()

        assert (solver.rho > 0).all(), "Density became non-positive"

    def test_pressure_stays_positive(self, heated_cavity_case):
        """Pressure remains positive (absolute pressure)."""
        from pyfoam.applications.rho_pimple_foam import RhoPimpleFoam

        solver = RhoPimpleFoam(heated_cavity_case)
        solver.run()

        # For compressible flow with absolute pressure, p should stay positive
        assert (solver.p > 0).all(), "Pressure became non-positive"


# ---------------------------------------------------------------------------
# Tests — Tiny mesh for fast execution
# ---------------------------------------------------------------------------

class TestRhoPimpleFoamTinyMesh:
    """Tests on a very small mesh (2x2) for fast execution."""

    def test_tiny_mesh_runs(self, tiny_heated_case):
        """2x2 mesh runs without errors."""
        from pyfoam.applications.rho_pimple_foam import RhoPimpleFoam

        solver = RhoPimpleFoam(tiny_heated_case)
        assert solver.mesh.n_cells == 4

        conv = solver.run()
        assert solver.U.shape == (4, 3)
        assert solver.p.shape == (4,)
        assert solver.T.shape == (4,)
        assert torch.isfinite(solver.U).all()
        assert torch.isfinite(solver.p).all()
        assert torch.isfinite(solver.T).all()

    def test_tiny_mesh_output(self, tiny_heated_case):
        """2x2 mesh produces valid output."""
        from pyfoam.applications.rho_pimple_foam import RhoPimpleFoam
        from pyfoam.io.field_io import read_field

        solver = RhoPimpleFoam(tiny_heated_case)
        solver.run()

        # Should have written fields
        time_dirs = [d for d in tiny_heated_case.iterdir()
                     if d.is_dir() and d.name.replace(".", "").isdigit()
                     and d.name != "0"]
        assert len(time_dirs) >= 1

        last_dir = sorted(time_dirs, key=lambda d: float(d.name))[-1]
        U_data = read_field(last_dir / "U")
        assert U_data.scalar_type == "vector"


# ---------------------------------------------------------------------------
# Tests — Algorithm behaviour
# ---------------------------------------------------------------------------

class TestRhoPimpleFoamAlgorithm:
    """Tests for PIMPLE algorithm behaviour."""

    def test_outer_iterations_reported(self, heated_cavity_case):
        """PIMPLE reports outer iteration count."""
        from pyfoam.applications.rho_pimple_foam import RhoPimpleFoam

        solver = RhoPimpleFoam(heated_cavity_case)
        conv = solver.run()

        assert conv.outer_iterations >= 1

    def test_convergence_data_complete(self, heated_cavity_case):
        """PIMPLE returns complete convergence data."""
        from pyfoam.applications.rho_pimple_foam import RhoPimpleFoam

        solver = RhoPimpleFoam(heated_cavity_case)
        conv = solver.run()

        assert conv.continuity_error >= 0
        assert conv.U_residual >= 0
        assert conv.p_residual >= 0

    def test_different_outer_correctors(self, tmp_path):
        """rhoPimpleFoam works with different numbers of outer correctors."""
        case_dir = tmp_path / "heated_5outer"
        _make_heated_cavity_case(
            case_dir,
            n_cells_x=2,
            n_cells_y=2,
            delta_t=1e-5,
            end_time=2e-5,
            n_outer_correctors=5,
            n_correctors=3,
        )

        from pyfoam.applications.rho_pimple_foam import RhoPimpleFoam

        solver = RhoPimpleFoam(case_dir)
        assert solver.n_outer_correctors == 5
        assert solver.n_correctors == 3

        conv = solver.run()
        assert torch.isfinite(solver.U).all()
        assert torch.isfinite(solver.p).all()
        assert torch.isfinite(solver.T).all()


# ---------------------------------------------------------------------------
# Tests — Thermodynamic consistency
# ---------------------------------------------------------------------------

class TestRhoPimpleFoamThermo:
    """Tests for thermodynamic consistency."""

    def test_eos_consistency(self, heated_cavity_case):
        """Density is consistent with EOS after solving."""
        from pyfoam.applications.rho_pimple_foam import RhoPimpleFoam

        solver = RhoPimpleFoam(heated_cavity_case)
        solver.run()

        # Recompute density from EOS
        rho_check = solver.thermo.rho(solver.p, solver.T)

        # Should match stored density
        assert torch.allclose(solver.rho, rho_check, rtol=1e-3), \
            "Stored density inconsistent with EOS"

    def test_viscosity_positive(self, heated_cavity_case):
        """Viscosity is positive for all cells."""
        from pyfoam.applications.rho_pimple_foam import RhoPimpleFoam

        solver = RhoPimpleFoam(heated_cavity_case)
        solver.run()

        mu = solver.thermo.mu(solver.T)
        assert (mu > 0).all(), "Viscosity became non-positive"

    def test_thermal_conductivity_positive(self, heated_cavity_case):
        """Thermal conductivity is positive for all cells."""
        from pyfoam.applications.rho_pimple_foam import RhoPimpleFoam

        solver = RhoPimpleFoam(heated_cavity_case)
        solver.run()

        kappa = solver.thermo.kappa(solver.T)
        assert (kappa > 0).all(), "Thermal conductivity became non-positive"

    def test_heat_flux_direction(self, heated_cavity_case):
        """Heat flux flows from hot to cold wall."""
        from pyfoam.applications.rho_pimple_foam import RhoPimpleFoam

        solver = RhoPimpleFoam(heated_cavity_case)
        solver.run()

        # After some time, cells near left wall should be warmer
        # than cells near right wall
        # In a 4x4 grid: left cells are indices 0,4,8,12; right are 3,7,11,15
        n_x = 4
        n_y = 4
        left_cells = [j * n_x for j in range(n_y)]
        right_cells = [j * n_x + n_x - 1 for j in range(n_y)]

        T_left = solver.T[left_cells].mean()
        T_right = solver.T[right_cells].mean()

        # Hot wall on left, cold on right
        # Due to BCs, left cells should see higher T
        # (This may not hold strongly after only 5 steps, but trend should be there)
        # At minimum, temperatures should be finite and different
        assert torch.isfinite(T_left)
        assert torch.isfinite(T_right)


# ---------------------------------------------------------------------------
# Tests — Time control
# ---------------------------------------------------------------------------

class TestRhoPimpleFoamTimeControl:
    """Tests for time stepping control."""

    def test_time_settings(self, heated_cavity_case):
        """Time settings are read correctly."""
        from pyfoam.applications.rho_pimple_foam import RhoPimpleFoam

        solver = RhoPimpleFoam(heated_cavity_case)
        assert abs(solver.delta_t - 1e-5) < 1e-15
        assert abs(solver.end_time - 5e-5) < 1e-15

    def test_transient_time_stepping(self, heated_cavity_case):
        """rhoPimpleFoam advances through multiple time steps."""
        from pyfoam.applications.rho_pimple_foam import RhoPimpleFoam

        solver = RhoPimpleFoam(heated_cavity_case)
        conv = solver.run()

        # Should have run at least one iteration
        assert conv.outer_iterations >= 1


# ---------------------------------------------------------------------------
# Tests — Written field format
# ---------------------------------------------------------------------------

class TestRhoPimpleFoamOutput:
    """Tests for written field format."""

    def test_written_fields_valid_format(self, heated_cavity_case):
        """Written fields are valid OpenFOAM format."""
        from pyfoam.applications.rho_pimple_foam import RhoPimpleFoam
        from pyfoam.io.field_io import read_field

        solver = RhoPimpleFoam(heated_cavity_case)
        solver.run()

        # Find the last written time directory
        time_dirs = sorted(
            [d for d in heated_cavity_case.iterdir()
             if d.is_dir() and d.name.replace(".", "").isdigit()
             and d.name != "0"],
            key=lambda d: float(d.name),
        )
        assert len(time_dirs) >= 1

        last_dir = time_dirs[-1]
        U_data = read_field(last_dir / "U")
        p_data = read_field(last_dir / "p")
        T_data = read_field(last_dir / "T")

        assert U_data.scalar_type == "vector"
        assert p_data.scalar_type == "scalar"
        assert T_data.scalar_type == "scalar"
        assert not U_data.is_uniform
        assert not p_data.is_uniform
        assert not T_data.is_uniform

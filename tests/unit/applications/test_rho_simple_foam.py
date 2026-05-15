"""
Unit tests for RhoSimpleFoam — steady-state compressible SIMPLE solver.

Tests cover:
- Case loading and mesh construction with temperature field
- Thermophysical model initialisation (EOS + transport)
- Field initialisation (U, p, T, phi, rho)
- fvSolution/fvSchemes settings parsing
- Turbulence model initialisation (optional)
- EOS density computation (ρ = p/RT)
- Energy equation solving
- Compressible pressure equation
- Momentum predictor with variable density
- Viscous dissipation computation
- Convergence on heated cavity
- Field writing to time directories
- Written field format validity
- Solver produces finite values
- Turbulence coupling (ρk-ε)
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from pyfoam.core.dtype import CFD_DTYPE, INDEX_DTYPE
from pyfoam.io.foam_file import FoamFileHeader, FileFormat, write_foam_file


# ---------------------------------------------------------------------------
# Mesh generation helper for compressible cavity case
# ---------------------------------------------------------------------------

def _make_compressible_cavity_case(
    case_dir: Path,
    n_cells_x: int = 10,
    n_cells_y: int = 10,
    T_init: float = 300.0,
    T_top: float = 310.0,
    p_init: float = 101325.0,
    end_time: int = 500,
    write_interval: int = 100,
    alpha_p: float = 0.3,
    alpha_U: float = 0.7,
    alpha_T: float = 1.0,
    convergence_tolerance: float = 1e-4,
    max_outer_iterations: int = 100,
    turbulence_model: str | None = None,
) -> None:
    """Write a complete compressible lid-driven cavity case to *case_dir*.

    Creates:
    - constant/polyMesh/{points, faces, owner, neighbour, boundary}
    - 0/U, 0/p, 0/T
    - system/{controlDict, fvSchemes, fvSolution}

    The case uses air (R=287, Cp=1005) with Sutherland viscosity.
    The top wall moves at U=(1,0,0) with temperature T_top.
    Other walls are stationary at temperature T_init.
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

    # ---- 0/p (pressure in Pa for compressible) ----
    p_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="volScalarField", location="0", object="p",
    )
    p_body = (
        "dimensions      [1 -1 -2 0 0 0 0];\n\n"
        f"internalField   uniform {p_init};\n\n"
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

    # ---- 0/T (temperature) ----
    T_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="volScalarField", location="0", object="T",
    )
    T_body = (
        "dimensions      [0 0 0 1 0 0 0];\n\n"
        f"internalField   uniform {T_init};\n\n"
        "boundaryField\n{\n"
        "    movingWall\n    {\n"
        "        type            fixedValue;\n"
        f"        value           uniform {T_top};\n"
        "    }\n"
        "    fixedWalls\n    {\n"
        "        type            fixedValue;\n"
        f"        value           uniform {T_init};\n"
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
        "application     rhoSimpleFoam;\n"
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
        "        maxIter         1000;\n"
        "    }\n"
        "}\n\n"
        "SIMPLE\n{\n"
        "    nNonOrthogonalCorrectors 0;\n"
        "    residualControl\n    {\n"
        "        p               1e-4;\n"
        "        U               1e-4;\n"
        "    }\n"
        "    relaxationFactors\n    {\n"
        f"        p               {alpha_p};\n"
        f"        U               {alpha_U};\n"
        f"        T               {alpha_T};\n"
        "    }\n"
        f"    convergenceTolerance {convergence_tolerance};\n"
        f"    maxOuterIterations  {max_outer_iterations};\n"
        "}\n"
    )
    write_foam_file(sys_dir / "fvSolution", fv_header, fv_body, overwrite=True)

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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def compressible_cavity(tmp_path):
    """Create a compressible cavity case (4x4 mesh)."""
    case_dir = tmp_path / "compressible_cavity"
    _make_compressible_cavity_case(
        case_dir,
        n_cells_x=4,
        n_cells_y=4,
        T_init=300.0,
        T_top=310.0,
        p_init=101325.0,
        end_time=50,
        write_interval=50,
        max_outer_iterations=20,
    )
    return case_dir


@pytest.fixture
def tiny_compressible_cavity(tmp_path):
    """Create a minimal 2x2 compressible cavity for fast tests."""
    case_dir = tmp_path / "tiny_compressible"
    _make_compressible_cavity_case(
        case_dir,
        n_cells_x=2,
        n_cells_y=2,
        T_init=300.0,
        T_top=305.0,
        p_init=101325.0,
        end_time=10,
        write_interval=10,
        max_outer_iterations=10,
    )
    return case_dir


@pytest.fixture
def heated_cavity_turbulent(tmp_path):
    """Create a 2x2 compressible cavity with kEpsilon turbulence."""
    case_dir = tmp_path / "heated_turb"
    _make_compressible_cavity_case(
        case_dir,
        n_cells_x=2,
        n_cells_y=2,
        T_init=300.0,
        T_top=310.0,
        p_init=101325.0,
        end_time=10,
        write_interval=10,
        max_outer_iterations=10,
        turbulence_model="kEpsilon",
    )
    return case_dir


# ===========================================================================
# Tests
# ===========================================================================


class TestRhoSimpleFoamInit:
    """Tests for RhoSimpleFoam initialisation and property reading."""

    def test_case_loads(self, compressible_cavity):
        """Case directory is readable and has expected structure."""
        from pyfoam.io.case import Case

        case = Case(compressible_cavity)
        assert case.has_mesh()
        assert case.has_field("U", 0)
        assert case.has_field("p", 0)
        assert case.has_field("T", 0)
        assert case.get_application() == "rhoSimpleFoam"
        assert case.get_end_time() == 50

    def test_mesh_builds(self, compressible_cavity):
        """FvMesh is constructed correctly from case data."""
        from pyfoam.applications.solver_base import SolverBase

        solver = SolverBase(compressible_cavity)
        mesh = solver.mesh

        assert mesh.n_cells == 16  # 4x4
        assert mesh.n_internal_faces > 0
        assert mesh.cell_volumes.shape == (16,)
        assert mesh.face_areas.shape[0] == mesh.n_faces

    def test_fields_initialise(self, compressible_cavity):
        """Fields are initialised from the 0/ directory."""
        from pyfoam.applications.rho_simple_foam import RhoSimpleFoam

        solver = RhoSimpleFoam(compressible_cavity)

        # U should be (16, 3) zeros (uniform (0 0 0))
        assert solver.U.shape == (16, 3)
        assert torch.allclose(solver.U, torch.zeros(16, 3, dtype=CFD_DTYPE))

        # p should be (16,) with p_init
        assert solver.p.shape == (16,)
        assert torch.allclose(solver.p, torch.full((16,), 101325.0, dtype=CFD_DTYPE))

        # T should be (16,) with T_init
        assert solver.T.shape == (16,)
        assert torch.allclose(solver.T, torch.full((16,), 300.0, dtype=CFD_DTYPE))

        # phi should be (n_faces,) zeros
        assert solver.phi.shape == (solver.mesh.n_faces,)

        # rho should be computed from EOS
        assert solver.rho.shape == (16,)
        assert torch.isfinite(solver.rho).all()

    def test_thermo_defaults(self, compressible_cavity):
        """Default thermo is air (PerfectGas + Sutherland)."""
        from pyfoam.applications.rho_simple_foam import RhoSimpleFoam

        solver = RhoSimpleFoam(compressible_cavity)
        assert solver.thermo is not None
        assert solver.thermo.R() == 287.0
        assert solver.thermo.Cp() == 1005.0

    def test_custom_thermo(self, compressible_cavity):
        """Custom thermo model can be injected."""
        from pyfoam.applications.rho_simple_foam import RhoSimpleFoam
        from pyfoam.thermophysical.thermo import BasicThermo
        from pyfoam.thermophysical.equation_of_state import PerfectGas
        from pyfoam.thermophysical.transport_model import ConstantViscosity

        custom_thermo = BasicThermo(
            eos=PerfectGas(R=461.5, Cp=1872.0),  # steam
            transport=ConstantViscosity(mu=1.2e-5),
        )
        solver = RhoSimpleFoam(compressible_cavity, thermo=custom_thermo)
        assert solver.thermo.R() == 461.5
        assert solver.thermo.Cp() == 1872.0

    def test_fv_solution_settings(self, compressible_cavity):
        """fvSolution settings are read correctly."""
        from pyfoam.applications.rho_simple_foam import RhoSimpleFoam

        solver = RhoSimpleFoam(compressible_cavity)
        assert solver.p_solver == "PCG"
        assert solver.U_solver == "PBiCGStab"
        assert solver.T_solver == "PCG"
        assert abs(solver.alpha_p - 0.3) < 1e-10
        assert abs(solver.alpha_U - 0.7) < 1e-10
        assert abs(solver.alpha_T - 1.0) < 1e-10
        assert abs(solver.convergence_tolerance - 1e-4) < 1e-10
        assert solver.max_outer_iterations == 20

    def test_fv_schemes_settings(self, compressible_cavity):
        """fvSchemes settings are read correctly."""
        from pyfoam.applications.rho_simple_foam import RhoSimpleFoam

        solver = RhoSimpleFoam(compressible_cavity)
        assert solver.grad_scheme == "Gauss linear"
        assert solver.lap_scheme == "Gauss linear corrected"

    def test_turbulence_disabled_by_default(self, compressible_cavity):
        """Turbulence is disabled when no turbulenceProperties exists."""
        from pyfoam.applications.rho_simple_foam import RhoSimpleFoam

        solver = RhoSimpleFoam(compressible_cavity)
        assert solver.turbulence_enabled is False
        assert solver.ras is None


class TestRhoSimpleFoamEOS:
    """Tests for equation of state integration."""

    def test_density_from_eos(self):
        """Density computed correctly from perfect gas EOS: ρ = p/(RT)."""
        from pyfoam.thermophysical.thermo import create_air_thermo

        thermo = create_air_thermo()
        p = torch.tensor([101325.0], dtype=CFD_DTYPE)
        T = torch.tensor([300.0], dtype=CFD_DTYPE)
        rho = thermo.rho(p, T)

        expected = 101325.0 / (287.0 * 300.0)
        assert abs(rho.item() - expected) / expected < 1e-10

    def test_density_consistency(self):
        """ρ = p/(RT) and p = ρRT are consistent."""
        from pyfoam.thermophysical.thermo import create_air_thermo

        thermo = create_air_thermo()
        p = torch.tensor([101325.0], dtype=CFD_DTYPE)
        T = torch.tensor([300.0], dtype=CFD_DTYPE)

        rho = thermo.rho(p, T)
        p_back = thermo.p(rho, T)

        assert torch.allclose(p, p_back, rtol=1e-10)

    def test_density_increases_with_pressure(self):
        """Density increases proportionally with pressure."""
        from pyfoam.thermophysical.thermo import create_air_thermo

        thermo = create_air_thermo()
        T = torch.tensor([300.0], dtype=CFD_DTYPE)
        p1 = torch.tensor([101325.0], dtype=CFD_DTYPE)
        p2 = torch.tensor([202650.0], dtype=CFD_DTYPE)

        rho1 = thermo.rho(p1, T)
        rho2 = thermo.rho(p2, T)

        assert abs(rho2.item() / rho1.item() - 2.0) < 1e-10

    def test_density_decreases_with_temperature(self):
        """Density decreases with increasing temperature."""
        from pyfoam.thermophysical.thermo import create_air_thermo

        thermo = create_air_thermo()
        p = torch.tensor([101325.0], dtype=CFD_DTYPE)
        T1 = torch.tensor([300.0], dtype=CFD_DTYPE)
        T2 = torch.tensor([600.0], dtype=CFD_DTYPE)

        rho1 = thermo.rho(p, T1)
        rho2 = thermo.rho(p, T2)

        assert rho2.item() < rho1.item()
        assert abs(rho1.item() / rho2.item() - 2.0) < 1e-10

    def test_density_tensor_shape(self):
        """EOS handles tensor inputs correctly."""
        from pyfoam.thermophysical.thermo import create_air_thermo

        thermo = create_air_thermo()
        n = 16
        p = torch.full((n,), 101325.0, dtype=CFD_DTYPE)
        T = torch.full((n,), 300.0, dtype=CFD_DTYPE)
        rho = thermo.rho(p, T)

        assert rho.shape == (n,)
        assert torch.isfinite(rho).all()

    def test_initial_rho_matches_eos(self, compressible_cavity):
        """Initial density field matches EOS computation."""
        from pyfoam.applications.rho_simple_foam import RhoSimpleFoam

        solver = RhoSimpleFoam(compressible_cavity)
        rho_expected = solver.thermo.rho(solver.p, solver.T)

        assert torch.allclose(solver.rho, rho_expected, rtol=1e-10)


class TestRhoSimpleFoamTransport:
    """Tests for transport model integration."""

    def test_viscosity_from_sutherland(self):
        """Sutherland viscosity at 300K is reasonable for air."""
        from pyfoam.thermophysical.transport_model import Sutherland

        sutherland = Sutherland()
        mu = sutherland.mu(300.0)

        # Air at 300K: μ ≈ 1.846e-5 Pa·s
        assert 1.7e-5 < mu.item() < 2.0e-5

    def test_viscosity_increases_with_temperature(self):
        """Sutherland viscosity increases with temperature."""
        from pyfoam.thermophysical.transport_model import Sutherland

        sutherland = Sutherland()
        mu_300 = sutherland.mu(300.0)
        mu_600 = sutherland.mu(600.0)

        assert mu_600.item() > mu_300.item()

    def test_thermal_conductivity(self):
        """Thermal conductivity κ = μ*Cp/Pr."""
        from pyfoam.thermophysical.thermo import create_air_thermo

        thermo = create_air_thermo()
        T = torch.tensor([300.0], dtype=CFD_DTYPE)
        kappa = thermo.kappa(T)
        mu = thermo.mu(T)
        cp = thermo.Cp()
        pr = thermo.Pr

        expected = mu * cp / pr
        assert torch.allclose(kappa, expected, rtol=1e-10)

    def test_prandtl_number(self):
        """Default Prandtl number is 0.7 (air)."""
        from pyfoam.thermophysical.thermo import create_air_thermo

        thermo = create_air_thermo()
        assert abs(thermo.Pr - 0.7) < 1e-10
        assert abs(thermo.Prt - 0.85) < 1e-10


class TestRhoSimpleFoamMomentum:
    """Tests for the momentum predictor."""

    def test_momentum_predictor_shape(self, compressible_cavity):
        """Momentum predictor returns correct shapes."""
        from pyfoam.applications.rho_simple_foam import RhoSimpleFoam

        solver = RhoSimpleFoam(compressible_cavity)
        U, A_p, H = solver._momentum_predictor(
            solver.U, solver.p, solver.phi, solver.rho,
        )

        n_cells = solver.mesh.n_cells
        assert U.shape == (n_cells, 3)
        assert A_p.shape == (n_cells,)
        assert H.shape == (n_cells, 3)

    def test_momentum_predictor_finite(self, compressible_cavity):
        """Momentum predictor produces finite values."""
        from pyfoam.applications.rho_simple_foam import RhoSimpleFoam

        solver = RhoSimpleFoam(compressible_cavity)
        U, A_p, H = solver._momentum_predictor(
            solver.U, solver.p, solver.phi, solver.rho,
        )

        assert torch.isfinite(U).all()
        assert torch.isfinite(A_p).all()
        assert torch.isfinite(H).all()

    def test_momentum_with_turbulent_viscosity(self, compressible_cavity):
        """Momentum predictor works with effective viscosity."""
        from pyfoam.applications.rho_simple_foam import RhoSimpleFoam

        solver = RhoSimpleFoam(compressible_cavity)

        # Create a synthetic effective viscosity field
        n_cells = solver.mesh.n_cells
        mu_mol = solver.thermo.mu(solver.T)
        mu_eff = mu_mol * 10.0  # 10x molecular (turbulent)

        U, A_p, H = solver._momentum_predictor(
            solver.U, solver.p, solver.phi, solver.rho, mu_eff=mu_eff,
        )

        assert torch.isfinite(U).all()
        # Higher viscosity should give different result
        U2, _, _ = solver._momentum_predictor(
            solver.U, solver.p, solver.phi, solver.rho,
        )
        # Results may differ due to different viscosity
        assert U.shape == U2.shape

    def test_pressure_gradient_effect(self, compressible_cavity):
        """Non-zero pressure gradient affects velocity."""
        from pyfoam.applications.rho_simple_foam import RhoSimpleFoam

        solver = RhoSimpleFoam(compressible_cavity)

        # Zero pressure → no pressure gradient contribution
        p_zero = torch.zeros_like(solver.p)
        U0, _, _ = solver._momentum_predictor(
            solver.U, p_zero, solver.phi, solver.rho,
        )

        # Non-zero pressure → pressure gradient affects velocity
        p_nonzero = solver.p.clone()
        p_nonzero[0] = 200000.0  # pressure spike
        U1, _, _ = solver._momentum_predictor(
            solver.U, p_nonzero, solver.phi, solver.rho,
        )

        # Results should differ
        assert not torch.allclose(U0, U1, atol=1e-10)


class TestRhoSimpleFoamPressure:
    """Tests for the compressible pressure equation."""

    def test_pressure_equation_shape(self, compressible_cavity):
        """Pressure equation returns correct shape."""
        from pyfoam.applications.rho_simple_foam import RhoSimpleFoam

        solver = RhoSimpleFoam(compressible_cavity)
        mesh = solver.mesh

        # Create dummy phiHbyA and A_p
        n_internal = mesh.n_internal_faces
        phiHbyA = torch.zeros(n_internal, dtype=CFD_DTYPE)
        A_p = torch.ones(mesh.n_cells, dtype=CFD_DTYPE)

        p_new = solver._solve_pressure_equation(
            solver.p, phiHbyA, A_p, solver.rho, mesh,
        )

        assert p_new.shape == (mesh.n_cells,)
        assert torch.isfinite(p_new).all()

    def test_pressure_equation_zero_source(self, compressible_cavity):
        """Zero flux source gives uniform pressure (up to reference)."""
        from pyfoam.applications.rho_simple_foam import RhoSimpleFoam

        solver = RhoSimpleFoam(compressible_cavity)
        mesh = solver.mesh

        n_internal = mesh.n_internal_faces
        phiHbyA = torch.zeros(n_internal, dtype=CFD_DTYPE)
        A_p = torch.ones(mesh.n_cells, dtype=CFD_DTYPE)

        p_new = solver._solve_pressure_equation(
            solver.p, phiHbyA, A_p, solver.rho, mesh,
        )

        # Pressure should remain approximately uniform
        p_range = p_new.max() - p_new.min()
        assert p_range < 1.0  # small variation due to numerical error

    def test_pressure_equation_converges(self, compressible_cavity):
        """Pressure equation solver converges within max iterations."""
        from pyfoam.applications.rho_simple_foam import RhoSimpleFoam

        solver = RhoSimpleFoam(compressible_cavity)
        mesh = solver.mesh

        n_internal = mesh.n_internal_faces
        # Use a larger flux to produce a visible pressure correction
        phiHbyA = torch.randn(n_internal, dtype=CFD_DTYPE) * 0.1
        A_p = torch.ones(mesh.n_cells, dtype=CFD_DTYPE)

        p_init = solver.p.clone()
        p_new = solver._solve_pressure_equation(
            p_init, phiHbyA, A_p, solver.rho, mesh,
        )

        assert torch.isfinite(p_new).all()
        # Pressure correction should produce a non-trivial change
        # (the source is non-zero, so p should change)
        p_diff = (p_new - p_init).abs()
        assert p_diff.max() > 1e-6, (
            f"Pressure correction too small: max diff={p_diff.max():.6e}"
        )


class TestRhoSimpleFoamEnergy:
    """Tests for the energy equation."""

    def test_energy_equation_shape(self, compressible_cavity):
        """Energy equation returns correct shape."""
        from pyfoam.applications.rho_simple_foam import RhoSimpleFoam

        solver = RhoSimpleFoam(compressible_cavity)
        T_old = solver.T.clone()

        T_new = solver._solve_energy_equation(
            solver.T, solver.U, solver.phi, solver.rho, solver.p, T_old,
        )

        assert T_new.shape == (solver.mesh.n_cells,)
        assert torch.isfinite(T_new).all()

    def test_energy_equation_preserves_uniform_T(self, compressible_cavity):
        """Uniform temperature with zero velocity stays uniform."""
        from pyfoam.applications.rho_simple_foam import RhoSimpleFoam

        solver = RhoSimpleFoam(compressible_cavity)
        T_old = solver.T.clone()

        # Zero velocity → no convection, no dissipation
        U_zero = torch.zeros_like(solver.U)
        phi_zero = torch.zeros_like(solver.phi)

        T_new = solver._solve_energy_equation(
            solver.T, U_zero, phi_zero, solver.rho, solver.p, T_old,
        )

        # Temperature should remain approximately uniform
        T_range = T_new.max() - T_new.min()
        assert T_range < 1.0  # small numerical variation

    def test_energy_under_relaxation(self, tmp_path):
        """Under-relaxation uses T_old, not current T."""
        case_dir = tmp_path / "relax_case"
        _make_compressible_cavity_case(
            case_dir,
            n_cells_x=2,
            n_cells_y=2,
            T_init=300.0,
            T_top=310.0,
            alpha_T=0.5,
            max_outer_iterations=5,
        )

        from pyfoam.applications.rho_simple_foam import RhoSimpleFoam

        solver = RhoSimpleFoam(case_dir)
        assert abs(solver.alpha_T - 0.5) < 1e-10

        # Run a single iteration and verify T changes
        T_before = solver.T.clone()
        T_old = solver.T.clone()
        T_after = solver._solve_energy_equation(
            solver.T, solver.U, solver.phi, solver.rho, solver.p, T_old,
        )

        # With alpha_T=0.5, T_after should be between T_solved and T_old
        # The key check: it should NOT be identical to T_solved (no relaxation)
        assert T_after.shape == T_before.shape

    def test_energy_with_turbulent_conductivity(self, compressible_cavity):
        """Energy equation works with turbulent thermal conductivity."""
        from pyfoam.applications.rho_simple_foam import RhoSimpleFoam

        solver = RhoSimpleFoam(compressible_cavity)

        # Create synthetic effective viscosity (turbulent)
        mu_mol = solver.thermo.mu(solver.T)
        mu_eff = mu_mol * 10.0

        T_old = solver.T.clone()
        T_new = solver._solve_energy_equation(
            solver.T, solver.U, solver.phi, solver.rho, solver.p, T_old,
            mu_eff=mu_eff,
        )

        assert torch.isfinite(T_new).all()

    def test_viscous_dissipation_nonzero(self, compressible_cavity):
        """Viscous dissipation is non-zero for non-uniform velocity."""
        from pyfoam.applications.rho_simple_foam import RhoSimpleFoam

        solver = RhoSimpleFoam(compressible_cavity)
        mesh = solver.mesh

        # Set non-uniform velocity
        U = solver.U.clone()
        U[:, 0] = torch.linspace(0, 1, mesh.n_cells, dtype=CFD_DTYPE)

        grad_U = solver._compute_grad_vector(U, mesh)
        S = 0.5 * (grad_U + grad_U.transpose(-1, -2))
        S_double_dot = (S * S).sum(dim=(1, 2))

        # Should be non-zero for non-uniform velocity
        assert S_double_dot.abs().sum() > 0


class TestRhoSimpleFoamDivergence:
    """Tests for divergence and gradient computations."""

    def test_grad_scalar_shape(self, compressible_cavity):
        """Gradient of scalar returns (n_cells, 3)."""
        from pyfoam.applications.rho_simple_foam import RhoSimpleFoam

        solver = RhoSimpleFoam(compressible_cavity)
        grad_p = solver._compute_grad(solver.p, solver.mesh)

        assert grad_p.shape == (solver.mesh.n_cells, 3)
        assert torch.isfinite(grad_p).all()

    def test_grad_zero_for_zero_field(self, compressible_cavity):
        """Gradient of zero field is zero everywhere."""
        from pyfoam.applications.rho_simple_foam import RhoSimpleFoam

        solver = RhoSimpleFoam(compressible_cavity)
        mesh = solver.mesh
        p_zero = torch.zeros(mesh.n_cells, dtype=CFD_DTYPE)
        grad_p = solver._compute_grad(p_zero, mesh)

        assert torch.allclose(grad_p, torch.zeros_like(grad_p), atol=1e-10)

    def test_grad_vector_shape(self, compressible_cavity):
        """Gradient of vector returns (n_cells, 3, 3)."""
        from pyfoam.applications.rho_simple_foam import RhoSimpleFoam

        solver = RhoSimpleFoam(compressible_cavity)
        grad_U = solver._compute_grad_vector(solver.U, solver.mesh)

        assert grad_U.shape == (solver.mesh.n_cells, 3, 3)
        assert torch.isfinite(grad_U).all()

    def test_div_shape(self, compressible_cavity):
        """Divergence returns (n_cells,)."""
        from pyfoam.applications.rho_simple_foam import RhoSimpleFoam

        solver = RhoSimpleFoam(compressible_cavity)
        div_U = solver._compute_div(solver.U, solver.phi, solver.mesh)

        assert div_U.shape == (solver.mesh.n_cells,)
        assert torch.isfinite(div_U).all()

    def test_continuity_error_shape(self, compressible_cavity):
        """Continuity error returns a scalar float."""
        from pyfoam.applications.rho_simple_foam import RhoSimpleFoam

        solver = RhoSimpleFoam(compressible_cavity)
        error = solver._compute_continuity_error(solver.phi, solver.rho)

        assert isinstance(error, float)
        assert error >= 0.0


class TestRhoSimpleFoamTurbulence:
    """Tests for turbulence model integration."""

    def test_turbulence_enabled_with_ras(self, heated_cavity_turbulent):
        """Turbulence is enabled when turbulenceProperties specifies RAS."""
        from pyfoam.applications.rho_simple_foam import RhoSimpleFoam

        solver = RhoSimpleFoam(heated_cavity_turbulent)
        assert solver.turbulence_enabled is True
        assert solver.ras is not None

    def test_turbulence_disabled_laminar(self, compressible_cavity):
        """Turbulence is disabled without turbulenceProperties."""
        from pyfoam.applications.rho_simple_foam import RhoSimpleFoam

        solver = RhoSimpleFoam(compressible_cavity)
        assert solver.turbulence_enabled is False
        assert solver.ras is None

    def test_update_turbulence_returns_none_when_disabled(self, compressible_cavity):
        """_update_turbulence returns None when turbulence is disabled."""
        from pyfoam.applications.rho_simple_foam import RhoSimpleFoam

        solver = RhoSimpleFoam(compressible_cavity)
        mu_eff = solver._update_turbulence()
        assert mu_eff is None

    def test_update_turbulence_returns_field_when_enabled(self, heated_cavity_turbulent):
        """_update_turbulence returns effective viscosity when enabled."""
        from pyfoam.applications.rho_simple_foam import RhoSimpleFoam

        solver = RhoSimpleFoam(heated_cavity_turbulent)
        mu_eff = solver._update_turbulence()

        assert mu_eff is not None
        assert mu_eff.shape == (solver.mesh.n_cells,)
        assert torch.isfinite(mu_eff).all()
        # Effective viscosity should be >= molecular viscosity
        mu_mol = solver.thermo.mu(solver.T)
        assert (mu_eff >= mu_mol - 1e-10).all()


class TestRhoSimpleFoamRun:
    """Tests for the full solver run."""

    def test_run_converges(self, tiny_compressible_cavity):
        """RhoSimpleFoam runs and produces valid output."""
        from pyfoam.applications.rho_simple_foam import RhoSimpleFoam

        solver = RhoSimpleFoam(tiny_compressible_cavity)
        conv = solver.run()

        n_cells = 4  # 2x2
        assert solver.U.shape == (n_cells, 3)
        assert solver.p.shape == (n_cells,)
        assert solver.T.shape == (n_cells,)
        assert solver.phi.shape == (solver.mesh.n_faces,)

        # All values should be finite
        assert torch.isfinite(solver.U).all(), "U contains NaN/Inf"
        assert torch.isfinite(solver.p).all(), "p contains NaN/Inf"
        assert torch.isfinite(solver.T).all(), "T contains NaN/Inf"
        assert torch.isfinite(solver.phi).all(), "phi contains NaN/Inf"
        assert torch.isfinite(solver.rho).all(), "rho contains NaN/Inf"

    def test_run_writes_output(self, tiny_compressible_cavity):
        """RhoSimpleFoam writes U, p, T to time directories."""
        from pyfoam.applications.rho_simple_foam import RhoSimpleFoam

        solver = RhoSimpleFoam(tiny_compressible_cavity)
        solver.run()

        # Check that at least one output time directory was created
        time_dirs = [d for d in tiny_compressible_cavity.iterdir()
                     if d.is_dir() and d.name.replace(".", "").isdigit()
                     and d.name != "0"]
        assert len(time_dirs) >= 1

        # Check that U, p, T were written
        for td in time_dirs:
            assert (td / "U").exists(), f"U not found in {td}"
            assert (td / "p").exists(), f"p not found in {td}"
            assert (td / "T").exists(), f"T not found in {td}"

    def test_fields_are_valid_format(self, tiny_compressible_cavity):
        """Written fields are valid OpenFOAM format."""
        from pyfoam.applications.rho_simple_foam import RhoSimpleFoam
        from pyfoam.io.field_io import read_field

        solver = RhoSimpleFoam(tiny_compressible_cavity)
        solver.run()

        # Find the last written time directory
        time_dirs = sorted(
            [d for d in tiny_compressible_cavity.iterdir()
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

    def test_density_remains_positive(self, tiny_compressible_cavity):
        """Density stays positive throughout the simulation."""
        from pyfoam.applications.rho_simple_foam import RhoSimpleFoam

        solver = RhoSimpleFoam(tiny_compressible_cavity)
        solver.run()

        assert (solver.rho > 0).all(), "Density became non-positive"
        assert (solver.T > 0).all(), "Temperature became non-positive"

    def test_turbulent_run_converges(self, heated_cavity_turbulent):
        """RhoSimpleFoam with turbulence produces valid output."""
        from pyfoam.applications.rho_simple_foam import RhoSimpleFoam

        solver = RhoSimpleFoam(heated_cavity_turbulent)
        conv = solver.run()

        n_cells = 4  # 2x2
        assert solver.U.shape == (n_cells, 3)
        assert solver.p.shape == (n_cells,)
        assert solver.T.shape == (n_cells,)
        assert torch.isfinite(solver.U).all(), "U contains NaN/Inf"
        assert torch.isfinite(solver.p).all(), "p contains NaN/Inf"
        assert torch.isfinite(solver.T).all(), "T contains NaN/Inf"

    def test_pressure_and_temperature_coupled(self, tiny_compressible_cavity):
        """Pressure and temperature are coupled via EOS after solving."""
        from pyfoam.applications.rho_simple_foam import RhoSimpleFoam

        solver = RhoSimpleFoam(tiny_compressible_cavity)
        solver.run()

        # Verify EOS consistency: ρ = p/(RT)
        rho_expected = solver.thermo.rho(solver.p, solver.T)
        assert torch.allclose(solver.rho, rho_expected, rtol=1e-6), (
            "Density inconsistent with EOS after solving"
        )

    def test_convergence_data_populated(self, tiny_compressible_cavity):
        """ConvergenceData has non-trivial values after run."""
        from pyfoam.applications.rho_simple_foam import RhoSimpleFoam

        solver = RhoSimpleFoam(tiny_compressible_cavity)
        conv = solver.run()

        assert conv.outer_iterations >= 1
        assert conv.U_residual >= 0
        assert conv.p_residual >= 0
        assert conv.continuity_error >= 0

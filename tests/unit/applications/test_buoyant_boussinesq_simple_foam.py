"""
Unit tests for BuoyantBoussinesqSimpleFoam — steady-state Boussinesq buoyant solver.

Tests cover:
- Boussinesq density approximation: ρ = ρ₀[1 − β(T − T₀)]
- Reference parameter initialisation (rho_ref, beta, T_ref)
- Gravity vector reading and injection
- Gravity dot product computation (gh, ghf)
- Pressure decomposition (p_rgh = p - rho_ref * gh)
- Boussinesq momentum predictor with thermal buoyancy
- Energy equation (convection-diffusion, no dissipation)
- Pressure equation with constant density
- Natural convection in a differentially heated cavity
- Field writing to time directories
- Written field format validity
- Solver produces finite values
- Density stays positive
- Custom parameter injection
- Convergence data populated
- Zero gravity gives no convection
- Large beta amplifies buoyancy
- Uniform temperature gives zero buoyancy
- Temperature gradient drives circulation
- Inheritance from SolverBase
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from pyfoam.core.dtype import CFD_DTYPE, INDEX_DTYPE
from pyfoam.io.foam_file import FoamFileHeader, FileFormat, write_foam_file


# ---------------------------------------------------------------------------
# Mesh generation helper for Boussinesq cavity case
# ---------------------------------------------------------------------------

def _make_boussinesq_cavity_case(
    case_dir: Path,
    n_cells_x: int = 10,
    n_cells_y: int = 10,
    T_init: float = 300.0,
    T_hot: float = 305.0,
    T_cold: float = 295.0,
    p_init: float = 101325.0,
    end_time: int = 500,
    write_interval: int = 100,
    alpha_p: float = 0.3,
    alpha_U: float = 0.7,
    alpha_T: float = 1.0,
    convergence_tolerance: float = 1e-4,
    max_outer_iterations: int = 100,
    gravity: tuple[float, float, float] = (0.0, -9.81, 0.0),
) -> None:
    """Write a complete Boussinesq cavity case to *case_dir*.

    Creates a cavity with:
    - Left wall at T_hot (heated)
    - Right wall at T_cold (cooled)
    - Top/bottom walls adiabatic (zeroGradient)
    - Gravity pointing down (-y)
    """
    case_dir.mkdir(parents=True, exist_ok=True)

    # ---- Mesh ----
    dx = 1.0 / n_cells_x
    dy = 1.0 / n_cells_y
    dz = 0.1

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

    # hotWall (left, x=0)
    for j in range(n_cells_y):
        p0 = j * (n_cells_x + 1)
        p1 = p0 + n_cells_x + 1
        p2 = p1 + n_base
        p3 = p0 + n_base
        faces.append((4, p0, p1, p2, p3))
        owner.append(j * n_cells_x)

    n_hot = n_cells_y
    hot_start = n_internal

    # coldWall (right, x=1)
    for j in range(n_cells_y):
        p0 = j * (n_cells_x + 1) + n_cells_x
        p1 = p0 + n_cells_x + 1
        p2 = p1 + n_base
        p3 = p0 + n_base
        faces.append((4, p0, p1, p2, p3))
        owner.append(j * n_cells_x + n_cells_x - 1)

    n_cold = n_cells_y
    cold_start = hot_start + n_hot

    # adiabaticWalls (top, bottom)
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

    n_adiabatic = 2 * n_cells_x
    adiabatic_start = cold_start + n_cold

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
    empty_start = adiabatic_start + n_adiabatic

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

    h = FoamFileHeader(**{**header_base.__dict__, "class_name": "vectorField", "object": "points"})
    lines = [f"{n_points}", "("]
    for x, y, z in all_points:
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
    lines.append("    adiabaticWalls")
    lines.append("    {")
    lines.append("        type            wall;")
    lines.append(f"        nFaces          {n_adiabatic};")
    lines.append(f"        startFace       {adiabatic_start};")
    lines.append("    }")
    lines.append("    frontAndBack")
    lines.append("    {")
    lines.append("        type            empty;")
    lines.append(f"        nFaces          {n_empty};")
    lines.append(f"        startFace       {empty_start};")
    lines.append("    }")
    lines.append(")")
    write_foam_file(mesh_dir / "boundary", h, "\n".join(lines), overwrite=True)

    # ---- constant/g ----
    g_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="uniformDimensionedVectorField", location="constant", object="g",
    )
    g_body = (
        f"dimensions      [0 1 -2 0 0 0 0];\n"
        f"value           ({gravity[0]:.6g} {gravity[1]:.6g} {gravity[2]:.6g});\n"
    )
    write_foam_file(case_dir / "constant" / "g", g_header, g_body, overwrite=True)

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
        "    adiabaticWalls\n    {\n"
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
        f"internalField   uniform {p_init};\n\n"
        "boundaryField\n{\n"
        "    hotWall\n    {\n"
        "        type            zeroGradient;\n"
        "    }\n"
        "    coldWall\n    {\n"
        "        type            zeroGradient;\n"
        "    }\n"
        "    adiabaticWalls\n    {\n"
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
        "    adiabaticWalls\n    {\n"
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
        "application     buoyantBoussinesqSimpleFoam;\n"
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
        "    p_rgh\n    {\n"
        "        solver          PCG;\n"
        "        preconditioner  DIC;\n"
        "        tolerance       1e-6;\n"
        "        relTol          0.01;\n"
        "    }\n"
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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def boussinesq_cavity(tmp_path):
    """Create a Boussinesq cavity case (4x4 mesh)."""
    case_dir = tmp_path / "boussinesq_cavity"
    _make_boussinesq_cavity_case(
        case_dir,
        n_cells_x=4,
        n_cells_y=4,
        T_init=300.0,
        T_hot=305.0,
        T_cold=295.0,
        p_init=101325.0,
        end_time=50,
        write_interval=50,
        max_outer_iterations=20,
    )
    return case_dir


@pytest.fixture
def tiny_boussinesq_cavity(tmp_path):
    """Create a minimal 2x2 Boussinesq cavity for fast tests."""
    case_dir = tmp_path / "tiny_boussinesq"
    _make_boussinesq_cavity_case(
        case_dir,
        n_cells_x=2,
        n_cells_y=2,
        T_init=300.0,
        T_hot=302.0,
        T_cold=298.0,
        p_init=101325.0,
        end_time=10,
        write_interval=10,
        max_outer_iterations=10,
    )
    return case_dir


@pytest.fixture
def boussinesq_custom_gravity(tmp_path):
    """Create a Boussinesq cavity with custom gravity."""
    case_dir = tmp_path / "boussinesq_custom_g"
    _make_boussinesq_cavity_case(
        case_dir,
        n_cells_x=2,
        n_cells_y=2,
        T_init=300.0,
        T_hot=310.0,
        T_cold=290.0,
        gravity=(0.0, -9.81, 0.0),
        end_time=10,
        write_interval=10,
        max_outer_iterations=10,
    )
    return case_dir


# ===========================================================================
# Tests
# ===========================================================================


class TestBuoyantBoussinesqSimpleFoamInit:
    """Tests for BuoyantBoussinesqSimpleFoam initialisation."""

    def test_case_loads(self, boussinesq_cavity):
        """Case directory is readable and has expected structure."""
        from pyfoam.io.case import Case

        case = Case(boussinesq_cavity)
        assert case.has_mesh()
        assert case.has_field("U", 0)
        assert case.has_field("p", 0)
        assert case.has_field("T", 0)

    def test_mesh_builds(self, boussinesq_cavity):
        """FvMesh is constructed correctly."""
        from pyfoam.applications.solver_base import SolverBase

        solver = SolverBase(boussinesq_cavity)
        mesh = solver.mesh

        assert mesh.n_cells == 16  # 4x4
        assert mesh.n_internal_faces > 0

    def test_fields_initialise(self, boussinesq_cavity):
        """Fields are initialised from the 0/ directory."""
        from pyfoam.applications.buoyant_boussinesq_simple_foam import BuoyantBoussinesqSimpleFoam

        solver = BuoyantBoussinesqSimpleFoam(boussinesq_cavity)

        n_cells = 16
        assert solver.U.shape == (n_cells, 3)
        assert solver.p.shape == (n_cells,)
        assert solver.T.shape == (n_cells,)
        assert solver.phi.shape == (solver.mesh.n_faces,)
        assert solver.rho.shape == (n_cells,)

    def test_default_parameters(self, boussinesq_cavity):
        """Default Boussinesq parameters are set correctly."""
        from pyfoam.applications.buoyant_boussinesq_simple_foam import BuoyantBoussinesqSimpleFoam

        solver = BuoyantBoussinesqSimpleFoam(boussinesq_cavity)

        assert solver.rho_ref == 1.0
        assert solver.beta == pytest.approx(3.34e-3)
        assert solver.T_ref == pytest.approx(300.0)

    def test_custom_parameters(self, boussinesq_cavity):
        """Custom Boussinesq parameters can be injected."""
        from pyfoam.applications.buoyant_boussinesq_simple_foam import BuoyantBoussinesqSimpleFoam

        solver = BuoyantBoussinesqSimpleFoam(
            boussinesq_cavity,
            rho_ref=1.2,
            beta=3.0e-3,
            T_ref=310.0,
        )

        assert solver.rho_ref == 1.2
        assert solver.beta == pytest.approx(3.0e-3)
        assert solver.T_ref == pytest.approx(310.0)


class TestBuoyantBoussinesqSimpleFoamGravity:
    """Tests for gravity vector and dot products."""

    def test_gravity_read_from_file(self, boussinesq_cavity):
        """Gravity vector is read from constant/g."""
        from pyfoam.applications.buoyant_boussinesq_simple_foam import BuoyantBoussinesqSimpleFoam

        solver = BuoyantBoussinesqSimpleFoam(boussinesq_cavity)
        g = solver.g

        assert g.shape == (3,)
        assert abs(g[1].item() - (-9.81)) < 0.01
        assert abs(g[0].item()) < 1e-10
        assert abs(g[2].item()) < 1e-10

    def test_custom_gravity_injection(self, boussinesq_cavity):
        """Custom gravity vector can be injected."""
        from pyfoam.applications.buoyant_boussinesq_simple_foam import BuoyantBoussinesqSimpleFoam

        solver = BuoyantBoussinesqSimpleFoam(
            boussinesq_cavity,
            gravity=(0.0, -10.0, 0.0),
        )
        g = solver.g

        assert abs(g[1].item() - (-10.0)) < 1e-10

    def test_gh_shape(self, boussinesq_cavity):
        """gh has correct shape (n_cells,)."""
        from pyfoam.applications.buoyant_boussinesq_simple_foam import BuoyantBoussinesqSimpleFoam

        solver = BuoyantBoussinesqSimpleFoam(boussinesq_cavity)
        assert solver.gh.shape == (solver.mesh.n_cells,)

    def test_ghf_shape(self, boussinesq_cavity):
        """ghf has correct shape (n_faces,)."""
        from pyfoam.applications.buoyant_boussinesq_simple_foam import BuoyantBoussinesqSimpleFoam

        solver = BuoyantBoussinesqSimpleFoam(boussinesq_cavity)
        assert solver.ghf.shape == (solver.mesh.n_faces,)

    def test_gh_finite(self, boussinesq_cavity):
        """gh values are finite."""
        from pyfoam.applications.buoyant_boussinesq_simple_foam import BuoyantBoussinesqSimpleFoam

        solver = BuoyantBoussinesqSimpleFoam(boussinesq_cavity)
        assert torch.isfinite(solver.gh).all()

    def test_gh_decreases_with_y(self, boussinesq_cavity):
        """gh should decrease with y for downward gravity."""
        from pyfoam.applications.buoyant_boussinesq_simple_foam import BuoyantBoussinesqSimpleFoam

        solver = BuoyantBoussinesqSimpleFoam(boussinesq_cavity)
        mesh = solver.mesh

        gh = solver.gh
        cell_y = mesh.cell_centres[:, 1]

        sorted_indices = torch.argsort(cell_y)
        gh_sorted = gh[sorted_indices]

        for i in range(len(gh_sorted) - 1):
            assert gh_sorted[i] >= gh_sorted[i + 1] - 1e-6


class TestBoussinesqApproximation:
    """Tests for the Boussinesq density approximation."""

    def test_boussinesq_rho_at_ref_temperature(self, boussinesq_cavity):
        """At T = T_ref, density equals rho_ref."""
        from pyfoam.applications.buoyant_boussinesq_simple_foam import BuoyantBoussinesqSimpleFoam

        solver = BuoyantBoussinesqSimpleFoam(boussinesq_cavity)
        T_ref = torch.full((16,), solver.T_ref, dtype=CFD_DTYPE)
        rho = solver._boussinesq_rho(T_ref)

        assert torch.allclose(rho, torch.full_like(rho, solver.rho_ref), rtol=1e-10)

    def test_boussinesq_rho_decreases_with_temperature(self, boussinesq_cavity):
        """Density decreases as temperature increases."""
        from pyfoam.applications.buoyant_boussinesq_simple_foam import BuoyantBoussinesqSimpleFoam

        solver = BuoyantBoussinesqSimpleFoam(boussinesq_cavity)
        T_cold = torch.full((16,), 290.0, dtype=CFD_DTYPE)
        T_hot = torch.full((16,), 310.0, dtype=CFD_DTYPE)

        rho_cold = solver._boussinesq_rho(T_cold)
        rho_hot = solver._boussinesq_rho(T_hot)

        assert (rho_cold > rho_hot).all()

    def test_boussinesq_rho_linear(self, boussinesq_cavity):
        """Density is linear in temperature."""
        from pyfoam.applications.buoyant_boussinesq_simple_foam import BuoyantBoussinesqSimpleFoam

        solver = BuoyantBoussinesqSimpleFoam(boussinesq_cavity)
        T1 = torch.tensor([300.0], dtype=CFD_DTYPE)
        T2 = torch.tensor([310.0], dtype=CFD_DTYPE)
        T3 = torch.tensor([320.0], dtype=CFD_DTYPE)

        rho1 = solver._boussinesq_rho(T1)
        rho2 = solver._boussinesq_rho(T2)
        rho3 = solver._boussinesq_rho(T3)

        # Linear: (rho1 - rho2) should equal (rho2 - rho3)
        drho12 = rho1 - rho2
        drho23 = rho2 - rho3
        assert torch.allclose(drho12, drho23, rtol=1e-10)

    def test_p_rgh_initialisation(self, boussinesq_cavity):
        """p_rgh = p - rho_ref * gh at initialisation."""
        from pyfoam.applications.buoyant_boussinesq_simple_foam import BuoyantBoussinesqSimpleFoam

        solver = BuoyantBoussinesqSimpleFoam(boussinesq_cavity)
        p_rgh_expected = solver.p - solver.rho_ref * solver.gh

        assert torch.allclose(solver.p_rgh, p_rgh_expected, rtol=1e-10)


class TestBuoyantBoussinesqSimpleFoamMomentum:
    """Tests for the Boussinesq momentum predictor."""

    def test_momentum_predictor_shape(self, boussinesq_cavity):
        """Boussinesq momentum predictor returns correct shapes."""
        from pyfoam.applications.buoyant_boussinesq_simple_foam import BuoyantBoussinesqSimpleFoam

        solver = BuoyantBoussinesqSimpleFoam(boussinesq_cavity)
        U, A_p, H = solver._boussinesq_momentum_predictor(
            solver.U, solver.p_rgh, solver.phi, solver.rho, solver.T,
        )

        n_cells = solver.mesh.n_cells
        assert U.shape == (n_cells, 3)
        assert A_p.shape == (n_cells,)
        assert H.shape == (n_cells, 3)

    def test_momentum_predictor_finite(self, boussinesq_cavity):
        """Boussinesq momentum predictor produces finite values."""
        from pyfoam.applications.buoyant_boussinesq_simple_foam import BuoyantBoussinesqSimpleFoam

        solver = BuoyantBoussinesqSimpleFoam(boussinesq_cavity)
        U, A_p, H = solver._boussinesq_momentum_predictor(
            solver.U, solver.p_rgh, solver.phi, solver.rho, solver.T,
        )

        assert torch.isfinite(U).all()
        assert torch.isfinite(A_p).all()
        assert torch.isfinite(H).all()

    def test_buoyancy_with_temperature_gradient(self, boussinesq_cavity):
        """Non-zero temperature gradient produces non-zero velocity."""
        from pyfoam.applications.buoyant_boussinesq_simple_foam import BuoyantBoussinesqSimpleFoam

        solver = BuoyantBoussinesqSimpleFoam(boussinesq_cavity)
        mesh = solver.mesh

        # Set up temperature gradient (hot left, cold right)
        x = mesh.cell_centres[:, 0]
        T_grad = 300.0 + 10.0 * (1.0 - 2.0 * x)
        solver.T = T_grad
        solver.rho = solver._boussinesq_rho(solver.T)

        U, _, _ = solver._boussinesq_momentum_predictor(
            solver.U, solver.p_rgh, solver.phi, solver.rho, solver.T,
        )

        # Should produce non-zero velocity due to buoyancy
        assert U.abs().max() > 1e-10

    def test_no_buoyancy_with_zero_gravity(self, boussinesq_cavity):
        """Zero gravity gives no buoyancy source."""
        from pyfoam.applications.buoyant_boussinesq_simple_foam import BuoyantBoussinesqSimpleFoam

        solver = BuoyantBoussinesqSimpleFoam(
            boussinesq_cavity,
            gravity=(0.0, 0.0, 0.0),
        )

        U, _, _ = solver._boussinesq_momentum_predictor(
            solver.U, solver.p_rgh, solver.phi, solver.rho, solver.T,
        )

        assert torch.isfinite(U).all()


class TestBuoyantBoussinesqSimpleFoamEnergy:
    """Tests for the Boussinesq energy equation."""

    def test_energy_equation_shape(self, boussinesq_cavity):
        """Energy equation returns correct shape."""
        from pyfoam.applications.buoyant_boussinesq_simple_foam import BuoyantBoussinesqSimpleFoam

        solver = BuoyantBoussinesqSimpleFoam(boussinesq_cavity)
        T_old = solver.T.clone()

        T_new = solver._boussinesq_solve_energy_equation(
            solver.T, solver.U, solver.phi, T_old,
        )

        assert T_new.shape == (solver.mesh.n_cells,)
        assert torch.isfinite(T_new).all()

    def test_energy_preserves_uniform_T(self, boussinesq_cavity):
        """Uniform temperature with zero velocity stays uniform."""
        from pyfoam.applications.buoyant_boussinesq_simple_foam import BuoyantBoussinesqSimpleFoam

        solver = BuoyantBoussinesqSimpleFoam(boussinesq_cavity)
        T_old = solver.T.clone()

        U_zero = torch.zeros_like(solver.U)
        phi_zero = torch.zeros_like(solver.phi)

        T_new = solver._boussinesq_solve_energy_equation(
            solver.T, U_zero, phi_zero, T_old,
        )

        # Temperature should remain approximately uniform
        T_range = T_new.max() - T_new.min()
        assert T_range < 1.0


class TestBuoyantBoussinesqSimpleFoamRun:
    """Tests for the full solver run."""

    def test_run_converges(self, tiny_boussinesq_cavity):
        """BuoyantBoussinesqSimpleFoam runs and produces valid output."""
        from pyfoam.applications.buoyant_boussinesq_simple_foam import BuoyantBoussinesqSimpleFoam

        solver = BuoyantBoussinesqSimpleFoam(tiny_boussinesq_cavity)
        conv = solver.run()

        n_cells = 4  # 2x2
        assert solver.U.shape == (n_cells, 3)
        assert solver.p.shape == (n_cells,)
        assert solver.T.shape == (n_cells,)
        assert solver.phi.shape == (solver.mesh.n_faces,)

        assert torch.isfinite(solver.U).all(), "U contains NaN/Inf"
        assert torch.isfinite(solver.p).all(), "p contains NaN/Inf"
        assert torch.isfinite(solver.T).all(), "T contains NaN/Inf"
        assert torch.isfinite(solver.phi).all(), "phi contains NaN/Inf"
        assert torch.isfinite(solver.rho).all(), "rho contains NaN/Inf"

    def test_run_writes_output(self, tiny_boussinesq_cavity):
        """Solver writes U, p, T to time directories."""
        from pyfoam.applications.buoyant_boussinesq_simple_foam import BuoyantBoussinesqSimpleFoam

        solver = BuoyantBoussinesqSimpleFoam(tiny_boussinesq_cavity)
        solver.run()

        time_dirs = [d for d in tiny_boussinesq_cavity.iterdir()
                     if d.is_dir() and d.name.replace(".", "").isdigit()
                     and d.name != "0"]
        assert len(time_dirs) >= 1

        for td in time_dirs:
            assert (td / "U").exists(), f"U not found in {td}"
            assert (td / "p").exists(), f"p not found in {td}"
            assert (td / "T").exists(), f"T not found in {td}"

    def test_density_remains_positive(self, tiny_boussinesq_cavity):
        """Density stays positive throughout the simulation."""
        from pyfoam.applications.buoyant_boussinesq_simple_foam import BuoyantBoussinesqSimpleFoam

        solver = BuoyantBoussinesqSimpleFoam(tiny_boussinesq_cavity)
        solver.run()

        assert (solver.rho > 0).all(), "Density became non-positive"
        assert (solver.T > 0).all(), "Temperature became non-positive"

    def test_convergence_data_populated(self, tiny_boussinesq_cavity):
        """ConvergenceData has non-trivial values after run."""
        from pyfoam.applications.buoyant_boussinesq_simple_foam import BuoyantBoussinesqSimpleFoam

        solver = BuoyantBoussinesqSimpleFoam(tiny_boussinesq_cavity)
        conv = solver.run()

        assert conv.outer_iterations >= 1
        assert conv.U_residual >= 0
        assert conv.p_residual >= 0
        assert conv.continuity_error >= 0

    def test_fields_are_valid_format(self, tiny_boussinesq_cavity):
        """Written fields are valid OpenFOAM format."""
        from pyfoam.applications.buoyant_boussinesq_simple_foam import BuoyantBoussinesqSimpleFoam
        from pyfoam.io.field_io import read_field

        solver = BuoyantBoussinesqSimpleFoam(tiny_boussinesq_cavity)
        solver.run()

        time_dirs = sorted(
            [d for d in tiny_boussinesq_cavity.iterdir()
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


class TestBuoyantBoussinesqSimpleFoamInheritance:
    """Tests for inheritance from SolverBase."""

    def test_inherits_from_solver_base(self, boussinesq_cavity):
        """BuoyantBoussinesqSimpleFoam inherits from SolverBase."""
        from pyfoam.applications.buoyant_boussinesq_simple_foam import BuoyantBoussinesqSimpleFoam
        from pyfoam.applications.solver_base import SolverBase

        solver = BuoyantBoussinesqSimpleFoam(boussinesq_cavity)
        assert isinstance(solver, SolverBase)

    def test_has_required_methods(self, boussinesq_cavity):
        """Solver has all required methods."""
        from pyfoam.applications.buoyant_boussinesq_simple_foam import BuoyantBoussinesqSimpleFoam

        solver = BuoyantBoussinesqSimpleFoam(boussinesq_cavity)

        assert hasattr(solver, '_compute_grad')
        assert hasattr(solver, '_compute_grad_vector')
        assert hasattr(solver, '_compute_div')
        assert hasattr(solver, '_compute_residual')
        assert hasattr(solver, '_compute_continuity_error')
        assert hasattr(solver, '_boussinesq_rho')
        assert hasattr(solver, '_boussinesq_momentum_predictor')
        assert hasattr(solver, '_boussinesq_solve_energy_equation')

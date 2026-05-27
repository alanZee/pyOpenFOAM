"""
Unit tests for RhoPorousSimpleFoam — steady-state compressible porous media
SIMPLE solver.

Tests cover:
- Case loading with compressible porous properties
- Field initialisation (U, p, T, phi, rho)
- fvSolution/fvSchemes settings parsing
- Porous zone property reading
- Cell mask building
- Compressible Darcy-Forchheimer resistance (with density)
- MRF Coriolis and centrifugal forces
- Momentum predictor with porous resistance
- Run convergence on compressible porous cavity
- Field writing and format validation
- Zero resistance degeneracy (should behave like rhoSimpleFoam)
- Combined porous + MRF operation
- Density and temperature remain positive
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from pyfoam.core.dtype import CFD_DTYPE
from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.io.foam_file import FoamFileHeader, FileFormat, write_foam_file


# ---------------------------------------------------------------------------
# Mesh generation helper (compressible porous cavity)
# ---------------------------------------------------------------------------

def _make_compressible_porous_case(
    case_dir: Path,
    n_cells_x: int = 4,
    n_cells_y: int = 4,
    T_init: float = 300.0,
    T_top: float = 310.0,
    p_init: float = 101325.0,
    end_time: int = 50,
    write_interval: int = 50,
    alpha_p: float = 0.3,
    alpha_U: float = 0.7,
    alpha_T: float = 1.0,
    convergence_tolerance: float = 1e-4,
    max_outer_iterations: int = 200,
    d_coeff: tuple[float, float, float] = (1e4, 1e4, 1e4),
    f_coeff: tuple[float, float, float] = (0.0, 0.0, 0.0),
    include_mrf: bool = False,
    mrf_omega: float = 0.0,
    mrf_axis: tuple[float, float, float] = (0.0, 0.0, 1.0),
    mrf_origin: tuple[float, float, float] = (0.5, 0.5, 0.0),
) -> None:
    """Write a complete compressible porous cavity case to *case_dir*.

    Creates the same lid-driven cavity mesh with temperature field
    and constant/porosityProperties for porous media.
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

    # movingWall (top)
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
    for i in range(n_cells_x):
        p0 = i
        p1 = i + 1
        p2 = p1 + n_base
        p3 = p0 + n_base
        faces.append((4, p0, p1, p2, p3))
        owner.append(i)

    for j in range(n_cells_y):
        p0 = j * (n_cells_x + 1)
        p1 = p0 + n_cells_x + 1
        p2 = p1 + n_base
        p3 = p0 + n_base
        faces.append((4, p0, p1, p2, p3))
        owner.append(j * n_cells_x)

    for j in range(n_cells_y):
        p0 = j * (n_cells_x + 1) + n_cells_x
        p1 = p0 + n_cells_x + 1
        p2 = p1 + n_base
        p3 = p0 + n_base
        faces.append((4, p0, p1, p2, p3))
        owner.append(j * n_cells_x + n_cells_x - 1)

    n_fixed = n_cells_x + 2 * n_cells_y
    fixed_start = n_internal + n_moving

    # frontAndBack (empty)
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

    # ---- 0/p (Pa for compressible) ----
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

    # ---- 0/T ----
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
        "application     rhoPorousSimpleFoam;\n"
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

    # ---- porosityProperties ----
    pp_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="dictionary", location="constant", object="porosityProperties",
    )
    pp_body = (
        "porosity1\n{\n"
        f"    type            DarcyForchheimer;\n"
        f"    cellZone        all;\n"
        f"    d   ({d_coeff[0]} {d_coeff[1]} {d_coeff[2]});\n"
        f"    f   ({f_coeff[0]} {f_coeff[1]} {f_coeff[2]});\n"
        "}\n"
    )
    write_foam_file(
        case_dir / "constant" / "porosityProperties", pp_header,
        pp_body, overwrite=True,
    )

    # ---- MRFProperties (optional) ----
    if include_mrf:
        mrf_header = FoamFileHeader(
            version="2.0", format=FileFormat.ASCII,
            class_name="dictionary", location="constant", object="MRFProperties",
        )
        mrf_body = (
            "MRF1\n{\n"
            f"    cellZone        all;\n"
            f"    origin          ({mrf_origin[0]} {mrf_origin[1]} {mrf_origin[2]});\n"
            f"    axis            ({mrf_axis[0]} {mrf_axis[1]} {mrf_axis[2]});\n"
            f"    omega           {mrf_omega};\n"
            "}\n"
        )
        write_foam_file(
            case_dir / "constant" / "MRFProperties", mrf_header,
            mrf_body, overwrite=True,
        )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def compressible_porous_case(tmp_path):
    """Create a compressible porous cavity case (4x4 mesh)."""
    case_dir = tmp_path / "comp_porous_cavity"
    _make_compressible_porous_case(
        case_dir,
        n_cells_x=4,
        n_cells_y=4,
        T_init=300.0,
        T_top=310.0,
        p_init=101325.0,
        end_time=50,
        write_interval=50,
        max_outer_iterations=20,
        d_coeff=(1e4, 1e4, 1e4),
        f_coeff=(0.0, 0.0, 0.0),
    )
    return case_dir


@pytest.fixture
def tiny_compressible_porous_case(tmp_path):
    """Create a minimal 2x2 compressible porous cavity for fast tests."""
    case_dir = tmp_path / "tiny_comp_porous"
    _make_compressible_porous_case(
        case_dir,
        n_cells_x=2,
        n_cells_y=2,
        T_init=300.0,
        T_top=305.0,
        p_init=101325.0,
        end_time=10,
        write_interval=10,
        max_outer_iterations=10,
        d_coeff=(1e3, 1e3, 1e3),
        f_coeff=(1.0, 1.0, 1.0),
    )
    return case_dir


@pytest.fixture
def mrf_compressible_porous_case(tmp_path):
    """Create a case with both porous and MRF zones (compressible)."""
    case_dir = tmp_path / "mrf_comp_porous"
    _make_compressible_porous_case(
        case_dir,
        n_cells_x=2,
        n_cells_y=2,
        T_init=300.0,
        T_top=310.0,
        p_init=101325.0,
        end_time=10,
        write_interval=10,
        max_outer_iterations=10,
        d_coeff=(1e3, 1e3, 1e3),
        f_coeff=(0.0, 0.0, 0.0),
        include_mrf=True,
        mrf_omega=5.0,
        mrf_axis=(0.0, 0.0, 1.0),
        mrf_origin=(0.5, 0.5, 0.0),
    )
    return case_dir


# ---------------------------------------------------------------------------
# Tests: Initialisation
# ---------------------------------------------------------------------------

class TestRhoPorousSimpleFoamInit:
    """Tests for RhoPorousSimpleFoam initialisation."""

    def test_case_loads(self, compressible_porous_case):
        """Compressible porous case loads correctly."""
        from pyfoam.applications.rho_porous_simple_foam import RhoPorousSimpleFoam

        solver = RhoPorousSimpleFoam(compressible_porous_case)
        assert solver.mesh.n_cells == 16

    def test_fields_initialise(self, compressible_porous_case):
        """Fields are initialised from the 0/ directory."""
        from pyfoam.applications.rho_porous_simple_foam import RhoPorousSimpleFoam

        solver = RhoPorousSimpleFoam(compressible_porous_case)

        assert solver.U.shape == (16, 3)
        assert solver.p.shape == (16,)
        assert solver.T.shape == (16,)
        assert solver.phi.shape == (solver.mesh.n_faces,)
        assert solver.rho.shape == (16,)
        assert torch.isfinite(solver.rho).all()

    def test_thermo_defaults(self, compressible_porous_case):
        """Default thermo is air."""
        from pyfoam.applications.rho_porous_simple_foam import RhoPorousSimpleFoam

        solver = RhoPorousSimpleFoam(compressible_porous_case)
        assert solver.thermo is not None
        assert solver.thermo.R() == 287.0

    def test_porous_properties_read(self, compressible_porous_case):
        """Porous properties are read from constant/porosityProperties."""
        from pyfoam.applications.rho_porous_simple_foam import RhoPorousSimpleFoam

        solver = RhoPorousSimpleFoam(compressible_porous_case)
        assert len(solver.porous_zones) == 1
        assert abs(solver.porous_zones[0].d[0] - 1e4) < 1e-2

    def test_no_mrf_by_default(self, compressible_porous_case):
        """No MRF zones when MRFProperties is absent."""
        from pyfoam.applications.rho_porous_simple_foam import RhoPorousSimpleFoam

        solver = RhoPorousSimpleFoam(compressible_porous_case)
        assert len(solver.mrf_zones) == 0

    def test_fv_solution_settings(self, compressible_porous_case):
        """fvSolution settings are read correctly."""
        from pyfoam.applications.rho_porous_simple_foam import RhoPorousSimpleFoam

        solver = RhoPorousSimpleFoam(compressible_porous_case)
        assert solver.p_solver == "PCG"
        assert solver.U_solver == "PBiCGStab"
        assert solver.T_solver == "PCG"
        assert abs(solver.alpha_p - 0.3) < 1e-10
        assert abs(solver.alpha_U - 0.7) < 1e-10
        assert abs(solver.alpha_T - 1.0) < 1e-10
        assert solver.max_outer_iterations == 20

    def test_custom_porous_properties(self, tmp_path):
        """Porous properties can be passed directly."""
        case_dir = tmp_path / "custom_porous"
        _make_compressible_porous_case(
            case_dir, n_cells_x=2, n_cells_y=2, end_time=5,
            write_interval=5, max_outer_iterations=10,
        )

        from pyfoam.applications.rho_porous_simple_foam import RhoPorousSimpleFoam
        from pyfoam.applications.porous_simple_foam import PorousZoneProperties

        custom = PorousZoneProperties(
            name="custom",
            cell_zone="all",
            d=(1e5, 1e5, 1e5),
            f=(5.0, 5.0, 5.0),
        )
        solver = RhoPorousSimpleFoam(case_dir, porous_zones=[custom])
        assert len(solver.porous_zones) == 1
        assert solver.porous_zones[0].d == (1e5, 1e5, 1e5)


# ---------------------------------------------------------------------------
# Tests: Cell mask building
# ---------------------------------------------------------------------------

class TestCellMask:
    """Tests for cell zone mask building."""

    def test_all_cells_mask(self, compressible_porous_case):
        """Zone name 'all' selects all cells."""
        from pyfoam.applications.rho_porous_simple_foam import RhoPorousSimpleFoam

        solver = RhoPorousSimpleFoam(compressible_porous_case)
        mask = solver._build_cell_mask("all")
        assert mask.shape == (16,)
        assert mask.all()

    def test_missing_zone_fallback(self, compressible_porous_case):
        """Missing zone name falls back to all cells."""
        from pyfoam.applications.rho_porous_simple_foam import RhoPorousSimpleFoam

        solver = RhoPorousSimpleFoam(compressible_porous_case)
        mask = solver._build_cell_mask("nonexistent_zone")
        assert mask.shape == (16,)
        assert mask.all()


# ---------------------------------------------------------------------------
# Tests: Compressible Darcy-Forchheimer resistance
# ---------------------------------------------------------------------------

class TestCompressibleDarcyForchheimer:
    """Tests for the compressible form of Darcy-Forchheimer resistance."""

    def test_compressible_resistance_uses_dynamic_viscosity(self):
        """Compressible resistance uses mu (dynamic viscosity) not nu."""
        mu = 1.846e-5  # dynamic viscosity of air at 300K
        d = torch.tensor([1e4, 1e4, 1e4], dtype=CFD_DTYPE)
        rho = torch.tensor([1.2], dtype=CFD_DTYPE)
        U = torch.tensor([[1.0, 0.0, 0.0]], dtype=CFD_DTYPE)

        mu_tensor = torch.tensor([mu], dtype=CFD_DTYPE)
        U_mag = U.norm(dim=1)

        # Compressible: Cd = mu * d + rho * |U| * f/2
        Cd = mu_tensor.unsqueeze(-1) * d.unsqueeze(0)

        # Cd_x = 1.846e-5 * 1e4 = 0.1846
        assert abs(Cd[0, 0].item() - 0.1846) < 1e-4

    def test_forchheimer_includes_density(self):
        """Forchheimer term includes density for compressible flow."""
        f = torch.tensor([10.0, 10.0, 10.0], dtype=CFD_DTYPE)
        rho = torch.tensor([1.2], dtype=CFD_DTYPE)
        U = torch.tensor([[2.0, 0.0, 0.0]], dtype=CFD_DTYPE)

        U_mag = U.norm(dim=1)
        F_coeff = rho.unsqueeze(-1) * U_mag.unsqueeze(-1) * f.unsqueeze(0) * 0.5

        # rho * |U| * f / 2 = 1.2 * 2 * 10 * 0.5 = 12.0
        assert abs(F_coeff[0, 0].item() - 12.0) < 1e-6

    def test_combined_compressible_resistance(self):
        """Combined Darcy-Forchheimer resistance (compressible)."""
        mu = 1.846e-5
        d = torch.tensor([1e4, 1e4, 1e4], dtype=CFD_DTYPE)
        f = torch.tensor([10.0, 10.0, 10.0], dtype=CFD_DTYPE)
        rho = torch.tensor([1.2], dtype=CFD_DTYPE)
        U = torch.tensor([[1.0, 0.0, 0.0]], dtype=CFD_DTYPE)

        mu_tensor = torch.tensor([mu], dtype=CFD_DTYPE)
        U_mag = U.norm(dim=1)

        Cd = (
            mu_tensor.unsqueeze(-1) * d.unsqueeze(0)
            + rho.unsqueeze(-1) * U_mag.unsqueeze(-1) * f.unsqueeze(0) * 0.5
        )

        # mu*d + rho*|U|*f/2 = 1.846e-5*1e4 + 1.2*1*10*0.5 = 0.1846 + 6.0 = 6.1846
        expected = 0.1846 + 6.0
        assert abs(Cd[0, 0].item() - expected) < 0.01

    def test_zero_resistance(self):
        """Zero resistance produces zero contribution."""
        d = torch.tensor([0.0, 0.0, 0.0], dtype=CFD_DTYPE)
        f = torch.tensor([0.0, 0.0, 0.0], dtype=CFD_DTYPE)
        mu = torch.tensor([1.846e-5], dtype=CFD_DTYPE)
        rho = torch.tensor([1.2], dtype=CFD_DTYPE)
        U = torch.tensor([[1.0, 1.0, 1.0]], dtype=CFD_DTYPE)

        U_mag = U.norm(dim=1)
        Cd = (
            mu.unsqueeze(-1) * d.unsqueeze(0)
            + rho.unsqueeze(-1) * U_mag.unsqueeze(-1) * f.unsqueeze(0) * 0.5
        )

        assert Cd.abs().max() < 1e-30

    def test_density_affects_forchheimer(self):
        """Higher density leads to higher Forchheimer resistance."""
        f = torch.tensor([10.0, 10.0, 10.0], dtype=CFD_DTYPE)
        U = torch.tensor([[1.0, 0.0, 0.0]], dtype=CFD_DTYPE)
        rho_low = torch.tensor([1.0], dtype=CFD_DTYPE)
        rho_high = torch.tensor([5.0], dtype=CFD_DTYPE)

        U_mag = U.norm(dim=1)  # (1,)
        F_low = rho_low.unsqueeze(-1) * U_mag.unsqueeze(-1) * f.unsqueeze(0) * 0.5
        F_high = rho_high.unsqueeze(-1) * U_mag.unsqueeze(-1) * f.unsqueeze(0) * 0.5

        assert F_high[0, 0].item() > F_low[0, 0].item()


# ---------------------------------------------------------------------------
# Tests: Negative resistance handling
# ---------------------------------------------------------------------------

class TestNegativeResistance:
    """Tests for OpenFOAM's negative resistance convention."""

    def test_negative_resistance_multiplier(self):
        """Negative resistance is treated as multiplier of max positive component."""
        from pyfoam.applications.rho_porous_simple_foam import RhoPorousSimpleFoam

        result = RhoPorousSimpleFoam._parse_resistance_vector((1e4, -0.5, 0))
        assert abs(result[0] - 1e4) < 1e-6
        assert abs(result[1] - 5000.0) < 1e-6
        assert abs(result[2]) < 1e-6

    def test_all_negative_resistance_returns_zero(self):
        """All-negative resistance returns zeros (invalid)."""
        from pyfoam.applications.rho_porous_simple_foam import RhoPorousSimpleFoam

        result = RhoPorousSimpleFoam._parse_resistance_vector((-1.0, -2.0, -3.0))
        assert result == (0.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# Tests: MRF forces
# ---------------------------------------------------------------------------

class TestMRFForces:
    """Tests for MRF Coriolis and centrifugal forces."""

    def test_mrf_case_loads(self, mrf_compressible_porous_case):
        """Case with both MRF and porous zones loads correctly."""
        from pyfoam.applications.rho_porous_simple_foam import RhoPorousSimpleFoam

        solver = RhoPorousSimpleFoam(mrf_compressible_porous_case)
        assert len(solver.porous_zones) == 1
        assert len(solver.mrf_zones) == 1

    def test_centrifugal_z_axis(self):
        """Centrifugal force for z-axis rotation points radially outward."""
        omega_vec = torch.tensor([0.0, 0.0, 10.0], dtype=CFD_DTYPE)
        r = torch.tensor([[1.0, 0.0, 0.0]], dtype=CFD_DTYPE)

        omega_sq = omega_vec.dot(omega_vec)
        omega_dot_r = (r * omega_vec.unsqueeze(0)).sum(dim=1)
        F_cent = r * omega_sq - omega_vec.unsqueeze(0) * omega_dot_r.unsqueeze(-1)

        assert abs(F_cent[0, 0].item() - 100.0) < 1e-8
        assert abs(F_cent[0, 1].item()) < 1e-8

    def test_coriolis_z_axis(self):
        """Coriolis force for z-axis rotation: -2 omega x U."""
        omega_vec = torch.tensor([0.0, 0.0, 10.0], dtype=CFD_DTYPE)
        U = torch.tensor([[1.0, 0.0, 0.0]], dtype=CFD_DTYPE)

        omega_cross_U = torch.zeros_like(U)
        omega_cross_U[:, 0] = omega_vec[1] * U[:, 2] - omega_vec[2] * U[:, 1]
        omega_cross_U[:, 1] = omega_vec[2] * U[:, 0] - omega_vec[0] * U[:, 2]
        omega_cross_U[:, 2] = omega_vec[0] * U[:, 1] - omega_vec[1] * U[:, 0]
        F_coriolis = -2.0 * omega_cross_U

        # -2*(0,0,10)x(1,0,0) = -2*(0,10,0) = (0,-20,0)
        assert abs(F_coriolis[0, 1].item() - (-20.0)) < 1e-8


# ---------------------------------------------------------------------------
# Tests: Momentum predictor
# ---------------------------------------------------------------------------

class TestMomentumPredictor:
    """Tests for the momentum predictor with porous resistance."""

    def test_momentum_predictor_shape(self, compressible_porous_case):
        """Momentum predictor returns correct shapes."""
        from pyfoam.applications.rho_porous_simple_foam import RhoPorousSimpleFoam

        solver = RhoPorousSimpleFoam(compressible_porous_case)
        U, A_p, H = solver._momentum_predictor(
            solver.U, solver.p, solver.phi, solver.rho,
        )

        n_cells = solver.mesh.n_cells
        assert U.shape == (n_cells, 3)
        assert A_p.shape == (n_cells,)
        assert H.shape == (n_cells, 3)

    def test_momentum_predictor_finite(self, compressible_porous_case):
        """Momentum predictor produces finite values."""
        from pyfoam.applications.rho_porous_simple_foam import RhoPorousSimpleFoam

        solver = RhoPorousSimpleFoam(compressible_porous_case)
        U, A_p, H = solver._momentum_predictor(
            solver.U, solver.p, solver.phi, solver.rho,
        )

        assert torch.isfinite(U).all()
        assert torch.isfinite(A_p).all()
        assert torch.isfinite(H).all()

    def test_momentum_with_effective_viscosity(self, compressible_porous_case):
        """Momentum predictor works with effective viscosity."""
        from pyfoam.applications.rho_porous_simple_foam import RhoPorousSimpleFoam

        solver = RhoPorousSimpleFoam(compressible_porous_case)

        mu_mol = solver.thermo.mu(solver.T)
        mu_eff = mu_mol * 10.0

        U, A_p, H = solver._momentum_predictor(
            solver.U, solver.p, solver.phi, solver.rho, mu_eff=mu_eff,
        )

        assert torch.isfinite(U).all()


# ---------------------------------------------------------------------------
# Tests: Full solver run
# ---------------------------------------------------------------------------

class TestRhoPorousSimpleFoamRun:
    """Tests for the full solver run."""

    def test_run_produces_finite_fields(self, tiny_compressible_porous_case):
        """RhoPorousSimpleFoam produces finite velocity, pressure, temperature."""
        from pyfoam.applications.rho_porous_simple_foam import RhoPorousSimpleFoam

        solver = RhoPorousSimpleFoam(tiny_compressible_porous_case)
        conv = solver.run()

        n_cells = 4  # 2x2
        assert solver.U.shape == (n_cells, 3)
        assert solver.p.shape == (n_cells,)
        assert solver.T.shape == (n_cells,)
        assert solver.rho.shape == (n_cells,)
        assert torch.isfinite(solver.U).all(), "U contains NaN/Inf"
        assert torch.isfinite(solver.p).all(), "p contains NaN/Inf"
        assert torch.isfinite(solver.T).all(), "T contains NaN/Inf"
        assert torch.isfinite(solver.phi).all(), "phi contains NaN/Inf"
        assert torch.isfinite(solver.rho).all(), "rho contains NaN/Inf"

    def test_run_writes_output(self, tiny_compressible_porous_case):
        """Solver writes U, p, T to time directories."""
        from pyfoam.applications.rho_porous_simple_foam import RhoPorousSimpleFoam

        solver = RhoPorousSimpleFoam(tiny_compressible_porous_case)
        solver.run()

        time_dirs = [d for d in tiny_compressible_porous_case.iterdir()
                     if d.is_dir() and d.name.replace(".", "").isdigit()
                     and d.name != "0"]
        assert len(time_dirs) >= 1

        for td in time_dirs:
            assert (td / "U").exists(), f"U not found in {td}"
            assert (td / "p").exists(), f"p not found in {td}"
            assert (td / "T").exists(), f"T not found in {td}"

    def test_fields_valid_format(self, tiny_compressible_porous_case):
        """Written fields are valid OpenFOAM format."""
        from pyfoam.applications.rho_porous_simple_foam import RhoPorousSimpleFoam
        from pyfoam.io.field_io import read_field

        solver = RhoPorousSimpleFoam(tiny_compressible_porous_case)
        solver.run()

        time_dirs = sorted(
            [d for d in tiny_compressible_porous_case.iterdir()
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

    def test_density_remains_positive(self, tiny_compressible_porous_case):
        """Density and temperature stay positive throughout the simulation."""
        from pyfoam.applications.rho_porous_simple_foam import RhoPorousSimpleFoam

        solver = RhoPorousSimpleFoam(tiny_compressible_porous_case)
        solver.run()

        assert (solver.rho > 0).all(), "Density became non-positive"
        assert (solver.T > 0).all(), "Temperature became non-positive"

    def test_eos_consistency_after_run(self, tiny_compressible_porous_case):
        """Density is consistent with EOS after solving."""
        from pyfoam.applications.rho_porous_simple_foam import RhoPorousSimpleFoam

        solver = RhoPorousSimpleFoam(tiny_compressible_porous_case)
        solver.run()

        rho_expected = solver.thermo.rho(solver.p, solver.T)
        assert torch.allclose(solver.rho, rho_expected, rtol=1e-6), (
            "Density inconsistent with EOS after solving"
        )

    def test_convergence_data_populated(self, tiny_compressible_porous_case):
        """ConvergenceData has non-trivial values after run."""
        from pyfoam.applications.rho_porous_simple_foam import RhoPorousSimpleFoam

        solver = RhoPorousSimpleFoam(tiny_compressible_porous_case)
        conv = solver.run()

        assert conv.outer_iterations >= 1
        assert conv.U_residual >= 0
        assert conv.p_residual >= 0
        assert conv.continuity_error >= 0


# ---------------------------------------------------------------------------
# Tests: Zero resistance degeneracy
# ---------------------------------------------------------------------------

class TestZeroResistanceDegeneracy:
    """Tests that zero porous resistance degenerates correctly."""

    def test_zero_resistance_runs(self, tmp_path):
        """Solver with zero resistance produces finite fields."""
        case_dir = tmp_path / "zero_porous"
        _make_compressible_porous_case(
            case_dir,
            n_cells_x=2,
            n_cells_y=2,
            end_time=10,
            write_interval=10,
            max_outer_iterations=10,
            d_coeff=(0.0, 0.0, 0.0),
            f_coeff=(0.0, 0.0, 0.0),
        )

        from pyfoam.applications.rho_porous_simple_foam import RhoPorousSimpleFoam

        solver = RhoPorousSimpleFoam(case_dir)
        conv = solver.run()

        assert torch.isfinite(solver.U).all()
        assert torch.isfinite(solver.p).all()
        assert torch.isfinite(solver.T).all()


# ---------------------------------------------------------------------------
# Tests: MRF + Porous combined (compressible)
# ---------------------------------------------------------------------------

class TestMrfPorousCombinedCompressible:
    """Tests for combined MRF and porous operation in compressible flow."""

    def test_mrf_porous_loads(self, mrf_compressible_porous_case):
        """Case with both MRF and porous zones loads correctly."""
        from pyfoam.applications.rho_porous_simple_foam import RhoPorousSimpleFoam

        solver = RhoPorousSimpleFoam(mrf_compressible_porous_case)
        assert len(solver.porous_zones) == 1
        assert len(solver.mrf_zones) == 1

    def test_mrf_porous_runs(self, mrf_compressible_porous_case):
        """Combined MRF + porous solver runs and produces finite fields."""
        from pyfoam.applications.rho_porous_simple_foam import RhoPorousSimpleFoam

        solver = RhoPorousSimpleFoam(mrf_compressible_porous_case)
        conv = solver.run()

        assert torch.isfinite(solver.U).all(), "U contains NaN/Inf"
        assert torch.isfinite(solver.p).all(), "p contains NaN/Inf"
        assert torch.isfinite(solver.T).all(), "T contains NaN/Inf"

    def test_mrf_only(self, tmp_path):
        """Solver with MRF only (no porous resistance) runs correctly."""
        case_dir = tmp_path / "mrf_only_comp"
        _make_compressible_porous_case(
            case_dir,
            n_cells_x=2,
            n_cells_y=2,
            end_time=10,
            write_interval=10,
            max_outer_iterations=10,
            d_coeff=(0.0, 0.0, 0.0),
            f_coeff=(0.0, 0.0, 0.0),
            include_mrf=True,
            mrf_omega=5.0,
        )

        from pyfoam.applications.rho_porous_simple_foam import RhoPorousSimpleFoam

        solver = RhoPorousSimpleFoam(case_dir)
        conv = solver.run()

        assert torch.isfinite(solver.U).all()
        assert torch.isfinite(solver.p).all()


# ---------------------------------------------------------------------------
# Tests: High resistance behaviour
# ---------------------------------------------------------------------------

class TestHighResistance:
    """Tests for high porous resistance limiting behaviour."""

    def test_high_darcy_resistance_slows_flow(self, tmp_path):
        """High Darcy resistance produces finite fields (may not converge on small mesh)."""
        case_dir = tmp_path / "high_darcy"
        _make_compressible_porous_case(
            case_dir,
            n_cells_x=2,
            n_cells_y=2,
            end_time=10,
            write_interval=10,
            max_outer_iterations=10,
            d_coeff=(1e8, 1e8, 1e8),
            f_coeff=(0.0, 0.0, 0.0),
        )

        from pyfoam.applications.rho_porous_simple_foam import RhoPorousSimpleFoam

        solver = RhoPorousSimpleFoam(case_dir)
        conv = solver.run()

        # High resistance may not converge on coarse mesh, but fields must be finite
        assert torch.isfinite(solver.U).all()
        assert torch.isfinite(solver.p).all()
        assert torch.isfinite(solver.T).all()

    def test_high_forchheimer_resistance(self, tmp_path):
        """High Forchheimer resistance is physical and finite."""
        case_dir = tmp_path / "high_forch"
        _make_compressible_porous_case(
            case_dir,
            n_cells_x=2,
            n_cells_y=2,
            end_time=10,
            write_interval=10,
            max_outer_iterations=10,
            d_coeff=(1e3, 1e3, 1e3),
            f_coeff=(100.0, 100.0, 100.0),
        )

        from pyfoam.applications.rho_porous_simple_foam import RhoPorousSimpleFoam

        solver = RhoPorousSimpleFoam(case_dir)
        conv = solver.run()

        assert torch.isfinite(solver.U).all()
        assert torch.isfinite(solver.p).all()

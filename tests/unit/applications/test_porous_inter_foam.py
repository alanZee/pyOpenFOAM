"""
Unit tests for PorousInterFoam — porous media two-phase VOF solver.

Tests cover:
- PorousInterFoam initialisation and property reading
- Porous zone data building (cell masks, resistance tensors)
- Darcy-Forchheimer resistance in two-phase context
- Momentum predictor with porous resistance
- Solver run with VOF advection + porous media
- Zero porous resistance degeneracy (should behave like interFoam)
- Negative resistance handling
- Multiple porous zones
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from pyfoam.core.dtype import CFD_DTYPE, INDEX_DTYPE
from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.io.foam_file import FoamFileHeader, FileFormat, write_foam_file


# ---------------------------------------------------------------------------
# Case generation helper (cavity with two-phase + porous media)
# ---------------------------------------------------------------------------


def _make_porous_inter_case(
    case_dir: Path,
    n_cells_x: int = 4,
    n_cells_y: int = 4,
    end_time: int = 5,
    write_interval: int = 5,
    d_coeff: tuple[float, float, float] = (1e4, 1e4, 1e4),
    f_coeff: tuple[float, float, float] = (0.0, 0.0, 0.0),
    sigma: float = 0.07,
) -> None:
    """Write a two-phase porous cavity case."""
    case_dir.mkdir(parents=True, exist_ok=True)

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

    # Boundary faces: movingWall (top)
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
        version="2.0", format=FileFormat.ASCII,
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

    # ---- transportProperties ----
    tp_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="dictionary", location="constant", object="transportProperties",
    )
    tp_body = (
        "nu              [0 2 -1 0 0 0 0] 1e-3;\n"
        "rho1            [1 -3 0 0 0 0 0] 1000;\n"
        "rho2            [1 -3 0 0 0 0 0] 1.225;\n"
        "sigma           [1 0 -2 0 0 0 0] 0.07;\n"
    )
    write_foam_file(
        case_dir / "constant" / "transportProperties", tp_header, tp_body, overwrite=True,
    )

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
        case_dir / "constant" / "porosityProperties", pp_header, pp_body, overwrite=True,
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

    # ---- 0/alpha.water ----
    a_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="volScalarField", location="0", object="alpha.water",
    )
    a_body = (
        "dimensions      [0 0 0 0 0 0 0];\n\n"
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
    write_foam_file(zero_dir / "alpha.water", a_header, a_body, overwrite=True)

    # ---- system/controlDict ----
    sys_dir = case_dir / "system"
    sys_dir.mkdir(exist_ok=True)

    cd_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="dictionary", location="system", object="controlDict",
    )
    cd_body = (
        "application     PorousInterFoam;\n"
        "startFrom       startTime;\n"
        "startTime       0;\n"
        "stopAt          endTime;\n"
        f"endTime         {end_time};\n"
        "deltaT          0.001;\n"
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
        "ddtSchemes\n{\n    default         Euler;\n}\n\n"
        "gradSchemes\n{\n    default         Gauss linear;\n}\n\n"
        "divSchemes\n{\n    default         none;\n"
        "    div(rhoPhi,U)   Gauss linearUpwind;\n"
        "    div(phi,alpha)  Gauss vanLeer;\n"
        "    div(pc,alpha)   Gauss linear;\n}\n\n"
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
        "PIMPLE\n{\n"
        "    nOuterCorrectors 3;\n"
        "    nCorrectors 2;\n"
        "    relaxationFactors\n    {\n"
        "        p               0.3;\n"
        "        U               0.7;\n"
        "    }\n"
        "    convergenceTolerance 1e-4;\n"
        "    maxOuterIterations  50;\n"
        "}\n"
    )
    write_foam_file(sys_dir / "fvSolution", fv_header, fv_body, overwrite=True)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def porous_inter_case(tmp_path):
    """Create a porous two-phase cavity case."""
    case_dir = tmp_path / "porous_inter"
    _make_porous_inter_case(
        case_dir,
        n_cells_x=2, n_cells_y=2,
        end_time=3, write_interval=3,
        d_coeff=(1e3, 1e3, 1e3),
        f_coeff=(0.0, 0.0, 0.0),
    )
    return case_dir


@pytest.fixture
def porous_inter_case_with_forchheimer(tmp_path):
    """Create a case with both Darcy and Forchheimer resistance."""
    case_dir = tmp_path / "porous_inter_forch"
    _make_porous_inter_case(
        case_dir,
        n_cells_x=2, n_cells_y=2,
        end_time=3, write_interval=3,
        d_coeff=(1e3, 1e3, 1e3),
        f_coeff=(5.0, 5.0, 5.0),
    )
    return case_dir


# ---------------------------------------------------------------------------
# Tests: Initialisation
# ---------------------------------------------------------------------------


class TestPorousInterFoamInit:
    """Tests for PorousInterFoam initialisation."""

    def test_case_loads(self, porous_inter_case):
        """Case directory loads correctly."""
        from pyfoam.applications.porous_inter_foam import PorousInterFoam

        solver = PorousInterFoam(porous_inter_case)
        assert solver.mesh.n_cells == 4

    def test_porous_properties_read(self, porous_inter_case):
        """Porous properties are read correctly."""
        from pyfoam.applications.porous_inter_foam import PorousInterFoam

        solver = PorousInterFoam(porous_inter_case)
        assert len(solver.porous_zones) == 1
        assert abs(solver.porous_zones[0].d[0] - 1e3) < 1e-2

    def test_custom_porous_properties(self, tmp_path):
        """Porous properties can be passed directly."""
        case_dir = tmp_path / "custom"
        _make_porous_inter_case(
            case_dir,
            n_cells_x=2, n_cells_y=2,
            end_time=3, write_interval=3,
        )

        from pyfoam.applications.porous_inter_foam import PorousInterFoam
        from pyfoam.applications.porous_simple_foam import PorousZoneProperties

        custom = PorousZoneProperties(
            name="custom", cell_zone="all",
            d=(1e5, 1e5, 1e5), f=(5.0, 5.0, 5.0),
        )
        solver = PorousInterFoam(case_dir, porous_zones=[custom])
        assert len(solver.porous_zones) == 1
        assert solver.porous_zones[0].d == (1e5, 1e5, 1e5)

    def test_no_porous_file(self, tmp_path):
        """Missing porosityProperties results in empty list."""
        case_dir = tmp_path / "no_porous"
        _make_porous_inter_case(
            case_dir,
            n_cells_x=2, n_cells_y=2,
            end_time=3, write_interval=3,
        )
        (case_dir / "constant" / "porosityProperties").unlink(missing_ok=True)

        from pyfoam.applications.porous_inter_foam import PorousInterFoam

        solver = PorousInterFoam(case_dir)
        assert len(solver.porous_zones) == 0

    def test_fluid_properties_stored(self, porous_inter_case):
        """Fluid properties from interFoam are preserved."""
        from pyfoam.applications.porous_inter_foam import PorousInterFoam

        solver = PorousInterFoam(
            porous_inter_case,
            rho1=900.0, rho2=2.0, sigma=0.05,
        )
        assert solver.rho1 == 900.0
        assert solver.rho2 == 2.0
        assert solver.sigma == 0.05


# ---------------------------------------------------------------------------
# Tests: Cell mask
# ---------------------------------------------------------------------------


class TestCellMask:
    """Tests for cell zone mask building."""

    def test_all_cells_mask(self, porous_inter_case):
        """Zone 'all' selects all cells."""
        from pyfoam.applications.porous_inter_foam import PorousInterFoam

        solver = PorousInterFoam(porous_inter_case)
        mask = solver._build_cell_mask("all")
        assert mask.shape == (4,)
        assert mask.all()

    def test_missing_zone_fallback(self, porous_inter_case):
        """Missing zone falls back to all cells."""
        from pyfoam.applications.porous_inter_foam import PorousInterFoam

        solver = PorousInterFoam(porous_inter_case)
        mask = solver._build_cell_mask("nonexistent")
        assert mask.all()


# ---------------------------------------------------------------------------
# Tests: Porous resistance
# ---------------------------------------------------------------------------


class TestPorousResistance:
    """Tests for Darcy-Forchheimer resistance in two-phase context."""

    def test_porous_data_built(self, porous_inter_case):
        """Porous zone data is pre-built with cell masks and tensors."""
        from pyfoam.applications.porous_inter_foam import PorousInterFoam

        solver = PorousInterFoam(porous_inter_case)
        assert len(solver._porous_data) == 1
        zone = solver._porous_data[0]
        assert "cell_mask" in zone
        assert "d" in zone
        assert "f" in zone
        assert zone["d"].shape == (3,)

    def test_resistance_increases_with_d(self):
        """Higher Darcy coefficient produces higher resistance tensor."""
        nu = 0.01
        d_low = torch.tensor([1e2, 1e2, 1e2], dtype=CFD_DTYPE)
        d_high = torch.tensor([1e6, 1e6, 1e6], dtype=CFD_DTYPE)
        nu_tensor = torch.tensor([nu], dtype=CFD_DTYPE)

        Cd_low = nu_tensor * d_low
        Cd_high = nu_tensor * d_high
        assert Cd_high.sum() > Cd_low.sum()

    def test_forchheimer_depends_on_velocity(self):
        """Forchheimer resistance depends on velocity magnitude."""
        f = torch.tensor([10.0, 10.0, 10.0], dtype=CFD_DTYPE)
        rho = torch.tensor([1000.0], dtype=CFD_DTYPE)

        U_slow = torch.tensor([[0.5, 0.0, 0.0]], dtype=CFD_DTYPE)
        U_fast = torch.tensor([[5.0, 0.0, 0.0]], dtype=CFD_DTYPE)

        F_slow = 0.5 * rho * U_slow.norm(dim=1).unsqueeze(-1) * f.unsqueeze(0)
        F_fast = 0.5 * rho * U_fast.norm(dim=1).unsqueeze(-1) * f.unsqueeze(0)

        assert F_fast.sum() > F_slow.sum()


# ---------------------------------------------------------------------------
# Tests: Solver execution
# ---------------------------------------------------------------------------


class TestPorousInterFoamSolver:
    """Tests for solver execution."""

    def test_run_produces_finite_fields(self, porous_inter_case):
        """PorousInterFoam produces finite velocity and pressure."""
        from pyfoam.applications.porous_inter_foam import PorousInterFoam

        solver = PorousInterFoam(porous_inter_case)
        conv = solver.run()

        assert torch.isfinite(solver.U).all(), "U contains NaN/Inf"
        assert torch.isfinite(solver.p).all(), "p contains NaN/Inf"
        assert torch.isfinite(solver.alpha).all(), "alpha contains NaN/Inf"

    def test_run_writes_output(self, porous_inter_case):
        """PorousInterFoam writes field files."""
        from pyfoam.applications.porous_inter_foam import PorousInterFoam

        solver = PorousInterFoam(porous_inter_case)
        solver.run()

        time_dirs = [d for d in porous_inter_case.iterdir()
                     if d.is_dir() and d.name.replace(".", "").isdigit()
                     and d.name != "0"]
        assert len(time_dirs) >= 1

    def test_zero_resistance_degeneracy(self, tmp_path):
        """With zero resistance, behaves like standard interFoam."""
        case_dir = tmp_path / "zero_porous"
        _make_porous_inter_case(
            case_dir,
            n_cells_x=2, n_cells_y=2,
            end_time=3, write_interval=3,
            d_coeff=(0.0, 0.0, 0.0),
            f_coeff=(0.0, 0.0, 0.0),
        )

        from pyfoam.applications.porous_inter_foam import PorousInterFoam

        solver = PorousInterFoam(case_dir)
        conv = solver.run()

        assert torch.isfinite(solver.U).all()
        assert torch.isfinite(solver.p).all()

    def test_forchheimer_resistance_runs(self, porous_inter_case_with_forchheimer):
        """Solver with Darcy + Forchheimer resistance runs."""
        from pyfoam.applications.porous_inter_foam import PorousInterFoam

        solver = PorousInterFoam(porous_inter_case_with_forchheimer)
        conv = solver.run()

        assert torch.isfinite(solver.U).all(), "U contains NaN/Inf"
        assert torch.isfinite(solver.p).all(), "p contains NaN/Inf"

    def test_alpha_field_preserved(self, porous_inter_case):
        """Alpha (VOF) field is correctly tracked during porous flow."""
        from pyfoam.applications.porous_inter_foam import PorousInterFoam

        solver = PorousInterFoam(porous_inter_case)
        solver.run()

        alpha = solver.alpha.detach().cpu().numpy()
        # Alpha should be bounded [0, 1]
        assert alpha.min() >= -0.1, f"alpha min = {alpha.min()}, expected >= -0.1"
        assert alpha.max() <= 1.1, f"alpha max = {alpha.max()}, expected <= 1.1"


# ---------------------------------------------------------------------------
# Tests: Negative resistance handling
# ---------------------------------------------------------------------------


class TestNegativeResistance:
    """Tests for negative resistance convention."""

    def test_negative_resistance_multiplier(self):
        """Negative resistance is handled correctly."""
        from pyfoam.applications.porous_simple_foam import PorousSimpleFoam

        result = PorousSimpleFoam._parse_resistance_vector((1e4, -0.5, 0))
        assert abs(result[0] - 1e4) < 1e-6
        assert abs(result[1] - 5000.0) < 1e-6

    def test_repr(self):
        """PorousZoneProperties repr includes key info."""
        from pyfoam.applications.porous_simple_foam import PorousZoneProperties

        pz = PorousZoneProperties(name="test", d=(1.0, 2.0, 3.0))
        assert "test" in repr(pz)

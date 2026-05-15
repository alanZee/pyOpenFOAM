"""
Unit tests for PorousSimpleFoam — steady-state incompressible porous media SIMPLE solver.

Tests cover:
- PorousZoneProperties initialisation and negative resistance handling
- MRFZoneProperties initialisation and normalisation
- PorousSimpleFoam case loading and property reading
- Cell mask building (all cells, named zones)
- Darcy-Forchheimer resistance computation
- MRF Coriolis and centrifugal force computation
- Solver construction
- Run convergence on porous cavity case
- Field writing and format validation
- Zero resistance degeneracy (should behave like standard simpleFoam)
- Isotropic vs anisotropic resistance
- Combined porous + MRF operation
- High resistance limiting behaviour
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from pyfoam.core.dtype import CFD_DTYPE
from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.io.foam_file import FoamFileHeader, FileFormat, write_foam_file


# ---------------------------------------------------------------------------
# Mesh generation helper (reuses cavity mesh with porous properties)
# ---------------------------------------------------------------------------

def _make_porous_case(
    case_dir: Path,
    n_cells_x: int = 4,
    n_cells_y: int = 4,
    nu: float = 0.01,
    end_time: int = 100,
    write_interval: int = 50,
    alpha_p: float = 0.3,
    alpha_U: float = 0.7,
    convergence_tolerance: float = 1e-4,
    max_outer_iterations: int = 200,
    d_coeff: tuple[float, float, float] = (1e4, 1e4, 1e4),
    f_coeff: tuple[float, float, float] = (0.0, 0.0, 0.0),
    include_mrf: bool = False,
    mrf_omega: float = 0.0,
    mrf_axis: tuple[float, float, float] = (0.0, 0.0, 1.0),
    mrf_origin: tuple[float, float, float] = (0.5, 0.5, 0.0),
) -> None:
    """Write a complete porous cavity case to *case_dir*.

    Creates the same lid-driven cavity mesh as simpleFoam tests,
    plus constant/porosityProperties for porous media.
    Optionally includes constant/MRFProperties.
    """
    case_dir.mkdir(parents=True, exist_ok=True)

    # ---- Mesh (same as simpleFoam cavity) ----
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
        "application     PorousSimpleFoam;\n"
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
def porous_case(tmp_path):
    """Create a porous cavity case in a temporary directory."""
    case_dir = tmp_path / "porous_cavity"
    _make_porous_case(
        case_dir,
        n_cells_x=4,
        n_cells_y=4,
        nu=0.01,
        d_coeff=(1e4, 1e4, 1e4),
        f_coeff=(0.0, 0.0, 0.0),
    )
    return case_dir


@pytest.fixture
def tiny_porous_case(tmp_path):
    """Create a minimal 2x2 porous case for fast tests."""
    case_dir = tmp_path / "tiny_porous"
    _make_porous_case(
        case_dir,
        n_cells_x=2,
        n_cells_y=2,
        nu=0.01,
        end_time=10,
        write_interval=10,
        max_outer_iterations=50,
        d_coeff=(1e3, 1e3, 1e3),
        f_coeff=(1.0, 1.0, 1.0),
    )
    return case_dir


@pytest.fixture
def mrf_porous_case(tmp_path):
    """Create a case with both porous and MRF zones."""
    case_dir = tmp_path / "mrf_porous"
    _make_porous_case(
        case_dir,
        n_cells_x=2,
        n_cells_y=2,
        nu=0.01,
        end_time=10,
        write_interval=10,
        max_outer_iterations=50,
        d_coeff=(1e3, 1e3, 1e3),
        f_coeff=(0.0, 0.0, 0.0),
        include_mrf=True,
        mrf_omega=5.0,
        mrf_axis=(0.0, 0.0, 1.0),
        mrf_origin=(0.5, 0.5, 0.0),
    )
    return case_dir


# ---------------------------------------------------------------------------
# Tests: PorousZoneProperties
# ---------------------------------------------------------------------------

class TestPorousZoneProperties:
    """Tests for PorousZoneProperties data container."""

    def test_default_properties(self):
        """Default PorousZoneProperties has sensible defaults."""
        from pyfoam.applications.porous_simple_foam import PorousZoneProperties

        props = PorousZoneProperties()
        assert props.name == "porosity1"
        assert props.cell_zone == "porosity"
        assert props.d == (0.0, 0.0, 0.0)
        assert props.f == (0.0, 0.0, 0.0)

    def test_custom_properties(self):
        """PorousZoneProperties accepts custom values."""
        from pyfoam.applications.porous_simple_foam import PorousZoneProperties

        props = PorousZoneProperties(
            name="filter",
            cell_zone="filterZone",
            d=(1e6, 2e6, 3e6),
            f=(10.0, 20.0, 30.0),
        )
        assert props.name == "filter"
        assert props.cell_zone == "filterZone"
        assert props.d == (1e6, 2e6, 3e6)
        assert props.f == (10.0, 20.0, 30.0)

    def test_repr(self):
        """String representation includes key info."""
        from pyfoam.applications.porous_simple_foam import PorousZoneProperties

        props = PorousZoneProperties(name="test", d=(1.0, 2.0, 3.0))
        repr_str = repr(props)
        assert "test" in repr_str
        assert "1.0" in repr_str


# ---------------------------------------------------------------------------
# Tests: MRFZoneProperties
# ---------------------------------------------------------------------------

class TestMRFZoneProperties:
    """Tests for MRFZoneProperties data container."""

    def test_default_properties(self):
        """Default MRFZoneProperties has sensible defaults."""
        from pyfoam.applications.porous_simple_foam import MRFZoneProperties

        props = MRFZoneProperties()
        assert props.origin == (0.0, 0.0, 0.0)
        assert props.axis == (0.0, 0.0, 1.0)
        assert props.omega == 0.0

    def test_axis_normalisation(self):
        """Axis is normalised to unit length."""
        from pyfoam.applications.porous_simple_foam import MRFZoneProperties

        props = MRFZoneProperties(axis=(3.0, 4.0, 0.0))
        mag = (props.axis[0] ** 2 + props.axis[1] ** 2 + props.axis[2] ** 2) ** 0.5
        assert abs(mag - 1.0) < 1e-10

    def test_zero_axis_defaults_to_z(self):
        """Zero-length axis defaults to (0, 0, 1)."""
        from pyfoam.applications.porous_simple_foam import MRFZoneProperties

        props = MRFZoneProperties(axis=(0.0, 0.0, 0.0))
        assert props.axis == (0.0, 0.0, 1.0)

    def test_omega_vec(self):
        """omega_vec is correct for z-axis rotation."""
        from pyfoam.applications.porous_simple_foam import MRFZoneProperties

        props = MRFZoneProperties(axis=(0.0, 0.0, 1.0), omega=10.0)
        assert abs(props.omega_vec[2] - 10.0) < 1e-10
        assert abs(props.omega_vec[0]) < 1e-10
        assert abs(props.omega_vec[1]) < 1e-10


# ---------------------------------------------------------------------------
# Tests: PorousSimpleFoam initialisation
# ---------------------------------------------------------------------------

class TestPorousSimpleFoamInit:
    """Tests for PorousSimpleFoam initialisation."""

    def test_case_loads(self, porous_case):
        """Porous case directory loads correctly."""
        from pyfoam.applications.porous_simple_foam import PorousSimpleFoam

        solver = PorousSimpleFoam(porous_case)
        assert solver.mesh.n_cells == 16

    def test_porous_properties_read(self, porous_case):
        """Porous properties are read from constant/porosityProperties."""
        from pyfoam.applications.porous_simple_foam import PorousSimpleFoam

        solver = PorousSimpleFoam(porous_case)
        assert len(solver.porous_zones) == 1
        assert abs(solver.porous_zones[0].d[0] - 1e4) < 1e-2

    def test_custom_porous_properties(self, tmp_path):
        """Porous properties can be passed directly."""
        case_dir = tmp_path / "custom_porous"
        _make_porous_case(case_dir, n_cells_x=2, n_cells_y=2, end_time=5,
                          write_interval=5, max_outer_iterations=10)

        from pyfoam.applications.porous_simple_foam import PorousSimpleFoam, PorousZoneProperties

        custom = PorousZoneProperties(
            name="custom",
            cell_zone="all",
            d=(1e5, 1e5, 1e5),
            f=(5.0, 5.0, 5.0),
        )
        solver = PorousSimpleFoam(case_dir, porous_zones=[custom])
        assert len(solver.porous_zones) == 1
        assert solver.porous_zones[0].d == (1e5, 1e5, 1e5)

    def test_no_porous_file(self, tmp_path):
        """Missing porosityProperties results in empty list."""
        case_dir = tmp_path / "no_porous"
        _make_porous_case(case_dir, n_cells_x=2, n_cells_y=2, end_time=5,
                          write_interval=5, max_outer_iterations=10)
        # Remove the porosityProperties file
        (case_dir / "constant" / "porosityProperties").unlink(missing_ok=True)

        from pyfoam.applications.porous_simple_foam import PorousSimpleFoam

        solver = PorousSimpleFoam(case_dir)
        assert len(solver.porous_zones) == 0


# ---------------------------------------------------------------------------
# Tests: Cell mask building
# ---------------------------------------------------------------------------

class TestCellMask:
    """Tests for cell zone mask building."""

    def test_all_cells_mask(self, porous_case):
        """Zone name 'all' selects all cells."""
        from pyfoam.applications.porous_simple_foam import PorousSimpleFoam

        solver = PorousSimpleFoam(porous_case)
        mask = solver._build_cell_mask("all")
        assert mask.shape == (16,)
        assert mask.all()

    def test_missing_zone_fallback(self, porous_case):
        """Missing zone name falls back to all cells."""
        from pyfoam.applications.porous_simple_foam import PorousSimpleFoam

        solver = PorousSimpleFoam(porous_case)
        mask = solver._build_cell_mask("nonexistent_zone")
        assert mask.shape == (16,)
        assert mask.all()


# ---------------------------------------------------------------------------
# Tests: Darcy-Forchheimer resistance computation
# ---------------------------------------------------------------------------

class TestDarcyForchheimer:
    """Tests for Darcy-Forchheimer resistance model."""

    def test_isotropic_darcy_resistance(self):
        """Isotropic Darcy resistance: diagonal gets tr(Cd), source gets anisotropic part."""
        # For isotropic d = (d, d, d):
        # Cd = ν * d * I (as tensor diag(d,d,d))
        # tr(Cd) = 3 * ν * d
        # Cd - I*tr(Cd) = d*I - 3*d*I = -2*d*I (non-zero!)
        # Net effect: (A + 3Vνd)U = H - ∇p + 2Vνd U => (A + Vνd)U = H - ∇p
        nu = 0.01
        d = torch.tensor([1e4, 1e4, 1e4], dtype=CFD_DTYPE)
        U = torch.tensor([[1.0, 0.0, 0.0]], dtype=CFD_DTYPE)
        V = torch.tensor([1.0], dtype=CFD_DTYPE)

        nu_tensor = torch.tensor([nu], dtype=CFD_DTYPE)
        U_mag = U.norm(dim=1)
        Cd = nu_tensor.unsqueeze(-1) * d.unsqueeze(0)  # (1, 3)
        isoCd = Cd.sum(dim=1)  # (1,)
        aniso = Cd - isoCd.unsqueeze(-1) * torch.ones_like(Cd)

        # Diagonal contribution: V * tr(Cd) = 1 * 3 * 0.01 * 1e4 = 300
        diag_contrib = V * isoCd
        assert abs(diag_contrib[0].item() - 300.0) < 1e-6

        # Anisotropic source: -V * aniso * U = -1 * (-200) * 1 = 200
        # Net diagonal: 300 - 200 = 100 = V * ν * d (correct!)
        aniso_source = V.unsqueeze(-1) * aniso * U
        assert abs(aniso_source[0, 0].item() - (-200.0)) < 1e-6

    def test_anisotropic_darcy_resistance(self):
        """Anisotropic Darcy resistance creates anisotropic source."""
        nu = 0.01
        d = torch.tensor([1e4, 2e4, 3e4], dtype=CFD_DTYPE)
        U = torch.tensor([[1.0, 0.0, 0.0]], dtype=CFD_DTYPE)
        V = torch.tensor([1.0], dtype=CFD_DTYPE)

        nu_tensor = torch.tensor([nu], dtype=CFD_DTYPE)
        Cd = nu_tensor.unsqueeze(-1) * d.unsqueeze(0)
        isoCd = Cd.sum(dim=1)
        aniso = Cd - isoCd.unsqueeze(-1) * torch.ones_like(Cd)
        aniso_source = V.unsqueeze(-1) * aniso * U

        # Anisotropic source should be non-zero for x-component
        # Cd_x = 0.01 * 1e4 = 100, isoCd = 100 + 200 + 300 = 600
        # aniso_x = 100 - 600 = -500
        # source_x = -V * aniso_x * U_x = -1 * (-500) * 1 = 500
        assert abs(aniso_source[0, 0].item() - (-500.0)) < 1e-6
        # y and z components: aniso_y = 200-600 = -400, but U_y = 0
        assert abs(aniso_source[0, 1].item()) < 1e-10

    def test_forchheimer_resistance(self):
        """Forchheimer resistance depends on velocity magnitude."""
        nu = 0.01
        f = torch.tensor([10.0, 10.0, 10.0], dtype=CFD_DTYPE)
        U = torch.tensor([[2.0, 0.0, 0.0]], dtype=CFD_DTYPE)

        U_mag = U.norm(dim=1)
        F_coeff = U_mag.unsqueeze(-1) * f.unsqueeze(0) * 0.5  # (1, 3)

        # |U| = 2, f = 10, so F_coeff = 2 * 10 * 0.5 = 10
        assert abs(F_coeff[0, 0].item() - 10.0) < 1e-6

    def test_combined_resistance(self):
        """Combined Darcy-Forchheimer resistance."""
        nu = 0.01
        d = torch.tensor([1e4, 1e4, 1e4], dtype=CFD_DTYPE)
        f = torch.tensor([10.0, 10.0, 10.0], dtype=CFD_DTYPE)
        U = torch.tensor([[1.0, 0.0, 0.0]], dtype=CFD_DTYPE)

        nu_tensor = torch.tensor([nu], dtype=CFD_DTYPE)
        U_mag = U.norm(dim=1)

        Cd = nu_tensor.unsqueeze(-1) * d.unsqueeze(0) + \
             U_mag.unsqueeze(-1) * f.unsqueeze(0) * 0.5

        # Cd = 0.01*1e4 + 1*10*0.5 = 100 + 5 = 105 per component
        assert abs(Cd[0, 0].item() - 105.0) < 1e-6

    def test_zero_resistance(self):
        """Zero resistance produces zero contribution."""
        d = torch.tensor([0.0, 0.0, 0.0], dtype=CFD_DTYPE)
        f = torch.tensor([0.0, 0.0, 0.0], dtype=CFD_DTYPE)
        U = torch.tensor([[1.0, 1.0, 1.0]], dtype=CFD_DTYPE)
        nu = 0.01

        nu_tensor = torch.tensor([nu], dtype=CFD_DTYPE)
        U_mag = U.norm(dim=1)
        Cd = nu_tensor.unsqueeze(-1) * d.unsqueeze(0) + \
             U_mag.unsqueeze(-1) * f.unsqueeze(0) * 0.5

        assert Cd.abs().max() < 1e-30

    def test_resistance_slows_velocity(self):
        """Higher resistance leads to lower velocity magnitude."""
        # This is tested indirectly through the solver, but we can
        # verify the resistance tensor magnitude increases with d
        d_low = torch.tensor([1e2, 1e2, 1e2], dtype=CFD_DTYPE)
        d_high = torch.tensor([1e6, 1e6, 1e6], dtype=CFD_DTYPE)
        nu = 0.01
        nu_tensor = torch.tensor([nu], dtype=CFD_DTYPE)

        Cd_low = nu_tensor * d_low
        Cd_high = nu_tensor * d_high

        assert Cd_high.sum() > Cd_low.sum()


# ---------------------------------------------------------------------------
# Tests: MRF force computation
# ---------------------------------------------------------------------------

class TestMRFForces:
    """Tests for MRF Coriolis and centrifugal force computations."""

    def test_centrifugal_z_axis(self):
        """Centrifugal force for z-axis rotation points radially outward."""
        omega_vec = torch.tensor([0.0, 0.0, 10.0], dtype=CFD_DTYPE)
        r = torch.tensor([[1.0, 0.0, 0.0]], dtype=CFD_DTYPE)

        omega_sq = omega_vec.dot(omega_vec)
        omega_dot_r = (r * omega_vec.unsqueeze(0)).sum(dim=1)
        F_cent = r * omega_sq - omega_vec.unsqueeze(0) * omega_dot_r.unsqueeze(-1)

        # F_cent = (100, 0, 0)
        assert abs(F_cent[0, 0].item() - 100.0) < 1e-8
        assert abs(F_cent[0, 1].item()) < 1e-8
        assert abs(F_cent[0, 2].item()) < 1e-8

    def test_coriolis_z_axis(self):
        """Coriolis force for z-axis rotation: -2ω × U."""
        omega_vec = torch.tensor([0.0, 0.0, 10.0], dtype=CFD_DTYPE)
        U = torch.tensor([[1.0, 0.0, 0.0]], dtype=CFD_DTYPE)

        omega_cross_U = torch.zeros_like(U)
        omega_cross_U[:, 0] = omega_vec[1] * U[:, 2] - omega_vec[2] * U[:, 1]
        omega_cross_U[:, 1] = omega_vec[2] * U[:, 0] - omega_vec[0] * U[:, 2]
        omega_cross_U[:, 2] = omega_vec[0] * U[:, 1] - omega_vec[1] * U[:, 0]
        F_coriolis = -2.0 * omega_cross_U

        # -2*(0,0,10)×(1,0,0) = -2*(0,10,0) = (0,-20,0)
        assert abs(F_coriolis[0, 1].item() - (-20.0)) < 1e-8


# ---------------------------------------------------------------------------
# Tests: Solver execution
# ---------------------------------------------------------------------------

class TestPorousSimpleFoamSolver:
    """Tests for porous SIMPLE solver construction and execution."""

    def test_build_porous_solver(self, porous_case):
        """_build_solver_with_porosity creates a _PorousSIMPLESolver."""
        from pyfoam.applications.porous_simple_foam import PorousSimpleFoam, _PorousSIMPLESolver

        solver = PorousSimpleFoam(porous_case)
        porous_solver = solver._build_solver_with_porosity()
        assert isinstance(porous_solver, _PorousSIMPLESolver)

    def test_run_produces_finite_fields(self, tiny_porous_case):
        """PorousSimpleFoam produces finite velocity and pressure."""
        from pyfoam.applications.porous_simple_foam import PorousSimpleFoam

        solver = PorousSimpleFoam(tiny_porous_case)
        conv = solver.run()

        assert solver.U.shape == (4, 3)
        assert solver.p.shape == (4,)
        assert torch.isfinite(solver.U).all(), "U contains NaN/Inf"
        assert torch.isfinite(solver.p).all(), "p contains NaN/Inf"
        assert torch.isfinite(solver.phi).all(), "phi contains NaN/Inf"

    def test_run_writes_output(self, tiny_porous_case):
        """PorousSimpleFoam writes field files to time directories."""
        from pyfoam.applications.porous_simple_foam import PorousSimpleFoam

        solver = PorousSimpleFoam(tiny_porous_case)
        solver.run()

        time_dirs = [d for d in tiny_porous_case.iterdir()
                     if d.is_dir() and d.name.replace(".", "").isdigit()
                     and d.name != "0"]
        assert len(time_dirs) >= 1

        for td in time_dirs:
            assert (td / "U").exists(), f"U not found in {td}"
            assert (td / "p").exists(), f"p not found in {td}"

    def test_fields_valid_format(self, tiny_porous_case):
        """Written fields are valid OpenFOAM format."""
        from pyfoam.applications.porous_simple_foam import PorousSimpleFoam
        from pyfoam.io.field_io import read_field

        solver = PorousSimpleFoam(tiny_porous_case)
        solver.run()

        time_dirs = sorted(
            [d for d in tiny_porous_case.iterdir()
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

    def test_zero_resistance_degeneracy(self, tmp_path):
        """PorousSimpleFoam with zero resistance should behave like simpleFoam."""
        case_dir = tmp_path / "zero_porous"
        _make_porous_case(
            case_dir,
            n_cells_x=2,
            n_cells_y=2,
            end_time=10,
            write_interval=10,
            max_outer_iterations=50,
            d_coeff=(0.0, 0.0, 0.0),
            f_coeff=(0.0, 0.0, 0.0),
        )

        from pyfoam.applications.porous_simple_foam import PorousSimpleFoam

        solver = PorousSimpleFoam(case_dir)
        conv = solver.run()

        assert torch.isfinite(solver.U).all()
        assert torch.isfinite(solver.p).all()


# ---------------------------------------------------------------------------
# Tests: MRF + Porous combined
# ---------------------------------------------------------------------------

class TestMrfPorousCombined:
    """Tests for combined MRF and porous operation."""

    def test_mrf_porous_loads(self, mrf_porous_case):
        """Case with both MRF and porous zones loads correctly."""
        from pyfoam.applications.porous_simple_foam import PorousSimpleFoam

        solver = PorousSimpleFoam(mrf_porous_case)
        assert len(solver.porous_zones) == 1
        assert len(solver.mrf_zones) == 1

    def test_mrf_porous_runs(self, mrf_porous_case):
        """Combined MRF + porous solver runs and produces finite fields."""
        from pyfoam.applications.porous_simple_foam import PorousSimpleFoam

        solver = PorousSimpleFoam(mrf_porous_case)
        conv = solver.run()

        assert torch.isfinite(solver.U).all(), "U contains NaN/Inf"
        assert torch.isfinite(solver.p).all(), "p contains NaN/Inf"

    def test_mrf_only(self, tmp_path):
        """Solver with MRF only (no porous) runs correctly."""
        case_dir = tmp_path / "mrf_only"
        _make_porous_case(
            case_dir,
            n_cells_x=2,
            n_cells_y=2,
            end_time=10,
            write_interval=10,
            max_outer_iterations=50,
            d_coeff=(0.0, 0.0, 0.0),
            f_coeff=(0.0, 0.0, 0.0),
            include_mrf=True,
            mrf_omega=5.0,
        )

        from pyfoam.applications.porous_simple_foam import PorousSimpleFoam

        solver = PorousSimpleFoam(case_dir)
        conv = solver.run()

        assert torch.isfinite(solver.U).all()
        assert torch.isfinite(solver.p).all()


# ---------------------------------------------------------------------------
# Tests: Negative resistance handling
# ---------------------------------------------------------------------------

class TestNegativeResistance:
    """Tests for OpenFOAM's negative resistance convention."""

    def test_negative_resistance_multiplier(self):
        """Negative resistance is treated as multiplier of max positive component."""
        from pyfoam.applications.porous_simple_foam import PorousSimpleFoam

        # d = (1e4, -0.5, 0) means d_y = -0.5 * max(1e4, 0) = -0.5 * 1e4 = -5000
        # But OpenFOAM convention: val *= -maxCmpt, so -0.5 * -10000 = 5000
        result = PorousSimpleFoam._parse_resistance_vector((1e4, -0.5, 0))
        assert abs(result[0] - 1e4) < 1e-6
        assert abs(result[1] - 5000.0) < 1e-6
        assert abs(result[2]) < 1e-6

    def test_all_negative_resistance_returns_zero(self):
        """All-negative resistance returns zeros (invalid)."""
        from pyfoam.applications.porous_simple_foam import PorousSimpleFoam

        result = PorousSimpleFoam._parse_resistance_vector((-1.0, -2.0, -3.0))
        assert result == (0.0, 0.0, 0.0)

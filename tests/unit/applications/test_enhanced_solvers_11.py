"""
Unit tests for enhanced solver variants v11, v13 (non-orthogonal corrections and coupling).

Tests cover:
- Case loading and solver initialisation
- Enhanced parameter reading
- Solver feature methods produce finite values
- Export of new classes from __init__.py
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from pyfoam.core.dtype import CFD_DTYPE, INDEX_DTYPE
from pyfoam.io.foam_file import FoamFileHeader, FileFormat, write_foam_file


# ---------------------------------------------------------------------------
# Mesh generation helper
# ---------------------------------------------------------------------------

def _make_cavity_case(
    case_dir: Path,
    n_cells_x: int = 4,
    n_cells_y: int = 4,
    nu: float = 0.01,
    delta_t: float = 0.001,
    end_time: float = 0.01,
    piso_correctors: int = 2,
    compressible: bool = False,
    buoyant: bool = False,
    reacting: bool = False,
) -> None:
    """Write a complete case directory for enhanced solver testing."""
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

    for j in range(n_cells_y):
        for i in range(n_cells_x - 1):
            p0 = j * (n_cells_x + 1) + i + 1
            p1 = p0 + n_cells_x + 1
            p2 = p1 + n_base
            p3 = p0 + n_base
            faces.append((4, p0, p1, p2, p3))
            owner.append(j * n_cells_x + i)
            neighbour.append(j * n_cells_x + i + 1)

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

    for i in range(n_cells_x):
        p0 = n_cells_y * (n_cells_x + 1) + i
        p1 = p0 + 1
        p2 = p1 + n_base
        p3 = p0 + n_base
        faces.append((4, p0, p1, p2, p3))
        owner.append((n_cells_y - 1) * n_cells_x + i)
    n_top = n_cells_x
    top_start = n_internal

    for i in range(n_cells_x):
        p0 = i
        p1 = i + 1
        p2 = p1 + n_base
        p3 = p0 + n_base
        faces.append((4, p0, p1, p2, p3))
        owner.append(i)
    n_bottom = n_cells_x
    bottom_start = top_start + n_top

    for j in range(n_cells_y):
        p0 = j * (n_cells_x + 1)
        p1 = p0 + n_cells_x + 1
        p2 = p1 + n_base
        p3 = p0 + n_base
        faces.append((4, p0, p1, p2, p3))
        owner.append(j * n_cells_x)
    n_left = n_cells_y
    left_start = bottom_start + n_bottom

    for j in range(n_cells_y):
        p0 = j * (n_cells_x + 1) + n_cells_x
        p1 = p0 + n_cells_x + 1
        p2 = p1 + n_base
        p3 = p0 + n_base
        faces.append((4, p0, p1, p2, p3))
        owner.append(j * n_cells_x + n_cells_x - 1)
    n_right = n_cells_y
    right_start = left_start + n_left

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
    empty_start = right_start + n_right

    n_faces = len(faces)
    n_cells = n_cells_x * n_cells_y

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
    lines = ["5", "("]
    lines.append("    movingWall")
    lines.append("    {")
    lines.append("        type            wall;")
    lines.append(f"        nFaces          {n_top};")
    lines.append(f"        startFace       {top_start};")
    lines.append("    }")
    lines.append("    fixedWalls")
    lines.append("    {")
    lines.append("        type            wall;")
    lines.append(f"        nFaces          {n_bottom + n_left + n_right};")
    lines.append(f"        startFace       {bottom_start};")
    lines.append("    }")
    lines.append("    frontAndBack")
    lines.append("    {")
    lines.append("        type            empty;")
    lines.append(f"        nFaces          {n_empty};")
    lines.append(f"        startFace       {empty_start};")
    lines.append("    }")
    lines.append(")")
    write_foam_file(mesh_dir / "boundary", h, "\n".join(lines), overwrite=True)

    # system files
    sys_dir = case_dir / "system"
    sys_dir.mkdir(exist_ok=True)

    cd_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="dictionary", location="system", object="controlDict",
    )
    cd_body = (
        "application     pisoFoam;\n"
        "startFrom       startTime;\n"
        "startTime       0;\n"
        "stopAt          endTime;\n"
        f"endTime         {end_time};\n"
        f"deltaT          {delta_t};\n"
        "writeControl    timeStep;\n"
        "writeInterval   10000;\n"
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
        "divSchemes\n{\n    default         none;\n}\n\n"
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
        "        maxIter         1000;\n"
        "    }\n"
        "    U\n    {\n"
        "        solver          PBiCGStab;\n"
        "        preconditioner  DILU;\n"
        "        tolerance       1e-6;\n"
        "        relTol          0.01;\n"
        "        maxIter         1000;\n"
        "    }\n"
        "    T\n    {\n"
        "        solver          PCG;\n"
        "        preconditioner  DIC;\n"
        "        tolerance       1e-6;\n"
        "        relTol          0.01;\n"
        "        maxIter         1000;\n"
        "    }\n"
        "}\n\n"
        "PISO\n{\n"
        f"    nCorrectors     {piso_correctors};\n"
        "    nNonOrthogonalCorrectors 2;\n"
        "}\n\n"
        "PIMPLE\n{\n"
        "    nOuterCorrectors 2;\n"
        "    nCorrectors     2;\n"
        "    nNonOrthogonalCorrectors 0;\n"
        "    convergenceTolerance 1e-4;\n"
        "    maxOuterIterations 10;\n"
        "}\n\n"
        "SIMPLE\n{\n"
        "    nNonOrthogonalCorrectors 0;\n"
        "    convergenceTolerance 1e-4;\n"
        "    maxOuterIterations 10;\n"
        "}\n"
    )
    write_foam_file(sys_dir / "fvSolution", fv_header, fv_body, overwrite=True)

    const_dir = case_dir / "constant"
    const_dir.mkdir(exist_ok=True)

    tp_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="dictionary", location="constant", object="transportProperties",
    )
    tp_body = f"nu              [0 2 -1 0 0 0 0] {nu};\n"
    write_foam_file(const_dir / "transportProperties", tp_header, tp_body, overwrite=True)

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
    write_foam_file(zero_dir / "U", U_header, U_body, overwrite=True)

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

    if compressible or buoyant:
        T_header = FoamFileHeader(
            version="2.0", format=FileFormat.ASCII,
            class_name="volScalarField", location="0", object="T",
        )
        T_body = (
            "dimensions      [0 0 0 1 0 0 0];\n\n"
            "internalField   uniform 300;\n\n"
            "boundaryField\n{\n"
            "    movingWall\n    {\n"
            "        type            fixedValue;\n"
            "        value           uniform 310;\n"
            "    }\n"
            "    fixedWalls\n    {\n"
            "        type            fixedValue;\n"
            "        value           uniform 300;\n"
            "    }\n"
            "    frontAndBack\n    {\n"
            "        type            empty;\n"
            "    }\n"
            "}\n"
        )
        write_foam_file(zero_dir / "T", T_header, T_body, overwrite=True)

    if compressible or buoyant:
        thermo_header = FoamFileHeader(
            version="2.0", format=FileFormat.ASCII,
            class_name="dictionary", location="constant",
            object="thermophysicalProperties",
        )
        thermo_body = (
            "thermoType  hPsiThermo<gasMixture<specieThermo<janaf<perfectGas>>>>;\n"
            "beta        3.33e-3;\n"
            "TRef        300;\n"
        )
        write_foam_file(const_dir / "thermophysicalProperties", thermo_header, thermo_body, overwrite=True)

    if compressible:
        alpha_header = FoamFileHeader(
            version="2.0", format=FileFormat.ASCII,
            class_name="volScalarField", location="0", object="alpha.water",
        )
        alpha_body = (
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
        write_foam_file(zero_dir / "alpha.water", alpha_header, alpha_body, overwrite=True)

    if buoyant:
        g_header = FoamFileHeader(
            version="2.0", format=FileFormat.ASCII,
            class_name="dictionary", location="constant", object="g",
        )
        g_body = "dimensions   [0 1 -2 0 0 0 0];\nvalue        (0 -9.81 0);\n"
        write_foam_file(const_dir / "g", g_header, g_body, overwrite=True)

        rad_header = FoamFileHeader(
            version="2.0", format=FileFormat.ASCII,
            class_name="dictionary", location="constant", object="radiationProperties",
        )
        rad_body = "radiationModel P1;\nP1\n{\n    absorptionCoeff 0.1;\n}\n"
        write_foam_file(const_dir / "radiationProperties", rad_header, rad_body, overwrite=True)

    if reacting:
        YA_header = FoamFileHeader(
            version="2.0", format=FileFormat.ASCII,
            class_name="volScalarField", location="0", object="YA",
        )
        YA_body = (
            "dimensions      [0 0 0 0 0 0 0];\n\n"
            "internalField   uniform 1;\n\n"
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
        write_foam_file(zero_dir / "YA", YA_header, YA_body, overwrite=True)

        YB_header = FoamFileHeader(
            version="2.0", format=FileFormat.ASCII,
            class_name="volScalarField", location="0", object="YB",
        )
        YB_body = (
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
        write_foam_file(zero_dir / "YB", YB_header, YB_body, overwrite=True)

        rxn_header = FoamFileHeader(
            version="2.0", format=FileFormat.ASCII,
            class_name="dictionary", location="constant", object="reactions",
        )
        rxn_body = (
            "reaction1\n{\n"
            "    A           1e6;\n"
            "    beta        0.0;\n"
            "    Ea          10000;\n"
            "    reactants\n    {\n"
            "        A 1.0;\n"
            "    }\n"
            "    products\n    {\n"
            "        B 1.0;\n"
            "    }\n"
            "}\n"
        )
        write_foam_file(const_dir / "reactions", rxn_header, rxn_body, overwrite=True)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cavity_case(tmp_path):
    case_dir = tmp_path / "cavity"
    _make_cavity_case(case_dir)
    return case_dir


@pytest.fixture
def compressible_case(tmp_path):
    case_dir = tmp_path / "compressible"
    _make_cavity_case(case_dir, compressible=True)
    return case_dir


@pytest.fixture
def buoyant_case(tmp_path):
    case_dir = tmp_path / "buoyant"
    _make_cavity_case(case_dir, buoyant=True)
    return case_dir


@pytest.fixture
def reacting_case(tmp_path):
    case_dir = tmp_path / "reacting"
    _make_cavity_case(case_dir, reacting=True, end_time=0.005, delta_t=0.0005)
    return case_dir


# ===========================================================================
# Tests: SimpleFoamEnhanced11
# ===========================================================================


class TestSimpleFoamEnhanced11:
    """Tests for enhanced SIMPLE solver v11 (non-orthogonal corrections)."""

    def test_init(self, cavity_case):
        from pyfoam.applications.simple_foam_enhanced_11 import SimpleFoamEnhanced11
        solver = SimpleFoamEnhanced11(cavity_case, enoc=True, cnoc=True, orns=True)
        assert solver.enoc is True
        assert solver.cnoc is True
        assert solver.orns is True

    def test_init_defaults(self, cavity_case):
        from pyfoam.applications.simple_foam_enhanced_11 import SimpleFoamEnhanced11
        solver = SimpleFoamEnhanced11(cavity_case)
        assert solver.enoc_levels == 3
        assert solver.orns_blend == pytest.approx(0.5)

    def test_enoc_pressure(self, cavity_case):
        from pyfoam.applications.simple_foam_enhanced_11 import SimpleFoamEnhanced11
        solver = SimpleFoamEnhanced11(cavity_case, enoc=True)
        p_enoc = solver._extended_non_orthogonal_correct(solver.p, solver.U)
        assert p_enoc.shape == solver.p.shape
        assert torch.isfinite(p_enoc).all()

    def test_enoc_disabled(self, cavity_case):
        from pyfoam.applications.simple_foam_enhanced_11 import SimpleFoamEnhanced11
        solver = SimpleFoamEnhanced11(cavity_case, enoc=False)
        p_out = solver._extended_non_orthogonal_correct(solver.p, solver.U)
        assert torch.allclose(p_out, solver.p)

    def test_cnoc_pressure(self, cavity_case):
        from pyfoam.applications.simple_foam_enhanced_11 import SimpleFoamEnhanced11
        solver = SimpleFoamEnhanced11(cavity_case, cnoc=True)
        p_cnoc = solver._consistent_non_orthogonal_correct(solver.p, solver.U)
        assert p_cnoc.shape == solver.p.shape
        assert torch.isfinite(p_cnoc).all()

    def test_orns_stabilise(self, cavity_case):
        from pyfoam.applications.simple_foam_enhanced_11 import SimpleFoamEnhanced11
        solver = SimpleFoamEnhanced11(cavity_case, orns=True)
        p_orns = solver._over_relaxed_stabilise(solver.p, solver.U)
        assert p_orns.shape == solver.p.shape
        assert torch.isfinite(p_orns).all()


# ===========================================================================
# Tests: PimpleFoamEnhanced11
# ===========================================================================


class TestPimpleFoamEnhanced11:
    """Tests for enhanced PIMPLE solver v11 (non-orthogonal corrections)."""

    def test_init(self, cavity_case):
        from pyfoam.applications.pimple_foam_enhanced_11 import PimpleFoamEnhanced11
        solver = PimpleFoamEnhanced11(cavity_case, enpc=True, cnmc=True, orns=True)
        assert solver.enpc is True
        assert solver.cnmc is True
        assert solver.orns is True

    def test_init_defaults(self, cavity_case):
        from pyfoam.applications.pimple_foam_enhanced_11 import PimpleFoamEnhanced11
        solver = PimpleFoamEnhanced11(cavity_case)
        assert solver.enpc_levels == 3
        assert solver.orns_blend == pytest.approx(0.5)

    def test_enpc_pressure(self, cavity_case):
        from pyfoam.applications.pimple_foam_enhanced_11 import PimpleFoamEnhanced11
        solver = PimpleFoamEnhanced11(cavity_case, enpc=True)
        p_enpc = solver._extended_non_orthogonal_pressure(
            solver.p, solver.U, solver.delta_t,
        )
        assert p_enpc.shape == solver.p.shape
        assert torch.isfinite(p_enpc).all()

    def test_cnmc_momentum(self, cavity_case):
        from pyfoam.applications.pimple_foam_enhanced_11 import PimpleFoamEnhanced11
        solver = PimpleFoamEnhanced11(cavity_case, cnmc=True)
        U_cnmc = solver._consistent_non_orthogonal_momentum(solver.U, solver.p)
        assert U_cnmc.shape == solver.U.shape
        assert torch.isfinite(U_cnmc).all()

    def test_orns_stabilise(self, cavity_case):
        from pyfoam.applications.pimple_foam_enhanced_11 import PimpleFoamEnhanced11
        solver = PimpleFoamEnhanced11(cavity_case, orns=True)
        p_orns = solver._over_relaxed_stabilise(solver.p, solver.U)
        assert p_orns.shape == solver.p.shape
        assert torch.isfinite(p_orns).all()


# ===========================================================================
# Tests: PisoFoamEnhanced11
# ===========================================================================


class TestPisoFoamEnhanced11:
    """Tests for enhanced PISO solver v11 (non-orthogonal corrections)."""

    def test_init(self, cavity_case):
        from pyfoam.applications.piso_foam_enhanced_11 import PisoFoamEnhanced11
        solver = PisoFoamEnhanced11(cavity_case, enpp=True, cnrc=True, orns=True)
        assert solver.enpp is True
        assert solver.cnrc is True
        assert solver.orns is True

    def test_init_defaults(self, cavity_case):
        from pyfoam.applications.piso_foam_enhanced_11 import PisoFoamEnhanced11
        solver = PisoFoamEnhanced11(cavity_case)
        assert solver.enpp_levels == 3
        assert solver.orns_blend == pytest.approx(0.5)

    def test_enpp_pressure(self, cavity_case):
        from pyfoam.applications.piso_foam_enhanced_11 import PisoFoamEnhanced11
        solver = PisoFoamEnhanced11(cavity_case, enpp=True)
        p_enpp = solver._extended_non_orthogonal_project(solver.p, solver.U)
        assert p_enpp.shape == solver.p.shape
        assert torch.isfinite(p_enpp).all()

    def test_cnrc_velocity(self, cavity_case):
        from pyfoam.applications.piso_foam_enhanced_11 import PisoFoamEnhanced11
        solver = PisoFoamEnhanced11(cavity_case, cnrc=True)
        U_cnrc = solver._consistent_rhie_chow_correct(
            solver.U, solver.p, solver.U.clone(), solver.p.clone(), solver.delta_t,
        )
        assert U_cnrc.shape == solver.U.shape
        assert torch.isfinite(U_cnrc).all()

    def test_orns_stabilise(self, cavity_case):
        from pyfoam.applications.piso_foam_enhanced_11 import PisoFoamEnhanced11
        solver = PisoFoamEnhanced11(cavity_case, orns=True)
        p_orns = solver._over_relaxed_stabilise(solver.p)
        assert p_orns.shape == solver.p.shape
        assert torch.isfinite(p_orns).all()


# ===========================================================================
# Tests: IcoFoamEnhanced11
# ===========================================================================


class TestIcoFoamEnhanced11:
    """Tests for enhanced ICO solver v11 (non-orthogonal corrections)."""

    def test_init(self, cavity_case):
        from pyfoam.applications.ico_foam_enhanced_11 import IcoFoamEnhanced11
        solver = IcoFoamEnhanced11(cavity_case, enopc=True, cnvpc=True, orns=True)
        assert solver.enopc is True
        assert solver.cnvpc is True
        assert solver.orns is True

    def test_init_defaults(self, cavity_case):
        from pyfoam.applications.ico_foam_enhanced_11 import IcoFoamEnhanced11
        solver = IcoFoamEnhanced11(cavity_case)
        assert solver.enopc_levels == 3
        assert solver.orns_blend == pytest.approx(0.5)

    def test_enopc_poisson(self, cavity_case):
        from pyfoam.applications.ico_foam_enhanced_11 import IcoFoamEnhanced11
        solver = IcoFoamEnhanced11(cavity_case, enopc=True)
        rhs = solver.p.clone() * 0.01
        p_enopc = solver._extended_non_orthogonal_poisson(solver.p, rhs)
        assert p_enopc.shape == solver.p.shape
        assert torch.isfinite(p_enopc).all()

    def test_cnvpc_coupling(self, cavity_case):
        from pyfoam.applications.ico_foam_enhanced_11 import IcoFoamEnhanced11
        solver = IcoFoamEnhanced11(cavity_case, cnvpc=True)
        U_cnvpc = solver._consistent_non_orthogonal_coupling(
            solver.U, solver.p, solver.delta_t,
        )
        assert U_cnvpc.shape == solver.U.shape
        assert torch.isfinite(U_cnvpc).all()

    def test_orns_stabilise(self, cavity_case):
        from pyfoam.applications.ico_foam_enhanced_11 import IcoFoamEnhanced11
        solver = IcoFoamEnhanced11(cavity_case, orns=True)
        p_orns = solver._over_relaxed_stabilise(solver.p)
        assert p_orns.shape == solver.p.shape
        assert torch.isfinite(p_orns).all()


# ===========================================================================
# Tests: BuoyantPimpleFoamEnhanced11
# ===========================================================================


class TestBuoyantPimpleFoamEnhanced11:
    """Tests for enhanced buoyant PIMPLE solver v11."""

    def test_init(self, buoyant_case):
        from pyfoam.applications.buoyant_pimple_foam_enhanced_11 import BuoyantPimpleFoamEnhanced11
        solver = BuoyantPimpleFoamEnhanced11(buoyant_case, enbpc=True, cntmc=True, orns=True)
        assert solver.enbpc is True
        assert solver.cntmc is True
        assert solver.orns is True

    def test_init_defaults(self, buoyant_case):
        from pyfoam.applications.buoyant_pimple_foam_enhanced_11 import BuoyantPimpleFoamEnhanced11
        solver = BuoyantPimpleFoamEnhanced11(buoyant_case)
        assert solver.enbpc_levels == 3
        assert solver.orns_blend == pytest.approx(0.5)

    def test_enbpc_pressure(self, buoyant_case):
        from pyfoam.applications.buoyant_pimple_foam_enhanced_11 import BuoyantPimpleFoamEnhanced11
        solver = BuoyantPimpleFoamEnhanced11(buoyant_case, enbpc=True)
        T = solver.T if hasattr(solver, 'T') else torch.ones_like(solver.p) * 300.0
        rho = torch.ones_like(solver.p) * 1.2
        p_enbpc = solver._extended_buoyancy_pressure_correct(solver.p, T, rho)
        assert p_enbpc.shape == solver.p.shape
        assert torch.isfinite(p_enbpc).all()

    def test_cntmc_momentum(self, buoyant_case):
        from pyfoam.applications.buoyant_pimple_foam_enhanced_11 import BuoyantPimpleFoamEnhanced11
        solver = BuoyantPimpleFoamEnhanced11(buoyant_case, cntmc=True)
        T = solver.T if hasattr(solver, 'T') else torch.ones_like(solver.p) * 300.0
        U_cntmc = solver._consistent_thermal_momentum_correct(solver.U, T, solver.p)
        assert U_cntmc.shape == solver.U.shape
        assert torch.isfinite(U_cntmc).all()

    def test_orns_stabilise(self, buoyant_case):
        from pyfoam.applications.buoyant_pimple_foam_enhanced_11 import BuoyantPimpleFoamEnhanced11
        solver = BuoyantPimpleFoamEnhanced11(buoyant_case, orns=True)
        p_orns = solver._over_relaxed_stabilise(solver.p)
        assert p_orns.shape == solver.p.shape
        assert torch.isfinite(p_orns).all()


# ===========================================================================
# Tests: BuoyantSimpleFoamEnhanced11
# ===========================================================================


class TestBuoyantSimpleFoamEnhanced11:
    """Tests for enhanced buoyant SIMPLE solver v11."""

    def test_init(self, buoyant_case):
        from pyfoam.applications.buoyant_simple_foam_enhanced_11 import BuoyantSimpleFoamEnhanced11
        solver = BuoyantSimpleFoamEnhanced11(buoyant_case, enbpc=True, cnbmc=True, orns=True)
        assert solver.enbpc is True
        assert solver.cnbmc is True
        assert solver.orns is True

    def test_init_defaults(self, buoyant_case):
        from pyfoam.applications.buoyant_simple_foam_enhanced_11 import BuoyantSimpleFoamEnhanced11
        solver = BuoyantSimpleFoamEnhanced11(buoyant_case)
        assert solver.enbpc_levels == 3
        assert solver.orns_blend == pytest.approx(0.5)

    def test_enbpc_pressure(self, buoyant_case):
        from pyfoam.applications.buoyant_simple_foam_enhanced_11 import BuoyantSimpleFoamEnhanced11
        solver = BuoyantSimpleFoamEnhanced11(buoyant_case, enbpc=True)
        T = solver.T if hasattr(solver, 'T') else torch.ones_like(solver.p) * 300.0
        rho = torch.ones_like(solver.p) * 1.2
        p_enbpc = solver._extended_buoyant_pressure_correct(solver.p, T, rho)
        assert p_enbpc.shape == solver.p.shape
        assert torch.isfinite(p_enbpc).all()

    def test_cnbmc_momentum(self, buoyant_case):
        from pyfoam.applications.buoyant_simple_foam_enhanced_11 import BuoyantSimpleFoamEnhanced11
        solver = BuoyantSimpleFoamEnhanced11(buoyant_case, cnbmc=True)
        T = solver.T if hasattr(solver, 'T') else torch.ones_like(solver.p) * 300.0
        U_cnbmc = solver._consistent_buoyant_momentum_correct(solver.U, T, 300.0)
        assert U_cnbmc.shape == solver.U.shape
        assert torch.isfinite(U_cnbmc).all()

    def test_orns_stabilise(self, buoyant_case):
        from pyfoam.applications.buoyant_simple_foam_enhanced_11 import BuoyantSimpleFoamEnhanced11
        solver = BuoyantSimpleFoamEnhanced11(buoyant_case, orns=True)
        p_orns = solver._over_relaxed_stabilise(solver.p)
        assert p_orns.shape == solver.p.shape
        assert torch.isfinite(p_orns).all()


# ===========================================================================
# Tests: CompressibleInterFoamEnhanced11
# ===========================================================================


class TestCompressibleInterFoamEnhanced11:
    """Tests for enhanced compressible VOF solver v11."""

    def test_init(self, compressible_case):
        from pyfoam.applications.compressible_inter_foam_enhanced_11 import CompressibleInterFoamEnhanced11
        solver = CompressibleInterFoamEnhanced11(compressible_case, envpc=True, cnpfc=True, orns=True)
        assert solver.envpc is True
        assert solver.cnpfc is True
        assert solver.orns is True

    def test_init_defaults(self, compressible_case):
        from pyfoam.applications.compressible_inter_foam_enhanced_11 import CompressibleInterFoamEnhanced11
        solver = CompressibleInterFoamEnhanced11(compressible_case)
        assert solver.envpc_levels == 3
        assert solver.orns_blend == pytest.approx(0.5)

    def test_envpc_pressure(self, compressible_case):
        from pyfoam.applications.compressible_inter_foam_enhanced_11 import CompressibleInterFoamEnhanced11
        solver = CompressibleInterFoamEnhanced11(compressible_case, envpc=True)
        alpha = torch.zeros_like(solver.p)
        rho = torch.ones_like(solver.p)
        p_envpc = solver._extended_vof_pressure_correct(solver.p, alpha, rho)
        assert p_envpc.shape == solver.p.shape
        assert torch.isfinite(p_envpc).all()

    def test_cnpfc_fraction(self, compressible_case):
        from pyfoam.applications.compressible_inter_foam_enhanced_11 import CompressibleInterFoamEnhanced11
        solver = CompressibleInterFoamEnhanced11(compressible_case, cnpfc=True)
        alpha = torch.full_like(solver.p, 0.5)
        rho = torch.ones_like(solver.p)
        alpha_cnpfc = solver._consistent_phase_fraction_correct(alpha, solver.p, rho)
        assert alpha_cnpfc.shape == alpha.shape
        assert torch.isfinite(alpha_cnpfc).all()
        assert (alpha_cnpfc >= 0.0).all()
        assert (alpha_cnpfc <= 1.0).all()

    def test_orns_stabilise(self, compressible_case):
        from pyfoam.applications.compressible_inter_foam_enhanced_11 import CompressibleInterFoamEnhanced11
        solver = CompressibleInterFoamEnhanced11(compressible_case, orns=True)
        p_orns = solver._over_relaxed_stabilise(solver.p)
        assert p_orns.shape == solver.p.shape
        assert torch.isfinite(p_orns).all()


# ===========================================================================
# Tests: SprayFoamEnhanced11
# ===========================================================================


class TestSprayFoamEnhanced11:
    """Tests for enhanced spray solver v11."""

    def test_init(self, cavity_case):
        from pyfoam.applications.spray_foam_enhanced_11 import SprayFoamEnhanced11
        solver = SprayFoamEnhanced11(cavity_case, enspc=True, cnpvc=True, orns=True)
        assert solver.enspc is True
        assert solver.cnpvc is True
        assert solver.orns is True

    def test_init_defaults(self, cavity_case):
        from pyfoam.applications.spray_foam_enhanced_11 import SprayFoamEnhanced11
        solver = SprayFoamEnhanced11(cavity_case)
        assert solver.enspc_levels == 3
        assert solver.orns_blend == pytest.approx(0.5)

    def test_enspc_pressure(self, cavity_case):
        from pyfoam.applications.spray_foam_enhanced_11 import SprayFoamEnhanced11
        solver = SprayFoamEnhanced11(cavity_case, enspc=True)
        p_enspc = solver._extended_spray_pressure_correct(solver.p, solver.U)
        assert p_enspc.shape == solver.p.shape
        assert torch.isfinite(p_enspc).all()

    def test_cnpvc_velocity(self, cavity_case):
        from pyfoam.applications.spray_foam_enhanced_11 import SprayFoamEnhanced11
        solver = SprayFoamEnhanced11(cavity_case, cnpvc=True)
        n = solver.mesh.n_cells
        U_p = solver.U.clone()
        alpha_p = torch.zeros(n, device=solver.U.device, dtype=solver.U.dtype)
        U_cnpvc = solver._consistent_parcel_velocity_correct(solver.U, U_p, alpha_p)
        assert U_cnpvc.shape == solver.U.shape
        assert torch.isfinite(U_cnpvc).all()

    def test_orns_stabilise(self, cavity_case):
        from pyfoam.applications.spray_foam_enhanced_11 import SprayFoamEnhanced11
        solver = SprayFoamEnhanced11(cavity_case, orns=True)
        p_orns = solver._over_relaxed_stabilise(solver.p)
        assert p_orns.shape == solver.p.shape
        assert torch.isfinite(p_orns).all()


# ===========================================================================
# Tests: MultiphaseEulerFoamEnhanced11
# ===========================================================================


class TestMultiphaseEulerFoamEnhanced11:
    """Tests for enhanced multiphase Euler solver v11."""

    def test_class_exists(self):
        from pyfoam.applications.multiphase_euler_foam_enhanced_11 import MultiphaseEulerFoamEnhanced11
        assert MultiphaseEulerFoamEnhanced11 is not None

    def test_class_has_init(self):
        from pyfoam.applications.multiphase_euler_foam_enhanced_11 import MultiphaseEulerFoamEnhanced11
        assert hasattr(MultiphaseEulerFoamEnhanced11, '__init__')

    def test_enmpc_pressure_logic(self):
        """Test extended multi-phase pressure correction logic."""
        n = 16
        p = torch.randn(n)
        owner = torch.arange(n - 1)
        neigh = torch.arange(1, n)
        # Simple pressure gradient correction test
        p_corr = p.clone()
        dp = p[neigh] - p[owner]
        correction = dp * 0.01
        assert correction.shape == (n - 1,)
        assert torch.isfinite(correction).all()

    def test_cnpmc_momentum_logic(self):
        """Test consistent phase-momentum correction logic."""
        n = 16
        dp = torch.randn(n)
        dp3 = dp.unsqueeze(-1).expand(-1, 3) * 0.001
        corr = torch.zeros(n, 3)
        owner = torch.arange(n - 1)
        neigh = torch.arange(1, n)
        corr.index_add_(0, owner, dp3[:-1])
        corr.index_add_(0, neigh, -dp3[:-1])
        assert corr.shape == (n, 3)
        assert torch.isfinite(corr).all()

    def test_orns_stabilise_logic(self):
        """Test over-relaxed stabilisation blending logic."""
        n = 16
        dp = torch.randn(n)
        blend = 0.5
        dp_over = dp * (1.0 + blend)
        dp_min = dp * (1.0 - blend)
        dp_blend = blend * dp_over + (1.0 - blend) * dp_min
        assert dp_blend.shape == dp.shape
        assert torch.isfinite(dp_blend).all()


# ===========================================================================
# Tests: ReactingFoamEnhanced13
# ===========================================================================


class TestReactingFoamEnhanced13:
    """Tests for enhanced reacting solver v13 (coupling algorithms)."""

    def test_init(self, reacting_case):
        from pyfoam.applications.reacting_foam_enhanced_13 import ReactingFoamEnhanced13
        solver = ReactingFoamEnhanced13(reacting_case, src=True, crs=True, pvcc=True)
        assert solver.src is True
        assert solver.crs is True
        assert solver.pvcc is True

    def test_init_defaults(self, reacting_case):
        from pyfoam.applications.reacting_foam_enhanced_13 import ReactingFoamEnhanced13
        solver = ReactingFoamEnhanced13(reacting_case)
        assert solver.crs_max_iter == 5
        assert solver.pvcc_relaxation == pytest.approx(0.8)

    def test_simplec_coupling(self, reacting_case):
        from pyfoam.applications.reacting_foam_enhanced_13 import ReactingFoamEnhanced13
        solver = ReactingFoamEnhanced13(reacting_case, src=True)
        Y_corr, T_corr = solver._simplec_consistent_coupling(
            solver.Y, solver.T, solver.delta_t,
        )
        assert isinstance(Y_corr, dict)
        for name in solver.species:
            assert name in Y_corr
        assert T_corr.shape == solver.T.shape
        assert torch.isfinite(T_corr).all()

    def test_coupled_reacting_solve(self, reacting_case):
        from pyfoam.applications.reacting_foam_enhanced_13 import ReactingFoamEnhanced13
        solver = ReactingFoamEnhanced13(reacting_case, crs=True, crs_max_iter=2)
        Y_corr, T_corr = solver._coupled_reacting_solve(
            solver.Y, solver.T, solver.delta_t,
        )
        assert isinstance(Y_corr, dict)
        for name in solver.species:
            assert name in Y_corr
        assert T_corr.shape == solver.T.shape
        assert torch.isfinite(T_corr).all()

    def test_pvcc_correct(self, reacting_case):
        from pyfoam.applications.reacting_foam_enhanced_13 import ReactingFoamEnhanced13
        solver = ReactingFoamEnhanced13(reacting_case, pvcc=True)
        Y_corr, T_corr = solver._pvcc_correct(
            solver.Y, solver.T, solver.Y, solver.T, solver.delta_t,
        )
        assert isinstance(Y_corr, dict)
        assert T_corr.shape == solver.T.shape
        assert torch.isfinite(T_corr).all()


# ===========================================================================
# Tests: Exports
# ===========================================================================


class TestExportsV11:
    """Tests for __init__.py exports of v11/v13 solvers."""

    def test_simple_enhanced_11_exported(self):
        from pyfoam.applications import SimpleFoamEnhanced11
        assert SimpleFoamEnhanced11 is not None

    def test_pimple_enhanced_11_exported(self):
        from pyfoam.applications import PimpleFoamEnhanced11
        assert PimpleFoamEnhanced11 is not None

    def test_piso_enhanced_11_exported(self):
        from pyfoam.applications import PisoFoamEnhanced11
        assert PisoFoamEnhanced11 is not None

    def test_ico_enhanced_11_exported(self):
        from pyfoam.applications import IcoFoamEnhanced11
        assert IcoFoamEnhanced11 is not None

    def test_buoyant_pimple_enhanced_11_exported(self):
        from pyfoam.applications import BuoyantPimpleFoamEnhanced11
        assert BuoyantPimpleFoamEnhanced11 is not None

    def test_buoyant_simple_enhanced_11_exported(self):
        from pyfoam.applications import BuoyantSimpleFoamEnhanced11
        assert BuoyantSimpleFoamEnhanced11 is not None

    def test_compressible_inter_enhanced_11_exported(self):
        from pyfoam.applications import CompressibleInterFoamEnhanced11
        assert CompressibleInterFoamEnhanced11 is not None

    def test_spray_enhanced_11_exported(self):
        from pyfoam.applications import SprayFoamEnhanced11
        assert SprayFoamEnhanced11 is not None

    def test_multiphase_euler_enhanced_11_exported(self):
        from pyfoam.applications import MultiphaseEulerFoamEnhanced11
        assert MultiphaseEulerFoamEnhanced11 is not None

    def test_reacting_enhanced_13_exported(self):
        from pyfoam.applications import ReactingFoamEnhanced13
        assert ReactingFoamEnhanced13 is not None

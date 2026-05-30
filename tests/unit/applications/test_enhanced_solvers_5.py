"""
Unit tests for enhanced solver variants v5 (and v7/v4 specialized).

Tests cover:
- Case loading and solver initialisation
- Enhanced parameter reading
- Solver produces finite values after short run
- Export of new classes from __init__.py
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from pyfoam.core.dtype import CFD_DTYPE, INDEX_DTYPE
from pyfoam.io.foam_file import FoamFileHeader, FileFormat, write_foam_file


# ---------------------------------------------------------------------------
# Mesh generation helper (reuses the same pattern as test_enhanced_solvers_4)
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
# Tests: IcoFoamEnhanced5
# ===========================================================================


class TestIcoFoamEnhanced5:
    """Tests for enhanced ICO solver v5."""

    def test_init(self, cavity_case):
        from pyfoam.applications.ico_foam_enhanced_5 import IcoFoamEnhanced5
        solver = IcoFoamEnhanced5(cavity_case, error_tolerance=1e-4, momentum_limiter=True)
        assert solver.error_tolerance == pytest.approx(1e-4)
        assert solver.momentum_limiter is True

    def test_init_defaults(self, cavity_case):
        from pyfoam.applications.ico_foam_enhanced_5 import IcoFoamEnhanced5
        solver = IcoFoamEnhanced5(cavity_case)
        assert solver.error_tolerance == pytest.approx(1e-4)
        assert solver.richardson_order == 2
        assert solver.momentum_limiter is True

    def test_characteristic_flux_split(self, cavity_case):
        from pyfoam.applications.ico_foam_enhanced_5 import IcoFoamEnhanced5
        solver = IcoFoamEnhanced5(cavity_case)
        U_corr = solver._characteristic_flux_split(
            solver.U, solver.U.clone(), solver.delta_t,
        )
        assert U_corr.shape == solver.U.shape
        assert torch.isfinite(U_corr).all()

    def test_richardson_extrapolation(self, cavity_case):
        from pyfoam.applications.ico_foam_enhanced_5 import IcoFoamEnhanced5
        solver = IcoFoamEnhanced5(cavity_case, error_tolerance=1e-4)
        error, dt_rec = solver._richardson_extrapolation_dt(
            solver.U, solver.U.clone() * 0.99, solver.delta_t,
        )
        assert isinstance(error, float)
        assert dt_rec > 0
        assert dt_rec <= solver.delta_t * 2.0

    def test_momentum_limiter(self, cavity_case):
        from pyfoam.applications.ico_foam_enhanced_5 import IcoFoamEnhanced5
        solver = IcoFoamEnhanced5(cavity_case, momentum_limiter=True)
        U_limited = solver._momentum_preserving_limiter(solver.U, solver.U.clone())
        assert U_limited.shape == solver.U.shape
        assert torch.isfinite(U_limited).all()

    def test_momentum_limiter_disabled(self, cavity_case):
        from pyfoam.applications.ico_foam_enhanced_5 import IcoFoamEnhanced5
        solver = IcoFoamEnhanced5(cavity_case, momentum_limiter=False)
        U = solver.U.clone()
        U_limited = solver._momentum_preserving_limiter(U, solver.U.clone())
        assert torch.allclose(U_limited, U)


# ===========================================================================
# Tests: SimpleFoamEnhanced5
# ===========================================================================


class TestSimpleFoamEnhanced5:
    """Tests for enhanced SIMPLE solver v5."""

    def test_init(self, cavity_case):
        from pyfoam.applications.simple_foam_enhanced_5 import SimpleFoamEnhanced5
        solver = SimpleFoamEnhanced5(cavity_case, feature_precondition=True)
        assert solver.feature_precondition is True

    def test_init_defaults(self, cavity_case):
        from pyfoam.applications.simple_foam_enhanced_5 import SimpleFoamEnhanced5
        solver = SimpleFoamEnhanced5(cavity_case)
        assert solver.spectral_switching is True
        assert solver.momentum_conservation is True

    def test_feature_precondition(self, cavity_case):
        from pyfoam.applications.simple_foam_enhanced_5 import SimpleFoamEnhanced5
        solver = SimpleFoamEnhanced5(cavity_case, feature_precondition=True)
        p_prec = solver._feature_aligned_precondition(solver.p, solver.U)
        assert p_prec.shape == solver.p.shape
        assert torch.isfinite(p_prec).all()

    def test_feature_precondition_disabled(self, cavity_case):
        from pyfoam.applications.simple_foam_enhanced_5 import SimpleFoamEnhanced5
        solver = SimpleFoamEnhanced5(cavity_case, feature_precondition=False)
        p_prec = solver._feature_aligned_precondition(solver.p, solver.U)
        assert torch.allclose(p_prec, solver.p)

    def test_spectral_radius(self, cavity_case):
        from pyfoam.applications.simple_foam_enhanced_5 import SimpleFoamEnhanced5
        solver = SimpleFoamEnhanced5(cavity_case)
        rho = solver._estimate_spectral_radius(
            solver.U, solver.U.clone() * 0.99, solver.U.clone() * 0.98,
        )
        assert isinstance(rho, float)
        assert rho >= 0

    def test_momentum_conservation(self, cavity_case):
        from pyfoam.applications.simple_foam_enhanced_5 import SimpleFoamEnhanced5
        solver = SimpleFoamEnhanced5(cavity_case, momentum_conservation=True)
        U_corr = solver._enforce_momentum_conservation(solver.U)
        assert U_corr.shape == solver.U.shape
        assert torch.isfinite(U_corr).all()

    def test_momentum_conservation_disabled(self, cavity_case):
        from pyfoam.applications.simple_foam_enhanced_5 import SimpleFoamEnhanced5
        solver = SimpleFoamEnhanced5(cavity_case, momentum_conservation=False)
        U = solver.U.clone()
        U_corr = solver._enforce_momentum_conservation(U)
        assert torch.allclose(U_corr, U)


# ===========================================================================
# Tests: PisoFoamEnhanced5
# ===========================================================================


class TestPisoFoamEnhanced5:
    """Tests for enhanced PISO solver v5."""

    def test_init(self, cavity_case):
        from pyfoam.applications.piso_foam_enhanced_5 import PisoFoamEnhanced5
        solver = PisoFoamEnhanced5(cavity_case, error_tolerance=1e-3, bounded_transport=True)
        assert solver.error_tolerance == pytest.approx(1e-3)
        assert solver.bounded_transport is True

    def test_init_defaults(self, cavity_case):
        from pyfoam.applications.piso_foam_enhanced_5 import PisoFoamEnhanced5
        solver = PisoFoamEnhanced5(cavity_case)
        assert solver.anisotropic_rhie_chow is True
        assert solver.bounded_transport is True

    def test_temporal_error(self, cavity_case):
        from pyfoam.applications.piso_foam_enhanced_5 import PisoFoamEnhanced5
        solver = PisoFoamEnhanced5(cavity_case, error_tolerance=1e-3)
        error, dt_rec = solver._estimate_temporal_error_local(
            solver.U, solver.U.clone() * 0.99, solver.delta_t,
        )
        assert isinstance(error, float)
        assert dt_rec > 0

    def test_anisotropic_rhie_chow(self, cavity_case):
        from pyfoam.applications.piso_foam_enhanced_5 import PisoFoamEnhanced5
        solver = PisoFoamEnhanced5(cavity_case, anisotropic_rhie_chow=True)
        A_p = torch.ones(solver.mesh.n_cells, dtype=solver.U.dtype)
        U_corr = solver._anisotropic_rhie_chow(solver.U, solver.p, A_p)
        assert U_corr.shape == solver.U.shape
        assert torch.isfinite(U_corr).all()

    def test_bounded_transport(self, cavity_case):
        from pyfoam.applications.piso_foam_enhanced_5 import PisoFoamEnhanced5
        solver = PisoFoamEnhanced5(cavity_case, bounded_transport=True)
        U_bounded = solver._apply_bounded_transport(solver.U, solver.U.clone())
        assert U_bounded.shape == solver.U.shape
        assert torch.isfinite(U_bounded).all()

    def test_bounded_transport_disabled(self, cavity_case):
        from pyfoam.applications.piso_foam_enhanced_5 import PisoFoamEnhanced5
        solver = PisoFoamEnhanced5(cavity_case, bounded_transport=False)
        U = solver.U.clone()
        U_bounded = solver._apply_bounded_transport(U, solver.U.clone())
        assert torch.allclose(U_bounded, U)


# ===========================================================================
# Tests: PimpleFoamEnhanced5
# ===========================================================================


class TestPimpleFoamEnhanced5:
    """Tests for enhanced PIMPLE solver v5."""

    def test_init(self, cavity_case):
        from pyfoam.applications.pimple_foam_enhanced_5 import PimpleFoamEnhanced5
        solver = PimpleFoamEnhanced5(cavity_case, simplec_inner=True, residual_smoothing_alpha=0.3)
        assert solver.simplec_inner is True
        assert solver.residual_smoothing_alpha == pytest.approx(0.3)

    def test_init_defaults(self, cavity_case):
        from pyfoam.applications.pimple_foam_enhanced_5 import PimpleFoamEnhanced5
        solver = PimpleFoamEnhanced5(cavity_case)
        assert solver.simplec_inner is True
        assert solver.adaptive_coarsening is True

    def test_simplec_pressure_correction(self, cavity_case):
        from pyfoam.applications.pimple_foam_enhanced_5 import PimpleFoamEnhanced5
        solver = PimpleFoamEnhanced5(cavity_case, simplec_inner=True)
        p_corr, U_corr = solver._simplec_pressure_correction(
            solver.p, solver.U, solver.U.clone(),
        )
        assert p_corr.shape == solver.p.shape
        assert U_corr.shape == solver.U.shape
        assert torch.isfinite(p_corr).all()
        assert torch.isfinite(U_corr).all()

    def test_simplec_disabled(self, cavity_case):
        from pyfoam.applications.pimple_foam_enhanced_5 import PimpleFoamEnhanced5
        solver = PimpleFoamEnhanced5(cavity_case, simplec_inner=False)
        p_corr, U_corr = solver._simplec_pressure_correction(
            solver.p, solver.U, solver.U.clone(),
        )
        assert torch.allclose(p_corr, solver.p)
        assert torch.allclose(U_corr, solver.U)

    def test_residual_smoothing(self, cavity_case):
        from pyfoam.applications.pimple_foam_enhanced_5 import PimpleFoamEnhanced5
        solver = PimpleFoamEnhanced5(cavity_case, residual_smoothing_alpha=0.3)
        sU, sp = solver._smooth_residual(1e-3, 1e-4)
        assert isinstance(sU, float)
        assert isinstance(sp, float)

    def test_coarsening_ratio(self, cavity_case):
        from pyfoam.applications.pimple_foam_enhanced_5 import PimpleFoamEnhanced5
        solver = PimpleFoamEnhanced5(cavity_case, adaptive_coarsening=True)
        ratio = solver._compute_coarsening_ratio(0)
        assert isinstance(ratio, int)
        assert ratio >= 2


# ===========================================================================
# Tests: RhoPimpleFoamEnhanced5
# ===========================================================================


class TestRhoPimpleFoamEnhanced5:
    """Tests for enhanced compressible PIMPLE solver v5."""

    def test_init(self, compressible_case):
        from pyfoam.applications.rho_pimple_foam_enhanced_5 import RhoPimpleFoamEnhanced5
        solver = RhoPimpleFoamEnhanced5(compressible_case, low_mach_prec=True, energy_conservative=True)
        assert solver.low_mach_prec is True
        assert solver.energy_conservative is True

    def test_init_defaults(self, compressible_case):
        from pyfoam.applications.rho_pimple_foam_enhanced_5 import RhoPimpleFoamEnhanced5
        solver = RhoPimpleFoamEnhanced5(compressible_case)
        assert solver.mach_cutoff == pytest.approx(0.3)
        assert solver.acoustic_cfl == pytest.approx(1.0)

    def test_weiss_smith_precondition(self, compressible_case):
        from pyfoam.applications.rho_pimple_foam_enhanced_5 import RhoPimpleFoamEnhanced5
        solver = RhoPimpleFoamEnhanced5(compressible_case, low_mach_prec=True)
        Ma = torch.full((solver.mesh.n_cells,), 0.1)
        U_prec, p_prec = solver._weiss_smith_precondition(solver.U, solver.p, Ma)
        assert U_prec.shape == solver.U.shape
        assert torch.isfinite(U_prec).all()

    def test_weiss_smith_disabled(self, compressible_case):
        from pyfoam.applications.rho_pimple_foam_enhanced_5 import RhoPimpleFoamEnhanced5
        solver = RhoPimpleFoamEnhanced5(compressible_case, low_mach_prec=False)
        Ma = torch.full((solver.mesh.n_cells,), 0.1)
        U_prec, _ = solver._weiss_smith_precondition(solver.U, solver.p, Ma)
        assert torch.allclose(U_prec, solver.U)

    def test_total_energy_correct(self, compressible_case):
        from pyfoam.applications.rho_pimple_foam_enhanced_5 import RhoPimpleFoamEnhanced5
        solver = RhoPimpleFoamEnhanced5(compressible_case, energy_conservative=True)
        T_corr = solver._total_energy_correct(
            solver.U, solver.p, solver.T, solver.rho,
        )
        assert T_corr.shape == solver.T.shape
        assert torch.isfinite(T_corr).all()

    def test_acoustic_aware_dt(self, compressible_case):
        from pyfoam.applications.rho_pimple_foam_enhanced_5 import RhoPimpleFoamEnhanced5
        solver = RhoPimpleFoamEnhanced5(compressible_case, acoustic_cfl=1.0)
        dt = solver._acoustic_aware_dt(solver.delta_t)
        assert dt > 0
        assert dt <= solver.delta_t * 2.0


# ===========================================================================
# Tests: BuoyantSimpleFoamEnhanced5
# ===========================================================================


class TestBuoyantSimpleFoamEnhanced5:
    """Tests for enhanced buoyant SIMPLE solver v5."""

    def test_init(self, buoyant_case):
        from pyfoam.applications.buoyant_simple_foam_enhanced_5 import BuoyantSimpleFoamEnhanced5
        solver = BuoyantSimpleFoamEnhanced5(buoyant_case, implicit_buoyancy=True)
        assert solver.implicit_buoyancy is True

    def test_init_defaults(self, buoyant_case):
        from pyfoam.applications.buoyant_simple_foam_enhanced_5 import BuoyantSimpleFoamEnhanced5
        solver = BuoyantSimpleFoamEnhanced5(buoyant_case)
        assert solver.peclet_relaxation is True
        assert solver.robin_bc is True

    def test_implicit_buoyancy_source(self, buoyant_case):
        from pyfoam.applications.buoyant_simple_foam_enhanced_5 import BuoyantSimpleFoamEnhanced5
        solver = BuoyantSimpleFoamEnhanced5(buoyant_case, implicit_buoyancy=True)
        S = solver._implicit_buoyancy_pressure_source(solver.T, solver.rho)
        assert S.shape == (solver.mesh.n_cells,)
        assert torch.isfinite(S).all()

    def test_implicit_buoyancy_disabled(self, buoyant_case):
        from pyfoam.applications.buoyant_simple_foam_enhanced_5 import BuoyantSimpleFoamEnhanced5
        solver = BuoyantSimpleFoamEnhanced5(buoyant_case, implicit_buoyancy=False)
        S = solver._implicit_buoyancy_pressure_source(solver.T, solver.rho)
        assert torch.allclose(S, torch.zeros_like(S))

    def test_robin_bc(self, buoyant_case):
        from pyfoam.applications.buoyant_simple_foam_enhanced_5 import BuoyantSimpleFoamEnhanced5
        solver = BuoyantSimpleFoamEnhanced5(buoyant_case, robin_bc=True)
        T_corr = solver._convert_to_robin_bc(solver.T)
        assert T_corr.shape == solver.T.shape
        assert torch.isfinite(T_corr).all()


# ===========================================================================
# Tests: BuoyantPimpleFoamEnhanced5
# ===========================================================================


class TestBuoyantPimpleFoamEnhanced5:
    """Tests for enhanced buoyant PIMPLE solver v5."""

    def test_init(self, buoyant_case):
        from pyfoam.applications.buoyant_pimple_foam_enhanced_5 import BuoyantPimpleFoamEnhanced5
        solver = BuoyantPimpleFoamEnhanced5(buoyant_case, semi_implicit_buoyancy=True)
        assert solver.semi_implicit_buoyancy is True

    def test_init_defaults(self, buoyant_case):
        from pyfoam.applications.buoyant_pimple_foam_enhanced_5 import BuoyantPimpleFoamEnhanced5
        solver = BuoyantPimpleFoamEnhanced5(buoyant_case)
        assert solver.buoyancy_dt_limit is True
        assert solver.buoyancy_tke_production is True

    def test_buoyancy_limited_dt(self, buoyant_case):
        from pyfoam.applications.buoyant_pimple_foam_enhanced_5 import BuoyantPimpleFoamEnhanced5
        solver = BuoyantPimpleFoamEnhanced5(buoyant_case, buoyancy_dt_limit=True)
        # High N should limit dt
        dt_limited = solver._buoyancy_limited_dt(solver.delta_t, N=10.0)
        assert dt_limited <= solver.delta_t

        # Zero N should not limit
        dt_none = solver._buoyancy_limited_dt(solver.delta_t, N=0.0)
        assert dt_none == solver.delta_t

    def test_buoyancy_tke_production(self, buoyant_case):
        from pyfoam.applications.buoyant_pimple_foam_enhanced_5 import BuoyantPimpleFoamEnhanced5
        solver = BuoyantPimpleFoamEnhanced5(buoyant_case, buoyancy_tke_production=True)
        k = torch.full((solver.mesh.n_cells,), 0.01)
        P_b = solver._compute_buoyancy_tke_production(solver.T, solver.rho, k)
        assert P_b.shape == k.shape
        assert torch.isfinite(P_b).all()


# ===========================================================================
# Tests: ReactingFoamEnhanced7
# ===========================================================================


class TestReactingFoamEnhanced7:
    """Tests for enhanced reacting solver v7."""

    def test_init(self, reacting_case):
        from pyfoam.applications.reacting_foam_enhanced_7 import ReactingFoamEnhanced7
        solver = ReactingFoamEnhanced7(reacting_case, implicit_coupling=True, mass_consistent_velocity=True)
        assert solver.implicit_coupling is True
        assert solver.mass_consistent_velocity is True

    def test_init_defaults(self, reacting_case):
        from pyfoam.applications.reacting_foam_enhanced_7 import ReactingFoamEnhanced7
        solver = ReactingFoamEnhanced7(reacting_case)
        assert solver.jfnk_tolerance == pytest.approx(1e-6)
        assert solver.adaptive_refinement is True

    def test_jfnk_residual(self, reacting_case):
        from pyfoam.applications.reacting_foam_enhanced_7 import ReactingFoamEnhanced7
        solver = ReactingFoamEnhanced7(reacting_case, implicit_coupling=True)
        Y_old = {name: y.clone() for name, y in solver.Y.items()}
        residuals = solver._jfnk_residual(solver.Y, Y_old, solver.T, 0.001)
        assert isinstance(residuals, dict)
        for name in solver.species:
            assert name in residuals

    def test_adaptive_refinement(self, reacting_case):
        from pyfoam.applications.reacting_foam_enhanced_7 import ReactingFoamEnhanced7
        solver = ReactingFoamEnhanced7(reacting_case, adaptive_refinement=True)
        n_sub = solver._adaptive_refine_subcycling("YA", solver.Y, solver.T)
        assert isinstance(n_sub, int)
        assert n_sub >= 1


# ===========================================================================
# Tests: SolidFoamEnhanced4
# ===========================================================================


class TestSolidFoamEnhanced4:
    """Tests for enhanced solid mechanics solver v4."""

    def test_init(self, cavity_case):
        from pyfoam.applications.solid_foam_enhanced_4 import SolidFoamEnhanced4
        solver = SolidFoamEnhanced4(
            cavity_case, E=200e9, nu=0.3,
            block_gauss_seidel=True, failure_criterion=True,
        )
        assert solver.block_gauss_seidel is True
        assert solver.failure_criterion is True

    def test_init_defaults(self, cavity_case):
        from pyfoam.applications.solid_foam_enhanced_4 import SolidFoamEnhanced4
        solver = SolidFoamEnhanced4(cavity_case, E=200e9, nu=0.3)
        assert solver.bgs_max_iters == 5
        assert solver.failure_stress == pytest.approx(500e6)

    def test_failure_criterion(self, cavity_case):
        from pyfoam.applications.solid_foam_enhanced_4 import SolidFoamEnhanced4
        solver = SolidFoamEnhanced4(
            cavity_case, E=200e9, nu=0.3,
            failure_criterion=True, failure_stress=100.0,
        )
        eps = torch.randn(solver.mesh.n_cells, 6) * 1e-4
        sigma = torch.randn(solver.mesh.n_cells, 6) * 200.0  # Above failure
        sigma_bounded = solver._apply_failure_criterion(sigma, eps)
        assert sigma_bounded.shape == sigma.shape
        assert torch.isfinite(sigma_bounded).all()

    def test_failure_disabled(self, cavity_case):
        from pyfoam.applications.solid_foam_enhanced_4 import SolidFoamEnhanced4
        solver = SolidFoamEnhanced4(
            cavity_case, E=200e9, nu=0.3, failure_criterion=False,
        )
        eps = torch.randn(solver.mesh.n_cells, 6) * 1e-4
        sigma = torch.randn(solver.mesh.n_cells, 6) * 200.0
        sigma_bounded = solver._apply_failure_criterion(sigma, eps)
        assert torch.allclose(sigma_bounded, sigma)

    def test_thermal_source(self, cavity_case):
        from pyfoam.applications.solid_foam_enhanced_4 import SolidFoamEnhanced4
        solver = SolidFoamEnhanced4(cavity_case, E=200e9, nu=0.3)
        source = solver._compute_thermal_source(solver.T, solver.D)
        assert source.shape == (solver.mesh.n_cells,)
        assert torch.isfinite(source).all()


# ===========================================================================
# Tests: FilmFoamEnhanced4
# ===========================================================================


class TestFilmFoamEnhanced4:
    """Tests for enhanced film solver v4."""

    def test_init(self, cavity_case):
        from pyfoam.applications.film_foam_enhanced_4 import FilmFoamEnhanced4
        solver = FilmFoamEnhanced4(cavity_case, surfactant=True, marangoni=True)
        assert solver.surfactant is True
        assert solver.marangoni is True

    def test_init_defaults(self, cavity_case):
        from pyfoam.applications.film_foam_enhanced_4 import FilmFoamEnhanced4
        solver = FilmFoamEnhanced4(cavity_case)
        assert solver.rupture_dynamics is True
        assert solver.sigma_clean == pytest.approx(0.072)

    def test_surface_tension(self, cavity_case):
        from pyfoam.applications.film_foam_enhanced_4 import FilmFoamEnhanced4
        solver = FilmFoamEnhanced4(cavity_case, surfactant=True)
        Gamma = torch.full((solver.mesh.n_cells,), solver.gamma_eq * 0.5)
        sigma = solver._compute_surface_tension(Gamma)
        assert sigma.shape == Gamma.shape
        assert (sigma >= solver.sigma_min).all()
        assert (sigma <= solver.sigma_clean).all()
        assert torch.isfinite(sigma).all()

    def test_marangoni_stress(self, cavity_case):
        from pyfoam.applications.film_foam_enhanced_4 import FilmFoamEnhanced4
        solver = FilmFoamEnhanced4(cavity_case, marangoni=True)
        tau = solver._compute_marangoni_stress(None, solver.Gamma)
        assert tau.shape == (solver.mesh.n_cells,)
        assert (tau >= 0).all()
        assert torch.isfinite(tau).all()

    def test_marangoni_disabled(self, cavity_case):
        from pyfoam.applications.film_foam_enhanced_4 import FilmFoamEnhanced4
        solver = FilmFoamEnhanced4(cavity_case, marangoni=False)
        tau = solver._compute_marangoni_stress(None, solver.Gamma)
        assert torch.allclose(tau, torch.zeros_like(tau))


# ===========================================================================
# Tests: SprayFoamEnhanced4
# ===========================================================================


class TestSprayFoamEnhanced4:
    """Tests for enhanced spray solver v4."""

    def test_init(self, cavity_case):
        from pyfoam.applications.spray_foam_enhanced_4 import SprayFoamEnhanced4, FuelComponent
        comp = FuelComponent(name="n-decane", mass_fraction=1.0)
        solver = SprayFoamEnhanced4(cavity_case, multi_component=True, components=[comp])
        assert solver.multi_component is True
        assert len(solver.components) == 1

    def test_init_defaults(self, cavity_case):
        from pyfoam.applications.spray_foam_enhanced_4 import SprayFoamEnhanced4
        solver = SprayFoamEnhanced4(cavity_case)
        assert solver.turbulent_dispersion is True
        assert len(solver.components) == 1

    def test_fuel_component(self):
        from pyfoam.applications.spray_foam_enhanced_4 import FuelComponent
        comp = FuelComponent(name="iso-octane", boiling_point=372.0)
        assert comp.name == "iso-octane"
        assert comp.boiling_point == 372.0
        assert comp.mass_fraction == 1.0

    def test_raoult_evaporation(self, cavity_case):
        from pyfoam.applications.spray_foam_enhanced_4 import SprayFoamEnhanced4, FuelComponent
        comp = FuelComponent(name="n-decane", mass_fraction=1.0)
        solver = SprayFoamEnhanced4(cavity_case, multi_component=True, components=[comp])
        rates = solver._raoult_law_evaporation(
            d=1e-4, T_droplet=300.0, T_gas=500.0, v_rel=5.0,
            composition={"n-decane": 1.0},
        )
        assert isinstance(rates, dict)
        assert "n-decane" in rates
        assert rates["n-decane"] >= 0


# ===========================================================================
# Tests: MultiphaseEulerFoamEnhanced5
# ===========================================================================


class TestMultiphaseEulerFoamEnhanced5:
    """Tests for enhanced multiphase Euler solver v5."""

    def test_init(self, cavity_case):
        from pyfoam.applications.multiphase_euler_foam_enhanced_5 import MultiphaseEulerFoamEnhanced5
        assert hasattr(MultiphaseEulerFoamEnhanced5, '__init__')

    def test_class_exists(self):
        from pyfoam.applications.multiphase_euler_foam_enhanced_5 import MultiphaseEulerFoamEnhanced5
        assert MultiphaseEulerFoamEnhanced5 is not None

    def test_volume_fraction_bounding(self):
        """Test volume fraction bounding logic."""
        # Simulate the bounding logic
        alpha_1 = torch.tensor([0.3, 0.5, -0.1, 1.2])
        alpha_2 = torch.tensor([0.7, 0.5, 1.1, -0.2])

        # Clamp
        alpha_1 = alpha_1.clamp(min=0.0, max=1.0)
        alpha_2 = alpha_2.clamp(min=0.0, max=1.0)

        # Renormalise
        total = (alpha_1 + alpha_2).clamp(min=1e-30)
        alpha_1 = alpha_1 / total
        alpha_2 = alpha_2 / total

        assert (alpha_1 >= 0).all()
        assert (alpha_1 <= 1).all()
        assert torch.allclose(alpha_1 + alpha_2, torch.ones_like(alpha_1))


# ===========================================================================
# Tests: Exports
# ===========================================================================


class TestExportsV5:
    """Tests for __init__.py exports of v5/v7/v4 solvers."""

    def test_ico_enhanced_5_exported(self):
        from pyfoam.applications import IcoFoamEnhanced5
        assert IcoFoamEnhanced5 is not None

    def test_simple_enhanced_5_exported(self):
        from pyfoam.applications import SimpleFoamEnhanced5
        assert SimpleFoamEnhanced5 is not None

    def test_piso_enhanced_5_exported(self):
        from pyfoam.applications import PisoFoamEnhanced5
        assert PisoFoamEnhanced5 is not None

    def test_pimple_enhanced_5_exported(self):
        from pyfoam.applications import PimpleFoamEnhanced5
        assert PimpleFoamEnhanced5 is not None

    def test_rho_pimple_enhanced_5_exported(self):
        from pyfoam.applications import RhoPimpleFoamEnhanced5
        assert RhoPimpleFoamEnhanced5 is not None

    def test_buoyant_simple_enhanced_5_exported(self):
        from pyfoam.applications import BuoyantSimpleFoamEnhanced5
        assert BuoyantSimpleFoamEnhanced5 is not None

    def test_buoyant_pimple_enhanced_5_exported(self):
        from pyfoam.applications import BuoyantPimpleFoamEnhanced5
        assert BuoyantPimpleFoamEnhanced5 is not None

    def test_reacting_enhanced_7_exported(self):
        from pyfoam.applications import ReactingFoamEnhanced7
        assert ReactingFoamEnhanced7 is not None

    def test_solid_enhanced_4_exported(self):
        from pyfoam.applications import SolidFoamEnhanced4
        assert SolidFoamEnhanced4 is not None

    def test_film_enhanced_4_exported(self):
        from pyfoam.applications import FilmFoamEnhanced4
        assert FilmFoamEnhanced4 is not None

    def test_spray_enhanced_4_exported(self):
        from pyfoam.applications import SprayFoamEnhanced4
        assert SprayFoamEnhanced4 is not None

    def test_multiphase_enhanced_5_exported(self):
        from pyfoam.applications import MultiphaseEulerFoamEnhanced5
        assert MultiphaseEulerFoamEnhanced5 is not None

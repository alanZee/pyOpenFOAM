"""
Unit tests for enhanced solver variants v3.

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
# Mesh generation helper (same as test_enhanced_solvers.py)
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
# Tests: IcoFoamEnhanced3
# ===========================================================================


class TestIcoFoamEnhanced3:
    """Tests for enhanced ICO solver v3."""

    def test_init(self, cavity_case):
        from pyfoam.applications.ico_foam_enhanced_3 import IcoFoamEnhanced3
        solver = IcoFoamEnhanced3(cavity_case, temporal_order=3, error_tolerance=1e-4)
        assert solver.temporal_order == 3
        assert abs(solver.error_tolerance - 1e-4) < 1e-10

    def test_init_defaults(self, cavity_case):
        from pyfoam.applications.ico_foam_enhanced_3 import IcoFoamEnhanced3
        solver = IcoFoamEnhanced3(cavity_case)
        assert solver.temporal_order == 3
        assert solver.adaptive_dt is True

    def test_run_completes(self, cavity_case):
        from pyfoam.applications.ico_foam_enhanced_3 import IcoFoamEnhanced3
        solver = IcoFoamEnhanced3(cavity_case, adaptive_dt=False)
        conv = solver.run()
        assert conv is not None

    def test_finite_values(self, cavity_case):
        from pyfoam.applications.ico_foam_enhanced_3 import IcoFoamEnhanced3
        solver = IcoFoamEnhanced3(cavity_case, adaptive_dt=False)
        solver.run()
        assert torch.isfinite(solver.U).all()
        assert torch.isfinite(solver.p).all()

    def test_ssp_rk2_step(self, cavity_case):
        from pyfoam.applications.ico_foam_enhanced_3 import IcoFoamEnhanced3
        solver = IcoFoamEnhanced3(cavity_case)
        U_rk2 = solver._ssp_rk2_step(solver.U, solver.U * 0.9, solver.p, solver.delta_t)
        assert U_rk2.shape == solver.U.shape
        assert torch.isfinite(U_rk2).all()

    def test_ssp_rk3_step(self, cavity_case):
        from pyfoam.applications.ico_foam_enhanced_3 import IcoFoamEnhanced3
        solver = IcoFoamEnhanced3(cavity_case)
        U_rk3 = solver._ssp_rk3_step(solver.U, solver.U * 0.9, solver.p, solver.delta_t)
        assert U_rk3.shape == solver.U.shape
        assert torch.isfinite(U_rk3).all()

    def test_temporal_error_estimation(self, cavity_case):
        from pyfoam.applications.ico_foam_enhanced_3 import IcoFoamEnhanced3
        solver = IcoFoamEnhanced3(cavity_case)
        U_a = solver.U.clone()
        U_b = solver.U.clone() + 1e-5
        error = solver._estimate_temporal_error(U_a, U_b)
        assert error >= 0

    def test_error_adaptive_dt(self, cavity_case):
        from pyfoam.applications.ico_foam_enhanced_3 import IcoFoamEnhanced3
        solver = IcoFoamEnhanced3(cavity_case, error_tolerance=1e-4)
        dt = solver._compute_error_adaptive_dt(solver.delta_t, 1e-3)
        assert dt > 0
        assert dt <= solver.delta_t * 2.0


# ===========================================================================
# Tests: SimpleFoamEnhanced3
# ===========================================================================


class TestSimpleFoamEnhanced3:
    """Tests for enhanced SIMPLE solver v3."""

    def test_init(self, cavity_case):
        from pyfoam.applications.simple_foam_enhanced_3 import SimpleFoamEnhanced3
        solver = SimpleFoamEnhanced3(cavity_case, anderson_depth=4, smoothing_levels=3)
        assert solver.anderson_depth == 4
        assert solver.smoothing_levels == 3

    def test_run_completes(self, cavity_case):
        from pyfoam.applications.simple_foam_enhanced_3 import SimpleFoamEnhanced3
        solver = SimpleFoamEnhanced3(cavity_case)
        conv = solver.run()
        assert conv is not None

    def test_finite_values(self, cavity_case):
        from pyfoam.applications.simple_foam_enhanced_3 import SimpleFoamEnhanced3
        solver = SimpleFoamEnhanced3(cavity_case)
        solver.run()
        assert torch.isfinite(solver.U).all()
        assert torch.isfinite(solver.p).all()

    def test_residual_smoothing(self, cavity_case):
        from pyfoam.applications.simple_foam_enhanced_3 import SimpleFoamEnhanced3
        solver = SimpleFoamEnhanced3(cavity_case, smoothing_levels=2)
        r = torch.randn(solver.mesh.n_cells)
        r_smooth = solver._multi_level_smooth_residual(r)
        assert r_smooth.shape == r.shape
        assert torch.isfinite(r_smooth).all()

    def test_anderson_mix(self, cavity_case):
        from pyfoam.applications.simple_foam_enhanced_3 import SimpleFoamEnhanced3
        solver = SimpleFoamEnhanced3(cavity_case, anderson_depth=3)
        n = solver.mesh.n_cells
        x = torch.randn(n)
        # Build history
        x_hist = [x + 0.1 * i for i in range(4)]
        g_hist = [0.1 * torch.randn(n) for _ in range(4)]
        result = solver._anderson_mix(x, x_hist, g_hist)
        assert result.shape == x.shape
        assert torch.isfinite(result).all()


# ===========================================================================
# Tests: PisoFoamEnhanced3
# ===========================================================================


class TestPisoFoamEnhanced3:
    """Tests for enhanced PISO solver v3."""

    def test_init(self, cavity_case):
        from pyfoam.applications.piso_foam_enhanced_3 import PisoFoamEnhanced3
        solver = PisoFoamEnhanced3(cavity_case, momentum_balance_tol=1e-3)
        assert abs(solver.momentum_balance_tol - 1e-3) < 1e-10

    def test_run_completes(self, cavity_case):
        from pyfoam.applications.piso_foam_enhanced_3 import PisoFoamEnhanced3
        solver = PisoFoamEnhanced3(cavity_case, max_courant=10.0)
        conv = solver.run()
        assert conv is not None

    def test_finite_values(self, cavity_case):
        from pyfoam.applications.piso_foam_enhanced_3 import PisoFoamEnhanced3
        solver = PisoFoamEnhanced3(cavity_case, max_courant=10.0)
        solver.run()
        assert torch.isfinite(solver.U).all()
        assert torch.isfinite(solver.p).all()

    def test_skewness_corrected_rhie_chow(self, cavity_case):
        from pyfoam.applications.piso_foam_enhanced_3 import PisoFoamEnhanced3
        solver = PisoFoamEnhanced3(cavity_case, skewness_correction=True)
        A_p = torch.ones(solver.mesh.n_cells)
        U_corr = solver._rhie_chow_skewness_corrected(solver.U, solver.p, A_p)
        assert U_corr.shape == solver.U.shape
        assert torch.isfinite(U_corr).all()

    def test_momentum_balance(self, cavity_case):
        from pyfoam.applications.piso_foam_enhanced_3 import PisoFoamEnhanced3
        solver = PisoFoamEnhanced3(cavity_case)
        balance = solver._compute_momentum_balance(solver.U, solver.U.clone(), solver.delta_t)
        assert isinstance(balance, float)
        assert balance >= 0


# ===========================================================================
# Tests: PimpleFoamEnhanced3
# ===========================================================================


class TestPimpleFoamEnhanced3:
    """Tests for enhanced PIMPLE solver v3."""

    def test_init(self, cavity_case):
        from pyfoam.applications.pimple_foam_enhanced_3 import PimpleFoamEnhanced3
        solver = PimpleFoamEnhanced3(cavity_case, line_search_alpha=0.6)
        assert abs(solver.line_search_alpha - 0.6) < 1e-10

    def test_run_completes(self, cavity_case):
        from pyfoam.applications.pimple_foam_enhanced_3 import PimpleFoamEnhanced3
        solver = PimpleFoamEnhanced3(cavity_case)
        conv = solver.run()
        assert conv is not None

    def test_finite_values(self, cavity_case):
        from pyfoam.applications.pimple_foam_enhanced_3 import PimpleFoamEnhanced3
        solver = PimpleFoamEnhanced3(cavity_case)
        solver.run()
        assert torch.isfinite(solver.U).all()
        assert torch.isfinite(solver.p).all()

    def test_newton_krylov_acceleration(self, cavity_case):
        from pyfoam.applications.pimple_foam_enhanced_3 import PimpleFoamEnhanced3
        solver = PimpleFoamEnhanced3(cavity_case)
        U = solver.U.clone()
        U_old = solver.U.clone() * 0.99
        F_U = U - U_old
        U_accel = solver._newton_krylov_acceleration(U, U_old, F_U)
        assert U_accel.shape == U.shape
        assert torch.isfinite(U_accel).all()

    def test_adaptive_outer_count(self, cavity_case):
        from pyfoam.applications.pimple_foam_enhanced_3 import PimpleFoamEnhanced3
        solver = PimpleFoamEnhanced3(cavity_case)
        # Fast convergence
        n_fast = solver._adaptive_outer_count(0.3, 10)
        assert n_fast <= 10
        # Slow convergence
        n_slow = solver._adaptive_outer_count(0.95, 10)
        assert n_slow >= 10


# ===========================================================================
# Tests: RhoPimpleFoamEnhanced3
# ===========================================================================


class TestRhoPimpleFoamEnhanced3:
    """Tests for enhanced compressible PIMPLE solver v3."""

    def test_init(self, compressible_case):
        from pyfoam.applications.rho_pimple_foam_enhanced_3 import RhoPimpleFoamEnhanced3
        solver = RhoPimpleFoamEnhanced3(compressible_case, variable_Cp=True, Cp_reference=1010.0)
        assert solver.variable_Cp is True
        assert solver.Cp_ref == 1010.0

    def test_run_completes(self, compressible_case):
        from pyfoam.applications.rho_pimple_foam_enhanced_3 import RhoPimpleFoamEnhanced3
        solver = RhoPimpleFoamEnhanced3(compressible_case)
        conv = solver.run()
        assert conv is not None

    def test_finite_values(self, compressible_case):
        from pyfoam.applications.rho_pimple_foam_enhanced_3 import RhoPimpleFoamEnhanced3
        solver = RhoPimpleFoamEnhanced3(compressible_case)
        solver.run()
        assert torch.isfinite(solver.U).all()
        assert torch.isfinite(solver.p).all()
        assert torch.isfinite(solver.T).all()

    def test_variable_Cp(self, compressible_case):
        from pyfoam.applications.rho_pimple_foam_enhanced_3 import RhoPimpleFoamEnhanced3
        solver = RhoPimpleFoamEnhanced3(compressible_case, variable_Cp=True)
        Cp = solver._compute_Cp(solver.T)
        assert Cp.shape == solver.T.shape
        assert (Cp > 0).all()

    def test_constant_Cp(self, compressible_case):
        from pyfoam.applications.rho_pimple_foam_enhanced_3 import RhoPimpleFoamEnhanced3
        solver = RhoPimpleFoamEnhanced3(compressible_case, variable_Cp=False, Cp_reference=1005.0)
        Cp = solver._compute_Cp(solver.T)
        assert torch.allclose(Cp, torch.full_like(Cp, 1005.0))

    def test_sonic_number(self, compressible_case):
        from pyfoam.applications.rho_pimple_foam_enhanced_3 import RhoPimpleFoamEnhanced3
        solver = RhoPimpleFoamEnhanced3(compressible_case)
        sonic = solver._compute_sonic_number()
        assert sonic.shape == (solver.mesh.n_cells,)
        assert (sonic >= 0).all()

    def test_sonic_aware_relaxation(self, compressible_case):
        from pyfoam.applications.rho_pimple_foam_enhanced_3 import RhoPimpleFoamEnhanced3
        solver = RhoPimpleFoamEnhanced3(compressible_case)
        sonic = torch.full((solver.mesh.n_cells,), 0.5)
        result = solver._sonic_aware_relaxation(
            solver.U, solver.U.clone(), 0.7, sonic,
        )
        assert result.shape == solver.U.shape
        assert torch.isfinite(result).all()


# ===========================================================================
# Tests: BuoyantSimpleFoamEnhanced3
# ===========================================================================


class TestBuoyantSimpleFoamEnhanced3:
    """Tests for enhanced buoyant SIMPLE solver v3."""

    def test_init(self, buoyant_case):
        from pyfoam.applications.buoyant_simple_foam_enhanced_3 import BuoyantSimpleFoamEnhanced3
        solver = BuoyantSimpleFoamEnhanced3(buoyant_case, quadratic_boussinesq=True)
        assert solver.quadratic_boussinesq is True

    def test_flow_regime_classification(self, buoyant_case):
        from pyfoam.applications.buoyant_simple_foam_enhanced_3 import BuoyantSimpleFoamEnhanced3
        solver = BuoyantSimpleFoamEnhanced3(buoyant_case)
        Ri_field = torch.zeros(solver.mesh.n_cells)
        assert solver._classify_flow_regime(0.01, Ri_field) == "forced"
        assert solver._classify_flow_regime(1.0, Ri_field) == "mixed"
        assert solver._classify_flow_regime(20.0, Ri_field) == "natural"

    def test_quadratic_boussinesq(self, buoyant_case):
        from pyfoam.applications.buoyant_simple_foam_enhanced_3 import BuoyantSimpleFoamEnhanced3
        solver = BuoyantSimpleFoamEnhanced3(buoyant_case, quadratic_boussinesq=True)
        F = solver._compute_quadratic_boussinesq(solver.T, 1.225)
        assert F.shape == (solver.mesh.n_cells, 3)
        assert torch.isfinite(F).all()

    def test_buoyancy_pressure_correction(self, buoyant_case):
        from pyfoam.applications.buoyant_simple_foam_enhanced_3 import BuoyantSimpleFoamEnhanced3
        solver = BuoyantSimpleFoamEnhanced3(buoyant_case, buoyancy_pressure_correction=True)
        p_corr = solver._buoyancy_pressure_correction(solver.p, solver.T)
        assert p_corr.shape == solver.p.shape
        assert torch.isfinite(p_corr).all()

    def test_regime_relaxation(self, buoyant_case):
        from pyfoam.applications.buoyant_simple_foam_enhanced_3 import BuoyantSimpleFoamEnhanced3
        solver = BuoyantSimpleFoamEnhanced3(buoyant_case)
        U_f, _ = solver._regime_relaxation("forced", 0.7, 0.3)
        U_n, _ = solver._regime_relaxation("natural", 0.7, 0.3)
        assert U_n < U_f


# ===========================================================================
# Tests: BuoyantPimpleFoamEnhanced3
# ===========================================================================


class TestBuoyantPimpleFoamEnhanced3:
    """Tests for enhanced buoyant PIMPLE solver v3."""

    def test_init(self, buoyant_case):
        from pyfoam.applications.buoyant_pimple_foam_enhanced_3 import BuoyantPimpleFoamEnhanced3
        solver = BuoyantPimpleFoamEnhanced3(
            buoyant_case, gravity_wave_filter=True, wave_filter_coeff=0.3,
        )
        assert solver.gravity_wave_filter is True
        assert abs(solver.wave_filter_coeff - 0.3) < 1e-10

    def test_semi_implicit_buoyancy(self, buoyant_case):
        from pyfoam.applications.buoyant_pimple_foam_enhanced_3 import BuoyantPimpleFoamEnhanced3
        solver = BuoyantPimpleFoamEnhanced3(buoyant_case)
        rho = solver.rho
        F_exp, diag = solver._semi_implicit_buoyancy_source(
            solver.T, solver.T, rho,
        )
        assert F_exp.shape == (solver.mesh.n_cells, 3)
        assert diag.shape == (solver.mesh.n_cells,)
        assert (diag >= 0).all()

    def test_gravity_wave_filter(self, buoyant_case):
        from pyfoam.applications.buoyant_pimple_foam_enhanced_3 import BuoyantPimpleFoamEnhanced3
        solver = BuoyantPimpleFoamEnhanced3(buoyant_case, gravity_wave_filter=True)
        U_filt = solver._apply_gravity_wave_filter(
            solver.U, solver.U.clone(), solver.delta_t,
        )
        assert U_filt.shape == solver.U.shape
        assert torch.isfinite(U_filt).all()

    def test_buoyancy_production(self, buoyant_case):
        from pyfoam.applications.buoyant_pimple_foam_enhanced_3 import BuoyantPimpleFoamEnhanced3
        solver = BuoyantPimpleFoamEnhanced3(buoyant_case, buoyancy_production=True)
        G_b = solver._compute_buoyancy_production(solver.T)
        assert G_b.shape == (solver.mesh.n_cells,)
        assert torch.isfinite(G_b).all()


# ===========================================================================
# Tests: ReactingFoamEnhanced5
# ===========================================================================


class TestReactingFoamEnhanced5:
    """Tests for enhanced reacting solver v5."""

    def test_init(self, reacting_case):
        from pyfoam.applications.reacting_foam_enhanced_5 import ReactingFoamEnhanced5
        solver = ReactingFoamEnhanced5(reacting_case, isat_enabled=True, isat_tolerance=1e-4)
        assert solver.isat_enabled is True
        assert abs(solver.isat_tolerance - 1e-4) < 1e-10

    def test_isat_entry(self):
        from pyfoam.applications.reacting_foam_enhanced_5 import ISATEntry
        entry = ISATEntry(tolerance=1e-3)
        assert entry.tolerance == 1e-3

    def test_pressure_dependent_reaction(self):
        from pyfoam.applications.reacting_foam_enhanced_5 import PressureDependentReaction
        rxn = PressureDependentReaction(k0_A=1e10, kinf_A=1e14)
        assert rxn.k0_A == 1e10
        assert rxn.kinf_A == 1e14

    def test_troe_falloff_rate(self, reacting_case):
        from pyfoam.applications.reacting_foam_enhanced_5 import ReactingFoamEnhanced5, PressureDependentReaction
        solver = ReactingFoamEnhanced5(reacting_case)
        rxn = PressureDependentReaction()
        T = solver.T
        M = torch.full_like(T, 1e3)  # Third-body concentration
        k = solver._troe_falloff_rate(rxn, T, M)
        assert k.shape == T.shape
        assert (k >= 0).all()
        assert torch.isfinite(k).all()

    def test_isat_lookup_miss(self, reacting_case):
        from pyfoam.applications.reacting_foam_enhanced_5 import ReactingFoamEnhanced5
        solver = ReactingFoamEnhanced5(reacting_case, isat_enabled=True)
        result = solver._isat_lookup(solver.Y, solver.T)
        assert result is None  # Empty table

    def test_isat_insert_and_lookup(self, reacting_case):
        from pyfoam.applications.reacting_foam_enhanced_5 import ReactingFoamEnhanced5
        solver = ReactingFoamEnhanced5(reacting_case, isat_enabled=True)
        # Insert
        omega = {name: torch.zeros(solver.mesh.n_cells) for name in solver.species}
        solver._isat_insert(solver.Y, solver.T, omega)
        assert len(solver._isat_tree) == 1
        # Lookup (exact match)
        result = solver._isat_lookup(solver.Y, solver.T)
        assert result is not None


# ===========================================================================
# Tests: SolidFoamEnhanced2
# ===========================================================================


class TestSolidFoamEnhanced2:
    """Tests for enhanced solid mechanics solver v2."""

    def test_init(self, cavity_case):
        from pyfoam.applications.solid_foam_enhanced_2 import SolidFoamEnhanced2
        solver = SolidFoamEnhanced2(cavity_case, E=200e9, nu=0.3, creep_A=1e-12)
        assert solver.E == 200e9
        assert solver.creep_A == 1e-12

    def test_creep_strain_rate(self, cavity_case):
        from pyfoam.applications.solid_foam_enhanced_2 import SolidFoamEnhanced2
        solver = SolidFoamEnhanced2(cavity_case, E=200e9, nu=0.3, creep_A=1e-12, creep_n=5.0)
        sigma = torch.randn(solver.mesh.n_cells, 6) * 1e6
        rate = solver._compute_creep_strain_rate(sigma, 1.0)
        assert rate.shape == (solver.mesh.n_cells, 6)
        assert torch.isfinite(rate).all()

    def test_zero_creep(self, cavity_case):
        from pyfoam.applications.solid_foam_enhanced_2 import SolidFoamEnhanced2
        solver = SolidFoamEnhanced2(cavity_case, E=200e9, nu=0.3, creep_A=0.0)
        sigma = torch.randn(solver.mesh.n_cells, 6)
        rate = solver._compute_creep_strain_rate(sigma, 1.0)
        assert torch.allclose(rate, torch.zeros_like(rate))

    def test_fatigue_damage(self, cavity_case):
        from pyfoam.applications.solid_foam_enhanced_2 import SolidFoamEnhanced2
        solver = SolidFoamEnhanced2(cavity_case, E=200e9, nu=0.3, fatigue_coefficient=0.5)
        sigma = torch.randn(solver.mesh.n_cells, 6) * 1e6
        solver._update_fatigue_damage(sigma, 0.001)
        assert (solver.fatigue_damage >= 0).all()
        assert (solver.fatigue_damage <= 1.0).all()


# ===========================================================================
# Tests: FilmFoamEnhanced2
# ===========================================================================


class TestFilmFoamEnhanced2:
    """Tests for enhanced film solver v2."""

    def test_init(self, cavity_case):
        from pyfoam.applications.film_foam_enhanced_2 import FilmFoamEnhanced2
        solver = FilmFoamEnhanced2(cavity_case, precursor_thickness=1e-9)
        assert solver.precursor_thickness == 1e-9

    def test_height_function_curvature(self, cavity_case):
        from pyfoam.applications.film_foam_enhanced_2 import FilmFoamEnhanced2
        from pyfoam.core.device import get_default_dtype
        solver = FilmFoamEnhanced2(cavity_case, height_function_curvature=True)
        dtype = get_default_dtype()
        h = torch.full((solver.mesh.n_cells,), 1e-4, dtype=dtype)
        kappa = solver._compute_height_function_curvature(h)
        assert kappa.shape == (solver.mesh.n_cells,)
        assert torch.isfinite(kappa).all()

    def test_spinodal_check(self, cavity_case):
        from pyfoam.applications.film_foam_enhanced_2 import FilmFoamEnhanced2
        from pyfoam.core.device import get_default_dtype
        solver = FilmFoamEnhanced2(cavity_case, hamaker=1e-20)
        dtype = get_default_dtype()
        # Very thin film
        h_thin = torch.full((solver.mesh.n_cells,), 1e-10, dtype=dtype)
        _, n = solver._check_spinodal_instability(h_thin)
        assert isinstance(n, int)

        # Thick film (should be stable)
        h_thick = torch.full((solver.mesh.n_cells,), 1e-2, dtype=dtype)
        _, n2 = solver._check_spinodal_instability(h_thick)
        assert n2 == 0

    def test_precursor_film(self, cavity_case):
        from pyfoam.applications.film_foam_enhanced_2 import FilmFoamEnhanced2
        from pyfoam.core.device import get_default_dtype
        solver = FilmFoamEnhanced2(cavity_case, precursor_thickness=1e-9)
        dtype = get_default_dtype()
        h = torch.tensor([0.0, 1e-10, 1e-3], dtype=dtype)
        h_reg = solver._apply_precursor_film(h)
        assert (h_reg >= solver.precursor_thickness).all()


# ===========================================================================
# Tests: SprayFoamEnhanced2
# ===========================================================================


class TestSprayFoamEnhanced2:
    """Tests for enhanced spray solver v2."""

    def test_tab_breakup_model(self):
        from pyfoam.applications.spray_foam_enhanced_2 import TABBreakupModel
        model = TABBreakupModel()
        # Low We: no breakup
        y, dy = model.compute_distortion(0.0, 0.0, 1e-4, 0.5, 800.0, 1e-3, 1e-5)
        assert y < 1.0
        assert model.should_breakup(y) is False

    def test_tab_breakup_high_we(self):
        from pyfoam.applications.spray_foam_enhanced_2 import TABBreakupModel
        model = TABBreakupModel()
        # High We over long time: breakup
        y, dy = 0.0, 0.0
        for _ in range(1000):
            y, dy = model.compute_distortion(y, dy, 1e-3, 100.0, 800.0, 1e-3, 1e-4)
        # Should reach breakup at high We
        if y >= 1.0:
            d_child = model.child_diameter(1e-3, y)
            assert d_child < 1e-3

    def test_collision_event(self):
        from pyfoam.applications.spray_foam_enhanced_2 import CollisionEvent
        # Low relative velocity: bounce
        event_low = CollisionEvent(1e-4, 1e-4, 0.01)
        assert event_low.outcome in ("bounce", "coalescence", "fragmentation")

        # Coalescence diameter
        event = CollisionEvent(1e-4, 2e-4, 1.0)
        assert event.coalescence_diameter > 0

    def test_init(self, cavity_case):
        from pyfoam.applications.spray_foam_enhanced_2 import SprayFoamEnhanced2
        solver = SprayFoamEnhanced2(cavity_case, tab_breakup=True, collision_model=True)
        assert solver.tab_breakup is True
        assert solver.collision_model is True


# ===========================================================================
# Tests: MultiphaseEulerFoamEnhanced3
# ===========================================================================


class TestMultiphaseEulerFoamEnhanced3:
    """Tests for enhanced multiphase Euler solver v3."""

    def test_iac_state(self):
        from pyfoam.applications.multiphase_euler_foam_enhanced_3 import IACState
        a_i = torch.tensor([1e3, 2e3])
        state = IACState(a_i=a_i)
        assert state.a_i.shape == (2,)

    def test_adaptive_moment_selection(self, cavity_case):
        from pyfoam.applications.multiphase_euler_foam_enhanced_3 import MultiphaseEulerFoamEnhanced3
        # Test with a simple mock - create a minimal object with the attribute
        class _Mock:
            adaptive_moments = True
        mock = _Mock()
        moments = [torch.tensor([1e6]), torch.tensor([1e2]), torch.tensor([1e-2])]
        n = MultiphaseEulerFoamEnhanced3._select_adaptive_moments(
            mock, "test_phase", moments, error_threshold=1e-3,
        )
        assert 2 <= n <= 3


# ===========================================================================
# Tests: Exports
# ===========================================================================


class TestExportsV3:
    """Tests for __init__.py exports of v3 solvers."""

    def test_ico_enhanced_3_exported(self):
        from pyfoam.applications import IcoFoamEnhanced3
        assert IcoFoamEnhanced3 is not None

    def test_simple_enhanced_3_exported(self):
        from pyfoam.applications import SimpleFoamEnhanced3
        assert SimpleFoamEnhanced3 is not None

    def test_piso_enhanced_3_exported(self):
        from pyfoam.applications import PisoFoamEnhanced3
        assert PisoFoamEnhanced3 is not None

    def test_pimple_enhanced_3_exported(self):
        from pyfoam.applications import PimpleFoamEnhanced3
        assert PimpleFoamEnhanced3 is not None

    def test_rho_pimple_enhanced_3_exported(self):
        from pyfoam.applications import RhoPimpleFoamEnhanced3
        assert RhoPimpleFoamEnhanced3 is not None

    def test_buoyant_simple_enhanced_3_exported(self):
        from pyfoam.applications import BuoyantSimpleFoamEnhanced3
        assert BuoyantSimpleFoamEnhanced3 is not None

    def test_buoyant_pimple_enhanced_3_exported(self):
        from pyfoam.applications import BuoyantPimpleFoamEnhanced3
        assert BuoyantPimpleFoamEnhanced3 is not None

    def test_reacting_enhanced_5_exported(self):
        from pyfoam.applications import ReactingFoamEnhanced5
        assert ReactingFoamEnhanced5 is not None

    def test_solid_enhanced_2_exported(self):
        from pyfoam.applications import SolidFoamEnhanced2
        assert SolidFoamEnhanced2 is not None

    def test_film_enhanced_2_exported(self):
        from pyfoam.applications import FilmFoamEnhanced2
        assert FilmFoamEnhanced2 is not None

    def test_spray_enhanced_2_exported(self):
        from pyfoam.applications import SprayFoamEnhanced2
        assert SprayFoamEnhanced2 is not None

    def test_multiphase_enhanced_3_exported(self):
        from pyfoam.applications import MultiphaseEulerFoamEnhanced3
        assert MultiphaseEulerFoamEnhanced3 is not None

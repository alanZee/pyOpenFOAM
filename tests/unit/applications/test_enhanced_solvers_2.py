"""
Unit tests for enhanced solver variants v2.

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
# Tests: IcoFoamEnhanced2
# ===========================================================================


class TestIcoFoamEnhanced2:
    """Tests for enhanced ICO solver v2."""

    def test_init(self, cavity_case):
        from pyfoam.applications.ico_foam_enhanced_2 import IcoFoamEnhanced2
        solver = IcoFoamEnhanced2(cavity_case, use_bdf2=True, cfl_safety=0.8)
        assert solver.use_bdf2 is True
        assert abs(solver.cfl_safety - 0.8) < 1e-10

    def test_init_defaults(self, cavity_case):
        from pyfoam.applications.ico_foam_enhanced_2 import IcoFoamEnhanced2
        solver = IcoFoamEnhanced2(cavity_case)
        assert solver.use_bdf2 is False
        assert solver.adaptive_dt is True

    def test_run_completes(self, cavity_case):
        from pyfoam.applications.ico_foam_enhanced_2 import IcoFoamEnhanced2
        solver = IcoFoamEnhanced2(cavity_case, adaptive_dt=False)
        conv = solver.run()
        assert conv is not None

    def test_finite_values(self, cavity_case):
        from pyfoam.applications.ico_foam_enhanced_2 import IcoFoamEnhanced2
        solver = IcoFoamEnhanced2(cavity_case, adaptive_dt=False)
        solver.run()
        assert torch.isfinite(solver.U).all()
        assert torch.isfinite(solver.p).all()

    def test_multi_stage_cfl(self, cavity_case):
        from pyfoam.applications.ico_foam_enhanced_2 import IcoFoamEnhanced2
        solver = IcoFoamEnhanced2(cavity_case)
        Co = solver._compute_max_courant_multi_stage()
        assert Co >= 0

    def test_bdf2_rhs(self, cavity_case):
        from pyfoam.applications.ico_foam_enhanced_2 import IcoFoamEnhanced2
        solver = IcoFoamEnhanced2(cavity_case)
        U_n = solver.U.clone()
        U_nm1 = solver.U.clone() * 0.9
        rhs = solver._compute_bdf2_rhs(U_n, U_nm1)
        assert rhs.shape == solver.U.shape


# ===========================================================================
# Tests: SimpleFoamEnhanced2
# ===========================================================================


class TestSimpleFoamEnhanced2:
    """Tests for enhanced SIMPLE solver v2."""

    def test_init(self, cavity_case):
        from pyfoam.applications.simple_foam_enhanced_2 import SimpleFoamEnhanced2
        solver = SimpleFoamEnhanced2(cavity_case, residual_smoothing_coeff=0.3)
        assert abs(solver.residual_smoothing_coeff - 0.3) < 1e-10
        assert solver.auto_switch is True

    def test_run_completes(self, cavity_case):
        from pyfoam.applications.simple_foam_enhanced_2 import SimpleFoamEnhanced2
        solver = SimpleFoamEnhanced2(cavity_case)
        conv = solver.run()
        assert conv is not None

    def test_finite_values(self, cavity_case):
        from pyfoam.applications.simple_foam_enhanced_2 import SimpleFoamEnhanced2
        solver = SimpleFoamEnhanced2(cavity_case)
        solver.run()
        assert torch.isfinite(solver.U).all()
        assert torch.isfinite(solver.p).all()

    def test_pressure_smoothing(self, cavity_case):
        from pyfoam.applications.simple_foam_enhanced_2 import SimpleFoamEnhanced2
        solver = SimpleFoamEnhanced2(cavity_case, residual_smoothing_coeff=0.2)
        p_corr = torch.randn(solver.mesh.n_cells)
        p_smooth = solver._smooth_pressure_correction(p_corr)
        assert p_smooth.shape == p_corr.shape
        assert torch.isfinite(p_smooth).all()

    def test_algorithm_switch(self, cavity_case):
        from pyfoam.applications.simple_foam_enhanced_2 import SimpleFoamEnhanced2
        solver = SimpleFoamEnhanced2(cavity_case, auto_switch=True, divergence_threshold=5.0)
        solver._prev_residual_U = 0.001
        solver._check_and_switch_algorithm(0.01)  # 10x increase
        assert solver._using_simplec is False


# ===========================================================================
# Tests: PisoFoamEnhanced2
# ===========================================================================


class TestPisoFoamEnhanced2:
    """Tests for enhanced PISO solver v2."""

    def test_init(self, cavity_case):
        from pyfoam.applications.piso_foam_enhanced_2 import PisoFoamEnhanced2
        solver = PisoFoamEnhanced2(cavity_case, max_piso_correctors=4)
        assert solver.max_piso_correctors == 4

    def test_run_completes(self, cavity_case):
        from pyfoam.applications.piso_foam_enhanced_2 import PisoFoamEnhanced2
        solver = PisoFoamEnhanced2(cavity_case, max_courant=10.0)
        conv = solver.run()
        assert conv is not None

    def test_finite_values(self, cavity_case):
        from pyfoam.applications.piso_foam_enhanced_2 import PisoFoamEnhanced2
        solver = PisoFoamEnhanced2(cavity_case, max_courant=10.0)
        solver.run()
        assert torch.isfinite(solver.U).all()
        assert torch.isfinite(solver.p).all()

    def test_consistent_flux_correction(self, cavity_case):
        from pyfoam.applications.piso_foam_enhanced_2 import PisoFoamEnhanced2
        solver = PisoFoamEnhanced2(cavity_case)
        phi = torch.zeros(solver.mesh.n_faces)
        phi_new = solver._correct_flux_consistent(solver.U, phi)
        assert phi_new.shape == phi.shape

    def test_ho_rhie_chow(self, cavity_case):
        from pyfoam.applications.piso_foam_enhanced_2 import PisoFoamEnhanced2
        solver = PisoFoamEnhanced2(cavity_case)
        A_p = torch.ones(solver.mesh.n_cells)
        U_corr = solver._rhie_chow_velocity_correction_ho(
            solver.U, solver.p, solver.p_old, A_p,
        )
        assert U_corr.shape == solver.U.shape
        assert torch.isfinite(U_corr).all()


# ===========================================================================
# Tests: PimpleFoamEnhanced2
# ===========================================================================


class TestPimpleFoamEnhanced2:
    """Tests for enhanced PIMPLE solver v2."""

    def test_init(self, cavity_case):
        from pyfoam.applications.pimple_foam_enhanced_2 import PimpleFoamEnhanced2
        solver = PimpleFoamEnhanced2(cavity_case, sor_weight=0.5)
        assert abs(solver.sor_weight - 0.5) < 1e-10

    def test_run_completes(self, cavity_case):
        from pyfoam.applications.pimple_foam_enhanced_2 import PimpleFoamEnhanced2
        solver = PimpleFoamEnhanced2(cavity_case)
        conv = solver.run()
        assert conv is not None

    def test_finite_values(self, cavity_case):
        from pyfoam.applications.pimple_foam_enhanced_2 import PimpleFoamEnhanced2
        solver = PimpleFoamEnhanced2(cavity_case)
        solver.run()
        assert torch.isfinite(solver.U).all()
        assert torch.isfinite(solver.p).all()

    def test_sor_aitken_relaxation(self, cavity_case):
        from pyfoam.applications.pimple_foam_enhanced_2 import PimpleFoamEnhanced2
        solver = PimpleFoamEnhanced2(cavity_case, sor_weight=0.3)
        field = solver.U.clone()
        field_old = solver.U.clone()
        result, alpha = solver._sor_aitken_relaxation(
            field, field_old, field_old, 0.7,
        )
        assert result.shape == solver.U.shape
        assert 0.05 <= alpha <= 1.0

    def test_residual_prediction(self, cavity_case):
        from pyfoam.applications.pimple_foam_enhanced_2 import PimpleFoamEnhanced2
        solver = PimpleFoamEnhanced2(cavity_case)
        # Converging residual history
        will_converge, iters = solver._predict_outer_convergence(
            [1.0, 0.5, 0.25], 1e-4,
        )
        assert isinstance(will_converge, bool)
        assert isinstance(iters, float)


# ===========================================================================
# Tests: RhoPimpleFoamEnhanced2
# ===========================================================================


class TestRhoPimpleFoamEnhanced2:
    """Tests for enhanced compressible PIMPLE solver v2."""

    def test_init(self, compressible_case):
        from pyfoam.applications.rho_pimple_foam_enhanced_2 import RhoPimpleFoamEnhanced2
        solver = RhoPimpleFoamEnhanced2(compressible_case, n_energy_correctors=3)
        assert solver.n_energy_correctors == 3
        assert solver.density_correction is True

    def test_run_completes(self, compressible_case):
        from pyfoam.applications.rho_pimple_foam_enhanced_2 import RhoPimpleFoamEnhanced2
        solver = RhoPimpleFoamEnhanced2(compressible_case)
        conv = solver.run()
        assert conv is not None

    def test_finite_values(self, compressible_case):
        from pyfoam.applications.rho_pimple_foam_enhanced_2 import RhoPimpleFoamEnhanced2
        solver = RhoPimpleFoamEnhanced2(compressible_case)
        solver.run()
        assert torch.isfinite(solver.U).all()
        assert torch.isfinite(solver.p).all()
        assert torch.isfinite(solver.T).all()

    def test_mach_aware_T_relaxation(self, compressible_case):
        from pyfoam.applications.rho_pimple_foam_enhanced_2 import RhoPimpleFoamEnhanced2
        solver = RhoPimpleFoamEnhanced2(compressible_case)
        T = solver.T.clone()
        T_old = solver.T.clone()
        Ma = torch.full((solver.mesh.n_cells,), 0.1)
        T_relaxed = solver._mach_aware_T_relaxation(T, T_old, Ma)
        assert T_relaxed.shape == T.shape
        assert torch.isfinite(T_relaxed).all()

    def test_density_correction(self, compressible_case):
        from pyfoam.applications.rho_pimple_foam_enhanced_2 import RhoPimpleFoamEnhanced2
        solver = RhoPimpleFoamEnhanced2(compressible_case)
        rho_new = solver._density_correction_step(solver.rho, solver.p, solver.T)
        assert rho_new.shape == solver.rho.shape
        assert torch.isfinite(rho_new).all()


# ===========================================================================
# Tests: BuoyantSimpleFoamEnhanced2
# ===========================================================================


class TestBuoyantSimpleFoamEnhanced2:
    """Tests for enhanced buoyant SIMPLE solver v2."""

    def test_init(self, buoyant_case):
        from pyfoam.applications.buoyant_simple_foam_enhanced_2 import BuoyantSimpleFoamEnhanced2
        solver = BuoyantSimpleFoamEnhanced2(buoyant_case, implicit_buoyancy=True)
        assert solver.implicit_buoyancy is True

    def test_gradient_richardson(self, buoyant_case):
        from pyfoam.applications.buoyant_simple_foam_enhanced_2 import BuoyantSimpleFoamEnhanced2
        solver = BuoyantSimpleFoamEnhanced2(buoyant_case)
        Ri_field = solver._compute_gradient_richardson_field(solver.U, solver.T)
        assert Ri_field.shape == (solver.mesh.n_cells,)
        assert (Ri_field >= 0).all()

    def test_implicit_buoyancy_source(self, buoyant_case):
        from pyfoam.applications.buoyant_simple_foam_enhanced_2 import BuoyantSimpleFoamEnhanced2
        solver = BuoyantSimpleFoamEnhanced2(buoyant_case)
        source, diag = solver._compute_implicit_buoyancy_source(
            solver.T, solver.U, 1.225, solver.delta_t,
        )
        assert source.shape == (solver.mesh.n_cells, 3)
        assert diag >= 0

    def test_hydrostatic_pressure(self, buoyant_case):
        from pyfoam.applications.buoyant_simple_foam_enhanced_2 import BuoyantSimpleFoamEnhanced2
        solver = BuoyantSimpleFoamEnhanced2(buoyant_case)
        p_hydro = solver._compute_hydrostatic_pressure(solver.T)
        assert p_hydro.shape == (solver.mesh.n_cells,)
        assert torch.isfinite(p_hydro).all()


# ===========================================================================
# Tests: BuoyantPimpleFoamEnhanced2
# ===========================================================================


class TestBuoyantPimpleFoamEnhanced2:
    """Tests for enhanced buoyant PIMPLE solver v2."""

    def test_init(self, buoyant_case):
        from pyfoam.applications.buoyant_pimple_foam_enhanced_2 import BuoyantPimpleFoamEnhanced2
        solver = BuoyantPimpleFoamEnhanced2(buoyant_case, T_min=250, T_max=4000)
        assert solver.T_min == 250
        assert solver.T_max == 4000

    def test_brunt_vaisala(self, buoyant_case):
        from pyfoam.applications.buoyant_pimple_foam_enhanced_2 import BuoyantPimpleFoamEnhanced2
        solver = BuoyantPimpleFoamEnhanced2(buoyant_case)
        N = solver._compute_brunt_vaisala_frequency(solver.T)
        assert N >= 0

    def test_temperature_limiting(self, buoyant_case):
        from pyfoam.applications.buoyant_pimple_foam_enhanced_2 import BuoyantPimpleFoamEnhanced2
        solver = BuoyantPimpleFoamEnhanced2(buoyant_case, T_min=250, T_max=4000)
        T = torch.tensor([100.0, 300.0, 6000.0])
        T_limited = solver._limit_temperature(T)
        assert T_limited[0] == 250
        assert T_limited[1] == 300
        assert T_limited[2] == 4000

    def test_stratification_time_step(self, buoyant_case):
        from pyfoam.applications.buoyant_pimple_foam_enhanced_2 import BuoyantPimpleFoamEnhanced2
        solver = BuoyantPimpleFoamEnhanced2(buoyant_case)
        # No stratification
        dt_no = solver._adapt_time_step_stratification(0.0)
        assert dt_no > 0
        # With stratification
        dt_yes = solver._adapt_time_step_stratification(10.0)
        assert dt_yes > 0


# ===========================================================================
# Tests: ReactingFoamEnhanced4
# ===========================================================================


class TestReactingFoamEnhanced4:
    """Tests for enhanced reacting solver v4."""

    def test_init(self, reacting_case):
        from pyfoam.applications.reacting_foam_enhanced_4 import ReactingFoamEnhanced4
        solver = ReactingFoamEnhanced4(reacting_case, partial_equilibrium_threshold=0.9)
        assert solver.partial_equilibrium_threshold == 0.9

    def test_reaction_graph(self, reacting_case):
        from pyfoam.applications.reacting_foam_enhanced_4 import ReactingFoamEnhanced4
        solver = ReactingFoamEnhanced4(reacting_case)
        assert len(solver.reaction_graph) >= 0
        # All nodes should have an order
        for node in solver.reaction_graph:
            assert node.order >= 0

    def test_mass_renormalisation(self, reacting_case):
        from pyfoam.applications.reacting_foam_enhanced_4 import ReactingFoamEnhanced4
        solver = ReactingFoamEnhanced4(reacting_case)
        Y_test = {name: y.clone() for name, y in solver.Y.items()}
        Y_norm = solver._renormalise_mass_fractions(Y_test)
        Y_sum = sum(Y_norm.values())
        assert torch.allclose(Y_sum, torch.ones_like(Y_sum), atol=1e-6)

    def test_species_adaptive(self, reacting_case):
        from pyfoam.applications.reacting_foam_enhanced_4 import ReactingFoamEnhanced4
        solver = ReactingFoamEnhanced4(reacting_case)
        Y_new = solver._advance_species_adaptive(solver.Y, solver.T, solver.delta_t)
        for name in solver.species:
            assert torch.isfinite(Y_new[name]).all()
            assert (Y_new[name] >= 0).all()


# ===========================================================================
# Tests: SolidFoamEnhanced
# ===========================================================================


class TestSolidFoamEnhanced:
    """Tests for enhanced solid mechanics solver."""

    def test_init(self, cavity_case):
        from pyfoam.applications.solid_foam_enhanced import SolidFoamEnhanced
        solver = SolidFoamEnhanced(cavity_case, E=200e9, nu=0.3)
        assert solver.E == 200e9
        assert solver.max_sub_iterations >= 1

    def test_stress_smoothing(self, cavity_case):
        from pyfoam.applications.solid_foam_enhanced import SolidFoamEnhanced
        solver = SolidFoamEnhanced(cavity_case, stress_smoothing_coeff=0.3)
        sigma = torch.randn(solver.mesh.n_cells, 6)
        sigma_smooth = solver._smooth_stress(sigma)
        assert sigma_smooth.shape == sigma.shape
        assert torch.isfinite(sigma_smooth).all()


# ===========================================================================
# Tests: FilmFoamEnhanced
# ===========================================================================


class TestFilmFoamEnhanced:
    """Tests for enhanced film solver."""

    def test_init(self, cavity_case):
        from pyfoam.applications.film_foam_enhanced import FilmFoamEnhanced
        solver = FilmFoamEnhanced(cavity_case, hamaker=1e-20, adaptive_dt=True)
        assert solver.hamaker == 1e-20
        assert solver.adaptive_dt is True

    def test_disjoining_pressure(self, cavity_case):
        from pyfoam.applications.film_foam_enhanced import FilmFoamEnhanced
        solver = FilmFoamEnhanced(cavity_case)
        from pyfoam.core.device import get_default_dtype
        dtype = get_default_dtype()
        h = torch.full((solver.mesh.n_cells,), 1e-4, dtype=dtype)
        grad_Pi = solver._compute_disjoining_pressure_gradient(h)
        assert grad_Pi.shape == (solver.mesh.n_cells, 3)
        assert torch.isfinite(grad_Pi).all()

    def test_capillary_dt(self, cavity_case):
        from pyfoam.applications.film_foam_enhanced import FilmFoamEnhanced
        from pyfoam.core.device import get_default_dtype
        solver = FilmFoamEnhanced(cavity_case)
        dtype = get_default_dtype()
        h = torch.full((solver.mesh.n_cells,), 1e-3, dtype=dtype)
        dt = solver._compute_capillary_dt(h)
        assert dt > 0


# ===========================================================================
# Tests: SprayFoamEnhanced
# ===========================================================================


class TestSprayFoamEnhanced:
    """Tests for enhanced spray solver."""

    def test_reitz_diwakar_breakup(self):
        from pyfoam.applications.spray_foam_enhanced import ReitzDiwakarBreakup
        model = ReitzDiwakarBreakup()
        # Below We_bag: no breakup
        d, tau = model.compute_breakup(1e-4, 1.0, 800.0)
        assert d == 1e-4
        # Above We_bag: breakup
        d2, tau2 = model.compute_breakup(1e-4, 10.0, 800.0)
        assert d2 <= 1e-4


# ===========================================================================
# Tests: MultiphaseEulerFoamEnhanced2
# ===========================================================================


class TestMultiphaseEulerFoamEnhanced2:
    """Tests for enhanced multiphase Euler solver v2."""

    def test_qmom_abscissas(self, cavity_case):
        from pyfoam.applications.multiphase_euler_foam_enhanced_2 import QuadratureMoment
        # Basic dataclass test
        qm = QuadratureMoment(n_moments=4)
        assert qm.n_moments == 4

    def test_saffman_turner_rate(self, cavity_case):
        from pyfoam.applications.multiphase_euler_foam_enhanced_2 import MultiphaseEulerFoamEnhanced2
        # Test the static method (call on class since we can't easily instantiate)
        rate = MultiphaseEulerFoamEnhanced2._saffman_turner_coalescence_rate(
            None, 1e-4, 1e-4, 0.01, 1e-5,
        )
        assert rate >= 0

    def test_zero_dissipation(self, cavity_case):
        from pyfoam.applications.multiphase_euler_foam_enhanced_2 import MultiphaseEulerFoamEnhanced2
        rate = MultiphaseEulerFoamEnhanced2._saffman_turner_coalescence_rate(
            None, 1e-4, 1e-4, 0.0, 1e-5,
        )
        assert rate == 0.0


# ===========================================================================
# Tests: Exports
# ===========================================================================


class TestExportsV2:
    """Tests for __init__.py exports of v2 solvers."""

    def test_ico_enhanced_2_exported(self):
        from pyfoam.applications import IcoFoamEnhanced2
        assert IcoFoamEnhanced2 is not None

    def test_simple_enhanced_2_exported(self):
        from pyfoam.applications import SimpleFoamEnhanced2
        assert SimpleFoamEnhanced2 is not None

    def test_piso_enhanced_2_exported(self):
        from pyfoam.applications import PisoFoamEnhanced2
        assert PisoFoamEnhanced2 is not None

    def test_pimple_enhanced_2_exported(self):
        from pyfoam.applications import PimpleFoamEnhanced2
        assert PimpleFoamEnhanced2 is not None

    def test_rho_pimple_enhanced_2_exported(self):
        from pyfoam.applications import RhoPimpleFoamEnhanced2
        assert RhoPimpleFoamEnhanced2 is not None

    def test_buoyant_simple_enhanced_2_exported(self):
        from pyfoam.applications import BuoyantSimpleFoamEnhanced2
        assert BuoyantSimpleFoamEnhanced2 is not None

    def test_buoyant_pimple_enhanced_2_exported(self):
        from pyfoam.applications import BuoyantPimpleFoamEnhanced2
        assert BuoyantPimpleFoamEnhanced2 is not None

    def test_reacting_enhanced_4_exported(self):
        from pyfoam.applications import ReactingFoamEnhanced4
        assert ReactingFoamEnhanced4 is not None

    def test_solid_enhanced_exported(self):
        from pyfoam.applications import SolidFoamEnhanced
        assert SolidFoamEnhanced is not None

    def test_film_enhanced_exported(self):
        from pyfoam.applications import FilmFoamEnhanced
        assert FilmFoamEnhanced is not None

    def test_spray_enhanced_exported(self):
        from pyfoam.applications import SprayFoamEnhanced
        assert SprayFoamEnhanced is not None

    def test_multiphase_enhanced_2_exported(self):
        from pyfoam.applications import MultiphaseEulerFoamEnhanced2
        assert MultiphaseEulerFoamEnhanced2 is not None

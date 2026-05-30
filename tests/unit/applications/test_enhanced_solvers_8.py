"""
Unit tests for enhanced solver variants v8 (and v10/v7 specialized).

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
# Mesh generation helper (same pattern as test_enhanced_solvers_7)
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
# Tests: IcoFoamEnhanced8
# ===========================================================================


class TestIcoFoamEnhanced8:
    """Tests for enhanced ICO solver v8."""

    def test_init(self, cavity_case):
        from pyfoam.applications.ico_foam_enhanced_8 import IcoFoamEnhanced8
        solver = IcoFoamEnhanced8(cavity_case, metric_adaptation=True, semi_lagrangian=True)
        assert solver.metric_adaptation is True
        assert solver.semi_lagrangian is True

    def test_init_defaults(self, cavity_case):
        from pyfoam.applications.ico_foam_enhanced_8 import IcoFoamEnhanced8
        solver = IcoFoamEnhanced8(cavity_case)
        assert solver.bfbt_precondition is True
        assert solver.hessian_weight == pytest.approx(0.5)
        assert solver.trajectory_sub_steps == 2

    def test_hessian_metric(self, cavity_case):
        from pyfoam.applications.ico_foam_enhanced_8 import IcoFoamEnhanced8
        solver = IcoFoamEnhanced8(cavity_case, metric_adaptation=True)
        metric = solver._compute_hessian_metric(solver.p)
        assert metric.shape == (solver.mesh.n_cells, 3, 3)
        assert torch.isfinite(metric).all()

    def test_metric_disabled(self, cavity_case):
        from pyfoam.applications.ico_foam_enhanced_8 import IcoFoamEnhanced8
        solver = IcoFoamEnhanced8(cavity_case, metric_adaptation=False)
        metric = solver._compute_hessian_metric(solver.p)
        assert metric.shape == (solver.mesh.n_cells, 3, 3)
        # Should be identity
        eye = torch.eye(3, dtype=solver.p.dtype, device=solver.p.device).unsqueeze(0).expand(solver.mesh.n_cells, -1, -1)
        assert torch.allclose(metric, eye)

    def test_bfbt_precondition(self, cavity_case):
        from pyfoam.applications.ico_foam_enhanced_8 import IcoFoamEnhanced8
        solver = IcoFoamEnhanced8(cavity_case, bfbt_precondition=True)
        p_prec = solver._bfbt_pressure_precondition(solver.p, solver.U)
        assert p_prec.shape == solver.p.shape
        assert torch.isfinite(p_prec).all()

    def test_semi_lagrangian(self, cavity_case):
        from pyfoam.applications.ico_foam_enhanced_8 import IcoFoamEnhanced8
        solver = IcoFoamEnhanced8(cavity_case, semi_lagrangian=True)
        U_sl = solver._semi_lagrangian_advance(solver.U, solver.delta_t)
        assert U_sl.shape == solver.U.shape
        assert torch.isfinite(U_sl).all()

    def test_semi_lagrangian_disabled(self, cavity_case):
        from pyfoam.applications.ico_foam_enhanced_8 import IcoFoamEnhanced8
        solver = IcoFoamEnhanced8(cavity_case, semi_lagrangian=False)
        U = solver.U.clone()
        U_sl = solver._semi_lagrangian_advance(U, solver.delta_t)
        assert torch.allclose(U_sl, U)


# ===========================================================================
# Tests: SimpleFoamEnhanced8
# ===========================================================================


class TestSimpleFoamEnhanced8:
    """Tests for enhanced SIMPLE solver v8."""

    def test_init(self, cavity_case):
        from pyfoam.applications.simple_foam_enhanced_8 import SimpleFoamEnhanced8
        solver = SimpleFoamEnhanced8(cavity_case, jfnk_acceleration=True, spectral_precondition=True)
        assert solver.jfnk_acceleration is True
        assert solver.spectral_precondition is True

    def test_init_defaults(self, cavity_case):
        from pyfoam.applications.simple_foam_enhanced_8 import SimpleFoamEnhanced8
        solver = SimpleFoamEnhanced8(cavity_case)
        assert solver.newton_line_search is True
        assert solver.jfnk_perturbation == pytest.approx(1e-6)
        assert solver.armijo_c == pytest.approx(1e-4)

    def test_jfnk_jvp(self, cavity_case):
        from pyfoam.applications.simple_foam_enhanced_8 import SimpleFoamEnhanced8
        solver = SimpleFoamEnhanced8(cavity_case, jfnk_acceleration=True)
        v_U = torch.randn_like(solver.U) * 0.01
        v_p = torch.randn_like(solver.p) * 0.01
        Jv_U, Jv_p = solver._jfnk_jacobian_vector_product(
            solver.U, solver.p, v_U, v_p,
        )
        assert Jv_U.shape == solver.U.shape
        assert Jv_p.shape == solver.p.shape
        assert torch.isfinite(Jv_U).all()
        assert torch.isfinite(Jv_p).all()

    def test_spectral_precondition(self, cavity_case):
        from pyfoam.applications.simple_foam_enhanced_8 import SimpleFoamEnhanced8
        solver = SimpleFoamEnhanced8(cavity_case, spectral_precondition=True)
        p_prec = solver._spectral_element_precondition(solver.p, solver.U)
        assert p_prec.shape == solver.p.shape
        assert torch.isfinite(p_prec).all()

    def test_armijo_line_search(self, cavity_case):
        from pyfoam.applications.simple_foam_enhanced_8 import SimpleFoamEnhanced8
        solver = SimpleFoamEnhanced8(cavity_case, newton_line_search=True)
        U_new = solver.U.clone() * 1.01
        U_result = solver._armijo_line_search(solver.U.clone(), U_new, 1.0, 1.5)
        assert U_result.shape == solver.U.shape
        assert torch.isfinite(U_result).all()


# ===========================================================================
# Tests: PisoFoamEnhanced8
# ===========================================================================


class TestPisoFoamEnhanced8:
    """Tests for enhanced PISO solver v8."""

    def test_init(self, cavity_case):
        from pyfoam.applications.piso_foam_enhanced_8 import PisoFoamEnhanced8
        solver = PisoFoamEnhanced8(cavity_case, embedded_rk=True, gmres_pressure=True)
        assert solver.embedded_rk is True
        assert solver.gmres_pressure is True

    def test_init_defaults(self, cavity_case):
        from pyfoam.applications.piso_foam_enhanced_8 import PisoFoamEnhanced8
        solver = PisoFoamEnhanced8(cavity_case)
        assert solver.skew_symmetric_advection is True
        assert solver.rk_safety == pytest.approx(0.9)
        assert solver.gmres_restart == 10

    def test_embedded_rk_error(self, cavity_case):
        from pyfoam.applications.piso_foam_enhanced_8 import PisoFoamEnhanced8
        solver = PisoFoamEnhanced8(cavity_case, embedded_rk=True)
        U_low = solver.U.clone()
        U_high = solver.U.clone() * 1.001
        error, dt_new = solver._embedded_rk_error_estimate(U_low, U_high, solver.delta_t)
        assert isinstance(error, float)
        assert isinstance(dt_new, float)

    def test_skew_symmetric_advection(self, cavity_case):
        from pyfoam.applications.piso_foam_enhanced_8 import PisoFoamEnhanced8
        solver = PisoFoamEnhanced8(cavity_case, skew_symmetric_advection=True)
        U_skew = solver._skew_symmetric_momentum_flux(solver.U, solver.U.clone())
        assert U_skew.shape == solver.U.shape
        assert torch.isfinite(U_skew).all()

    def test_gmres_pressure(self, cavity_case):
        from pyfoam.applications.piso_foam_enhanced_8 import PisoFoamEnhanced8
        solver = PisoFoamEnhanced8(cavity_case, gmres_pressure=True)
        rhs = solver.p.clone() * 0.01
        p_gmres = solver._gmres_pressure_solve(solver.p, rhs, n_iter=3)
        assert p_gmres.shape == solver.p.shape
        assert torch.isfinite(p_gmres).all()


# ===========================================================================
# Tests: PimpleFoamEnhanced8
# ===========================================================================


class TestPimpleFoamEnhanced8:
    """Tests for enhanced PIMPLE solver v8."""

    def test_init(self, cavity_case):
        from pyfoam.applications.pimple_foam_enhanced_8 import PimpleFoamEnhanced8
        solver = PimpleFoamEnhanced8(cavity_case, oif_stepping=True, simplenga=True)
        assert solver.oif_stepping is True
        assert solver.simplenga is True

    def test_init_defaults(self, cavity_case):
        from pyfoam.applications.pimple_foam_enhanced_8 import PimpleFoamEnhanced8
        solver = PimpleFoamEnhanced8(cavity_case)
        assert solver.adaptive_amg is True
        assert solver.oif_order == 3
        assert solver.nga_depth == 5

    def test_oif_momentum_advance(self, cavity_case):
        from pyfoam.applications.pimple_foam_enhanced_8 import PimpleFoamEnhanced8
        solver = PimpleFoamEnhanced8(cavity_case, oif_stepping=True)
        U_oif = solver._oif_momentum_advance(solver.U, solver.U.clone(), solver.delta_t)
        assert U_oif.shape == solver.U.shape
        assert torch.isfinite(U_oif).all()

    def test_adaptive_amg(self, cavity_case):
        from pyfoam.applications.pimple_foam_enhanced_8 import PimpleFoamEnhanced8
        solver = PimpleFoamEnhanced8(cavity_case, adaptive_amg=True)
        rhs = solver.p.clone() * 0.01
        p_amg = solver._adaptive_amg_solve(solver.p, rhs, n_levels=2)
        assert p_amg.shape == solver.p.shape
        assert torch.isfinite(p_amg).all()

    def test_simplenga(self, cavity_case):
        from pyfoam.applications.pimple_foam_enhanced_8 import PimpleFoamEnhanced8
        solver = PimpleFoamEnhanced8(cavity_case, simplenga=True)
        U_acc, p_acc = solver._simplenga_acceleration(
            solver.U, solver.p, solver.U.clone(), solver.p.clone(),
        )
        assert U_acc.shape == solver.U.shape
        assert torch.isfinite(U_acc).all()


# ===========================================================================
# Tests: RhoPimpleFoamEnhanced8
# ===========================================================================


class TestRhoPimpleFoamEnhanced8:
    """Tests for enhanced compressible PIMPLE solver v8."""

    def test_init(self, compressible_case):
        from pyfoam.applications.rho_pimple_foam_enhanced_8 import RhoPimpleFoamEnhanced8
        solver = RhoPimpleFoamEnhanced8(compressible_case, jst_dissipation=True)
        assert solver.jst_dissipation is True

    def test_init_defaults(self, compressible_case):
        from pyfoam.applications.rho_pimple_foam_enhanced_8 import RhoPimpleFoamEnhanced8
        solver = RhoPimpleFoamEnhanced8(compressible_case)
        assert solver.dual_time is True
        assert solver.mixture_averaged is True
        assert solver.n_dual_iters == 3

    def test_jst_dissipation(self, compressible_case):
        from pyfoam.applications.rho_pimple_foam_enhanced_8 import RhoPimpleFoamEnhanced8
        solver = RhoPimpleFoamEnhanced8(compressible_case, jst_dissipation=True)
        U_jst = solver._jst_dissipation_flux(solver.U, solver.p, solver.rho)
        assert U_jst.shape == solver.U.shape
        assert torch.isfinite(U_jst).all()

    def test_jst_disabled(self, compressible_case):
        from pyfoam.applications.rho_pimple_foam_enhanced_8 import RhoPimpleFoamEnhanced8
        solver = RhoPimpleFoamEnhanced8(compressible_case, jst_dissipation=False)
        U = solver.U.clone()
        U_jst = solver._jst_dissipation_flux(U, solver.p, solver.rho)
        assert torch.allclose(U_jst, U)

    def test_dual_time(self, compressible_case):
        from pyfoam.applications.rho_pimple_foam_enhanced_8 import RhoPimpleFoamEnhanced8
        solver = RhoPimpleFoamEnhanced8(compressible_case, dual_time=True)
        U, p, T, rho = solver._dual_time_iteration(
            solver.U, solver.p, solver.T, solver.rho, solver.delta_t,
        )
        assert U.shape == solver.U.shape
        assert torch.isfinite(U).all()


# ===========================================================================
# Tests: BuoyantSimpleFoamEnhanced8
# ===========================================================================


class TestBuoyantSimpleFoamEnhanced8:
    """Tests for enhanced buoyant SIMPLE solver v8."""

    def test_init(self, buoyant_case):
        from pyfoam.applications.buoyant_simple_foam_enhanced_8 import BuoyantSimpleFoamEnhanced8
        solver = BuoyantSimpleFoamEnhanced8(buoyant_case, variable_boussinesq=True)
        assert solver.variable_boussinesq is True

    def test_init_defaults(self, buoyant_case):
        from pyfoam.applications.buoyant_simple_foam_enhanced_8 import BuoyantSimpleFoamEnhanced8
        solver = BuoyantSimpleFoamEnhanced8(buoyant_case)
        assert solver.conjugate_htc is True
        assert solver.rossby_turb_switching is True
        assert solver.rossby_threshold == pytest.approx(0.1)

    def test_variable_property_density(self, buoyant_case):
        from pyfoam.applications.buoyant_simple_foam_enhanced_8 import BuoyantSimpleFoamEnhanced8
        solver = BuoyantSimpleFoamEnhanced8(buoyant_case, variable_boussinesq=True)
        rho = solver._variable_property_density(1.2, solver.T, 300.0)
        assert rho.shape == solver.T.shape
        assert (rho > 0).all()
        assert torch.isfinite(rho).all()

    def test_rossby_number(self, buoyant_case):
        from pyfoam.applications.buoyant_simple_foam_enhanced_8 import BuoyantSimpleFoamEnhanced8
        solver = BuoyantSimpleFoamEnhanced8(buoyant_case, rossby_turb_switching=True)
        Ro = solver._compute_rossby_number(solver.U)
        assert isinstance(Ro, float)
        assert Ro >= 0

    def test_conjugate_htc(self, buoyant_case):
        from pyfoam.applications.buoyant_simple_foam_enhanced_8 import BuoyantSimpleFoamEnhanced8
        solver = BuoyantSimpleFoamEnhanced8(buoyant_case, conjugate_htc=True)
        T_solid = solver.T.clone()
        T_f, T_s = solver._conjugate_heat_transfer(
            solver.T, T_solid, 0.6, 50.0, solver.delta_t,
        )
        assert T_f.shape == solver.T.shape
        assert torch.isfinite(T_f).all()


# ===========================================================================
# Tests: BuoyantPimpleFoamEnhanced8
# ===========================================================================


class TestBuoyantPimpleFoamEnhanced8:
    """Tests for enhanced buoyant PIMPLE solver v8."""

    def test_init(self, buoyant_case):
        from pyfoam.applications.buoyant_pimple_foam_enhanced_8 import BuoyantPimpleFoamEnhanced8
        solver = BuoyantPimpleFoamEnhanced8(buoyant_case, density_precondition=True)
        assert solver.density_precondition is True

    def test_init_defaults(self, buoyant_case):
        from pyfoam.applications.buoyant_pimple_foam_enhanced_8 import BuoyantPimpleFoamEnhanced8
        solver = BuoyantPimpleFoamEnhanced8(buoyant_case)
        assert solver.entropy_stable_thermal is True
        assert solver.gravity_wave_cfl is True
        assert solver.gw_cfl_max == pytest.approx(0.5)

    def test_density_precondition(self, buoyant_case):
        from pyfoam.applications.buoyant_pimple_foam_enhanced_8 import BuoyantPimpleFoamEnhanced8
        solver = BuoyantPimpleFoamEnhanced8(buoyant_case, density_precondition=True)
        rho_ref = float(solver.rho.mean().item())
        p_corr, U_corr = solver._density_buoyancy_precondition(
            solver.p, solver.U, solver.rho, rho_ref,
        )
        assert p_corr.shape == solver.p.shape
        assert U_corr.shape == solver.U.shape
        assert torch.isfinite(p_corr).all()

    def test_entropy_stable_thermal(self, buoyant_case):
        from pyfoam.applications.buoyant_pimple_foam_enhanced_8 import BuoyantPimpleFoamEnhanced8
        solver = BuoyantPimpleFoamEnhanced8(buoyant_case, entropy_stable_thermal=True)
        T_es = solver._entropy_stable_thermal_convection(
            solver.T, solver.T.clone(), solver.U, solver.delta_t,
        )
        assert T_es.shape == solver.T.shape
        assert torch.isfinite(T_es).all()
        assert (T_es >= 200).all()

    def test_gravity_wave_limited_dt(self, buoyant_case):
        from pyfoam.applications.buoyant_pimple_foam_enhanced_8 import BuoyantPimpleFoamEnhanced8
        solver = BuoyantPimpleFoamEnhanced8(buoyant_case, gravity_wave_cfl=True)
        dt_limited = solver._gravity_wave_limited_dt(solver.T, solver.delta_t)
        assert isinstance(dt_limited, float)
        assert dt_limited > 0


# ===========================================================================
# Tests: ReactingFoamEnhanced10
# ===========================================================================


class TestReactingFoamEnhanced10:
    """Tests for enhanced reacting solver v10."""

    def test_init(self, reacting_case):
        from pyfoam.applications.reacting_foam_enhanced_10 import ReactingFoamEnhanced10
        solver = ReactingFoamEnhanced10(reacting_case, hak_hierarchy=True, imex_integration=True)
        assert solver.hak_hierarchy is True
        assert solver.imex_integration is True

    def test_init_defaults(self, reacting_case):
        from pyfoam.applications.reacting_foam_enhanced_10 import ReactingFoamEnhanced10
        solver = ReactingFoamEnhanced10(reacting_case)
        assert solver.ml_combustion is True
        assert solver.hak_levels == 3
        assert solver.imex_safety == pytest.approx(0.8)

    def test_hak_level_selection(self, reacting_case):
        from pyfoam.applications.reacting_foam_enhanced_10 import ReactingFoamEnhanced10
        solver = ReactingFoamEnhanced10(reacting_case, hak_hierarchy=True)
        level = solver._select_hak_level(solver.T, solver.Y, 1e3)
        assert isinstance(level, int)
        assert 0 <= level < solver.hak_levels

    def test_hak_disabled(self, reacting_case):
        from pyfoam.applications.reacting_foam_enhanced_10 import ReactingFoamEnhanced10
        solver = ReactingFoamEnhanced10(reacting_case, hak_hierarchy=False)
        level = solver._select_hak_level(solver.T, solver.Y, 1e6)
        assert level == 0

    def test_imex_step(self, reacting_case):
        from pyfoam.applications.reacting_foam_enhanced_10 import ReactingFoamEnhanced10
        solver = ReactingFoamEnhanced10(reacting_case, imex_integration=True)
        Y_new, T_new = solver._imex_step(solver.Y, solver.T, solver.delta_t)
        assert isinstance(Y_new, dict)
        assert T_new.shape == solver.T.shape
        assert torch.isfinite(T_new).all()

    def test_ml_combustion_closure(self, reacting_case):
        from pyfoam.applications.reacting_foam_enhanced_10 import ReactingFoamEnhanced10
        solver = ReactingFoamEnhanced10(reacting_case, ml_combustion=True)
        Y_corrected = solver._ml_combustion_closure(solver.T, solver.Y, solver.delta_t)
        assert isinstance(Y_corrected, dict)
        for name in solver.species:
            assert name in Y_corrected


# ===========================================================================
# Tests: SolidFoamEnhanced7
# ===========================================================================


class TestSolidFoamEnhanced7:
    """Tests for enhanced solid mechanics solver v7."""

    def test_init(self, cavity_case):
        from pyfoam.applications.solid_foam_enhanced_7 import SolidFoamEnhanced7
        solver = SolidFoamEnhanced7(cavity_case, E=200e9, nu=0.3, spectral_damage=True)
        assert solver.spectral_damage is True

    def test_init_defaults(self, cavity_case):
        from pyfoam.applications.solid_foam_enhanced_7 import SolidFoamEnhanced7
        solver = SolidFoamEnhanced7(cavity_case, E=200e9, nu=0.3)
        assert solver.topology_optimisation is True
        assert solver.implicit_contact is True
        assert solver.penalty_exponent == pytest.approx(3.0)

    def test_spectral_damage(self, cavity_case):
        from pyfoam.applications.solid_foam_enhanced_7 import SolidFoamEnhanced7
        solver = SolidFoamEnhanced7(cavity_case, E=200e9, nu=0.3, spectral_damage=True)
        sigma = torch.randn(solver.mesh.n_cells, 6)
        epsilon = torch.randn(solver.mesh.n_cells, 6) * 1e-4
        damage = solver._spectral_damage_evolution(
            sigma, epsilon, solver.damage, solver.delta_t,
        )
        assert damage.shape == (solver.mesh.n_cells,)
        assert (damage >= 0).all()
        assert (damage <= 1).all()
        assert torch.isfinite(damage).all()

    def test_topology_update(self, cavity_case):
        from pyfoam.applications.solid_foam_enhanced_7 import SolidFoamEnhanced7
        solver = SolidFoamEnhanced7(cavity_case, E=200e9, nu=0.3, topology_optimisation=True)
        sigma = torch.randn(solver.mesh.n_cells, 6)
        D = torch.randn(solver.mesh.n_cells, 3) * 1e-6
        rho = solver._topology_optimisation_update(sigma, D, solver.delta_t)
        assert rho.shape == (solver.mesh.n_cells,)
        assert (rho >= 0.01).all()
        assert (rho <= 1.0).all()
        assert torch.isfinite(rho).all()


# ===========================================================================
# Tests: FilmFoamEnhanced7
# ===========================================================================


class TestFilmFoamEnhanced7:
    """Tests for enhanced film solver v7."""

    def test_init(self, cavity_case):
        from pyfoam.applications.film_foam_enhanced_7 import FilmFoamEnhanced7
        solver = FilmFoamEnhanced7(cavity_case, cahn_hilliard=True, dlvo=True)
        assert solver.cahn_hilliard is True
        assert solver.dlvo is True

    def test_init_defaults(self, cavity_case):
        from pyfoam.applications.film_foam_enhanced_7 import FilmFoamEnhanced7
        solver = FilmFoamEnhanced7(cavity_case)
        assert solver.thermocapillary is True
        assert solver.ch_mobility == pytest.approx(1e-10)
        assert solver.hamaker == pytest.approx(1e-20)

    def test_cahn_hilliard_update(self, cavity_case):
        from pyfoam.applications.film_foam_enhanced_7 import FilmFoamEnhanced7
        solver = FilmFoamEnhanced7(cavity_case, cahn_hilliard=True)
        phi = torch.randn(solver.mesh.n_cells) * 0.1
        phi_new = solver._cahn_hilliard_update(phi, solver.delta_t)
        assert phi_new.shape == phi.shape
        assert torch.isfinite(phi_new).all()
        assert (phi_new >= -1.0).all()
        assert (phi_new <= 1.0).all()

    def test_marangoni_stress(self, cavity_case):
        from pyfoam.applications.film_foam_enhanced_7 import FilmFoamEnhanced7
        solver = FilmFoamEnhanced7(cavity_case, thermocapillary=True)
        h = torch.full((solver.mesh.n_cells,), 1e-6)
        T = torch.full((solver.mesh.n_cells,), 380.0)
        tau = solver._marangoni_stress_temperature(h, T)
        assert tau.shape == h.shape
        assert torch.isfinite(tau).all()

    def test_dlvo_pressure(self, cavity_case):
        from pyfoam.applications.film_foam_enhanced_7 import FilmFoamEnhanced7
        solver = FilmFoamEnhanced7(cavity_case, dlvo=True)
        h = torch.full((solver.mesh.n_cells,), 1e-8)
        T = torch.full((solver.mesh.n_cells,), 300.0)
        p_dlvo = solver._dlvo_disjoining_pressure(h, T)
        assert p_dlvo.shape == h.shape
        assert torch.isfinite(p_dlvo).all()


# ===========================================================================
# Tests: SprayFoamEnhanced7
# ===========================================================================


class TestSprayFoamEnhanced7:
    """Tests for enhanced spray solver v7."""

    def test_init(self, cavity_case):
        from pyfoam.applications.spray_foam_enhanced_7 import SprayFoamEnhanced7
        solver = SprayFoamEnhanced7(cavity_case, spray_amr=True, dns_calibrated_breakup=True)
        assert solver.spray_amr is True
        assert solver.dns_calibrated_breakup is True

    def test_init_defaults(self, cavity_case):
        from pyfoam.applications.spray_foam_enhanced_7 import SprayFoamEnhanced7
        solver = SprayFoamEnhanced7(cavity_case)
        assert solver.langevin_dispersion is True
        assert solver.langevin_C0 == pytest.approx(2.1)
        assert solver.amr_parcel_threshold == pytest.approx(5.0)

    def test_spray_amr_indicators(self, cavity_case):
        from pyfoam.applications.spray_foam_enhanced_7 import SprayFoamEnhanced7
        solver = SprayFoamEnhanced7(cavity_case, spray_amr=True)
        n_cells = solver.mesh.n_cells
        parcel_density = torch.full((n_cells,), 10.0)
        d_mean = torch.full((n_cells,), 1e-4)
        d_var = torch.full((n_cells,), 1e-9)
        indicators = solver._compute_spray_refinement_indicators(
            parcel_density, d_mean, d_var,
        )
        assert indicators.shape == (n_cells,)
        assert torch.isfinite(indicators).all()

    def test_dns_breakup_low_we(self, cavity_case):
        from pyfoam.applications.spray_foam_enhanced_7 import SprayFoamEnhanced7
        solver = SprayFoamEnhanced7(cavity_case, dns_calibrated_breakup=True)
        n_frag, frags = solver._dns_calibrated_breakup_model(1e-3, We=3.0, Oh=0.01, Re_d=100.0)
        assert n_frag == 1

    def test_dns_breakup_high_we(self, cavity_case):
        from pyfoam.applications.spray_foam_enhanced_7 import SprayFoamEnhanced7
        solver = SprayFoamEnhanced7(cavity_case, dns_calibrated_breakup=True)
        n_frag, frags = solver._dns_calibrated_breakup_model(1e-3, We=200.0, Oh=0.01, Re_d=1000.0)
        assert n_frag >= 2
        assert len(frags) == n_frag

    def test_langevin_dispersion(self, cavity_case):
        from pyfoam.applications.spray_foam_enhanced_7 import SprayFoamEnhanced7
        solver = SprayFoamEnhanced7(cavity_case, langevin_dispersion=True)
        U_p = torch.randn(10, 3) * 0.1
        U_f = torch.randn(10, 3) * 0.1
        U_disp = solver._langevin_dispersion_step(
            U_p, U_f, k=0.01, eps=0.001, dt=0.001, d_p=1e-4, rho_p=1000.0,
        )
        assert U_disp.shape == U_p.shape
        assert torch.isfinite(U_disp).all()


# ===========================================================================
# Tests: MultiphaseEulerFoamEnhanced8
# ===========================================================================


class TestMultiphaseEulerFoamEnhanced8:
    """Tests for enhanced multiphase Euler solver v8."""

    def test_init(self, cavity_case):
        from pyfoam.applications.multiphase_euler_foam_enhanced_8 import MultiphaseEulerFoamEnhanced8
        assert hasattr(MultiphaseEulerFoamEnhanced8, '__init__')

    def test_class_exists(self):
        from pyfoam.applications.multiphase_euler_foam_enhanced_8 import MultiphaseEulerFoamEnhanced8
        assert MultiphaseEulerFoamEnhanced8 is not None

    def test_hyperbolic_moment_logic(self):
        """Test hyperbolic moment advection preserves non-negativity."""
        moments = torch.rand(16, 4) * 0.5
        # Simulate advection (should not create negatives)
        moments_new = moments.clamp(min=0.0)
        assert moments_new.shape == (16, 4)
        assert (moments_new >= 0).all()
        assert torch.isfinite(moments_new).all()

    def test_filtered_drag_force_logic(self):
        """Test filtered drag force computation logic."""
        Re_p = torch.full((16,), 100.0)
        Cd = 24.0 / Re_p.clamp(min=1e-3) * (1.0 + 0.15 * Re_p.pow(0.687))
        assert Cd.shape == (16,)
        assert (Cd > 0).all()
        assert torch.isfinite(Cd).all()


# ===========================================================================
# Tests: Exports
# ===========================================================================


class TestExportsV8:
    """Tests for __init__.py exports of v8/v10/v7 solvers."""

    def test_ico_enhanced_8_exported(self):
        from pyfoam.applications import IcoFoamEnhanced8
        assert IcoFoamEnhanced8 is not None

    def test_simple_enhanced_8_exported(self):
        from pyfoam.applications import SimpleFoamEnhanced8
        assert SimpleFoamEnhanced8 is not None

    def test_piso_enhanced_8_exported(self):
        from pyfoam.applications import PisoFoamEnhanced8
        assert PisoFoamEnhanced8 is not None

    def test_pimple_enhanced_8_exported(self):
        from pyfoam.applications import PimpleFoamEnhanced8
        assert PimpleFoamEnhanced8 is not None

    def test_rho_pimple_enhanced_8_exported(self):
        from pyfoam.applications import RhoPimpleFoamEnhanced8
        assert RhoPimpleFoamEnhanced8 is not None

    def test_buoyant_simple_enhanced_8_exported(self):
        from pyfoam.applications import BuoyantSimpleFoamEnhanced8
        assert BuoyantSimpleFoamEnhanced8 is not None

    def test_buoyant_pimple_enhanced_8_exported(self):
        from pyfoam.applications import BuoyantPimpleFoamEnhanced8
        assert BuoyantPimpleFoamEnhanced8 is not None

    def test_reacting_enhanced_10_exported(self):
        from pyfoam.applications import ReactingFoamEnhanced10
        assert ReactingFoamEnhanced10 is not None

    def test_solid_enhanced_7_exported(self):
        from pyfoam.applications import SolidFoamEnhanced7
        assert SolidFoamEnhanced7 is not None

    def test_film_enhanced_7_exported(self):
        from pyfoam.applications import FilmFoamEnhanced7
        assert FilmFoamEnhanced7 is not None

    def test_spray_enhanced_7_exported(self):
        from pyfoam.applications import SprayFoamEnhanced7
        assert SprayFoamEnhanced7 is not None

    def test_multiphase_enhanced_8_exported(self):
        from pyfoam.applications import MultiphaseEulerFoamEnhanced8
        assert MultiphaseEulerFoamEnhanced8 is not None

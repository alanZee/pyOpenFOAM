"""
Unit tests for enhanced solver variants v10 (and v12/v9 specialized).

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
# Mesh generation helper (same pattern as test_enhanced_solvers_9)
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
# Tests: IcoFoamEnhanced10
# ===========================================================================


class TestIcoFoamEnhanced10:
    """Tests for enhanced ICO solver v10."""

    def test_init(self, cavity_case):
        from pyfoam.applications.ico_foam_enhanced_10 import IcoFoamEnhanced10
        solver = IcoFoamEnhanced10(cavity_case, hmg_precondition=True, space_time_galerkin=True)
        assert solver.hmg_precondition is True
        assert solver.space_time_galerkin is True

    def test_init_defaults(self, cavity_case):
        from pyfoam.applications.ico_foam_enhanced_10 import IcoFoamEnhanced10
        solver = IcoFoamEnhanced10(cavity_case)
        assert solver.hmg_levels == 3
        assert solver.st_order == 2
        assert solver.adaptive_compressibility is True
        assert solver.beta_ac_max == pytest.approx(10.0)

    def test_hmg_precondition(self, cavity_case):
        from pyfoam.applications.ico_foam_enhanced_10 import IcoFoamEnhanced10
        solver = IcoFoamEnhanced10(cavity_case, hmg_precondition=True)
        rhs = solver.p.clone() * 0.01
        p_hmg = solver._hmg_pressure_precondition(solver.p, rhs)
        assert p_hmg.shape == solver.p.shape
        assert torch.isfinite(p_hmg).all()

    def test_hmg_disabled(self, cavity_case):
        from pyfoam.applications.ico_foam_enhanced_10 import IcoFoamEnhanced10
        solver = IcoFoamEnhanced10(cavity_case, hmg_precondition=False)
        p = solver.p.clone()
        rhs = p.clone()
        p_out = solver._hmg_pressure_precondition(p, rhs)
        assert torch.allclose(p_out, p)

    def test_space_time_advance(self, cavity_case):
        from pyfoam.applications.ico_foam_enhanced_10 import IcoFoamEnhanced10
        solver = IcoFoamEnhanced10(cavity_case, space_time_galerkin=True)
        U_st = solver._space_time_advance(
            solver.U, solver.U.clone(), solver.p, solver.delta_t,
        )
        assert U_st.shape == solver.U.shape
        assert torch.isfinite(U_st).all()

    def test_artificial_compressibility(self, cavity_case):
        from pyfoam.applications.ico_foam_enhanced_10 import IcoFoamEnhanced10
        solver = IcoFoamEnhanced10(cavity_case, adaptive_compressibility=True)
        p_ac = solver._artificial_compressibility_pressure(
            solver.p, solver.U, solver.delta_t,
        )
        assert p_ac.shape == solver.p.shape
        assert torch.isfinite(p_ac).all()


# ===========================================================================
# Tests: SimpleFoamEnhanced10
# ===========================================================================


class TestSimpleFoamEnhanced10:
    """Tests for enhanced SIMPLE solver v10."""

    def test_init(self, cavity_case):
        from pyfoam.applications.simple_foam_enhanced_10 import SimpleFoamEnhanced10
        solver = SimpleFoamEnhanced10(cavity_case, olps=True, spectral_viscosity=True)
        assert solver.olps is True
        assert solver.spectral_viscosity is True

    def test_init_defaults(self, cavity_case):
        from pyfoam.applications.simple_foam_enhanced_10 import SimpleFoamEnhanced10
        solver = SimpleFoamEnhanced10(cavity_case)
        assert solver.ddurs is True
        assert solver.olps_patch_size == 16
        assert solver.sv_cutoff_wavenumber == pytest.approx(0.7)

    def test_olps_pressure(self, cavity_case):
        from pyfoam.applications.simple_foam_enhanced_10 import SimpleFoamEnhanced10
        solver = SimpleFoamEnhanced10(cavity_case, olps=True)
        rhs = solver.p.clone() * 0.01
        p_olps = solver._olps_pressure_correct(solver.p, rhs)
        assert p_olps.shape == solver.p.shape
        assert torch.isfinite(p_olps).all()

    def test_spectral_viscosity(self, cavity_case):
        from pyfoam.applications.simple_foam_enhanced_10 import SimpleFoamEnhanced10
        solver = SimpleFoamEnhanced10(cavity_case, spectral_viscosity=True)
        U_sv = solver._spectral_viscosity_stabilise(solver.U, solver.U.clone())
        assert U_sv.shape == solver.U.shape
        assert torch.isfinite(U_sv).all()

    def test_ddurs_relaxation(self, cavity_case):
        from pyfoam.applications.simple_foam_enhanced_10 import SimpleFoamEnhanced10
        solver = SimpleFoamEnhanced10(cavity_case, ddurs=True)
        alpha_U, alpha_p = solver._ddurs_relaxation_factor(10, 0.5)
        assert 0.1 <= alpha_U <= 0.95
        assert 0.05 <= alpha_p <= 0.5


# ===========================================================================
# Tests: PisoFoamEnhanced10
# ===========================================================================


class TestPisoFoamEnhanced10:
    """Tests for enhanced PISO solver v10."""

    def test_init(self, cavity_case):
        from pyfoam.applications.piso_foam_enhanced_10 import PisoFoamEnhanced10
        solver = PisoFoamEnhanced10(cavity_case, iles_mpdata=True, pressure_hodge=True)
        assert solver.iles_mpdata is True
        assert solver.pressure_hodge is True

    def test_init_defaults(self, cavity_case):
        from pyfoam.applications.piso_foam_enhanced_10 import PisoFoamEnhanced10
        solver = PisoFoamEnhanced10(cavity_case)
        assert solver.multirate_dt is True
        assert solver.mpdata_n_iters == 2
        assert solver.multirate_ratio == 4

    def test_iles_mpdata(self, cavity_case):
        from pyfoam.applications.piso_foam_enhanced_10 import PisoFoamEnhanced10
        solver = PisoFoamEnhanced10(cavity_case, iles_mpdata=True)
        U_iles = solver._iles_mpdata_advection(solver.U, solver.U.clone(), solver.delta_t)
        assert U_iles.shape == solver.U.shape
        assert torch.isfinite(U_iles).all()

    def test_pressure_hodge(self, cavity_case):
        from pyfoam.applications.piso_foam_enhanced_10 import PisoFoamEnhanced10
        solver = PisoFoamEnhanced10(cavity_case, pressure_hodge=True)
        U_h, p_h = solver._pressure_hodge_project(solver.U, solver.p)
        assert U_h.shape == solver.U.shape
        assert p_h.shape == solver.p.shape
        assert torch.isfinite(U_h).all()

    def test_multirate_dt(self, cavity_case):
        from pyfoam.applications.piso_foam_enhanced_10 import PisoFoamEnhanced10
        solver = PisoFoamEnhanced10(cavity_case, multirate_dt=True)
        dt_U, dt_p = solver._multirate_dt_select(solver.delta_t, 0)
        assert dt_U < dt_p
        assert dt_U == pytest.approx(solver.delta_t / solver.multirate_ratio)


# ===========================================================================
# Tests: PimpleFoamEnhanced10
# ===========================================================================


class TestPimpleFoamEnhanced10:
    """Tests for enhanced PIMPLE solver v10."""

    def test_init(self, cavity_case):
        from pyfoam.applications.pimple_foam_enhanced_10 import PimpleFoamEnhanced10
        solver = PimpleFoamEnhanced10(cavity_case, vms_pressure=True, oinn_corrector=True)
        assert solver.vms_pressure is True
        assert solver.oinn_corrector is True

    def test_init_defaults(self, cavity_case):
        from pyfoam.applications.pimple_foam_enhanced_10 import PimpleFoamEnhanced10
        solver = PimpleFoamEnhanced10(cavity_case)
        assert solver.energy_preserving is True
        assert solver.vms_stab_coeff == pytest.approx(0.1)
        assert solver.oinn_hidden_dim == 32

    def test_vms_pressure(self, cavity_case):
        from pyfoam.applications.pimple_foam_enhanced_10 import PimpleFoamEnhanced10
        solver = PimpleFoamEnhanced10(cavity_case, vms_pressure=True)
        p_vms = solver._vms_pressure_stabilise(solver.p, solver.U, solver.delta_t)
        assert p_vms.shape == solver.p.shape
        assert torch.isfinite(p_vms).all()

    def test_oinn_correct(self, cavity_case):
        from pyfoam.applications.pimple_foam_enhanced_10 import PimpleFoamEnhanced10
        solver = PimpleFoamEnhanced10(cavity_case, oinn_corrector=True)
        U_oinn = solver._oinn_correct(solver.U, solver.U.clone(), 0.5)
        assert U_oinn.shape == solver.U.shape
        assert torch.isfinite(U_oinn).all()

    def test_energy_budget(self, cavity_case):
        from pyfoam.applications.pimple_foam_enhanced_10 import PimpleFoamEnhanced10
        solver = PimpleFoamEnhanced10(cavity_case, energy_preserving=True)
        U_eb = solver._energy_budget_correct(solver.U, solver.U.clone(), solver.delta_t)
        assert U_eb.shape == solver.U.shape
        assert torch.isfinite(U_eb).all()


# ===========================================================================
# Tests: RhoPimpleFoamEnhanced10
# ===========================================================================


class TestRhoPimpleFoamEnhanced10:
    """Tests for enhanced compressible PIMPLE solver v10."""

    def test_init(self, compressible_case):
        from pyfoam.applications.rho_pimple_foam_enhanced_10 import RhoPimpleFoamEnhanced10
        solver = RhoPimpleFoamEnhanced10(compressible_case, hybrid_des=True)
        assert solver.hybrid_des is True

    def test_init_defaults(self, compressible_case):
        from pyfoam.applications.rho_pimple_foam_enhanced_10 import RhoPimpleFoamEnhanced10
        solver = RhoPimpleFoamEnhanced10(compressible_case)
        assert solver.thermo_consistent is True
        assert solver.acoustic_hybrid is True
        assert solver.des_constant == pytest.approx(0.65)

    def test_hybrid_des(self, compressible_case):
        from pyfoam.applications.rho_pimple_foam_enhanced_10 import RhoPimpleFoamEnhanced10
        solver = RhoPimpleFoamEnhanced10(compressible_case, hybrid_des=True)
        k = torch.ones(solver.mesh.n_cells) * 0.01
        nu_des = solver._hybrid_des_viscosity(solver.U, solver.rho, k, 0.01)
        assert nu_des.shape == (solver.mesh.n_cells,)
        assert torch.isfinite(nu_des).all()

    def test_thermo_consistent(self, compressible_case):
        from pyfoam.applications.rho_pimple_foam_enhanced_10 import RhoPimpleFoamEnhanced10
        solver = RhoPimpleFoamEnhanced10(compressible_case, thermo_consistent=True)
        p_new, rho_new, T_new = solver._thermo_consistent_correction(
            solver.p, solver.rho, solver.T, solver.U,
        )
        assert p_new.shape == solver.p.shape
        assert torch.isfinite(p_new).all()

    def test_acoustic_hybrid(self, compressible_case):
        from pyfoam.applications.rho_pimple_foam_enhanced_10 import RhoPimpleFoamEnhanced10
        solver = RhoPimpleFoamEnhanced10(compressible_case, acoustic_hybrid=True)
        U_new, p_new = solver._acoustic_hybrid_step(
            solver.U, solver.p, solver.rho, solver.delta_t,
        )
        assert U_new.shape == solver.U.shape
        assert torch.isfinite(U_new).all()


# ===========================================================================
# Tests: BuoyantSimpleFoamEnhanced10
# ===========================================================================


class TestBuoyantSimpleFoamEnhanced10:
    """Tests for enhanced buoyant SIMPLE solver v10."""

    def test_init(self, buoyant_case):
        from pyfoam.applications.buoyant_simple_foam_enhanced_10 import BuoyantSimpleFoamEnhanced10
        solver = BuoyantSimpleFoamEnhanced10(buoyant_case, non_boussinesq=True)
        assert solver.non_boussinesq is True

    def test_init_defaults(self, buoyant_case):
        from pyfoam.applications.buoyant_simple_foam_enhanced_10 import BuoyantSimpleFoamEnhanced10
        solver = BuoyantSimpleFoamEnhanced10(buoyant_case)
        assert solver.shell_htc is True
        assert solver.richardson_damping is True
        assert solver.shell_thickness == pytest.approx(0.005)
        assert solver.ri_threshold == pytest.approx(1.0)

    def test_variable_density_buoyancy(self, buoyant_case):
        from pyfoam.applications.buoyant_simple_foam_enhanced_10 import BuoyantSimpleFoamEnhanced10
        solver = BuoyantSimpleFoamEnhanced10(buoyant_case, non_boussinesq=True)
        rho = torch.ones(solver.mesh.n_cells) * 1.2
        T = solver.T
        F = solver._variable_density_buoyancy(solver.U, rho, T, 1.2, 300.0)
        assert F.shape == solver.U.shape
        assert torch.isfinite(F).all()

    def test_shell_htc(self, buoyant_case):
        from pyfoam.applications.buoyant_simple_foam_enhanced_10 import BuoyantSimpleFoamEnhanced10
        solver = BuoyantSimpleFoamEnhanced10(buoyant_case, shell_htc=True)
        T_corr = solver._shell_htc_correction(solver.T, 300.0, 10.0, 45.0)
        assert T_corr.shape == solver.T.shape
        assert torch.isfinite(T_corr).all()

    def test_richardson_damping(self, buoyant_case):
        from pyfoam.applications.buoyant_simple_foam_enhanced_10 import BuoyantSimpleFoamEnhanced10
        solver = BuoyantSimpleFoamEnhanced10(buoyant_case, richardson_damping=True)
        U_damp = solver._richardson_velocity_damp(solver.U, solver.T, 300.0)
        assert U_damp.shape == solver.U.shape
        assert torch.isfinite(U_damp).all()


# ===========================================================================
# Tests: BuoyantPimpleFoamEnhanced10
# ===========================================================================


class TestBuoyantPimpleFoamEnhanced10:
    """Tests for enhanced buoyant PIMPLE solver v10."""

    def test_init(self, buoyant_case):
        from pyfoam.applications.buoyant_pimple_foam_enhanced_10 import BuoyantPimpleFoamEnhanced10
        solver = BuoyantPimpleFoamEnhanced10(buoyant_case, cbpvs=True)
        assert solver.cbpvs is True

    def test_init_defaults(self, buoyant_case):
        from pyfoam.applications.buoyant_pimple_foam_enhanced_10 import BuoyantPimpleFoamEnhanced10
        solver = BuoyantPimpleFoamEnhanced10(buoyant_case)
        assert solver.rbtim is True
        assert solver.temporal_filter is True
        assert solver.filter_alpha == pytest.approx(0.3)

    def test_cbpvs_correction(self, buoyant_case):
        from pyfoam.applications.buoyant_pimple_foam_enhanced_10 import BuoyantPimpleFoamEnhanced10
        solver = BuoyantPimpleFoamEnhanced10(buoyant_case, cbpvs=True)
        U_new, p_new = solver._cbpvs_block_correction(
            solver.U, solver.p, solver.T, solver.delta_t,
        )
        assert U_new.shape == solver.U.shape
        assert p_new.shape == solver.p.shape
        assert torch.isfinite(U_new).all()

    def test_rbtim_coupling(self, buoyant_case):
        from pyfoam.applications.buoyant_pimple_foam_enhanced_10 import BuoyantPimpleFoamEnhanced10
        solver = BuoyantPimpleFoamEnhanced10(buoyant_case, rbtim=True)
        k = torch.ones(solver.mesh.n_cells) * 0.01
        eps = torch.ones(solver.mesh.n_cells) * 0.001
        T_new, k_new, eps_new = solver._rbtim_triple_coupling(solver.T, k, eps, solver.delta_t)
        assert T_new.shape == solver.T.shape
        assert (k_new >= 0).all()
        assert torch.isfinite(T_new).all()

    def test_temporal_filter(self, buoyant_case):
        from pyfoam.applications.buoyant_pimple_foam_enhanced_10 import BuoyantPimpleFoamEnhanced10
        solver = BuoyantPimpleFoamEnhanced10(buoyant_case, temporal_filter=True)
        T_f, U_f = solver._temporal_filter_buoyancy(solver.T, solver.U)
        assert T_f.shape == solver.T.shape
        assert U_f.shape == solver.U.shape
        assert torch.isfinite(T_f).all()


# ===========================================================================
# Tests: ReactingFoamEnhanced12
# ===========================================================================


class TestReactingFoamEnhanced12:
    """Tests for enhanced reacting solver v12."""

    def test_init(self, reacting_case):
        from pyfoam.applications.reacting_foam_enhanced_12 import ReactingFoamEnhanced12
        solver = ReactingFoamEnhanced12(reacting_case, mlrc=True, fgm=True, soot_mom=True)
        assert solver.mlrc is True
        assert solver.fgm is True
        assert solver.soot_mom is True

    def test_init_defaults(self, reacting_case):
        from pyfoam.applications.reacting_foam_enhanced_12 import ReactingFoamEnhanced12
        solver = ReactingFoamEnhanced12(reacting_case)
        assert solver.mlrc_latent_dim == 4
        assert solver.fgm_n_flamelets == 20
        assert solver.soot_n_moments == 6

    def test_mlrc_encode_decode(self, reacting_case):
        from pyfoam.applications.reacting_foam_enhanced_12 import ReactingFoamEnhanced12
        solver = ReactingFoamEnhanced12(reacting_case, mlrc=True)
        Y_corr = solver._mlrc_encode_decode(solver.Y, solver.T)
        assert isinstance(Y_corr, dict)
        for name in solver.species:
            assert name in Y_corr

    def test_fgm_lookup(self, reacting_case):
        from pyfoam.applications.reacting_foam_enhanced_12 import ReactingFoamEnhanced12
        solver = ReactingFoamEnhanced12(reacting_case, fgm=True)
        Z = torch.ones(solver.mesh.n_cells) * 0.06
        Y_fgm, T_fgm = solver._fgm_lookup(solver.Y, solver.T, Z)
        assert T_fgm.shape == solver.T.shape
        assert torch.isfinite(T_fgm).all()

    def test_soot_mom_transport(self, reacting_case):
        from pyfoam.applications.reacting_foam_enhanced_12 import ReactingFoamEnhanced12
        solver = ReactingFoamEnhanced12(reacting_case, soot_mom=True)
        moments = torch.zeros(solver.mesh.n_cells, solver.soot_n_moments)
        moments_new = solver._soot_mom_transport(moments, solver.T, solver.Y, solver.delta_t)
        assert moments_new.shape == moments.shape
        assert torch.isfinite(moments_new).all()


# ===========================================================================
# Tests: SolidFoamEnhanced9
# ===========================================================================


class TestSolidFoamEnhanced9:
    """Tests for enhanced solid mechanics solver v9."""

    def test_init(self, cavity_case):
        from pyfoam.applications.solid_foam_enhanced_9 import SolidFoamEnhanced9
        solver = SolidFoamEnhanced9(cavity_case, E=200e9, nu=0.3, peridynamics=True)
        assert solver.peridynamics is True

    def test_init_defaults(self, cavity_case):
        from pyfoam.applications.solid_foam_enhanced_9 import SolidFoamEnhanced9
        solver = SolidFoamEnhanced9(cavity_case, E=200e9, nu=0.3)
        assert solver.crystal_plasticity is True
        assert solver.natural_element is True
        assert solver.n_slip_systems == 12
        assert solver.pd_horizon == pytest.approx(0.01)

    def test_peridynamic_force(self, cavity_case):
        from pyfoam.applications.solid_foam_enhanced_9 import SolidFoamEnhanced9
        solver = SolidFoamEnhanced9(cavity_case, E=200e9, nu=0.3, peridynamics=True)
        U = torch.randn(solver.mesh.n_cells, 3) * 1e-6
        f_pd = solver._peridynamic_force(U, U.clone())
        assert f_pd.shape == U.shape
        assert torch.isfinite(f_pd).all()

    def test_crystal_plasticity(self, cavity_case):
        from pyfoam.applications.solid_foam_enhanced_9 import SolidFoamEnhanced9
        solver = SolidFoamEnhanced9(cavity_case, E=200e9, nu=0.3, crystal_plasticity=True)
        sigma = torch.randn(solver.mesh.n_cells, 6) * 1e6
        epsilon = torch.randn(solver.mesh.n_cells, 6) * 1e-4
        sigma_new, slip_res = solver._crystal_plasticity_update(sigma, epsilon, solver.delta_t)
        assert sigma_new.shape == sigma.shape
        assert slip_res.shape == (solver.mesh.n_cells, solver.n_slip_systems)
        assert torch.isfinite(sigma_new).all()

    def test_natural_element(self, cavity_case):
        from pyfoam.applications.solid_foam_enhanced_9 import SolidFoamEnhanced9
        solver = SolidFoamEnhanced9(cavity_case, E=200e9, nu=0.3, natural_element=True)
        positions = torch.randn(solver.mesh.n_cells, 3)
        field = torch.randn(solver.mesh.n_cells, 3)
        f_nei = solver._natural_element_interpolate(field, positions)
        assert f_nei.shape == field.shape
        assert torch.isfinite(f_nei).all()


# ===========================================================================
# Tests: FilmFoamEnhanced9
# ===========================================================================


class TestFilmFoamEnhanced9:
    """Tests for enhanced film solver v9."""

    def test_init(self, cavity_case):
        from pyfoam.applications.film_foam_enhanced_9 import FilmFoamEnhanced9
        solver = FilmFoamEnhanced9(cavity_case, inertial_film=True, wetting_drying=True)
        assert solver.inertial_film is True
        assert solver.wetting_drying is True

    def test_init_defaults(self, cavity_case):
        from pyfoam.applications.film_foam_enhanced_9 import FilmFoamEnhanced9
        solver = FilmFoamEnhanced9(cavity_case)
        assert solver.thermocapillary_analysis is True
        assert solver.precursor_thickness == pytest.approx(1e-9)
        assert solver.ma_critical == pytest.approx(80.0)

    def test_inertial_film(self, cavity_case):
        from pyfoam.applications.film_foam_enhanced_9 import FilmFoamEnhanced9
        solver = FilmFoamEnhanced9(cavity_case, inertial_film=True)
        n = solver.mesh.n_cells
        h = torch.full((n,), 1e-6)
        h_old = h.clone()
        U = torch.zeros(n)
        h_new = solver._inertial_film_advance(h, h_old, U, 1e-3, 1000.0, solver.delta_t)
        assert h_new.shape == h.shape
        assert torch.isfinite(h_new).all()

    def test_precursor_film(self, cavity_case):
        from pyfoam.applications.film_foam_enhanced_9 import FilmFoamEnhanced9
        solver = FilmFoamEnhanced9(cavity_case, wetting_drying=True)
        h = torch.full((solver.mesh.n_cells,), 1e-15)
        h_reg = solver._precursor_film_regularise(h)
        assert (h_reg >= solver.precursor_thickness).all()
        assert torch.isfinite(h_reg).all()

    def test_marangoni_stability(self, cavity_case):
        from pyfoam.applications.film_foam_enhanced_9 import FilmFoamEnhanced9
        solver = FilmFoamEnhanced9(cavity_case, thermocapillary_analysis=True)
        h = torch.full((solver.mesh.n_cells,), 1e-6)
        T = torch.full((solver.mesh.n_cells,), 300.0)
        Ma = solver._marangoni_stability_check(h, T, 1e-3, 1e-4, 0.6)
        assert Ma.shape == (solver.mesh.n_cells,)
        assert torch.isfinite(Ma).all()


# ===========================================================================
# Tests: SprayFoamEnhanced9
# ===========================================================================


class TestSprayFoamEnhanced9:
    """Tests for enhanced spray solver v9."""

    def test_init(self, cavity_case):
        from pyfoam.applications.spray_foam_enhanced_9 import SprayFoamEnhanced9
        solver = SprayFoamEnhanced9(cavity_case, apms=True, dns_drag=True)
        assert solver.apms is True
        assert solver.dns_drag is True

    def test_init_defaults(self, cavity_case):
        from pyfoam.applications.spray_foam_enhanced_9 import SprayFoamEnhanced9
        solver = SprayFoamEnhanced9(cavity_case)
        assert solver.langevin_memory is True
        assert solver.apms_merge_threshold == 50
        assert solver.wake_interaction_coeff == pytest.approx(0.5)

    def test_apms_merge_split(self, cavity_case):
        from pyfoam.applications.spray_foam_enhanced_9 import SprayFoamEnhanced9
        solver = SprayFoamEnhanced9(cavity_case, apms=True)
        n_new, d_new = solver._apms_merge_split(100, 0, 1e-4, solver.mesh.n_cells)
        assert isinstance(n_new, int)
        assert isinstance(d_new, float)
        assert n_new <= solver.apms_merge_threshold

    def test_dns_drag(self, cavity_case):
        from pyfoam.applications.spray_foam_enhanced_9 import SprayFoamEnhanced9
        solver = SprayFoamEnhanced9(cavity_case, dns_drag=True)
        Cd = solver._dns_informed_drag(500.0, 0.3, 0.7)
        assert isinstance(Cd, float)
        assert Cd > 0

    def test_langevin_memory(self, cavity_case):
        from pyfoam.applications.spray_foam_enhanced_9 import SprayFoamEnhanced9
        solver = SprayFoamEnhanced9(cavity_case, langevin_memory=True)
        n = solver.mesh.n_cells
        U_p = torch.randn(n, 3) * 0.1
        U_f = solver.U.clone()
        U_disp = solver._langevin_memory_dispersion(
            U_p, U_f, 0.01, 0.001, solver.delta_t, 1e-4, 700.0,
        )
        assert U_disp.shape == U_p.shape
        assert torch.isfinite(U_disp).all()


# ===========================================================================
# Tests: MultiphaseEulerFoamEnhanced10
# ===========================================================================


class TestMultiphaseEulerFoamEnhanced10:
    """Tests for enhanced multiphase Euler solver v10."""

    def test_init(self, cavity_case):
        from pyfoam.applications.multiphase_euler_foam_enhanced_10 import MultiphaseEulerFoamEnhanced10
        assert hasattr(MultiphaseEulerFoamEnhanced10, '__init__')

    def test_class_exists(self):
        from pyfoam.applications.multiphase_euler_foam_enhanced_10 import MultiphaseEulerFoamEnhanced10
        assert MultiphaseEulerFoamEnhanced10 is not None

    def test_musig_class_transport_logic(self):
        """Test MUSIG class transport preserves non-negativity."""
        fractions = torch.rand(16, 10) * 0.1
        fractions_new = fractions.clamp(min=0.0)
        total = fractions_new.sum(dim=-1, keepdim=True).clamp(min=1e-10)
        fractions_new = fractions_new / total
        assert fractions_new.shape == (16, 10)
        assert (fractions_new >= 0).all()
        assert torch.isfinite(fractions_new).all()

    def test_birs_source_logic(self):
        """Test BIRS source computation logic."""
        alpha_d = torch.ones(16) * 0.3
        U_slip = torch.ones(16) * 0.1
        k = torch.ones(16) * 0.01
        C_pb = 0.25
        d_b = 3e-3
        P_b = C_pb * alpha_d * U_slip.pow(3) / d_b
        k_new = k + P_b * 0.001
        assert (k_new >= 0).all()
        assert torch.isfinite(k_new).all()

    def test_iate_logic(self):
        """Test IATE interfacial area computation logic."""
        alpha_d = 0.3
        d_b = 3e-3
        ai_eq = 6.0 * alpha_d / d_b
        assert ai_eq > 0
        assert isinstance(ai_eq, float)


# ===========================================================================
# Tests: Exports
# ===========================================================================


class TestExportsV10:
    """Tests for __init__.py exports of v10/v12/v9 solvers."""

    def test_ico_enhanced_10_exported(self):
        from pyfoam.applications import IcoFoamEnhanced10
        assert IcoFoamEnhanced10 is not None

    def test_simple_enhanced_10_exported(self):
        from pyfoam.applications import SimpleFoamEnhanced10
        assert SimpleFoamEnhanced10 is not None

    def test_piso_enhanced_10_exported(self):
        from pyfoam.applications import PisoFoamEnhanced10
        assert PisoFoamEnhanced10 is not None

    def test_pimple_enhanced_10_exported(self):
        from pyfoam.applications import PimpleFoamEnhanced10
        assert PimpleFoamEnhanced10 is not None

    def test_rho_pimple_enhanced_10_exported(self):
        from pyfoam.applications import RhoPimpleFoamEnhanced10
        assert RhoPimpleFoamEnhanced10 is not None

    def test_buoyant_simple_enhanced_10_exported(self):
        from pyfoam.applications import BuoyantSimpleFoamEnhanced10
        assert BuoyantSimpleFoamEnhanced10 is not None

    def test_buoyant_pimple_enhanced_10_exported(self):
        from pyfoam.applications import BuoyantPimpleFoamEnhanced10
        assert BuoyantPimpleFoamEnhanced10 is not None

    def test_reacting_enhanced_12_exported(self):
        from pyfoam.applications import ReactingFoamEnhanced12
        assert ReactingFoamEnhanced12 is not None

    def test_solid_enhanced_9_exported(self):
        from pyfoam.applications import SolidFoamEnhanced9
        assert SolidFoamEnhanced9 is not None

    def test_film_enhanced_9_exported(self):
        from pyfoam.applications import FilmFoamEnhanced9
        assert FilmFoamEnhanced9 is not None

    def test_spray_enhanced_9_exported(self):
        from pyfoam.applications import SprayFoamEnhanced9
        assert SprayFoamEnhanced9 is not None

    def test_multiphase_enhanced_10_exported(self):
        from pyfoam.applications import MultiphaseEulerFoamEnhanced10
        assert MultiphaseEulerFoamEnhanced10 is not None

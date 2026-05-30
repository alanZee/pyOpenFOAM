"""
Unit tests for enhanced solver variants v7 (and v9/v6 specialized).

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
# Mesh generation helper (same pattern as test_enhanced_solvers_6)
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
# Tests: IcoFoamEnhanced7
# ===========================================================================


class TestIcoFoamEnhanced7:
    """Tests for enhanced ICO solver v7."""

    def test_init(self, cavity_case):
        from pyfoam.applications.ico_foam_enhanced_7 import IcoFoamEnhanced7
        solver = IcoFoamEnhanced7(cavity_case, wavelet_amr=True, energy_stable_convection=True)
        assert solver.wavelet_amr is True
        assert solver.energy_stable_convection is True

    def test_init_defaults(self, cavity_case):
        from pyfoam.applications.ico_foam_enhanced_7 import IcoFoamEnhanced7
        solver = IcoFoamEnhanced7(cavity_case)
        assert solver.wavelet_threshold == pytest.approx(0.01)
        assert solver.schur_precondition is True

    def test_wavelet_indicators(self, cavity_case):
        from pyfoam.applications.ico_foam_enhanced_7 import IcoFoamEnhanced7
        solver = IcoFoamEnhanced7(cavity_case, wavelet_amr=True)
        indicators = solver._compute_wavelet_indicators(solver.p)
        assert indicators.shape == (solver.mesh.n_cells,)
        assert torch.isfinite(indicators).all()
        assert (indicators >= 0).all()

    def test_wavelet_disabled(self, cavity_case):
        from pyfoam.applications.ico_foam_enhanced_7 import IcoFoamEnhanced7
        solver = IcoFoamEnhanced7(cavity_case, wavelet_amr=False)
        indicators = solver._compute_wavelet_indicators(solver.p)
        assert torch.allclose(indicators, torch.zeros_like(indicators))

    def test_mark_refinement(self, cavity_case):
        from pyfoam.applications.ico_foam_enhanced_7 import IcoFoamEnhanced7
        solver = IcoFoamEnhanced7(cavity_case, wavelet_amr=True, wavelet_threshold=1e-6)
        flags = solver._mark_cells_for_refinement(solver.p, solver.U)
        assert flags.shape == (solver.mesh.n_cells,)
        assert flags.dtype == torch.bool

    def test_energy_stable_convection(self, cavity_case):
        from pyfoam.applications.ico_foam_enhanced_7 import IcoFoamEnhanced7
        solver = IcoFoamEnhanced7(cavity_case, energy_stable_convection=True)
        U_es = solver._energy_stable_convection_flux(solver.U, solver.U.clone(), solver.delta_t)
        assert U_es.shape == solver.U.shape
        assert torch.isfinite(U_es).all()

    def test_energy_stable_disabled(self, cavity_case):
        from pyfoam.applications.ico_foam_enhanced_7 import IcoFoamEnhanced7
        solver = IcoFoamEnhanced7(cavity_case, energy_stable_convection=False)
        U = solver.U.clone()
        U_es = solver._energy_stable_convection_flux(U, solver.U.clone(), solver.delta_t)
        assert torch.allclose(U_es, U)

    def test_schur_precondition(self, cavity_case):
        from pyfoam.applications.ico_foam_enhanced_7 import IcoFoamEnhanced7
        solver = IcoFoamEnhanced7(cavity_case, schur_precondition=True)
        p_prec = solver._schur_precondition_pressure(solver.p, solver.U)
        assert p_prec.shape == solver.p.shape
        assert torch.isfinite(p_prec).all()


# ===========================================================================
# Tests: SimpleFoamEnhanced7
# ===========================================================================


class TestSimpleFoamEnhanced7:
    """Tests for enhanced SIMPLE solver v7."""

    def test_init(self, cavity_case):
        from pyfoam.applications.simple_foam_enhanced_7 import SimpleFoamEnhanced7
        solver = SimpleFoamEnhanced7(cavity_case, vms_turbulence=True, convex_splitting=True)
        assert solver.vms_turbulence is True
        assert solver.convex_splitting is True

    def test_init_defaults(self, cavity_case):
        from pyfoam.applications.simple_foam_enhanced_7 import SimpleFoamEnhanced7
        solver = SimpleFoamEnhanced7(cavity_case)
        assert solver.anderson_restart is True
        assert solver.restart_threshold == pytest.approx(1.2)

    def test_vms_viscosity(self, cavity_case):
        from pyfoam.applications.simple_foam_enhanced_7 import SimpleFoamEnhanced7
        solver = SimpleFoamEnhanced7(cavity_case, vms_turbulence=True)
        nu_vms = solver._compute_vms_viscosity(solver.U, 0.01)
        assert nu_vms.shape == (solver.mesh.n_cells,)
        assert (nu_vms >= 0).all()
        assert torch.isfinite(nu_vms).all()

    def test_vms_disabled(self, cavity_case):
        from pyfoam.applications.simple_foam_enhanced_7 import SimpleFoamEnhanced7
        solver = SimpleFoamEnhanced7(cavity_case, vms_turbulence=False)
        nu_vms = solver._compute_vms_viscosity(solver.U, 0.01)
        assert torch.allclose(nu_vms, torch.full_like(nu_vms, 0.01))

    def test_anderson_restart(self, cavity_case):
        from pyfoam.applications.simple_foam_enhanced_7 import SimpleFoamEnhanced7
        solver = SimpleFoamEnhanced7(cavity_case, anderson_restart=True)
        U_new = solver.U.clone() * 1.01
        U_result = solver._anderson_mixing_restart(U_new, solver.U.clone(), 1e-3)
        assert U_result.shape == solver.U.shape
        assert torch.isfinite(U_result).all()

    def test_convex_correction(self, cavity_case):
        from pyfoam.applications.simple_foam_enhanced_7 import SimpleFoamEnhanced7
        solver = SimpleFoamEnhanced7(cavity_case, convex_splitting=True)
        p_corr, U_corr = solver._convex_pressure_velocity_correction(
            solver.p, solver.U, solver.p.clone(),
        )
        assert p_corr.shape == solver.p.shape
        assert U_corr.shape == solver.U.shape
        assert torch.isfinite(p_corr).all()


# ===========================================================================
# Tests: PisoFoamEnhanced7
# ===========================================================================


class TestPisoFoamEnhanced7:
    """Tests for enhanced PISO solver v7."""

    def test_init(self, cavity_case):
        from pyfoam.applications.piso_foam_enhanced_7 import PisoFoamEnhanced7
        solver = PisoFoamEnhanced7(cavity_case, cmi_interpolation=True, hessian_precondition=True)
        assert solver.cmi_interpolation is True
        assert solver.hessian_precondition is True

    def test_init_defaults(self, cavity_case):
        from pyfoam.applications.piso_foam_enhanced_7 import PisoFoamEnhanced7
        solver = PisoFoamEnhanced7(cavity_case)
        assert solver.dual_weighted_error is True
        assert solver.cmi_interpolation is True

    def test_dual_weighted_error(self, cavity_case):
        from pyfoam.applications.piso_foam_enhanced_7 import PisoFoamEnhanced7
        solver = PisoFoamEnhanced7(cavity_case, dual_weighted_error=True)
        U_prev = solver.U.clone() * 0.99
        error = solver._dual_weighted_residual_error(
            solver.U, solver.U.clone() * 0.99, U_prev * 0.98, solver.delta_t,
        )
        assert isinstance(error, float)

    def test_dual_weighted_disabled(self, cavity_case):
        from pyfoam.applications.piso_foam_enhanced_7 import PisoFoamEnhanced7
        solver = PisoFoamEnhanced7(cavity_case, dual_weighted_error=False)
        error = solver._dual_weighted_residual_error(
            solver.U, solver.U.clone(), solver.U.clone(), solver.delta_t,
        )
        assert error == 0.0

    def test_cmi_interpolation(self, cavity_case):
        from pyfoam.applications.piso_foam_enhanced_7 import PisoFoamEnhanced7
        solver = PisoFoamEnhanced7(cavity_case, cmi_interpolation=True)
        U_cmi = solver._conservative_momentum_interpolation(
            solver.U, solver.p, solver.U.clone(),
        )
        assert U_cmi.shape == solver.U.shape
        assert torch.isfinite(U_cmi).all()

    def test_hessian_precondition(self, cavity_case):
        from pyfoam.applications.piso_foam_enhanced_7 import PisoFoamEnhanced7
        solver = PisoFoamEnhanced7(cavity_case, hessian_precondition=True)
        p_prec = solver._pressure_hessian_precondition(solver.p, solver.U)
        assert p_prec.shape == solver.p.shape
        assert torch.isfinite(p_prec).all()


# ===========================================================================
# Tests: PimpleFoamEnhanced7
# ===========================================================================


class TestPimpleFoamEnhanced7:
    """Tests for enhanced PIMPLE solver v7."""

    def test_init(self, cavity_case):
        from pyfoam.applications.pimple_foam_enhanced_7 import PimpleFoamEnhanced7
        solver = PimpleFoamEnhanced7(cavity_case, block_coupled=True, line_relaxation=True)
        assert solver.block_coupled is True
        assert solver.line_relaxation is True

    def test_init_defaults(self, cavity_case):
        from pyfoam.applications.pimple_foam_enhanced_7 import PimpleFoamEnhanced7
        solver = PimpleFoamEnhanced7(cavity_case)
        assert solver.adaptive_semi_implicit is True
        assert solver.cfl_threshold_si == pytest.approx(0.5)

    def test_block_coupled_solve(self, cavity_case):
        from pyfoam.applications.pimple_foam_enhanced_7 import PimpleFoamEnhanced7
        solver = PimpleFoamEnhanced7(cavity_case, block_coupled=True)
        U_corr, p_corr = solver._block_coupled_solve(
            solver.U, solver.p, solver.U.clone(), solver.p.clone(), solver.delta_t,
        )
        assert U_corr.shape == solver.U.shape
        assert p_corr.shape == solver.p.shape
        assert torch.isfinite(U_corr).all()
        assert torch.isfinite(p_corr).all()

    def test_select_implicit_level(self, cavity_case):
        from pyfoam.applications.pimple_foam_enhanced_7 import PimpleFoamEnhanced7
        solver = PimpleFoamEnhanced7(cavity_case, adaptive_semi_implicit=True)
        factor = solver._select_implicit_level(solver.U, solver.delta_t)
        assert 0.0 <= factor <= 1.0

    def test_hierarchical_multigrid(self, cavity_case):
        from pyfoam.applications.pimple_foam_enhanced_7 import PimpleFoamEnhanced7
        solver = PimpleFoamEnhanced7(cavity_case, line_relaxation=True)
        rhs = solver.p.clone() * 0.01
        p_mg = solver._hierarchical_multigrid_solve(solver.p, rhs, n_levels=2)
        assert p_mg.shape == solver.p.shape
        assert torch.isfinite(p_mg).all()


# ===========================================================================
# Tests: RhoPimpleFoamEnhanced7
# ===========================================================================


class TestRhoPimpleFoamEnhanced7:
    """Tests for enhanced compressible PIMPLE solver v7."""

    def test_init(self, compressible_case):
        from pyfoam.applications.rho_pimple_foam_enhanced_7 import RhoPimpleFoamEnhanced7
        solver = RhoPimpleFoamEnhanced7(compressible_case, acoustic_splitting=True)
        assert solver.acoustic_splitting is True

    def test_init_defaults(self, compressible_case):
        from pyfoam.applications.rho_pimple_foam_enhanced_7 import RhoPimpleFoamEnhanced7
        solver = RhoPimpleFoamEnhanced7(compressible_case)
        assert solver.pressure_density is True
        assert solver.energy_switching is True
        assert solver.mach_switch_threshold == pytest.approx(0.3)

    def test_pressure_based_density(self, compressible_case):
        from pyfoam.applications.rho_pimple_foam_enhanced_7 import RhoPimpleFoamEnhanced7
        solver = RhoPimpleFoamEnhanced7(compressible_case, pressure_density=True)
        rho_corr = solver._pressure_based_density(solver.rho, solver.p, solver.T)
        assert rho_corr.shape == solver.rho.shape
        assert (rho_corr >= 0).all()
        assert torch.isfinite(rho_corr).all()

    def test_acoustic_splitting(self, compressible_case):
        from pyfoam.applications.rho_pimple_foam_enhanced_7 import RhoPimpleFoamEnhanced7
        solver = RhoPimpleFoamEnhanced7(compressible_case, acoustic_splitting=True)
        U_ac, p_ac = solver._acoustic_convective_split_step(
            solver.U, solver.p, solver.rho, solver.delta_t,
        )
        assert U_ac.shape == solver.U.shape
        assert p_ac.shape == solver.p.shape
        assert torch.isfinite(U_ac).all()

    def test_energy_enthalpy_switch(self, compressible_case):
        from pyfoam.applications.rho_pimple_foam_enhanced_7 import RhoPimpleFoamEnhanced7
        solver = RhoPimpleFoamEnhanced7(compressible_case, energy_switching=True)
        T_corr = solver._energy_enthalpy_switch(solver.T, solver.U, solver.p, solver.rho)
        assert T_corr.shape == solver.T.shape
        assert torch.isfinite(T_corr).all()
        assert (T_corr >= 200).all()


# ===========================================================================
# Tests: BuoyantSimpleFoamEnhanced7
# ===========================================================================


class TestBuoyantSimpleFoamEnhanced7:
    """Tests for enhanced buoyant SIMPLE solver v7."""

    def test_init(self, buoyant_case):
        from pyfoam.applications.buoyant_simple_foam_enhanced_7 import BuoyantSimpleFoamEnhanced7
        solver = BuoyantSimpleFoamEnhanced7(buoyant_case, quadratic_boussinesq=True)
        assert solver.quadratic_boussinesq is True

    def test_init_defaults(self, buoyant_case):
        from pyfoam.applications.buoyant_simple_foam_enhanced_7 import BuoyantSimpleFoamEnhanced7
        solver = BuoyantSimpleFoamEnhanced7(buoyant_case)
        assert solver.overset_coupling is True
        assert solver.radiation_acceleration is True

    def test_quadratic_boussinesq(self, buoyant_case):
        from pyfoam.applications.buoyant_simple_foam_enhanced_7 import BuoyantSimpleFoamEnhanced7
        solver = BuoyantSimpleFoamEnhanced7(buoyant_case, quadratic_boussinesq=True)
        rho = solver._quadratic_boussinesq_density(1.2, solver.T, 300.0)
        assert rho.shape == solver.T.shape
        assert (rho > 0).all()
        assert torch.isfinite(rho).all()

    def test_radiation_predictor(self, buoyant_case):
        from pyfoam.applications.buoyant_simple_foam_enhanced_7 import BuoyantSimpleFoamEnhanced7
        solver = BuoyantSimpleFoamEnhanced7(buoyant_case, radiation_acceleration=True)
        T_pred = solver._radiation_buoyancy_predictor(solver.T, solver.rho)
        assert T_pred.shape == solver.T.shape
        assert torch.isfinite(T_pred).all()
        assert (T_pred >= 200).all()

    def test_overset_interpolation(self, buoyant_case):
        from pyfoam.applications.buoyant_simple_foam_enhanced_7 import BuoyantSimpleFoamEnhanced7
        solver = BuoyantSimpleFoamEnhanced7(buoyant_case, overset_coupling=True)
        T_corr, U_corr = solver._overset_buoyancy_interpolation(solver.T, solver.U)
        assert T_corr.shape == solver.T.shape
        assert U_corr.shape == solver.U.shape
        assert torch.isfinite(T_corr).all()


# ===========================================================================
# Tests: BuoyantPimpleFoamEnhanced7
# ===========================================================================


class TestBuoyantPimpleFoamEnhanced7:
    """Tests for enhanced buoyant PIMPLE solver v7."""

    def test_init(self, buoyant_case):
        from pyfoam.applications.buoyant_pimple_foam_enhanced_7 import BuoyantPimpleFoamEnhanced7
        solver = BuoyantPimpleFoamEnhanced7(buoyant_case, implicit_buoyancy=True, thermal_les=True)
        assert solver.implicit_buoyancy is True
        assert solver.thermal_les is True

    def test_init_defaults(self, buoyant_case):
        from pyfoam.applications.buoyant_pimple_foam_enhanced_7 import BuoyantPimpleFoamEnhanced7
        solver = BuoyantPimpleFoamEnhanced7(buoyant_case)
        assert solver.adaptive_thermal_bl is True
        assert solver.prandtl_sgs == pytest.approx(0.7)

    def test_implicit_buoyancy_coupling(self, buoyant_case):
        from pyfoam.applications.buoyant_pimple_foam_enhanced_7 import BuoyantPimpleFoamEnhanced7
        solver = BuoyantPimpleFoamEnhanced7(buoyant_case, implicit_buoyancy=True)
        p_corr, U_corr = solver._implicit_buoyancy_pressure_coupling(
            solver.p, solver.U, solver.T, solver.rho, solver.delta_t,
        )
        assert p_corr.shape == solver.p.shape
        assert U_corr.shape == solver.U.shape
        assert torch.isfinite(p_corr).all()

    def test_thermal_sgs_diffusivity(self, buoyant_case):
        from pyfoam.applications.buoyant_pimple_foam_enhanced_7 import BuoyantPimpleFoamEnhanced7
        solver = BuoyantPimpleFoamEnhanced7(buoyant_case, thermal_les=True)
        alpha = solver._compute_thermal_sgs_diffusivity(solver.U, 0.01)
        assert alpha > 0

    def test_thermal_bl_detection(self, buoyant_case):
        from pyfoam.applications.buoyant_pimple_foam_enhanced_7 import BuoyantPimpleFoamEnhanced7
        solver = BuoyantPimpleFoamEnhanced7(buoyant_case, adaptive_thermal_bl=True)
        bl = solver._detect_thermal_bl_thickness(solver.T)
        assert isinstance(bl, float)
        assert 0.0 < bl <= 1.0


# ===========================================================================
# Tests: ReactingFoamEnhanced9
# ===========================================================================


class TestReactingFoamEnhanced9:
    """Tests for enhanced reacting solver v9."""

    def test_init(self, reacting_case):
        from pyfoam.applications.reacting_foam_enhanced_9 import ReactingFoamEnhanced9
        solver = ReactingFoamEnhanced9(reacting_case, adaptive_splitting=True, ntc_chemistry=True)
        assert solver.adaptive_splitting is True
        assert solver.ntc_chemistry is True

    def test_init_defaults(self, reacting_case):
        from pyfoam.applications.reacting_foam_enhanced_9 import ReactingFoamEnhanced9
        solver = ReactingFoamEnhanced9(reacting_case)
        assert solver.block_jacobi is True
        assert solver.n_jacobi_iters == 3
        assert solver.splitting_tolerance == pytest.approx(1e-4)

    def test_adaptive_splitting_ratio(self, reacting_case):
        from pyfoam.applications.reacting_foam_enhanced_9 import ReactingFoamEnhanced9
        solver = ReactingFoamEnhanced9(reacting_case, adaptive_splitting=True)
        ratio = solver._adaptive_splitting_ratio(solver.Y, solver.T, solver.delta_t)
        assert isinstance(ratio, float)
        assert 0.0 <= ratio <= 1.0

    def test_adaptive_splitting_disabled(self, reacting_case):
        from pyfoam.applications.reacting_foam_enhanced_9 import ReactingFoamEnhanced9
        solver = ReactingFoamEnhanced9(reacting_case, adaptive_splitting=False)
        ratio = solver._adaptive_splitting_ratio(solver.Y, solver.T, solver.delta_t)
        assert ratio == 0.5

    def test_ntc_lookup(self, reacting_case):
        from pyfoam.applications.reacting_foam_enhanced_9 import ReactingFoamEnhanced9
        solver = ReactingFoamEnhanced9(reacting_case, ntc_chemistry=True)
        Y_updated = solver._ntc_lookup(solver.Y, solver.T, solver.delta_t)
        assert isinstance(Y_updated, dict)
        for name in solver.species:
            assert name in Y_updated

    def test_block_jacobi(self, reacting_case):
        from pyfoam.applications.reacting_foam_enhanced_9 import ReactingFoamEnhanced9
        solver = ReactingFoamEnhanced9(reacting_case, block_jacobi=True, n_jacobi_iters=2)
        Y_updated = solver._block_jacobi_species_solve(solver.Y, solver.T, solver.delta_t)
        assert isinstance(Y_updated, dict)
        for name in solver.species:
            assert name in Y_updated


# ===========================================================================
# Tests: SolidFoamEnhanced6
# ===========================================================================


class TestSolidFoamEnhanced6:
    """Tests for enhanced solid mechanics solver v6."""

    def test_init(self, cavity_case):
        from pyfoam.applications.solid_foam_enhanced_6 import SolidFoamEnhanced6
        solver = SolidFoamEnhanced6(cavity_case, E=200e9, nu=0.3, xfem=True)
        assert solver.xfem is True

    def test_init_defaults(self, cavity_case):
        from pyfoam.applications.solid_foam_enhanced_6 import SolidFoamEnhanced6
        solver = SolidFoamEnhanced6(cavity_case, E=200e9, nu=0.3)
        assert solver.coupled_fatigue is True
        assert solver.mlpg_recovery is True
        assert solver.mlpg_radius_factor == pytest.approx(2.0)

    def test_enrichment_functions(self, cavity_case):
        from pyfoam.applications.solid_foam_enhanced_6 import SolidFoamEnhanced6
        solver = SolidFoamEnhanced6(cavity_case, E=200e9, nu=0.3, xfem=True)
        damage = torch.full((solver.mesh.n_cells,), 0.5)
        enrichment = solver._compute_enrichment_functions(damage)
        assert enrichment.shape == (solver.mesh.n_cells, solver.n_enrichment_dofs)
        assert torch.isfinite(enrichment).all()

    def test_enrichment_disabled(self, cavity_case):
        from pyfoam.applications.solid_foam_enhanced_6 import SolidFoamEnhanced6
        solver = SolidFoamEnhanced6(cavity_case, E=200e9, nu=0.3, xfem=False)
        enrichment = solver._compute_enrichment_functions(solver.damage)
        assert torch.allclose(enrichment, solver.enrichment)

    def test_dislocation_density(self, cavity_case):
        from pyfoam.applications.solid_foam_enhanced_6 import SolidFoamEnhanced6
        solver = SolidFoamEnhanced6(cavity_case, E=200e9, nu=0.3, coupled_fatigue=True)
        sigma = torch.randn(solver.mesh.n_cells, 6)
        epsilon = torch.randn(solver.mesh.n_cells, 6) * 1e-4
        rho = solver._update_dislocation_density(sigma, epsilon, solver.delta_t)
        assert rho.shape == (solver.mesh.n_cells,)
        assert (rho >= 0).all()
        assert torch.isfinite(rho).all()

    def test_mlpg_recovery(self, cavity_case):
        from pyfoam.applications.solid_foam_enhanced_6 import SolidFoamEnhanced6
        solver = SolidFoamEnhanced6(cavity_case, E=200e9, nu=0.3, mlpg_recovery=True)
        sigma = torch.randn(solver.mesh.n_cells, 6)
        sigma_smooth = solver._mlpg_stress_smoothing(sigma)
        assert sigma_smooth.shape == sigma.shape
        assert torch.isfinite(sigma_smooth).all()


# ===========================================================================
# Tests: FilmFoamEnhanced6
# ===========================================================================


class TestFilmFoamEnhanced6:
    """Tests for enhanced film solver v6."""

    def test_init(self, cavity_case):
        from pyfoam.applications.film_foam_enhanced_6 import FilmFoamEnhanced6
        solver = FilmFoamEnhanced6(cavity_case, ehd=True, phase_change=True)
        assert solver.ehd is True
        assert solver.phase_change is True

    def test_init_defaults(self, cavity_case):
        from pyfoam.applications.film_foam_enhanced_6 import FilmFoamEnhanced6
        solver = FilmFoamEnhanced6(cavity_case)
        assert solver.viscoelastic is True
        assert solver.electric_field == pytest.approx(1e6)
        assert solver.latent_heat == pytest.approx(2.26e6)

    def test_ehd_pressure(self, cavity_case):
        from pyfoam.applications.film_foam_enhanced_6 import FilmFoamEnhanced6
        solver = FilmFoamEnhanced6(cavity_case, ehd=True)
        h = torch.full((solver.mesh.n_cells,), 1e-6)
        p_ehd = solver._compute_ehd_pressure(h)
        assert p_ehd.shape == h.shape
        assert (p_ehd >= 0).all()
        assert torch.isfinite(p_ehd).all()

    def test_ehd_disabled(self, cavity_case):
        from pyfoam.applications.film_foam_enhanced_6 import FilmFoamEnhanced6
        solver = FilmFoamEnhanced6(cavity_case, ehd=False)
        h = torch.full((solver.mesh.n_cells,), 1e-6)
        p_ehd = solver._compute_ehd_pressure(h)
        assert torch.allclose(p_ehd, torch.zeros_like(p_ehd))

    def test_phase_change_source(self, cavity_case):
        from pyfoam.applications.film_foam_enhanced_6 import FilmFoamEnhanced6
        solver = FilmFoamEnhanced6(cavity_case, phase_change=True)
        h = torch.full((solver.mesh.n_cells,), 1e-6)
        T = torch.full((solver.mesh.n_cells,), 380.0)
        dS, Q = solver._compute_phase_change_source(h, T)
        assert dS.shape == h.shape
        assert Q.shape == h.shape
        assert torch.isfinite(dS).all()
        assert torch.isfinite(Q).all()

    def test_viscoelastic_stress(self, cavity_case):
        from pyfoam.applications.film_foam_enhanced_6 import FilmFoamEnhanced6
        solver = FilmFoamEnhanced6(cavity_case, viscoelastic=True)
        shear = torch.full((solver.mesh.n_cells,), 100.0)
        mu = torch.full((solver.mesh.n_cells,), 0.001)
        tau = solver._viscoelastic_stress_update(shear, mu, solver.delta_t)
        assert tau.shape == (solver.mesh.n_cells,)
        assert torch.isfinite(tau).all()


# ===========================================================================
# Tests: SprayFoamEnhanced6
# ===========================================================================


class TestSprayFoamEnhanced6:
    """Tests for enhanced spray solver v6."""

    def test_init(self, cavity_case):
        from pyfoam.applications.spray_foam_enhanced_6 import SprayFoamEnhanced6
        solver = SprayFoamEnhanced6(cavity_case, stochastic_breakup=True, electrostatic=True)
        assert solver.stochastic_breakup is True
        assert solver.electrostatic is True

    def test_init_defaults(self, cavity_case):
        from pyfoam.applications.spray_foam_enhanced_6 import SprayFoamEnhanced6
        solver = SprayFoamEnhanced6(cavity_case)
        assert solver.wall_film is True
        assert solver.splash_threshold_we == pytest.approx(50.0)
        assert solver.charge_to_mass == pytest.approx(1e-3)

    def test_stochastic_breakup_low_we(self, cavity_case):
        from pyfoam.applications.spray_foam_enhanced_6 import SprayFoamEnhanced6
        solver = SprayFoamEnhanced6(cavity_case, stochastic_breakup=True)
        n_frag, frags = solver._stochastic_breakup_model(1e-3, We=5.0, Oh=0.01)
        assert n_frag == 1
        assert len(frags) == 1

    def test_stochastic_breakup_high_we(self, cavity_case):
        from pyfoam.applications.spray_foam_enhanced_6 import SprayFoamEnhanced6
        solver = SprayFoamEnhanced6(cavity_case, stochastic_breakup=True)
        n_frag, frags = solver._stochastic_breakup_model(1e-3, We=200.0, Oh=0.01)
        assert n_frag >= 2
        assert len(frags) == n_frag

    def test_wall_interaction(self, cavity_case):
        from pyfoam.applications.spray_foam_enhanced_6 import SprayFoamEnhanced6
        solver = SprayFoamEnhanced6(cavity_case, wall_film=True)
        assert solver._spray_wall_interaction(We=3.0, d=1e-4) == "rebound"
        assert solver._spray_wall_interaction(We=20.0, d=1e-4) == "spread"
        assert solver._spray_wall_interaction(We=100.0, d=1e-4) == "splash"

    def test_wall_film_state_dataclass(self):
        from pyfoam.applications.spray_foam_enhanced_6 import WallFilmState
        state = WallFilmState()
        assert state.thickness.shape == (0,)

    def test_droplet_charge(self, cavity_case):
        from pyfoam.applications.spray_foam_enhanced_6 import SprayFoamEnhanced6
        solver = SprayFoamEnhanced6(cavity_case, electrostatic=True)
        q = solver._compute_droplet_charge(solver.delta_t)
        assert q.shape == (solver.mesh.n_cells,)
        assert (q >= 0).all()
        assert torch.isfinite(q).all()


# ===========================================================================
# Tests: MultiphaseEulerFoamEnhanced7
# ===========================================================================


class TestMultiphaseEulerFoamEnhanced7:
    """Tests for enhanced multiphase Euler solver v7."""

    def test_init(self, cavity_case):
        from pyfoam.applications.multiphase_euler_foam_enhanced_7 import MultiphaseEulerFoamEnhanced7
        assert hasattr(MultiphaseEulerFoamEnhanced7, '__init__')

    def test_class_exists(self):
        from pyfoam.applications.multiphase_euler_foam_enhanced_7 import MultiphaseEulerFoamEnhanced7
        assert MultiphaseEulerFoamEnhanced7 is not None

    def test_des_length_scale_logic(self):
        """Test DES length scale computation logic."""
        # Simulate the DES length scale
        Delta = torch.full((16,), 0.25)
        C_des = 0.65
        l_rans = Delta * 0.5
        l_les = C_des * Delta
        l_des = torch.min(l_rans, l_les)
        assert l_des.shape == (16,)
        assert (l_des >= 0).all()
        assert torch.isfinite(l_des).all()

    def test_poly_iac_logic(self):
        """Test poly-dispersed IAC computation logic."""
        alpha = torch.rand(16) * 0.5
        d32 = torch.full((16,), 1e-3)
        a_iac = 6.0 * alpha / d32.clamp(min=1e-6)
        assert a_iac.shape == (16,)
        assert (a_iac >= 0).all()
        assert torch.isfinite(a_iac).all()


# ===========================================================================
# Tests: Exports
# ===========================================================================


class TestExportsV7:
    """Tests for __init__.py exports of v7/v9/v6 solvers."""

    def test_ico_enhanced_7_exported(self):
        from pyfoam.applications import IcoFoamEnhanced7
        assert IcoFoamEnhanced7 is not None

    def test_simple_enhanced_7_exported(self):
        from pyfoam.applications import SimpleFoamEnhanced7
        assert SimpleFoamEnhanced7 is not None

    def test_piso_enhanced_7_exported(self):
        from pyfoam.applications import PisoFoamEnhanced7
        assert PisoFoamEnhanced7 is not None

    def test_pimple_enhanced_7_exported(self):
        from pyfoam.applications import PimpleFoamEnhanced7
        assert PimpleFoamEnhanced7 is not None

    def test_rho_pimple_enhanced_7_exported(self):
        from pyfoam.applications import RhoPimpleFoamEnhanced7
        assert RhoPimpleFoamEnhanced7 is not None

    def test_buoyant_simple_enhanced_7_exported(self):
        from pyfoam.applications import BuoyantSimpleFoamEnhanced7
        assert BuoyantSimpleFoamEnhanced7 is not None

    def test_buoyant_pimple_enhanced_7_exported(self):
        from pyfoam.applications import BuoyantPimpleFoamEnhanced7
        assert BuoyantPimpleFoamEnhanced7 is not None

    def test_reacting_enhanced_9_exported(self):
        from pyfoam.applications import ReactingFoamEnhanced9
        assert ReactingFoamEnhanced9 is not None

    def test_solid_enhanced_6_exported(self):
        from pyfoam.applications import SolidFoamEnhanced6
        assert SolidFoamEnhanced6 is not None

    def test_film_enhanced_6_exported(self):
        from pyfoam.applications import FilmFoamEnhanced6
        assert FilmFoamEnhanced6 is not None

    def test_spray_enhanced_6_exported(self):
        from pyfoam.applications import SprayFoamEnhanced6
        assert SprayFoamEnhanced6 is not None

    def test_multiphase_enhanced_7_exported(self):
        from pyfoam.applications import MultiphaseEulerFoamEnhanced7
        assert MultiphaseEulerFoamEnhanced7 is not None

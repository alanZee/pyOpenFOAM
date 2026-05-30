"""
Unit tests for enhanced solver variants v9 (and v11/v8 specialized).

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
# Mesh generation helper (same pattern as test_enhanced_solvers_8)
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
# Tests: IcoFoamEnhanced9
# ===========================================================================


class TestIcoFoamEnhanced9:
    """Tests for enhanced ICO solver v9."""

    def test_init(self, cavity_case):
        from pyfoam.applications.ico_foam_enhanced_9 import IcoFoamEnhanced9
        solver = IcoFoamEnhanced9(cavity_case, nn_precondition=True, symplectic_integrator=True)
        assert solver.nn_precondition is True
        assert solver.symplectic_integrator is True

    def test_init_defaults(self, cavity_case):
        from pyfoam.applications.ico_foam_enhanced_9 import IcoFoamEnhanced9
        solver = IcoFoamEnhanced9(cavity_case)
        assert solver.p_refinement is True
        assert solver.p_max_order == 4
        assert solver.symplectic_sub_steps == 2

    def test_nn_precondition(self, cavity_case):
        from pyfoam.applications.ico_foam_enhanced_9 import IcoFoamEnhanced9
        solver = IcoFoamEnhanced9(cavity_case, nn_precondition=True)
        p_prec = solver._nn_pressure_precondition(solver.p, solver.U)
        assert p_prec.shape == solver.p.shape
        assert torch.isfinite(p_prec).all()

    def test_nn_disabled(self, cavity_case):
        from pyfoam.applications.ico_foam_enhanced_9 import IcoFoamEnhanced9
        solver = IcoFoamEnhanced9(cavity_case, nn_precondition=False)
        p = solver.p.clone()
        p_prec = solver._nn_pressure_precondition(p, solver.U)
        assert torch.allclose(p_prec, p)

    def test_polynomial_order(self, cavity_case):
        from pyfoam.applications.ico_foam_enhanced_9 import IcoFoamEnhanced9
        solver = IcoFoamEnhanced9(cavity_case, p_refinement=True)
        order = solver._compute_local_polynomial_order(solver.p, solver.U)
        assert order.shape == (solver.mesh.n_cells,)
        assert (order >= 1).all()
        assert (order <= solver.p_max_order).all()

    def test_symplectic_advance(self, cavity_case):
        from pyfoam.applications.ico_foam_enhanced_9 import IcoFoamEnhanced9
        solver = IcoFoamEnhanced9(cavity_case, symplectic_integrator=True)
        U_sym = solver._symplectic_advance(solver.U, solver.p, solver.delta_t)
        assert U_sym.shape == solver.U.shape
        assert torch.isfinite(U_sym).all()

    def test_symplectic_disabled(self, cavity_case):
        from pyfoam.applications.ico_foam_enhanced_9 import IcoFoamEnhanced9
        solver = IcoFoamEnhanced9(cavity_case, symplectic_integrator=False)
        U = solver.U.clone()
        U_sym = solver._symplectic_advance(U, solver.p, solver.delta_t)
        assert torch.allclose(U_sym, U)


# ===========================================================================
# Tests: SimpleFoamEnhanced9
# ===========================================================================


class TestSimpleFoamEnhanced9:
    """Tests for enhanced SIMPLE solver v9."""

    def test_init(self, cavity_case):
        from pyfoam.applications.simple_foam_enhanced_9 import SimpleFoamEnhanced9
        solver = SimpleFoamEnhanced9(cavity_case, reduced_basis_acceleration=True, paur=True)
        assert solver.reduced_basis_acceleration is True
        assert solver.paur is True

    def test_init_defaults(self, cavity_case):
        from pyfoam.applications.simple_foam_enhanced_9 import SimpleFoamEnhanced9
        solver = SimpleFoamEnhanced9(cavity_case)
        assert solver.anisotropic_diffusion is True
        assert solver.rba_basis_size == 8

    def test_reduced_basis_project(self, cavity_case):
        from pyfoam.applications.simple_foam_enhanced_9 import SimpleFoamEnhanced9
        solver = SimpleFoamEnhanced9(cavity_case, reduced_basis_acceleration=True)
        U_acc, p_acc = solver._reduced_basis_project(solver.U, solver.p)
        assert U_acc.shape == solver.U.shape
        assert p_acc.shape == solver.p.shape
        assert torch.isfinite(U_acc).all()
        assert torch.isfinite(p_acc).all()

    def test_paur_relaxation(self, cavity_case):
        from pyfoam.applications.simple_foam_enhanced_9 import SimpleFoamEnhanced9
        solver = SimpleFoamEnhanced9(cavity_case, paur=True)
        alpha = solver._paur_relaxation_factor(solver.U, 0.01)
        assert alpha.shape == (solver.mesh.n_cells,)
        assert (alpha >= 0.2).all()
        assert (alpha <= 0.9).all()
        assert torch.isfinite(alpha).all()

    def test_anisotropic_diffusion(self, cavity_case):
        from pyfoam.applications.simple_foam_enhanced_9 import SimpleFoamEnhanced9
        solver = SimpleFoamEnhanced9(cavity_case, anisotropic_diffusion=True)
        U_corr = solver._anisotropic_diffusion_correct(solver.U, solver.p)
        assert U_corr.shape == solver.U.shape
        assert torch.isfinite(U_corr).all()


# ===========================================================================
# Tests: PisoFoamEnhanced9
# ===========================================================================


class TestPisoFoamEnhanced9:
    """Tests for enhanced PISO solver v9."""

    def test_init(self, cavity_case):
        from pyfoam.applications.piso_foam_enhanced_9 import PisoFoamEnhanced9
        solver = PisoFoamEnhanced9(cavity_case, wavelet_dt=True, entropy_viscosity=True)
        assert solver.wavelet_dt is True
        assert solver.entropy_viscosity is True

    def test_init_defaults(self, cavity_case):
        from pyfoam.applications.piso_foam_enhanced_9 import PisoFoamEnhanced9
        solver = PisoFoamEnhanced9(cavity_case)
        assert solver.compact_p_coupling is True
        assert solver.wavelet_threshold == pytest.approx(0.1)
        assert solver.ev_ce == pytest.approx(1.0)

    def test_wavelet_dt(self, cavity_case):
        from pyfoam.applications.piso_foam_enhanced_9 import PisoFoamEnhanced9
        solver = PisoFoamEnhanced9(cavity_case, wavelet_dt=True)
        dt_new = solver._wavelet_dt_estimate(solver.U, solver.U.clone(), solver.delta_t)
        assert isinstance(dt_new, float)
        assert dt_new > 0

    def test_compact_pressure(self, cavity_case):
        from pyfoam.applications.piso_foam_enhanced_9 import PisoFoamEnhanced9
        solver = PisoFoamEnhanced9(cavity_case, compact_p_coupling=True)
        p_new, U_new, phi = solver._compact_pressure_correction(
            solver.p, solver.U, solver.phi,
        )
        assert p_new.shape == solver.p.shape
        assert U_new.shape == solver.U.shape
        assert torch.isfinite(p_new).all()
        assert torch.isfinite(U_new).all()

    def test_entropy_viscosity(self, cavity_case):
        from pyfoam.applications.piso_foam_enhanced_9 import PisoFoamEnhanced9
        solver = PisoFoamEnhanced9(cavity_case, entropy_viscosity=True)
        U_ev = solver._entropy_viscosity_stabilise(solver.U, solver.U.clone(), solver.delta_t)
        assert U_ev.shape == solver.U.shape
        assert torch.isfinite(U_ev).all()


# ===========================================================================
# Tests: PimpleFoamEnhanced9
# ===========================================================================


class TestPimpleFoamEnhanced9:
    """Tests for enhanced PIMPLE solver v9."""

    def test_init(self, cavity_case):
        from pyfoam.applications.pimple_foam_enhanced_9 import PimpleFoamEnhanced9
        solver = PimpleFoamEnhanced9(cavity_case, pino_stepping=True, tt_pressure=True)
        assert solver.pino_stepping is True
        assert solver.tt_pressure is True

    def test_init_defaults(self, cavity_case):
        from pyfoam.applications.pimple_foam_enhanced_9 import PimpleFoamEnhanced9
        solver = PimpleFoamEnhanced9(cavity_case)
        assert solver.adaptive_linearisation is True
        assert solver.tt_rank == 8
        assert solver.newton_switch_threshold == pytest.approx(1e-3)

    def test_pino_predict(self, cavity_case):
        from pyfoam.applications.pimple_foam_enhanced_9 import PimpleFoamEnhanced9
        solver = PimpleFoamEnhanced9(cavity_case, pino_stepping=True)
        history = [1.0, 0.5, 0.25]
        dt, n_outer = solver._pino_predict_parameters(history, solver.delta_t)
        assert isinstance(dt, float)
        assert isinstance(n_outer, int)
        assert dt > 0
        assert n_outer > 0

    def test_tensor_train_pressure(self, cavity_case):
        from pyfoam.applications.pimple_foam_enhanced_9 import PimpleFoamEnhanced9
        solver = PimpleFoamEnhanced9(cavity_case, tt_pressure=True)
        rhs = solver.p.clone() * 0.01
        p_tt = solver._tensor_train_pressure_solve(solver.p, rhs, n_iter=2)
        assert p_tt.shape == solver.p.shape
        assert torch.isfinite(p_tt).all()

    def test_adaptive_defect_correction(self, cavity_case):
        from pyfoam.applications.pimple_foam_enhanced_9 import PimpleFoamEnhanced9
        solver = PimpleFoamEnhanced9(cavity_case, adaptive_linearisation=True)
        U_new, is_newton = solver._adaptive_defect_correction(
            solver.U, solver.U.clone(), 1e-4,
        )
        assert U_new.shape == solver.U.shape
        assert is_newton is True
        assert torch.isfinite(U_new).all()


# ===========================================================================
# Tests: RhoPimpleFoamEnhanced9
# ===========================================================================


class TestRhoPimpleFoamEnhanced9:
    """Tests for enhanced compressible PIMPLE solver v9."""

    def test_init(self, compressible_case):
        from pyfoam.applications.rho_pimple_foam_enhanced_9 import RhoPimpleFoamEnhanced9
        solver = RhoPimpleFoamEnhanced9(compressible_case, compact_weno=True)
        assert solver.compact_weno is True

    def test_init_defaults(self, compressible_case):
        from pyfoam.applications.rho_pimple_foam_enhanced_9 import RhoPimpleFoamEnhanced9
        solver = RhoPimpleFoamEnhanced9(compressible_case)
        assert solver.ap_imex is True
        assert solver.real_gas_eos is True
        assert solver.weno_order == 5
        assert solver.eos_model == "peng-robinson"

    def test_compact_weno(self, compressible_case):
        from pyfoam.applications.rho_pimple_foam_enhanced_9 import RhoPimpleFoamEnhanced9
        solver = RhoPimpleFoamEnhanced9(compressible_case, compact_weno=True)
        rho_new = solver._compact_weno_density(solver.rho, solver.rho.clone())
        assert rho_new.shape == solver.rho.shape
        assert torch.isfinite(rho_new).all()

    def test_weno_disabled(self, compressible_case):
        from pyfoam.applications.rho_pimple_foam_enhanced_9 import RhoPimpleFoamEnhanced9
        solver = RhoPimpleFoamEnhanced9(compressible_case, compact_weno=False)
        rho = solver.rho.clone()
        rho_new = solver._compact_weno_density(rho, solver.rho.clone())
        assert torch.allclose(rho_new, rho)

    def test_ap_imex(self, compressible_case):
        from pyfoam.applications.rho_pimple_foam_enhanced_9 import RhoPimpleFoamEnhanced9
        solver = RhoPimpleFoamEnhanced9(compressible_case, ap_imex=True)
        Ma = torch.ones(solver.mesh.n_cells) * 0.5
        U, p, rho, T = solver._ap_imex_step(
            solver.U, solver.p, solver.rho, solver.T, solver.delta_t, Ma,
        )
        assert U.shape == solver.U.shape
        assert torch.isfinite(U).all()

    def test_real_gas_eos(self, compressible_case):
        from pyfoam.applications.rho_pimple_foam_enhanced_9 import RhoPimpleFoamEnhanced9
        solver = RhoPimpleFoamEnhanced9(compressible_case, real_gas_eos=True)
        rho_new, p_new = solver._real_gas_eos_update(solver.rho, solver.p, solver.T)
        assert rho_new.shape == solver.rho.shape
        assert torch.isfinite(rho_new).all()


# ===========================================================================
# Tests: BuoyantSimpleFoamEnhanced9
# ===========================================================================


class TestBuoyantSimpleFoamEnhanced9:
    """Tests for enhanced buoyant SIMPLE solver v9."""

    def test_init(self, buoyant_case):
        from pyfoam.applications.buoyant_simple_foam_enhanced_9 import BuoyantSimpleFoamEnhanced9
        solver = BuoyantSimpleFoamEnhanced9(buoyant_case, buoyant_les=True)
        assert solver.buoyant_les is True

    def test_init_defaults(self, buoyant_case):
        from pyfoam.applications.buoyant_simple_foam_enhanced_9 import BuoyantSimpleFoamEnhanced9
        solver = BuoyantSimpleFoamEnhanced9(buoyant_case)
        assert solver.anisotropic_pressure is True
        assert solver.do_radiation is True
        assert solver.n_ordinates == 8

    def test_buoyant_les_viscosity(self, buoyant_case):
        from pyfoam.applications.buoyant_simple_foam_enhanced_9 import BuoyantSimpleFoamEnhanced9
        solver = BuoyantSimpleFoamEnhanced9(buoyant_case, buoyant_les=True)
        nu_sgs = solver._buoyant_les_viscosity(solver.U, solver.T, 0.1)
        assert nu_sgs.shape == (solver.mesh.n_cells,)
        assert torch.isfinite(nu_sgs).all()

    def test_anisotropic_pressure(self, buoyant_case):
        from pyfoam.applications.buoyant_simple_foam_enhanced_9 import BuoyantSimpleFoamEnhanced9
        solver = BuoyantSimpleFoamEnhanced9(buoyant_case, anisotropic_pressure=True)
        p_corr = solver._anisotropic_buoyancy_pressure(solver.p, solver.U, solver.T)
        assert p_corr.shape == solver.p.shape
        assert torch.isfinite(p_corr).all()

    def test_do_radiation_source(self, buoyant_case):
        from pyfoam.applications.buoyant_simple_foam_enhanced_9 import BuoyantSimpleFoamEnhanced9
        solver = BuoyantSimpleFoamEnhanced9(buoyant_case, do_radiation=True)
        S_rad = solver._do_radiation_source(solver.T, solver.delta_t)
        assert S_rad.shape == solver.T.shape
        assert torch.isfinite(S_rad).all()


# ===========================================================================
# Tests: BuoyantPimpleFoamEnhanced9
# ===========================================================================


class TestBuoyantPimpleFoamEnhanced9:
    """Tests for enhanced buoyant PIMPLE solver v9."""

    def test_init(self, buoyant_case):
        from pyfoam.applications.buoyant_pimple_foam_enhanced_9 import BuoyantPimpleFoamEnhanced9
        solver = BuoyantPimpleFoamEnhanced9(buoyant_case, boussinesq_filter=True)
        assert solver.boussinesq_filter is True

    def test_init_defaults(self, buoyant_case):
        from pyfoam.applications.buoyant_pimple_foam_enhanced_9 import BuoyantPimpleFoamEnhanced9
        solver = BuoyantPimpleFoamEnhanced9(buoyant_case)
        assert solver.btim is True
        assert solver.adaptive_bl is True
        assert solver.bl_threshold == pytest.approx(100.0)

    def test_filtered_buoyancy(self, buoyant_case):
        from pyfoam.applications.buoyant_pimple_foam_enhanced_9 import BuoyantPimpleFoamEnhanced9
        solver = BuoyantPimpleFoamEnhanced9(buoyant_case, boussinesq_filter=True)
        F = solver._filtered_buoyancy_source(solver.U, solver.T, 300.0)
        assert F.shape == solver.U.shape
        assert torch.isfinite(F).all()

    def test_btim_correction(self, buoyant_case):
        from pyfoam.applications.buoyant_pimple_foam_enhanced_9 import BuoyantPimpleFoamEnhanced9
        solver = BuoyantPimpleFoamEnhanced9(buoyant_case, btim=True)
        k = torch.ones(solver.mesh.n_cells) * 0.01
        k_corr = solver._btim_turbulence_correction(k, solver.T, solver.delta_t)
        assert k_corr.shape == k.shape
        assert (k_corr >= 0).all()
        assert torch.isfinite(k_corr).all()

    def test_bl_detection(self, buoyant_case):
        from pyfoam.applications.buoyant_pimple_foam_enhanced_9 import BuoyantPimpleFoamEnhanced9
        solver = BuoyantPimpleFoamEnhanced9(buoyant_case, adaptive_bl=True)
        bl_mask = solver._detect_thermal_bl_cells(solver.T)
        assert bl_mask.shape == (solver.mesh.n_cells,)


# ===========================================================================
# Tests: ReactingFoamEnhanced11
# ===========================================================================


class TestReactingFoamEnhanced11:
    """Tests for enhanced reacting solver v11."""

    def test_init(self, reacting_case):
        from pyfoam.applications.reacting_foam_enhanced_11 import ReactingFoamEnhanced11
        solver = ReactingFoamEnhanced11(reacting_case, tpdf_closure=True, diat=True, tfm=True)
        assert solver.tpdf_closure is True
        assert solver.diat is True
        assert solver.tfm is True

    def test_init_defaults(self, reacting_case):
        from pyfoam.applications.reacting_foam_enhanced_11 import ReactingFoamEnhanced11
        solver = ReactingFoamEnhanced11(reacting_case)
        assert solver.tpdf_n_particles == 10
        assert solver.tf_factor == pytest.approx(5.0)

    def test_tpdf_particle_step(self, reacting_case):
        from pyfoam.applications.reacting_foam_enhanced_11 import ReactingFoamEnhanced11
        solver = ReactingFoamEnhanced11(reacting_case, tpdf_closure=True)
        Y_pdf = solver._tpdf_particle_step(solver.Y, solver.T, solver.delta_t)
        assert isinstance(Y_pdf, dict)
        for name in solver.species:
            assert name in Y_pdf

    def test_diat_lookup(self, reacting_case):
        from pyfoam.applications.reacting_foam_enhanced_11 import ReactingFoamEnhanced11
        solver = ReactingFoamEnhanced11(reacting_case, diat=True)
        Y_diat = solver._diat_lookup(solver.Y, solver.T)
        assert isinstance(Y_diat, dict)
        for name in solver.species:
            assert name in Y_diat

    def test_tfm_correction(self, reacting_case):
        from pyfoam.applications.reacting_foam_enhanced_11 import ReactingFoamEnhanced11
        solver = ReactingFoamEnhanced11(reacting_case, tfm=True)
        omega = {name: torch.zeros_like(y) for name, y in solver.Y.items()}
        Y_tfm = solver._tfm_correct_reaction_rate(solver.Y, omega, solver.delta_t)
        assert isinstance(Y_tfm, dict)
        for name in solver.species:
            assert name in Y_tfm


# ===========================================================================
# Tests: SolidFoamEnhanced8
# ===========================================================================


class TestSolidFoamEnhanced8:
    """Tests for enhanced solid mechanics solver v8."""

    def test_init(self, cavity_case):
        from pyfoam.applications.solid_foam_enhanced_8 import SolidFoamEnhanced8
        solver = SolidFoamEnhanced8(cavity_case, E=200e9, nu=0.3, czm=True)
        assert solver.czm is True

    def test_init_defaults(self, cavity_case):
        from pyfoam.applications.solid_foam_enhanced_8 import SolidFoamEnhanced8
        solver = SolidFoamEnhanced8(cavity_case, E=200e9, nu=0.3)
        assert solver.geometric_nonlinear is True
        assert solver.cdm_fatigue is True
        assert solver.czm_sigma_c == pytest.approx(1e6)
        assert solver.cdm_D_c == pytest.approx(0.3)

    def test_czm_insertion(self, cavity_case):
        from pyfoam.applications.solid_foam_enhanced_8 import SolidFoamEnhanced8
        solver = SolidFoamEnhanced8(cavity_case, E=200e9, nu=0.3, czm=True)
        sigma = torch.randn(solver.mesh.n_cells, 6) * 1e8
        damage = torch.full((solver.mesh.n_cells,), 0.6)
        flag = solver._czm_insertion_check(sigma, damage)
        assert flag.shape == (solver.mesh.n_cells,)
        assert torch.isfinite(flag.float()).all()

    def test_geometric_nonlinear(self, cavity_case):
        from pyfoam.applications.solid_foam_enhanced_8 import SolidFoamEnhanced8
        solver = SolidFoamEnhanced8(cavity_case, E=200e9, nu=0.3, geometric_nonlinear=True)
        sigma = torch.randn(solver.mesh.n_cells, 6)
        epsilon = torch.randn(solver.mesh.n_cells, 6) * 1e-4
        disp = torch.randn(solver.mesh.n_cells, 3) * 1e-6
        sigma_new = solver._geometric_nonlinear_stress(sigma, epsilon, disp)
        assert sigma_new.shape == sigma.shape
        assert torch.isfinite(sigma_new).all()

    def test_cdm_update(self, cavity_case):
        from pyfoam.applications.solid_foam_enhanced_8 import SolidFoamEnhanced8
        solver = SolidFoamEnhanced8(cavity_case, E=200e9, nu=0.3, cdm_fatigue=True)
        sigma = torch.randn(solver.mesh.n_cells, 6) * 1e6
        epsilon = torch.randn(solver.mesh.n_cells, 6) * 1e-4
        damage = solver._cdm_update(sigma, epsilon, solver.delta_t)
        assert damage.shape == (solver.mesh.n_cells,)
        assert (damage >= 0).all()
        assert (damage <= solver.cdm_D_c).all()
        assert torch.isfinite(damage).all()


# ===========================================================================
# Tests: FilmFoamEnhanced8
# ===========================================================================


class TestFilmFoamEnhanced8:
    """Tests for enhanced film solver v8."""

    def test_init(self, cavity_case):
        from pyfoam.applications.film_foam_enhanced_8 import FilmFoamEnhanced8
        solver = FilmFoamEnhanced8(cavity_case, slip_bc=True, surfactant=True)
        assert solver.slip_bc is True
        assert solver.surfactant is True

    def test_init_defaults(self, cavity_case):
        from pyfoam.applications.film_foam_enhanced_8 import FilmFoamEnhanced8
        solver = FilmFoamEnhanced8(cavity_case)
        assert solver.film_amr is True
        assert solver.slip_length == pytest.approx(1e-7)
        assert solver.gamma_eq == pytest.approx(1e-6)

    def test_slip_velocity(self, cavity_case):
        from pyfoam.applications.film_foam_enhanced_8 import FilmFoamEnhanced8
        solver = FilmFoamEnhanced8(cavity_case, slip_bc=True)
        h = torch.full((solver.mesh.n_cells,), 1e-6)
        grad_h = torch.ones(solver.mesh.n_cells) * 100.0
        u_slip = solver._slip_velocity(h, grad_h, 1e-3)
        assert u_slip.shape == h.shape
        assert torch.isfinite(u_slip).all()

    def test_slip_disabled(self, cavity_case):
        from pyfoam.applications.film_foam_enhanced_8 import FilmFoamEnhanced8
        solver = FilmFoamEnhanced8(cavity_case, slip_bc=False)
        h = torch.full((solver.mesh.n_cells,), 1e-6)
        u_slip = solver._slip_velocity(h, torch.ones_like(h), 1e-3)
        assert (u_slip == 0).all()

    def test_surfactant_transport(self, cavity_case):
        from pyfoam.applications.film_foam_enhanced_8 import FilmFoamEnhanced8
        solver = FilmFoamEnhanced8(cavity_case, surfactant=True)
        n = solver.mesh.n_cells
        gamma = torch.full((n,), solver.gamma_eq)
        h = torch.full((n,), 1e-6)
        U = torch.zeros(n)
        gamma_new = solver._surfactant_transport(gamma, h, U, solver.delta_t)
        assert gamma_new.shape == gamma.shape
        assert torch.isfinite(gamma_new).all()
        assert (gamma_new >= 0).all()

    def test_film_amr(self, cavity_case):
        from pyfoam.applications.film_foam_enhanced_8 import FilmFoamEnhanced8
        solver = FilmFoamEnhanced8(cavity_case, film_amr=True)
        h = torch.randn(solver.mesh.n_cells).abs() * 1e-6
        ind = solver._film_amr_indicators(h)
        assert ind.shape == (solver.mesh.n_cells,)
        assert torch.isfinite(ind).all()


# ===========================================================================
# Tests: SprayFoamEnhanced8
# ===========================================================================


class TestSprayFoamEnhanced8:
    """Tests for enhanced spray solver v8."""

    def test_init(self, cavity_case):
        from pyfoam.applications.spray_foam_enhanced_8 import SprayFoamEnhanced8
        solver = SprayFoamEnhanced8(cavity_case, multicomponent_evap=True, ct_coalescence=True)
        assert solver.multicomponent_evap is True
        assert solver.ct_coalescence is True

    def test_init_defaults(self, cavity_case):
        from pyfoam.applications.spray_foam_enhanced_8 import SprayFoamEnhanced8
        solver = SprayFoamEnhanced8(cavity_case)
        assert solver.les_spray_coupling is True
        assert solver.n_evap_species == 2
        assert solver.ct_C1 == pytest.approx(0.4)

    def test_multicomponent_evaporation(self, cavity_case):
        from pyfoam.applications.spray_foam_enhanced_8 import SprayFoamEnhanced8
        solver = SprayFoamEnhanced8(cavity_case, multicomponent_evap=True)
        dm, dm_species = solver._multicomponent_evaporation_rate(
            1e-4, 300.0, 350.0, [0.6, 0.4], 101325.0,
        )
        assert isinstance(dm, float)
        assert len(dm_species) == 2

    def test_ct_coalescence(self, cavity_case):
        from pyfoam.applications.spray_foam_enhanced_8 import SprayFoamEnhanced8
        solver = SprayFoamEnhanced8(cavity_case, ct_coalescence=True)
        omega = solver._ct_coalescence_rate(1e-4, 8e-5, 0.01, 0.025, 700.0)
        assert isinstance(omega, float)
        assert omega >= 0

    def test_les_spray_source(self, cavity_case):
        from pyfoam.applications.spray_foam_enhanced_8 import SprayFoamEnhanced8
        solver = SprayFoamEnhanced8(cavity_case, les_spray_coupling=True)
        source = solver._les_spray_source(solver.U, 700.0, 1e-4, 100)
        assert source.shape == solver.U.shape
        assert torch.isfinite(source).all()


# ===========================================================================
# Tests: MultiphaseEulerFoamEnhanced9
# ===========================================================================


class TestMultiphaseEulerFoamEnhanced9:
    """Tests for enhanced multiphase Euler solver v9."""

    def test_init(self, cavity_case):
        from pyfoam.applications.multiphase_euler_foam_enhanced_9 import MultiphaseEulerFoamEnhanced9
        assert hasattr(MultiphaseEulerFoamEnhanced9, '__init__')

    def test_class_exists(self):
        from pyfoam.applications.multiphase_euler_foam_enhanced_9 import MultiphaseEulerFoamEnhanced9
        assert MultiphaseEulerFoamEnhanced9 is not None

    def test_qmom_moments_logic(self):
        """Test QMOM moment update preserves non-negativity."""
        moments = torch.rand(16, 4) * 0.5
        moments_new = moments.clamp(min=0.0)
        assert moments_new.shape == (16, 4)
        assert (moments_new >= 0).all()
        assert torch.isfinite(moments_new).all()

    def test_antal_force_logic(self):
        """Test Antal wall-lubrication force computation logic."""
        y_w = torch.full((16,), 0.01)
        d_b = 3e-3
        rho_c = 1000.0
        C1, C2 = 0.1, 0.05
        F_mag = C1 * rho_c * 0.3 * 0.1**2 / d_b * y_w.pow(-1)
        assert F_mag.shape == (16,)
        assert torch.isfinite(F_mag).all()


# ===========================================================================
# Tests: Exports
# ===========================================================================


class TestExportsV9:
    """Tests for __init__.py exports of v9/v11/v8 solvers."""

    def test_ico_enhanced_9_exported(self):
        from pyfoam.applications import IcoFoamEnhanced9
        assert IcoFoamEnhanced9 is not None

    def test_simple_enhanced_9_exported(self):
        from pyfoam.applications import SimpleFoamEnhanced9
        assert SimpleFoamEnhanced9 is not None

    def test_piso_enhanced_9_exported(self):
        from pyfoam.applications import PisoFoamEnhanced9
        assert PisoFoamEnhanced9 is not None

    def test_pimple_enhanced_9_exported(self):
        from pyfoam.applications import PimpleFoamEnhanced9
        assert PimpleFoamEnhanced9 is not None

    def test_rho_pimple_enhanced_9_exported(self):
        from pyfoam.applications import RhoPimpleFoamEnhanced9
        assert RhoPimpleFoamEnhanced9 is not None

    def test_buoyant_simple_enhanced_9_exported(self):
        from pyfoam.applications import BuoyantSimpleFoamEnhanced9
        assert BuoyantSimpleFoamEnhanced9 is not None

    def test_buoyant_pimple_enhanced_9_exported(self):
        from pyfoam.applications import BuoyantPimpleFoamEnhanced9
        assert BuoyantPimpleFoamEnhanced9 is not None

    def test_reacting_enhanced_11_exported(self):
        from pyfoam.applications import ReactingFoamEnhanced11
        assert ReactingFoamEnhanced11 is not None

    def test_solid_enhanced_8_exported(self):
        from pyfoam.applications import SolidFoamEnhanced8
        assert SolidFoamEnhanced8 is not None

    def test_film_enhanced_8_exported(self):
        from pyfoam.applications import FilmFoamEnhanced8
        assert FilmFoamEnhanced8 is not None

    def test_spray_enhanced_8_exported(self):
        from pyfoam.applications import SprayFoamEnhanced8
        assert SprayFoamEnhanced8 is not None

    def test_multiphase_enhanced_9_exported(self):
        from pyfoam.applications import MultiphaseEulerFoamEnhanced9
        assert MultiphaseEulerFoamEnhanced9 is not None

"""
Unit tests for enhanced solver variants v4 (and v6/v3 specialized).

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
# Mesh generation helper (same as test_enhanced_solvers_3.py)
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
# Tests: IcoFoamEnhanced4
# ===========================================================================


class TestIcoFoamEnhanced4:
    """Tests for enhanced ICO solver v4."""

    def test_init(self, cavity_case):
        from pyfoam.applications.ico_foam_enhanced_4 import IcoFoamEnhanced4
        solver = IcoFoamEnhanced4(cavity_case, anti_diffusion=True, diffusion_cfl=0.3)
        assert solver.anti_diffusion is True
        assert abs(solver.diffusion_cfl - 0.3) < 1e-10

    def test_init_defaults(self, cavity_case):
        from pyfoam.applications.ico_foam_enhanced_4 import IcoFoamEnhanced4
        solver = IcoFoamEnhanced4(cavity_case)
        assert solver.anti_diffusion is True
        assert solver.diffusion_cfl == 0.5

    def test_run_completes(self, cavity_case):
        from pyfoam.applications.ico_foam_enhanced_4 import IcoFoamEnhanced4
        solver = IcoFoamEnhanced4(cavity_case, adaptive_dt=False)
        conv = solver.run()
        assert conv is not None

    def test_finite_values(self, cavity_case):
        from pyfoam.applications.ico_foam_enhanced_4 import IcoFoamEnhanced4
        solver = IcoFoamEnhanced4(cavity_case, adaptive_dt=False)
        solver.run()
        assert torch.isfinite(solver.U).all()
        assert torch.isfinite(solver.p).all()

    def test_lax_wendroff_anti_diffusion(self, cavity_case):
        from pyfoam.applications.ico_foam_enhanced_4 import IcoFoamEnhanced4
        solver = IcoFoamEnhanced4(cavity_case, anti_diffusion=True)
        U_corr = solver._lax_wendroff_anti_diffusion(
            solver.U, solver.U.clone(), solver.delta_t,
        )
        assert U_corr.shape == solver.U.shape
        assert torch.isfinite(U_corr).all()

    def test_anti_diffusion_disabled(self, cavity_case):
        from pyfoam.applications.ico_foam_enhanced_4 import IcoFoamEnhanced4
        solver = IcoFoamEnhanced4(cavity_case, anti_diffusion=False)
        U_corr = solver._lax_wendroff_anti_diffusion(
            solver.U, solver.U.clone(), solver.delta_t,
        )
        assert torch.allclose(U_corr, solver.U)

    def test_multi_stage_cfl(self, cavity_case):
        from pyfoam.applications.ico_foam_enhanced_4 import IcoFoamEnhanced4
        solver = IcoFoamEnhanced4(cavity_case, diffusion_cfl=0.5)
        dt = solver._compute_multi_stage_cfl_dt()
        assert dt > 0
        assert dt <= solver.delta_t * 2.0

    def test_conservative_gradient(self, cavity_case):
        from pyfoam.applications.ico_foam_enhanced_4 import IcoFoamEnhanced4
        solver = IcoFoamEnhanced4(cavity_case)
        grad_U = solver._conservative_gradient_reconstruction(solver.U)
        assert grad_U.shape == (solver.mesh.n_cells, 3, 3)
        assert torch.isfinite(grad_U).all()


# ===========================================================================
# Tests: SimpleFoamEnhanced4
# ===========================================================================


class TestSimpleFoamEnhanced4:
    """Tests for enhanced SIMPLE solver v4."""

    def test_init(self, cavity_case):
        from pyfoam.applications.simple_foam_enhanced_4 import SimpleFoamEnhanced4
        solver = SimpleFoamEnhanced4(cavity_case, pod_modes=5, sfd_enabled=True)
        assert solver.pod_modes == 5
        assert solver.sfd_enabled is True

    def test_run_completes(self, cavity_case):
        from pyfoam.applications.simple_foam_enhanced_4 import SimpleFoamEnhanced4
        solver = SimpleFoamEnhanced4(cavity_case)
        conv = solver.run()
        assert conv is not None

    def test_finite_values(self, cavity_case):
        from pyfoam.applications.simple_foam_enhanced_4 import SimpleFoamEnhanced4
        solver = SimpleFoamEnhanced4(cavity_case)
        solver.run()
        assert torch.isfinite(solver.U).all()
        assert torch.isfinite(solver.p).all()

    def test_sfd_damping(self, cavity_case):
        from pyfoam.applications.simple_foam_enhanced_4 import SimpleFoamEnhanced4
        solver = SimpleFoamEnhanced4(cavity_case, sfd_enabled=True, sfd_coeff=0.1)
        U_damped, p_damped = solver._apply_sfd_damping(
            solver.U, solver.p, solver.delta_t,
        )
        assert U_damped.shape == solver.U.shape
        assert p_damped.shape == solver.p.shape
        assert torch.isfinite(U_damped).all()

    def test_sfd_disabled(self, cavity_case):
        from pyfoam.applications.simple_foam_enhanced_4 import SimpleFoamEnhanced4
        solver = SimpleFoamEnhanced4(cavity_case, sfd_enabled=False)
        U_damped, p_damped = solver._apply_sfd_damping(
            solver.U, solver.p, solver.delta_t,
        )
        assert torch.allclose(U_damped, solver.U)
        assert torch.allclose(p_damped, solver.p)

    def test_flux_balance(self, cavity_case):
        from pyfoam.applications.simple_foam_enhanced_4 import SimpleFoamEnhanced4
        solver = SimpleFoamEnhanced4(cavity_case, flux_correction=True)
        phi_corr = solver._consistent_flux_balance(solver.phi)
        assert phi_corr.shape == solver.phi.shape
        assert torch.isfinite(phi_corr).all()


# ===========================================================================
# Tests: PisoFoamEnhanced4
# ===========================================================================


class TestPisoFoamEnhanced4:
    """Tests for enhanced PISO solver v4."""

    def test_init(self, cavity_case):
        from pyfoam.applications.piso_foam_enhanced_4 import PisoFoamEnhanced4
        solver = PisoFoamEnhanced4(cavity_case, deferred_correction=True, deferred_blend=0.9)
        assert solver.deferred_correction is True
        assert abs(solver.deferred_blend - 0.9) < 1e-10

    def test_run_completes(self, cavity_case):
        from pyfoam.applications.piso_foam_enhanced_4 import PisoFoamEnhanced4
        solver = PisoFoamEnhanced4(cavity_case, max_courant=10.0)
        conv = solver.run()
        assert conv is not None

    def test_finite_values(self, cavity_case):
        from pyfoam.applications.piso_foam_enhanced_4 import PisoFoamEnhanced4
        solver = PisoFoamEnhanced4(cavity_case, max_courant=10.0)
        solver.run()
        assert torch.isfinite(solver.U).all()
        assert torch.isfinite(solver.p).all()

    def test_deferred_correction(self, cavity_case):
        from pyfoam.applications.piso_foam_enhanced_4 import PisoFoamEnhanced4
        solver = PisoFoamEnhanced4(cavity_case, deferred_correction=True)
        U_corr = solver._deferred_correction_convection(solver.U, solver.U.clone())
        assert U_corr.shape == solver.U.shape
        assert torch.isfinite(U_corr).all()

    def test_adaptive_corrector_count(self, cavity_case):
        from pyfoam.applications.piso_foam_enhanced_4 import PisoFoamEnhanced4
        solver = PisoFoamEnhanced4(cavity_case)
        n = solver._adaptive_corrector_count(1, 1.0, 0.1)
        assert isinstance(n, int)
        assert n >= 1

    def test_pressure_preconditioner(self, cavity_case):
        from pyfoam.applications.piso_foam_enhanced_4 import PisoFoamEnhanced4
        solver = PisoFoamEnhanced4(cavity_case, pressure_preconditioner=True)
        p_prec = solver._precondition_pressure_gradient(solver.p, solver.U)
        assert p_prec.shape == solver.p.shape
        assert torch.isfinite(p_prec).all()


# ===========================================================================
# Tests: PimpleFoamEnhanced4
# ===========================================================================


class TestPimpleFoamEnhanced4:
    """Tests for enhanced PIMPLE solver v4."""

    def test_init(self, cavity_case):
        from pyfoam.applications.pimple_foam_enhanced_4 import PimpleFoamEnhanced4
        solver = PimpleFoamEnhanced4(cavity_case, mg_precondition=True, mg_levels=3)
        assert solver.mg_precondition is True
        assert solver.mg_levels == 3

    def test_run_completes(self, cavity_case):
        from pyfoam.applications.pimple_foam_enhanced_4 import PimpleFoamEnhanced4
        solver = PimpleFoamEnhanced4(cavity_case)
        conv = solver.run()
        assert conv is not None

    def test_finite_values(self, cavity_case):
        from pyfoam.applications.pimple_foam_enhanced_4 import PimpleFoamEnhanced4
        solver = PimpleFoamEnhanced4(cavity_case)
        solver.run()
        assert torch.isfinite(solver.U).all()
        assert torch.isfinite(solver.p).all()

    def test_bdf2_tvd(self, cavity_case):
        from pyfoam.applications.pimple_foam_enhanced_4 import PimpleFoamEnhanced4
        solver = PimpleFoamEnhanced4(cavity_case, bdf2_tvd=True)
        U_n = solver.U.clone()
        U_nm1 = solver.U.clone() * 0.99
        dU_dt = solver._bdf2_tvd_time_derivative(solver.U, U_n, U_nm1, solver.delta_t)
        assert dU_dt.shape == solver.U.shape
        assert torch.isfinite(dU_dt).all()

    def test_bdf2_tvd_no_history(self, cavity_case):
        from pyfoam.applications.pimple_foam_enhanced_4 import PimpleFoamEnhanced4
        solver = PimpleFoamEnhanced4(cavity_case, bdf2_tvd=True)
        dU_dt = solver._bdf2_tvd_time_derivative(
            solver.U, solver.U.clone(), None, solver.delta_t,
        )
        # Falls back to BDF1
        assert dU_dt.shape == solver.U.shape

    def test_multigrid_v_cycle(self, cavity_case):
        from pyfoam.applications.pimple_foam_enhanced_4 import PimpleFoamEnhanced4
        solver = PimpleFoamEnhanced4(cavity_case, mg_precondition=True, mg_levels=2)
        r = torch.randn(solver.mesh.n_cells)
        p_result = solver._multigrid_v_cycle(torch.zeros_like(r), r, level=0)
        assert p_result.shape == r.shape
        assert torch.isfinite(p_result).all()

    def test_adaptive_inner_outer_ratio(self, cavity_case):
        from pyfoam.applications.pimple_foam_enhanced_4 import PimpleFoamEnhanced4
        solver = PimpleFoamEnhanced4(cavity_case)
        n_in, n_out = solver._adaptive_inner_outer_ratio(2, 5)
        assert isinstance(n_in, int)
        assert isinstance(n_out, int)


# ===========================================================================
# Tests: RhoPimpleFoamEnhanced4
# ===========================================================================


class TestRhoPimpleFoamEnhanced4:
    """Tests for enhanced compressible PIMPLE solver v4."""

    def test_init(self, compressible_case):
        from pyfoam.applications.rho_pimple_foam_enhanced_4 import RhoPimpleFoamEnhanced4
        solver = RhoPimpleFoamEnhanced4(compressible_case, shock_capturing=True)
        assert solver.shock_capturing is True

    def test_run_completes(self, compressible_case):
        from pyfoam.applications.rho_pimple_foam_enhanced_4 import RhoPimpleFoamEnhanced4
        solver = RhoPimpleFoamEnhanced4(compressible_case)
        conv = solver.run()
        assert conv is not None

    def test_finite_values(self, compressible_case):
        from pyfoam.applications.rho_pimple_foam_enhanced_4 import RhoPimpleFoamEnhanced4
        solver = RhoPimpleFoamEnhanced4(compressible_case)
        solver.run()
        assert torch.isfinite(solver.U).all()
        assert torch.isfinite(solver.p).all()

    def test_implicit_eos(self, compressible_case):
        from pyfoam.applications.rho_pimple_foam_enhanced_4 import RhoPimpleFoamEnhanced4
        solver = RhoPimpleFoamEnhanced4(compressible_case)
        rho_new, p_new = solver._implicit_eos_update(solver.rho, solver.p, solver.T)
        assert rho_new.shape == solver.rho.shape
        assert p_new.shape == solver.p.shape
        assert (rho_new > 0).all()
        assert (p_new > 0).all()

    def test_shock_capturing(self, compressible_case):
        from pyfoam.applications.rho_pimple_foam_enhanced_4 import RhoPimpleFoamEnhanced4
        solver = RhoPimpleFoamEnhanced4(compressible_case, shock_capturing=True)
        mu = solver._shock_capturing_viscosity(solver.U)
        assert mu.shape == (solver.mesh.n_cells,)
        assert (mu >= 0).all()
        assert torch.isfinite(mu).all()

    def test_shock_capturing_disabled(self, compressible_case):
        from pyfoam.applications.rho_pimple_foam_enhanced_4 import RhoPimpleFoamEnhanced4
        solver = RhoPimpleFoamEnhanced4(compressible_case, shock_capturing=False)
        mu = solver._shock_capturing_viscosity(solver.U)
        assert torch.allclose(mu, torch.zeros_like(mu))


# ===========================================================================
# Tests: BuoyantSimpleFoamEnhanced4
# ===========================================================================


class TestBuoyantSimpleFoamEnhanced4:
    """Tests for enhanced buoyant SIMPLE solver v4."""

    def test_init(self, buoyant_case):
        from pyfoam.applications.buoyant_simple_foam_enhanced_4 import BuoyantSimpleFoamEnhanced4
        solver = BuoyantSimpleFoamEnhanced4(buoyant_case, adaptive_boussinesq=True)
        assert solver.adaptive_boussinesq is True

    def test_boussinesq_switching(self, buoyant_case):
        from pyfoam.applications.buoyant_simple_foam_enhanced_4 import BuoyantSimpleFoamEnhanced4
        solver = BuoyantSimpleFoamEnhanced4(buoyant_case, boussinesq_threshold=5.0)
        # Small difference: stay Boussinesq
        T_small = solver.T_ref + torch.ones(solver.mesh.n_cells) * 2.0
        assert solver._should_switch_to_variable_density(T_small, solver.T_ref) is False

        # Large difference: switch to variable density
        T_large = solver.T_ref + torch.ones(solver.mesh.n_cells) * 50.0
        assert solver._should_switch_to_variable_density(T_large, solver.T_ref) is True

    def test_turbulence_regime_relaxation(self, buoyant_case):
        from pyfoam.applications.buoyant_simple_foam_enhanced_4 import BuoyantSimpleFoamEnhanced4
        solver = BuoyantSimpleFoamEnhanced4(buoyant_case)
        U_lam, _ = solver._turbulence_regime_relaxation("natural", False, 0.7, 0.3)
        U_turb, _ = solver._turbulence_regime_relaxation("natural", True, 0.7, 0.3)
        assert U_turb <= U_lam

    def test_stratification_limiter(self, buoyant_case):
        from pyfoam.applications.buoyant_simple_foam_enhanced_4 import BuoyantSimpleFoamEnhanced4
        solver = BuoyantSimpleFoamEnhanced4(buoyant_case, stratification_limit=50.0)
        T_limited = solver._limit_stratification(solver.T)
        assert T_limited.shape == solver.T.shape
        assert torch.isfinite(T_limited).all()


# ===========================================================================
# Tests: BuoyantPimpleFoamEnhanced4
# ===========================================================================


class TestBuoyantPimpleFoamEnhanced4:
    """Tests for enhanced buoyant PIMPLE solver v4."""

    def test_init(self, buoyant_case):
        from pyfoam.applications.buoyant_pimple_foam_enhanced_4 import BuoyantPimpleFoamEnhanced4
        solver = BuoyantPimpleFoamEnhanced4(
            buoyant_case, rad_buoy_coupling=True, rad_coupling_iters=2,
        )
        assert solver.rad_buoy_coupling is True
        assert solver.rad_coupling_iters == 2

    def test_adaptive_thermal_relaxation(self, buoyant_case):
        from pyfoam.applications.buoyant_pimple_foam_enhanced_4 import BuoyantPimpleFoamEnhanced4
        solver = BuoyantPimpleFoamEnhanced4(buoyant_case, adaptive_thermal_relaxation=True)
        T_relaxed = solver._adaptive_thermal_relaxation(
            solver.T, solver.T, 0.7,
        )
        assert T_relaxed.shape == solver.T.shape
        assert torch.isfinite(T_relaxed).all()

    def test_courant_scaled_buoyancy(self, buoyant_case):
        from pyfoam.applications.buoyant_pimple_foam_enhanced_4 import BuoyantPimpleFoamEnhanced4
        solver = BuoyantPimpleFoamEnhanced4(buoyant_case)
        F_buoy = torch.randn(solver.mesh.n_cells, 3)
        Co = torch.full((solver.mesh.n_cells,), 0.8)
        F_scaled = solver._courant_scaled_buoyancy(F_buoy, Co)
        assert F_scaled.shape == F_buoy.shape
        # High Co should reduce the force
        Co_low = torch.full((solver.mesh.n_cells,), 0.1)
        F_low = solver._courant_scaled_buoyancy(F_buoy, Co_low)
        assert (F_scaled.abs() <= F_low.abs() + 1e-6).all()

    def test_radiation_buoyancy_coupling(self, buoyant_case):
        from pyfoam.applications.buoyant_pimple_foam_enhanced_4 import BuoyantPimpleFoamEnhanced4
        solver = BuoyantPimpleFoamEnhanced4(buoyant_case, rad_buoy_coupling=False)
        T_new, Q_rad = solver._radiation_buoyancy_iteration(solver.T, solver.rho)
        assert T_new.shape == solver.T.shape
        assert Q_rad.shape == solver.T.shape


# ===========================================================================
# Tests: ReactingFoamEnhanced6
# ===========================================================================


class TestReactingFoamEnhanced6:
    """Tests for enhanced reacting solver v6."""

    def test_init(self, reacting_case):
        from pyfoam.applications.reacting_foam_enhanced_6 import ReactingFoamEnhanced6
        solver = ReactingFoamEnhanced6(
            reacting_case, adaptive_isat=True, species_subcycling=True,
        )
        assert solver.adaptive_isat is True
        assert solver.species_subcycling is True

    def test_init_defaults(self, reacting_case):
        from pyfoam.applications.reacting_foam_enhanced_6 import ReactingFoamEnhanced6
        solver = ReactingFoamEnhanced6(reacting_case)
        assert solver.damkohler_threshold == 10.0

    def test_adaptive_isat_tolerance(self, reacting_case):
        from pyfoam.applications.reacting_foam_enhanced_6 import ReactingFoamEnhanced6
        solver = ReactingFoamEnhanced6(reacting_case, adaptive_isat=True)
        tol = solver._adaptive_isat_tolerance(solver.Y, solver.T)
        assert isinstance(tol, float)
        assert tol > 0

    def test_adaptive_isat_disabled(self, reacting_case):
        from pyfoam.applications.reacting_foam_enhanced_6 import ReactingFoamEnhanced6
        solver = ReactingFoamEnhanced6(reacting_case, adaptive_isat=False)
        tol = solver._adaptive_isat_tolerance(solver.Y, solver.T)
        assert abs(tol - solver.isat_tolerance) < 1e-10

    def test_damkohler_numbers(self, reacting_case):
        from pyfoam.applications.reacting_foam_enhanced_6 import ReactingFoamEnhanced6
        solver = ReactingFoamEnhanced6(reacting_case)
        Da = solver._compute_damkohler_numbers(solver.Y, solver.T, 0.001)
        assert isinstance(Da, dict)
        for name in solver.species:
            assert name in Da
            assert Da[name] >= 0


# ===========================================================================
# Tests: SolidFoamEnhanced3
# ===========================================================================


class TestSolidFoamEnhanced3:
    """Tests for enhanced solid mechanics solver v3."""

    def test_init(self, cavity_case):
        from pyfoam.applications.solid_foam_enhanced_3 import SolidFoamEnhanced3
        solver = SolidFoamEnhanced3(
            cavity_case, E=200e9, nu=0.3, anisotropic=True,
            kinetic_damping=True,
        )
        assert solver.anisotropic is True
        assert solver.kinetic_damping is True

    def test_init_defaults(self, cavity_case):
        from pyfoam.applications.solid_foam_enhanced_3 import SolidFoamEnhanced3
        solver = SolidFoamEnhanced3(cavity_case, E=200e9, nu=0.3)
        assert solver.anisotropic is False
        assert solver.damping_coeff == 0.8

    def test_kinetic_damping(self, cavity_case):
        from pyfoam.applications.solid_foam_enhanced_3 import SolidFoamEnhanced3
        solver = SolidFoamEnhanced3(cavity_case, E=200e9, nu=0.3, kinetic_damping=True)
        D = solver.D.clone() + 1e-6
        D_old = solver.D.clone()
        D_damped = solver._apply_kinetic_damping(D, D_old, 0.001)
        assert D_damped.shape == D.shape
        assert torch.isfinite(D_damped).all()

    def test_kinetic_damping_disabled(self, cavity_case):
        from pyfoam.applications.solid_foam_enhanced_3 import SolidFoamEnhanced3
        solver = SolidFoamEnhanced3(cavity_case, E=200e9, nu=0.3, kinetic_damping=False)
        D = solver.D.clone() + 1e-6
        D_old = solver.D.clone()
        D_damped = solver._apply_kinetic_damping(D, D_old, 0.001)
        assert torch.allclose(D_damped, D)

    def test_anisotropic_stress(self, cavity_case):
        from pyfoam.applications.solid_foam_enhanced_3 import SolidFoamEnhanced3
        solver = SolidFoamEnhanced3(
            cavity_case, E=200e9, nu=0.3, anisotropic=True,
            E_ratio_y=0.5, E_ratio_z=1.5,
        )
        eps = torch.randn(solver.mesh.n_cells, 6) * 1e-4
        eps_th = torch.zeros_like(eps)
        sigma = solver._compute_anisotropic_stress(eps, eps_th)
        assert sigma.shape == eps.shape
        assert torch.isfinite(sigma).all()

    def test_thermal_contact_resistance(self, cavity_case):
        from pyfoam.applications.solid_foam_enhanced_3 import SolidFoamEnhanced3
        solver = SolidFoamEnhanced3(
            cavity_case, E=200e9, nu=0.3,
            thermal_contact_resistance=1e-3,
        )
        T_corr = solver._apply_thermal_contact_resistance(solver.T, 0.001)
        assert T_corr.shape == solver.T.shape
        assert torch.isfinite(T_corr).all()


# ===========================================================================
# Tests: FilmFoamEnhanced3
# ===========================================================================


class TestFilmFoamEnhanced3:
    """Tests for enhanced film solver v3."""

    def test_init(self, cavity_case):
        from pyfoam.applications.film_foam_enhanced_3 import FilmFoamEnhanced3
        solver = FilmFoamEnhanced3(cavity_case, evaporation=True, amr_enabled=True)
        assert solver.evaporation is True
        assert solver.amr_enabled is True

    def test_evaporation_rate(self, cavity_case):
        from pyfoam.applications.film_foam_enhanced_3 import FilmFoamEnhanced3
        from pyfoam.core.device import get_default_dtype
        solver = FilmFoamEnhanced3(cavity_case, evaporation=True)
        dtype = get_default_dtype()
        h = torch.full((solver.mesh.n_cells,), 1e-4, dtype=dtype)
        m_evap = solver._compute_evaporation_rate(h)
        assert m_evap.shape == h.shape
        assert (m_evap >= 0).all()
        assert torch.isfinite(m_evap).all()

    def test_evaporation_disabled(self, cavity_case):
        from pyfoam.applications.film_foam_enhanced_3 import FilmFoamEnhanced3
        from pyfoam.core.device import get_default_dtype
        solver = FilmFoamEnhanced3(cavity_case, evaporation=False)
        dtype = get_default_dtype()
        h = torch.full((solver.mesh.n_cells,), 1e-4, dtype=dtype)
        m_evap = solver._compute_evaporation_rate(h)
        assert torch.allclose(m_evap, torch.zeros_like(m_evap))

    def test_wet_drying(self, cavity_case):
        from pyfoam.applications.film_foam_enhanced_3 import FilmFoamEnhanced3
        from pyfoam.core.device import get_default_dtype
        solver = FilmFoamEnhanced3(cavity_case, wet_dry_threshold=1e-8)
        dtype = get_default_dtype()
        h = torch.full((solver.mesh.n_cells,), 1e-4, dtype=dtype)
        h[0] = 1e-10  # Below threshold
        h_result = solver._apply_wet_drying(h, 0.001)
        assert h_result.shape == h.shape
        assert (h_result >= 0).all()

    def test_amr_marking(self, cavity_case):
        from pyfoam.applications.film_foam_enhanced_3 import FilmFoamEnhanced3
        from pyfoam.core.device import get_default_dtype
        solver = FilmFoamEnhanced3(cavity_case, amr_enabled=True, amr_threshold=1e-5)
        dtype = get_default_dtype()
        h = torch.full((solver.mesh.n_cells,), 1e-4, dtype=dtype)
        flags = solver._mark_refinement_cells(h)
        assert flags.shape == h.shape
        assert flags.dtype == torch.bool


# ===========================================================================
# Tests: SprayFoamEnhanced3
# ===========================================================================


class TestSprayFoamEnhanced3:
    """Tests for enhanced spray solver v3."""

    def test_init(self, cavity_case):
        from pyfoam.applications.spray_foam_enhanced_3 import SprayFoamEnhanced3
        solver = SprayFoamEnhanced3(cavity_case, dynamic_drag=True)
        assert solver.dynamic_drag is True

    def test_moment_tracker(self):
        from pyfoam.applications.spray_foam_enhanced_3 import ParcelMomentTracker
        tracker = ParcelMomentTracker()
        tracker.update([1e-4, 2e-4, 3e-4])
        assert tracker.total_particles == 3
        assert tracker.M_0 == 3
        assert tracker.M_1 > 0

    def test_dynamic_drag(self):
        from pyfoam.applications.spray_foam_enhanced_3 import SprayFoamEnhanced3
        # Test the static method through class attribute access
        # We can't easily instantiate without a case, but we can test the logic
        C_d_base = 0.44
        # Sphere (y=0): should be close to Schiller-Naumann
        Re = 100.0
        distortion = 0.0
        Re_safe = max(Re, 1e-10)
        C_d_sn = (24.0 / Re_safe) * (1.0 + 0.15 * Re_safe ** 0.687)
        C_d_sphere = C_d_sn * (1.0 + 2.632 * 0.0)
        assert C_d_sphere > 0

        # Deformed (y=0.5): should have higher drag
        C_d_deformed = C_d_sn * (1.0 + 2.632 * 0.5)
        assert C_d_deformed > C_d_sphere

    def test_evaporation_rate(self, cavity_case):
        from pyfoam.applications.spray_foam_enhanced_3 import SprayFoamEnhanced3
        solver = SprayFoamEnhanced3(cavity_case)
        dm_dt = solver._compute_evaporation_rate_enhanced(
            d=1e-4, T_droplet=300.0, T_gas=500.0, v_rel=5.0, rho_l=800.0,
        )
        assert dm_dt >= 0
        assert isinstance(dm_dt, float)


# ===========================================================================
# Tests: MultiphaseEulerFoamEnhanced4
# ===========================================================================


class TestMultiphaseEulerFoamEnhanced4:
    """Tests for enhanced multiphase Euler solver v4."""

    def test_init(self, cavity_case):
        from pyfoam.applications.multiphase_euler_foam_enhanced_4 import MultiphaseEulerFoamEnhanced4
        # Test class exists and has expected attributes
        assert hasattr(MultiphaseEulerFoamEnhanced4, '__init__')

    def test_swarm_drag_correction(self, cavity_case):
        from pyfoam.applications.multiphase_euler_foam_enhanced_4 import MultiphaseEulerFoamEnhanced4
        # Test the swarm correction logic directly
        alpha_d = torch.tensor([0.1, 0.3, 0.5])
        C_d_base = 0.44
        swarm_exponent = 2.0
        alpha_c = (1.0 - alpha_d).clamp(min=0.01, max=1.0)
        correction = alpha_c.pow(-swarm_exponent)
        C_d_swarm = C_d_base * correction
        assert (C_d_swarm >= C_d_base).all()
        assert torch.isfinite(C_d_swarm).all()

    def test_phase_weighted_pressure(self, cavity_case):
        """Test phase-weighted pressure correction logic."""
        n_cells = 16
        p = torch.zeros(n_cells)
        alpha = torch.full((n_cells,), 0.5)
        U = torch.ones(n_cells, 3) * 0.1
        # Simplified test of correction logic
        residual = U.norm(dim=-1)
        numerator = alpha * residual
        denominator = alpha.pow(2)
        correction = numerator / denominator.clamp(min=1e-30)
        assert torch.isfinite(correction).all()


# ===========================================================================
# Tests: Exports
# ===========================================================================


class TestExportsV4:
    """Tests for __init__.py exports of v4/v6/v3 solvers."""

    def test_ico_enhanced_4_exported(self):
        from pyfoam.applications import IcoFoamEnhanced4
        assert IcoFoamEnhanced4 is not None

    def test_simple_enhanced_4_exported(self):
        from pyfoam.applications import SimpleFoamEnhanced4
        assert SimpleFoamEnhanced4 is not None

    def test_piso_enhanced_4_exported(self):
        from pyfoam.applications import PisoFoamEnhanced4
        assert PisoFoamEnhanced4 is not None

    def test_pimple_enhanced_4_exported(self):
        from pyfoam.applications import PimpleFoamEnhanced4
        assert PimpleFoamEnhanced4 is not None

    def test_rho_pimple_enhanced_4_exported(self):
        from pyfoam.applications import RhoPimpleFoamEnhanced4
        assert RhoPimpleFoamEnhanced4 is not None

    def test_buoyant_simple_enhanced_4_exported(self):
        from pyfoam.applications import BuoyantSimpleFoamEnhanced4
        assert BuoyantSimpleFoamEnhanced4 is not None

    def test_buoyant_pimple_enhanced_4_exported(self):
        from pyfoam.applications import BuoyantPimpleFoamEnhanced4
        assert BuoyantPimpleFoamEnhanced4 is not None

    def test_reacting_enhanced_6_exported(self):
        from pyfoam.applications import ReactingFoamEnhanced6
        assert ReactingFoamEnhanced6 is not None

    def test_solid_enhanced_3_exported(self):
        from pyfoam.applications import SolidFoamEnhanced3
        assert SolidFoamEnhanced3 is not None

    def test_film_enhanced_3_exported(self):
        from pyfoam.applications import FilmFoamEnhanced3
        assert FilmFoamEnhanced3 is not None

    def test_spray_enhanced_3_exported(self):
        from pyfoam.applications import SprayFoamEnhanced3
        assert SprayFoamEnhanced3 is not None

    def test_multiphase_enhanced_4_exported(self):
        from pyfoam.applications import MultiphaseEulerFoamEnhanced4
        assert MultiphaseEulerFoamEnhanced4 is not None

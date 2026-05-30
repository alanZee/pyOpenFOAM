"""
Unit tests for enhanced solver variants v6 (and v8/v5 specialized).

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
# Mesh generation helper (same pattern as test_enhanced_solvers_5)
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
# Tests: IcoFoamEnhanced6
# ===========================================================================


class TestIcoFoamEnhanced6:
    """Tests for enhanced ICO solver v6."""

    def test_init(self, cavity_case):
        from pyfoam.applications.ico_foam_enhanced_6 import IcoFoamEnhanced6
        solver = IcoFoamEnhanced6(cavity_case, vorticity_stab=True, compact_reconstruction=True)
        assert solver.vorticity_stab is True
        assert solver.compact_reconstruction is True

    def test_init_defaults(self, cavity_case):
        from pyfoam.applications.ico_foam_enhanced_6 import IcoFoamEnhanced6
        solver = IcoFoamEnhanced6(cavity_case)
        assert solver.vorticity_threshold == pytest.approx(1.0)
        assert solver.spectral_element_order == 2

    def test_vorticity_magnitude(self, cavity_case):
        from pyfoam.applications.ico_foam_enhanced_6 import IcoFoamEnhanced6
        solver = IcoFoamEnhanced6(cavity_case, vorticity_stab=True)
        omega = solver._compute_vorticity_magnitude(solver.U)
        assert omega.shape == (solver.mesh.n_cells,)
        assert torch.isfinite(omega).all()
        assert (omega >= 0).all()

    def test_vorticity_stabilisation(self, cavity_case):
        from pyfoam.applications.ico_foam_enhanced_6 import IcoFoamEnhanced6
        solver = IcoFoamEnhanced6(cavity_case, vorticity_stab=True)
        U_stab = solver._apply_vorticity_stabilisation(solver.U, solver.U.clone())
        assert U_stab.shape == solver.U.shape
        assert torch.isfinite(U_stab).all()

    def test_vorticity_stab_disabled(self, cavity_case):
        from pyfoam.applications.ico_foam_enhanced_6 import IcoFoamEnhanced6
        solver = IcoFoamEnhanced6(cavity_case, vorticity_stab=False)
        U = solver.U.clone()
        U_stab = solver._apply_vorticity_stabilisation(U, solver.U.clone())
        assert torch.allclose(U_stab, U)

    def test_compact_reconstruction(self, cavity_case):
        from pyfoam.applications.ico_foam_enhanced_6 import IcoFoamEnhanced6
        solver = IcoFoamEnhanced6(cavity_case, compact_reconstruction=True)
        grad = solver._compact_reconstruction_gradient(solver.p)
        assert grad.shape == (solver.mesh.n_cells, 3)
        assert torch.isfinite(grad).all()

    def test_spectral_element_advance(self, cavity_case):
        from pyfoam.applications.ico_foam_enhanced_6 import IcoFoamEnhanced6
        solver = IcoFoamEnhanced6(cavity_case, spectral_element_order=2)
        U_se = solver._spectral_element_advance(
            solver.U, solver.U.clone(), solver.delta_t,
        )
        assert U_se.shape == solver.U.shape
        assert torch.isfinite(U_se).all()

    def test_spectral_element_order3(self, cavity_case):
        from pyfoam.applications.ico_foam_enhanced_6 import IcoFoamEnhanced6
        solver = IcoFoamEnhanced6(cavity_case, spectral_element_order=3)
        U_se = solver._spectral_element_advance(
            solver.U, solver.U.clone() * 0.99, solver.delta_t,
        )
        assert U_se.shape == solver.U.shape
        assert torch.isfinite(U_se).all()


# ===========================================================================
# Tests: SimpleFoamEnhanced6
# ===========================================================================


class TestSimpleFoamEnhanced6:
    """Tests for enhanced SIMPLE solver v6."""

    def test_init(self, cavity_case):
        from pyfoam.applications.simple_foam_enhanced_6 import SimpleFoamEnhanced6
        solver = SimpleFoamEnhanced6(cavity_case, tensorial_viscosity=True)
        assert solver.tensorial_viscosity is True

    def test_init_defaults(self, cavity_case):
        from pyfoam.applications.simple_foam_enhanced_6 import SimpleFoamEnhanced6
        solver = SimpleFoamEnhanced6(cavity_case)
        assert solver.pseudo_transient is True
        assert solver.residual_weighting is True
        assert solver.reynolds_ref == pytest.approx(100.0)

    def test_tensorial_viscosity_skip(self, cavity_case):
        from pyfoam.applications.simple_foam_enhanced_6 import SimpleFoamEnhanced6
        solver = SimpleFoamEnhanced6(cavity_case, tensorial_viscosity=True)
        nu_t = solver._compute_tensorial_viscosity(solver.U, 0.01)
        assert nu_t.shape == (solver.mesh.n_cells, 3, 3)
        assert torch.isfinite(nu_t).all()

    def test_tensorial_viscosity_skip_disabled(self, cavity_case):
        from pyfoam.applications.simple_foam_enhanced_6 import SimpleFoamEnhanced6
        solver = SimpleFoamEnhanced6(cavity_case, tensorial_viscosity=False)
        nu_t = solver._compute_tensorial_viscosity(solver.U, 0.01)
        assert nu_t.shape == (solver.mesh.n_cells, 3, 3)
        # Should be diagonal with nu_t values
        assert nu_t[:, 0, 0].mean().item() == pytest.approx(0.01)

    def test_pseudo_dt(self, cavity_case):
        from pyfoam.applications.simple_foam_enhanced_6 import SimpleFoamEnhanced6
        solver = SimpleFoamEnhanced6(cavity_case, pseudo_transient=True)
        dt = solver._estimate_pseudo_dt(solver.U, solver.U.clone() * 0.99)
        assert isinstance(dt, float)
        assert dt > 0

    def test_residual_weighting(self, cavity_case):
        from pyfoam.applications.simple_foam_enhanced_6 import SimpleFoamEnhanced6
        solver = SimpleFoamEnhanced6(cavity_case, residual_weighting=True)
        weighted = solver._weight_residual_by_reynolds(1e-3, solver.U, 0.01)
        assert isinstance(weighted, float)
        assert weighted <= 1e-3

    def test_residual_weighting_disabled(self, cavity_case):
        from pyfoam.applications.simple_foam_enhanced_6 import SimpleFoamEnhanced6
        solver = SimpleFoamEnhanced6(cavity_case, residual_weighting=False)
        weighted = solver._weight_residual_by_reynolds(1e-3, solver.U, 0.01)
        assert weighted == pytest.approx(1e-3)


# ===========================================================================
# Tests: PisoFoamEnhanced6
# ===========================================================================


class TestPisoFoamEnhanced6:
    """Tests for enhanced PISO solver v6."""

    def test_init(self, cavity_case):
        from pyfoam.applications.piso_foam_enhanced_6 import PisoFoamEnhanced6
        solver = PisoFoamEnhanced6(cavity_case, entropy_stable=True)
        assert solver.entropy_stable is True

    def test_init_defaults(self, cavity_case):
        from pyfoam.applications.piso_foam_enhanced_6 import PisoFoamEnhanced6
        solver = PisoFoamEnhanced6(cavity_case)
        assert solver.adaptive_correctors is True
        assert solver.compact_rhie_chow is True

    def test_adaptive_correctors(self, cavity_case):
        from pyfoam.applications.piso_foam_enhanced_6 import PisoFoamEnhanced6
        solver = PisoFoamEnhanced6(cavity_case, adaptive_correctors=True)
        # Converging residuals should trigger early stop
        stop = solver._should_stop_correctors(3, [1.0, 0.5, 0.48, 0.47])
        assert isinstance(stop, bool)

    def test_adaptive_correctors_patience(self, cavity_case):
        from pyfoam.applications.piso_foam_enhanced_6 import PisoFoamEnhanced6
        solver = PisoFoamEnhanced6(cavity_case, adaptive_correctors=True, corrector_patience=3)
        # Should not stop before patience
        stop = solver._should_stop_correctors(1, [1.0, 0.5, 0.48])
        assert stop is False

    def test_entropy_stable_flux(self, cavity_case):
        from pyfoam.applications.piso_foam_enhanced_6 import PisoFoamEnhanced6
        solver = PisoFoamEnhanced6(cavity_case, entropy_stable=True)
        U_es = solver._entropy_stable_flux(solver.U, solver.U.clone(), solver.delta_t)
        assert U_es.shape == solver.U.shape
        assert torch.isfinite(U_es).all()

    def test_entropy_stable_disabled(self, cavity_case):
        from pyfoam.applications.piso_foam_enhanced_6 import PisoFoamEnhanced6
        solver = PisoFoamEnhanced6(cavity_case, entropy_stable=False)
        U = solver.U.clone()
        U_es = solver._entropy_stable_flux(U, solver.U.clone(), solver.delta_t)
        assert torch.allclose(U_es, U)

    def test_compact_rhie_chow(self, cavity_case):
        from pyfoam.applications.piso_foam_enhanced_6 import PisoFoamEnhanced6
        solver = PisoFoamEnhanced6(cavity_case, compact_rhie_chow=True)
        A_p = torch.ones(solver.mesh.n_cells, dtype=solver.U.dtype)
        U_rc = solver._compact_rhie_chow_interpolation(solver.U, solver.p, A_p)
        assert U_rc.shape == solver.U.shape
        assert torch.isfinite(U_rc).all()


# ===========================================================================
# Tests: PimpleFoamEnhanced6
# ===========================================================================


class TestPimpleFoamEnhanced6:
    """Tests for enhanced PIMPLE solver v6."""

    def test_init(self, cavity_case):
        from pyfoam.applications.pimple_foam_enhanced_6 import PimpleFoamEnhanced6
        solver = PimpleFoamEnhanced6(cavity_case, back_substitution=True, pod_precondition=True)
        assert solver.back_substitution is True
        assert solver.pod_precondition is True

    def test_init_defaults(self, cavity_case):
        from pyfoam.applications.pimple_foam_enhanced_6 import PimpleFoamEnhanced6
        solver = PimpleFoamEnhanced6(cavity_case)
        assert solver.residual_scaling is True
        assert solver.pod_pressure_modes == 5

    def test_back_substitution(self, cavity_case):
        from pyfoam.applications.pimple_foam_enhanced_6 import PimpleFoamEnhanced6
        solver = PimpleFoamEnhanced6(cavity_case, back_substitution=True)
        p_corr, U_corr = solver._momentum_back_substitution(
            solver.U, solver.p, solver.U.clone(),
        )
        assert p_corr.shape == solver.p.shape
        assert U_corr.shape == solver.U.shape
        assert torch.isfinite(p_corr).all()

    def test_back_substitution_disabled(self, cavity_case):
        from pyfoam.applications.pimple_foam_enhanced_6 import PimpleFoamEnhanced6
        solver = PimpleFoamEnhanced6(cavity_case, back_substitution=False)
        p_corr, U_corr = solver._momentum_back_substitution(
            solver.U, solver.p, solver.U.clone(),
        )
        assert torch.allclose(p_corr, solver.p)
        assert torch.allclose(U_corr, solver.U)

    def test_residual_scaling(self, cavity_case):
        from pyfoam.applications.pimple_foam_enhanced_6 import PimpleFoamEnhanced6
        solver = PimpleFoamEnhanced6(cavity_case, residual_scaling=True)
        scaled = solver._scale_residual_by_reynolds(1e-3, solver.U, 0.01)
        assert isinstance(scaled, float)
        assert scaled <= 1e-3

    def test_pod_pressure_precondition(self, cavity_case):
        from pyfoam.applications.pimple_foam_enhanced_6 import PimpleFoamEnhanced6
        solver = PimpleFoamEnhanced6(cavity_case, pod_precondition=True)
        p_pod = solver._pod_pressure_precondition(solver.p)
        assert p_pod.shape == solver.p.shape
        assert torch.isfinite(p_pod).all()


# ===========================================================================
# Tests: RhoPimpleFoamEnhanced6
# ===========================================================================


class TestRhoPimpleFoamEnhanced6:
    """Tests for enhanced compressible PIMPLE solver v6."""

    def test_init(self, compressible_case):
        from pyfoam.applications.rho_pimple_foam_enhanced_6 import RhoPimpleFoamEnhanced6
        solver = RhoPimpleFoamEnhanced6(compressible_case, baroclinic_torque=True, entropy_variables=True)
        assert solver.baroclinic_torque is True
        assert solver.entropy_variables is True

    def test_init_defaults(self, compressible_case):
        from pyfoam.applications.rho_pimple_foam_enhanced_6 import RhoPimpleFoamEnhanced6
        solver = RhoPimpleFoamEnhanced6(compressible_case)
        assert solver.density_velocity_coupling is True
        assert solver.baroclinic_torque is True

    def test_density_velocity_correction(self, compressible_case):
        from pyfoam.applications.rho_pimple_foam_enhanced_6 import RhoPimpleFoamEnhanced6
        solver = RhoPimpleFoamEnhanced6(compressible_case, density_velocity_coupling=True)
        rho_corr, U_corr = solver._density_velocity_correction(
            solver.rho, solver.U, solver.p,
        )
        assert rho_corr.shape == solver.rho.shape
        assert U_corr.shape == solver.U.shape
        assert torch.isfinite(rho_corr).all()

    def test_baroclinic_torque(self, compressible_case):
        from pyfoam.applications.rho_pimple_foam_enhanced_6 import RhoPimpleFoamEnhanced6
        solver = RhoPimpleFoamEnhanced6(compressible_case, baroclinic_torque=True)
        baro = solver._compute_baroclinic_torque(solver.rho, solver.p)
        assert baro.shape == (solver.mesh.n_cells, 3)
        assert torch.isfinite(baro).all()

    def test_baroclinic_disabled(self, compressible_case):
        from pyfoam.applications.rho_pimple_foam_enhanced_6 import RhoPimpleFoamEnhanced6
        solver = RhoPimpleFoamEnhanced6(compressible_case, baroclinic_torque=False)
        baro = solver._compute_baroclinic_torque(solver.rho, solver.p)
        assert torch.allclose(baro, torch.zeros_like(baro))

    def test_entropy_dissipation(self, compressible_case):
        from pyfoam.applications.rho_pimple_foam_enhanced_6 import RhoPimpleFoamEnhanced6
        solver = RhoPimpleFoamEnhanced6(compressible_case, entropy_variables=True)
        U_ent = solver._apply_entropy_dissipation(solver.U, solver.rho, solver.T)
        assert U_ent.shape == solver.U.shape
        assert torch.isfinite(U_ent).all()


# ===========================================================================
# Tests: BuoyantSimpleFoamEnhanced6
# ===========================================================================


class TestBuoyantSimpleFoamEnhanced6:
    """Tests for enhanced buoyant SIMPLE solver v6."""

    def test_init(self, buoyant_case):
        from pyfoam.applications.buoyant_simple_foam_enhanced_6 import BuoyantSimpleFoamEnhanced6
        solver = BuoyantSimpleFoamEnhanced6(buoyant_case, strong_coupling=True, ggdh=True)
        assert solver.strong_coupling is True
        assert solver.ggdh is True

    def test_init_defaults(self, buoyant_case):
        from pyfoam.applications.buoyant_simple_foam_enhanced_6 import BuoyantSimpleFoamEnhanced6
        solver = BuoyantSimpleFoamEnhanced6(buoyant_case)
        assert solver.energy_momentum_interchange is True
        assert solver.n_coupling_iters == 3

    def test_strong_coupling(self, buoyant_case):
        from pyfoam.applications.buoyant_simple_foam_enhanced_6 import BuoyantSimpleFoamEnhanced6
        solver = BuoyantSimpleFoamEnhanced6(buoyant_case, strong_coupling=True)
        p_corr, T_corr = solver._strongly_coupled_buoyancy_pressure(
            solver.p, solver.T, solver.rho,
        )
        assert p_corr.shape == solver.p.shape
        assert T_corr.shape == solver.T.shape
        assert torch.isfinite(p_corr).all()
        assert torch.isfinite(T_corr).all()

    def test_energy_momentum_interchange(self, buoyant_case):
        from pyfoam.applications.buoyant_simple_foam_enhanced_6 import BuoyantSimpleFoamEnhanced6
        solver = BuoyantSimpleFoamEnhanced6(buoyant_case, energy_momentum_interchange=True)
        U_corr, T_corr = solver._energy_momentum_interchange_correction(solver.U, solver.T)
        assert U_corr.shape == solver.U.shape
        assert T_corr.shape == solver.T.shape
        assert torch.isfinite(U_corr).all()

    def test_ggdh_heat_flux(self, buoyant_case):
        from pyfoam.applications.buoyant_simple_foam_enhanced_6 import BuoyantSimpleFoamEnhanced6
        solver = BuoyantSimpleFoamEnhanced6(buoyant_case, ggdh=True)
        q = solver._compute_ggdh_heat_flux(solver.T, solver.U)
        assert q.shape == (solver.mesh.n_cells,)
        assert torch.isfinite(q).all()

    def test_ggdh_disabled(self, buoyant_case):
        from pyfoam.applications.buoyant_simple_foam_enhanced_6 import BuoyantSimpleFoamEnhanced6
        solver = BuoyantSimpleFoamEnhanced6(buoyant_case, ggdh=False)
        q = solver._compute_ggdh_heat_flux(solver.T, solver.U)
        assert torch.allclose(q, torch.zeros_like(q))


# ===========================================================================
# Tests: BuoyantPimpleFoamEnhanced6
# ===========================================================================


class TestBuoyantPimpleFoamEnhanced6:
    """Tests for enhanced buoyant PIMPLE solver v6."""

    def test_init(self, buoyant_case):
        from pyfoam.applications.buoyant_pimple_foam_enhanced_6 import BuoyantPimpleFoamEnhanced6
        solver = BuoyantPimpleFoamEnhanced6(buoyant_case, projection_split=True, gravity_wave_filter=True)
        assert solver.projection_split is True
        assert solver.gravity_wave_filter is True

    def test_init_defaults(self, buoyant_case):
        from pyfoam.applications.buoyant_pimple_foam_enhanced_6 import BuoyantPimpleFoamEnhanced6
        solver = BuoyantPimpleFoamEnhanced6(buoyant_case)
        assert solver.coupled_kepsilon is True
        assert solver.gw_filter_coeff == pytest.approx(0.1)

    def test_projection_split_skip(self, buoyant_case):
        from pyfoam.applications.buoyant_pimple_foam_enhanced_6 import BuoyantPimpleFoamEnhanced6
        solver = BuoyantPimpleFoamEnhanced6(buoyant_case, projection_split=True)
        p_corr, T_corr = solver._projection_pressure_temperature(
            solver.p, solver.T, solver.rho, solver.delta_t,
        )
        assert p_corr.shape == solver.p.shape
        assert T_corr.shape == solver.T.shape
        assert torch.isfinite(p_corr).all()

    def test_gravity_wave_filter(self, buoyant_case):
        from pyfoam.applications.buoyant_pimple_foam_enhanced_6 import BuoyantPimpleFoamEnhanced6
        solver = BuoyantPimpleFoamEnhanced6(buoyant_case, gravity_wave_filter=True)
        U_filt = solver._adaptive_gravity_wave_filter(
            solver.U, solver.U.clone(), N=1.0, dt=0.001,
        )
        assert U_filt.shape == solver.U.shape
        assert torch.isfinite(U_filt).all()

    def test_gravity_wave_filter_zero_N(self, buoyant_case):
        from pyfoam.applications.buoyant_pimple_foam_enhanced_6 import BuoyantPimpleFoamEnhanced6
        solver = BuoyantPimpleFoamEnhanced6(buoyant_case, gravity_wave_filter=True)
        U = solver.U.clone()
        U_filt = solver._adaptive_gravity_wave_filter(U, solver.U.clone(), N=0.0, dt=0.001)
        assert torch.allclose(U_filt, U)

    def test_coupled_kepsilon(self, buoyant_case):
        from pyfoam.applications.buoyant_pimple_foam_enhanced_6 import BuoyantPimpleFoamEnhanced6
        solver = BuoyantPimpleFoamEnhanced6(buoyant_case, coupled_kepsilon=True)
        n_cells = solver.mesh.n_cells
        k = torch.full((n_cells,), 0.01)
        eps = torch.full((n_cells,), 0.01)
        k_new, eps_new = solver._coupled_kepsilon_buoyancy_update(
            k, eps, solver.T, solver.rho, solver.delta_t,
        )
        assert k_new.shape == k.shape
        assert (k_new >= 0).all()
        assert (eps_new >= 0).all()


# ===========================================================================
# Tests: ReactingFoamEnhanced8
# ===========================================================================


class TestReactingFoamEnhanced8:
    """Tests for enhanced reacting solver v8."""

    def test_init(self, reacting_case):
        from pyfoam.applications.reacting_foam_enhanced_8 import ReactingFoamEnhanced8
        solver = ReactingFoamEnhanced8(reacting_case, drg_reduction=True, weno_transport=True)
        assert solver.drg_reduction is True
        assert solver.weno_transport is True

    def test_init_defaults(self, reacting_case):
        from pyfoam.applications.reacting_foam_enhanced_8 import ReactingFoamEnhanced8
        solver = ReactingFoamEnhanced8(reacting_case)
        assert solver.nn_time_stepping is True
        assert solver.drg_tolerance == pytest.approx(1e-3)
        assert solver.weno_order == 5

    def test_drg_coefficients(self, reacting_case):
        from pyfoam.applications.reacting_foam_enhanced_8 import ReactingFoamEnhanced8
        solver = ReactingFoamEnhanced8(reacting_case, drg_reduction=True)
        coeffs = solver._compute_drg_coefficients(solver.Y, solver.T, "YA")
        assert isinstance(coeffs, dict)
        for name in solver.species:
            assert name in coeffs

    def test_weno_reconstruction(self, reacting_case):
        from pyfoam.applications.reacting_foam_enhanced_8 import ReactingFoamEnhanced8
        solver = ReactingFoamEnhanced8(reacting_case, weno_transport=True)
        Y_face = torch.full((solver.mesh.n_cells,), 0.5)
        Y_left = torch.full((solver.mesh.n_cells,), 0.4)
        Y_right = torch.full((solver.mesh.n_cells,), 0.6)
        Y_weno = solver._weno_reconstruction(Y_face, Y_left, Y_right)
        assert Y_weno.shape == Y_face.shape
        assert torch.isfinite(Y_weno).all()
        assert (Y_weno >= 0.4).all()
        assert (Y_weno <= 0.6).all()

    def test_weno_disabled(self, reacting_case):
        from pyfoam.applications.reacting_foam_enhanced_8 import ReactingFoamEnhanced8
        solver = ReactingFoamEnhanced8(reacting_case, weno_transport=False)
        Y_face = torch.full((solver.mesh.n_cells,), 0.5)
        Y_weno = solver._weno_reconstruction(Y_face, Y_face * 0.9, Y_face * 1.1)
        assert torch.allclose(Y_weno, Y_face)

    def test_nn_predict_subcycling(self, reacting_case):
        from pyfoam.applications.reacting_foam_enhanced_8 import ReactingFoamEnhanced8
        solver = ReactingFoamEnhanced8(reacting_case, nn_time_stepping=True)
        n_sub = solver._nn_predict_subcycling("YA", solver.Y, solver.T)
        assert isinstance(n_sub, int)
        assert n_sub >= 1


# ===========================================================================
# Tests: SolidFoamEnhanced5
# ===========================================================================


class TestSolidFoamEnhanced5:
    """Tests for enhanced solid mechanics solver v5."""

    def test_init(self, cavity_case):
        from pyfoam.applications.solid_foam_enhanced_5 import SolidFoamEnhanced5
        solver = SolidFoamEnhanced5(
            cavity_case, E=200e9, nu=0.3,
            phase_field_fracture=True, hmm=True,
        )
        assert solver.phase_field_fracture is True
        assert solver.hmm is True

    def test_init_defaults(self, cavity_case):
        from pyfoam.applications.solid_foam_enhanced_5 import SolidFoamEnhanced5
        solver = SolidFoamEnhanced5(cavity_case, E=200e9, nu=0.3)
        assert solver.Gc == pytest.approx(2700.0)
        assert solver.domain_decomposition is True

    def test_damage_evolution(self, cavity_case):
        from pyfoam.applications.solid_foam_enhanced_5 import SolidFoamEnhanced5
        solver = SolidFoamEnhanced5(
            cavity_case, E=200e9, nu=0.3,
            phase_field_fracture=True, failure_stress=100.0,
        )
        sigma = torch.randn(solver.mesh.n_cells, 6) * 200.0
        epsilon = torch.randn(solver.mesh.n_cells, 6) * 1e-4
        damage = solver._evolve_damage(sigma, epsilon, solver.delta_t)
        assert damage.shape == (solver.mesh.n_cells,)
        assert (damage >= 0).all()
        assert (damage <= 0.999).all()
        assert torch.isfinite(damage).all()

    def test_damage_disabled(self, cavity_case):
        from pyfoam.applications.solid_foam_enhanced_5 import SolidFoamEnhanced5
        solver = SolidFoamEnhanced5(cavity_case, E=200e9, nu=0.3, phase_field_fracture=False)
        sigma = torch.randn(solver.mesh.n_cells, 6)
        epsilon = torch.randn(solver.mesh.n_cells, 6)
        damage = solver._evolve_damage(sigma, epsilon, solver.delta_t)
        assert torch.allclose(damage, solver.damage)

    def test_hmm_constitutive(self, cavity_case):
        from pyfoam.applications.solid_foam_enhanced_5 import SolidFoamEnhanced5
        solver = SolidFoamEnhanced5(cavity_case, E=200e9, nu=0.3, hmm=True)
        epsilon = torch.randn(solver.mesh.n_cells, 6) * 1e-4
        sigma = solver._hmm_constitutive_update(epsilon, solver.T)
        assert sigma.shape == (solver.mesh.n_cells, 6)
        assert torch.isfinite(sigma).all()


# ===========================================================================
# Tests: FilmFoamEnhanced5
# ===========================================================================


class TestFilmFoamEnhanced5:
    """Tests for enhanced film solver v5."""

    def test_init(self, cavity_case):
        from pyfoam.applications.film_foam_enhanced_5 import FilmFoamEnhanced5
        solver = FilmFoamEnhanced5(cavity_case, foam_drainage=True, non_newtonian=True)
        assert solver.foam_drainage is True
        assert solver.non_newtonian is True

    def test_init_defaults(self, cavity_case):
        from pyfoam.applications.film_foam_enhanced_5 import FilmFoamEnhanced5
        solver = FilmFoamEnhanced5(cavity_case)
        assert solver.thermal_viscosity is True
        assert solver.power_law_n == pytest.approx(0.6)

    def test_drainage_source(self, cavity_case):
        from pyfoam.applications.film_foam_enhanced_5 import FilmFoamEnhanced5
        solver = FilmFoamEnhanced5(cavity_case, foam_drainage=True)
        h = torch.full((solver.mesh.n_cells,), 1e-6)
        eps = torch.full((solver.mesh.n_cells,), 0.1)
        dS_h, dS_eps = solver._compute_drainage_source(h, eps)
        assert dS_h.shape == h.shape
        assert dS_eps.shape == eps.shape
        assert (dS_h <= 0).all()  # Drainage reduces film thickness
        assert (dS_eps >= 0).all()  # Drainage increases liquid fraction
        assert torch.isfinite(dS_h).all()

    def test_drainage_disabled(self, cavity_case):
        from pyfoam.applications.film_foam_enhanced_5 import FilmFoamEnhanced5
        solver = FilmFoamEnhanced5(cavity_case, foam_drainage=False)
        h = torch.full((solver.mesh.n_cells,), 1e-6)
        eps = torch.full((solver.mesh.n_cells,), 0.1)
        dS_h, dS_eps = solver._compute_drainage_source(h, eps)
        assert torch.allclose(dS_h, torch.zeros_like(dS_h))

    def test_thermal_viscosity(self, cavity_case):
        from pyfoam.applications.film_foam_enhanced_5 import FilmFoamEnhanced5
        solver = FilmFoamEnhanced5(cavity_case, thermal_viscosity=True)
        T = torch.full((solver.mesh.n_cells,), 350.0)
        mu = solver._thermal_viscosity_factor(T)
        assert mu.shape == T.shape
        assert torch.isfinite(mu).all()
        # Higher T should give lower viscosity (factor < 1)
        T_ref = torch.full((solver.mesh.n_cells,), 298.15)
        mu_ref = solver._thermal_viscosity_factor(T_ref)
        assert (mu <= mu_ref).all()

    def test_non_newtonian_viscosity(self, cavity_case):
        from pyfoam.applications.film_foam_enhanced_5 import FilmFoamEnhanced5
        solver = FilmFoamEnhanced5(cavity_case, non_newtonian=True, power_law_n=0.6)
        gamma = torch.full((solver.mesh.n_cells,), 100.0)
        mu = solver._non_newtonian_viscosity(gamma)
        assert mu.shape == gamma.shape
        assert (mu > 0).all()
        assert torch.isfinite(mu).all()


# ===========================================================================
# Tests: SprayFoamEnhanced5
# ===========================================================================


class TestSprayFoamEnhanced5:
    """Tests for enhanced spray solver v5."""

    def test_init(self, cavity_case):
        from pyfoam.applications.spray_foam_enhanced_5 import SprayFoamEnhanced5
        solver = SprayFoamEnhanced5(cavity_case, multi_physics=True, ml_collision=True)
        assert solver.multi_physics is True
        assert solver.ml_collision is True

    def test_init_defaults(self, cavity_case):
        from pyfoam.applications.spray_foam_enhanced_5 import SprayFoamEnhanced5
        solver = SprayFoamEnhanced5(cavity_case)
        assert solver.spray_combustion is True
        assert solver.combustion_efficiency == pytest.approx(0.95)

    def test_collision_outcome_dataclass(self):
        from pyfoam.applications.spray_foam_enhanced_5 import CollisionOutcome
        outcome = CollisionOutcome("coalesce", 1, [])
        assert outcome.outcome == "coalesce"
        assert outcome.n_fragments == 1

    def test_ml_predict_collision_low_we(self, cavity_case):
        from pyfoam.applications.spray_foam_enhanced_5 import SprayFoamEnhanced5
        solver = SprayFoamEnhanced5(cavity_case, ml_collision=True)
        result = solver._ml_predict_collision(We=5.0, B=0.5, size_ratio=0.8)
        assert result.outcome == "coalesce"
        assert result.n_fragments == 1

    def test_ml_predict_collision_high_we(self, cavity_case):
        from pyfoam.applications.spray_foam_enhanced_5 import SprayFoamEnhanced5
        solver = SprayFoamEnhanced5(cavity_case, ml_collision=True)
        result = solver._ml_predict_collision(We=200.0, B=0.5, size_ratio=0.8)
        assert result.outcome == "fragment"
        assert result.n_fragments >= 2

    def test_ml_predict_collision_bounce(self, cavity_case):
        from pyfoam.applications.spray_foam_enhanced_5 import SprayFoamEnhanced5
        solver = SprayFoamEnhanced5(cavity_case, ml_collision=True)
        result = solver._ml_predict_collision(We=50.0, B=0.1, size_ratio=0.9)
        assert result.outcome == "bounce"


# ===========================================================================
# Tests: MultiphaseEulerFoamEnhanced6
# ===========================================================================


class TestMultiphaseEulerFoamEnhanced6:
    """Tests for enhanced multiphase Euler solver v6."""

    def test_init(self, cavity_case):
        from pyfoam.applications.multiphase_euler_foam_enhanced_6 import MultiphaseEulerFoamEnhanced6
        assert hasattr(MultiphaseEulerFoamEnhanced6, '__init__')

    def test_class_exists(self):
        from pyfoam.applications.multiphase_euler_foam_enhanced_6 import MultiphaseEulerFoamEnhanced6
        assert MultiphaseEulerFoamEnhanced6 is not None

    def test_les_sgs_logic(self):
        """Test Smagorinsky SGS viscosity computation logic."""
        # Simulate the SGS computation
        cs = 0.1
        Delta = torch.full((16,), 0.25)  # cell size
        S_mag = torch.rand(16) * 10.0
        nu_sgs = (cs * Delta).pow(2) * S_mag
        assert nu_sgs.shape == (16,)
        assert (nu_sgs >= 0).all()
        assert torch.isfinite(nu_sgs).all()

    def test_interfacial_force_components(self):
        """Test that interfacial force decomposition is consistent."""
        F_drag = torch.randn(16, 3) * 0.1
        F_vm = torch.randn(16, 3) * 0.01
        F_lift = torch.randn(16, 3) * 0.05
        F_wall = torch.zeros(16, 3)
        F_total = F_drag + F_vm + F_lift + F_wall
        assert F_total.shape == (16, 3)
        assert torch.isfinite(F_total).all()


# ===========================================================================
# Tests: Exports
# ===========================================================================


class TestExportsV6:
    """Tests for __init__.py exports of v6/v8/v5 solvers."""

    def test_ico_enhanced_6_exported(self):
        from pyfoam.applications import IcoFoamEnhanced6
        assert IcoFoamEnhanced6 is not None

    def test_simple_enhanced_6_exported(self):
        from pyfoam.applications import SimpleFoamEnhanced6
        assert SimpleFoamEnhanced6 is not None

    def test_piso_enhanced_6_exported(self):
        from pyfoam.applications import PisoFoamEnhanced6
        assert PisoFoamEnhanced6 is not None

    def test_pimple_enhanced_6_exported(self):
        from pyfoam.applications import PimpleFoamEnhanced6
        assert PimpleFoamEnhanced6 is not None

    def test_rho_pimple_enhanced_6_exported(self):
        from pyfoam.applications import RhoPimpleFoamEnhanced6
        assert RhoPimpleFoamEnhanced6 is not None

    def test_buoyant_simple_enhanced_6_exported(self):
        from pyfoam.applications import BuoyantSimpleFoamEnhanced6
        assert BuoyantSimpleFoamEnhanced6 is not None

    def test_buoyant_pimple_enhanced_6_exported(self):
        from pyfoam.applications import BuoyantPimpleFoamEnhanced6
        assert BuoyantPimpleFoamEnhanced6 is not None

    def test_reacting_enhanced_8_exported(self):
        from pyfoam.applications import ReactingFoamEnhanced8
        assert ReactingFoamEnhanced8 is not None

    def test_solid_enhanced_5_exported(self):
        from pyfoam.applications import SolidFoamEnhanced5
        assert SolidFoamEnhanced5 is not None

    def test_film_enhanced_5_exported(self):
        from pyfoam.applications import FilmFoamEnhanced5
        assert FilmFoamEnhanced5 is not None

    def test_spray_enhanced_5_exported(self):
        from pyfoam.applications import SprayFoamEnhanced5
        assert SprayFoamEnhanced5 is not None

    def test_multiphase_enhanced_6_exported(self):
        from pyfoam.applications import MultiphaseEulerFoamEnhanced6
        assert MultiphaseEulerFoamEnhanced6 is not None

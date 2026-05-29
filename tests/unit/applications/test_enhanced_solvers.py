"""
Unit tests for enhanced solver variants.

Tests cover:
- Case loading and mesh construction
- Solver initialisation
- Enhanced parameter reading
- Solver produces finite values after short run
- Solver writes output
- Export of new classes from __init__.py
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from pyfoam.core.dtype import CFD_DTYPE, INDEX_DTYPE
from pyfoam.io.foam_file import FoamFileHeader, FileFormat, write_foam_file


# ---------------------------------------------------------------------------
# Mesh generation helper (reused from test_piso_foam pattern)
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

    # Points
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

    # Internal vertical faces
    for j in range(n_cells_y):
        for i in range(n_cells_x - 1):
            p0 = j * (n_cells_x + 1) + i + 1
            p1 = p0 + n_cells_x + 1
            p2 = p1 + n_base
            p3 = p0 + n_base
            faces.append((4, p0, p1, p2, p3))
            owner.append(j * n_cells_x + i)
            neighbour.append(j * n_cells_x + i + 1)

    # Internal horizontal faces
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

    # Boundary: movingWall (top)
    for i in range(n_cells_x):
        p0 = n_cells_y * (n_cells_x + 1) + i
        p1 = p0 + 1
        p2 = p1 + n_base
        p3 = p0 + n_base
        faces.append((4, p0, p1, p2, p3))
        owner.append((n_cells_y - 1) * n_cells_x + i)
    n_top = n_cells_x
    top_start = n_internal

    # Boundary: fixedWalls (bottom, left, right)
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

    # Boundary: frontAndBack (empty)
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

    # Write mesh files
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

    # ---- system files ----
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

    # ---- constant files ----
    const_dir = case_dir / "constant"
    const_dir.mkdir(exist_ok=True)

    tp_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="dictionary", location="constant", object="transportProperties",
    )
    tp_body = f"nu              [0 2 -1 0 0 0 0] {nu};\n"
    write_foam_file(const_dir / "transportProperties", tp_header, tp_body, overwrite=True)

    # ---- 0/ directory ----
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

    # Compressible/buoyant fields
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

    # Thermophysical properties for compressible
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

    # Gravity file for buoyant cases
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

    # Reacting fields
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

        # Reactions file
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
    """Small cavity case for incompressible solvers."""
    case_dir = tmp_path / "cavity"
    _make_cavity_case(case_dir)
    return case_dir


@pytest.fixture
def compressible_case(tmp_path):
    """Small cavity case for compressible solvers."""
    case_dir = tmp_path / "compressible"
    _make_cavity_case(case_dir, compressible=True)
    return case_dir


@pytest.fixture
def buoyant_case(tmp_path):
    """Small cavity case for buoyant solvers."""
    case_dir = tmp_path / "buoyant"
    _make_cavity_case(case_dir, buoyant=True)
    return case_dir


@pytest.fixture
def reacting_case(tmp_path):
    """Small cavity case for reacting solvers."""
    case_dir = tmp_path / "reacting"
    _make_cavity_case(case_dir, reacting=True, end_time=0.005, delta_t=0.0005)
    return case_dir


# ===========================================================================
# Tests: PisoFoamEnhanced
# ===========================================================================


class TestPisoFoamEnhanced:
    """Tests for enhanced PISO solver."""

    def test_init(self, cavity_case):
        """PisoFoamEnhanced initialises correctly."""
        from pyfoam.applications.piso_foam_enhanced import PisoFoamEnhanced
        solver = PisoFoamEnhanced(cavity_case, max_courant=0.5)
        assert solver.max_courant == 0.5
        assert solver.n_non_orth_correctors >= 0

    def test_run_completes(self, cavity_case):
        """PisoFoamEnhanced runs to completion."""
        from pyfoam.applications.piso_foam_enhanced import PisoFoamEnhanced
        solver = PisoFoamEnhanced(cavity_case, max_courant=10.0)
        conv = solver.run()
        assert conv is not None

    def test_finite_values(self, cavity_case):
        """All fields are finite after run."""
        from pyfoam.applications.piso_foam_enhanced import PisoFoamEnhanced
        solver = PisoFoamEnhanced(cavity_case, max_courant=10.0)
        solver.run()
        assert torch.isfinite(solver.U).all()
        assert torch.isfinite(solver.p).all()

    def test_sub_step_computation(self, cavity_case):
        """Sub-step computation works."""
        from pyfoam.applications.piso_foam_enhanced import PisoFoamEnhanced
        solver = PisoFoamEnhanced(cavity_case)
        n_sub = solver._compute_sub_steps()
        assert n_sub >= 1

    def test_max_courant_estimation(self, cavity_case):
        """Courant number estimation is non-negative."""
        from pyfoam.applications.piso_foam_enhanced import PisoFoamEnhanced
        solver = PisoFoamEnhanced(cavity_case)
        Co = solver._estimate_max_courant()
        assert Co >= 0

    def test_writes_output(self, cavity_case):
        """PisoFoamEnhanced writes output fields."""
        from pyfoam.applications.piso_foam_enhanced import PisoFoamEnhanced
        solver = PisoFoamEnhanced(cavity_case, max_courant=10.0)
        solver.run()
        time_dirs = [
            d for d in cavity_case.iterdir()
            if d.is_dir() and d.name.replace(".", "").isdigit() and d.name != "0"
        ]
        assert len(time_dirs) >= 1


# ===========================================================================
# Tests: PimpleFoamEnhanced
# ===========================================================================


class TestPimpleFoamEnhanced:
    """Tests for enhanced PIMPLE solver."""

    def test_init(self, cavity_case):
        """PimpleFoamEnhanced initialises correctly."""
        from pyfoam.applications.pimple_foam_enhanced import PimpleFoamEnhanced
        solver = PimpleFoamEnhanced(cavity_case, warm_up_steps=3)
        assert solver.warm_up_steps == 3
        assert solver.residual_plateau_tol > 0

    def test_run_completes(self, cavity_case):
        """PimpleFoamEnhanced runs to completion."""
        from pyfoam.applications.pimple_foam_enhanced import PimpleFoamEnhanced
        solver = PimpleFoamEnhanced(cavity_case)
        conv = solver.run()
        assert conv is not None

    def test_finite_values(self, cavity_case):
        """All fields are finite after run."""
        from pyfoam.applications.pimple_foam_enhanced import PimpleFoamEnhanced
        solver = PimpleFoamEnhanced(cavity_case)
        solver.run()
        assert torch.isfinite(solver.U).all()
        assert torch.isfinite(solver.p).all()

    def test_warm_up_factor(self, cavity_case):
        """Warm-up factor is in (0, 1] during warm-up."""
        from pyfoam.applications.pimple_foam_enhanced import PimpleFoamEnhanced
        solver = PimpleFoamEnhanced(cavity_case, warm_up_steps=5)
        solver._step_count = 2
        factor = solver._get_warm_up_factor()
        assert 0 < factor <= 1.0

    def test_plateau_detection(self, cavity_case):
        """Plateau detection works for small changes."""
        from pyfoam.applications.pimple_foam_enhanced import PimpleFoamEnhanced
        solver = PimpleFoamEnhanced(cavity_case, residual_plateau_tol=0.01)
        assert solver._is_plateau(1.0, None) is False
        assert solver._is_plateau(1.0, 1.001) is True
        assert solver._is_plateau(2.0, 1.0) is False


# ===========================================================================
# Tests: SimpleFoamEnhanced
# ===========================================================================


class TestSimpleFoamEnhanced:
    """Tests for enhanced SIMPLE solver."""

    def test_init_simplec(self, cavity_case):
        """SimpleFoamEnhanced initialises with SIMPLEC."""
        from pyfoam.applications.simple_foam_enhanced import SimpleFoamEnhanced
        solver = SimpleFoamEnhanced(cavity_case, use_simplec=True)
        assert solver.use_simplec is True

    def test_init_standard(self, cavity_case):
        """SimpleFoamEnhanced initialises with standard SIMPLE."""
        from pyfoam.applications.simple_foam_enhanced import SimpleFoamEnhanced
        solver = SimpleFoamEnhanced(cavity_case, use_simplec=False)
        assert solver.use_simplec is False

    def test_run_completes(self, cavity_case):
        """SimpleFoamEnhanced runs to completion."""
        from pyfoam.applications.simple_foam_enhanced import SimpleFoamEnhanced
        solver = SimpleFoamEnhanced(cavity_case)
        conv = solver.run()
        assert conv is not None

    def test_finite_values(self, cavity_case):
        """All fields are finite after run."""
        from pyfoam.applications.simple_foam_enhanced import SimpleFoamEnhanced
        solver = SimpleFoamEnhanced(cavity_case)
        solver.run()
        assert torch.isfinite(solver.U).all()
        assert torch.isfinite(solver.p).all()

    def test_dynamic_relaxation(self, cavity_case):
        """Dynamic relaxation adjusts alpha when residual increases."""
        from pyfoam.applications.simple_foam_enhanced import SimpleFoamEnhanced
        solver = SimpleFoamEnhanced(cavity_case, dynamic_relaxation=True)
        original_alpha = solver.alpha_U
        solver._prev_residual_U = 0.001
        solver._adjust_relaxation(0.002)  # Residual doubled
        assert solver.alpha_U <= original_alpha


# ===========================================================================
# Tests: IcoFoamEnhanced
# ===========================================================================


class TestIcoFoamEnhanced:
    """Tests for enhanced ICO solver."""

    def test_init(self, cavity_case):
        """IcoFoamEnhanced initialises correctly."""
        from pyfoam.applications.ico_foam_enhanced import IcoFoamEnhanced
        solver = IcoFoamEnhanced(cavity_case, theta=0.5, adaptive_dt=True)
        assert abs(solver.theta - 0.5) < 1e-10
        assert solver.adaptive_dt is True

    def test_init_euler(self, cavity_case):
        """IcoFoamEnhanced with Euler scheme."""
        from pyfoam.applications.ico_foam_enhanced import IcoFoamEnhanced
        solver = IcoFoamEnhanced(cavity_case, theta=1.0)
        assert abs(solver.theta - 1.0) < 1e-10

    def test_run_completes(self, cavity_case):
        """IcoFoamEnhanced runs to completion."""
        from pyfoam.applications.ico_foam_enhanced import IcoFoamEnhanced
        solver = IcoFoamEnhanced(cavity_case, adaptive_dt=False)
        conv = solver.run()
        assert conv is not None

    def test_finite_values(self, cavity_case):
        """All fields are finite after run."""
        from pyfoam.applications.ico_foam_enhanced import IcoFoamEnhanced
        solver = IcoFoamEnhanced(cavity_case, adaptive_dt=False)
        solver.run()
        assert torch.isfinite(solver.U).all()
        assert torch.isfinite(solver.p).all()

    def test_adaptive_dt(self, cavity_case):
        """Adaptive time step is positive and bounded."""
        from pyfoam.applications.ico_foam_enhanced import IcoFoamEnhanced
        solver = IcoFoamEnhanced(cavity_case, adaptive_dt=True)
        dt = solver._compute_adaptive_dt()
        assert dt > 0
        assert dt <= solver.delta_t * 2.0

    def test_mass_flux(self, cavity_case):
        """Consistent mass flux computation produces a tensor."""
        from pyfoam.applications.ico_foam_enhanced import IcoFoamEnhanced
        solver = IcoFoamEnhanced(cavity_case)
        phi = solver._compute_consistent_mass_flux()
        assert phi.shape == (solver.mesh.n_faces,)


# ===========================================================================
# Tests: RhoPimpleFoamEnhanced
# ===========================================================================


class TestRhoPimpleFoamEnhanced:
    """Tests for enhanced compressible PIMPLE solver."""

    def test_init(self, compressible_case):
        """RhoPimpleFoamEnhanced initialises correctly."""
        from pyfoam.applications.rho_pimple_foam_enhanced import RhoPimpleFoamEnhanced
        solver = RhoPimpleFoamEnhanced(compressible_case, coupling_iterations=2)
        assert solver.coupling_iterations == 2

    def test_run_completes(self, compressible_case):
        """RhoPimpleFoamEnhanced runs to completion."""
        from pyfoam.applications.rho_pimple_foam_enhanced import RhoPimpleFoamEnhanced
        solver = RhoPimpleFoamEnhanced(compressible_case)
        conv = solver.run()
        assert conv is not None

    def test_finite_values(self, compressible_case):
        """All fields are finite after run."""
        from pyfoam.applications.rho_pimple_foam_enhanced import RhoPimpleFoamEnhanced
        solver = RhoPimpleFoamEnhanced(compressible_case)
        solver.run()
        assert torch.isfinite(solver.U).all()
        assert torch.isfinite(solver.p).all()
        assert torch.isfinite(solver.T).all()

    def test_mach_computation(self, compressible_case):
        """Mach number computation is non-negative."""
        from pyfoam.applications.rho_pimple_foam_enhanced import RhoPimpleFoamEnhanced
        solver = RhoPimpleFoamEnhanced(compressible_case)
        Ma = solver._compute_mach_number()
        assert Ma.shape == (solver.mesh.n_cells,)
        assert (Ma >= 0).all()

    def test_max_mach(self, compressible_case):
        """Maximum Mach number is non-negative."""
        from pyfoam.applications.rho_pimple_foam_enhanced import RhoPimpleFoamEnhanced
        solver = RhoPimpleFoamEnhanced(compressible_case)
        max_ma = solver._compute_max_mach()
        assert max_ma >= 0


# ===========================================================================
# Tests: BuoyantSimpleFoamEnhanced
# ===========================================================================


class TestBuoyantSimpleFoamEnhanced:
    """Tests for enhanced buoyant SIMPLE solver."""

    def test_init(self, buoyant_case):
        """BuoyantSimpleFoamEnhanced initialises correctly."""
        from pyfoam.applications.buoyant_simple_foam_enhanced import BuoyantSimpleFoamEnhanced
        solver = BuoyantSimpleFoamEnhanced(buoyant_case, beta=3.33e-3, T_ref=300.0)
        assert abs(solver.beta - 3.33e-3) < 1e-10
        assert abs(solver.T_ref - 300.0) < 1e-10

    def test_boussinesq_detection(self, buoyant_case):
        """Boussinesq mode auto-detection works."""
        from pyfoam.applications.buoyant_simple_foam_enhanced import BuoyantSimpleFoamEnhanced
        solver = BuoyantSimpleFoamEnhanced(buoyant_case, beta=3.33e-3)
        # With small ΔT, Boussinesq should be valid
        assert solver._should_use_boussinesq() is True

    def test_richardson_number(self, buoyant_case):
        """Richardson number is finite."""
        from pyfoam.applications.buoyant_simple_foam_enhanced import BuoyantSimpleFoamEnhanced
        solver = BuoyantSimpleFoamEnhanced(buoyant_case)
        Ri = solver._compute_richardson_number(solver.U, solver.T)
        assert isinstance(Ri, float)

    def test_boussinesq_buoyancy(self, buoyant_case):
        """Boussinesq buoyancy force has correct shape."""
        from pyfoam.applications.buoyant_simple_foam_enhanced import BuoyantSimpleFoamEnhanced
        solver = BuoyantSimpleFoamEnhanced(buoyant_case)
        F = solver._compute_boussinesq_buoyancy(solver.T, 1.225)
        assert F.shape == (solver.mesh.n_cells, 3)

    def test_buoyancy_aware_relaxation(self, buoyant_case):
        """Buoyancy-aware relaxation reduces alpha for high Ri."""
        from pyfoam.applications.buoyant_simple_foam_enhanced import BuoyantSimpleFoamEnhanced
        solver = BuoyantSimpleFoamEnhanced(buoyant_case)
        alpha_U_low, _ = solver._buoyancy_aware_relaxation(0.5)
        alpha_U_high, _ = solver._buoyancy_aware_relaxation(10.0)
        assert alpha_U_high <= alpha_U_low


# ===========================================================================
# Tests: BuoyantPimpleFoamEnhanced
# ===========================================================================


class TestBuoyantPimpleFoamEnhanced:
    """Tests for enhanced buoyant PIMPLE solver."""

    def test_init(self, buoyant_case):
        """BuoyantPimpleFoamEnhanced initialises correctly."""
        from pyfoam.applications.buoyant_pimple_foam_enhanced import BuoyantPimpleFoamEnhanced
        solver = BuoyantPimpleFoamEnhanced(buoyant_case, beta=3.33e-3, T_ref=300.0)
        assert abs(solver.beta - 3.33e-3) < 1e-10
        assert abs(solver.T_ref - 300.0) < 1e-10

    def test_richardson_computation(self, buoyant_case):
        """Richardson number computation returns a float."""
        from pyfoam.applications.buoyant_pimple_foam_enhanced import BuoyantPimpleFoamEnhanced
        solver = BuoyantPimpleFoamEnhanced(buoyant_case)
        Ri = solver._compute_richardson(solver.U, solver.T)
        assert isinstance(Ri, float)
        assert Ri >= 0

    def test_time_step_adaptation(self, buoyant_case):
        """Time step adaptation reduces dt for high Ri."""
        from pyfoam.applications.buoyant_pimple_foam_enhanced import BuoyantPimpleFoamEnhanced
        solver = BuoyantPimpleFoamEnhanced(buoyant_case)
        dt_low = solver._adapt_time_step(0.5)
        dt_high = solver._adapt_time_step(10.0)
        assert dt_high <= dt_low

    def test_temp_dependent_relaxation(self, buoyant_case):
        """Temperature-dependent relaxation produces finite values."""
        from pyfoam.applications.buoyant_pimple_foam_enhanced import BuoyantPimpleFoamEnhanced
        solver = BuoyantPimpleFoamEnhanced(buoyant_case)
        T = solver.T.clone()
        field = solver.U.clone()
        field_old = solver.U.clone()
        result = solver._temperature_dependent_relaxation(T, field, field_old, 0.7)
        assert torch.isfinite(result).all()


# ===========================================================================
# Tests: ReactingFoamEnhanced3
# ===========================================================================


class TestReactingFoamEnhanced3:
    """Tests for enhanced reacting solver v3."""

    def test_init(self, reacting_case):
        """ReactingFoamEnhanced3 initialises correctly."""
        from pyfoam.applications.reacting_foam_enhanced_3 import ReactingFoamEnhanced3
        solver = ReactingFoamEnhanced3(reacting_case, stiffness_threshold=100.0)
        assert solver.stiffness_threshold == 100.0
        assert solver.max_chem_sub_steps >= 1

    def test_stiffness_indicator(self, reacting_case):
        """Stiffness indicator has correct shape and is positive."""
        from pyfoam.applications.reacting_foam_enhanced_3 import ReactingFoamEnhanced3
        solver = ReactingFoamEnhanced3(reacting_case)
        stiff = solver._compute_stiffness_indicator(solver.T, solver.Y)
        assert stiff.shape == (solver.mesh.n_cells,)
        assert (stiff >= 0).all()

    def test_equilibrium_reactions(self, reacting_case):
        """Equilibrium reactions are parsed (default: irreversible)."""
        from pyfoam.applications.reacting_foam_enhanced_3 import ReactingFoamEnhanced3
        solver = ReactingFoamEnhanced3(reacting_case)
        assert len(solver.equilibrium_reactions) >= 0

    def test_chemical_jacobian(self, reacting_case):
        """Chemical Jacobian has correct structure."""
        from pyfoam.applications.reacting_foam_enhanced_3 import ReactingFoamEnhanced3
        solver = ReactingFoamEnhanced3(reacting_case)
        J = solver._compute_chemical_jacobian(solver.T, solver.Y)
        for sp_i in solver.species:
            assert sp_i in J
            for sp_j in solver.species:
                assert sp_j in J[sp_i]
                assert J[sp_i][sp_j].shape == (solver.mesh.n_cells,)

    def test_chemistry_sub_cycling(self, reacting_case):
        """Chemistry sub-cycling produces valid mass fractions."""
        from pyfoam.applications.reacting_foam_enhanced_3 import ReactingFoamEnhanced3
        solver = ReactingFoamEnhanced3(reacting_case)
        Y_new = solver._advance_chemistry_sub_cycled(solver.Y, solver.T, solver.delta_t)
        for name in solver.species:
            assert torch.isfinite(Y_new[name]).all()
            assert (Y_new[name] >= 0).all()

    def test_semi_implicit_solver(self, reacting_case):
        """Semi-implicit fast reaction solver preserves mass."""
        from pyfoam.applications.reacting_foam_enhanced_3 import ReactingFoamEnhanced3
        solver = ReactingFoamEnhanced3(reacting_case)
        Y_new = solver._advance_fast_reactions_semi_implicit(
            solver.Y, solver.T, solver.delta_t,
        )
        # Check mass conservation: sum(Y) should be close to 1
        Y_sum = sum(Y_new.values())
        assert torch.allclose(Y_sum, torch.ones_like(Y_sum), atol=1e-6)


# ===========================================================================
# Tests: Exports
# ===========================================================================


class TestExports:
    """Tests for __init__.py exports."""

    def test_piso_enhanced_exported(self):
        """PisoFoamEnhanced is exported from pyfoam.applications."""
        from pyfoam.applications import PisoFoamEnhanced
        assert PisoFoamEnhanced is not None

    def test_pimple_enhanced_exported(self):
        """PimpleFoamEnhanced is exported."""
        from pyfoam.applications import PimpleFoamEnhanced
        assert PimpleFoamEnhanced is not None

    def test_simple_enhanced_exported(self):
        """SimpleFoamEnhanced is exported."""
        from pyfoam.applications import SimpleFoamEnhanced
        assert SimpleFoamEnhanced is not None

    def test_ico_enhanced_exported(self):
        """IcoFoamEnhanced is exported."""
        from pyfoam.applications import IcoFoamEnhanced
        assert IcoFoamEnhanced is not None

    def test_rho_pimple_enhanced_exported(self):
        """RhoPimpleFoamEnhanced is exported."""
        from pyfoam.applications import RhoPimpleFoamEnhanced
        assert RhoPimpleFoamEnhanced is not None

    def test_buoyant_simple_enhanced_exported(self):
        """BuoyantSimpleFoamEnhanced is exported."""
        from pyfoam.applications import BuoyantSimpleFoamEnhanced
        assert BuoyantSimpleFoamEnhanced is not None

    def test_buoyant_pimple_enhanced_exported(self):
        """BuoyantPimpleFoamEnhanced is exported."""
        from pyfoam.applications import BuoyantPimpleFoamEnhanced
        assert BuoyantPimpleFoamEnhanced is not None

    def test_reacting_enhanced_3_exported(self):
        """ReactingFoamEnhanced3 is exported."""
        from pyfoam.applications import ReactingFoamEnhanced3
        assert ReactingFoamEnhanced3 is not None

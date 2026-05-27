"""
Unit tests for PDRFoam — premixed combustion solver with b-Xi model.

Tests cover:
- Case loading and field initialisation
- Flame model computation (laminar flame speed, Xi, turbulent flame speed)
- Density model from progress variable
- Progress variable transport
- Solver execution
- Field output
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from pyfoam.core.dtype import CFD_DTYPE
from pyfoam.io.foam_file import FoamFileHeader, FileFormat, write_foam_file


# ---------------------------------------------------------------------------
# 网格生成辅助函数
# ---------------------------------------------------------------------------

def _make_pdr_case(
    case_dir: Path,
    n_cells: int = 5,
    L: float = 1.0,
    end_time: float = 2e-4,
    delta_t: float = 1e-5,
    n_outer_correctors: int = 3,
    n_correctors: int = 2,
    p_init: float = 101325.0,
    b_init: float = 0.0,
) -> None:
    """Write a 1D premixed combustion channel case."""
    case_dir.mkdir(parents=True, exist_ok=True)

    dx = L / n_cells
    dy = 0.1
    dz = 0.1

    # 节点
    points = []
    for i in range(n_cells + 1):
        x = i * dx
        points.append((x, 0.0, 0.0))
        points.append((x, dy, 0.0))
        points.append((x, dy, dz))
        points.append((x, 0.0, dz))

    n_points = len(points)

    # 面
    faces = []
    owner = []
    neighbour = []

    for i in range(n_cells - 1):
        faces.append((4, i * 4 + 0, i * 4 + 1, i * 4 + 2, i * 4 + 3))
        owner.append(i)
        neighbour.append(i + 1)

    n_internal = len(neighbour)

    # 入口
    inlet_start = n_internal
    faces.append((4, 0, 3, 2, 1))
    owner.append(0)

    # 出口
    outlet_start = inlet_start + 1
    level = n_cells
    faces.append((4, level * 4 + 0, level * 4 + 1, level * 4 + 2, level * 4 + 3))
    owner.append(n_cells - 1)

    # 空 patches
    empty_start = outlet_start + 1
    for i in range(n_cells):
        faces.append((4, i * 4 + 0, (i + 1) * 4 + 0, (i + 1) * 4 + 3, i * 4 + 3))
        owner.append(i)
    for i in range(n_cells):
        faces.append((4, i * 4 + 1, i * 4 + 2, (i + 1) * 4 + 2, (i + 1) * 4 + 1))
        owner.append(i)
    for i in range(n_cells):
        faces.append((4, i * 4 + 0, i * 4 + 1, (i + 1) * 4 + 1, (i + 1) * 4 + 0))
        owner.append(i)
    for i in range(n_cells):
        faces.append((4, i * 4 + 3, (i + 1) * 4 + 3, (i + 1) * 4 + 2, i * 4 + 2))
        owner.append(i)

    n_faces = len(faces)

    # 写网格文件
    mesh_dir = case_dir / "constant" / "polyMesh"
    mesh_dir.mkdir(parents=True, exist_ok=True)

    header_base = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII, location="constant/polyMesh",
    )

    h = FoamFileHeader(**{**header_base.__dict__, "class_name": "vectorField", "object": "points"})
    lines = [f"{n_points}", "("]
    for p in points:
        lines.append(f"({p[0]:.10g} {p[1]:.10g} {p[2]:.10g})")
    lines.append(")")
    write_foam_file(mesh_dir / "points", h, "\n".join(lines), overwrite=True)

    h = FoamFileHeader(**{**header_base.__dict__, "class_name": "faceList", "object": "faces"})
    lines = [f"{n_faces}", "("]
    for face in faces:
        verts = " ".join(str(v) for v in face[1:])
        lines.append(f"{face[0]}({verts})")
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
    n_empty = 4 * n_cells
    lines = ["2", "("]
    lines.append("    inlet")
    lines.append("    {")
    lines.append("        type            patch;")
    lines.append(f"        nFaces          1;")
    lines.append(f"        startFace       {inlet_start};")
    lines.append("    }")
    lines.append("    outlet")
    lines.append("    {")
    lines.append("        type            patch;")
    lines.append(f"        nFaces          1;")
    lines.append(f"        startFace       {outlet_start};")
    lines.append("    }")
    lines.append("    walls")
    lines.append("    {")
    lines.append("        type            empty;")
    lines.append(f"        nFaces          {n_empty};")
    lines.append(f"        startFace       {empty_start};")
    lines.append("    }")
    lines.append(")")
    write_foam_file(mesh_dir / "boundary", h, "\n".join(lines), overwrite=True)

    # 0/b
    zero_dir = case_dir / "0"
    zero_dir.mkdir(exist_ok=True)

    b_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="volScalarField", location="0", object="b",
    )
    write_foam_file(zero_dir / "b", b_header, (
        "dimensions      [0 0 0 0 0 0 0];\n\n"
        f"internalField   uniform {b_init};\n\n"
        "boundaryField\n{\n"
        "    inlet\n    {\n"
        "        type            fixedValue;\n"
        "        value           uniform 0;\n"
        "    }\n"
        "    outlet\n    {\n"
        "        type            zeroGradient;\n"
        "    }\n"
        "    walls\n    {\n"
        "        type            empty;\n"
        "    }\n"
        "}\n"
    ), overwrite=True)

    # 0/U
    u_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="volVectorField", location="0", object="U",
    )
    write_foam_file(zero_dir / "U", u_header, (
        "dimensions      [0 1 -1 0 0 0 0];\n\n"
        "internalField   uniform (1 0 0);\n\n"
        "boundaryField\n{\n"
        "    inlet\n    {\n"
        "        type            fixedValue;\n"
        "        value           uniform (1 0 0);\n"
        "    }\n"
        "    outlet\n    {\n"
        "        type            zeroGradient;\n"
        "    }\n"
        "    walls\n    {\n"
        "        type            empty;\n"
        "    }\n"
        "}\n"
    ), overwrite=True)

    # 0/p
    p_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="volScalarField", location="0", object="p",
    )
    write_foam_file(zero_dir / "p", p_header, (
        "dimensions      [1 -1 -2 0 0 0 0];\n\n"
        f"internalField   uniform {p_init};\n\n"
        "boundaryField\n{\n"
        "    inlet\n    {\n"
        "        type            zeroGradient;\n"
        "    }\n"
        "    outlet\n    {\n"
        "        type            zeroGradient;\n"
        "    }\n"
        "    walls\n    {\n"
        "        type            empty;\n"
        "    }\n"
        "}\n"
    ), overwrite=True)

    # 0/T
    t_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="volScalarField", location="0", object="T",
    )
    write_foam_file(zero_dir / "T", t_header, (
        "dimensions      [0 0 0 1 0 0 0];\n\n"
        "internalField   uniform 300;\n\n"
        "boundaryField\n{\n"
        "    inlet\n    {\n"
        "        type            fixedValue;\n"
        "        value           uniform 300;\n"
        "    }\n"
        "    outlet\n    {\n"
        "        type            zeroGradient;\n"
        "    }\n"
        "    walls\n    {\n"
        "        type            empty;\n"
        "    }\n"
        "}\n"
    ), overwrite=True)

    # system
    sys_dir = case_dir / "system"
    sys_dir.mkdir(exist_ok=True)

    cd_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="dictionary", location="system", object="controlDict",
    )
    write_foam_file(sys_dir / "controlDict", cd_header, (
        "application     pdrFoam;\n"
        "startTime       0;\n"
        f"endTime         {end_time:g};\n"
        f"deltaT          {delta_t:g};\n"
        "writeControl    timeStep;\n"
        f"writeInterval   100;\n"
    ), overwrite=True)

    fs_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="dictionary", location="system", object="fvSchemes",
    )
    write_foam_file(sys_dir / "fvSchemes", fs_header, (
        "ddtSchemes\n{\n    default         Euler;\n}\n\n"
        "divSchemes\n{\n    default         Gauss linear;\n}\n\n"
        "laplacianSchemes\n{\n    default         Gauss linear corrected;\n}\n"
    ), overwrite=True)

    fv_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="dictionary", location="system", object="fvSolution",
    )
    write_foam_file(sys_dir / "fvSolution", fv_header, (
        "solvers\n{\n"
        "    p\n    {\n"
        "        solver          PCG;\n"
        "        preconditioner  DIC;\n"
        "        tolerance       1e-6;\n"
        "        relTol          0.01;\n"
        "    }\n"
        "    U\n    {\n"
        "        solver          PBiCGStab;\n"
        "        preconditioner  DILU;\n"
        "        tolerance       1e-6;\n"
        "        relTol          0.01;\n"
        "    }\n"
        "    b\n    {\n"
        "        solver          PBiCGStab;\n"
        "        preconditioner  DILU;\n"
        "        tolerance       1e-6;\n"
        "        relTol          0.01;\n"
        "    }\n"
        "}\n\n"
        "PIMPLE\n{\n"
        f"    nOuterCorrectors    {n_outer_correctors};\n"
        f"    nCorrectors         {n_correctors};\n"
        "    nNonOrthogonalCorrectors 0;\n"
        "    convergenceTolerance 1e-4;\n"
        "    maxOuterIterations  100;\n"
        "    relaxationFactors\n    {\n"
        "        p               0.3;\n"
        "        U               0.7;\n"
        "        b               0.5;\n"
        "    }\n"
        "}\n"
    ), overwrite=True)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def pdr_case(tmp_path):
    """Create a 1D premixed combustion case."""
    case_dir = tmp_path / "pdr"
    _make_pdr_case(
        case_dir,
        n_cells=4,
        end_time=3e-5,
        delta_t=1e-5,
        n_outer_correctors=3,
        n_correctors=2,
    )
    return case_dir


# ---------------------------------------------------------------------------
# Tests — 初始化
# ---------------------------------------------------------------------------

class TestPDRFoamInit:
    """Tests for PDRFoam initialisation."""

    def test_case_loads(self, pdr_case):
        """Case directory is readable."""
        from pyfoam.io.case import Case
        case = Case(pdr_case)
        assert case.has_mesh()

    def test_solver_initialises(self, pdr_case):
        """PDRFoam initialises without errors."""
        from pyfoam.applications.pdr_foam import PDRFoam

        solver = PDRFoam(pdr_case)
        assert solver is not None
        assert solver.mesh is not None

    def test_progress_variable_shape(self, pdr_case):
        """Progress variable b has correct shape."""
        from pyfoam.applications.pdr_foam import PDRFoam

        solver = PDRFoam(pdr_case)
        n_cells = solver.mesh.n_cells
        assert solver.b.shape == (n_cells,)
        assert solver.U.shape == (n_cells, 3)
        assert solver.p.shape == (n_cells,)
        assert solver.T.shape == (n_cells,)

    def test_flame_parameters(self, pdr_case):
        """Flame parameters are set correctly."""
        from pyfoam.applications.pdr_foam import PDRFoam

        solver = PDRFoam(pdr_case, SL0=0.5, Xi0=3.0)
        assert abs(solver.SL0 - 0.5) < 1e-10
        assert abs(solver.Xi0 - 3.0) < 1e-10


# ---------------------------------------------------------------------------
# Tests — 火焰模型
# ---------------------------------------------------------------------------

class TestPDRFoamFlameModel:
    """Tests for b-Xi flame model computations."""

    def test_density_unburnt(self, pdr_case):
        """Density equals rho_unburnt when b=0."""
        from pyfoam.applications.pdr_foam import PDRFoam

        solver = PDRFoam(pdr_case)
        b_unburnt = torch.zeros(5, dtype=torch.float64)
        rho = solver._rho_from_b(b_unburnt)
        assert torch.allclose(
            rho,
            torch.full_like(rho, solver.rho_unburnt),
            rtol=1e-6,
        )

    def test_density_burnt(self, pdr_case):
        """Density equals rho_burnt when b=1."""
        from pyfoam.applications.pdr_foam import PDRFoam

        solver = PDRFoam(pdr_case)
        b_burnt = torch.ones(5, dtype=torch.float64)
        rho = solver._rho_from_b(b_burnt)
        assert torch.allclose(
            rho,
            torch.full_like(rho, solver.rho_burnt),
            rtol=1e-6,
        )

    def test_laminar_flame_speed_positive(self, pdr_case):
        """Laminar flame speed is always positive."""
        from pyfoam.applications.pdr_foam import PDRFoam

        solver = PDRFoam(pdr_case)
        T = torch.full((5,), 300.0, dtype=torch.float64)
        p = torch.full((5,), 101325.0, dtype=torch.float64)
        SL = solver._laminar_flame_speed(T, p)
        assert (SL > 0).all()
        assert torch.isfinite(SL).all()

    def test_turbulent_flame_speed_exceeds_laminar(self, pdr_case):
        """Turbulent flame speed >= laminar flame speed."""
        from pyfoam.applications.pdr_foam import PDRFoam

        solver = PDRFoam(pdr_case)
        SL = torch.full((5,), 0.4, dtype=torch.float64)
        u_prime = torch.full((5,), 1.0, dtype=torch.float64)
        ST, Xi = solver._turbulent_flame_speed(SL, u_prime)
        assert (ST >= SL).all(), "S_T < S_L"
        assert (Xi >= 1.0).all(), "Xi < 1"

    def test_temperature_from_progress(self, pdr_case):
        """Temperature increases from unburnt to adiabatic."""
        from pyfoam.applications.pdr_foam import PDRFoam

        solver = PDRFoam(pdr_case)
        T_unburnt = solver._T_from_b(torch.tensor(0.0, dtype=torch.float64))
        T_burnt = solver._T_from_b(torch.tensor(1.0, dtype=torch.float64))
        assert abs(float(T_unburnt) - solver.T_unburnt) < 1e-6
        assert abs(float(T_burnt) - solver.T_adiabatic) < 1e-6


# ---------------------------------------------------------------------------
# Tests — 求解器执行
# ---------------------------------------------------------------------------

class TestPDRFoamRun:
    """Tests for solver execution."""

    def test_run_completes(self, pdr_case):
        """Solver runs without errors."""
        from pyfoam.applications.pdr_foam import PDRFoam

        solver = PDRFoam(pdr_case)
        conv = solver.run()

        assert conv is not None
        assert conv.continuity_error >= 0

    def test_progress_bounded(self, pdr_case):
        """Progress variable b stays in [0, 1] after solving."""
        from pyfoam.applications.pdr_foam import PDRFoam

        solver = PDRFoam(pdr_case)
        solver.run()

        assert (solver.b >= 0.0).all(), "b < 0"
        assert (solver.b <= 1.0).all(), "b > 1"

    def test_fields_finite(self, pdr_case):
        """All fields are finite after solving."""
        from pyfoam.applications.pdr_foam import PDRFoam

        solver = PDRFoam(pdr_case)
        solver.run()

        assert torch.isfinite(solver.U).all(), "U contains NaN/Inf"
        assert torch.isfinite(solver.p).all(), "p contains NaN/Inf"
        assert torch.isfinite(solver.b).all(), "b contains NaN/Inf"
        assert torch.isfinite(solver.T).all(), "T contains NaN/Inf"

    def test_density_stays_positive(self, pdr_case):
        """Density remains positive throughout simulation."""
        from pyfoam.applications.pdr_foam import PDRFoam

        solver = PDRFoam(pdr_case)
        solver.run()

        assert (solver.rho > 0).all(), "Density became non-positive"

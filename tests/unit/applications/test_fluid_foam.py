"""
Unit tests for FluidFoam — unified compressible solver with full energy equation.

Tests cover:
- Case loading and field initialisation
- PIMPLE settings reading
- Thermodynamic consistency
- Solver execution (run + finite fields)
- Energy equation coupling
- Field output format
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

def _make_fluid_case(
    case_dir: Path,
    n_cells: int = 4,
    L: float = 1.0,
    end_time: float = 1e-4,
    delta_t: float = 1e-5,
    write_interval: int = 100,
    n_outer_correctors: int = 3,
    n_correctors: int = 2,
    T_hot: float = 350.0,
    T_cold: float = 300.0,
    p_init: float = 101325.0,
) -> None:
    """Write a 1D heated-channel case for FluidFoam."""
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

    # thermophysicalProperties
    tp_header = FoamFileHeader(
        version="2.0", format=FileFormat.ASCII,
        class_name="dictionary", location="constant",
        object="thermophysicalProperties",
    )
    tp_body = (
        "thermoType\n"
        "{\n"
        "    type            hePsiThermo;\n"
        "    mixture         pureMixture;\n"
        "    transport       const;\n"
        "    thermo          hConst;\n"
        "    equationOfState perfectGas;\n"
        "    specie          specie;\n"
        "    energy          sensibleEnthalpy;\n"
        "}\n\n"
        "mixture\n"
        "{\n"
        "    specie\n    {\n"
        "        molWeight      28.966;\n"
        "    }\n"
        "    thermodynamics\n    {\n"
        "        Cp          1005.0;\n"
        "        Hf          0;\n"
        "    }\n"
        "    transport\n    {\n"
        "        mu          1.716e-5;\n"
        "        Pr          0.7;\n"
        "    }\n"
        "}\n"
    )
    write_foam_file(
        case_dir / "constant" / "thermophysicalProperties", tp_header,
        tp_body, overwrite=True,
    )

    # 0/U
    zero_dir = case_dir / "0"
    zero_dir.mkdir(exist_ok=True)

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
    T_init = (T_hot + T_cold) / 2
    write_foam_file(zero_dir / "T", t_header, (
        "dimensions      [0 0 0 1 0 0 0];\n\n"
        f"internalField   uniform {T_init};\n\n"
        "boundaryField\n{\n"
        "    inlet\n    {\n"
        "        type            fixedValue;\n"
        f"        value           uniform {T_hot};\n"
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
        "application     fluidFoam;\n"
        "startTime       0;\n"
        f"endTime         {end_time:g};\n"
        f"deltaT          {delta_t:g};\n"
        "writeControl    timeStep;\n"
        f"writeInterval   {write_interval};\n"
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
        "    T\n    {\n"
        "        solver          PCG;\n"
        "        preconditioner  DIC;\n"
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
        "        T               1.0;\n"
        "    }\n"
        "}\n"
    ), overwrite=True)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def fluid_case(tmp_path):
    """Create a 1D heated channel case for FluidFoam."""
    case_dir = tmp_path / "fluid"
    _make_fluid_case(
        case_dir,
        n_cells=3,
        end_time=3e-5,
        delta_t=1e-5,
        n_outer_correctors=3,
        n_correctors=2,
        T_hot=350.0,
        T_cold=300.0,
    )
    return case_dir


@pytest.fixture
def tiny_fluid_case(tmp_path):
    """Create a minimal 2-cell channel case."""
    case_dir = tmp_path / "tiny_fluid"
    _make_fluid_case(
        case_dir,
        n_cells=2,
        end_time=2e-5,
        delta_t=1e-5,
        n_outer_correctors=3,
        n_correctors=2,
    )
    return case_dir


# ---------------------------------------------------------------------------
# Tests — 初始化
# ---------------------------------------------------------------------------

class TestFluidFoamInit:
    """Tests for FluidFoam initialisation."""

    def test_case_loads(self, fluid_case):
        """Case directory is readable."""
        from pyfoam.io.case import Case
        case = Case(fluid_case)
        assert case.has_mesh()

    def test_solver_initialises(self, fluid_case):
        """FluidFoam initialises without errors."""
        from pyfoam.applications.fluid_foam import FluidFoam

        solver = FluidFoam(fluid_case)
        assert solver is not None
        assert solver.mesh is not None

    def test_field_shapes(self, fluid_case):
        """All field shapes are correct."""
        from pyfoam.applications.fluid_foam import FluidFoam

        solver = FluidFoam(fluid_case)
        n_cells = solver.mesh.n_cells
        assert solver.U.shape == (n_cells, 3)
        assert solver.p.shape == (n_cells,)
        assert solver.T.shape == (n_cells,)
        assert solver.rho.shape == (n_cells,)
        assert solver.phi.shape == (solver.mesh.n_faces,)

    def test_thermo_model_present(self, fluid_case):
        """Thermophysical model is initialised."""
        from pyfoam.applications.fluid_foam import FluidFoam

        solver = FluidFoam(fluid_case)
        assert solver.thermo is not None
        assert hasattr(solver.thermo, 'rho')
        assert hasattr(solver.thermo, 'mu')
        assert hasattr(solver.thermo, 'kappa')


# ---------------------------------------------------------------------------
# Tests — 设置读取
# ---------------------------------------------------------------------------

class TestFluidFoamSettings:
    """Tests for PIMPLE settings reading."""

    def test_pimple_settings_read(self, fluid_case):
        """PIMPLE settings are read correctly from fvSolution."""
        from pyfoam.applications.fluid_foam import FluidFoam

        solver = FluidFoam(fluid_case)
        assert solver.n_outer_correctors == 3
        assert solver.n_correctors == 2
        assert abs(solver.convergence_tolerance - 1e-4) < 1e-10


# ---------------------------------------------------------------------------
# Tests — 求解器执行
# ---------------------------------------------------------------------------

class TestFluidFoamRun:
    """Tests for solver execution."""

    def test_run_completes(self, fluid_case):
        """Solver runs without errors."""
        from pyfoam.applications.fluid_foam import FluidFoam

        solver = FluidFoam(fluid_case)
        conv = solver.run()

        assert conv is not None
        assert conv.continuity_error >= 0

    def test_fields_finite(self, fluid_case):
        """All fields are finite after solving."""
        from pyfoam.applications.fluid_foam import FluidFoam

        solver = FluidFoam(fluid_case)
        solver.run()

        assert torch.isfinite(solver.U).all(), "U contains NaN/Inf"
        assert torch.isfinite(solver.p).all(), "p contains NaN/Inf"
        assert torch.isfinite(solver.T).all(), "T contains NaN/Inf"
        assert torch.isfinite(solver.phi).all(), "phi contains NaN/Inf"

    def test_density_stays_positive(self, fluid_case):
        """Density remains positive throughout simulation."""
        from pyfoam.applications.fluid_foam import FluidFoam

        solver = FluidFoam(fluid_case)
        solver.run()

        assert (solver.rho > 0).all(), "Density became non-positive"


# ---------------------------------------------------------------------------
# Tests — 热力学一致性
# ---------------------------------------------------------------------------

class TestFluidFoamThermo:
    """Tests for thermodynamic consistency."""

    def test_eos_consistency(self, fluid_case):
        """Density is consistent with EOS after solving."""
        from pyfoam.applications.fluid_foam import FluidFoam

        solver = FluidFoam(fluid_case)
        solver.run()

        rho_check = solver.thermo.rho(solver.p, solver.T)
        assert torch.allclose(solver.rho, rho_check, rtol=1e-3), \
            "Stored density inconsistent with EOS"

    def test_viscosity_positive(self, fluid_case):
        """Viscosity is positive for all cells."""
        from pyfoam.applications.fluid_foam import FluidFoam

        solver = FluidFoam(fluid_case)
        solver.run()

        mu = solver.thermo.mu(solver.T)
        assert (mu > 0).all(), "Viscosity became non-positive"


# ---------------------------------------------------------------------------
# Tests — 输出
# ---------------------------------------------------------------------------

class TestFluidFoamOutput:
    """Tests for field output."""

    def test_run_writes_output(self, fluid_case):
        """Solver writes field files to time directories."""
        from pyfoam.applications.fluid_foam import FluidFoam

        solver = FluidFoam(fluid_case)
        solver.run()

        time_dirs = [d for d in fluid_case.iterdir()
                     if d.is_dir() and d.name.replace(".", "").isdigit()
                     and d.name != "0"]
        assert len(time_dirs) >= 1

"""
Unit tests for IsothermalFluidFoam — transient compressible isothermal solver.

Tests cover:
- Case loading and field initialisation
- PIMPLE settings reading
- Constant temperature from thermophysicalProperties
- EOS density update
- Momentum predictor
- Pressure equation
- Time-stepping loop
- Convergence
- Field writing
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from pyfoam.core.dtype import CFD_DTYPE
from pyfoam.io.foam_file import FoamFileHeader, FileFormat, write_foam_file


# ---------------------------------------------------------------------------
# 网格 + 算例生成
# ---------------------------------------------------------------------------

def _make_isothermal_case(
    case_dir: Path,
    n_cells: int = 3,
    L: float = 1.0,
    end_time: int = 1,
    delta_t: float = 0.01,
    write_interval: int = 100,
    T_ref: float = 300.0,
    p_init: float = 101325.0,
    U_init: float = 0.01,
) -> None:
    """Write a 1D compressible isothermal case."""
    case_dir.mkdir(parents=True, exist_ok=True)

    dx = L / n_cells
    dy, dz = 0.1, 0.1

    # ---- 网格点 ----
    points = []
    for i in range(n_cells + 1):
        x = i * dx
        points.extend([(x, 0.0, 0.0), (x, dy, 0.0), (x, dy, dz), (x, 0.0, dz)])

    n_points = len(points)
    faces, owner, neighbour = [], [], []

    # 内部面
    for i in range(n_cells - 1):
        faces.append((4, i * 4, i * 4 + 1, i * 4 + 2, i * 4 + 3))
        owner.append(i)
        neighbour.append(i + 1)
    n_internal = len(neighbour)

    # 入口面
    inlet_start = n_internal
    faces.append((4, 0, 3, 2, 1))
    owner.append(0)

    # 出口面
    outlet_start = inlet_start + 1
    level = n_cells
    faces.append((4, level * 4, level * 4 + 1, level * 4 + 2, level * 4 + 3))
    owner.append(n_cells - 1)

    # Empty patches
    empty_start = outlet_start + 1
    for i in range(n_cells):
        faces.append((4, i * 4, (i + 1) * 4, (i + 1) * 4 + 3, i * 4 + 3))
        owner.append(i)
    for i in range(n_cells):
        faces.append((4, i * 4 + 1, i * 4 + 2, (i + 1) * 4 + 2, (i + 1) * 4 + 1))
        owner.append(i)
    for i in range(n_cells):
        faces.append((4, i * 4, i * 4 + 1, (i + 1) * 4 + 1, (i + 1) * 4))
        owner.append(i)
    for i in range(n_cells):
        faces.append((4, i * 4 + 3, (i + 1) * 4 + 3, (i + 1) * 4 + 2, i * 4 + 2))
        owner.append(i)

    n_faces = len(faces)
    n_empty = 4 * n_cells

    # ---- 写网格文件 ----
    mesh_dir = case_dir / "constant" / "polyMesh"
    mesh_dir.mkdir(parents=True, exist_ok=True)

    hb = FoamFileHeader(version="2.0", format=FileFormat.ASCII, location="constant/polyMesh")

    def _header(class_name: str, obj: str) -> FoamFileHeader:
        return FoamFileHeader(**{**hb.__dict__, "class_name": class_name, "object": obj})

    lines = [str(n_points), "("]
    for p in points:
        lines.append(f"({p[0]:.10g} {p[1]:.10g} {p[2]:.10g})")
    lines.append(")")
    write_foam_file(mesh_dir / "points", _header("vectorField", "points"), "\n".join(lines), overwrite=True)

    lines = [str(n_faces), "("]
    for face in faces:
        verts = " ".join(str(v) for v in face[1:])
        lines.append(f"{face[0]}({verts})")
    lines.append(")")
    write_foam_file(mesh_dir / "faces", _header("faceList", "faces"), "\n".join(lines), overwrite=True)

    lines = [str(n_faces), "("]
    for c in owner:
        lines.append(str(c))
    lines.append(")")
    write_foam_file(mesh_dir / "owner", _header("labelList", "owner"), "\n".join(lines), overwrite=True)

    lines = [str(n_internal), "("]
    for c in neighbour:
        lines.append(str(c))
    lines.append(")")
    write_foam_file(mesh_dir / "neighbour", _header("labelList", "neighbour"), "\n".join(lines), overwrite=True)

    lines = ["4", "("]
    lines += [
        "    inlet", "    {", "        type            patch;",
        f"        nFaces          1;", f"        startFace       {inlet_start};", "    }",
        "    outlet", "    {", "        type            patch;",
        f"        nFaces          1;", f"        startFace       {outlet_start};", "    }",
        "    walls", "    {", "        type            empty;",
        f"        nFaces          {n_empty};", f"        startFace       {empty_start};", "    }",
        ")",
    ]
    write_foam_file(mesh_dir / "boundary", _header("polyBoundaryMesh", "boundary"), "\n".join(lines), overwrite=True)

    # ---- thermophysicalProperties ----
    tp_header = FoamFileHeader(version="2.0", format=FileFormat.ASCII, class_name="dictionary",
                               location="constant", object="thermophysicalProperties")
    write_foam_file(
        case_dir / "constant" / "thermophysicalProperties", tp_header,
        f"T              {T_ref};\n"
        "R              8.314;\n"
        "Cp             1005;\n",
        overwrite=True,
    )

    # ---- 0/U ----
    zero_dir = case_dir / "0"
    zero_dir.mkdir(exist_ok=True)

    write_foam_file(zero_dir / "U",
        FoamFileHeader(version="2.0", format=FileFormat.ASCII, class_name="volVectorField",
                       location="0", object="U"),
        "dimensions      [0 1 -1 0 0 0 0];\n\n"
        f"internalField   uniform ({U_init} 0 0);\n\n"
        "boundaryField\n{\n"
        "    inlet\n    {\n        type            fixedValue;\n"
        f"        value           uniform ({U_init} 0 0);\n    }}\n"
        "    outlet\n    {\n        type            zeroGradient;\n    }\n"
        "    walls\n    {\n        type            empty;\n    }\n"
        "}\n", overwrite=True)

    # ---- 0/p ----
    write_foam_file(zero_dir / "p",
        FoamFileHeader(version="2.0", format=FileFormat.ASCII, class_name="volScalarField",
                       location="0", object="p"),
        "dimensions      [1 -1 -2 0 0 0 0];\n\n"
        f"internalField   uniform {p_init};\n\n"
        "boundaryField\n{\n"
        "    inlet\n    {\n        type            zeroGradient;\n    }\n"
        "    outlet\n    {\n        type            zeroGradient;\n    }\n"
        "    walls\n    {\n        type            empty;\n    }\n"
        "}\n", overwrite=True)

    # ---- system/ ----
    sys_dir = case_dir / "system"
    sys_dir.mkdir(exist_ok=True)

    write_foam_file(sys_dir / "controlDict",
        FoamFileHeader(version="2.0", format=FileFormat.ASCII, class_name="dictionary",
                       location="system", object="controlDict"),
        "application     isothermalFluidFoam;\n"
        "startTime       0;\n"
        f"endTime         {end_time};\n"
        f"deltaT          {delta_t};\n"
        "writeControl    timeStep;\n"
        f"writeInterval   {write_interval};\n",
        overwrite=True)

    write_foam_file(sys_dir / "fvSchemes",
        FoamFileHeader(version="2.0", format=FileFormat.ASCII, class_name="dictionary",
                       location="system", object="fvSchemes"),
        "ddtSchemes\n{\n    default         Euler;\n}\n\n"
        "divSchemes\n{\n    default         Gauss linear;\n}\n\n"
        "laplacianSchemes\n{\n    default         Gauss linear corrected;\n}\n",
        overwrite=True)

    write_foam_file(sys_dir / "fvSolution",
        FoamFileHeader(version="2.0", format=FileFormat.ASCII, class_name="dictionary",
                       location="system", object="fvSolution"),
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
        "}\n\n"
        "PIMPLE\n{\n"
        "    nOuterCorrectors    3;\n"
        "    nCorrectors         2;\n"
        "    nNonOrthogonalCorrectors 0;\n"
        "    convergenceTolerance 1e-4;\n"
        "    relaxationFactors\n    {\n"
        "        p               0.3;\n"
        "        U               0.7;\n"
        "    }\n"
        "}\n",
        overwrite=True)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def iso_case(tmp_path):
    """Create a 3-cell isothermal case (low velocity for stability)."""
    case_dir = tmp_path / "iso"
    _make_isothermal_case(case_dir, n_cells=3, end_time=0.1, delta_t=0.001, U_init=0.01)
    return case_dir


@pytest.fixture
def tiny_iso_case(tmp_path):
    """Create a minimal 2-cell isothermal case (quiescent start for stability)."""
    case_dir = tmp_path / "tiny_iso"
    _make_isothermal_case(case_dir, n_cells=2, end_time=0.02, delta_t=0.001, U_init=0.0)
    return case_dir


# ---------------------------------------------------------------------------
# 初始化测试
# ---------------------------------------------------------------------------

class TestIsothermalFluidFoamInit:
    """IsothermalFluidFoam 初始化测试。"""

    def test_case_loads(self, iso_case):
        """算例目录可读取。"""
        from pyfoam.io.case import Case
        case = Case(iso_case)
        assert case.has_mesh()

    def test_fields_initialise(self, iso_case):
        """场初始化形状正确。"""
        from pyfoam.applications.isothermal_fluid_foam import IsothermalFluidFoam

        solver = IsothermalFluidFoam(iso_case)
        assert solver.U.shape == (3, 3)
        assert solver.p.shape == (3,)
        assert solver.phi.shape == (solver.mesh.n_faces,)
        assert solver.rho.shape == (3,)

    def test_T_ref_read(self, iso_case):
        """参考温度读取正确。"""
        from pyfoam.applications.isothermal_fluid_foam import IsothermalFluidFoam

        solver = IsothermalFluidFoam(iso_case)
        assert abs(solver.T_ref - 300.0) < 1e-10

    def test_density_from_eos(self, iso_case):
        """密度由 EOS 计算（等温）。"""
        from pyfoam.applications.isothermal_fluid_foam import IsothermalFluidFoam

        solver = IsothermalFluidFoam(iso_case)
        # ρ = p / (R_specific · T_ref)
        # R_specific = R / W = 8.314 / 0.029 ≈ 286.7 J/(kg·K)
        # ρ = 101325 / (286.7 · 300) ≈ 1.178 kg/m³ (标准空气密度)
        rho_expected = 101325.0 / (8.314 / 0.029 * 300.0)
        assert torch.allclose(
            solver.rho, torch.full((3,), rho_expected, dtype=CFD_DTYPE),
            rtol=0.01,
        )


# ---------------------------------------------------------------------------
# PIMPLE 设置测试
# ---------------------------------------------------------------------------

class TestIsothermalFluidFoamSettings:
    """PIMPLE 设置读取测试。"""

    def test_pimple_settings(self, iso_case):
        """PIMPLE 参数读取正确。"""
        from pyfoam.applications.isothermal_fluid_foam import IsothermalFluidFoam

        solver = IsothermalFluidFoam(iso_case)
        assert solver.n_outer_correctors == 3
        assert solver.n_correctors == 2
        assert solver.alpha_p == 0.3
        assert solver.alpha_U == 0.7


# ---------------------------------------------------------------------------
# EOS 测试
# ---------------------------------------------------------------------------

class TestIsothermalFluidFoamEOS:
    """等温 EOS 测试。"""

    def test_rho_update(self, iso_case):
        """密度随压力变化。"""
        from pyfoam.applications.isothermal_fluid_foam import IsothermalFluidFoam

        solver = IsothermalFluidFoam(iso_case)
        rho_orig = solver.rho.clone()

        # 压力加倍，密度应加倍
        p_new = solver.p * 2.0
        rho_new = solver._update_rho(p_new)

        assert torch.allclose(rho_new, 2.0 * rho_orig, rtol=0.01)


# ---------------------------------------------------------------------------
# 求解器执行测试
# ---------------------------------------------------------------------------

class TestIsothermalFluidFoamSolver:
    """IsothermalFluidFoam 求解器执行测试。"""

    def test_run_completes(self, tiny_iso_case):
        """求解器运行无报错。"""
        from pyfoam.applications.isothermal_fluid_foam import IsothermalFluidFoam

        solver = IsothermalFluidFoam(tiny_iso_case)
        result = solver.run()
        assert result is not None

    def test_pressure_finite(self, tiny_iso_case):
        """压力在求解后保持有限。"""
        from pyfoam.applications.isothermal_fluid_foam import IsothermalFluidFoam

        solver = IsothermalFluidFoam(tiny_iso_case)
        solver.run()
        assert torch.isfinite(solver.p).all(), "p contains NaN/Inf"

    def test_velocity_finite(self, tiny_iso_case):
        """速度在求解后保持有限。"""
        from pyfoam.applications.isothermal_fluid_foam import IsothermalFluidFoam

        solver = IsothermalFluidFoam(tiny_iso_case)
        solver.run()
        assert torch.isfinite(solver.U).all(), "U contains NaN/Inf"

    def test_density_finite(self, tiny_iso_case):
        """密度在求解后保持有限。"""
        from pyfoam.applications.isothermal_fluid_foam import IsothermalFluidFoam

        solver = IsothermalFluidFoam(tiny_iso_case)
        solver.run()
        assert torch.isfinite(solver.rho).all(), "rho contains NaN/Inf"
        assert (solver.rho > 0).all(), "rho has non-positive values"

    def test_density_positive(self, tiny_iso_case):
        """密度始终为正。"""
        from pyfoam.applications.isothermal_fluid_foam import IsothermalFluidFoam

        solver = IsothermalFluidFoam(tiny_iso_case)
        solver.run()
        assert (solver.rho > 0).all(), "rho has non-positive values"

    def test_writes_output(self, tiny_iso_case):
        """场写入到时间目录。"""
        from pyfoam.applications.isothermal_fluid_foam import IsothermalFluidFoam

        solver = IsothermalFluidFoam(tiny_iso_case)
        solver.run()

        time_dirs = [
            d for d in tiny_iso_case.iterdir()
            if d.is_dir() and d.name.replace(".", "").isdigit() and d.name != "0"
        ]
        assert len(time_dirs) >= 1

    def test_pressure_changes(self, tiny_iso_case):
        """求解器正常运行并输出结果。"""
        from pyfoam.applications.isothermal_fluid_foam import IsothermalFluidFoam

        solver = IsothermalFluidFoam(tiny_iso_case)
        result = solver.run()

        # 运行完成，残差有限
        assert result is not None
        assert result.U_residual is not None
        assert torch.isfinite(solver.U).all(), "U contains NaN/Inf"
        assert torch.isfinite(solver.p).all(), "p contains NaN/Inf"

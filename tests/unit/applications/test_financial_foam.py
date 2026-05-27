"""
Unit tests for FinancialFoam — Black-Scholes equation solver.

Tests cover:
- Case loading and field initialisation
- Option payoff computation (call and put)
- Theta-method time stepping (explicit and implicit)
- Thomas algorithm for tridiagonal systems
- Boundary condition application
- Solver run to completion
- Option pricing output
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from pyfoam.core.dtype import CFD_DTYPE
from pyfoam.io.foam_file import FoamFileHeader, FileFormat, write_foam_file


# ---------------------------------------------------------------------------
# Case generation helper
# ---------------------------------------------------------------------------

def _make_financial_case(
    case_dir: Path,
    n_cells: int = 20,
    end_time: int = 1,
    delta_t: float = 0.01,
) -> None:
    """Write a 1D case for financialFoam."""
    case_dir.mkdir(parents=True, exist_ok=True)

    dx = 1.0 / n_cells
    dy, dz = 0.1, 0.1

    # 网格点
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
    n_empty = 0
    for i in range(n_cells):
        faces.append((4, i * 4, (i + 1) * 4, (i + 1) * 4 + 3, i * 4 + 3))
        owner.append(i)
    n_empty += n_cells
    for i in range(n_cells):
        faces.append((4, i * 4 + 1, i * 4 + 2, (i + 1) * 4 + 2, (i + 1) * 4 + 1))
        owner.append(i)
    n_empty += n_cells
    for i in range(n_cells):
        faces.append((4, i * 4, i * 4 + 1, (i + 1) * 4 + 1, i * 4))
        owner.append(i)
    n_empty += n_cells
    for i in range(n_cells):
        faces.append((4, i * 4 + 3, (i + 1) * 4 + 3, (i + 1) * 4 + 2, i * 4 + 2))
        owner.append(i)
    n_empty += n_cells

    n_faces = len(faces)

    # 写网格文件
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

    lines = ["2", "("]
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

    # 0/ 目录
    zero_dir = case_dir / "0"
    zero_dir.mkdir(exist_ok=True)

    # V (期权价值)
    write_foam_file(zero_dir / "V",
        FoamFileHeader(version="2.0", format=FileFormat.ASCII, class_name="volScalarField",
                       location="0", object="V"),
        "dimensions      [0 0 0 0 0 0 0];\n\n"
        "internalField   uniform 0;\n\n"
        "boundaryField\n{\n"
        "    inlet\n    {\n        type            fixedValue;\n"
        "        value           uniform 0;\n    }\n"
        "    outlet\n    {\n        type            zeroGradient;\n    }\n"
        "    walls\n    {\n        type            empty;\n    }\n"
        "}\n", overwrite=True)

    # U (SolverBase 需要)
    write_foam_file(zero_dir / "U",
        FoamFileHeader(version="2.0", format=FileFormat.ASCII, class_name="volVectorField",
                       location="0", object="U"),
        "dimensions      [0 1 -1 0 0 0 0];\n\n"
        "internalField   uniform (0 0 0);\n\n"
        "boundaryField\n{\n"
        "    inlet\n    {\n        type            fixedValue;\n"
        "        value           uniform (0 0 0);\n    }\n"
        "    outlet\n    {\n        type            zeroGradient;\n    }\n"
        "    walls\n    {\n        type            empty;\n    }\n"
        "}\n", overwrite=True)

    # p (SolverBase 需要)
    write_foam_file(zero_dir / "p",
        FoamFileHeader(version="2.0", format=FileFormat.ASCII, class_name="volScalarField",
                       location="0", object="p"),
        "dimensions      [1 -1 -2 0 0 0 0];\n\n"
        "internalField   uniform 101325;\n\n"
        "boundaryField\n{\n"
        "    inlet\n    {\n        type            zeroGradient;\n    }\n"
        "    outlet\n    {\n        type            zeroGradient;\n    }\n"
        "    walls\n    {\n        type            empty;\n    }\n"
        "}\n", overwrite=True)

    # system/
    sys_dir = case_dir / "system"
    sys_dir.mkdir(exist_ok=True)

    write_foam_file(sys_dir / "controlDict",
        FoamFileHeader(version="2.0", format=FileFormat.ASCII, class_name="dictionary",
                       location="system", object="controlDict"),
        "application     financialFoam;\n"
        "startTime       0;\n"
        f"endTime         {end_time};\n"
        f"deltaT          {delta_t};\n"
        "writeControl    timeStep;\n"
        "writeInterval   100;\n",
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
        "financialFoam\n{\n"
        "    convergenceTolerance 1e-6;\n"
        "    scheme              implicit;\n"
        "}\n",
        overwrite=True)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def financial_case(tmp_path):
    """Create a 20-cell financial case."""
    case_dir = tmp_path / "financial"
    _make_financial_case(case_dir, n_cells=20, end_time=1, delta_t=0.01)
    return case_dir


@pytest.fixture
def call_solver(financial_case):
    """Create a call option solver."""
    from pyfoam.applications.financial_foam import FinancialFoam
    return FinancialFoam(
        financial_case, option_type="call",
        K=100.0, r=0.05, sigma=0.2, S_max=300.0,
    )


@pytest.fixture
def put_solver(financial_case):
    """Create a put option solver."""
    from pyfoam.applications.financial_foam import FinancialFoam
    return FinancialFoam(
        financial_case, option_type="put",
        K=100.0, r=0.05, sigma=0.2, S_max=300.0,
    )


# ===========================================================================
# Tests
# ===========================================================================


class TestFinancialFoamInit:
    """FinancialFoam 初始化测试。"""

    def test_case_loads(self, financial_case):
        """算例目录可读取。"""
        from pyfoam.io.case import Case
        case = Case(financial_case)
        assert case.has_mesh()

    def test_solver_creates_call(self, financial_case):
        """Call option solver 创建成功。"""
        from pyfoam.applications.financial_foam import FinancialFoam

        solver = FinancialFoam(
            financial_case, option_type="call",
            K=100.0, r=0.05, sigma=0.2, S_max=300.0,
        )
        assert solver.option_type == "call"
        assert solver.K == 100.0

    def test_solver_creates_put(self, financial_case):
        """Put option solver 创建成功。"""
        from pyfoam.applications.financial_foam import FinancialFoam

        solver = FinancialFoam(
            financial_case, option_type="put",
            K=100.0, r=0.05, sigma=0.2, S_max=300.0,
        )
        assert solver.option_type == "put"


class TestFinancialFoamPayoff:
    """期权收益函数测试。"""

    def test_call_payoff(self, call_solver):
        """Call payoff = max(S-K, 0)。"""
        solver = call_solver
        # 期权价值应非负
        assert (solver.V >= 0).all(), "Option value must be non-negative"

    def test_put_payoff(self, put_solver):
        """Put payoff = max(K-S, 0)。"""
        solver = put_solver
        assert (solver.V >= 0).all(), "Option value must be non-negative"

    def test_intrinsic_value_call(self, call_solver):
        """Call 内在价值正确。"""
        solver = call_solver
        iv = solver.intrinsic_value
        # S > K 时内在价值为正
        mask = solver.S > solver.K
        assert (iv[mask] > 0).all()

    def test_intrinsic_value_put(self, put_solver):
        """Put 内在价值正确。"""
        solver = put_solver
        iv = solver.intrinsic_value
        # S < K 时内在价值为正
        mask = solver.S < solver.K
        assert (iv[mask] > 0).all()


class TestFinancialFoamThomasAlgorithm:
    """Thomas 算法测试。"""

    def test_thomas_algorithm_simple(self, financial_case):
        """Thomas 算法求解简单三对角系统。"""
        from pyfoam.applications.financial_foam import FinancialFoam

        # 构造简单系统: 2x + y = 5, x + 2y = 4 → x=2, y=1
        main = torch.tensor([2.0, 2.0])
        lower = torch.tensor([0.0, 1.0])
        upper = torch.tensor([1.0, 0.0])
        rhs = torch.tensor([5.0, 4.0])

        x = FinancialFoam._thomas_algorithm(lower, main, upper, rhs)
        assert abs(x[0].item() - 2.0) < 1e-10
        assert abs(x[1].item() - 1.0) < 1e-10

    def test_thomas_algorithm_identity(self, financial_case):
        """Thomas 算法处理单位矩阵。"""
        from pyfoam.applications.financial_foam import FinancialFoam

        n = 5
        main = torch.ones(n)
        lower = torch.zeros(n)
        upper = torch.zeros(n)
        rhs = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])

        x = FinancialFoam._thomas_algorithm(lower, main, upper, rhs)
        assert torch.allclose(x, rhs, atol=1e-12)


class TestFinancialFoamSolver:
    """FinancialFoam 求解器执行测试。"""

    def test_run_completes_call(self, call_solver):
        """Call option solver 运行无报错。"""
        solver = call_solver
        solver.end_time = 0.1
        result = solver.run()
        assert "converged" in result
        assert "V_at_K" in result

    def test_run_completes_put(self, put_solver):
        """Put option solver 运行无报错。"""
        solver = put_solver
        solver.end_time = 0.1
        result = solver.run()
        assert "converged" in result
        assert "V_at_K" in result

    def test_option_values_finite(self, call_solver):
        """期权价值在求解后保持有限。"""
        solver = call_solver
        solver.end_time = 0.1
        solver.run()
        assert torch.isfinite(solver.V).all(), "V contains NaN/Inf"

    def test_option_values_non_negative(self, call_solver):
        """期权价值在求解后保持非负。"""
        solver = call_solver
        solver.end_time = 0.1
        solver.run()
        assert (solver.V >= 0).all(), "Option value must be non-negative"

    def test_explicit_solver(self, financial_case):
        """显式求解器可运行。"""
        from pyfoam.applications.financial_foam import FinancialFoam

        solver = FinancialFoam(
            financial_case, option_type="call",
            K=100.0, r=0.05, sigma=0.2, S_max=300.0,
            theta=0.0,
        )
        solver.end_time = 0.01
        solver.delta_t = 0.0001  # 显式需要小时间步
        result = solver.run()
        assert "converged" in result

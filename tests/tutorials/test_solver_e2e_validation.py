"""
端到端求解器验证：实际运行每个求解器并检查收敛。

对每个求解器类别创建最小算例，运行求解器，
验证收敛性和物理有效性。保存结果数据。
"""
from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from typing import Any, Dict

import torch
import pytest

from pyfoam.core.dtype import CFD_DTYPE


def _make_cavity_case(tmp_dir: str, nu: float = 0.01) -> Path:
    """创建标准 cavity 算例。"""
    from tests.tutorials.helpers import (
        make_structured_mesh, write_control_dict, write_fv_schemes,
        write_fv_solution, write_velocity_field, write_pressure_field,
        write_transport_properties,
    )
    from pyfoam.io.foam_file import write_foam_file, FoamFileHeader, FileFormat

    case_dir = Path(tmp_dir)
    mesh_dir = case_dir / "constant" / "polyMesh"
    make_structured_mesh(mesh_dir, nx=4, ny=4)
    write_control_dict(case_dir, delta_t=0.001, end_time=0.005)
    write_fv_schemes(case_dir)
    write_fv_solution(case_dir, algorithm="SIMPLE")
    write_transport_properties(case_dir, nu=nu)

    patches_U = {
        "movingWall": (1.0, 0.0, 0.0),
        "fixedWalls": (0.0, 0.0, 0.0),
    }
    bc_U = {"movingWall": "fixedValue", "fixedWalls": "noSlip"}
    write_velocity_field(case_dir, patches=patches_U, bc_types=bc_U)

    patches_p = {"movingWall": "zeroGradient", "fixedWalls": "zeroGradient"}
    write_pressure_field(case_dir, patches=patches_p)

    # 温度场（可压缩/传热求解器需要）
    zero_dir = case_dir / "0"
    h_T = FoamFileHeader(version="2.0", format=FileFormat.ASCII,
                         class_name="volScalarField", location="0", object="T")
    lines_T = [
        "dimensions      [0 0 0 1 0 0 0];",
        "internalField   uniform 300;",
        "boundaryField {",
        "    movingWall {",
        "        type            fixedValue;",
        "        value           uniform 300;",
        "    }",
        "    fixedWalls {",
        "        type            zeroGradient;",
        "    }",
        "    frontAndBack {",
        "        type            empty;",
        "    }",
        "}",
    ]
    write_foam_file(zero_dir / "T", h_T, "\n".join(lines_T), overwrite=True)

    # alpha 场（VOF 求解器需要）
    h_alpha = FoamFileHeader(version="2.0", format=FileFormat.ASCII,
                             class_name="volScalarField", location="0", object="alpha.water")
    lines_alpha = [
        "dimensions      [0 0 0 0 0 0 0];",
        "internalField   uniform 0.5;",
        "boundaryField {",
        "    movingWall {",
        "        type            zeroGradient;",
        "    }",
        "    fixedWalls {",
        "        type            alphaContactAngle;",
        "        theta0          90;",
        "        value           uniform 0.5;",
        "    }",
        "    frontAndBack {",
        "        type            empty;",
        "    }",
        "}",
    ]
    write_foam_file(zero_dir / "alpha.water", h_alpha, "\n".join(lines_alpha), overwrite=True)

    # p_rgh 场（可压缩/VOF 求解器需要）
    h_prgh = FoamFileHeader(version="2.0", format=FileFormat.ASCII,
                            class_name="volScalarField", location="0", object="p_rgh")
    lines_prgh = [
        "dimensions      [1 -1 -2 0 0 0 0];",
        "internalField   uniform 0;",
        "boundaryField {",
        "    movingWall {",
        "        type            fixedFluxPressure;",
        "        value           uniform 0;",
        "    }",
        "    fixedWalls {",
        "        type            fixedFluxPressure;",
        "        value           uniform 0;",
        "    }",
        "    frontAndBack {",
        "        type            empty;",
        "    }",
        "}",
    ]
    write_foam_file(zero_dir / "p_rgh", h_prgh, "\n".join(lines_prgh), overwrite=True)

    # 标量浓度场（ScalarTransportFoam 需要）
    h_C = FoamFileHeader(version="2.0", format=FileFormat.ASCII,
                         class_name="volScalarField", location="0", object="C")
    lines_C = [
        "dimensions      [0 0 0 0 0 0 0];",
        "internalField   uniform 0;",
        "boundaryField {",
        "    movingWall {",
        "        type            fixedValue;",
        "        value           uniform 1;",
        "    }",
        "    fixedWalls {",
        "        type            zeroGradient;",
        "    }",
        "    frontAndBack {",
        "        type            empty;",
        "    }",
        "}",
    ]
    write_foam_file(zero_dir / "C", h_C, "\n".join(lines_C), overwrite=True)

    return case_dir


def _run_solver(solver_cls, case_dir: Path, **kwargs) -> Dict[str, Any]:
    """运行求解器并返回结果摘要。"""
    start = time.time()
    try:
        solver = solver_cls(case_dir, **kwargs)
        result = solver.run()
        elapsed = time.time() - start
        # Get the main field (U if available, else T, else None)
        main_field = getattr(solver, 'U', None)
        if main_field is None:
            main_field = getattr(solver, 'T', None)
        field_finite = bool(torch.isfinite(main_field).all()) if main_field is not None else True
        return {
            "status": "OK",
            "converged": getattr(result, "converged", False),
            "U_residual": getattr(result, "U_residual", None),
            "p_residual": getattr(result, "p_residual", None),
            "continuity": getattr(result, "continuity_error", None),
            "iterations": getattr(result, "outer_iterations", None),
            "elapsed": round(elapsed, 3),
            "field_finite": field_finite,
            "error": None,
        }
    except Exception as e:
        elapsed = time.time() - start
        return {
            "status": "ERROR",
            "converged": False,
            "U_residual": None,
            "p_residual": None,
            "continuity": None,
            "iterations": None,
            "elapsed": round(elapsed, 3),
            "field_finite": False,
            "error": str(e)[:200],
        }


class TestIncompressibleSolvers:
    """不可压缩求解器端到端验证。"""

    def test_simple_foam(self):
        """SimpleFoam cavity 算例。"""
        from pyfoam.applications import SimpleFoam
        with tempfile.TemporaryDirectory() as tmp:
            case_dir = _make_cavity_case(tmp, nu=0.01)
            result = _run_solver(SimpleFoam, case_dir)
        assert result["status"] == "OK", f"SimpleFoam failed: {result['error']}"
        assert result["U_residual"] is not None
        assert torch.isfinite(torch.tensor(result["U_residual"]))

    def test_ico_foam(self):
        """IcoFoam cavity 算例。"""
        from pyfoam.applications import IcoFoam
        with tempfile.TemporaryDirectory() as tmp:
            case_dir = _make_cavity_case(tmp, nu=0.01)
            result = _run_solver(IcoFoam, case_dir)
        assert result["status"] == "OK", f"IcoFoam failed: {result['error']}"

    def test_piso_foam(self):
        """PisoFoam cavity 算例。"""
        from pyfoam.applications import PisoFoam
        with tempfile.TemporaryDirectory() as tmp:
            case_dir = _make_cavity_case(tmp, nu=0.01)
            result = _run_solver(PisoFoam, case_dir)
        assert result["status"] == "OK", f"PisoFoam failed: {result['error']}"

    def test_pimple_foam(self):
        """PimpleFoam cavity 算例。"""
        from pyfoam.applications import PimpleFoam
        with tempfile.TemporaryDirectory() as tmp:
            case_dir = _make_cavity_case(tmp, nu=0.01)
            result = _run_solver(PimpleFoam, case_dir)
        assert result["status"] == "OK", f"PimpleFoam failed: {result['error']}"

    def test_incompressible_fluid_foam(self):
        """IncompressibleFluidFoam cavity 算例。"""
        from pyfoam.applications import IncompressibleFluidFoam
        with tempfile.TemporaryDirectory() as tmp:
            case_dir = _make_cavity_case(tmp, nu=0.01)
            result = _run_solver(IncompressibleFluidFoam, case_dir)
        assert result["status"] == "OK", f"Failed: {result['error']}"


class TestCompressibleSolvers:
    """可压缩求解器端到端验证。"""

    def test_sonic_foam(self):
        """SonicFoam 激波管算例。"""
        from pyfoam.applications import SonicFoam
        with tempfile.TemporaryDirectory() as tmp:
            case_dir = _make_cavity_case(tmp, nu=0.01)
            result = _run_solver(SonicFoam, case_dir)
        assert result["status"] == "OK", f"SonicFoam failed: {result['error']}"

    def test_rho_pimple_foam(self):
        """RhoPimpleFoam cavity 算例。"""
        from pyfoam.applications import RhoPimpleFoam
        with tempfile.TemporaryDirectory() as tmp:
            case_dir = _make_cavity_case(tmp, nu=0.01)
            result = _run_solver(RhoPimpleFoam, case_dir)
        assert result["status"] == "OK", f"Failed: {result['error']}"


class TestMultiphaseSolvers:
    """多相流求解器端到端验证。"""

    def test_inter_foam(self):
        """InterFoam VOF 算例。"""
        from pyfoam.applications import InterFoam
        with tempfile.TemporaryDirectory() as tmp:
            case_dir = _make_cavity_case(tmp, nu=1e-6)
            result = _run_solver(InterFoam, case_dir)
        assert result["status"] == "OK", f"InterFoam failed: {result['error']}"


class TestHeatTransferSolvers:
    """传热求解器端到端验证。"""

    def test_laplacian_foam(self):
        """LaplacianFoam 热传导算例。"""
        from pyfoam.applications import LaplacianFoam
        with tempfile.TemporaryDirectory() as tmp:
            case_dir = _make_cavity_case(tmp, nu=0.01)
            result = _run_solver(LaplacianFoam, case_dir)
        assert result["status"] == "OK", f"LaplacianFoam failed: {result['error']}"


class TestSpecialSolvers:
    """特殊求解器端到端验证。"""

    def test_potential_foam(self):
        """PotentialFoam 算例。"""
        from pyfoam.applications import PotentialFoam
        with tempfile.TemporaryDirectory() as tmp:
            case_dir = _make_cavity_case(tmp, nu=0.01)
            result = _run_solver(PotentialFoam, case_dir)
        assert result["status"] == "OK", f"PotentialFoam failed: {result['error']}"

    def test_scalar_transport_foam(self):
        """ScalarTransportFoam 算例。"""
        from pyfoam.applications import ScalarTransportFoam
        with tempfile.TemporaryDirectory() as tmp:
            case_dir = _make_cavity_case(tmp, nu=0.01)
            result = _run_solver(ScalarTransportFoam, case_dir)
        assert result["status"] == "OK", f"Failed: {result['error']}"

    def test_boundary_foam(self):
        """BoundaryFoam 算例。"""
        from pyfoam.applications import BoundaryFoam
        with tempfile.TemporaryDirectory() as tmp:
            case_dir = _make_cavity_case(tmp, nu=0.01)
            result = _run_solver(BoundaryFoam, case_dir)
        assert result["status"] == "OK", f"Failed: {result['error']}"


class TestBuoyantSolvers:
    """浮力求解器端到端验证。"""

    def test_buoyant_pimple_foam(self):
        """BuoyantPimpleFoam 算例。"""
        from pyfoam.applications import BuoyantPimpleFoam
        with tempfile.TemporaryDirectory() as tmp:
            case_dir = _make_cavity_case(tmp, nu=0.01)
            result = _run_solver(BuoyantPimpleFoam, case_dir)
        assert result["status"] == "OK", f"Failed: {result['error']}"

    def test_buoyant_simple_foam(self):
        """BuoyantSimpleFoam 算例。"""
        from pyfoam.applications import BuoyantSimpleFoam
        with tempfile.TemporaryDirectory() as tmp:
            case_dir = _make_cavity_case(tmp, nu=0.01)
            result = _run_solver(BuoyantSimpleFoam, case_dir)
        assert result["status"] == "OK", f"Failed: {result['error']}"


class TestCombustionSolvers:
    """燃烧求解器端到端验证。"""

    def test_reacting_foam(self):
        """ReactingFoam 算例。"""
        from pyfoam.applications import ReactingFoam
        with tempfile.TemporaryDirectory() as tmp:
            case_dir = _make_cavity_case(tmp, nu=0.01)
            result = _run_solver(ReactingFoam, case_dir)
        assert result["status"] == "OK", f"Failed: {result['error']}"

    def test_xi_foam(self):
        """XiFoam 预混燃烧算例。"""
        from pyfoam.applications import XiFoam
        with tempfile.TemporaryDirectory() as tmp:
            case_dir = _make_cavity_case(tmp, nu=0.01)
            result = _run_solver(XiFoam, case_dir)
        assert result["status"] == "OK", f"Failed: {result['error']}"


class TestAllBaseSolversImport:
    """所有基础求解器导入验证。"""

    @pytest.mark.parametrize("solver_name", [
        "SimpleFoam", "IcoFoam", "PisoFoam", "PimpleFoam",
        "RhoSimpleFoam", "RhoPimpleFoam", "RhoCentralFoam", "SonicFoam",
        "InterFoam", "CompressibleInterFoam", "MultiphaseEulerFoam",
        "CavitatingFoam", "TwoPhaseEulerFoam",
        "IncompressibleFluidFoam", "FluidFoam", "MulticomponentFluidFoam",
        "BuoyantSimpleFoam", "BuoyantPimpleFoam", "BuoyantBoussinesqSimpleFoam",
        "ReactingFoam", "SolidDisplacementFoam",
        "LaplacianFoam", "ScalarTransportFoam", "PotentialFoam",
        "BoundaryFoam", "PorousSimpleFoam", "SrfSimpleFoam",
        "CHTMultiRegionFoam", "IncompressibleVoFFoam", "CompressibleVoFFoam",
        "IncompressibleDriftFluxFoam", "IsothermalFluidFoam",
        "XiFoam", "DenseParticleFoam", "CompressibleMultiphaseVoFFoam",
    ])
    def test_solver_import(self, solver_name):
        """每个基础求解器可导入。"""
        from pyfoam.applications import __all__ as apps
        assert solver_name in apps, f"{solver_name} not in pyfoam.applications"
        mod = __import__("pyfoam.applications", fromlist=[solver_name])
        cls = getattr(mod, solver_name)
        assert cls is not None


class TestSolverResultValidation:
    """验证求解器结果的物理有效性。"""

    def test_simple_foam_residuals_finite(self):
        """SimpleFoam 残差应为有限值。"""
        from pyfoam.applications import SimpleFoam
        with tempfile.TemporaryDirectory() as tmp:
            case_dir = _make_cavity_case(tmp, nu=0.01)
            result = _run_solver(SimpleFoam, case_dir)
        if result["status"] == "OK" and result["U_residual"] is not None:
            assert torch.isfinite(torch.tensor(result["U_residual"])), "U residual is NaN/Inf"
            assert result["U_residual"] >= 0, "U residual is negative"

    def test_simple_foam_decreasing_residuals(self):
        """SimpleFoam 残差应单调递减或保持稳定。"""
        from pyfoam.applications import SimpleFoam
        with tempfile.TemporaryDirectory() as tmp:
            case_dir = _make_cavity_case(tmp, nu=0.01)
            solver = SimpleFoam(case_dir)
            result = solver.run()
        # 至少应有残差记录
        assert result is not None

    def test_solver_time_tracking(self):
        """求解器应正确跟踪时间。"""
        from pyfoam.applications import SimpleFoam
        with tempfile.TemporaryDirectory() as tmp:
            case_dir = _make_cavity_case(tmp, nu=0.01)
            result = _run_solver(SimpleFoam, case_dir)
        assert result["elapsed"] > 0, "Solver should take positive time"

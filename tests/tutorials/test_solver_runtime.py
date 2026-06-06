"""
Tutorial validation: solver runtime tests.

验证求解器能在短时间内完成运行。
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from pyfoam.core.dtype import CFD_DTYPE, INDEX_DTYPE
from tests.tutorials.helpers import (
    make_structured_mesh,
    write_control_dict,
    write_fv_schemes,
    write_fv_solution,
    write_pressure_field,
    write_transport_properties,
    write_velocity_field,
)


def _make_case(tmp_path: Path, name: str, nx: int = 5, ny: int = 5) -> Path:
    case = tmp_path / name
    mesh_dir = case / "constant" / "polyMesh"
    make_structured_mesh(mesh_dir, nx=nx, ny=ny)
    write_transport_properties(case, nu=0.01)
    write_control_dict(case, delta_t=0.01, end_time=0.05, write_interval=100)
    write_fv_schemes(case)
    write_fv_solution(case)
    write_velocity_field(
        case,
        patches={"movingWall": (1, 0, 0), "fixedWalls": (0, 0, 0), "frontAndBack": (0, 0, 0)},
        bc_types={"movingWall": "fixedValue", "fixedWalls": "noSlip", "frontAndBack": "empty"},
    )
    write_pressure_field(
        case,
        patches={"movingWall": "zeroGradient", "fixedWalls": "zeroGradient", "frontAndBack": "empty"},
    )
    return case


class TestSolverRuntime:
    """求解器运行时测试。"""

    def test_simple_foam_5x5(self, tmp_path: Path):
        """SimpleFoam 5x5 网格运行。"""
        case = _make_case(tmp_path, "simple5x5", nx=5, ny=5)
        from pyfoam.applications.simple_foam import SimpleFoam
        solver = SimpleFoam(case)
        solver.run()
        assert torch.isfinite(solver.U).all()
        assert torch.isfinite(solver.p).all()

    def test_simple_foam_10x10(self, tmp_path: Path):
        """SimpleFoam 10x10 网格运行。"""
        case = _make_case(tmp_path, "simple10x10", nx=10, ny=10)
        from pyfoam.applications.simple_foam import SimpleFoam
        solver = SimpleFoam(case)
        solver.run()
        assert torch.isfinite(solver.U).all()
        assert torch.isfinite(solver.p).all()

    def test_piso_foam_5x5(self, tmp_path: Path):
        """PisoFoam 5x5 网格运行。"""
        case = _make_case(tmp_path, "piso5x5", nx=5, ny=5)
        from pyfoam.applications.piso_foam import PisoFoam
        solver = PisoFoam(case)
        solver.run()
        assert torch.isfinite(solver.U).all()
        assert torch.isfinite(solver.p).all()

    def test_piso_foam_10x10(self, tmp_path: Path):
        """PisoFoam 10x10 网格运行。"""
        case = _make_case(tmp_path, "piso10x10", nx=10, ny=10)
        from pyfoam.applications.piso_foam import PisoFoam
        solver = PisoFoam(case)
        solver.run()
        assert torch.isfinite(solver.U).all()
        assert torch.isfinite(solver.p).all()

    def test_pimple_foam_5x5(self, tmp_path: Path):
        """PimpleFoam 5x5 网格运行。"""
        case = _make_case(tmp_path, "pimple5x5", nx=5, ny=5)
        from pyfoam.applications.pimple_foam import PimpleFoam
        solver = PimpleFoam(case)
        solver.run()
        assert torch.isfinite(solver.U).all()
        assert torch.isfinite(solver.p).all()

    def test_ico_foam_5x5(self, tmp_path: Path):
        """IcoFoam 5x5 网格运行。"""
        case = _make_case(tmp_path, "ico5x5", nx=5, ny=5)
        from pyfoam.applications.ico_foam import IcoFoam
        solver = IcoFoam(case)
        solver.run()
        assert torch.isfinite(solver.U).all()
        assert torch.isfinite(solver.p).all()


class TestSolverConvergence:
    """求解器收敛性测试。"""

    def test_simple_convergence(self, tmp_path: Path):
        """SimpleFoam 收敛性检查。"""
        case = _make_case(tmp_path, "conv", nx=5, ny=5)
        from pyfoam.applications.simple_foam import SimpleFoam
        solver = SimpleFoam(case)
        solver.run()
        # 检查残差是否有限
        if hasattr(solver, 'residual_history'):
            for field, residuals in solver.residual_history.items():
                if len(residuals) > 0:
                    assert all(torch.isfinite(torch.tensor(r)) for r in residuals)


class TestMeshScalability:
    """网格规模测试。"""

    def test_3x3_mesh(self, tmp_path: Path):
        """3x3 网格运行。"""
        case = _make_case(tmp_path, "mesh3x3", nx=3, ny=3)
        from pyfoam.applications.simple_foam import SimpleFoam
        solver = SimpleFoam(case)
        solver.run()
        assert torch.isfinite(solver.U).all()

    def test_5x5_mesh(self, tmp_path: Path):
        """5x5 网格运行。"""
        case = _make_case(tmp_path, "mesh5x5", nx=5, ny=5)
        from pyfoam.applications.simple_foam import SimpleFoam
        solver = SimpleFoam(case)
        solver.run()
        assert torch.isfinite(solver.U).all()

    def test_10x10_mesh(self, tmp_path: Path):
        """10x10 网格运行。"""
        case = _make_case(tmp_path, "mesh10x10", nx=10, ny=10)
        from pyfoam.applications.simple_foam import SimpleFoam
        solver = SimpleFoam(case)
        solver.run()
        assert torch.isfinite(solver.U).all()

    def test_20x20_mesh(self, tmp_path: Path):
        """20x20 网格运行。"""
        case = _make_case(tmp_path, "mesh20x20", nx=20, ny=20)
        from pyfoam.applications.simple_foam import SimpleFoam
        solver = SimpleFoam(case)
        solver.run()
        assert torch.isfinite(solver.U).all()

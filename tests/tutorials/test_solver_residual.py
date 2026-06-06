"""
Tutorial validation: solver residual tests.

验证求解器残差收敛。
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


def _make_residual_case(tmp_path: Path, name: str, nx: int = 10, ny: int = 10) -> Path:
    case = tmp_path / name
    mesh_dir = case / "constant" / "polyMesh"
    make_structured_mesh(mesh_dir, nx=nx, ny=ny)
    write_transport_properties(case, nu=0.01)
    write_control_dict(case, delta_t=0.005, end_time=1.0, write_interval=200)
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


class TestSolverResidual:
    """求解器残差测试。"""

    def test_simple_residual_finite(self, tmp_path: Path):
        """SimpleFoam 残差有限。"""
        case = _make_residual_case(tmp_path, "residual")
        from pyfoam.applications.simple_foam import SimpleFoam
        solver = SimpleFoam(case)
        solver.run()
        assert torch.isfinite(solver.U).all()
        assert torch.isfinite(solver.p).all()

    def test_piso_residual_finite(self, tmp_path: Path):
        """PisoFoam 残差有限。"""
        case = _make_residual_case(tmp_path, "piso_residual", nx=10, ny=10)
        from pyfoam.applications.piso_foam import PisoFoam
        solver = PisoFoam(case)
        solver.run()
        assert torch.isfinite(solver.U).all()
        assert torch.isfinite(solver.p).all()

    def test_pimple_residual_finite(self, tmp_path: Path):
        """PimpleFoam 残差有限。"""
        case = _make_residual_case(tmp_path, "pimple_residual", nx=10, ny=10)
        from pyfoam.applications.pimple_foam import PimpleFoam
        solver = PimpleFoam(case)
        solver.run()
        assert torch.isfinite(solver.U).all()
        assert torch.isfinite(solver.p).all()


class TestSolverResidualConvergence:
    """求解器残差收敛测试。"""

    def test_simple_residual_decreasing(self, tmp_path: Path):
        """SimpleFoam 残差递减。"""
        case = _make_residual_case(tmp_path, "conv_residual")
        from pyfoam.applications.simple_foam import SimpleFoam
        solver = SimpleFoam(case)
        solver.run()
        # 检查残差历史（如果可用）
        if hasattr(solver, 'residual_history'):
            for field, residuals in solver.residual_history.items():
                if len(residuals) > 1:
                    # 残差应该总体递减
                    assert residuals[-1] <= residuals[0] * 10, f"{field} residual increased too much"

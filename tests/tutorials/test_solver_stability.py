"""
Tutorial validation: solver numerical stability tests.

验证求解器数值稳定性。
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


def _make_stability_case(
    tmp_path: Path,
    name: str,
    nu: float = 0.01,
    delta_t: float = 0.01,
    end_time: float = 0.05,
    nx: int = 5,
    ny: int = 5,
) -> Path:
    case = tmp_path / name
    mesh_dir = case / "constant" / "polyMesh"
    make_structured_mesh(mesh_dir, nx=nx, ny=ny)
    write_transport_properties(case, nu=nu)
    write_control_dict(case, delta_t=delta_t, end_time=end_time, write_interval=100)
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


class TestNumericalStability:
    """数值稳定性测试。"""

    def test_low_reynolds_stability(self, tmp_path: Path):
        """低雷诺数稳定性。"""
        case = _make_stability_case(tmp_path, "low_re", nu=1.0)
        from pyfoam.applications.simple_foam import SimpleFoam
        solver = SimpleFoam(case)
        solver.run()
        assert torch.isfinite(solver.U).all()

    def test_medium_reynolds_stability(self, tmp_path: Path):
        """中雷诺数稳定性。"""
        case = _make_stability_case(tmp_path, "med_re", nu=0.01)
        from pyfoam.applications.simple_foam import SimpleFoam
        solver = SimpleFoam(case)
        solver.run()
        assert torch.isfinite(solver.U).all()

    def test_small_time_step_stability(self, tmp_path: Path):
        """小时间步稳定性。"""
        case = _make_stability_case(tmp_path, "small_dt", delta_t=0.001, end_time=0.01)
        from pyfoam.applications.simple_foam import SimpleFoam
        solver = SimpleFoam(case)
        solver.run()
        assert torch.isfinite(solver.U).all()

    def test_large_time_step_stability(self, tmp_path: Path):
        """大时间步稳定性。"""
        case = _make_stability_case(tmp_path, "large_dt", delta_t=0.1, end_time=0.5)
        from pyfoam.applications.simple_foam import SimpleFoam
        solver = SimpleFoam(case)
        solver.run()
        assert torch.isfinite(solver.U).all()


class TestNumericalAccuracy:
    """数值精度测试。"""

    def test_fine_mesh_accuracy(self, tmp_path: Path):
        """细网格精度。"""
        case = _make_stability_case(tmp_path, "fine", nx=20, ny=20)
        from pyfoam.applications.simple_foam import SimpleFoam
        solver = SimpleFoam(case)
        solver.run()
        assert torch.isfinite(solver.U).all()
        assert solver.U.abs().sum() > 0

    def test_coarse_mesh_accuracy(self, tmp_path: Path):
        """粗网格精度。"""
        case = _make_stability_case(tmp_path, "coarse", nx=5, ny=5)
        from pyfoam.applications.simple_foam import SimpleFoam
        solver = SimpleFoam(case)
        solver.run()
        assert torch.isfinite(solver.U).all()
        assert solver.U.abs().sum() > 0

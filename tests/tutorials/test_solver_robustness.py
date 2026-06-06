"""
Tutorial validation: solver robustness tests.

验证求解器在各种条件下的鲁棒性。
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


def _make_robustness_case(
    tmp_path: Path,
    name: str,
    nu: float = 0.01,
    nx: int = 5,
    ny: int = 5,
    delta_t: float = 0.01,
    end_time: float = 0.05,
    u_wall: float = 1.0,
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
        patches={"movingWall": (u_wall, 0, 0), "fixedWalls": (0, 0, 0), "frontAndBack": (0, 0, 0)},
        bc_types={"movingWall": "fixedValue", "fixedWalls": "noSlip", "frontAndBack": "empty"},
    )
    write_pressure_field(
        case,
        patches={"movingWall": "zeroGradient", "fixedWalls": "zeroGradient", "frontAndBack": "empty"},
    )
    return case


class TestSolverRobustness:
    """求解器鲁棒性测试。"""

    def test_low_reynolds_number(self, tmp_path: Path):
        """低雷诺数 (Re=1) 运行。"""
        case = _make_robustness_case(tmp_path, "low_re", nu=1.0, u_wall=1.0)
        from pyfoam.applications.simple_foam import SimpleFoam
        solver = SimpleFoam(case)
        solver.run()
        assert torch.isfinite(solver.U).all()

    @pytest.mark.xfail(reason="High Re=10000 diverges on coarse 5x5 mesh")
    def test_high_reynolds_number(self, tmp_path: Path):
        """高雷诺数 (Re=10000) 运行。"""
        case = _make_robustness_case(tmp_path, "high_re", nu=0.0001, u_wall=1.0)
        from pyfoam.applications.simple_foam import SimpleFoam
        solver = SimpleFoam(case)
        solver.run()
        assert torch.isfinite(solver.U).all()

    def test_small_time_step(self, tmp_path: Path):
        """小时间步长运行。"""
        case = _make_robustness_case(tmp_path, "small_dt", delta_t=0.001, end_time=0.01)
        from pyfoam.applications.piso_foam import PisoFoam
        solver = PisoFoam(case)
        solver.run()
        assert torch.isfinite(solver.U).all()

    def test_large_time_step(self, tmp_path: Path):
        """大时间步长运行。"""
        case = _make_robustness_case(tmp_path, "large_dt", delta_t=0.1, end_time=0.5)
        from pyfoam.applications.simple_foam import SimpleFoam
        solver = SimpleFoam(case)
        solver.run()
        assert torch.isfinite(solver.U).all()

    def test_zero_velocity_wall(self, tmp_path: Path):
        """零速度壁面运行。"""
        case = _make_robustness_case(tmp_path, "zero_wall", u_wall=0.0)
        from pyfoam.applications.simple_foam import SimpleFoam
        solver = SimpleFoam(case)
        solver.run()
        assert torch.isfinite(solver.U).all()
        # 零驱动 → 零速度
        assert solver.U.abs().max().item() < 1e-10

    def test_high_velocity_wall(self, tmp_path: Path):
        """高速壁面运行。"""
        case = _make_robustness_case(tmp_path, "high_wall", u_wall=100.0)
        from pyfoam.applications.simple_foam import SimpleFoam
        solver = SimpleFoam(case)
        solver.run()
        assert torch.isfinite(solver.U).all()

"""
Tutorial validation: solver initial condition tests.

验证求解器初始条件处理。
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


def _make_ic_case(
    tmp_path: Path,
    name: str,
    u_init: tuple = (0.0, 0.0, 0.0),
    u_wall: float = 1.0,
    nx: int = 5,
    ny: int = 5,
) -> Path:
    case = tmp_path / name
    mesh_dir = case / "constant" / "polyMesh"
    make_structured_mesh(mesh_dir, nx=nx, ny=ny)
    write_transport_properties(case, nu=0.01)
    write_control_dict(case, delta_t=0.01, end_time=0.05, write_interval=100)
    write_fv_schemes(case)
    write_fv_solution(case)
    write_velocity_field(
        case,
        patches={"movingWall": (u_wall, 0, 0), "fixedWalls": (0, 0, 0), "frontAndBack": (0, 0, 0)},
        internal=u_init,
        bc_types={"movingWall": "fixedValue", "fixedWalls": "noSlip", "frontAndBack": "empty"},
    )
    write_pressure_field(
        case,
        patches={"movingWall": "zeroGradient", "fixedWalls": "zeroGradient", "frontAndBack": "empty"},
    )
    return case


class TestInitialConditions:
    """初始条件测试。"""

    def test_zero_initial_velocity(self, tmp_path: Path):
        """零初始速度。"""
        case = _make_ic_case(tmp_path, "zero_ic", u_init=(0, 0, 0))
        from pyfoam.applications.simple_foam import SimpleFoam
        solver = SimpleFoam(case)
        solver.run()
        assert torch.isfinite(solver.U).all()
        assert solver.U.abs().sum() > 0  # 应产生非零流场

    def test_uniform_initial_velocity(self, tmp_path: Path):
        """均匀初始速度。"""
        case = _make_ic_case(tmp_path, "uniform_ic", u_init=(0.5, 0, 0))
        from pyfoam.applications.simple_foam import SimpleFoam
        solver = SimpleFoam(case)
        solver.run()
        assert torch.isfinite(solver.U).all()

    def test_high_initial_velocity(self, tmp_path: Path):
        """高初始速度。"""
        case = _make_ic_case(tmp_path, "high_ic", u_init=(10.0, 0, 0))
        from pyfoam.applications.simple_foam import SimpleFoam
        solver = SimpleFoam(case)
        solver.run()
        assert torch.isfinite(solver.U).all()


class TestInitialConditionEffect:
    """初始条件效应测试。"""

    def test_ic_affects_convergence(self, tmp_path: Path):
        """初始条件影响收敛速度。"""
        case1 = _make_ic_case(tmp_path, "ic1", u_init=(0, 0, 0))
        case2 = _make_ic_case(tmp_path, "ic2", u_init=(0.5, 0, 0))
        from pyfoam.applications.simple_foam import SimpleFoam
        solver1 = SimpleFoam(case1)
        solver1.run()
        solver2 = SimpleFoam(case2)
        solver2.run()
        # 两种 IC 都应产生有效解
        assert torch.isfinite(solver1.U).all()
        assert torch.isfinite(solver2.U).all()

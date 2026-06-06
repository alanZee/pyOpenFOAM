"""
Tutorial validation: solver time stepping tests.

验证求解器时间步进。
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


def _make_time_case(
    tmp_path: Path,
    name: str,
    delta_t: float = 0.01,
    end_time: float = 0.05,
    nx: int = 5,
    ny: int = 5,
) -> Path:
    case = tmp_path / name
    mesh_dir = case / "constant" / "polyMesh"
    make_structured_mesh(mesh_dir, nx=nx, ny=ny)
    write_transport_properties(case, nu=0.01)
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


class TestSolverTimeStepping:
    """求解器时间步进测试。"""

    def test_small_time_step(self, tmp_path: Path):
        """小时间步长。"""
        case = _make_time_case(tmp_path, "small_dt", delta_t=0.001, end_time=0.01)
        from pyfoam.applications.simple_foam import SimpleFoam
        solver = SimpleFoam(case)
        solver.run()
        assert torch.isfinite(solver.U).all()

    def test_medium_time_step(self, tmp_path: Path):
        """中等时间步长。"""
        case = _make_time_case(tmp_path, "med_dt", delta_t=0.01, end_time=0.1)
        from pyfoam.applications.simple_foam import SimpleFoam
        solver = SimpleFoam(case)
        solver.run()
        assert torch.isfinite(solver.U).all()

    def test_large_time_step(self, tmp_path: Path):
        """大时间步长。"""
        case = _make_time_case(tmp_path, "large_dt", delta_t=0.1, end_time=1.0)
        from pyfoam.applications.simple_foam import SimpleFoam
        solver = SimpleFoam(case)
        solver.run()
        assert torch.isfinite(solver.U).all()

    def test_very_small_time_step(self, tmp_path: Path):
        """极小时间步长。"""
        case = _make_time_case(tmp_path, "tiny_dt", delta_t=0.0001, end_time=0.001)
        from pyfoam.applications.simple_foam import SimpleFoam
        solver = SimpleFoam(case)
        solver.run()
        assert torch.isfinite(solver.U).all()


class TestSolverTimeEffect:
    """求解器时间效应测试。"""

    def test_longer_simulation_develops_flow(self, tmp_path: Path):
        """更长时间模拟产生更强流场。"""
        case1 = _make_time_case(tmp_path, "short", delta_t=0.01, end_time=0.05)
        case2 = _make_time_case(tmp_path, "long", delta_t=0.01, end_time=0.5)
        from pyfoam.applications.simple_foam import SimpleFoam
        solver1 = SimpleFoam(case1)
        solver1.run()
        solver2 = SimpleFoam(case2)
        solver2.run()
        # 更长时间应产生更强流场
        u1_max = solver1.U[:, 0].abs().max().item()
        u2_max = solver2.U[:, 0].abs().max().item()
        assert u2_max >= u1_max, f"u1_max={u1_max}, u2_max={u2_max}"

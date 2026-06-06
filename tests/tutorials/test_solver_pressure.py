"""
Tutorial validation: solver pressure reference tests.

验证求解器压力参考。
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


def _make_pressure_case(tmp_path: Path, name: str, nx: int = 10, ny: int = 10) -> Path:
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


class TestPressureReference:
    """压力参考测试。"""

    def test_zero_pressure_reference(self, tmp_path: Path):
        """零压力参考。"""
        case = _make_pressure_case(tmp_path, "zero_pref")
        from pyfoam.applications.simple_foam import SimpleFoam
        solver = SimpleFoam(case)
        solver.run()
        assert torch.isfinite(solver.p).all()

    def test_pressure_field_finite(self, tmp_path: Path):
        """压力场有限。"""
        case = _make_pressure_case(tmp_path, "p_finite")
        from pyfoam.applications.simple_foam import SimpleFoam
        solver = SimpleFoam(case)
        solver.run()
        assert torch.isfinite(solver.p).all()
        assert solver.p.abs().max().item() < 1000.0


class TestPressureEffect:
    """压力效应测试。"""

    def test_pressure_gradient_exists(self, tmp_path: Path):
        """压力梯度存在。"""
        case = _make_pressure_case(tmp_path, "p_grad")
        from pyfoam.applications.simple_foam import SimpleFoam
        solver = SimpleFoam(case)
        solver.run()
        # 检查压力场不均匀
        p_range = solver.p.max() - solver.p.min()
        assert p_range > 0, "Pressure field is uniform"

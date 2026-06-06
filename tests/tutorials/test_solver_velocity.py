"""
Tutorial validation: solver velocity field tests.

验证求解器速度场处理。
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


def _make_velocity_case(tmp_path: Path, name: str, nx: int = 10, ny: int = 10) -> Path:
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


class TestVelocityField:
    """速度场测试。"""

    def test_velocity_field_shape(self, tmp_path: Path):
        """速度场形状正确。"""
        case = _make_velocity_case(tmp_path, "shape")
        from pyfoam.applications.simple_foam import SimpleFoam
        solver = SimpleFoam(case)
        solver.run()
        n_cells = solver.mesh.n_cells
        assert solver.U.shape == (n_cells, 3)

    def test_velocity_field_finite(self, tmp_path: Path):
        """速度场有限。"""
        case = _make_velocity_case(tmp_path, "finite")
        from pyfoam.applications.simple_foam import SimpleFoam
        solver = SimpleFoam(case)
        solver.run()
        assert torch.isfinite(solver.U).all()

    def test_velocity_field_nonzero(self, tmp_path: Path):
        """速度场非零。"""
        case = _make_velocity_case(tmp_path, "nonzero")
        from pyfoam.applications.simple_foam import SimpleFoam
        solver = SimpleFoam(case)
        solver.run()
        assert solver.U.abs().sum() > 0

    def test_velocity_field_bounded(self, tmp_path: Path):
        """速度场有界。"""
        case = _make_velocity_case(tmp_path, "bounded")
        from pyfoam.applications.simple_foam import SimpleFoam
        solver = SimpleFoam(case)
        solver.run()
        u_max = solver.U.abs().max().item()
        assert u_max < 100.0, f"Velocity too high: {u_max}"


class TestVelocityEffect:
    """速度效应测试。"""

    def test_wall_velocity_effect(self, tmp_path: Path):
        """壁面速度效应。"""
        case = _make_velocity_case(tmp_path, "wall_effect")
        from pyfoam.applications.simple_foam import SimpleFoam
        solver = SimpleFoam(case)
        solver.run()
        # 检查速度场有梯度
        u_range = solver.U[:, 0].max() - solver.U[:, 0].min()
        assert u_range > 0, "No velocity gradient"

"""
Tutorial validation: solver boundary condition comprehensive tests.

全面验证求解器边界条件处理。
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


def _make_bc_comprehensive_case(
    tmp_path: Path,
    name: str,
    u_wall: float = 1.0,
    nu: float = 0.01,
    nx: int = 10,
    ny: int = 10,
) -> Path:
    case = tmp_path / name
    mesh_dir = case / "constant" / "polyMesh"
    make_structured_mesh(mesh_dir, nx=nx, ny=ny)
    write_transport_properties(case, nu=nu)
    write_control_dict(case, delta_t=0.005, end_time=1.0, write_interval=200)
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


class TestBoundaryConditionComprehensive:
    """全面边界条件测试。"""

    def test_moving_wall_drives_flow(self, tmp_path: Path):
        """运动壁面驱动流场。"""
        case = _make_bc_comprehensive_case(tmp_path, "bc_drive")
        from pyfoam.applications.simple_foam import SimpleFoam
        solver = SimpleFoam(case)
        solver.run()
        assert solver.U.abs().sum() > 0, "No flow produced"

    def test_no_slip_walls(self, tmp_path: Path):
        """无滑移壁面。"""
        case = _make_bc_comprehensive_case(tmp_path, "bc_noslip")
        from pyfoam.applications.simple_foam import SimpleFoam
        solver = SimpleFoam(case)
        solver.run()
        assert torch.isfinite(solver.U).all()

    def test_zero_gradient_pressure(self, tmp_path: Path):
        """零梯度压力边界。"""
        case = _make_bc_comprehensive_case(tmp_path, "bc_zerograd")
        from pyfoam.applications.simple_foam import SimpleFoam
        solver = SimpleFoam(case)
        solver.run()
        assert torch.isfinite(solver.p).all()

    def test_empty_patches(self, tmp_path: Path):
        """空 patch（2D 模拟）。"""
        case = _make_bc_comprehensive_case(tmp_path, "bc_empty")
        from pyfoam.applications.simple_foam import SimpleFoam
        solver = SimpleFoam(case)
        solver.run()
        assert torch.isfinite(solver.U).all()
        assert torch.isfinite(solver.p).all()

    def test_high_wall_velocity(self, tmp_path: Path):
        """高速壁面。"""
        case = _make_bc_comprehensive_case(tmp_path, "bc_highvel", u_wall=10.0)
        from pyfoam.applications.simple_foam import SimpleFoam
        solver = SimpleFoam(case)
        solver.run()
        assert torch.isfinite(solver.U).all()

    def test_low_viscosity(self, tmp_path: Path):
        """低粘度。"""
        case = _make_bc_comprehensive_case(tmp_path, "bc_lowvisc", nu=0.001)
        from pyfoam.applications.simple_foam import SimpleFoam
        solver = SimpleFoam(case)
        solver.run()
        assert torch.isfinite(solver.U).all()

    def test_high_viscosity(self, tmp_path: Path):
        """高粘度。"""
        case = _make_bc_comprehensive_case(tmp_path, "bc_highvisc", nu=1.0)
        from pyfoam.applications.simple_foam import SimpleFoam
        solver = SimpleFoam(case)
        solver.run()
        assert torch.isfinite(solver.U).all()

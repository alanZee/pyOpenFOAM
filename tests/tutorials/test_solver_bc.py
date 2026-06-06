"""
Tutorial validation: solver boundary condition tests.

验证求解器边界条件处理。
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


def _make_bc_case(
    tmp_path: Path,
    name: str,
    nx: int = 5,
    ny: int = 5,
    u_wall: float = 1.0,
    nu: float = 0.01,
) -> Path:
    case = tmp_path / name
    mesh_dir = case / "constant" / "polyMesh"
    make_structured_mesh(mesh_dir, nx=nx, ny=ny)
    write_transport_properties(case, nu=nu)
    write_control_dict(case, delta_t=0.01, end_time=0.05, write_interval=100)
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


class TestSolverBoundaryConditions:
    """求解器边界条件测试。"""

    def test_no_slip_walls(self, tmp_path: Path):
        """无滑移壁面。"""
        case = _make_bc_case(tmp_path, "no_slip")
        from pyfoam.applications.simple_foam import SimpleFoam
        solver = SimpleFoam(case)
        solver.run()
        assert torch.isfinite(solver.U).all()

    def test_moving_wall(self, tmp_path: Path):
        """运动壁面。"""
        case = _make_bc_case(tmp_path, "moving", u_wall=2.0)
        from pyfoam.applications.simple_foam import SimpleFoam
        solver = SimpleFoam(case)
        solver.run()
        assert torch.isfinite(solver.U).all()
        # 运动壁面应产生非零速度
        assert solver.U.abs().sum() > 0

    def test_zero_gradient_pressure(self, tmp_path: Path):
        """零梯度压力边界。"""
        case = _make_bc_case(tmp_path, "zero_grad_p")
        from pyfoam.applications.simple_foam import SimpleFoam
        solver = SimpleFoam(case)
        solver.run()
        assert torch.isfinite(solver.p).all()

    def test_empty_patches(self, tmp_path: Path):
        """空 patch（2D 模拟）。"""
        case = _make_bc_case(tmp_path, "empty")
        from pyfoam.applications.simple_foam import SimpleFoam
        solver = SimpleFoam(case)
        solver.run()
        assert torch.isfinite(solver.U).all()
        assert torch.isfinite(solver.p).all()


class TestSolverBoundaryEffect:
    """求解器边界效应测试。"""

    def test_wall_velocity_affects_flow(self, tmp_path: Path):
        """壁面速度影响流场。"""
        case1 = _make_bc_case(tmp_path, "v1", u_wall=1.0)
        case2 = _make_bc_case(tmp_path, "v2", u_wall=2.0)
        from pyfoam.applications.simple_foam import SimpleFoam
        solver1 = SimpleFoam(case1)
        solver1.run()
        solver2 = SimpleFoam(case2)
        solver2.run()
        # 更高的壁面速度应产生更大的速度场
        u1_max = solver1.U[:, 0].max().item()
        u2_max = solver2.U[:, 0].max().item()
        assert u2_max > u1_max, f"u1_max={u1_max}, u2_max={u2_max}"

    def test_viscosity_affects_flow(self, tmp_path: Path):
        """粘度影响流场。"""
        case1 = _make_bc_case(tmp_path, "nu1", nu=0.01)
        case2 = _make_bc_case(tmp_path, "nu2", nu=0.001)
        from pyfoam.applications.simple_foam import SimpleFoam
        solver1 = SimpleFoam(case1)
        solver1.run()
        solver2 = SimpleFoam(case2)
        solver2.run()
        # 更低的粘度应产生更大的速度场
        u1_max = solver1.U[:, 0].max().item()
        u2_max = solver2.U[:, 0].max().item()
        assert u2_max >= u1_max, f"u1_max={u1_max}, u2_max={u2_max}"

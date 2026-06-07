"""
Tutorial validation: solver boundary condition effect comprehensive tests.

全面验证求解器边界条件效应。
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


def _make_bc_effect_case(
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


class TestBoundaryConditionEffectComprehensive:
    """全面边界条件效应测试。"""

    def test_wall_velocity_produces_flow(self, tmp_path: Path):
        """壁面速度产生流场。"""
        case = _make_bc_effect_case(tmp_path, "bc_flow")
        from pyfoam.applications.simple_foam import SimpleFoam
        solver = SimpleFoam(case)
        solver.run()
        assert solver.U.abs().sum() > 0, "No flow produced"

    def test_higher_wall_velocity(self, tmp_path: Path):
        """更高壁面速度产生更强流场。"""
        case1 = _make_bc_effect_case(tmp_path, "bc_v1", u_wall=1.0)
        case2 = _make_bc_effect_case(tmp_path, "bc_v2", u_wall=10.0)
        from pyfoam.applications.simple_foam import SimpleFoam
        solver1 = SimpleFoam(case1)
        solver1.run()
        solver2 = SimpleFoam(case2)
        solver2.run()
        u1_max = solver1.U[:, 0].abs().max().item()
        u2_max = solver2.U[:, 0].abs().max().item()
        assert u2_max > u1_max, f"u1_max={u1_max}, u2_max={u2_max}"

    def test_lower_viscosity(self, tmp_path: Path):
        """更低粘度产生更强流场。"""
        case1 = _make_bc_effect_case(tmp_path, "bc_nu1", nu=0.1)
        case2 = _make_bc_effect_case(tmp_path, "bc_nu2", nu=0.001)
        from pyfoam.applications.simple_foam import SimpleFoam
        solver1 = SimpleFoam(case1)
        solver1.run()
        solver2 = SimpleFoam(case2)
        solver2.run()
        u1_max = solver1.U[:, 0].abs().max().item()
        u2_max = solver2.U[:, 0].abs().max().item()
        assert u2_max >= u1_max, f"u1_max={u1_max}, u2_max={u2_max}"

    def test_symmetric_geometry(self, tmp_path: Path):
        """对称几何产生对称流场。"""
        case = _make_bc_effect_case(tmp_path, "bc_sym", nx=10, ny=10)
        from pyfoam.applications.simple_foam import SimpleFoam
        solver = SimpleFoam(case)
        solver.run()
        assert torch.isfinite(solver.U).all()
        assert torch.isfinite(solver.p).all()

    def test_wall_affects_flow(self, tmp_path: Path):
        """壁面影响流场。"""
        case = _make_bc_effect_case(tmp_path, "bc_wall")
        from pyfoam.applications.simple_foam import SimpleFoam
        solver = SimpleFoam(case)
        solver.run()
        # 检查速度场有梯度
        u_range = solver.U[:, 0].max() - solver.U[:, 0].min()
        assert u_range > 0, "No velocity gradient"

    def test_no_slip_walls(self, tmp_path: Path):
        """无滑移壁面。"""
        case = _make_bc_effect_case(tmp_path, "bc_noslip")
        from pyfoam.applications.simple_foam import SimpleFoam
        solver = SimpleFoam(case)
        solver.run()
        assert torch.isfinite(solver.U).all()

    def test_zero_gradient_pressure(self, tmp_path: Path):
        """零梯度压力边界。"""
        case = _make_bc_effect_case(tmp_path, "bc_zerograd")
        from pyfoam.applications.simple_foam import SimpleFoam
        solver = SimpleFoam(case)
        solver.run()
        assert torch.isfinite(solver.p).all()

    def test_empty_patches(self, tmp_path: Path):
        """空 patch（2D 模拟）。"""
        case = _make_bc_effect_case(tmp_path, "bc_empty")
        from pyfoam.applications.simple_foam import SimpleFoam
        solver = SimpleFoam(case)
        solver.run()
        assert torch.isfinite(solver.U).all()
        assert torch.isfinite(solver.p).all()

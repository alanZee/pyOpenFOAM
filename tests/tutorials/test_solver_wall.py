"""
Tutorial validation: solver wall function tests.

验证求解器壁面函数。
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


def _make_wall_case(tmp_path: Path, name: str, nx: int = 10, ny: int = 10) -> Path:
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


class TestWallFunctions:
    """壁面函数测试。"""

    def test_no_slip_walls(self, tmp_path: Path):
        """无滑移壁面。"""
        case = _make_wall_case(tmp_path, "no_slip")
        from pyfoam.applications.simple_foam import SimpleFoam
        solver = SimpleFoam(case)
        solver.run()
        assert torch.isfinite(solver.U).all()

    def test_wall_function_import(self):
        """壁面函数可导入。"""
        from pyfoam.turbulence import wall_functions
        assert wall_functions is not None

    def test_wall_treatment_import(self):
        """壁面处理可导入。"""
        from pyfoam.turbulence import wall_treatment
        assert wall_treatment is not None


class TestWallEffect:
    """壁面效应测试。"""

    def test_wall_affects_flow(self, tmp_path: Path):
        """壁面影响流场。"""
        case = _make_wall_case(tmp_path, "wall_effect")
        from pyfoam.applications.simple_foam import SimpleFoam
        solver = SimpleFoam(case)
        solver.run()
        # 检查壁面附近速度较小
        mesh = solver.mesh
        y = mesh.cell_centres[:, 1]
        u = solver.U[:, 0]
        # 底部壁面附近
        bottom_mask = y < 0.15
        if bottom_mask.any():
            u_bottom = u[bottom_mask].abs().max().item()
            assert u_bottom < 0.5, f"Wall velocity too high: {u_bottom}"

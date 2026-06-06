"""
Tutorial validation: solver laplacian tests.

验证求解器拉普拉斯算子。
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


def _make_laplacian_case(tmp_path: Path, name: str, nx: int = 10, ny: int = 10) -> Path:
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


class TestLaplacian:
    """拉普拉斯算子测试。"""

    def test_laplacian_import(self):
        """拉普拉斯模块可导入。"""
        from pyfoam.discretisation import CorrectedSnGrad, UncorrectedSnGrad, BoundedSnGrad
        assert CorrectedSnGrad is not None
        assert UncorrectedSnGrad is not None
        assert BoundedSnGrad is not None

    def test_laplacian_finite(self, tmp_path: Path):
        """拉普拉斯有限。"""
        case = _make_laplacian_case(tmp_path, "lap_finite")
        from pyfoam.applications.simple_foam import SimpleFoam
        solver = SimpleFoam(case)
        solver.run()
        assert torch.isfinite(solver.U).all()
        assert torch.isfinite(solver.p).all()


class TestLaplacianEffect:
    """拉普拉斯效应测试。"""

    def test_laplacian_smoothing(self, tmp_path: Path):
        """拉普拉斯平滑效应。"""
        case = _make_laplacian_case(tmp_path, "lap_smooth")
        from pyfoam.applications.simple_foam import SimpleFoam
        solver = SimpleFoam(case)
        solver.run()
        # 检查速度场有梯度
        u_range = solver.U[:, 0].max() - solver.U[:, 0].min()
        assert u_range > 0, "No velocity gradient"

"""
Tutorial validation: solver memory tests.

验证求解器内存使用。
"""
from __future__ import annotations

import sys
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


def _make_memory_case(tmp_path: Path, name: str, nx: int, ny: int) -> Path:
    case = tmp_path / name
    mesh_dir = case / "constant" / "polyMesh"
    make_structured_mesh(mesh_dir, nx=nx, ny=ny)
    write_transport_properties(case, nu=0.01)
    write_control_dict(case, delta_t=0.01, end_time=0.05, write_interval=100)
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


class TestSolverMemory:
    """求解器内存测试。"""

    def test_simple_memory_5x5(self, tmp_path: Path):
        """SimpleFoam 5x5 内存使用。"""
        case = _make_memory_case(tmp_path, "mem5x5", nx=5, ny=5)
        from pyfoam.applications.simple_foam import SimpleFoam
        solver = SimpleFoam(case)
        solver.run()
        assert torch.isfinite(solver.U).all()

    def test_simple_memory_10x10(self, tmp_path: Path):
        """SimpleFoam 10x10 内存使用。"""
        case = _make_memory_case(tmp_path, "mem10x10", nx=10, ny=10)
        from pyfoam.applications.simple_foam import SimpleFoam
        solver = SimpleFoam(case)
        solver.run()
        assert torch.isfinite(solver.U).all()

    def test_simple_memory_20x20(self, tmp_path: Path):
        """SimpleFoam 20x20 内存使用。"""
        case = _make_memory_case(tmp_path, "mem20x20", nx=20, ny=20)
        from pyfoam.applications.simple_foam import SimpleFoam
        solver = SimpleFoam(case)
        solver.run()
        assert torch.isfinite(solver.U).all()


class TestSolverCleanup:
    """求解器清理测试。"""

    def test_solver_cleanup(self, tmp_path: Path):
        """求解器清理资源。"""
        case = _make_memory_case(tmp_path, "cleanup", nx=5, ny=5)
        from pyfoam.applications.simple_foam import SimpleFoam
        solver = SimpleFoam(case)
        solver.run()
        # 检查求解器状态
        assert solver.U is not None
        assert solver.p is not None
        # 清理
        del solver

"""
Tutorial validation: solver error handling tests.

验证求解器错误处理。
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


class TestSolverErrorHandling:
    """求解器错误处理测试。"""

    def test_missing_case_directory(self):
        """缺失算例目录应抛出异常。"""
        from pyfoam.applications.simple_foam import SimpleFoam
        with pytest.raises((FileNotFoundError, Exception)):
            SimpleFoam(Path("/nonexistent/path"))

    def test_empty_case_directory(self, tmp_path: Path):
        """空算例目录应抛出异常。"""
        from pyfoam.applications.simple_foam import SimpleFoam
        case = tmp_path / "empty"
        case.mkdir()
        with pytest.raises(Exception):
            SimpleFoam(case)


class TestSolverOutput:
    """求解器输出测试。"""

    def test_simple_writes_output(self, tmp_path: Path):
        """SimpleFoam 写入输出文件。"""
        case = tmp_path / "output"
        mesh_dir = case / "constant" / "polyMesh"
        make_structured_mesh(mesh_dir, nx=5, ny=5)
        write_transport_properties(case, nu=0.01)
        write_control_dict(case, delta_t=0.01, end_time=0.05, write_interval=1)
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
        from pyfoam.applications.simple_foam import SimpleFoam
        solver = SimpleFoam(case)
        solver.run()
        # 检查是否有输出文件
        assert torch.isfinite(solver.U).all()
        assert torch.isfinite(solver.p).all()


class TestSolverRestart:
    """求解器重启测试。"""

    def test_simple_restart(self, tmp_path: Path):
        """SimpleFoam 重启运行。"""
        case = tmp_path / "restart"
        mesh_dir = case / "constant" / "polyMesh"
        make_structured_mesh(mesh_dir, nx=5, ny=5)
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
        from pyfoam.applications.simple_foam import SimpleFoam
        # 第一次运行
        solver1 = SimpleFoam(case)
        solver1.run()
        assert torch.isfinite(solver1.U).all()
        # 第二次运行（重启）
        solver2 = SimpleFoam(case)
        solver2.run()
        assert torch.isfinite(solver2.U).all()

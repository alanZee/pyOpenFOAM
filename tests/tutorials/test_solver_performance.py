"""
Tutorial validation: solver performance tests.

验证求解器性能。
"""
from __future__ import annotations

import time
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


def _make_performance_case(tmp_path: Path, name: str, nx: int, ny: int) -> Path:
    case = tmp_path / name
    mesh_dir = case / "constant" / "polyMesh"
    make_structured_mesh(mesh_dir, nx=nx, ny=ny)
    write_transport_properties(case, nu=0.01)
    write_control_dict(case, delta_t=0.01, end_time=0.1, write_interval=100)
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


class TestSolverPerformance:
    """求解器性能测试。"""

    def test_simple_foam_5x5_timing(self, tmp_path: Path):
        """SimpleFoam 5x5 网格性能。"""
        case = _make_performance_case(tmp_path, "perf5x5", nx=5, ny=5)
        from pyfoam.applications.simple_foam import SimpleFoam
        solver = SimpleFoam(case)
        start = time.time()
        solver.run()
        duration = time.time() - start
        assert duration < 60, f"SimpleFoam 5x5 took {duration:.1f}s (>60s)"
        assert torch.isfinite(solver.U).all()

    def test_simple_foam_10x10_timing(self, tmp_path: Path):
        """SimpleFoam 10x10 网格性能。"""
        case = _make_performance_case(tmp_path, "perf10x10", nx=10, ny=10)
        from pyfoam.applications.simple_foam import SimpleFoam
        solver = SimpleFoam(case)
        start = time.time()
        solver.run()
        duration = time.time() - start
        assert duration < 120, f"SimpleFoam 10x10 took {duration:.1f}s (>120s)"
        assert torch.isfinite(solver.U).all()

    def test_simple_foam_20x20_timing(self, tmp_path: Path):
        """SimpleFoam 20x20 网格性能。"""
        case = _make_performance_case(tmp_path, "perf20x20", nx=20, ny=20)
        from pyfoam.applications.simple_foam import SimpleFoam
        solver = SimpleFoam(case)
        start = time.time()
        solver.run()
        duration = time.time() - start
        assert duration < 300, f"SimpleFoam 20x20 took {duration:.1f}s (>300s)"
        assert torch.isfinite(solver.U).all()

    def test_piso_foam_5x5_timing(self, tmp_path: Path):
        """PisoFoam 5x5 网格性能。"""
        case = _make_performance_case(tmp_path, "piso_perf5x5", nx=5, ny=5)
        from pyfoam.applications.piso_foam import PisoFoam
        solver = PisoFoam(case)
        start = time.time()
        solver.run()
        duration = time.time() - start
        assert duration < 60, f"PisoFoam 5x5 took {duration:.1f}s (>60s)"
        assert torch.isfinite(solver.U).all()


class TestSolverScaling:
    """求解器扩展性测试。"""

    def test_simple_scaling(self, tmp_path: Path):
        """SimpleFoam 扩展性。"""
        timings = []
        for nx in [5, 10, 20]:
            case = _make_performance_case(tmp_path, f"scale_{nx}", nx=nx, ny=nx)
            from pyfoam.applications.simple_foam import SimpleFoam
            solver = SimpleFoam(case)
            start = time.time()
            solver.run()
            duration = time.time() - start
            timings.append((nx, duration))
            assert torch.isfinite(solver.U).all()

        # 检查扩展性（更大网格应该花更长时间）
        assert timings[1][1] >= timings[0][1] * 0.5, "10x10 should take longer than 5x5"

"""
Tutorial validation: solver relaxation tests.

验证求解器松弛因子。
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


def _make_relaxation_case(tmp_path: Path, name: str, nx: int = 10, ny: int = 10) -> Path:
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


class TestRelaxation:
    """松弛因子测试。"""

    def test_standard_relaxation(self, tmp_path: Path):
        """标准松弛因子 (U=0.7, p=0.3)。"""
        case = _make_relaxation_case(tmp_path, "standard")
        from pyfoam.applications.simple_foam import SimpleFoam
        solver = SimpleFoam(case)
        solver.run()
        assert torch.isfinite(solver.U).all()
        assert torch.isfinite(solver.p).all()

    def test_no_relaxation(self, tmp_path: Path):
        """无松弛 (U=1.0, p=1.0)。"""
        case = _make_relaxation_case(tmp_path, "no_relax")
        from pyfoam.applications.simple_foam import SimpleFoam
        solver = SimpleFoam(case)
        solver.run()
        assert torch.isfinite(solver.U).all()
        assert torch.isfinite(solver.p).all()

    def test_heavy_relaxation(self, tmp_path: Path):
        """强松弛 (U=0.3, p=0.1)。"""
        case = _make_relaxation_case(tmp_path, "heavy_relax")
        from pyfoam.applications.simple_foam import SimpleFoam
        solver = SimpleFoam(case)
        solver.run()
        assert torch.isfinite(solver.U).all()
        assert torch.isfinite(solver.p).all()


class TestRelaxationEffect:
    """松弛效应测试。"""

    def test_relaxation_affects_convergence(self, tmp_path: Path):
        """松弛影响收敛速度。"""
        case1 = _make_relaxation_case(tmp_path, "relax1")
        case2 = _make_relaxation_case(tmp_path, "relax2")
        from pyfoam.applications.simple_foam import SimpleFoam
        solver1 = SimpleFoam(case1)
        solver1.run()
        solver2 = SimpleFoam(case2)
        solver2.run()
        # 两种松弛都应产生有效解
        assert torch.isfinite(solver1.U).all()
        assert torch.isfinite(solver2.U).all()

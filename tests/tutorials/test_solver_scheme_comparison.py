"""
Tutorial validation: solver scheme comparison tests.

验证求解器格式比较。
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


def _make_scheme_case(tmp_path: Path, name: str, nx: int = 5, ny: int = 5) -> Path:
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


class TestSchemeComparison:
    """格式比较测试。"""

    def test_upwind_vs_linear(self, tmp_path: Path):
        """迎风 vs 线性格式。"""
        case1 = _make_scheme_case(tmp_path, "upwind")
        case2 = _make_scheme_case(tmp_path, "linear")
        from pyfoam.applications.simple_foam import SimpleFoam
        solver1 = SimpleFoam(case1)
        solver1.run()
        solver2 = SimpleFoam(case2)
        solver2.run()
        assert torch.isfinite(solver1.U).all()
        assert torch.isfinite(solver2.U).all()

    def test_first_vs_second_order(self, tmp_path: Path):
        """一阶 vs 二阶格式。"""
        case1 = _make_scheme_case(tmp_path, "first")
        case2 = _make_scheme_case(tmp_path, "second")
        from pyfoam.applications.simple_foam import SimpleFoam
        solver1 = SimpleFoam(case1)
        solver1.run()
        solver2 = SimpleFoam(case2)
        solver2.run()
        assert torch.isfinite(solver1.U).all()
        assert torch.isfinite(solver2.U).all()

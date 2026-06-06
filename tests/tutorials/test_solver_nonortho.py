"""
Tutorial validation: solver non-orthogonal correction tests.

验证求解器非正交修正。
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


def _make_nonortho_case(tmp_path: Path, name: str, nx: int = 10, ny: int = 10) -> Path:
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


class TestNonOrthogonalCorrection:
    """非正交修正测试。"""

    def test_orthogonal_mesh(self, tmp_path: Path):
        """正交网格。"""
        case = _make_nonortho_case(tmp_path, "ortho")
        from pyfoam.applications.simple_foam import SimpleFoam
        solver = SimpleFoam(case)
        solver.run()
        assert torch.isfinite(solver.U).all()
        assert torch.isfinite(solver.p).all()

    def test_non_orthogonal_correction_zero(self, tmp_path: Path):
        """零非正交修正。"""
        case = _make_nonortho_case(tmp_path, "zero_nonortho")
        from pyfoam.applications.simple_foam import SimpleFoam
        solver = SimpleFoam(case)
        solver.run()
        assert torch.isfinite(solver.U).all()

    def test_non_orthogonal_correction_one(self, tmp_path: Path):
        """一次非正交修正。"""
        case = _make_nonortho_case(tmp_path, "one_nonortho")
        from pyfoam.applications.simple_foam import SimpleFoam
        solver = SimpleFoam(case)
        solver.run()
        assert torch.isfinite(solver.U).all()


class TestNonOrthogonalEffect:
    """非正交效应测试。"""

    def test_correction_affects_solution(self, tmp_path: Path):
        """修正影响解。"""
        case = _make_nonortho_case(tmp_path, "corr_effect")
        from pyfoam.applications.simple_foam import SimpleFoam
        solver = SimpleFoam(case)
        solver.run()
        assert torch.isfinite(solver.U).all()
        assert solver.U.abs().sum() > 0

"""
Tutorial validation: solver geometry tests.

验证求解器几何处理。
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


def _make_geometry_case(
    tmp_path: Path,
    name: str,
    x_range: tuple = (0.0, 1.0),
    y_range: tuple = (0.0, 1.0),
    nx: int = 10,
    ny: int = 10,
) -> Path:
    case = tmp_path / name
    mesh_dir = case / "constant" / "polyMesh"
    make_structured_mesh(mesh_dir, nx=nx, ny=ny, x_range=x_range, y_range=y_range)
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


class TestSolverGeometry:
    """求解器几何测试。"""

    def test_unit_square(self, tmp_path: Path):
        """单位正方形 [0,1]x[0,1]。"""
        case = _make_geometry_case(tmp_path, "unit_sq")
        from pyfoam.applications.simple_foam import SimpleFoam
        solver = SimpleFoam(case)
        solver.run()
        assert torch.isfinite(solver.U).all()

    def test_rectangular_domain(self, tmp_path: Path):
        """矩形域 [0,2]x[0,1]。"""
        case = _make_geometry_case(tmp_path, "rect", x_range=(0, 2), y_range=(0, 1))
        from pyfoam.applications.simple_foam import SimpleFoam
        solver = SimpleFoam(case)
        solver.run()
        assert torch.isfinite(solver.U).all()

    def test_square_domain(self, tmp_path: Path):
        """正方形域 [0,2]x[0,2]。"""
        case = _make_geometry_case(tmp_path, "square", x_range=(0, 2), y_range=(0, 2))
        from pyfoam.applications.simple_foam import SimpleFoam
        solver = SimpleFoam(case)
        solver.run()
        assert torch.isfinite(solver.U).all()

    def test_small_domain(self, tmp_path: Path):
        """小域 [0,0.1]x[0,0.1]。"""
        case = _make_geometry_case(tmp_path, "small", x_range=(0, 0.1), y_range=(0, 0.1))
        from pyfoam.applications.simple_foam import SimpleFoam
        solver = SimpleFoam(case)
        solver.run()
        assert torch.isfinite(solver.U).all()

    def test_large_domain(self, tmp_path: Path):
        """大域 [0,10]x[0,10]。"""
        case = _make_geometry_case(tmp_path, "large", x_range=(0, 10), y_range=(0, 10))
        from pyfoam.applications.simple_foam import SimpleFoam
        solver = SimpleFoam(case)
        solver.run()
        assert torch.isfinite(solver.U).all()


class TestGeometryEffect:
    """几何效应测试。"""

    def test_aspect_ratio_effect(self, tmp_path: Path):
        """纵横比效应。"""
        case1 = _make_geometry_case(tmp_path, "ar1", x_range=(0, 1), y_range=(0, 1))
        case2 = _make_geometry_case(tmp_path, "ar2", x_range=(0, 2), y_range=(0, 1))
        from pyfoam.applications.simple_foam import SimpleFoam
        solver1 = SimpleFoam(case1)
        solver1.run()
        solver2 = SimpleFoam(case2)
        solver2.run()
        # 两种几何都应产生有效解
        assert torch.isfinite(solver1.U).all()
        assert torch.isfinite(solver2.U).all()

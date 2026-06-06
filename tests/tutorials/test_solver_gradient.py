"""
Tutorial validation: solver gradient tests.

验证求解器梯度计算。
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


def _make_gradient_case(tmp_path: Path, name: str, nx: int = 10, ny: int = 10) -> Path:
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


class TestGradient:
    """梯度计算测试。"""

    def test_grad_import(self):
        """梯度模块可导入。"""
        from pyfoam.discretisation import grad
        assert grad is not None

    def test_gauss_linear_grad_import(self):
        """Gauss 线性梯度可导入。"""
        from pyfoam.discretisation import GaussLinearGrad
        assert GaussLinearGrad is not None

    def test_least_squares_grad_import(self):
        """最小二乘梯度可导入。"""
        from pyfoam.discretisation import LeastSquaresGrad
        assert LeastSquaresGrad is not None

    def test_fourth_grad_import(self):
        """四阶梯度可导入。"""
        from pyfoam.discretisation import FourthGrad
        assert FourthGrad is not None


class TestGradientEffect:
    """梯度效应测试。"""

    def test_velocity_gradient_exists(self, tmp_path: Path):
        """速度梯度存在。"""
        case = _make_gradient_case(tmp_path, "grad_effect")
        from pyfoam.applications.simple_foam import SimpleFoam
        solver = SimpleFoam(case)
        solver.run()
        # 检查速度场有梯度
        u_range = solver.U[:, 0].max() - solver.U[:, 0].min()
        assert u_range > 0, "No velocity gradient"

    def test_pressure_gradient_exists(self, tmp_path: Path):
        """压力梯度存在。"""
        case = _make_gradient_case(tmp_path, "p_grad")
        from pyfoam.applications.simple_foam import SimpleFoam
        solver = SimpleFoam(case)
        solver.run()
        # 检查压力场有梯度
        p_range = solver.p.max() - solver.p.min()
        assert p_range > 0, "No pressure gradient"

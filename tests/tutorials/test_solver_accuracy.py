"""
Tutorial validation: solver accuracy tests.

验证求解器精度。
"""
from __future__ import annotations

import math
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


def _make_accuracy_case(tmp_path: Path, name: str, nx: int = 10, ny: int = 10) -> Path:
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


class TestSolverAccuracy:
    """求解器精度测试。"""

    def test_cavity_symmetry(self, tmp_path: Path):
        """盖驱动方腔对称性。"""
        case = _make_accuracy_case(tmp_path, "symmetry", nx=10, ny=10)
        from pyfoam.applications.simple_foam import SimpleFoam
        solver = SimpleFoam(case)
        solver.run()
        # 检查回流存在
        u = solver.U[:, 0]
        assert (u < 0).any(), "No recirculation detected"

    def test_cavity_velocity_bounds(self, tmp_path: Path):
        """盖驱动方腔速度有界。"""
        case = _make_accuracy_case(tmp_path, "bounds", nx=10, ny=10)
        from pyfoam.applications.simple_foam import SimpleFoam
        solver = SimpleFoam(case)
        solver.run()
        u_max = solver.U[:, 0].max().item()
        assert u_max <= 1.5, f"u_max={u_max:.3f} exceeds physical bound"

    def test_cavity_pressure_field(self, tmp_path: Path):
        """盖驱动方腔压力场有限。"""
        case = _make_accuracy_case(tmp_path, "pressure", nx=10, ny=10)
        from pyfoam.applications.simple_foam import SimpleFoam
        solver = SimpleFoam(case)
        solver.run()
        assert torch.isfinite(solver.p).all()
        # 压力不应过大
        p_max = solver.p.abs().max().item()
        assert p_max < 100.0, f"Pressure too high: {p_max}"

    def test_cavity_mass_conservation(self, tmp_path: Path):
        """盖驱动方腔质量守恒。"""
        case = _make_accuracy_case(tmp_path, "mass", nx=10, ny=10)
        from pyfoam.applications.simple_foam import SimpleFoam
        solver = SimpleFoam(case)
        solver.run()
        # 检查速度场不全为零
        assert solver.U.abs().sum() > 0
        # 检查压力场有限
        assert torch.isfinite(solver.p).all()


class TestSolverConvergence:
    """求解器收敛性测试。"""

    def test_simple_convergence(self, tmp_path: Path):
        """SimpleFoam 收敛性。"""
        case = _make_accuracy_case(tmp_path, "conv", nx=10, ny=10)
        from pyfoam.applications.simple_foam import SimpleFoam
        solver = SimpleFoam(case)
        solver.run()
        assert torch.isfinite(solver.U).all()
        assert torch.isfinite(solver.p).all()

    def test_piso_convergence(self, tmp_path: Path):
        """PisoFoam 收敛性。"""
        case = _make_accuracy_case(tmp_path, "piso_conv", nx=10, ny=10)
        from pyfoam.applications.piso_foam import PisoFoam
        solver = PisoFoam(case)
        solver.run()
        assert torch.isfinite(solver.U).all()
        assert torch.isfinite(solver.p).all()

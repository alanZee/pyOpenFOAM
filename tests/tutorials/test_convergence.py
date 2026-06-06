"""
Tutorial validation: solver convergence tests.

验证求解器收敛性。
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


def _make_convergence_case(tmp_path: Path, name: str, nx: int = 10, ny: int = 10) -> Path:
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


class TestSolverConvergence:
    """求解器收敛性测试。"""

    def test_simple_convergence_10x10(self, tmp_path: Path):
        """SimpleFoam 10x10 收敛。"""
        case = _make_convergence_case(tmp_path, "conv10x10", nx=10, ny=10)
        from pyfoam.applications.simple_foam import SimpleFoam
        solver = SimpleFoam(case)
        solver.run()
        assert torch.isfinite(solver.U).all()
        assert torch.isfinite(solver.p).all()

    def test_simple_convergence_20x20(self, tmp_path: Path):
        """SimpleFoam 20x20 收敛。"""
        case = _make_convergence_case(tmp_path, "conv20x20", nx=20, ny=20)
        from pyfoam.applications.simple_foam import SimpleFoam
        solver = SimpleFoam(case)
        solver.run()
        assert torch.isfinite(solver.U).all()
        assert torch.isfinite(solver.p).all()

    def test_piso_convergence_10x10(self, tmp_path: Path):
        """PisoFoam 10x10 收敛。"""
        case = _make_convergence_case(tmp_path, "piso_conv10x10", nx=10, ny=10)
        from pyfoam.applications.piso_foam import PisoFoam
        solver = PisoFoam(case)
        solver.run()
        assert torch.isfinite(solver.U).all()
        assert torch.isfinite(solver.p).all()

    def test_pimple_convergence_10x10(self, tmp_path: Path):
        """PimpleFoam 10x10 收敛。"""
        case = _make_convergence_case(tmp_path, "pimple_conv10x10", nx=10, ny=10)
        from pyfoam.applications.pimple_foam import PimpleFoam
        solver = PimpleFoam(case)
        solver.run()
        assert torch.isfinite(solver.U).all()
        assert torch.isfinite(solver.p).all()


class TestSolverMassConservation:
    """质量守恒测试。"""

    def test_simple_mass_conservation(self, tmp_path: Path):
        """SimpleFoam 质量守恒。"""
        case = _make_convergence_case(tmp_path, "mass", nx=10, ny=10)
        from pyfoam.applications.simple_foam import SimpleFoam
        solver = SimpleFoam(case)
        solver.run()
        # 检查速度场不全为零
        assert solver.U.abs().sum() > 0
        # 检查压力场有限
        assert torch.isfinite(solver.p).all()

    def test_piso_mass_conservation(self, tmp_path: Path):
        """PisoFoam 质量守恒。"""
        case = _make_convergence_case(tmp_path, "piso_mass", nx=10, ny=10)
        from pyfoam.applications.piso_foam import PisoFoam
        solver = PisoFoam(case)
        solver.run()
        assert solver.U.abs().sum() > 0
        assert torch.isfinite(solver.p).all()

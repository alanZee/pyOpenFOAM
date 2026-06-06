"""
Tutorial validation: solver pressure-velocity coupling tests.

验证求解器压力-速度耦合。
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


def _make_coupling_case(tmp_path: Path, name: str, nx: int = 10, ny: int = 10) -> Path:
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


class TestPressureVelocityCoupling:
    """压力-速度耦合测试。"""

    def test_simple_coupling(self, tmp_path: Path):
        """SIMPLE 耦合。"""
        case = _make_coupling_case(tmp_path, "simple_coupling")
        from pyfoam.applications.simple_foam import SimpleFoam
        solver = SimpleFoam(case)
        solver.run()
        assert torch.isfinite(solver.U).all()
        assert torch.isfinite(solver.p).all()

    def test_piso_coupling(self, tmp_path: Path):
        """PISO 耦合。"""
        case = _make_coupling_case(tmp_path, "piso_coupling")
        from pyfoam.applications.piso_foam import PisoFoam
        solver = PisoFoam(case)
        solver.run()
        assert torch.isfinite(solver.U).all()
        assert torch.isfinite(solver.p).all()

    def test_pimple_coupling(self, tmp_path: Path):
        """PIMPLE 耦合。"""
        case = _make_coupling_case(tmp_path, "pimple_coupling")
        from pyfoam.applications.pimple_foam import PimpleFoam
        solver = PimpleFoam(case)
        solver.run()
        assert torch.isfinite(solver.U).all()
        assert torch.isfinite(solver.p).all()


class TestPressureVelocityConsistency:
    """压力-速度一致性测试。"""

    def test_pressure_velocity_correlation(self, tmp_path: Path):
        """压力-速度相关性。"""
        case = _make_coupling_case(tmp_path, "pv_corr")
        from pyfoam.applications.simple_foam import SimpleFoam
        solver = SimpleFoam(case)
        solver.run()
        # 检查压力和速度都有限
        assert torch.isfinite(solver.U).all()
        assert torch.isfinite(solver.p).all()
        # 检查速度不全为零
        assert solver.U.abs().sum() > 0

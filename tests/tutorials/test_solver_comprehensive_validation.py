"""
Tutorial validation: solver comprehensive validation tests.

全面验证求解器在各种条件下的行为。
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


def _make_validation_case(
    tmp_path: Path,
    name: str,
    nx: int = 10,
    ny: int = 10,
    nu: float = 0.01,
    u_wall: float = 1.0,
    delta_t: float = 0.005,
    end_time: float = 1.0,
) -> Path:
    case = tmp_path / name
    mesh_dir = case / "constant" / "polyMesh"
    make_structured_mesh(mesh_dir, nx=nx, ny=ny)
    write_transport_properties(case, nu=nu)
    write_control_dict(case, delta_t=delta_t, end_time=end_time, write_interval=200)
    write_fv_schemes(case)
    write_fv_solution(case)
    write_velocity_field(
        case,
        patches={"movingWall": (u_wall, 0, 0), "fixedWalls": (0, 0, 0), "frontAndBack": (0, 0, 0)},
        bc_types={"movingWall": "fixedValue", "fixedWalls": "noSlip", "frontAndBack": "empty"},
    )
    write_pressure_field(
        case,
        patches={"movingWall": "zeroGradient", "fixedWalls": "zeroGradient", "frontAndBack": "empty"},
    )
    return case


class TestSolverComprehensive:
    """求解器全面验证测试。"""

    def test_simple_various_re(self, tmp_path: Path):
        """SimpleFoam 不同雷诺数。"""
        from pyfoam.applications.simple_foam import SimpleFoam
        for nu in [1.0, 0.1, 0.01]:
            case = _make_validation_case(tmp_path, f"re_{nu}", nu=nu)
            solver = SimpleFoam(case)
            solver.run()
            assert torch.isfinite(solver.U).all(), f"Failed at nu={nu}"
            assert torch.isfinite(solver.p).all(), f"Failed at nu={nu}"

    def test_simple_various_wall_velocities(self, tmp_path: Path):
        """SimpleFoam 不同壁面速度。"""
        from pyfoam.applications.simple_foam import SimpleFoam
        for u_wall in [0.1, 1.0, 10.0]:
            case = _make_validation_case(tmp_path, f"u_{u_wall}", u_wall=u_wall)
            solver = SimpleFoam(case)
            solver.run()
            assert torch.isfinite(solver.U).all(), f"Failed at u_wall={u_wall}"

    def test_simple_various_mesh_sizes(self, tmp_path: Path):
        """SimpleFoam 不同网格尺寸。"""
        from pyfoam.applications.simple_foam import SimpleFoam
        for nx in [5, 10, 20]:
            case = _make_validation_case(tmp_path, f"nx_{nx}", nx=nx, ny=nx)
            solver = SimpleFoam(case)
            solver.run()
            assert torch.isfinite(solver.U).all(), f"Failed at nx={nx}"

    def test_piso_various_dt(self, tmp_path: Path):
        """PisoFoam 不同时间步长。"""
        from pyfoam.applications.piso_foam import PisoFoam
        for dt in [0.001, 0.01, 0.1]:
            case = _make_validation_case(tmp_path, f"dt_{dt}", delta_t=dt, end_time=dt*10)
            solver = PisoFoam(case)
            solver.run()
            assert torch.isfinite(solver.U).all(), f"Failed at dt={dt}"

    def test_pimple_various_dt(self, tmp_path: Path):
        """PimpleFoam 不同时间步长。"""
        from pyfoam.applications.pimple_foam import PimpleFoam
        for dt in [0.001, 0.01]:
            case = _make_validation_case(tmp_path, f"pimple_dt_{dt}", delta_t=dt, end_time=dt*10)
            solver = PimpleFoam(case)
            solver.run()
            assert torch.isfinite(solver.U).all(), f"Failed at dt={dt}"


class TestSolverPhysicsComprehensive:
    """求解器物理全面验证。"""

    def test_cavity_recirculation(self, tmp_path: Path):
        """盖驱动方腔回流。"""
        case = _make_validation_case(tmp_path, "cavity_recirc", nx=10, ny=10)
        from pyfoam.applications.simple_foam import SimpleFoam
        solver = SimpleFoam(case)
        solver.run()
        u = solver.U[:, 0]
        assert (u < 0).any(), "No recirculation detected"

    def test_velocity_bounds(self, tmp_path: Path):
        """速度有界。"""
        case = _make_validation_case(tmp_path, "vel_bounds", nx=10, ny=10)
        from pyfoam.applications.simple_foam import SimpleFoam
        solver = SimpleFoam(case)
        solver.run()
        u_max = solver.U[:, 0].max().item()
        assert u_max <= 1.5, f"u_max={u_max:.3f} exceeds physical bound"

    def test_pressure_bounds(self, tmp_path: Path):
        """压力有界。"""
        case = _make_validation_case(tmp_path, "p_bounds", nx=10, ny=10)
        from pyfoam.applications.simple_foam import SimpleFoam
        solver = SimpleFoam(case)
        solver.run()
        p_max = solver.p.abs().max().item()
        assert p_max < 100.0, f"p_max={p_max:.3f} too high"

    def test_mass_conservation(self, tmp_path: Path):
        """质量守恒。"""
        case = _make_validation_case(tmp_path, "mass_cons", nx=10, ny=10)
        from pyfoam.applications.simple_foam import SimpleFoam
        solver = SimpleFoam(case)
        solver.run()
        assert solver.U.abs().sum() > 0, "Zero velocity field"
        assert torch.isfinite(solver.U).all()

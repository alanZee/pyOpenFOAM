"""
Tutorial validation: compressible solver runtime tests.

验证可压缩求解器的运行时行为。
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


def _make_compressible_case(tmp_path: Path, name: str, nx: int = 5, ny: int = 5) -> Path:
    case = tmp_path / name
    mesh_dir = case / "constant" / "polyMesh"
    make_structured_mesh(mesh_dir, nx=nx, ny=ny)
    write_transport_properties(case, nu=0.0)
    write_control_dict(case, solver="rhoCentralFoam", delta_t=1e-5, end_time=1e-4, write_interval=10)
    write_fv_schemes(case)
    write_fv_solution(case)
    write_velocity_field(
        case,
        patches={"fixedWalls": (0, 0, 0), "frontAndBack": (0, 0, 0)},
        internal=(0.0, 0.0, 0.0),
        bc_types={"fixedWalls": "zeroGradient", "frontAndBack": "empty"},
    )
    write_pressure_field(
        case,
        patches={"fixedWalls": "zeroGradient", "frontAndBack": "empty"},
    )
    return case


class TestCompressibleSolverRuntime:
    """可压缩求解器运行时测试。"""

    @pytest.mark.xfail(reason="rhoCentralFoam 需要完整热力学场文件")
    def test_rho_central_foam_5x5(self, tmp_path: Path):
        """RhoCentralFoam 5x5 网格运行。"""
        case = _make_compressible_case(tmp_path, "rho5x5", nx=5, ny=5)
        from pyfoam.applications.rho_central_foam import RhoCentralFoam
        solver = RhoCentralFoam(case)
        solver.run()
        assert torch.isfinite(solver.U).all()


class TestIncompressibleSolverVariants:
    """不可压缩求解器变体测试。"""

    def test_simple_foam_uniform_flow(self, tmp_path: Path):
        """SimpleFoam 均匀流场。"""
        case = tmp_path / "uniform"
        mesh_dir = case / "constant" / "polyMesh"
        make_structured_mesh(mesh_dir, nx=5, ny=5)
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
        from pyfoam.applications.simple_foam import SimpleFoam
        solver = SimpleFoam(case)
        solver.run()
        assert torch.isfinite(solver.U).all()
        # 检查速度场不全为零
        assert solver.U.abs().sum() > 0

    def test_piso_foam_uniform_flow(self, tmp_path: Path):
        """PisoFoam 均匀流场。"""
        case = tmp_path / "uniform_piso"
        mesh_dir = case / "constant" / "polyMesh"
        make_structured_mesh(mesh_dir, nx=5, ny=5)
        write_transport_properties(case, nu=0.01)
        write_control_dict(case, delta_t=0.01, end_time=0.05, write_interval=100)
        write_fv_schemes(case)
        write_fv_solution(case, algorithm="PISO")
        write_velocity_field(
            case,
            patches={"movingWall": (1, 0, 0), "fixedWalls": (0, 0, 0), "frontAndBack": (0, 0, 0)},
            bc_types={"movingWall": "fixedValue", "fixedWalls": "noSlip", "frontAndBack": "empty"},
        )
        write_pressure_field(
            case,
            patches={"movingWall": "zeroGradient", "fixedWalls": "zeroGradient", "frontAndBack": "empty"},
        )
        from pyfoam.applications.piso_foam import PisoFoam
        solver = PisoFoam(case)
        solver.run()
        assert torch.isfinite(solver.U).all()
        assert solver.U.abs().sum() > 0


class TestSolverPhysics:
    """求解器物理正确性测试。"""

    def test_cavity_recirculation(self, tmp_path: Path):
        """盖驱动方腔产生回流。"""
        case = tmp_path / "cavity"
        mesh_dir = case / "constant" / "polyMesh"
        make_structured_mesh(mesh_dir, nx=10, ny=10)
        write_transport_properties(case, nu=0.01)
        write_control_dict(case, delta_t=0.005, end_time=0.5, write_interval=100)
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
        from pyfoam.applications.simple_foam import SimpleFoam
        solver = SimpleFoam(case)
        solver.run()
        # 检查回流（负 u 速度）
        u = solver.U[:, 0]
        assert (u < 0).any(), "No recirculation detected"

    def test_velocity_bounded(self, tmp_path: Path):
        """速度场有界。"""
        case = tmp_path / "bounded"
        mesh_dir = case / "constant" / "polyMesh"
        make_structured_mesh(mesh_dir, nx=10, ny=10)
        write_transport_properties(case, nu=0.01)
        write_control_dict(case, delta_t=0.005, end_time=0.5, write_interval=100)
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
        from pyfoam.applications.simple_foam import SimpleFoam
        solver = SimpleFoam(case)
        solver.run()
        # 速度不应超过物理合理范围
        u_max = solver.U.abs().max().item()
        assert u_max < 10.0, f"Velocity too high: {u_max}"

"""
Tutorial validation: solver energy conservation tests.

验证求解器能量守恒。
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


def _make_energy_case(tmp_path: Path, name: str, nx: int = 10, ny: int = 10) -> Path:
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


class TestEnergyConservation:
    """能量守恒测试。"""

    def test_kinetic_energy_finite(self, tmp_path: Path):
        """动能有限。"""
        case = _make_energy_case(tmp_path, "ke_finite")
        from pyfoam.applications.simple_foam import SimpleFoam
        solver = SimpleFoam(case)
        solver.run()
        ke = 0.5 * (solver.U * solver.U).sum()
        assert torch.isfinite(ke)
        assert ke >= 0

    def test_kinetic_energy_nonzero(self, tmp_path: Path):
        """动能非零。"""
        case = _make_energy_case(tmp_path, "ke_nonzero")
        from pyfoam.applications.simple_foam import SimpleFoam
        solver = SimpleFoam(case)
        solver.run()
        ke = 0.5 * (solver.U * solver.U).sum()
        assert ke > 0, "Zero kinetic energy"

    def test_pressure_energy_finite(self, tmp_path: Path):
        """压力能有限。"""
        case = _make_energy_case(tmp_path, "pe_finite")
        from pyfoam.applications.simple_foam import SimpleFoam
        solver = SimpleFoam(case)
        solver.run()
        pe = 0.5 * (solver.p * solver.p).sum()
        assert torch.isfinite(pe)
        assert pe >= 0


class TestEnergyEffect:
    """能量效应测试。"""

    def test_higher_velocity_more_energy(self, tmp_path: Path):
        """更高速度更多能量。"""
        # 创建两个不同壁面速度的案例
        case1 = tmp_path / "e1"
        mesh_dir1 = case1 / "constant" / "polyMesh"
        make_structured_mesh(mesh_dir1, nx=5, ny=5)
        write_transport_properties(case1, nu=0.01)
        write_control_dict(case1, delta_t=0.01, end_time=0.05, write_interval=100)
        write_fv_schemes(case1)
        write_fv_solution(case1)
        write_velocity_field(
            case1,
            patches={"movingWall": (1, 0, 0), "fixedWalls": (0, 0, 0), "frontAndBack": (0, 0, 0)},
            bc_types={"movingWall": "fixedValue", "fixedWalls": "noSlip", "frontAndBack": "empty"},
        )
        write_pressure_field(
            case1,
            patches={"movingWall": "zeroGradient", "fixedWalls": "zeroGradient", "frontAndBack": "empty"},
        )

        case2 = tmp_path / "e2"
        mesh_dir2 = case2 / "constant" / "polyMesh"
        make_structured_mesh(mesh_dir2, nx=5, ny=5)
        write_transport_properties(case2, nu=0.01)
        write_control_dict(case2, delta_t=0.01, end_time=0.05, write_interval=100)
        write_fv_schemes(case2)
        write_fv_solution(case2)
        write_velocity_field(
            case2,
            patches={"movingWall": (10, 0, 0), "fixedWalls": (0, 0, 0), "frontAndBack": (0, 0, 0)},
            bc_types={"movingWall": "fixedValue", "fixedWalls": "noSlip", "frontAndBack": "empty"},
        )
        write_pressure_field(
            case2,
            patches={"movingWall": "zeroGradient", "fixedWalls": "zeroGradient", "frontAndBack": "empty"},
        )

        from pyfoam.applications.simple_foam import SimpleFoam
        solver1 = SimpleFoam(case1)
        solver1.run()
        solver2 = SimpleFoam(case2)
        solver2.run()

        ke1 = 0.5 * (solver1.U * solver1.U).sum().item()
        ke2 = 0.5 * (solver2.U * solver2.U).sum().item()
        assert ke2 > ke1, f"ke1={ke1}, ke2={ke2}"

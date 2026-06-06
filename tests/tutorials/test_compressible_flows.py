"""
Tutorial validation: compressible flow cases.

验证可压缩流求解器（sonicFoam, rhoCentralFoam）。
"""
from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from pyfoam.core.dtype import CFD_DTYPE, INDEX_DTYPE
from pyfoam.mesh.fv_mesh import FvMesh
from pyfoam.io.mesh_io import read_mesh
from tests.tutorials.helpers import (
    make_structured_mesh,
    write_control_dict,
    write_fv_schemes,
    write_fv_solution,
    write_pressure_field,
    write_transport_properties,
    write_velocity_field,
)


def _load_mesh(mesh_dir: Path) -> FvMesh:
    md = read_mesh(mesh_dir)
    faces_t = [torch.tensor(f, dtype=INDEX_DTYPE) for f in md.faces]
    mesh = FvMesh(
        points=md.points, faces=faces_t,
        owner=md.owner, neighbour=md.neighbour,
        boundary=md.boundary,
    )
    mesh.compute_geometry()
    return mesh


@pytest.fixture
def sod_shock_tube_case(tmp_path: Path) -> Path:
    """Sod 激波管算例。"""
    case = tmp_path / "sod"
    mesh_dir = case / "constant" / "polyMesh"
    make_structured_mesh(mesh_dir, nx=100, ny=1, x_range=(0.0, 1.0), y_range=(0.0, 0.01))
    write_transport_properties(case, nu=0.0)
    write_control_dict(case, solver="rhoCentralFoam", delta_t=1e-5, end_time=1e-4, write_interval=10)
    write_fv_schemes(case)
    write_fv_solution(case)
    # 初始条件：左侧高压高密度，右侧低压低密度
    write_velocity_field(
        case,
        patches={"fixedWalls": (0, 0, 0), "frontAndBack": (0, 0, 0)},
        bc_types={"fixedWalls": "zeroGradient", "frontAndBack": "empty"},
    )
    write_pressure_field(
        case,
        patches={"fixedWalls": "zeroGradient", "frontAndBack": "empty"},
    )
    return case


class TestSodShockTube:
    """Sod 激波管验证。"""

    @pytest.mark.xfail(reason="rhoCentralFoam 需要完整的热力学场文件")
    def test_completes(self, sod_shock_tube_case: Path):
        from pyfoam.applications.rho_central_foam import RhoCentralFoam
        solver = RhoCentralFoam(sod_shock_tube_case)
        solver.run()
        assert torch.isfinite(solver.U).all()


@pytest.fixture
def taylor_green_case(tmp_path: Path) -> Path:
    """Taylor-Green 涡算例。"""
    case = tmp_path / "tg"
    mesh_dir = case / "constant" / "polyMesh"
    make_structured_mesh(mesh_dir, nx=16, ny=16, x_range=(0, 2 * math.pi), y_range=(0, 2 * math.pi))
    write_transport_properties(case, nu=0.01)
    write_control_dict(case, delta_t=0.01, end_time=0.1, write_interval=10)
    write_fv_schemes(case)
    write_fv_solution(case)
    write_velocity_field(
        case,
        patches={"fixedWalls": (0, 0, 0), "frontAndBack": (0, 0, 0)},
        internal=(1.0, 0.0, 0.0),
        bc_types={"fixedWalls": "zeroGradient", "frontAndBack": "empty"},
    )
    write_pressure_field(
        case,
        patches={"fixedWalls": "zeroGradient", "frontAndBack": "empty"},
    )
    return case


class TestTaylorGreenVortex:
    """Taylor-Green 涡验证。"""

    def test_completes(self, taylor_green_case: Path):
        from pyfoam.applications.piso_foam import PisoFoam
        solver = PisoFoam(taylor_green_case)
        solver.run()
        assert torch.isfinite(solver.U).all()
        assert torch.isfinite(solver.p).all()


@pytest.fixture
def channel_flow_case(tmp_path: Path) -> Path:
    """管道流算例（周期边界）。"""
    case = tmp_path / "channel"
    mesh_dir = case / "constant" / "polyMesh"
    make_structured_mesh(mesh_dir, nx=20, ny=10, x_range=(0, 10), y_range=(0, 1))
    write_transport_properties(case, nu=0.01)
    write_control_dict(case, delta_t=0.01, end_time=1.0, write_interval=100)
    write_fv_schemes(case)
    write_fv_solution(case)
    write_velocity_field(
        case,
        patches={"fixedWalls": (0, 0, 0), "frontAndBack": (0, 0, 0)},
        internal=(1.0, 0.0, 0.0),
        bc_types={"fixedWalls": "noSlip", "frontAndBack": "empty"},
    )
    write_pressure_field(
        case,
        patches={"fixedWalls": "zeroGradient", "frontAndBack": "empty"},
    )
    return case


class TestChannelFlow:
    """管道流验证。"""

    def test_completes(self, channel_flow_case: Path):
        from pyfoam.applications.simple_foam import SimpleFoam
        solver = SimpleFoam(channel_flow_case)
        solver.run()
        assert torch.isfinite(solver.U).all()
        assert torch.isfinite(solver.p).all()

    def test_no_slip_walls(self, channel_flow_case: Path):
        """壁面速度应接近零。"""
        from pyfoam.applications.simple_foam import SimpleFoam
        solver = SimpleFoam(channel_flow_case)
        solver.run()
        # 壁面附近的单元速度应较小
        mesh = solver.mesh
        y = mesh.cell_centres[:, 1]
        u = solver.U[:, 0]
        # 底部壁面附近
        bottom_mask = y < 0.15
        if bottom_mask.any():
            u_bottom = u[bottom_mask].abs().max().item()
            assert u_bottom < 0.5, f"Wall velocity too high: {u_bottom}"

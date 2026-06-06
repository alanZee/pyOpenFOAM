"""
Tutorial validation: multiphase flow cases.

验证多相流求解器（interFoam, multiphaseEulerFoam）。
"""
from __future__ import annotations

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
def dam_break_case(tmp_path: Path) -> Path:
    """溃坝算例（VOF）。"""
    case = tmp_path / "damBreak"
    mesh_dir = case / "constant" / "polyMesh"
    make_structured_mesh(mesh_dir, nx=20, ny=10, x_range=(0, 4), y_range=(0, 2))
    write_transport_properties(case, nu=1e-6)
    write_control_dict(case, solver="interFoam", delta_t=0.001, end_time=0.1, write_interval=10)
    write_fv_schemes(case)
    write_fv_solution(case)
    write_velocity_field(
        case,
        patches={"fixedWalls": (0, 0, 0), "frontAndBack": (0, 0, 0)},
        bc_types={"fixedWalls": "noSlip", "frontAndBack": "empty"},
    )
    write_pressure_field(
        case,
        patches={"fixedWalls": "zeroGradient", "frontAndBack": "empty"},
    )
    return case


class TestDamBreak:
    """溃坝验证（需要完整 VOF 场文件）。"""

    @pytest.mark.xfail(reason="interFoam 需要 alpha.water 场文件")
    def test_completes(self, dam_break_case: Path):
        from pyfoam.applications.inter_foam import InterFoam
        solver = InterFoam(dam_break_case)
        solver.run()
        assert torch.isfinite(solver.U).all()


@pytest.fixture
def natural_convection_case(tmp_path: Path) -> Path:
    """自然对流方腔算例。"""
    case = tmp_path / "naturalConvection"
    mesh_dir = case / "constant" / "polyMesh"
    make_structured_mesh(mesh_dir, nx=16, ny=16)
    write_transport_properties(case, nu=1e-5)
    write_control_dict(case, solver="buoyantSimpleFoam", delta_t=0.001, end_time=0.5, write_interval=50)
    write_fv_schemes(case)
    write_fv_solution(case)
    write_velocity_field(
        case,
        patches={"movingWall": (0, 0, 0), "fixedWalls": (0, 0, 0), "frontAndBack": (0, 0, 0)},
        bc_types={"movingWall": "fixedValue", "fixedWalls": "noSlip", "frontAndBack": "empty"},
    )
    write_pressure_field(
        case,
        patches={"movingWall": "zeroGradient", "fixedWalls": "zeroGradient", "frontAndBack": "empty"},
    )
    return case


class TestNaturalConvection:
    """自然对流验证（需要温度场和浮力模型）。"""

    @pytest.mark.xfail(reason="buoyantSimpleFoam 需要完整温度场和重力设置")
    def test_completes(self, natural_convection_case: Path):
        from pyfoam.applications.buoyant_simple_foam import BuoyantSimpleFoam
        solver = BuoyantSimpleFoam(natural_convection_case)
        solver.run()
        assert torch.isfinite(solver.U).all()

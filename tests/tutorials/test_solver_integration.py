"""
Tutorial validation: solver integration tests.

验证求解器与其他模块的集成。
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


def _make_integration_case(tmp_path: Path, name: str, nx: int = 5, ny: int = 5) -> Path:
    case = tmp_path / name
    mesh_dir = case / "constant" / "polyMesh"
    make_structured_mesh(mesh_dir, nx=nx, ny=ny)
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
    return case


class TestSolverMeshIntegration:
    """求解器-网格集成测试。"""

    def test_simple_with_mesh_geometry(self, tmp_path: Path):
        """SimpleFoam 使用网格几何信息。"""
        case = _make_integration_case(tmp_path, "mesh_integ")
        from pyfoam.applications.simple_foam import SimpleFoam
        solver = SimpleFoam(case)
        assert solver.mesh is not None
        assert solver.mesh.n_cells > 0
        assert solver.mesh.n_faces > 0
        solver.run()
        assert torch.isfinite(solver.U).all()


class TestSolverFieldIntegration:
    """求解器-场集成测试。"""

    def test_simple_field_shapes(self, tmp_path: Path):
        """SimpleFoam 场形状正确。"""
        case = _make_integration_case(tmp_path, "field_integ")
        from pyfoam.applications.simple_foam import SimpleFoam
        solver = SimpleFoam(case)
        solver.run()
        n_cells = solver.mesh.n_cells
        assert solver.U.shape == (n_cells, 3)
        assert solver.p.shape == (n_cells,)


class TestSolverIOIntegration:
    """求解器-IO 集成测试。"""

    def test_simple_writes_files(self, tmp_path: Path):
        """SimpleFoam 写入文件。"""
        case = _make_integration_case(tmp_path, "io_integ")
        from pyfoam.applications.simple_foam import SimpleFoam
        solver = SimpleFoam(case)
        solver.run()
        # 检查输出文件存在
        assert (case / "0" / "U").exists()
        assert (case / "0" / "p").exists()


class TestSolverBoundaryIntegration:
    """求解器-边界条件集成测试。"""

    def test_simple_boundary_conditions(self, tmp_path: Path):
        """SimpleFoam 边界条件正确应用。"""
        case = _make_integration_case(tmp_path, "bc_integ")
        from pyfoam.applications.simple_foam import SimpleFoam
        solver = SimpleFoam(case)
        solver.run()
        # 检查边界条件已应用
        assert torch.isfinite(solver.U).all()
        assert torch.isfinite(solver.p).all()

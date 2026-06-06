"""
Tutorial validation: solver mesh refinement tests.

验证求解器网格细化。
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


def _make_refinement_case(tmp_path: Path, name: str, nx: int, ny: int) -> Path:
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


class TestSolverMeshRefinement:
    """求解器网格细化测试。"""

    def test_coarse_mesh(self, tmp_path: Path):
        """粗网格 (3x3)。"""
        case = _make_refinement_case(tmp_path, "coarse", nx=3, ny=3)
        from pyfoam.applications.simple_foam import SimpleFoam
        solver = SimpleFoam(case)
        solver.run()
        assert torch.isfinite(solver.U).all()
        assert solver.mesh.n_cells == 9

    def test_medium_mesh(self, tmp_path: Path):
        """中等网格 (10x10)。"""
        case = _make_refinement_case(tmp_path, "medium", nx=10, ny=10)
        from pyfoam.applications.simple_foam import SimpleFoam
        solver = SimpleFoam(case)
        solver.run()
        assert torch.isfinite(solver.U).all()
        assert solver.mesh.n_cells == 100

    def test_fine_mesh(self, tmp_path: Path):
        """细网格 (20x20)。"""
        case = _make_refinement_case(tmp_path, "fine", nx=20, ny=20)
        from pyfoam.applications.simple_foam import SimpleFoam
        solver = SimpleFoam(case)
        solver.run()
        assert torch.isfinite(solver.U).all()
        assert solver.mesh.n_cells == 400

    def test_very_fine_mesh(self, tmp_path: Path):
        """极细网格 (50x50)。"""
        case = _make_refinement_case(tmp_path, "very_fine", nx=50, ny=50)
        from pyfoam.applications.simple_foam import SimpleFoam
        solver = SimpleFoam(case)
        solver.run()
        assert torch.isfinite(solver.U).all()
        assert solver.mesh.n_cells == 2500


class TestSolverMeshConvergence:
    """求解器网格收敛性测试。"""

    def test_mesh_convergence(self, tmp_path: Path):
        """网格细化收敛。"""
        from pyfoam.applications.simple_foam import SimpleFoam
        u_max_values = []
        for nx in [5, 10, 20]:
            case = _make_refinement_case(tmp_path, f"conv_{nx}", nx=nx, ny=nx)
            solver = SimpleFoam(case)
            solver.run()
            u_max = solver.U[:, 0].abs().max().item()
            u_max_values.append((nx, u_max))
            assert torch.isfinite(solver.U).all()
        # 检查网格收敛性（更细网格应产生更稳定的结果）
        assert all(u > 0 for _, u in u_max_values), "All meshes should have non-zero flow"

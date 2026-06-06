"""
Tutorial validation: solver consistency tests.

验证求解器结果一致性。
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


def _make_consistency_case(tmp_path: Path, name: str, nx: int = 5, ny: int = 5) -> Path:
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


class TestSolverConsistency:
    """求解器一致性测试。"""

    def test_simple_deterministic(self, tmp_path: Path):
        """SimpleFoam 结果确定性。"""
        case1 = _make_consistency_case(tmp_path, "det1", nx=5, ny=5)
        case2 = _make_consistency_case(tmp_path, "det2", nx=5, ny=5)
        from pyfoam.applications.simple_foam import SimpleFoam
        solver1 = SimpleFoam(case1)
        solver1.run()
        solver2 = SimpleFoam(case2)
        solver2.run()
        # 相同输入应产生相同输出
        assert torch.allclose(solver1.U, solver2.U, atol=1e-10)
        assert torch.allclose(solver1.p, solver2.p, atol=1e-10)

    def test_simple_reproducible(self, tmp_path: Path):
        """SimpleFoam 结果可重现。"""
        case = _make_consistency_case(tmp_path, "repr", nx=5, ny=5)
        from pyfoam.applications.simple_foam import SimpleFoam
        solver1 = SimpleFoam(case)
        solver1.run()
        # 清理并重新运行
        solver2 = SimpleFoam(case)
        solver2.run()
        # 结果应相同
        assert torch.allclose(solver1.U, solver2.U, atol=1e-10)
        assert torch.allclose(solver1.p, solver2.p, atol=1e-10)


class TestSolverFieldConsistency:
    """求解器场一致性测试。"""

    def test_simple_field_shapes_consistent(self, tmp_path: Path):
        """SimpleFoam 场形状一致。"""
        case = _make_consistency_case(tmp_path, "shapes", nx=5, ny=5)
        from pyfoam.applications.simple_foam import SimpleFoam
        solver = SimpleFoam(case)
        solver.run()
        n_cells = solver.mesh.n_cells
        assert solver.U.shape == (n_cells, 3)
        assert solver.p.shape == (n_cells,)

    def test_simple_field_finite(self, tmp_path: Path):
        """SimpleFoam 场值有限。"""
        case = _make_consistency_case(tmp_path, "finite", nx=5, ny=5)
        from pyfoam.applications.simple_foam import SimpleFoam
        solver = SimpleFoam(case)
        solver.run()
        assert torch.isfinite(solver.U).all()
        assert torch.isfinite(solver.p).all()

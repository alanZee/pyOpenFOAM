"""
Tutorial validation: solver output validation tests.

验证求解器输出正确性。
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


def _make_output_case(tmp_path: Path, name: str, nx: int = 10, ny: int = 10) -> Path:
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


class TestSolverOutput:
    """求解器输出测试。"""

    def test_output_field_shape(self, tmp_path: Path):
        """输出场形状正确。"""
        case = _make_output_case(tmp_path, "output_shape")
        from pyfoam.applications.simple_foam import SimpleFoam
        solver = SimpleFoam(case)
        solver.run()
        n_cells = solver.mesh.n_cells
        assert solver.U.shape == (n_cells, 3)
        assert solver.p.shape == (n_cells,)

    def test_output_field_finite(self, tmp_path: Path):
        """输出场有限。"""
        case = _make_output_case(tmp_path, "output_finite")
        from pyfoam.applications.simple_foam import SimpleFoam
        solver = SimpleFoam(case)
        solver.run()
        assert torch.isfinite(solver.U).all()
        assert torch.isfinite(solver.p).all()

    def test_output_field_nonzero(self, tmp_path: Path):
        """输出场非零。"""
        case = _make_output_case(tmp_path, "output_nonzero")
        from pyfoam.applications.simple_foam import SimpleFoam
        solver = SimpleFoam(case)
        solver.run()
        assert solver.U.abs().sum() > 0

    def test_output_field_bounded(self, tmp_path: Path):
        """输出场有界。"""
        case = _make_output_case(tmp_path, "output_bounded")
        from pyfoam.applications.simple_foam import SimpleFoam
        solver = SimpleFoam(case)
        solver.run()
        u_max = solver.U.abs().max().item()
        assert u_max < 100.0, f"Velocity too high: {u_max}"
        p_max = solver.p.abs().max().item()
        assert p_max < 1000.0, f"Pressure too high: {p_max}"


class TestSolverOutputConsistency:
    """求解器输出一致性测试。"""

    def test_deterministic_output(self, tmp_path: Path):
        """确定性输出。"""
        case1 = _make_output_case(tmp_path, "det1")
        case2 = _make_output_case(tmp_path, "det2")
        from pyfoam.applications.simple_foam import SimpleFoam
        solver1 = SimpleFoam(case1)
        solver1.run()
        solver2 = SimpleFoam(case2)
        solver2.run()
        assert torch.allclose(solver1.U, solver2.U, atol=1e-10)
        assert torch.allclose(solver1.p, solver2.p, atol=1e-10)

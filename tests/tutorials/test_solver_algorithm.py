"""
Tutorial validation: solver algorithm tests.

验证求解器算法正确性。
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


def _make_algorithm_case(tmp_path: Path, name: str, nx: int = 10, ny: int = 10) -> Path:
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


class TestAlgorithmCorrectness:
    """算法正确性测试。"""

    def test_simple_algorithm(self, tmp_path: Path):
        """SIMPLE 算法。"""
        case = _make_algorithm_case(tmp_path, "simple_algo")
        from pyfoam.applications.simple_foam import SimpleFoam
        solver = SimpleFoam(case)
        solver.run()
        assert torch.isfinite(solver.U).all()
        assert torch.isfinite(solver.p).all()
        assert solver.U.abs().sum() > 0

    def test_piso_algorithm(self, tmp_path: Path):
        """PISO 算法。"""
        case = _make_algorithm_case(tmp_path, "piso_algo")
        from pyfoam.applications.piso_foam import PisoFoam
        solver = PisoFoam(case)
        solver.run()
        assert torch.isfinite(solver.U).all()
        assert torch.isfinite(solver.p).all()

    def test_pimple_algorithm(self, tmp_path: Path):
        """PIMPLE 算法。"""
        case = _make_algorithm_case(tmp_path, "pimple_algo")
        from pyfoam.applications.pimple_foam import PimpleFoam
        solver = PimpleFoam(case)
        solver.run()
        assert torch.isfinite(solver.U).all()
        assert torch.isfinite(solver.p).all()


class TestAlgorithmConsistency:
    """算法一致性测试。"""

    def test_simple_deterministic(self, tmp_path: Path):
        """SIMPLE 算法确定性。"""
        case1 = _make_algorithm_case(tmp_path, "det1")
        case2 = _make_algorithm_case(tmp_path, "det2")
        from pyfoam.applications.simple_foam import SimpleFoam
        solver1 = SimpleFoam(case1)
        solver1.run()
        solver2 = SimpleFoam(case2)
        solver2.run()
        assert torch.allclose(solver1.U, solver2.U, atol=1e-10)
        assert torch.allclose(solver1.p, solver2.p, atol=1e-10)

    def test_piso_deterministic(self, tmp_path: Path):
        """PISO 算法确定性。"""
        case1 = _make_algorithm_case(tmp_path, "piso_det1")
        case2 = _make_algorithm_case(tmp_path, "piso_det2")
        from pyfoam.applications.piso_foam import PisoFoam
        solver1 = PisoFoam(case1)
        solver1.run()
        solver2 = PisoFoam(case2)
        solver2.run()
        assert torch.allclose(solver1.U, solver2.U, atol=1e-10)
        assert torch.allclose(solver1.p, solver2.p, atol=1e-10)

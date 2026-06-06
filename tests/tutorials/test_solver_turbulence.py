"""
Tutorial validation: solver turbulence coupling tests.

验证求解器-湍流耦合。
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


def _make_turb_case(tmp_path: Path, name: str, nx: int = 10, ny: int = 10) -> Path:
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


class TestTurbulenceCoupling:
    """湍流耦合测试。"""

    def test_k_epsilon_import(self):
        """k-epsilon 模型可导入。"""
        from pyfoam.turbulence.k_epsilon import KEpsilonModel
        assert KEpsilonModel is not None

    def test_k_omega_sst_import(self):
        """k-omega SST 模型可导入。"""
        from pyfoam.turbulence.k_omega_sst import KOmegaSSTModel
        assert KOmegaSSTModel is not None

    def test_spalart_allmaras_import(self):
        """Spalart-Allmaras 模型可导入。"""
        from pyfoam.turbulence.spalart_allmaras import SpalartAllmarasModel
        assert SpalartAllmarasModel is not None

    def test_smagorinsky_import(self):
        """Smagorinsky LES 模型可导入。"""
        from pyfoam.turbulence.smagorinsky import SmagorinskyModel
        assert SmagorinskyModel is not None

    def test_wale_import(self):
        """WALE LES 模型可导入。"""
        from pyfoam.turbulence.wale import WALEModel
        assert WALEModel is not None


class TestTurbulenceModelProperties:
    """湍流模型属性测试。"""

    def test_k_epsilon_properties(self):
        """k-epsilon 模型属性。"""
        from pyfoam.turbulence.k_epsilon import KEpsilonModel
        assert hasattr(KEpsilonModel, 'k')
        assert hasattr(KEpsilonModel, 'epsilon')
        assert hasattr(KEpsilonModel, 'nut')

    def test_k_omega_sst_properties(self):
        """k-omega SST 模型属性。"""
        from pyfoam.turbulence.k_omega_sst import KOmegaSSTModel
        assert hasattr(KOmegaSSTModel, 'k')
        assert hasattr(KOmegaSSTModel, 'omega')
        assert hasattr(KOmegaSSTModel, 'nut')

    def test_spalart_allmaras_properties(self):
        """Spalart-Allmaras 模型属性。"""
        from pyfoam.turbulence.spalart_allmaras import SpalartAllmarasModel
        assert hasattr(SpalartAllmarasModel, 'nuTilde_field')
        assert hasattr(SpalartAllmarasModel, 'nut')

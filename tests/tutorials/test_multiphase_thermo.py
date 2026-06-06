"""
Tutorial validation: multiphase model smoke tests.

验证多相流模型的基本功能。
"""
from __future__ import annotations

import torch
import pytest
from pyfoam.core.dtype import CFD_DTYPE


class TestMultiphaseModelSmoke:
    """多相流模型 smoke 测试。"""

    def test_vof_import(self):
        """VOF 模型可导入。"""
        from pyfoam.multiphase import CompressibleMultiphaseVoF
        assert CompressibleMultiphaseVoF is not None

    def test_surface_tension_import(self):
        """表面张力模型可导入。"""
        from pyfoam.multiphase import CSFSurfaceTension
        assert CSFSurfaceTension is not None

    def test_drag_model_import(self):
        """阻力模型可导入。"""
        from pyfoam.multiphase import BubbleModel
        assert BubbleModel is not None


class TestThermophysicalModelSmoke:
    """热物理模型 smoke 测试。"""

    def test_perfect_gas_import(self):
        """理想气体 EOS 可导入。"""
        from pyfoam.thermophysical import PerfectGas
        assert PerfectGas is not None

    def test_constant_transport_import(self):
        """常输运模型可导入。"""
        from pyfoam.thermophysical import ConstantTransport
        assert ConstantTransport is not None

    def test_sutherland_import(self):
        """Sutherland 输运模型可导入。"""
        from pyfoam.thermophysical import Sutherland
        assert Sutherland is not None

    def test_basic_thermo_import(self):
        """BasicThermo 可导入。"""
        from pyfoam.thermophysical import BasicThermo
        assert BasicThermo is not None

    def test_combustion_model_import(self):
        """燃烧模型可导入。"""
        from pyfoam.thermophysical import CombustionModel
        assert CombustionModel is not None


class TestLagrangianSmoke:
    """拉格朗日模型 smoke 测试。"""

    def test_breakup_model_import(self):
        """破碎模型可导入。"""
        from pyfoam.lagrangian import BreakupModel
        assert BreakupModel is not None

    def test_buoyancy_force_import(self):
        """浮力模型可导入。"""
        from pyfoam.lagrangian import BuoyancyForce
        assert BuoyancyForce is not None

    def test_brownian_motion_import(self):
        """布朗运动模型可导入。"""
        from pyfoam.lagrangian import BrownianMotionForce
        assert BrownianMotionForce is not None

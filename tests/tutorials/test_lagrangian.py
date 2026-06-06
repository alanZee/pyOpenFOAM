"""
Tutorial validation: lagrangian smoke tests.

验证拉格朗日粒子模型的基本功能。
"""
from __future__ import annotations

import torch
import pytest
from pyfoam.core.dtype import CFD_DTYPE


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

    def test_drag_force_import(self):
        """阻力模型可导入。"""
        from pyfoam.lagrangian import DragForce
        assert DragForce is not None

    def test_lift_force_import(self):
        """升力模型可导入。"""
        from pyfoam.lagrangian import LiftForce
        assert LiftForce is not None

    def test_virtual_mass_force_import(self):
        """虚拟质量力可导入。"""
        from pyfoam.lagrangian import VirtualMassForce
        assert VirtualMassForce is not None

    def test_collision_model_import(self):
        """碰撞模型可导入。"""
        from pyfoam.lagrangian import CollisionModel
        assert CollisionModel is not None

    def test_injection_model_import(self):
        """注入模型可导入。"""
        from pyfoam.lagrangian import Injector
        assert Injector is not None

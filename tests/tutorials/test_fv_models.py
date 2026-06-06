"""
Tutorial validation: fvModels smoke tests.

验证 fvModels 源项框架的基本功能。
"""
from __future__ import annotations

import torch
import pytest
from pyfoam.core.dtype import CFD_DTYPE


class TestFvModelsSmoke:
    """fvModels smoke 测试。"""

    def test_fv_model_import(self):
        """FvModel 可导入。"""
        from pyfoam.fv import FvModel
        assert FvModel is not None

    def test_explicit_source_import(self):
        """ExplicitSource 可导入。"""
        from pyfoam.fv import ExplicitSource
        assert ExplicitSource is not None

    def test_semi_implicit_source_import(self):
        """SemiImplicitSource 可导入。"""
        from pyfoam.fv import SemiImplicitSource
        assert SemiImplicitSource is not None

    def test_buoyancy_force_import(self):
        """BuoyancyForce 可导入。"""
        from pyfoam.fv import BuoyancyForce
        assert BuoyancyForce is not None

    def test_mrf_source_import(self):
        """MRFSource 可导入。"""
        from pyfoam.fv import MRFSource
        assert MRFSource is not None

    def test_heat_source_import(self):
        """HeatSource 可导入。"""
        from pyfoam.fv import HeatSource
        assert HeatSource is not None

    def test_mass_source_import(self):
        """MassSource 可导入。"""
        from pyfoam.fv import MassSource
        assert MassSource is not None

    def test_porosity_force_import(self):
        """PorosityForce 可导入。"""
        from pyfoam.fv import PorosityForce
        assert PorosityForce is not None


class TestFvConstraintsSmoke:
    """fvConstraints smoke 测试。"""

    def test_fv_constraint_import(self):
        """FvConstraint 可导入。"""
        from pyfoam.fv import FvConstraint
        assert FvConstraint is not None

    def test_min_max_import(self):
        """MinMax 约束可导入。"""
        from pyfoam.fv import MinMaxConstraint
        assert MinMaxConstraint is not None

    def test_velocity_limits_import(self):
        """VelocityLimits 约束可导入。"""
        from pyfoam.fv import VelocityLimitsConstraint
        assert VelocityLimitsConstraint is not None

    def test_pressure_limits_import(self):
        """PressureLimits 约束可导入。"""
        from pyfoam.fv import LimitPressureConstraint
        assert LimitPressureConstraint is not None

    def test_temperature_limits_import(self):
        """TemperatureLimits 约束可导入。"""
        from pyfoam.fv import LimitTemperatureConstraint
        assert LimitTemperatureConstraint is not None

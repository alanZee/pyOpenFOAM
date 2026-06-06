"""
Tutorial validation: radiation and combustion smoke tests.

验证辐射和燃烧模型的基本功能。
"""
from __future__ import annotations

import torch
import pytest
from pyfoam.core.dtype import CFD_DTYPE


class TestRadiationSmoke:
    """辐射模型 smoke 测试。"""

    def test_p1_radiation_import(self):
        """P1 辐射模型可导入。"""
        from pyfoam.models.radiation import P1Radiation
        assert P1Radiation is not None

    def test_radiation_model_import(self):
        """RadiationModel 基类可导入。"""
        from pyfoam.models.radiation import RadiationModel
        assert RadiationModel is not None

    @pytest.mark.xfail(reason="ViewFactor not yet implemented")
    def test_view_factor_import(self):
        """视角因子模型可导入。"""
        from pyfoam.models.radiation import ViewFactor
        assert ViewFactor is not None

    @pytest.mark.xfail(reason="OpaqueSolid not yet implemented")
    def test_opaque_solid_import(self):
        """不透明固体辐射可导入。"""
        from pyfoam.models.radiation import OpaqueSolid
        assert OpaqueSolid is not None


class TestCombustionSmoke:
    """燃烧模型 smoke 测试。"""

    def test_arrhenius_import(self):
        """Arrhenius 反应速率可导入。"""
        from pyfoam.thermophysical import ArrheniusReaction
        assert ArrheniusReaction is not None

    def test_chemistry_model_import(self):
        """化学模型可导入。"""
        from pyfoam.thermophysical import ChemistryModel
        assert ChemistryModel is not None

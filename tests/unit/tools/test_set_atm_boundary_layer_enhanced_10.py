"""Tests for set_atm_boundary_layer_enhanced_10."""
from __future__ import annotations
import pytest
from pyfoam.tools.set_atm_boundary_layer_enhanced_10 import EnhancedABL10Result, CanopyModelResult, UrbanHeatIslandResult, set_atm_boundary_layer_enhanced_10


class TestEnhancedABL10Result:
    def test_returns_result(self):
        r = set_atm_boundary_layer_enhanced_10()
        assert isinstance(r, EnhancedABL10Result)

    def test_canopy(self):
        r = set_atm_boundary_layer_enhanced_10(enable_canopy=True)
        assert isinstance(r.canopy, CanopyModelResult)
        assert r.canopy.enabled is True

    def test_urban_heat(self):
        r = set_atm_boundary_layer_enhanced_10(enable_urban_heat=True)
        assert isinstance(r.urban_heat, UrbanHeatIslandResult)
        assert r.urban_heat.enabled is True

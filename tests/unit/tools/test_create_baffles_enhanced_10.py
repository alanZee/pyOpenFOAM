"""Tests for create_baffles_enhanced_10."""
from __future__ import annotations
import pytest
from pyfoam.tools.create_baffles_enhanced_10 import BaffleEnhanced10Result, PorousBaffleModel, ThermalResistanceBaffle, create_baffles_enhanced_10


class TestBaffleEnhanced10Result:
    def test_returns_result(self):
        r = create_baffles_enhanced_10()
        assert isinstance(r, BaffleEnhanced10Result)

    def test_porous_model(self):
        r = create_baffles_enhanced_10(enable_porous_model=True)
        assert isinstance(r.porous_model, PorousBaffleModel)
        assert r.porous_model.enabled is True

    def test_thermal_resistance(self):
        r = create_baffles_enhanced_10(enable_thermal_resistance=True)
        assert isinstance(r.thermal_resistance, ThermalResistanceBaffle)
        assert r.thermal_resistance.enabled is True

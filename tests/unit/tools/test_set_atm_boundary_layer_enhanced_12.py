"""Tests for set_atm_boundary_layer_enhanced_12."""
from __future__ import annotations
import pytest
from pyfoam.tools.set_atm_boundary_layer_enhanced_12 import EnhancedABL12Result, WeatherDataResult, MultiScaleAtmosphericResult, set_atm_boundary_layer_enhanced_12


class TestEnhancedABL12Result:
    def test_returns_result(self):
        r = set_atm_boundary_layer_enhanced_12()
        assert isinstance(r, EnhancedABL12Result)

    def test_weather(self):
        r = set_atm_boundary_layer_enhanced_12(enable_weather=True)
        assert isinstance(r.weather, WeatherDataResult)
        assert r.weather.enabled is True

    def test_multi_scale(self):
        r = set_atm_boundary_layer_enhanced_12(enable_multi_scale=True)
        assert isinstance(r.multi_scale, MultiScaleAtmosphericResult)
        assert r.multi_scale.enabled is True

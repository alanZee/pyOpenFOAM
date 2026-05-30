"""Tests for set_atm_boundary_layer_enhanced_11."""
from __future__ import annotations
import pytest
from pyfoam.tools.set_atm_boundary_layer_enhanced_11 import EnhancedABL11Result, PollutantDispersionResult, TerrainFollowingResult, set_atm_boundary_layer_enhanced_11


class TestEnhancedABL11Result:
    def test_returns_result(self):
        r = set_atm_boundary_layer_enhanced_11()
        assert isinstance(r, EnhancedABL11Result)

    def test_pollutant(self):
        r = set_atm_boundary_layer_enhanced_11(enable_pollutant=True)
        assert isinstance(r.pollutant, PollutantDispersionResult)
        assert r.pollutant.enabled is True

    def test_terrain_coords(self):
        r = set_atm_boundary_layer_enhanced_11(enable_terrain_coords=True)
        assert isinstance(r.terrain_coords, TerrainFollowingResult)
        assert r.terrain_coords.enabled is True

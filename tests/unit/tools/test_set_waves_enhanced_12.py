"""Tests for set_waves_enhanced_12."""
from __future__ import annotations
import pytest
from pyfoam.tools.set_waves_enhanced_12 import EnhancedWave12Result, WaveFarmResult, CoupledWaveStructureResult, set_waves_enhanced_12


class TestEnhancedWave12Result:
    def test_returns_result(self):
        r = set_waves_enhanced_12()
        assert isinstance(r, EnhancedWave12Result)

    def test_farm(self):
        r = set_waves_enhanced_12(enable_farm=True)
        assert isinstance(r.farm, WaveFarmResult)
        assert r.farm.enabled is True

    def test_structure_coupling(self):
        r = set_waves_enhanced_12(enable_structure_coupling=True)
        assert isinstance(r.structure_coupling, CoupledWaveStructureResult)
        assert r.structure_coupling.enabled is True

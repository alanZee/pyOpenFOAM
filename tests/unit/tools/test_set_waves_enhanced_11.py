"""Tests for set_waves_enhanced_11."""
from __future__ import annotations
import pytest
from pyfoam.tools.set_waves_enhanced_11 import EnhancedWave11Result, WaveFloatingBodyResult, WaveSedimentResult, set_waves_enhanced_11


class TestEnhancedWave11Result:
    def test_returns_result(self):
        r = set_waves_enhanced_11()
        assert isinstance(r, EnhancedWave11Result)

    def test_floating_body(self):
        r = set_waves_enhanced_11(enable_floating_body=True)
        assert isinstance(r.floating_body, WaveFloatingBodyResult)
        assert r.floating_body.enabled is True

    def test_sediment(self):
        r = set_waves_enhanced_11(enable_sediment=True)
        assert isinstance(r.sediment, WaveSedimentResult)
        assert r.sediment.enabled is True

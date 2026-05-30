"""Tests for set_waves_enhanced_10."""
from __future__ import annotations
import pytest
from pyfoam.tools.set_waves_enhanced_10 import EnhancedWave10Result, WaveBreakingResult, WaveCurrentResult, set_waves_enhanced_10


class TestEnhancedWave10Result:
    def test_returns_result(self):
        r = set_waves_enhanced_10()
        assert isinstance(r, EnhancedWave10Result)

    def test_breaking(self):
        r = set_waves_enhanced_10(enable_breaking=True)
        assert isinstance(r.breaking, WaveBreakingResult)
        assert r.breaking.enabled is True

    def test_current_interaction(self):
        r = set_waves_enhanced_10(enable_current_interaction=True)
        assert isinstance(r.current_interaction, WaveCurrentResult)
        assert r.current_interaction.enabled is True

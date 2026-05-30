"""Tests for create_baffles_enhanced_12."""
from __future__ import annotations
import pytest
from pyfoam.tools.create_baffles_enhanced_12 import BaffleEnhanced12Result, SmartBaffleControl, BaffleFatigueResult, create_baffles_enhanced_12


class TestBaffleEnhanced12Result:
    def test_returns_result(self):
        r = create_baffles_enhanced_12()
        assert isinstance(r, BaffleEnhanced12Result)

    def test_smart_control(self):
        r = create_baffles_enhanced_12(enable_smart_control=True)
        assert isinstance(r.smart_control, SmartBaffleControl)
        assert r.smart_control.enabled is True

    def test_fatigue(self):
        r = create_baffles_enhanced_12(enable_fatigue=True)
        assert isinstance(r.fatigue, BaffleFatigueResult)
        assert r.fatigue.enabled is True

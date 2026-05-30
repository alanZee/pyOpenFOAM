"""Tests for create_baffles_enhanced_11."""
from __future__ import annotations
import pytest
from pyfoam.tools.create_baffles_enhanced_11 import BaffleEnhanced11Result, AdaptiveBaffleGeometry, MultiZoneBaffleResult, create_baffles_enhanced_11


class TestBaffleEnhanced11Result:
    def test_returns_result(self):
        r = create_baffles_enhanced_11()
        assert isinstance(r, BaffleEnhanced11Result)

    def test_adaptive_geometry(self):
        r = create_baffles_enhanced_11(enable_adaptive_geometry=True)
        assert isinstance(r.adaptive_geometry, AdaptiveBaffleGeometry)
        assert r.adaptive_geometry.enabled is True

    def test_multi_zone(self):
        r = create_baffles_enhanced_11(enable_multi_zone=True)
        assert isinstance(r.multi_zone, MultiZoneBaffleResult)
        assert r.multi_zone.enabled is True

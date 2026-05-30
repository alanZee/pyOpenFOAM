"""Tests for surface_check_enhanced_10."""
from __future__ import annotations
import pytest
from pyfoam.tools.surface_check_enhanced_10 import SurfaceCheckEnhanced10Result, IntersectionResult, NormalConsistencyResult, surface_check_enhanced_10


class TestSurfaceCheckEnhanced10Result:
    def test_returns_result(self):
        r = surface_check_enhanced_10()
        assert isinstance(r, SurfaceCheckEnhanced10Result)

    def test_intersections(self):
        r = surface_check_enhanced_10(enable_intersections=True)
        assert isinstance(r.intersections, IntersectionResult)
        assert r.intersections.enabled is True

    def test_normal_consistency(self):
        r = surface_check_enhanced_10(enable_normal_consistency=True)
        assert isinstance(r.normal_consistency, NormalConsistencyResult)
        assert r.normal_consistency.enabled is True

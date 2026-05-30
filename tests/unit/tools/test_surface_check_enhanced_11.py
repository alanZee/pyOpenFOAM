"""Tests for surface_check_enhanced_11."""
from __future__ import annotations
import pytest
from pyfoam.tools.surface_check_enhanced_11 import SurfaceCheckEnhanced11Result, SelfProximityResult, CurvatureQualityResult, surface_check_enhanced_11


class TestSurfaceCheckEnhanced11Result:
    def test_returns_result(self):
        r = surface_check_enhanced_11()
        assert isinstance(r, SurfaceCheckEnhanced11Result)

    def test_proximity(self):
        r = surface_check_enhanced_11(enable_proximity=True)
        assert isinstance(r.proximity, SelfProximityResult)
        assert r.proximity.enabled is True

    def test_curvature_quality(self):
        r = surface_check_enhanced_11(enable_curvature_quality=True)
        assert isinstance(r.curvature_quality, CurvatureQualityResult)
        assert r.curvature_quality.enabled is True

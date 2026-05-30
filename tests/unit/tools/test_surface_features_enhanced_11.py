"""Tests for surface_features_enhanced_11."""
from __future__ import annotations
import pytest
from pyfoam.tools.surface_features_enhanced_11 import SurfaceFeaturesEnhanced11Result, CurvatureAdaptiveFeatureResult, FeaturePersistenceResult, surface_features_enhanced_11


class TestSurfaceFeaturesEnhanced11Result:
    def test_returns_result(self):
        r = surface_features_enhanced_11()
        assert isinstance(r, SurfaceFeaturesEnhanced11Result)

    def test_curvature_adaptive(self):
        r = surface_features_enhanced_11(enable_curvature_adaptive=True)
        assert isinstance(r.curvature_adaptive, CurvatureAdaptiveFeatureResult)
        assert r.curvature_adaptive.enabled is True

    def test_persistence(self):
        r = surface_features_enhanced_11(enable_persistence=True)
        assert isinstance(r.persistence, FeaturePersistenceResult)
        assert r.persistence.enabled is True

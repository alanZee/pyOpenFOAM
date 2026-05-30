"""Tests for surface_features_enhanced_10."""
from __future__ import annotations
import pytest
from pyfoam.tools.surface_features_enhanced_10 import SurfaceFeaturesEnhanced10Result, HierarchicalFeaturesResult, FeatureClusteringResult, surface_features_enhanced_10


class TestSurfaceFeaturesEnhanced10Result:
    def test_returns_result(self):
        r = surface_features_enhanced_10()
        assert isinstance(r, SurfaceFeaturesEnhanced10Result)

    def test_hierarchical(self):
        r = surface_features_enhanced_10(enable_hierarchical=True)
        assert isinstance(r.hierarchical, HierarchicalFeaturesResult)
        assert r.hierarchical.enabled is True

    def test_clustering(self):
        r = surface_features_enhanced_10(enable_clustering=True)
        assert isinstance(r.clustering, FeatureClusteringResult)
        assert r.clustering.enabled is True

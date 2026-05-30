"""Tests for surface_features_enhanced_12."""
from __future__ import annotations
import pytest
from pyfoam.tools.surface_features_enhanced_12 import SurfaceFeaturesEnhanced12Result, MLFeatureDetectionResult, FeatureTopologyResult, surface_features_enhanced_12


class TestSurfaceFeaturesEnhanced12Result:
    def test_returns_result(self):
        r = surface_features_enhanced_12()
        assert isinstance(r, SurfaceFeaturesEnhanced12Result)

    def test_ml_detection(self):
        r = surface_features_enhanced_12(enable_ml_detection=True)
        assert isinstance(r.ml_detection, MLFeatureDetectionResult)
        assert r.ml_detection.enabled is True

    def test_topology_graph(self):
        r = surface_features_enhanced_12(enable_topology_graph=True)
        assert isinstance(r.topology_graph, FeatureTopologyResult)
        assert r.topology_graph.enabled is True

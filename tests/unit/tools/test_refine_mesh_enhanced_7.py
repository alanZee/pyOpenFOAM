"""Tests for refine_mesh_enhanced_7."""
from __future__ import annotations
import pytest
from pyfoam.tools.refine_mesh_enhanced_7 import RefineEnhanced7Result, RefinementZonesResult, FeatureRefineResult, refine_mesh_enhanced_7


class TestRefineEnhanced7Result:
    def test_returns_result(self):
        r = refine_mesh_enhanced_7()
        assert isinstance(r, RefineEnhanced7Result)

    def test_zones(self):
        r = refine_mesh_enhanced_7(enable_zones=True)
        assert isinstance(r.zones, RefinementZonesResult)
        assert r.zones.enabled is True

    def test_feature_based(self):
        r = refine_mesh_enhanced_7(enable_feature_based=True)
        assert isinstance(r.feature_based, FeatureRefineResult)
        assert r.feature_based.enabled is True

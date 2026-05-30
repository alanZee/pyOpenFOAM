"""Tests for surface_auto_patch_enhanced_10."""
from __future__ import annotations
import pytest
from pyfoam.tools.surface_auto_patch_enhanced_10 import SurfaceAutoPatchEnhanced10Result, RegionGrowingResult, FeatureAwarePatchingResult, surface_auto_patch_enhanced_10


class TestSurfaceAutoPatchEnhanced10Result:
    def test_returns_result(self):
        r = surface_auto_patch_enhanced_10()
        assert isinstance(r, SurfaceAutoPatchEnhanced10Result)

    def test_region_growing(self):
        r = surface_auto_patch_enhanced_10(enable_region_growing=True)
        assert isinstance(r.region_growing, RegionGrowingResult)
        assert r.region_growing.enabled is True

    def test_feature_aware(self):
        r = surface_auto_patch_enhanced_10(enable_feature_aware=True)
        assert isinstance(r.feature_aware, FeatureAwarePatchingResult)
        assert r.feature_aware.enabled is True

"""Tests for surface_auto_patch_enhanced_11."""
from __future__ import annotations
import pytest
from pyfoam.tools.surface_auto_patch_enhanced_11 import SurfaceAutoPatchEnhanced11Result, CurvatureAdaptivePatchResult, MultiScaleSegmentationResult, surface_auto_patch_enhanced_11


class TestSurfaceAutoPatchEnhanced11Result:
    def test_returns_result(self):
        r = surface_auto_patch_enhanced_11()
        assert isinstance(r, SurfaceAutoPatchEnhanced11Result)

    def test_curvature_adaptive(self):
        r = surface_auto_patch_enhanced_11(enable_curvature_adaptive=True)
        assert isinstance(r.curvature_adaptive, CurvatureAdaptivePatchResult)
        assert r.curvature_adaptive.enabled is True

    def test_multi_scale(self):
        r = surface_auto_patch_enhanced_11(enable_multi_scale=True)
        assert isinstance(r.multi_scale, MultiScaleSegmentationResult)
        assert r.multi_scale.enabled is True

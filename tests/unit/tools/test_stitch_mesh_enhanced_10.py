"""Tests for stitch_mesh_enhanced_10."""
from __future__ import annotations
import pytest
from pyfoam.tools.stitch_mesh_enhanced_10 import StitchEnhanced10Result, NonConformalStitchResult, StitchQualityResult, stitch_mesh_enhanced_10


class TestStitchEnhanced10Result:
    def test_returns_result(self):
        r = stitch_mesh_enhanced_10()
        assert isinstance(r, StitchEnhanced10Result)

    def test_non_conformal(self):
        r = stitch_mesh_enhanced_10(enable_non_conformal=True)
        assert isinstance(r.non_conformal, NonConformalStitchResult)
        assert r.non_conformal.enabled is True

    def test_quality(self):
        r = stitch_mesh_enhanced_10(enable_quality=True)
        assert isinstance(r.quality, StitchQualityResult)
        assert r.quality.enabled is True

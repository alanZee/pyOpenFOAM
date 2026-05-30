"""Tests for stitch_mesh_enhanced_11."""
from __future__ import annotations
import pytest
from pyfoam.tools.stitch_mesh_enhanced_11 import StitchEnhanced11Result, AdaptiveStitchResult, MultiPatchStitchResult, stitch_mesh_enhanced_11


class TestStitchEnhanced11Result:
    def test_returns_result(self):
        r = stitch_mesh_enhanced_11()
        assert isinstance(r, StitchEnhanced11Result)

    def test_adaptive(self):
        r = stitch_mesh_enhanced_11(enable_adaptive=True)
        assert isinstance(r.adaptive, AdaptiveStitchResult)
        assert r.adaptive.enabled is True

    def test_multi_patch(self):
        r = stitch_mesh_enhanced_11(enable_multi_patch=True)
        assert isinstance(r.multi_patch, MultiPatchStitchResult)
        assert r.multi_patch.enabled is True

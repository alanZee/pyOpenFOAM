"""Tests for refine_mesh_enhanced_4."""
from __future__ import annotations
import pytest
from pyfoam.tools.refine_mesh_enhanced_4 import RefineEnhanced4Result, MultiLevelRefineResult, RefineQualityResult, refine_mesh_enhanced_4


class TestRefineEnhanced4Result:
    def test_returns_result(self):
        r = refine_mesh_enhanced_4()
        assert isinstance(r, RefineEnhanced4Result)

    def test_multi_level(self):
        r = refine_mesh_enhanced_4(enable_multi_level=True)
        assert isinstance(r.multi_level, MultiLevelRefineResult)
        assert r.multi_level.enabled is True

    def test_quality(self):
        r = refine_mesh_enhanced_4(enable_quality=True)
        assert isinstance(r.quality, RefineQualityResult)
        assert r.quality.enabled is True

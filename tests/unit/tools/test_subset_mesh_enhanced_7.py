"""Tests for subset_mesh_enhanced_7."""
from __future__ import annotations
import pytest
from pyfoam.tools.subset_mesh_enhanced_7 import SubsetEnhanced7Result, ZoneBasedSubsetResult, QualityPreservingResult, subset_mesh_enhanced_7


class TestSubsetEnhanced7Result:
    def test_returns_result(self):
        r = subset_mesh_enhanced_7()
        assert isinstance(r, SubsetEnhanced7Result)

    def test_zone_based(self):
        r = subset_mesh_enhanced_7(enable_zone_based=True)
        assert isinstance(r.zone_based, ZoneBasedSubsetResult)
        assert r.zone_based.enabled is True

    def test_quality(self):
        r = subset_mesh_enhanced_7(enable_quality=True)
        assert isinstance(r.quality, QualityPreservingResult)
        assert r.quality.enabled is True

"""Tests for subset_mesh_enhanced_5."""
from __future__ import annotations
import pytest
from pyfoam.tools.subset_mesh_enhanced_5 import SubsetEnhanced5Result, AdaptiveSubsetResult, SubsetStatisticsResult, NestedSubsetResult, subset_mesh_enhanced_5


class TestSubsetEnhanced5Result:
    def test_returns_result(self):
        r = subset_mesh_enhanced_5()
        assert isinstance(r, SubsetEnhanced5Result)

    def test_adaptive(self):
        r = subset_mesh_enhanced_5(enable_adaptive=True)
        assert isinstance(r.adaptive, AdaptiveSubsetResult)
        assert r.adaptive.enabled is True

    def test_statistics(self):
        r = subset_mesh_enhanced_5(enable_statistics=True)
        assert isinstance(r.statistics, SubsetStatisticsResult)
        assert r.statistics.enabled is True

    def test_nested(self):
        r = subset_mesh_enhanced_5(enable_nested=True)
        assert isinstance(r.nested, NestedSubsetResult)
        assert r.nested.enabled is True

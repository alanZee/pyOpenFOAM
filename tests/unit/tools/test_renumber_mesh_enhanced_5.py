"""Tests for renumber_mesh_enhanced_5."""
from __future__ import annotations
import pytest
from pyfoam.tools.renumber_mesh_enhanced_5 import RenumberEnhanced5Result, AdaptiveRenumberResult, CacheOptimizedResult, HierarchicalRenumberResult, renumber_mesh_enhanced_5


class TestRenumberEnhanced5Result:
    def test_returns_result(self):
        r = renumber_mesh_enhanced_5()
        assert isinstance(r, RenumberEnhanced5Result)

    def test_adaptive(self):
        r = renumber_mesh_enhanced_5(enable_adaptive=True)
        assert isinstance(r.adaptive, AdaptiveRenumberResult)
        assert r.adaptive.enabled is True

    def test_cache(self):
        r = renumber_mesh_enhanced_5(enable_cache=True)
        assert isinstance(r.cache, CacheOptimizedResult)
        assert r.cache.enabled is True

    def test_hierarchical(self):
        r = renumber_mesh_enhanced_5(enable_hierarchical=True)
        assert isinstance(r.hierarchical, HierarchicalRenumberResult)
        assert r.hierarchical.enabled is True

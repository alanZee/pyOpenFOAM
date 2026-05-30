"""Tests for merge_meshes_enhanced_10."""
from __future__ import annotations
import pytest
from pyfoam.tools.merge_meshes_enhanced_10 import MergeEnhanced10Result, HierarchicalMergeResult, MergeQualityResult, merge_meshes_enhanced_10


class TestMergeEnhanced10Result:
    def test_returns_result(self):
        r = merge_meshes_enhanced_10()
        assert isinstance(r, MergeEnhanced10Result)

    def test_hierarchical(self):
        r = merge_meshes_enhanced_10(enable_hierarchical=True)
        assert isinstance(r.hierarchical, HierarchicalMergeResult)
        assert r.hierarchical.enabled is True

    def test_quality_preservation(self):
        r = merge_meshes_enhanced_10(enable_quality_preservation=True)
        assert isinstance(r.quality_preservation, MergeQualityResult)
        assert r.quality_preservation.enabled is True

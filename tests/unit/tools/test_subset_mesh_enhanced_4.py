"""Tests for subset_mesh_enhanced_4."""
from __future__ import annotations
import pytest
from pyfoam.tools.subset_mesh_enhanced_4 import SubsetEnhanced4Result, MultiCriteriaSubsetResult, BoundaryPreservingSubsetResult, subset_mesh_enhanced_4


class TestSubsetEnhanced4Result:
    def test_returns_result(self):
        r = subset_mesh_enhanced_4()
        assert isinstance(r, SubsetEnhanced4Result)

    def test_multi_criteria(self):
        r = subset_mesh_enhanced_4(enable_multi_criteria=True)
        assert isinstance(r.multi_criteria, MultiCriteriaSubsetResult)
        assert r.multi_criteria.enabled is True

    def test_boundary_preserving(self):
        r = subset_mesh_enhanced_4(enable_boundary_preserving=True)
        assert isinstance(r.boundary_preserving, BoundaryPreservingSubsetResult)
        assert r.boundary_preserving.enabled is True

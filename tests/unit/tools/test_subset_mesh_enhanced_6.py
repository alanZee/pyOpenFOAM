"""Tests for subset_mesh_enhanced_6."""
from __future__ import annotations
import pytest
from pyfoam.tools.subset_mesh_enhanced_6 import SubsetEnhanced6Result, LayerAdditionResult, BLSubsetResult, subset_mesh_enhanced_6


class TestSubsetEnhanced6Result:
    def test_returns_result(self):
        r = subset_mesh_enhanced_6()
        assert isinstance(r, SubsetEnhanced6Result)

    def test_layer_add(self):
        r = subset_mesh_enhanced_6(enable_layer_add=True)
        assert isinstance(r.layer_add, LayerAdditionResult)
        assert r.layer_add.enabled is True

    def test_bl_subset(self):
        r = subset_mesh_enhanced_6(enable_bl_subset=True)
        assert isinstance(r.bl_subset, BLSubsetResult)
        assert r.bl_subset.enabled is True

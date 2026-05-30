"""Tests for subset_mesh_enhanced_3."""
from __future__ import annotations
import pytest
from pyfoam.tools.subset_mesh_enhanced_3 import SubsetEnhanced3Result, FieldThresholdSubsetResult, ConnectedRegionSubsetResult, subset_mesh_enhanced_3


class TestSubsetEnhanced3Result:
    def test_returns_result(self):
        r = subset_mesh_enhanced_3()
        assert isinstance(r, SubsetEnhanced3Result)

    def test_field_threshold(self):
        r = subset_mesh_enhanced_3(enable_field_threshold=True)
        assert isinstance(r.field_threshold, FieldThresholdSubsetResult)
        assert r.field_threshold.enabled is True

    def test_connected(self):
        r = subset_mesh_enhanced_3(enable_connected=True)
        assert isinstance(r.connected, ConnectedRegionSubsetResult)
        assert r.connected.enabled is True

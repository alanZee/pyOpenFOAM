"""Tests for subset_mesh_enhanced_9."""
from __future__ import annotations
import pytest
from pyfoam.tools.subset_mesh_enhanced_9 import SubsetEnhanced9Result, DynamicSubsetResult, SubsetCouplingResult, subset_mesh_enhanced_9


class TestSubsetEnhanced9Result:
    def test_returns_result(self):
        r = subset_mesh_enhanced_9()
        assert isinstance(r, SubsetEnhanced9Result)

    def test_dynamic(self):
        r = subset_mesh_enhanced_9(enable_dynamic=True)
        assert isinstance(r.dynamic, DynamicSubsetResult)
        assert r.dynamic.enabled is True

    def test_coupling(self):
        r = subset_mesh_enhanced_9(enable_coupling=True)
        assert isinstance(r.coupling, SubsetCouplingResult)
        assert r.coupling.enabled is True

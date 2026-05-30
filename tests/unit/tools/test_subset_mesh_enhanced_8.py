"""Tests for subset_mesh_enhanced_8."""
from __future__ import annotations
import pytest
from pyfoam.tools.subset_mesh_enhanced_8 import SubsetEnhanced8Result, HierarchicalSubsetResult, InterfaceManagementResult, subset_mesh_enhanced_8


class TestSubsetEnhanced8Result:
    def test_returns_result(self):
        r = subset_mesh_enhanced_8()
        assert isinstance(r, SubsetEnhanced8Result)

    def test_hierarchical(self):
        r = subset_mesh_enhanced_8(enable_hierarchical=True)
        assert isinstance(r.hierarchical, HierarchicalSubsetResult)
        assert r.hierarchical.enabled is True

    def test_interface(self):
        r = subset_mesh_enhanced_8(enable_interface=True)
        assert isinstance(r.interface, InterfaceManagementResult)
        assert r.interface.enabled is True

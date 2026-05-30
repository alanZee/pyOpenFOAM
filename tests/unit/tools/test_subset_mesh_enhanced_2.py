"""Tests for subset_mesh_enhanced_2."""
from __future__ import annotations
import pytest
from pyfoam.tools.subset_mesh_enhanced_2 import SubsetEnhanced2Result, SphereSubsetResult, PatchBasedSubsetResult, subset_mesh_enhanced_2


class TestSubsetEnhanced2Result:
    def test_returns_result(self):
        r = subset_mesh_enhanced_2()
        assert isinstance(r, SubsetEnhanced2Result)

    def test_sphere(self):
        r = subset_mesh_enhanced_2(enable_sphere=True)
        assert isinstance(r.sphere, SphereSubsetResult)
        assert r.sphere.enabled is True

    def test_patch_based(self):
        r = subset_mesh_enhanced_2(enable_patch_based=True)
        assert isinstance(r.patch_based, PatchBasedSubsetResult)
        assert r.patch_based.enabled is True

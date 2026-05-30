"""Tests for merge_meshes_enhanced_11."""
from __future__ import annotations
import pytest
from pyfoam.tools.merge_meshes_enhanced_11 import MergeEnhanced11Result, AdaptiveInterfaceResult, MultiResolutionMergeResult, merge_meshes_enhanced_11


class TestMergeEnhanced11Result:
    def test_returns_result(self):
        r = merge_meshes_enhanced_11()
        assert isinstance(r, MergeEnhanced11Result)

    def test_adaptive_interface(self):
        r = merge_meshes_enhanced_11(enable_adaptive_interface=True)
        assert isinstance(r.adaptive_interface, AdaptiveInterfaceResult)
        assert r.adaptive_interface.enabled is True

    def test_multi_resolution(self):
        r = merge_meshes_enhanced_11(enable_multi_resolution=True)
        assert isinstance(r.multi_resolution, MultiResolutionMergeResult)
        assert r.multi_resolution.enabled is True

"""Tests for create_patch_enhanced_12."""
from __future__ import annotations
import pytest
from pyfoam.tools.create_patch_enhanced_12 import PatchEnhanced12Result, AdaptivePatchResult, PatchTopologyResult, create_patch_enhanced_12


class TestPatchEnhanced12Result:
    def test_returns_result(self):
        r = create_patch_enhanced_12()
        assert isinstance(r, PatchEnhanced12Result)

    def test_adaptive_resolution(self):
        r = create_patch_enhanced_12(enable_adaptive_resolution=True)
        assert isinstance(r.adaptive_resolution, AdaptivePatchResult)
        assert r.adaptive_resolution.enabled is True

    def test_topology(self):
        r = create_patch_enhanced_12(enable_topology=True)
        assert isinstance(r.topology, PatchTopologyResult)
        assert r.topology.enabled is True

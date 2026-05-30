"""Tests for surface_auto_patch_enhanced_12."""
from __future__ import annotations
import pytest
from pyfoam.tools.surface_auto_patch_enhanced_12 import SurfaceAutoPatchEnhanced12Result, MLGuidedPatchResult, TopologyPreservingPatchResult, surface_auto_patch_enhanced_12


class TestSurfaceAutoPatchEnhanced12Result:
    def test_returns_result(self):
        r = surface_auto_patch_enhanced_12()
        assert isinstance(r, SurfaceAutoPatchEnhanced12Result)

    def test_ml_guided(self):
        r = surface_auto_patch_enhanced_12(enable_ml_guided=True)
        assert isinstance(r.ml_guided, MLGuidedPatchResult)
        assert r.ml_guided.enabled is True

    def test_topology_preserving(self):
        r = surface_auto_patch_enhanced_12(enable_topology_preserving=True)
        assert isinstance(r.topology_preserving, TopologyPreservingPatchResult)
        assert r.topology_preserving.enabled is True

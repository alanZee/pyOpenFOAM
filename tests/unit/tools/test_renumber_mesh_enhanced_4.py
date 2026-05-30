"""Tests for renumber_mesh_enhanced_4."""
from __future__ import annotations
import pytest
from pyfoam.tools.renumber_mesh_enhanced_4 import RenumberEnhanced4Result, HybridOrderingResult, BandwidthOptResult, renumber_mesh_enhanced_4


class TestRenumberEnhanced4Result:
    def test_returns_result(self):
        r = renumber_mesh_enhanced_4()
        assert isinstance(r, RenumberEnhanced4Result)

    def test_hybrid(self):
        r = renumber_mesh_enhanced_4(enable_hybrid=True)
        assert isinstance(r.hybrid, HybridOrderingResult)
        assert r.hybrid.enabled is True

    def test_bandwidth(self):
        r = renumber_mesh_enhanced_4(enable_bandwidth=True)
        assert isinstance(r.bandwidth, BandwidthOptResult)
        assert r.bandwidth.enabled is True

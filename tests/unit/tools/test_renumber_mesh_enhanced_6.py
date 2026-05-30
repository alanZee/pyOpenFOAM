"""Tests for renumber_mesh_enhanced_6."""
from __future__ import annotations
import pytest
from pyfoam.tools.renumber_mesh_enhanced_6 import RenumberEnhanced6Result, DualGraphOrderingResult, WavefrontReductionResult, renumber_mesh_enhanced_6


class TestRenumberEnhanced6Result:
    def test_returns_result(self):
        r = renumber_mesh_enhanced_6()
        assert isinstance(r, RenumberEnhanced6Result)

    def test_dual_graph(self):
        r = renumber_mesh_enhanced_6(enable_dual_graph=True)
        assert isinstance(r.dual_graph, DualGraphOrderingResult)
        assert r.dual_graph.enabled is True

    def test_wavefront(self):
        r = renumber_mesh_enhanced_6(enable_wavefront=True)
        assert isinstance(r.wavefront, WavefrontReductionResult)
        assert r.wavefront.enabled is True

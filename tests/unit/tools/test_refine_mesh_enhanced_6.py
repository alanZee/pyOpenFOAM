"""Tests for refine_mesh_enhanced_6."""
from __future__ import annotations
import pytest
from pyfoam.tools.refine_mesh_enhanced_6 import RefineEnhanced6Result, HexDominantRefineResult, TetRefineResult, refine_mesh_enhanced_6


class TestRefineEnhanced6Result:
    def test_returns_result(self):
        r = refine_mesh_enhanced_6()
        assert isinstance(r, RefineEnhanced6Result)

    def test_hex_dominant(self):
        r = refine_mesh_enhanced_6(enable_hex_dominant=True)
        assert isinstance(r.hex_dominant, HexDominantRefineResult)
        assert r.hex_dominant.enabled is True

    def test_tetrahedral(self):
        r = refine_mesh_enhanced_6(enable_tetrahedral=True)
        assert isinstance(r.tetrahedral, TetRefineResult)
        assert r.tetrahedral.enabled is True

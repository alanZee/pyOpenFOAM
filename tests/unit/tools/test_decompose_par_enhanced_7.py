"""Tests for decompose_par_enhanced_7."""
from __future__ import annotations
import pytest
from pyfoam.tools.decompose_par_enhanced_7 import DecomposeParEnhanced7Result, GhostCellResult, OverlapDecompResult, decompose_par_enhanced_7


class TestDecomposeParEnhanced7Result:
    def test_returns_result(self):
        r = decompose_par_enhanced_7()
        assert isinstance(r, DecomposeParEnhanced7Result)

    def test_ghost_cells(self):
        r = decompose_par_enhanced_7(enable_ghost_cells=True)
        assert isinstance(r.ghost_cells, GhostCellResult)
        assert r.ghost_cells.enabled is True

    def test_overlap(self):
        r = decompose_par_enhanced_7(enable_overlap=True)
        assert isinstance(r.overlap, OverlapDecompResult)
        assert r.overlap.enabled is True

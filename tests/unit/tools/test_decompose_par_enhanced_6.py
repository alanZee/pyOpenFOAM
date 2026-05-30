"""Tests for decompose_par_enhanced_6."""
from __future__ import annotations
import pytest
from pyfoam.tools.decompose_par_enhanced_6 import DecomposeParEnhanced6Result, MetisDecompResult, NeighbourDecompResult, decompose_par_enhanced_6


class TestDecomposeParEnhanced6Result:
    def test_returns_result(self):
        r = decompose_par_enhanced_6()
        assert isinstance(r, DecomposeParEnhanced6Result)

    def test_metis(self):
        r = decompose_par_enhanced_6(enable_metis=True)
        assert isinstance(r.metis, MetisDecompResult)
        assert r.metis.enabled is True

    def test_neighbour(self):
        r = decompose_par_enhanced_6(enable_neighbour=True)
        assert isinstance(r.neighbour, NeighbourDecompResult)
        assert r.neighbour.enabled is True

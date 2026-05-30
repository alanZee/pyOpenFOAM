"""Tests for decompose_par_enhanced_2."""
from __future__ import annotations
import pytest
from pyfoam.tools.decompose_par_enhanced_2 import DecomposeParEnhanced2Result, ScotchDecompResult, GraphPartitionResult, decompose_par_enhanced_2


class TestDecomposeParEnhanced2Result:
    def test_returns_result(self):
        r = decompose_par_enhanced_2()
        assert isinstance(r, DecomposeParEnhanced2Result)

    def test_scotch(self):
        r = decompose_par_enhanced_2(enable_scotch=True)
        assert isinstance(r.scotch, ScotchDecompResult)
        assert r.scotch.enabled is True

    def test_graph(self):
        r = decompose_par_enhanced_2(enable_graph=True)
        assert isinstance(r.graph, GraphPartitionResult)
        assert r.graph.enabled is True

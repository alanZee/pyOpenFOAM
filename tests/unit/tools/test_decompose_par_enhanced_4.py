"""Tests for decompose_par_enhanced_4."""
from __future__ import annotations
import pytest
from pyfoam.tools.decompose_par_enhanced_4 import DecomposeParEnhanced4Result, HierarchicalDecompResult, LoadBalancingResult, decompose_par_enhanced_4


class TestDecomposeParEnhanced4Result:
    def test_returns_result(self):
        r = decompose_par_enhanced_4()
        assert isinstance(r, DecomposeParEnhanced4Result)

    def test_hierarchical(self):
        r = decompose_par_enhanced_4(enable_hierarchical=True)
        assert isinstance(r.hierarchical, HierarchicalDecompResult)
        assert r.hierarchical.enabled is True

    def test_load_balance(self):
        r = decompose_par_enhanced_4(enable_load_balance=True)
        assert isinstance(r.load_balance, LoadBalancingResult)
        assert r.load_balance.enabled is True

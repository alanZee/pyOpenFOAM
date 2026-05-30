"""Tests for decompose_par_enhanced_3."""
from __future__ import annotations
import pytest
from pyfoam.tools.decompose_par_enhanced_3 import DecomposeParEnhanced3Result, MultiConstraintResult, WeightedDecompResult, decompose_par_enhanced_3


class TestDecomposeParEnhanced3Result:
    def test_returns_result(self):
        r = decompose_par_enhanced_3()
        assert isinstance(r, DecomposeParEnhanced3Result)

    def test_multi_constraint(self):
        r = decompose_par_enhanced_3(enable_multi_constraint=True)
        assert isinstance(r.multi_constraint, MultiConstraintResult)
        assert r.multi_constraint.enabled is True

    def test_weighted(self):
        r = decompose_par_enhanced_3(enable_weighted=True)
        assert isinstance(r.weighted, WeightedDecompResult)
        assert r.weighted.enabled is True

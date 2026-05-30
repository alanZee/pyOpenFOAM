"""Tests for decompose_par_enhanced_8."""
from __future__ import annotations
import pytest
from pyfoam.tools.decompose_par_enhanced_8 import DecomposeParEnhanced8Result, DynamicLoadBalanceResult, ProcessorAffinityResult, decompose_par_enhanced_8


class TestDecomposeParEnhanced8Result:
    def test_returns_result(self):
        r = decompose_par_enhanced_8()
        assert isinstance(r, DecomposeParEnhanced8Result)

    def test_dynamic(self):
        r = decompose_par_enhanced_8(enable_dynamic=True)
        assert isinstance(r.dynamic, DynamicLoadBalanceResult)
        assert r.dynamic.enabled is True

    def test_affinity(self):
        r = decompose_par_enhanced_8(enable_affinity=True)
        assert isinstance(r.affinity, ProcessorAffinityResult)
        assert r.affinity.enabled is True

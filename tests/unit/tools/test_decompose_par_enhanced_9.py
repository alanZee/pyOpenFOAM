"""Tests for decompose_par_enhanced_9."""
from __future__ import annotations
import pytest
from pyfoam.tools.decompose_par_enhanced_9 import DecomposeParEnhanced9Result, DecompVisualResult, CommMinResult, decompose_par_enhanced_9


class TestDecomposeParEnhanced9Result:
    def test_returns_result(self):
        r = decompose_par_enhanced_9()
        assert isinstance(r, DecomposeParEnhanced9Result)

    def test_visualisation(self):
        r = decompose_par_enhanced_9(enable_visualisation=True)
        assert isinstance(r.visualisation, DecompVisualResult)
        assert r.visualisation.enabled is True

    def test_comm_min(self):
        r = decompose_par_enhanced_9(enable_comm_min=True)
        assert isinstance(r.comm_min, CommMinResult)
        assert r.comm_min.enabled is True

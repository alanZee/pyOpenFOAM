"""Tests for refine_mesh_enhanced_8."""
from __future__ import annotations
import pytest
from pyfoam.tools.refine_mesh_enhanced_8 import RefineEnhanced8Result, ErrorEstimatorResult, RefinementHistoryResult, refine_mesh_enhanced_8


class TestRefineEnhanced8Result:
    def test_returns_result(self):
        r = refine_mesh_enhanced_8()
        assert isinstance(r, RefineEnhanced8Result)

    def test_error_estimator(self):
        r = refine_mesh_enhanced_8(enable_error_estimator=True)
        assert isinstance(r.error_estimator, ErrorEstimatorResult)
        assert r.error_estimator.enabled is True

    def test_history(self):
        r = refine_mesh_enhanced_8(enable_history=True)
        assert isinstance(r.history, RefinementHistoryResult)
        assert r.history.enabled is True

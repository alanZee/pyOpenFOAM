"""Tests for refine_mesh_enhanced_9."""
from __future__ import annotations
import pytest
from pyfoam.tools.refine_mesh_enhanced_9 import RefineEnhanced9Result, ParallelRefineResult, RefineConsistencyResult, refine_mesh_enhanced_9


class TestRefineEnhanced9Result:
    def test_returns_result(self):
        r = refine_mesh_enhanced_9()
        assert isinstance(r, RefineEnhanced9Result)

    def test_parallel(self):
        r = refine_mesh_enhanced_9(enable_parallel=True)
        assert isinstance(r.parallel, ParallelRefineResult)
        assert r.parallel.enabled is True

    def test_consistency(self):
        r = refine_mesh_enhanced_9(enable_consistency=True)
        assert isinstance(r.consistency, RefineConsistencyResult)
        assert r.consistency.enabled is True

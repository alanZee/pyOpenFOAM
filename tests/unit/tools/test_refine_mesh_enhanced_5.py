"""Tests for refine_mesh_enhanced_5."""
from __future__ import annotations
import pytest
from pyfoam.tools.refine_mesh_enhanced_5 import RefineEnhanced5Result, AdaptiveRefineResult, RefineStatisticsResult, CoarseningResult, refine_mesh_enhanced_5


class TestRefineEnhanced5Result:
    def test_returns_result(self):
        r = refine_mesh_enhanced_5()
        assert isinstance(r, RefineEnhanced5Result)

    def test_adaptive(self):
        r = refine_mesh_enhanced_5(enable_adaptive=True)
        assert isinstance(r.adaptive, AdaptiveRefineResult)
        assert r.adaptive.enabled is True

    def test_statistics(self):
        r = refine_mesh_enhanced_5(enable_statistics=True)
        assert isinstance(r.statistics, RefineStatisticsResult)
        assert r.statistics.enabled is True

    def test_coarsening(self):
        r = refine_mesh_enhanced_5(enable_coarsening=True)
        assert isinstance(r.coarsening, CoarseningResult)
        assert r.coarsening.enabled is True

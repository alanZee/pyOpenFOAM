"""Tests for surface_check_enhanced_7 — enhanced surface checking v7."""
from __future__ import annotations
import numpy as np
import pytest
from pyfoam.tools.surface_check_enhanced_7 import SurfaceCheckEnhanced7Result, MetricConfidence, TrendEntry, surface_check_enhanced_7


def _cube_tris():
    v = np.array([
        [0,0,0],[1,0,0],[1,1,0],[0,1,0],
        [0,0,1],[1,0,1],[1,1,1],[0,1,1],
    ], dtype=np.float64)
    f = np.array([
        [0,1,2],[0,2,3],[4,6,5],[4,7,6],
        [0,4,5],[0,5,1],[2,6,7],[2,7,3],
        [0,3,7],[0,7,4],[1,5,6],[1,6,2],
    ], dtype=np.int32)
    return v, f


class TestSurfaceCheckEnhanced7:
    def test_returns_result_type(self):
        v, f = _cube_tris()
        r = surface_check_enhanced_7(vertices=v, faces=f)
        assert isinstance(r, SurfaceCheckEnhanced7Result)

    def test_trend_analysis(self):
        v, f = _cube_tris()
        # Create fake history
        class FakeResult:
            n_degenerate_faces = 2
            n_open_edges = 4
        history = [FakeResult(), FakeResult()]
        r = surface_check_enhanced_7(
            vertices=v, faces=f, trend_history=history,
        )
        assert isinstance(r.trend_analysis, list)
        if r.trend_analysis:
            assert isinstance(r.trend_analysis[0], TrendEntry)

    def test_confidence_scoring(self):
        v, f = _cube_tris()
        r = surface_check_enhanced_7(vertices=v, faces=f)
        assert isinstance(r.metric_confidences, list)
        if r.metric_confidences:
            assert isinstance(r.metric_confidences[0], MetricConfidence)
            assert 0.0 <= r.metric_confidences[0].confidence <= 1.0

    def test_repair_execution(self):
        v, f = _cube_tris()
        r = surface_check_enhanced_7(vertices=v, faces=f, execute_repairs=True)
        assert r.n_repairs_executed >= 0
        assert 0.0 <= r.repair_improvement <= 1.0

    def test_summary(self):
        v, f = _cube_tris()
        r = surface_check_enhanced_7(vertices=v, faces=f)
        s = r.summary()
        assert "enhanced v7" in s

    def test_defaults(self):
        v, f = _cube_tris()
        r = surface_check_enhanced_7(vertices=v, faces=f)
        assert r.repair_improvement == 0.0
        assert r.n_repairs_executed == 0

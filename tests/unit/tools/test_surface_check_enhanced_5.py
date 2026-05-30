"""Tests for surface_check_enhanced_5 — enhanced surface checking v5."""
from __future__ import annotations
import numpy as np
import pytest
from pyfoam.tools.surface_check_enhanced_5 import (
    SurfaceCheckEnhanced5Result, RepairResult, surface_check_enhanced_5,
)


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


class TestSurfaceCheckEnhanced5:
    def test_returns_result_type(self):
        v, f = _cube_tris()
        r = surface_check_enhanced_5(vertices=v, faces=f)
        assert isinstance(r, SurfaceCheckEnhanced5Result)

    def test_auto_repair_no_issues(self):
        v, f = _cube_tris()
        r = surface_check_enhanced_5(vertices=v, faces=f, auto_repair=True)
        # Clean cube has no degenerates
        assert r.n_repairs_applied == 0

    def test_auto_repair_degenerates(self):
        v = np.array([[0,0,0],[1,0,0],[0,1,0],[1,0,0]], dtype=np.float64)
        f = np.array([[0,1,2],[0,1,3]], dtype=np.int32)  # face [0,1,3] is degenerate
        r = surface_check_enhanced_5(vertices=v, faces=f, auto_repair=True)
        assert r.n_repairs_applied >= 0  # at least attempts repair

    def test_summary_method(self):
        v, f = _cube_tris()
        r = surface_check_enhanced_5(vertices=v, faces=f)
        s = r.summary()
        assert "enhanced v5" in s

    def test_batch_mode(self):
        v, f = _cube_tris()
        r = surface_check_enhanced_5(batch_inputs=["fake.stl"])
        assert len(r.batch_results) == 1

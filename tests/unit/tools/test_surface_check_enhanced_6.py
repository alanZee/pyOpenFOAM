"""Tests for surface_check_enhanced_6 — enhanced surface checking v6."""
from __future__ import annotations
import numpy as np
import pytest
from pyfoam.tools.surface_check_enhanced_6 import (
    SurfaceCheckEnhanced6Result, DifferentialReport, RepairPriority,
    surface_check_enhanced_6,
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


class TestSurfaceCheckEnhanced6:
    def test_returns_result_type(self):
        r = surface_check_enhanced_6(vertices=np.zeros((3, 3)), faces=np.array([[0, 1, 2]]))
        assert isinstance(r, SurfaceCheckEnhanced6Result)

    def test_differential_check(self):
        v, f = _cube_tris()
        prev = surface_check_enhanced_6(vertices=v, faces=f)
        curr = surface_check_enhanced_6(vertices=v, faces=f, previous_result=prev)
        assert isinstance(curr.differential, DifferentialReport)
        assert curr.differential.n_new_degenerates == 0

    def test_generate_report(self):
        v, f = _cube_tris()
        r = surface_check_enhanced_6(
            vertices=v, faces=f, generate_report=True,
        )
        assert r.report_text is not None
        assert "surfaceCheckReport" in r.report_text

    def test_prioritize_repairs(self):
        v, f = _cube_tris()
        r = surface_check_enhanced_6(
            vertices=v, faces=f, prioritize_repairs=True,
        )
        assert isinstance(r.repair_priorities, list)
        assert r.n_prioritised_repairs == len(r.repair_priorities)

    def test_summary(self):
        v, f = _cube_tris()
        r = surface_check_enhanced_6(vertices=v, faces=f)
        s = r.summary()
        assert "enhanced v6" in s

    def test_no_differential(self):
        v, f = _cube_tris()
        r = surface_check_enhanced_6(vertices=v, faces=f)
        assert r.differential is None

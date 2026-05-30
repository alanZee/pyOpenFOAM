"""Tests for surface_check_enhanced_4 — enhanced surface checking v4."""
from __future__ import annotations
import numpy as np
import pytest
from pyfoam.tools.surface_check_enhanced_4 import (
    SurfaceCheckEnhanced4Result,
    surface_check_enhanced_4,
)


def _cube_tris():
    v = np.array([
        [0,0,0],[1,0,0],[1,1,0],[0,1,0],
        [0,0,1],[1,0,1],[1,1,1],[0,1,1],
    ], dtype=np.float64)
    f = np.array([
        [0,3,2],[0,2,1],[4,5,6],[4,6,7],
        [0,1,5],[0,5,4],[2,3,7],[2,7,6],
        [0,4,7],[0,7,3],[1,2,6],[1,6,5],
    ], dtype=np.int32)
    return v, f


class TestSurfaceCheckEnhanced4:
    def test_returns_result_type(self):
        v, f = _cube_tris()
        r = surface_check_enhanced_4(vertices=v, faces=f)
        assert isinstance(r, SurfaceCheckEnhanced4Result)

    def test_overall_grade(self):
        v, f = _cube_tris()
        r = surface_check_enhanced_4(vertices=v, faces=f)
        assert r.overall_grade in ("A", "B", "C", "D", "F")

    def test_heal_actions(self):
        v, f = _cube_tris()
        r = surface_check_enhanced_4(vertices=v, faces=f)
        assert isinstance(r.heal_actions, list)

    def test_watertight_cube_grade(self):
        v, f = _cube_tris()
        r = surface_check_enhanced_4(vertices=v, faces=f)
        # Cube is watertight, so grade should be good
        assert r.is_watertight is True
        assert r.overall_grade in ("A", "B")

    def test_summary_string(self):
        v, f = _cube_tris()
        r = surface_check_enhanced_4(vertices=v, faces=f)
        s = r.summary()
        assert "Overall grade" in s

    def test_empty_surface(self):
        v = np.array([[0,0,0]], dtype=np.float64)
        f = np.empty((0, 3), dtype=np.int32)
        r = surface_check_enhanced_4(vertices=v, faces=f)
        assert r.overall_grade == "F"

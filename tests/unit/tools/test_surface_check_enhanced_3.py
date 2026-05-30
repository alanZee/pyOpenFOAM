"""Tests for surface_check_enhanced_3 — enhanced surface checking v3."""
from __future__ import annotations
import numpy as np
import pytest
from pyfoam.tools.surface_check_enhanced_3 import (
    SurfaceCheckEnhanced3Result,
    surface_check_enhanced_3,
)


def _simple_triangles():
    verts = np.array([
        [0, 0, 0], [1, 0, 0], [0.5, 0.5, 0], [0.5, 0.5, 1.0],
    ], dtype=np.float64)
    faces = np.array([[0, 1, 2], [0, 1, 3]], dtype=np.int32)
    return verts, faces


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


class TestSurfaceCheckEnhanced3:
    def test_returns_result_type(self):
        v, f = _simple_triangles()
        r = surface_check_enhanced_3(vertices=v, faces=f)
        assert isinstance(r, SurfaceCheckEnhanced3Result)

    def test_radius_ratio_populated(self):
        v, f = _simple_triangles()
        r = surface_check_enhanced_3(vertices=v, faces=f)
        assert r.face_radius_ratios.shape == (2,)
        assert r.mean_radius_ratio > 0

    def test_condition_number_populated(self):
        v, f = _simple_triangles()
        r = surface_check_enhanced_3(vertices=v, faces=f)
        assert r.face_condition_numbers.shape == (2,)
        assert r.mean_condition_number >= 1.0

    def test_summary_string(self):
        v, f = _simple_triangles()
        r = surface_check_enhanced_3(vertices=v, faces=f)
        s = r.summary()
        assert "enhanced v3" in s
        assert "radius ratio" in s

    def test_cube_check(self):
        v, f = _cube_tris()
        r = surface_check_enhanced_3(vertices=v, faces=f)
        assert r.n_faces == 12
        assert r.is_watertight

    def test_quality_grades(self):
        v, f = _simple_triangles()
        r = surface_check_enhanced_3(vertices=v, faces=f)
        assert "A" in r.face_grades
        total = sum(r.face_grades.values())
        assert total == 2

    def test_empty_surface(self):
        v = np.array([[0,0,0]], dtype=np.float64)
        f = np.empty((0, 3), dtype=np.int32)
        r = surface_check_enhanced_3(vertices=v, faces=f)
        assert r.n_faces == 0
        assert len(r.warnings) > 0

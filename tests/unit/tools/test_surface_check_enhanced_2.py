"""Tests for surface_check_enhanced_2 — enhanced surface checking v2."""
from __future__ import annotations
import numpy as np
import pytest
from pyfoam.tools.surface_check_enhanced_2 import (
    SurfaceCheckEnhanced2Result,
    surface_check_enhanced_2,
)


def _simple_triangles():
    verts = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0.5, 0.866, 0],
        [0.5, 0.289, 0.816],
    ], dtype=np.float64)
    faces = np.array([[0, 1, 2], [0, 1, 3], [1, 2, 3], [0, 2, 3]], dtype=np.int32)
    return verts, faces


def _open_surface():
    """Two triangles sharing an edge (open surface)."""
    verts = np.array([
        [0, 0, 0], [1, 0, 0], [0.5, 0.5, 0], [0.5, 0.5, 1.0],
    ], dtype=np.float64)
    faces = np.array([[0, 1, 2], [0, 1, 3]], dtype=np.int32)
    return verts, faces


class TestSurfaceCheckEnhanced2:
    def test_returns_result_type(self):
        v, f = _simple_triangles()
        r = surface_check_enhanced_2(vertices=v, faces=f)
        assert isinstance(r, SurfaceCheckEnhanced2Result)

    def test_n_faces(self):
        v, f = _simple_triangles()
        r = surface_check_enhanced_2(vertices=v, faces=f)
        assert r.n_faces == 4

    def test_euler_characteristic(self):
        v, f = _simple_triangles()
        r = surface_check_enhanced_2(vertices=v, faces=f)
        assert isinstance(r.euler_characteristic, int)

    def test_connected_components(self):
        v, f = _simple_triangles()
        r = surface_check_enhanced_2(vertices=v, faces=f)
        assert r.n_connected_components >= 1

    def test_face_grades(self):
        v, f = _simple_triangles()
        r = surface_check_enhanced_2(vertices=v, faces=f)
        assert "A" in r.face_grades
        total = sum(r.face_grades.values())
        assert total == 4

    def test_min_max_angles(self):
        v, f = _simple_triangles()
        r = surface_check_enhanced_2(vertices=v, faces=f)
        assert r.min_angle_mean > 0
        assert r.max_angle_mean > 0

    def test_open_surface_not_watertight(self):
        v, f = _open_surface()
        r = surface_check_enhanced_2(vertices=v, faces=f)
        assert not r.is_watertight

    def test_summary(self):
        v, f = _simple_triangles()
        r = surface_check_enhanced_2(vertices=v, faces=f)
        s = r.summary()
        assert "Surface check" in s

    def test_empty_surface(self):
        v = np.array([[0, 0, 0]], dtype=np.float64)
        f = np.empty((0, 3), dtype=np.int32)
        r = surface_check_enhanced_2(vertices=v, faces=f)
        assert r.n_faces == 0

    def test_quality_thresholds(self):
        v, f = _simple_triangles()
        r = surface_check_enhanced_2(vertices=v, faces=f,
                                      quality_thresholds={"ar_A": 1.1, "ar_B": 2.0})
        assert "A" in r.face_grades

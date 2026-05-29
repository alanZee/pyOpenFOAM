"""Tests for surface_check_enhanced — enhanced surface checking."""
from __future__ import annotations
import numpy as np
import pytest
from pyfoam.tools.surface_check_enhanced import (
    SurfaceCheckEnhancedResult,
    surface_check_enhanced,
)


def _unit_cube_triangles():
    pts = np.array([
        [0,0,0],[1,0,0],[1,1,0],[0,1,0],
        [0,0,1],[1,0,1],[1,1,1],[0,1,1],
    ], dtype=np.float64)
    tris = np.array([
        [0,1,2],[0,2,3],[4,6,5],[4,7,6],
        [0,5,1],[0,4,5],[2,7,3],[2,6,7],
        [0,3,7],[0,7,4],[1,5,6],[1,6,2],
    ], dtype=np.int32)
    return pts, tris


def _single_triangle():
    pts = np.array([[0,0,0],[1,0,0],[0,1,0]], dtype=np.float64)
    tris = np.array([[0,1,2]], dtype=np.int32)
    return pts, tris


def _degenerate_triangle():
    pts = np.array([[0,0,0],[1,0,0],[0.5,0,0]], dtype=np.float64)
    tris = np.array([[0,1,2]], dtype=np.int32)
    return pts, tris


class TestSurfaceCheckEnhanced:
    def test_returns_result_type(self):
        pts, tris = _unit_cube_triangles()
        r = surface_check_enhanced("", vertices=pts, faces=tris)
        assert isinstance(r, SurfaceCheckEnhancedResult)

    def test_cube_metrics(self):
        pts, tris = _unit_cube_triangles()
        r = surface_check_enhanced("", vertices=pts, faces=tris)
        assert r.n_points == 8
        assert r.n_faces == 12
        assert r.total_area > 0

    def test_face_areas_array(self):
        pts, tris = _unit_cube_triangles()
        r = surface_check_enhanced("", vertices=pts, faces=tris)
        assert r.face_areas.shape == (12,)
        assert np.all(r.face_areas > 0)

    def test_aspect_ratios(self):
        pts, tris = _unit_cube_triangles()
        r = surface_check_enhanced("", vertices=pts, faces=tris)
        assert r.face_aspect_ratios.shape == (12,)
        assert r.mean_aspect_ratio > 0
        assert r.max_aspect_ratio >= r.mean_aspect_ratio

    def test_skewness(self):
        pts, tris = _unit_cube_triangles()
        r = surface_check_enhanced("", vertices=pts, faces=tris)
        assert r.face_skewness.shape == (12,)
        assert 0 <= r.mean_skewness <= 1

    def test_degenerate_face_detection(self):
        pts, tris = _degenerate_triangle()
        r = surface_check_enhanced("", vertices=pts, faces=tris)
        assert r.n_degenerate_faces == 1
        assert 0 in r.degenerate_face_indices

    def test_single_triangle_open_edges(self):
        pts, tris = _single_triangle()
        r = surface_check_enhanced("", vertices=pts, faces=tris)
        assert r.n_open_edges == 3
        assert not r.is_watertight

    def test_summary_string(self):
        pts, tris = _unit_cube_triangles()
        r = surface_check_enhanced("", vertices=pts, faces=tris)
        s = r.summary()
        assert "enhanced" in s
        assert "aspect ratio" in s.lower() or "Aspect" in s

    def test_bbox(self):
        pts, tris = _unit_cube_triangles()
        r = surface_check_enhanced("", vertices=pts, faces=tris)
        assert np.allclose(r.bbox_min, [0, 0, 0])
        assert np.allclose(r.bbox_max, [1, 1, 1])

    def test_empty_surface(self):
        pts = np.array([[0,0,0]], dtype=np.float64)
        tris = np.empty((0, 3), dtype=np.int32)
        r = surface_check_enhanced("", vertices=pts, faces=tris)
        assert r.n_faces == 0

    def test_nonexistent_file_raises(self):
        with pytest.raises(FileNotFoundError):
            surface_check_enhanced("/nonexistent/path.stl")

    def test_warnings_populated(self):
        pts, tris = _single_triangle()
        r = surface_check_enhanced("", vertices=pts, faces=tris)
        assert len(r.warnings) > 0

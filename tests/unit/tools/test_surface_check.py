"""Tests for surface_check — surface quality checker."""
from __future__ import annotations
import numpy as np
import pytest
from pyfoam.tools.surface_check import surface_check, SurfaceCheckResult


def _unit_cube_triangles():
    """Watertight unit cube as 12 triangles."""
    pts = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
    ], dtype=np.float64)
    tris = np.array([
        [0, 1, 2], [0, 2, 3],
        [4, 6, 5], [4, 7, 6],
        [0, 5, 1], [0, 4, 5],
        [2, 7, 3], [2, 6, 7],
        [0, 3, 7], [0, 7, 4],
        [1, 5, 6], [1, 6, 2],
    ], dtype=np.int32)
    return pts, tris


def _single_triangle():
    """Single triangle — open surface."""
    pts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
    tris = np.array([[0, 1, 2]], dtype=np.int32)
    return pts, tris


def _degenerate_triangle():
    """Triangle with zero area (collinear points)."""
    pts = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float64)
    tris = np.array([[0, 1, 2]], dtype=np.int32)
    return pts, tris


class TestSurfaceCheck:
    def test_watertight_cube(self):
        """Cube should be watertight."""
        pts, tris = _unit_cube_triangles()
        result = surface_check(vertices=pts, faces=tris)
        assert isinstance(result, SurfaceCheckResult)
        assert result.is_watertight is True
        assert result.n_open_edges == 0
        assert result.n_non_manifold_edges == 0

    def test_single_triangle_not_watertight(self):
        """Single triangle has 3 open edges."""
        pts, tris = _single_triangle()
        result = surface_check(vertices=pts, faces=tris)
        assert result.is_watertight is False
        assert result.n_open_edges == 3

    def test_face_count(self):
        """Face count should match input."""
        pts, tris = _unit_cube_triangles()
        result = surface_check(vertices=pts, faces=tris)
        assert result.n_faces == 12

    def test_point_count(self):
        """Point count should match input."""
        pts, tris = _unit_cube_triangles()
        result = surface_check(vertices=pts, faces=tris)
        assert result.n_points == 8

    def test_bounding_box(self):
        """Bounding box should match the unit cube."""
        pts, tris = _unit_cube_triangles()
        result = surface_check(vertices=pts, faces=tris)
        assert np.allclose(result.bbox_min, [0, 0, 0])
        assert np.allclose(result.bbox_max, [1, 1, 1])

    def test_total_area_cube(self):
        """Unit cube total surface area should be 6.0."""
        pts, tris = _unit_cube_triangles()
        result = surface_check(vertices=pts, faces=tris)
        assert abs(result.total_area - 6.0) < 1e-10

    def test_degenerate_face_detected(self):
        """Collinear triangle should be flagged as degenerate."""
        pts, tris = _degenerate_triangle()
        result = surface_check(vertices=pts, faces=tris)
        assert result.n_degenerate_faces == 1

    def test_summary(self):
        """Summary should be a non-empty string."""
        pts, tris = _unit_cube_triangles()
        result = surface_check(vertices=pts, faces=tris)
        s = result.summary()
        assert isinstance(s, str)
        assert len(s) > 0
        assert "watertight" in s.lower() or "Watertight" in s

    def test_empty_faces(self):
        """Surface with no faces should return early with warning."""
        pts = np.array([[0, 0, 0]], dtype=np.float64)
        tris = np.empty((0, 3), dtype=np.int32)
        result = surface_check(vertices=pts, faces=tris)
        assert result.n_faces == 0
        assert len(result.warnings) > 0

    def test_nonexistent_file_raises(self):
        """Should raise FileNotFoundError for non-existent file."""
        with pytest.raises(FileNotFoundError):
            surface_check("/nonexistent/path.stl")

    def test_edge_count_cube(self):
        """Cube should have 18 unique edges (12 face edges + ...)."""
        # A cube has 12 geometric edges. With 12 triangles, each face is 2 tris
        # sharing 1 diagonal, so 5 edges per face * 6 faces = 30, minus shared.
        # Actually: 12 boundary edges + 6 face diagonals = 18 unique edges.
        pts, tris = _unit_cube_triangles()
        result = surface_check(vertices=pts, faces=tris)
        assert result.n_edges == 18

    def test_warnings_populated(self):
        """Open surface should produce warnings."""
        pts, tris = _single_triangle()
        result = surface_check(vertices=pts, faces=tris)
        assert any("watertight" in w.lower() or "open" in w.lower() for w in result.warnings)

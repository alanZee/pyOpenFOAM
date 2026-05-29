"""Tests for surface_mesh_info — surface mesh statistics."""

from __future__ import annotations

import numpy as np
import pytest

from pyfoam.tools.surface_mesh_info import surface_mesh_info, SurfaceMeshInfo


def _unit_cube_triangles():
    """Create a unit cube as 12 triangles."""
    pts = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
    ], dtype=np.float64)
    tris = np.array([
        [0, 1, 2], [0, 2, 3],   # bottom
        [4, 6, 5], [4, 7, 6],   # top
        [0, 5, 1], [0, 4, 5],   # front
        [2, 7, 3], [2, 6, 7],   # back
        [0, 3, 7], [0, 7, 4],   # left
        [1, 5, 6], [1, 6, 2],   # right
    ], dtype=np.int32)
    return pts, tris


def _tetrahedron():
    """Create a single tetrahedron."""
    pts = np.array([
        [0, 0, 0], [1, 0, 0], [0.5, 1, 0], [0.5, 0.5, 1],
    ], dtype=np.float64)
    tris = np.array([
        [0, 1, 2], [0, 1, 3], [1, 2, 3], [0, 2, 3],
    ], dtype=np.int32)
    return pts, tris


class TestSurfaceMeshInfo:
    """Test the surface_mesh_info function."""

    def test_cube_basic_counts(self):
        """Cube should have correct point, face, edge counts."""
        pts, tris = _unit_cube_triangles()
        info = surface_mesh_info("", vertices=pts, faces=tris)
        assert isinstance(info, SurfaceMeshInfo)
        assert info.n_points == 8
        assert info.n_faces == 12
        assert info.n_edges > 0

    def test_cube_watertight(self):
        """Closed cube should be watertight and manifold."""
        pts, tris = _unit_cube_triangles()
        info = surface_mesh_info("", vertices=pts, faces=tris)
        assert info.is_watertight
        assert info.is_manifold
        assert info.n_open_edges == 0
        assert info.n_non_manifold_edges == 0

    def test_cube_bounding_box(self):
        """Bounding box should span [0,0,0] to [1,1,1]."""
        pts, tris = _unit_cube_triangles()
        info = surface_mesh_info("", vertices=pts, faces=tris)
        np.testing.assert_allclose(info.bbox_min, [0, 0, 0])
        np.testing.assert_allclose(info.bbox_max, [1, 1, 1])
        np.testing.assert_allclose(info.bbox_size, [1, 1, 1])

    def test_cube_total_area(self):
        """Unit cube total area should be 6 (6 faces of area 1)."""
        pts, tris = _unit_cube_triangles()
        info = surface_mesh_info("", vertices=pts, faces=tris)
        np.testing.assert_allclose(info.total_area, 6.0, rtol=1e-10)

    def test_cube_face_area_range(self):
        """All cube face areas should be 0.5 (each triangle is half a square)."""
        pts, tris = _unit_cube_triangles()
        info = surface_mesh_info("", vertices=pts, faces=tris)
        np.testing.assert_allclose(info.min_face_area, 0.5, rtol=1e-10)
        np.testing.assert_allclose(info.max_face_area, 0.5, rtol=1e-10)
        np.testing.assert_allclose(info.mean_face_area, 0.5, rtol=1e-10)

    def test_cube_edge_length_range(self):
        """Unit cube edges should all be length 1 or sqrt(2)."""
        pts, tris = _unit_cube_triangles()
        info = surface_mesh_info("", vertices=pts, faces=tris)
        assert info.min_edge_length == pytest.approx(1.0, rel=1e-10)
        assert info.max_edge_length == pytest.approx(np.sqrt(2.0), rel=1e-10)

    def test_cube_genus(self):
        """Cube genus should be 0 (topologically a sphere)."""
        pts, tris = _unit_cube_triangles()
        info = surface_mesh_info("", vertices=pts, faces=tris)
        assert info.genus == 0

    def test_tetrahedron(self):
        """Tetrahedron should have 4 vertices, 4 faces, 6 edges."""
        pts, tris = _tetrahedron()
        info = surface_mesh_info("", vertices=pts, faces=tris)
        assert info.n_points == 4
        assert info.n_faces == 4
        assert info.n_edges == 6
        assert info.is_watertight

    def test_single_triangle_not_watertight(self):
        """Single triangle should have open edges."""
        pts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
        tris = np.array([[0, 1, 2]], dtype=np.int32)
        info = surface_mesh_info("", vertices=pts, faces=tris)
        assert not info.is_watertight
        assert info.n_open_edges == 3

    def test_summary_method(self):
        """summary() should return a non-empty string."""
        pts, tris = _unit_cube_triangles()
        info = surface_mesh_info("", vertices=pts, faces=tris)
        text = info.summary()
        assert "8 points" in text
        assert "12 faces" in text
        assert "Watertight: True" in text

    def test_nonexistent_file_raises(self):
        """Should raise FileNotFoundError for non-existent path."""
        with pytest.raises(FileNotFoundError):
            surface_mesh_info("/nonexistent/path.stl")

    def test_empty_faces(self):
        """Empty faces array should return basic info."""
        pts = np.array([[0, 0, 0]], dtype=np.float64)
        tris = np.empty((0, 3), dtype=np.int32)
        info = surface_mesh_info("", vertices=pts, faces=tris)
        assert info.n_faces == 0
        assert info.n_edges == 0

    def test_open_surface_valence(self):
        """Valence should be computed for all vertices."""
        pts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
        tris = np.array([[0, 1, 2]], dtype=np.int32)
        info = surface_mesh_info("", vertices=pts, faces=tris)
        # Each vertex of a triangle shares 2 edges
        assert info.min_valence >= 1
        assert info.max_valence >= info.min_valence

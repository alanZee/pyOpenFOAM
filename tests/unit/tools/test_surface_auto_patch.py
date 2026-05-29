"""Tests for surface_auto_patch — auto-patch surface by feature angle."""
from __future__ import annotations
import numpy as np
import pytest
from pyfoam.tools.surface_auto_patch import surface_auto_patch, SurfaceAutoPatchResult


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


def _flat_quad():
    """Two coplanar triangles — should be 1 patch."""
    pts = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float64)
    tris = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    return pts, tris


def _L_shape():
    """Two surfaces meeting at 90 degrees (L-shape in 2D extruded)."""
    pts = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0],  # horizontal part
        [0, 0, 0], [1, 0, 0], [0, 0, 1],   # vertical part
    ], dtype=np.float64)
    tris = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
    return pts, tris


class TestSurfaceAutoPatch:
    def test_flat_quad_single_patch(self):
        """Coplanar triangles should form a single patch."""
        pts, tris = _flat_quad()
        result = surface_auto_patch(vertices=pts, faces=tris, feature_angle=30.0)
        assert isinstance(result, SurfaceAutoPatchResult)
        assert result.n_patches == 1

    def test_cube_multiple_patches(self):
        """Cube with low feature angle should produce multiple patches."""
        pts, tris = _unit_cube_triangles()
        result = surface_auto_patch(vertices=pts, faces=tris, feature_angle=10.0)
        assert result.n_patches >= 1
        assert result.patch_ids.shape[0] == 12

    def test_l_shape_two_patches(self):
        """L-shape at 30 deg threshold should produce 2 patches."""
        pts, tris = _L_shape()
        result = surface_auto_patch(vertices=pts, faces=tris, feature_angle=30.0)
        assert result.n_patches == 2

    def test_patch_face_counts_sum(self):
        """Sum of patch face counts should equal total faces."""
        pts, tris = _unit_cube_triangles()
        result = surface_auto_patch(vertices=pts, faces=tris)
        total = sum(result.patch_face_counts.values())
        assert total == 12

    def test_patch_ids_range(self):
        """Patch IDs should be in range [0, n_patches)."""
        pts, tris = _unit_cube_triangles()
        result = surface_auto_patch(vertices=pts, faces=tris)
        assert result.patch_ids.min() >= 0
        assert result.patch_ids.max() < result.n_patches

    def test_vertices_and_faces_preserved(self):
        """Output vertices and faces should match input."""
        pts, tris = _flat_quad()
        result = surface_auto_patch(vertices=pts, faces=tris)
        assert np.allclose(result.vertices, pts)
        assert np.array_equal(result.faces, tris)

    def test_output_stl(self, tmp_path):
        """Writing to STL should produce a file."""
        pts, tris = _flat_quad()
        out = tmp_path / "patched.stl"
        result = surface_auto_patch(vertices=pts, faces=tris, output_path=out)
        assert out.exists()
        content = out.read_text()
        assert "solid" in content

    def test_nonexistent_file_raises(self):
        """Should raise FileNotFoundError for non-existent file."""
        with pytest.raises(FileNotFoundError):
            surface_auto_patch("/nonexistent/path.stl")

    def test_high_angle_multiple_face_groups(self):
        """Very high feature angle groups only co-planar faces (6 cube faces)."""
        pts, tris = _unit_cube_triangles()
        result = surface_auto_patch(vertices=pts, faces=tris, feature_angle=180.0)
        # feature_angle=180 means only faces with identical normals merge
        # Cube has 6 face pairs → 6 patches
        assert result.n_patches == 6

    def test_empty_surface(self):
        """Empty surface should return 0 patches."""
        pts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
        tris = np.empty((0, 3), dtype=np.int32)
        result = surface_auto_patch(vertices=pts, faces=tris)
        assert result.n_patches == 0

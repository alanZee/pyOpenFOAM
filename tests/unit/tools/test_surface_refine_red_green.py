"""Tests for surface_refine_red_green — surface mesh refinement."""

from __future__ import annotations

import numpy as np
import pytest

from pyfoam.tools.surface_refine_red_green import surface_refine, RefineResult


def _single_triangle():
    """A single equilateral-ish triangle."""
    pts = np.array([[0, 0, 0], [1, 0, 0], [0.5, 0.866, 0]], dtype=np.float64)
    tris = np.array([[0, 1, 2]], dtype=np.int32)
    return pts, tris


def _two_triangles():
    """Two triangles sharing an edge (a quad split diagonally)."""
    pts = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
    ], dtype=np.float64)
    tris = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    return pts, tris


def _unit_cube():
    """Unit cube as 12 triangles."""
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


class TestSurfaceRefine:
    """Test the surface_refine function."""

    def test_single_triangle_red_refine(self):
        """Red refinement of 1 triangle should produce 4 triangles."""
        pts, tris = _single_triangle()
        result = surface_refine(pts, tris, levels=1)
        assert isinstance(result, RefineResult)
        assert result.n_output_faces == 4
        assert result.n_red_splits == 1
        assert result.n_input_faces == 1

    def test_single_triangle_2_levels(self):
        """Two levels of refinement should produce 16 triangles."""
        pts, tris = _single_triangle()
        result = surface_refine(pts, tris, levels=2)
        # First level: 1->4; second level: 4->16
        assert result.n_output_faces == 16
        assert result.levels == 2

    def test_two_triangles_refine(self):
        """Refining 2 triangles should produce 8 triangles."""
        pts, tris = _two_triangles()
        result = surface_refine(pts, tris, levels=1)
        assert result.n_output_faces == 8

    def test_selective_refine(self):
        """Refining only one of two triangles should trigger green refinement."""
        pts, tris = _two_triangles()
        mask = np.array([True, False], dtype=bool)
        result = surface_refine(pts, tris, refine_mask=mask)
        # 1 red + 1 green -> 4 + 2 = 6 faces
        assert result.n_red_splits == 1
        assert result.n_green_splits == 1
        assert result.n_output_faces == 6

    def test_output_shapes(self):
        """Output arrays should have correct shapes."""
        pts, tris = _unit_cube()
        result = surface_refine(pts, tris, levels=1)
        assert result.vertices.ndim == 2
        assert result.vertices.shape[1] == 3
        assert result.faces.ndim == 2
        assert result.faces.shape[1] == 3

    def test_no_refine_mask_all(self):
        """With no mask, all faces should be refined."""
        pts, tris = _two_triangles()
        result = surface_refine(pts, tris)
        assert result.n_red_splits == 2

    def test_angle_threshold(self):
        """Angle threshold should select appropriate faces."""
        pts = np.array([
            [0, 0, 0], [1, 0, 0], [10, 0, 0],  # long edge triangle
            [0, 1, 0],
        ], dtype=np.float64)
        tris = np.array([[0, 1, 3], [1, 2, 3]], dtype=np.int32)
        result = surface_refine(pts, tris, angle_threshold=1.5)
        # At least the face with the long edge should be selected
        assert result.n_red_splits >= 1

    def test_empty_faces_raises(self):
        """Empty face array should raise ValueError."""
        pts = np.array([[0, 0, 0]], dtype=np.float64)
        tris = np.empty((0, 3), dtype=np.int32)
        with pytest.raises(ValueError, match="at least one face"):
            surface_refine(pts, tris)

    def test_invalid_face_shape_raises(self):
        """Faces with wrong shape should raise ValueError."""
        pts = np.array([[0, 0, 0]], dtype=np.float64)
        tris = np.array([[0, 0]], dtype=np.int32).reshape(1, 2)
        with pytest.raises(ValueError, match="n_faces, 3"):
            surface_refine(pts, tris)

    def test_vertices_not_degenerate(self):
        """Refined vertices should be finite and not NaN."""
        pts, tris = _unit_cube()
        result = surface_refine(pts, tris, levels=1)
        assert np.all(np.isfinite(result.vertices))

    def test_refined_faces_valid_indices(self):
        """All face vertex indices should be valid."""
        pts, tris = _unit_cube()
        result = surface_refine(pts, tris, levels=1)
        n_verts = result.vertices.shape[0]
        assert np.all(result.faces >= 0)
        assert np.all(result.faces < n_verts)

    def test_total_area_conservation(self):
        """Total surface area should be conserved after refinement."""
        pts, tris = _unit_cube()
        # Compute original area
        areas_orig = _compute_areas(pts, tris)
        total_orig = areas_orig.sum()

        result = surface_refine(pts, tris, levels=1)
        areas_refined = _compute_areas(result.vertices, result.faces)
        total_refined = areas_refined.sum()

        np.testing.assert_allclose(total_refined, total_orig, rtol=1e-6)


def _compute_areas(verts, faces):
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    return 0.5 * np.linalg.norm(cross, axis=1)

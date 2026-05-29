"""Tests for surface_features_enhanced_2 — enhanced feature extraction v2."""
from __future__ import annotations
import numpy as np
import pytest
from pyfoam.tools.surface_features_enhanced_2 import (
    SurfaceFeaturesEnhanced2Result,
    surface_features_enhanced_2,
)


def _simple_triangles():
    """Two triangles sharing an edge (forming a 90-degree dihedral)."""
    verts = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0.5, 0.5, 0],
        [0.5, 0.5, 1.0],
    ], dtype=np.float64)
    faces = np.array([[0, 1, 2], [0, 1, 3]], dtype=np.int32)
    return verts, faces


def _cube_tris():
    """Triangulated cube (12 triangles)."""
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


class TestSurfaceFeaturesEnhanced2:
    def test_returns_result_type(self):
        v, f = _simple_triangles()
        r = surface_features_enhanced_2(vertices=v, faces=f)
        assert isinstance(r, SurfaceFeaturesEnhanced2Result)

    def test_n_edges_positive(self):
        v, f = _simple_triangles()
        r = surface_features_enhanced_2(vertices=v, faces=f)
        assert r.n_edges > 0

    def test_n_features_non_negative(self):
        v, f = _simple_triangles()
        r = surface_features_enhanced_2(vertices=v, faces=f)
        assert r.n_features >= 0

    def test_angle_bins_populated(self):
        v, f = _simple_triangles()
        r = surface_features_enhanced_2(vertices=v, faces=f)
        assert isinstance(r.angle_bins, dict)

    def test_topology_fields(self):
        v, f = _simple_triangles()
        r = surface_features_enhanced_2(vertices=v, faces=f)
        assert r.n_chains >= 0
        assert r.n_junctions >= 0
        assert r.n_open_chains >= 0

    def test_curvature_features(self):
        v, f = _simple_triangles()
        r = surface_features_enhanced_2(vertices=v, faces=f, curvature_threshold=10.0)
        assert r.curvature_features >= 0

    def test_min_feature_length_filter(self):
        v, f = _simple_triangles()
        r = surface_features_enhanced_2(vertices=v, faces=f, min_feature_length=100.0)
        assert r.n_features == 0

    def test_cube_features(self):
        v, f = _cube_tris()
        r = surface_features_enhanced_2(vertices=v, faces=f, included_angle=150.0)
        assert r.n_features > 0
        assert r.n_chains > 0

    def test_empty_surface_raises(self):
        v = np.array([[0,0,0]], dtype=np.float64)
        f = np.empty((0, 3), dtype=np.int32)
        with pytest.raises(ValueError, match="no faces"):
            surface_features_enhanced_2(vertices=v, faces=f)

"""Tests for surface_features_enhanced_4 — enhanced feature extraction v4."""
from __future__ import annotations
import numpy as np
import pytest
from pyfoam.tools.surface_features_enhanced_4 import (
    SurfaceFeaturesEnhanced4Result,
    surface_features_enhanced_4,
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


def _simple_triangles():
    verts = np.array([
        [0, 0, 0], [1, 0, 0], [0.5, 0.5, 0], [0.5, 0.5, 1.0],
    ], dtype=np.float64)
    faces = np.array([[0, 1, 2], [0, 1, 3]], dtype=np.int32)
    return verts, faces


class TestSurfaceFeaturesEnhanced4:
    def test_returns_result_type(self):
        v, f = _simple_triangles()
        r = surface_features_enhanced_4(vertices=v, faces=f)
        assert isinstance(r, SurfaceFeaturesEnhanced4Result)

    def test_persistence_array(self):
        v, f = _cube_tris()
        r = surface_features_enhanced_4(vertices=v, faces=f, included_angle=150.0)
        assert r.feature_persistence.shape == (r.n_features,)

    def test_multi_scale_analysis(self):
        v, f = _cube_tris()
        r = surface_features_enhanced_4(
            vertices=v, faces=f, included_angle=150.0,
            multi_scale_angles=[120.0, 150.0, 170.0],
        )
        assert len(r.per_scale_counts) == 3

    def test_hierarchical_classification(self):
        v, f = _cube_tris()
        r = surface_features_enhanced_4(
            vertices=v, faces=f, included_angle=150.0,
            multi_scale_angles=[120.0, 150.0, 170.0],
        )
        assert r.primary_features + r.secondary_features + r.tertiary_features == r.n_features

    def test_empty_surface_raises(self):
        v = np.array([[0,0,0]], dtype=np.float64)
        f = np.empty((0, 3), dtype=np.int32)
        with pytest.raises(ValueError, match="no faces"):
            surface_features_enhanced_4(vertices=v, faces=f)

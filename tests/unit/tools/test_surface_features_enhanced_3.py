"""Tests for surface_features_enhanced_3 — enhanced feature extraction v3."""
from __future__ import annotations
import numpy as np
import pytest
from pyfoam.tools.surface_features_enhanced_3 import (
    SurfaceFeaturesEnhanced3Result,
    surface_features_enhanced_3,
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


class TestSurfaceFeaturesEnhanced3:
    def test_returns_result_type(self):
        v, f = _simple_triangles()
        r = surface_features_enhanced_3(vertices=v, faces=f)
        assert isinstance(r, SurfaceFeaturesEnhanced3Result)

    def test_feature_sharpness_array(self):
        v, f = _cube_tris()
        r = surface_features_enhanced_3(vertices=v, faces=f, included_angle=150.0)
        assert r.feature_sharpness.shape == (r.n_features,)

    def test_weighted_mean_angle(self):
        v, f = _cube_tris()
        r = surface_features_enhanced_3(vertices=v, faces=f, included_angle=150.0)
        if r.n_features > 0:
            assert r.weighted_mean_angle > 0

    def test_per_region_features_empty_no_regions(self):
        v, f = _simple_triangles()
        r = surface_features_enhanced_3(vertices=v, faces=f)
        assert isinstance(r.per_region_features, dict)

    def test_per_region_features_with_regions(self):
        v, f = _cube_tris()
        regions = np.zeros(12, dtype=np.int32)
        regions[6:] = 1
        r = surface_features_enhanced_3(
            vertices=v, faces=f, included_angle=150.0, region_faces=regions,
        )
        assert isinstance(r.per_region_features, dict)

    def test_min_feature_length_filter(self):
        v, f = _simple_triangles()
        r = surface_features_enhanced_3(vertices=v, faces=f, min_feature_length=100.0)
        assert r.n_features == 0

    def test_empty_surface_raises(self):
        v = np.array([[0,0,0]], dtype=np.float64)
        f = np.empty((0, 3), dtype=np.int32)
        with pytest.raises(ValueError, match="no faces"):
            surface_features_enhanced_3(vertices=v, faces=f)

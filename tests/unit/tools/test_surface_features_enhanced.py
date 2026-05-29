"""Tests for surface_features_enhanced — enhanced feature extraction."""
from __future__ import annotations
import numpy as np
import pytest
from pyfoam.tools.surface_features_enhanced import (
    SurfaceFeaturesEnhancedResult,
    surface_features_enhanced,
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


def _two_perpendicular():
    pts = np.array([
        [0,0,0],[1,0,0],[0,1,0],
        [0,0,0],[1,0,0],[0,0,1],
    ], dtype=np.float64)
    tris = np.array([[0,1,2],[3,4,5]], dtype=np.int32)
    return pts, tris


class TestSurfaceFeaturesEnhanced:
    def test_returns_result_type(self):
        pts, tris = _unit_cube_triangles()
        r = surface_features_enhanced("", vertices=pts, faces=tris)
        assert isinstance(r, SurfaceFeaturesEnhancedResult)

    def test_cube_has_features(self):
        pts, tris = _unit_cube_triangles()
        r = surface_features_enhanced("", vertices=pts, faces=tris)
        assert r.n_features > 0
        assert r.feature_points.shape[0] == r.n_features

    def test_angle_bins_populated(self):
        pts, tris = _unit_cube_triangles()
        r = surface_features_enhanced("", vertices=pts, faces=tris)
        assert isinstance(r.angle_bins, dict)
        total = sum(r.angle_bins.values())
        assert total == r.n_features

    def test_feature_lengths(self):
        pts, tris = _unit_cube_triangles()
        r = surface_features_enhanced("", vertices=pts, faces=tris)
        assert r.feature_lengths.shape[0] == r.n_features
        assert np.all(r.feature_lengths > 0)

    def test_min_feature_length_filter(self):
        pts, tris = _unit_cube_triangles()
        r_all = surface_features_enhanced("", vertices=pts, faces=tris, min_feature_length=0.0)
        r_filtered = surface_features_enhanced("", vertices=pts, faces=tris, min_feature_length=100.0)
        assert r_filtered.n_features <= r_all.n_features

    def test_custom_angle_bins(self):
        pts, tris = _unit_cube_triangles()
        r = surface_features_enhanced(
            "", vertices=pts, faces=tris, angle_bins=[45, 90, 135],
        )
        assert isinstance(r.angle_bins, dict)

    def test_region_faces(self):
        pts, tris = _unit_cube_triangles()
        regions = np.zeros(tris.shape[0], dtype=np.int32)
        regions[:6] = 0
        regions[6:] = 1
        r = surface_features_enhanced(
            "", vertices=pts, faces=tris, region_faces=regions,
        )
        if r.region_ids is not None:
            assert r.region_ids.shape[0] == r.n_features

    def test_open_edges_have_angle_180(self):
        pts = np.array([[0,0,0],[1,0,0],[0,1,0]], dtype=np.float64)
        tris = np.array([[0,1,2]], dtype=np.int32)
        r = surface_features_enhanced("", vertices=pts, faces=tris)
        assert np.allclose(r.feature_angles, 180.0)

    def test_nonexistent_file_raises(self):
        with pytest.raises(FileNotFoundError):
            surface_features_enhanced("/nonexistent/path.stl")

    def test_empty_surface_raises(self):
        pts = np.array([[0,0,0]], dtype=np.float64)
        tris = np.empty((0, 3), dtype=np.int32)
        with pytest.raises(ValueError):
            surface_features_enhanced("", vertices=pts, faces=tris)

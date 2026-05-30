"""Tests for surface_features_enhanced_7 — enhanced feature extraction v7."""
from __future__ import annotations
import numpy as np
import pytest
from pyfoam.tools.surface_features_enhanced_7 import SurfaceFeaturesEnhanced7Result, surface_features_enhanced_7


def _cube_tris():
    v = np.array([
        [0,0,0],[1,0,0],[1,1,0],[0,1,0],
        [0,0,1],[1,0,1],[1,1,1],[0,1,1],
    ], dtype=np.float64)
    f = np.array([
        [0,1,2],[0,2,3],[4,6,5],[4,7,6],
        [0,4,5],[0,5,1],[2,6,7],[2,7,3],
        [0,3,7],[0,7,4],[1,5,6],[1,6,2],
    ], dtype=np.int32)
    return v, f


class TestSurfaceFeaturesEnhanced7:
    def test_returns_result_type(self):
        v, f = _cube_tris()
        r = surface_features_enhanced_7(vertices=v, faces=f, included_angle=150.0)
        assert isinstance(r, SurfaceFeaturesEnhanced7Result)

    def test_curvature_adaptive(self):
        v, f = _cube_tris()
        r = surface_features_enhanced_7(
            vertices=v, faces=f, included_angle=150.0,
            curvature_adaptive=True, adaptive_base_angle=100.0,
        )
        assert r.n_curvature_adapted >= 0
        assert isinstance(r.adaptive_thresholds, list)

    def test_smooth_features(self):
        v, f = _cube_tris()
        r = surface_features_enhanced_7(
            vertices=v, faces=f, included_angle=150.0,
            smooth_features=True, smooth_iterations=2,
        )
        assert r.n_smoothed >= 0

    def test_proximity_query(self):
        v, f = _cube_tris()
        query_pts = np.array([[0.5, 0.5, 0.5]], dtype=np.float64)
        r = surface_features_enhanced_7(
            vertices=v, faces=f, included_angle=150.0,
            proximity_query_points=query_pts, proximity_radius=0.5,
        )
        assert r.query_points is not None
        assert r.query_points.ndim == 2

    def test_empty_surface_raises(self):
        v = np.array([[0,0,0]], dtype=np.float64)
        f = np.empty((0, 3), dtype=np.int32)
        with pytest.raises(ValueError):
            surface_features_enhanced_7(vertices=v, faces=f)

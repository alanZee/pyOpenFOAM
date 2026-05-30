"""Tests for surface_features_enhanced_5 — enhanced feature extraction v5."""
from __future__ import annotations
import numpy as np
import pytest
from pyfoam.tools.surface_features_enhanced_5 import (
    SurfaceFeaturesEnhanced5Result, FeatureGroup, surface_features_enhanced_5,
)


def _simple_tri_mesh():
    """Single triangle surface."""
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    return verts, faces


def _cube_tris():
    """Cube surface with 12 triangles."""
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


class TestSurfaceFeaturesEnhanced5:
    def test_returns_result_type(self):
        v, f = _cube_tris()
        r = surface_features_enhanced_5(vertices=v, faces=f, included_angle=150.0)
        assert isinstance(r, SurfaceFeaturesEnhanced5Result)

    def test_feature_groups(self):
        v, f = _cube_tris()
        r = surface_features_enhanced_5(vertices=v, faces=f, included_angle=150.0)
        assert isinstance(r.feature_groups, list)
        assert r.n_groups == len(r.feature_groups)

    def test_importance_scores(self):
        v, f = _cube_tris()
        r = surface_features_enhanced_5(vertices=v, faces=f, included_angle=150.0)
        assert r.importance_scores.shape[0] == r.n_features

    def test_top_features(self):
        v, f = _cube_tris()
        r = surface_features_enhanced_5(vertices=v, faces=f, included_angle=150.0)
        assert len(r.top_features) <= 10

    def test_generate_dict(self):
        v, f = _cube_tris()
        r = surface_features_enhanced_5(
            vertices=v, faces=f, included_angle=150.0,
            generate_dict=True, dict_surface_name="cube.obj",
        )
        assert r.dict_snippet is not None
        assert "surfaceFeaturesDict" in r.dict_snippet

    def test_empty_surface_raises(self):
        v = np.array([[0,0,0]], dtype=np.float64)
        f = np.empty((0, 3), dtype=np.int32)
        with pytest.raises(ValueError):
            surface_features_enhanced_5(vertices=v, faces=f)

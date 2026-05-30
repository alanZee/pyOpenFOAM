"""Tests for surface_features_enhanced_6 — enhanced feature extraction v6."""
from __future__ import annotations
import numpy as np
import pytest
from pyfoam.tools.surface_features_enhanced_6 import (
    SurfaceFeaturesEnhanced6Result, FeatureHierarchy, MeshingParam,
    surface_features_enhanced_6,
)


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


class TestSurfaceFeaturesEnhanced6:
    def test_returns_result_type(self):
        v, f = _cube_tris()
        r = surface_features_enhanced_6(vertices=v, faces=f, included_angle=150.0)
        assert isinstance(r, SurfaceFeaturesEnhanced6Result)

    def test_simplify_features(self):
        v, f = _cube_tris()
        r = surface_features_enhanced_6(
            vertices=v, faces=f, included_angle=150.0,
            simplify_features=True, simplify_min_angle=5.0,
        )
        assert r.n_simplified >= 0

    def test_hierarchy(self):
        v, f = _cube_tris()
        r = surface_features_enhanced_6(
            vertices=v, faces=f, included_angle=150.0,
            hierarchy_angles=[30.0, 60.0, 90.0],
        )
        assert len(r.hierarchy) == 3
        assert isinstance(r.hierarchy[0], FeatureHierarchy)

    def test_meshing_params(self):
        v, f = _cube_tris()
        r = surface_features_enhanced_6(
            vertices=v, faces=f, included_angle=150.0,
            generate_meshing_params=True, base_refinement_level=2,
        )
        assert isinstance(r.meshing_params, list)
        assert isinstance(r.meshing_dict_snippet, str)

    def test_empty_hierarchy(self):
        v, f = _cube_tris()
        r = surface_features_enhanced_6(
            vertices=v, faces=f, included_angle=150.0,
        )
        assert len(r.hierarchy) == 0

    def test_empty_surface_raises(self):
        v = np.array([[0,0,0]], dtype=np.float64)
        f = np.empty((0, 3), dtype=np.int32)
        with pytest.raises(ValueError):
            surface_features_enhanced_6(vertices=v, faces=f)

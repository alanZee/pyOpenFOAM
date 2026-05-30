"""Tests for surface_auto_patch_enhanced_6 — enhanced auto-patching v6."""
from __future__ import annotations
import numpy as np
import pytest
from pyfoam.tools.surface_auto_patch_enhanced_6 import (
    SurfaceAutoPatchEnhanced6Result, surface_auto_patch_enhanced_6,
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


class TestSurfaceAutoPatchEnhanced6:
    def test_returns_result_type(self):
        v, f = _cube_tris()
        r = surface_auto_patch_enhanced_6(vertices=v, faces=f, feature_angle=30.0)
        assert isinstance(r, SurfaceAutoPatchEnhanced6Result)

    def test_curvature_aware(self):
        v, f = _cube_tris()
        r = surface_auto_patch_enhanced_6(
            vertices=v, faces=f, feature_angle=30.0,
            curvature_aware=True, curvature_threshold=0.1,
        )
        assert r.n_curvature_refined >= 0

    def test_max_patch_faces(self):
        v, f = _cube_tris()
        r = surface_auto_patch_enhanced_6(
            vertices=v, faces=f, feature_angle=30.0,
            max_patch_faces=3,
        )
        assert r.n_optimised_patches >= 0

    def test_feature_alignment(self):
        v, f = _cube_tris()
        r = surface_auto_patch_enhanced_6(
            vertices=v, faces=f, feature_angle=30.0,
            align_to_features=True,
        )
        assert isinstance(r.feature_aligned, bool)
        assert r.boundary_edge_count >= 0

    def test_default_no_curvature(self):
        v, f = _cube_tris()
        r = surface_auto_patch_enhanced_6(vertices=v, faces=f, feature_angle=30.0)
        assert r.n_curvature_refined == 0

    def test_empty_result_fields(self):
        v, f = _cube_tris()
        r = surface_auto_patch_enhanced_6(vertices=v, faces=f, feature_angle=30.0)
        assert r.n_patches >= 0

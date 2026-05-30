"""Tests for surface_auto_patch_enhanced_4 — enhanced auto-patching v4."""
from __future__ import annotations
import numpy as np
import pytest
from pyfoam.tools.surface_auto_patch_enhanced_4 import (
    SurfaceAutoPatchEnhanced4Result,
    surface_auto_patch_enhanced_4,
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


class TestSurfaceAutoPatchEnhanced4:
    def test_returns_result_type(self):
        v, f = _cube_tris()
        r = surface_auto_patch_enhanced_4(vertices=v, faces=f, feature_angle=40.0)
        assert isinstance(r, SurfaceAutoPatchEnhanced4Result)

    def test_compactness_score(self):
        v, f = _cube_tris()
        r = surface_auto_patch_enhanced_4(vertices=v, faces=f, feature_angle=40.0)
        for pid, score in r.patch_compactness.items():
            assert 0.0 <= score <= 1.0

    def test_concave_patches(self):
        v, f = _cube_tris()
        r = surface_auto_patch_enhanced_4(vertices=v, faces=f, feature_angle=40.0)
        assert isinstance(r.n_concave_patches, int)

    def test_adaptive_angle(self):
        v, f = _cube_tris()
        r = surface_auto_patch_enhanced_4(
            vertices=v, faces=f, feature_angle=40.0, adaptive_angle=True,
        )
        assert r.mean_adaptive_angle > 0

    def test_empty_surface(self):
        v = np.array([[0,0,0]], dtype=np.float64)
        f = np.empty((0, 3), dtype=np.int32)
        r = surface_auto_patch_enhanced_4(vertices=v, faces=f)
        assert r.n_patches == 0

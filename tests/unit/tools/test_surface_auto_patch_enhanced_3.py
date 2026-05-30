"""Tests for surface_auto_patch_enhanced_3 — enhanced auto-patching v3."""
from __future__ import annotations
import numpy as np
import pytest
from pyfoam.tools.surface_auto_patch_enhanced_3 import (
    SurfaceAutoPatchEnhanced3Result,
    surface_auto_patch_enhanced_3,
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


def _two_plane_tris():
    """Two disconnected planes."""
    v = np.array([
        [0,0,0],[1,0,0],[0.5,0.5,0],
        [0,0,5],[1,0,5],[0.5,0.5,5],
    ], dtype=np.float64)
    f = np.array([[0,1,2],[3,4,5]], dtype=np.int32)
    return v, f


class TestSurfaceAutoPatchEnhanced3:
    def test_returns_result_type(self):
        v, f = _cube_tris()
        r = surface_auto_patch_enhanced_3(vertices=v, faces=f)
        assert isinstance(r, SurfaceAutoPatchEnhanced3Result)

    def test_n_patches_positive(self):
        v, f = _cube_tris()
        r = surface_auto_patch_enhanced_3(vertices=v, faces=f, feature_angle=30.0)
        assert r.n_patches > 0

    def test_patch_ids_shape(self):
        v, f = _cube_tris()
        r = surface_auto_patch_enhanced_3(vertices=v, faces=f)
        assert r.patch_ids.shape == (12,)

    def test_quality_scores_populated(self):
        v, f = _cube_tris()
        r = surface_auto_patch_enhanced_3(vertices=v, faces=f)
        assert len(r.patch_quality_scores) == r.n_patches

    def test_boundary_edges_populated(self):
        v, f = _cube_tris()
        r = surface_auto_patch_enhanced_3(vertices=v, faces=f)
        assert len(r.patch_boundary_edges) == r.n_patches

    def test_disconnected_regions(self):
        v, f = _two_plane_tris()
        r = surface_auto_patch_enhanced_3(vertices=v, faces=f, feature_angle=30.0)
        assert r.n_patches >= 1

    def test_with_region_ids(self):
        v, f = _cube_tris()
        regions = np.zeros(12, dtype=np.int32)
        regions[6:] = 1
        r = surface_auto_patch_enhanced_3(
            vertices=v, faces=f, feature_angle=30.0, region_ids=regions,
        )
        assert r.n_regions == 2

    def test_min_patch_faces_merge(self):
        v, f = _cube_tris()
        r = surface_auto_patch_enhanced_3(
            vertices=v, faces=f, feature_angle=30.0, min_patch_faces=100,
        )
        # Very high threshold should merge everything
        assert r.n_patches >= 1

    def test_name_by_direction(self):
        v, f = _cube_tris()
        r = surface_auto_patch_enhanced_3(
            vertices=v, faces=f, feature_angle=30.0, name_by_direction=True,
        )
        for pid, name in r.patch_names.items():
            assert isinstance(name, str)

    def test_smooth_iterations(self):
        v, f = _cube_tris()
        r = surface_auto_patch_enhanced_3(
            vertices=v, faces=f, feature_angle=30.0, smooth_iterations=2,
        )
        assert r.n_patches > 0

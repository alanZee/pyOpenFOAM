"""Tests for surface_auto_patch_enhanced_5 — enhanced auto-patching v5."""
from __future__ import annotations
import numpy as np
import pytest
from pyfoam.tools.surface_auto_patch_enhanced_5 import (
    SurfaceAutoPatchEnhanced5Result, PatchStatistics, surface_auto_patch_enhanced_5,
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


class TestSurfaceAutoPatchEnhanced5:
    def test_returns_result_type(self):
        v, f = _cube_tris()
        r = surface_auto_patch_enhanced_5(vertices=v, faces=f, feature_angle=30.0)
        assert isinstance(r, SurfaceAutoPatchEnhanced5Result)

    def test_export_statistics(self):
        v, f = _cube_tris()
        r = surface_auto_patch_enhanced_5(
            vertices=v, faces=f, feature_angle=30.0,
            export_statistics=True,
        )
        assert isinstance(r.patch_statistics, list)
        if r.patch_statistics:
            assert isinstance(r.patch_statistics[0], PatchStatistics)

    def test_generate_dict(self):
        v, f = _cube_tris()
        r = surface_auto_patch_enhanced_5(
            vertices=v, faces=f, feature_angle=30.0,
            generate_dict=True,
        )
        assert r.dict_snippet is not None
        assert "createPatchDict" in r.dict_snippet

    def test_n_patches(self):
        v, f = _cube_tris()
        r = surface_auto_patch_enhanced_5(vertices=v, faces=f, feature_angle=30.0)
        assert r.n_patches >= 1

    def test_patch_ids_shape(self):
        v, f = _cube_tris()
        r = surface_auto_patch_enhanced_5(vertices=v, faces=f, feature_angle=30.0)
        assert r.patch_ids.shape[0] == f.shape[0]

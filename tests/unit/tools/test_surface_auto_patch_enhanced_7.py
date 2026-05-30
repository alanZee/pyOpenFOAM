"""Tests for surface_auto_patch_enhanced_7 — enhanced auto-patching v7."""
from __future__ import annotations
import numpy as np
import pytest
from pyfoam.tools.surface_auto_patch_enhanced_7 import SurfaceAutoPatchEnhanced7Result, PatchAdjacency, MeshSizeEstimate, surface_auto_patch_enhanced_7


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


class TestSurfaceAutoPatchEnhanced7:
    def test_returns_result_type(self):
        v, f = _cube_tris()
        r = surface_auto_patch_enhanced_7(vertices=v, faces=f, feature_angle=30.0)
        assert isinstance(r, SurfaceAutoPatchEnhanced7Result)

    def test_semantic_labelling(self):
        v, f = _cube_tris()
        r = surface_auto_patch_enhanced_7(
            vertices=v, faces=f, feature_angle=30.0,
            semantic_labelling=True, flow_direction=(1.0, 0.0, 0.0),
        )
        assert isinstance(r.semantic_labels, dict)

    def test_adjacency(self):
        v, f = _cube_tris()
        r = surface_auto_patch_enhanced_7(vertices=v, faces=f, feature_angle=30.0)
        assert isinstance(r.adjacencies, list)
        if r.adjacencies:
            assert isinstance(r.adjacencies[0], PatchAdjacency)

    def test_mesh_size_estimation(self):
        v, f = _cube_tris()
        r = surface_auto_patch_enhanced_7(
            vertices=v, faces=f, feature_angle=30.0,
            estimate_mesh_size=True, target_cell_count=1000,
        )
        assert isinstance(r.mesh_size_estimates, list)
        if r.mesh_size_estimates:
            assert isinstance(r.mesh_size_estimates[0], MeshSizeEstimate)

    def test_no_semantic_by_default(self):
        v, f = _cube_tris()
        r = surface_auto_patch_enhanced_7(vertices=v, faces=f, feature_angle=30.0)
        assert len(r.semantic_labels) == 0

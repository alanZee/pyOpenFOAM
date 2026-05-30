"""Tests for surface_auto_patch_enhanced_9 — enhanced auto-patching v9."""
from __future__ import annotations
import pytest
import numpy as np
from pyfoam.tools.surface_auto_patch_enhanced_9 import (
    SurfaceAutoPatchEnhanced9Result, ClusterResult, RefinementIteration,
    PatchOptimization, surface_auto_patch_enhanced_9,
)


class TestSurfaceAutoPatchEnhanced9:
    def test_returns_result_type(self):
        verts = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0]], dtype=np.float64)
        faces = np.array([[0,1,2],[0,2,3]], dtype=np.int32)
        r = surface_auto_patch_enhanced_9(vertices=verts, faces=faces)
        assert isinstance(r, SurfaceAutoPatchEnhanced9Result)

    def test_clustering(self):
        pts = np.random.rand(30, 3)
        faces = np.array([[i, i+1, i+2] for i in range(28)])
        r = surface_auto_patch_enhanced_9(
            vertices=pts, faces=faces,
            clustering_detection=True, n_clusters=3,
        )
        assert isinstance(r.clustering, ClusterResult)
        assert r.clustering.n_clusters > 0

    def test_adaptive_refinement(self):
        verts = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0]], dtype=np.float64)
        faces = np.array([[0,1,2],[0,2,3]], dtype=np.int32)
        r = surface_auto_patch_enhanced_9(
            vertices=verts, faces=faces,
            adaptive_refinement=True, max_refinement_iterations=3,
        )
        assert isinstance(r.refinement_history, list)

    def test_patch_optimization(self):
        verts = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0]], dtype=np.float64)
        faces = np.array([[0,1,2],[0,2,3]], dtype=np.int32)
        r = surface_auto_patch_enhanced_9(
            vertices=verts, faces=faces,
            optimize_patches=True, target_patch_count=3,
        )
        assert isinstance(r.optimization, PatchOptimization)

    def test_default_values(self):
        r = SurfaceAutoPatchEnhanced9Result()
        assert r.n_patches == 0
        assert r.clustering.n_clusters == 0

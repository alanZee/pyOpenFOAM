"""Tests for stitch_mesh_enhanced_4 — enhanced mesh stitching v4."""
from __future__ import annotations
import pytest
import torch
from pyfoam.core.dtype import INDEX_DTYPE
from pyfoam.mesh.fv_mesh import FvMesh
from pyfoam.tools.stitch_mesh_enhanced_4 import StitchEnhanced4Result, stitch_mesh_enhanced_4


def _mesh_with_patches():
    pts = [[0,0,0],[1,0,0],[1,1,0],[0,1,0],
           [0,0,1],[1,0,1],[1,1,1],[0,1,1]]
    fc = [[0,3,2,1],[4,5,6,7],[0,1,5,4],[2,3,7,6],[0,4,7,3],[1,2,6,5]]
    m = FvMesh(points=torch.tensor(pts, dtype=torch.float64),
               faces=[torch.tensor(f, dtype=INDEX_DTYPE) for f in fc],
               owner=torch.zeros(6, dtype=INDEX_DTYPE),
               neighbour=torch.tensor([], dtype=INDEX_DTYPE),
               boundary=[
                   {"name": "patch_a", "type": "wall", "startFace": 0, "nFaces": 1},
                   {"name": "patch_b", "type": "wall", "startFace": 1, "nFaces": 1},
                   {"name": "others", "type": "wall", "startFace": 2, "nFaces": 4},
               ],
               validate=False)
    m.compute_geometry()
    return m


class TestStitchEnhanced4:
    def test_returns_result_type(self):
        m = _mesh_with_patches()
        r = stitch_mesh_enhanced_4(m, "patch_a", "patch_b")
        assert isinstance(r, StitchEnhanced4Result)

    def test_icp_improvement_metric(self):
        m = _mesh_with_patches()
        r = stitch_mesh_enhanced_4(m, "patch_a", "patch_b", icp_iterations=2)
        assert isinstance(r.icp_improvement, float)

    def test_stitch_strength_metric(self):
        m = _mesh_with_patches()
        r = stitch_mesh_enhanced_4(m, "patch_a", "patch_b")
        assert isinstance(r.mean_stitch_strength, float)

    def test_gap_fill_disabled(self):
        m = _mesh_with_patches()
        r = stitch_mesh_enhanced_4(m, "patch_a", "patch_b", gap_fill_threshold=0.0)
        assert r.n_gap_filled == 0

    def test_nonexistent_patch_raises(self):
        m = _mesh_with_patches()
        with pytest.raises(ValueError, match="not found"):
            stitch_mesh_enhanced_4(m, "nonexistent", "patch_b")

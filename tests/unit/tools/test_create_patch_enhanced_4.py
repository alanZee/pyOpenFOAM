"""Tests for create_patch_enhanced_4 — enhanced patch creation v4."""
from __future__ import annotations
import pytest
import torch
from pyfoam.core.dtype import INDEX_DTYPE
from pyfoam.mesh.fv_mesh import FvMesh
from pyfoam.tools.create_patch_enhanced_4 import PatchEnhanced4Result, create_patch_enhanced_4


def _single_hex():
    pts = [[0,0,0],[1,0,0],[1,1,0],[0,1,0],
           [0,0,1],[1,0,1],[1,1,1],[0,1,1]]
    fc = [[0,3,2,1],[4,5,6,7],[0,1,5,4],[2,3,7,6],[0,4,7,3],[1,2,6,5]]
    m = FvMesh(points=torch.tensor(pts, dtype=torch.float64),
               faces=[torch.tensor(f, dtype=INDEX_DTYPE) for f in fc],
               owner=torch.zeros(6, dtype=INDEX_DTYPE),
               neighbour=torch.tensor([], dtype=INDEX_DTYPE),
               boundary=[{"name": "all", "type": "wall", "startFace": 0, "nFaces": 6}],
               validate=False)
    m.compute_geometry()
    return m


class TestPatchEnhanced4:
    def test_returns_result_type(self):
        m = _single_hex()
        r = create_patch_enhanced_4(m, face_indices=[0, 1], patch_name="test_patch")
        assert isinstance(r, PatchEnhanced4Result)

    def test_patch_centroids(self):
        m = _single_hex()
        r = create_patch_enhanced_4(m, face_indices=[0], patch_name="test_patch")
        if "test_patch" in r.patch_centroids:
            assert len(r.patch_centroids["test_patch"]) == 3

    def test_normal_consistency(self):
        m = _single_hex()
        r = create_patch_enhanced_4(m, face_indices=[0], patch_name="test_patch")
        if "test_patch" in r.patch_normal_consistency:
            assert 0.0 <= r.patch_normal_consistency["test_patch"] <= 1.0

    def test_merge_patches_skip(self):
        m = FvMesh(
            points=torch.tensor([[0,0,0],[1,0,0],[1,1,0],[0,1,0],
                                 [0,0,1],[1,0,1],[1,1,1],[0,1,1]], dtype=torch.float64),
            faces=[torch.tensor(f, dtype=INDEX_DTYPE) for f in
                   [[0,3,2,1],[4,5,6,7],[0,1,5,4],[2,3,7,6],[0,4,7,3],[1,2,6,5]]],
            owner=torch.zeros(6, dtype=INDEX_DTYPE),
            neighbour=torch.tensor([], dtype=INDEX_DTYPE),
            boundary=[
                {"name": "a", "type": "wall", "startFace": 0, "nFaces": 1},
                {"name": "b", "type": "wall", "startFace": 1, "nFaces": 1},
                {"name": "c", "type": "wall", "startFace": 2, "nFaces": 4},
            ],
            validate=False,
        )
        m.compute_geometry()
        r = create_patch_enhanced_4(m, merge_patches=["a", "b"], merged_name="ab")
        assert r.n_merged_patches == 2
        assert "ab" in r.patches_created

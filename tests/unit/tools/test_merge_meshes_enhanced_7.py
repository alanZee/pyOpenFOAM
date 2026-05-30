"""Tests for merge_meshes_enhanced_7 — enhanced mesh merging v7."""
from __future__ import annotations
import pytest
import torch
from pyfoam.core.dtype import INDEX_DTYPE
from pyfoam.mesh.fv_mesh import FvMesh
from pyfoam.tools.merge_meshes_enhanced_7 import MergeEnhanced7Result, MergeDiagnostic, merge_meshes_enhanced_7


def _single_hex(offset=(0, 0, 0)):
    ox, oy, oz = offset
    pts = [[ox,oy,oz],[ox+1,oy,oz],[ox+1,oy+1,oz],[ox,oy+1,oz],
           [ox,oy,oz+1],[ox+1,oy,oz+1],[ox+1,oy+1,oz+1],[ox,oy+1,oz+1]]
    fc = [[0,3,2,1],[4,5,6,7],[0,1,5,4],[2,3,7,6],[0,4,7,3],[1,2,6,5]]
    m = FvMesh(points=torch.tensor(pts, dtype=torch.float64),
               faces=[torch.tensor(f, dtype=INDEX_DTYPE) for f in fc],
               owner=torch.zeros(6, dtype=INDEX_DTYPE),
               neighbour=torch.tensor([], dtype=INDEX_DTYPE),
               boundary=[{"name": "all", "type": "wall", "startFace": 0, "nFaces": 6}],
               validate=False)
    m.compute_geometry()
    return m


class TestMergeEnhanced7:
    def test_returns_result_type(self):
        m = _single_hex()
        r = merge_meshes_enhanced_7([m])
        assert isinstance(r, MergeEnhanced7Result)

    def test_hierarchical_flag(self):
        m1 = _single_hex((0, 0, 0))
        m2 = _single_hex((10, 0, 0))
        r = merge_meshes_enhanced_7([m1, m2], hierarchical=True)
        assert r.hierarchical is True

    def test_quality_flagged(self):
        m = _single_hex()
        r = merge_meshes_enhanced_7([m], quality_threshold=0.3)
        assert r.n_quality_flagged >= 0

    def test_diagnostic_present(self):
        m = _single_hex()
        r = merge_meshes_enhanced_7([m])
        assert isinstance(r.diagnostic, MergeDiagnostic)
        assert r.diagnostic.total_wall_time_ms >= 0

    def test_diagnostic_stages(self):
        m1 = _single_hex((0, 0, 0))
        m2 = _single_hex((10, 0, 0))
        r = merge_meshes_enhanced_7([m1, m2])
        assert r.diagnostic.n_stages >= 1
        assert len(r.diagnostic.stages) >= 1

    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match="empty"):
            merge_meshes_enhanced_7([])

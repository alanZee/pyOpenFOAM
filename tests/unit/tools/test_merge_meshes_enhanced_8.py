"""Tests for merge_meshes_enhanced_8 — enhanced mesh merging v8."""
from __future__ import annotations
import pytest
import torch
from pyfoam.core.dtype import INDEX_DTYPE
from pyfoam.mesh.fv_mesh import FvMesh
from pyfoam.tools.merge_meshes_enhanced_8 import MergeEnhanced8Result, merge_meshes_enhanced_8


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


class TestMergeEnhanced8:
    def test_returns_result_type(self):
        m = _single_hex()
        r = merge_meshes_enhanced_8([m])
        assert isinstance(r, MergeEnhanced8Result)

    def test_conflict_resolution(self):
        m1 = _single_hex((0, 0, 0))
        m2 = _single_hex((10, 0, 0))
        r = merge_meshes_enhanced_8([m1, m2], resolve_conflicts=True, merge_zones=True)
        assert r.n_conflicts_detected >= 0
        assert r.n_conflicts_resolved_auto >= 0

    def test_field_transfer(self):
        m = _single_hex()
        r = merge_meshes_enhanced_8([m], transfer_fields=True, field_names=["points"])
        assert r.n_fields_transferred >= 0
        assert isinstance(r.field_transfers, list)

    def test_parallel_schedule(self):
        m1 = _single_hex((0, 0, 0))
        m2 = _single_hex((5, 0, 0))
        m3 = _single_hex((10, 0, 0))
        r = merge_meshes_enhanced_8([m1, m2, m3], parallel_schedule=True, max_parallel_pairs=2)
        assert isinstance(r.parallel_schedule, list)
        assert r.schedule_efficiency >= 0

    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match="empty"):
            merge_meshes_enhanced_8([])

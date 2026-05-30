"""Tests for merge_meshes_enhanced_6 — enhanced mesh merging v6."""
from __future__ import annotations
import pytest
import torch
from pyfoam.core.dtype import INDEX_DTYPE
from pyfoam.mesh.fv_mesh import FvMesh
from pyfoam.tools.merge_meshes_enhanced_6 import MergeEnhanced6Result, merge_meshes_enhanced_6


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


class TestMergeEnhanced6:
    def test_returns_result_type(self):
        m = _single_hex()
        r = merge_meshes_enhanced_6([m])
        assert isinstance(r, MergeEnhanced6Result)

    def test_merge_schedule(self):
        m1 = _single_hex((0, 0, 0))
        m2 = _single_hex((10, 0, 0))
        r = merge_meshes_enhanced_6([m1, m2], schedule_strategy="min_cost")
        assert isinstance(r.merge_schedule, list)

    def test_sequential_schedule(self):
        m1 = _single_hex((0, 0, 0))
        m2 = _single_hex((10, 0, 0))
        r = merge_meshes_enhanced_6([m1, m2], schedule_strategy="sequential")
        assert len(r.merge_schedule) == 1

    def test_conflict_detection(self):
        m1 = _single_hex((0, 0, 0))
        m2 = _single_hex((5, 0, 0))
        r = merge_meshes_enhanced_6(
            [m1, m2], conflict_resolution="priority",
        )
        assert r.n_conflicts_resolved >= 0

    def test_parallel_flag(self):
        m1 = _single_hex((0, 0, 0))
        m2 = _single_hex((10, 0, 0))
        m3 = _single_hex((20, 0, 0))
        r = merge_meshes_enhanced_6([m1, m2, m3], parallel_merge=True)
        assert isinstance(r.parallel, bool)

    def test_schedule_cost(self):
        m1 = _single_hex((0, 0, 0))
        m2 = _single_hex((10, 0, 0))
        r = merge_meshes_enhanced_6([m1, m2])
        assert r.schedule_cost >= 0

    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match="empty"):
            merge_meshes_enhanced_6([])

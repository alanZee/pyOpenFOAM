"""Tests for merge_meshes_enhanced_5 — enhanced mesh merging v5."""
from __future__ import annotations
import pytest
import torch
from pyfoam.core.dtype import INDEX_DTYPE
from pyfoam.mesh.fv_mesh import FvMesh
from pyfoam.tools.merge_meshes_enhanced_5 import MergeEnhanced5Result, merge_meshes_enhanced_5


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


class TestSingleMesh:
    def test_returns_result_type(self):
        m = _single_hex()
        r = merge_meshes_enhanced_5([m])
        assert isinstance(r, MergeEnhanced5Result)

    def test_quality_score(self):
        m = _single_hex()
        r = merge_meshes_enhanced_5([m])
        assert 0.0 <= r.quality_score <= 1.0

    def test_topology_valid(self):
        m = _single_hex()
        r = merge_meshes_enhanced_5([m])
        assert r.topology_valid is True

    def test_volume_conserved(self):
        m = _single_hex()
        r = merge_meshes_enhanced_5([m])
        assert r.volume_conserved is True


class TestTwoMeshes:
    def test_quality_after_merge(self):
        m1 = _single_hex((0, 0, 0))
        m2 = _single_hex((10, 0, 0))
        r = merge_meshes_enhanced_5([m1, m2])
        assert r.quality_score >= 0.0
        assert r.mesh.n_cells == 2

    def test_boundary_layer_axis(self):
        m1 = _single_hex((0, 0, 0))
        m2 = _single_hex((10, 0, 0))
        r = merge_meshes_enhanced_5([m1, m2], boundary_layer_axis=2)
        assert r.mesh.n_cells == 2


class TestEdgeCases:
    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match="empty"):
            merge_meshes_enhanced_5([])

    def test_no_topology_validation(self):
        m = _single_hex()
        r = merge_meshes_enhanced_5([m], validate_topology=False)
        assert r.topology_valid is True

"""Tests for merge_meshes_enhanced_4 — enhanced mesh merging v4."""
from __future__ import annotations
import pytest
import torch
from pyfoam.core.dtype import INDEX_DTYPE
from pyfoam.mesh.fv_mesh import FvMesh
from pyfoam.tools.merge_meshes_enhanced_4 import MergeEnhanced4Result, merge_meshes_enhanced_4


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
        r = merge_meshes_enhanced_4([m])
        assert isinstance(r, MergeEnhanced4Result)

    def test_volume_conserved_single(self):
        m = _single_hex()
        r = merge_meshes_enhanced_4([m])
        assert r.volume_conserved is True
        assert r.volume_error < 1e-10

    def test_connectivity_single(self):
        m = _single_hex()
        r = merge_meshes_enhanced_4([m])
        assert r.is_connected is True
        assert r.n_components == 1


class TestTwoMeshes:
    def test_disjoint_meshes_volume(self):
        m1 = _single_hex((0, 0, 0))
        m2 = _single_hex((10, 0, 0))
        r = merge_meshes_enhanced_4([m1, m2])
        assert r.mesh.n_cells == 2

    def test_volume_conservation(self):
        m1 = _single_hex((0, 0, 0))
        m2 = _single_hex((10, 0, 0))
        r = merge_meshes_enhanced_4([m1, m2])
        assert r.volume_conserved is True

    def test_adjacent_connectivity(self):
        m1 = _single_hex((0, 0, 0))
        m2 = _single_hex((1, 0, 0))
        r = merge_meshes_enhanced_4([m1, m2])
        assert r.mesh.n_cells == 2
        assert r.n_components >= 1


class TestWeightedTolerance:
    def test_weighted_tolerance_enabled(self):
        m1 = _single_hex((0, 0, 0))
        m2 = _single_hex((1, 0, 0))
        r = merge_meshes_enhanced_4([m1, m2], weighted_tolerance=True)
        assert r.mesh.n_cells == 2

    def test_weighted_tolerance_disabled(self):
        m1 = _single_hex((0, 0, 0))
        m2 = _single_hex((1, 0, 0))
        r = merge_meshes_enhanced_4([m1, m2], weighted_tolerance=False)
        assert r.mesh.n_cells == 2


class TestEdgeCases:
    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match="empty"):
            merge_meshes_enhanced_4([])

    def test_does_not_modify_original(self):
        m = _single_hex()
        nc = m.n_cells
        merge_meshes_enhanced_4([m])
        assert m.n_cells == nc

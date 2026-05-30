"""Tests for merge_meshes_enhanced_3 — enhanced mesh merging v3."""
from __future__ import annotations
import pytest
import torch
from pyfoam.core.dtype import INDEX_DTYPE
from pyfoam.mesh.fv_mesh import FvMesh
from pyfoam.tools.merge_meshes_enhanced_3 import MergeEnhanced3Result, merge_meshes_enhanced_3


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
        r = merge_meshes_enhanced_3([m])
        assert isinstance(r, MergeEnhanced3Result)

    def test_single_mesh_dedup_ratio_zero(self):
        m = _single_hex()
        r = merge_meshes_enhanced_3([m])
        assert r.dedup_ratio == 0.0

    def test_overlap_count_zero_single(self):
        m = _single_hex()
        r = merge_meshes_enhanced_3([m])
        assert r.overlap_count == 0


class TestTwoMeshes:
    def test_disjoint_meshes(self):
        m1 = _single_hex((0, 0, 0))
        m2 = _single_hex((10, 0, 0))
        r = merge_meshes_enhanced_3([m1, m2])
        assert r.mesh.n_cells == 2
        assert r.per_mesh_cells == [1, 1]

    def test_adjacent_meshes_merge(self):
        m1 = _single_hex((0, 0, 0))
        m2 = _single_hex((1, 0, 0))
        r = merge_meshes_enhanced_3([m1, m2])
        assert r.mesh.n_cells == 2
        assert r.mesh.n_internal_faces >= 1

    def test_total_volume(self):
        m1 = _single_hex((0, 0, 0))
        m2 = _single_hex((10, 0, 0))
        r = merge_meshes_enhanced_3([m1, m2])
        r.mesh.compute_geometry()
        assert abs(r.mesh.total_volume.item() - 2.0) < 1e-8


class TestMultiPassHashing:
    def test_default_two_passes(self):
        m1 = _single_hex((0, 0, 0))
        m2 = _single_hex((1, 0, 0))
        r = merge_meshes_enhanced_3([m1, m2], n_hash_passes=2)
        assert r.mesh.n_cells == 2

    def test_three_passes(self):
        m1 = _single_hex((0, 0, 0))
        m2 = _single_hex((10, 0, 0))
        r = merge_meshes_enhanced_3([m1, m2], n_hash_passes=3)
        assert r.dedup_ratio >= 0


class TestZonePriority:
    def test_merge_zones_with_priority(self):
        m1 = _single_hex((0, 0, 0))
        m2 = _single_hex((10, 0, 0))
        r = merge_meshes_enhanced_3(
            [m1, m2], merge_zones=True,
            zone_priority={"all": 10},
        )
        assert isinstance(r.zone_face_counts, dict)

    def test_overlap_count_for_adjacent(self):
        m1 = _single_hex((0, 0, 0))
        m2 = _single_hex((1, 0, 0))
        r = merge_meshes_enhanced_3([m1, m2])
        assert r.overlap_count >= 0


class TestEdgeCases:
    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match="empty"):
            merge_meshes_enhanced_3([])

    def test_does_not_modify_original(self):
        m = _single_hex()
        nc = m.n_cells
        merge_meshes_enhanced_3([m])
        assert m.n_cells == nc

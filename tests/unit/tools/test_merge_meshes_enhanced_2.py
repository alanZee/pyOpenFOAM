"""Tests for merge_meshes_enhanced_2 — enhanced mesh merging v2."""
from __future__ import annotations
import pytest
import torch
from pyfoam.core.dtype import INDEX_DTYPE
from pyfoam.mesh.fv_mesh import FvMesh
from pyfoam.tools.merge_meshes_enhanced_2 import MergeEnhanced2Result, merge_meshes_enhanced_2


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
        r = merge_meshes_enhanced_2([m])
        assert isinstance(r, MergeEnhanced2Result)

    def test_single_mesh_clone(self):
        m = _single_hex()
        r = merge_meshes_enhanced_2([m])
        assert r.mesh.n_cells == 1
        assert torch.allclose(r.mesh.points, m.points)

    def test_per_mesh_cells(self):
        m = _single_hex()
        r = merge_meshes_enhanced_2([m])
        assert r.per_mesh_cells == [1]


class TestTwoMeshes:
    def test_disjoint_meshes(self):
        m1 = _single_hex((0, 0, 0))
        m2 = _single_hex((10, 0, 0))
        r = merge_meshes_enhanced_2([m1, m2])
        assert r.mesh.n_cells == 2
        assert r.per_mesh_cells == [1, 1]

    def test_adjacent_meshes_merge_shared_face(self):
        m1 = _single_hex((0, 0, 0))
        m2 = _single_hex((1, 0, 0))
        r = merge_meshes_enhanced_2([m1, m2])
        assert r.mesh.n_cells == 2
        assert r.mesh.n_internal_faces >= 1

    def test_total_volume(self):
        m1 = _single_hex((0, 0, 0))
        m2 = _single_hex((10, 0, 0))
        r = merge_meshes_enhanced_2([m1, m2])
        r.mesh.compute_geometry()
        assert abs(r.mesh.total_volume.item() - 2.0) < 1e-8


class TestAdaptiveTolerance:
    def test_adaptive_tolerance_reported(self):
        m1 = _single_hex((0, 0, 0))
        m2 = _single_hex((10, 0, 0))
        r = merge_meshes_enhanced_2([m1, m2], adaptive_tolerance=True)
        assert r.adaptive_tol > 0

    def test_adaptive_off(self):
        m1 = _single_hex((0, 0, 0))
        m2 = _single_hex((10, 0, 0))
        r = merge_meshes_enhanced_2([m1, m2], adaptive_tolerance=False, tolerance=1e-6)
        assert r.adaptive_tol == 1e-6


class TestZoneMerging:
    def test_merge_zones_combines_patches(self):
        m1 = _single_hex((0, 0, 0))
        m2 = _single_hex((10, 0, 0))
        r = merge_meshes_enhanced_2([m1, m2], merge_zones=True)
        assert r.n_zones_merged >= 0
        assert isinstance(r.zone_face_counts, dict)

    def test_zone_face_counts(self):
        m1 = _single_hex((0, 0, 0))
        m2 = _single_hex((10, 0, 0))
        r = merge_meshes_enhanced_2([m1, m2], merge_zones=True)
        total = sum(r.zone_face_counts.values())
        assert total > 0


class TestEdgeCases:
    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match="empty"):
            merge_meshes_enhanced_2([])

    def test_does_not_modify_original(self):
        m = _single_hex()
        nc = m.n_cells
        merge_meshes_enhanced_2([m])
        assert m.n_cells == nc

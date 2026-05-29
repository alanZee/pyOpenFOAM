"""Tests for merge_meshes — merge multiple meshes into one."""
from __future__ import annotations
import pytest, torch
from pyfoam.core.dtype import INDEX_DTYPE
from pyfoam.mesh.fv_mesh import FvMesh
from pyfoam.tools.merge_meshes import merge_meshes


def _single_hex(offset=(0, 0, 0)):
    """创建一个单位立方体 mesh，可指定偏移。"""
    ox, oy, oz = offset
    pts = [[ox, oy, oz], [ox+1, oy, oz], [ox+1, oy+1, oz], [ox, oy+1, oz],
           [ox, oy, oz+1], [ox+1, oy, oz+1], [ox+1, oy+1, oz+1], [ox, oy+1, oz+1]]
    fc = [[0,3,2,1],[4,5,6,7],[0,1,5,4],[2,3,7,6],[0,4,7,3],[1,2,6,5]]
    m = FvMesh(points=torch.tensor(pts, dtype=torch.float64),
               faces=[torch.tensor(f, dtype=INDEX_DTYPE) for f in fc],
               owner=torch.zeros(6, dtype=INDEX_DTYPE),
               neighbour=torch.tensor([], dtype=INDEX_DTYPE),
               boundary=[{"name": "all", "type": "wall", "startFace": 0, "nFaces": 6}],
               validate=False)
    m.compute_geometry()
    return m


def _two_cell_hex():
    """两个堆叠的单位立方体 mesh（共享面）。"""
    pts = [[0,0,0],[1,0,0],[1,1,0],[0,1,0],
           [0,0,1],[1,0,1],[1,1,1],[0,1,1],
           [0,0,2],[1,0,2],[1,1,2],[0,1,2]]
    fc = [[4,5,6,7],[0,3,2,1],[0,1,5,4],[2,3,7,6],[0,4,7,3],[1,2,6,5],
          [8,9,10,11],[4,5,9,8],[7,11,10,6],[4,8,11,7],[5,6,10,9]]
    m = FvMesh(points=torch.tensor(pts, dtype=torch.float64),
               faces=[torch.tensor(f, dtype=INDEX_DTYPE) for f in fc],
               owner=torch.tensor([0,0,0,0,0,0,1,1,1,1,1], dtype=INDEX_DTYPE),
               neighbour=torch.tensor([1], dtype=INDEX_DTYPE),
               boundary=[{"name":"bottom","type":"wall","startFace":1,"nFaces":5},
                         {"name":"top","type":"wall","startFace":6,"nFaces":5}],
               validate=False)
    m.compute_geometry()
    return m


class TestSingleMesh:
    def test_single_mesh_returns_clone(self):
        m = _single_hex()
        r = merge_meshes([m])
        assert isinstance(r, FvMesh)
        assert r.n_cells == 1

    def test_single_preserves_points(self):
        m = _single_hex()
        r = merge_meshes([m])
        assert torch.allclose(r.points, m.points)

    def test_single_preserves_face_count(self):
        m = _single_hex()
        r = merge_meshes([m])
        assert r.n_faces == m.n_faces


class TestTwoMeshes:
    def test_two_disjoint_meshes(self):
        m1 = _single_hex((0, 0, 0))
        m2 = _single_hex((10, 0, 0))
        r = merge_meshes([m1, m2])
        assert r.n_cells == 2
        assert r.n_faces == 12  # 6 + 6

    def test_two_adjacent_merge_shared_face(self):
        """两个相邻 mesh 共享一个面，应产生 2 cells + 1 internal face。"""
        m1 = _single_hex((0, 0, 0))
        m2 = _single_hex((1, 0, 0))
        r = merge_meshes([m1, m2])
        assert r.n_cells == 2
        # 应该有 1 个 internal face（共享面）
        assert r.n_internal_faces >= 1

    def test_total_volume(self):
        m1 = _single_hex((0, 0, 0))
        m2 = _single_hex((10, 0, 0))
        r = merge_meshes([m1, m2])
        r.compute_geometry()
        assert abs(r.total_volume.item() - 2.0) < 1e-8


class TestThreeMeshes:
    def test_three_meshes(self):
        m1 = _single_hex((0, 0, 0))
        m2 = _single_hex((10, 0, 0))
        m3 = _single_hex((20, 0, 0))
        r = merge_meshes([m1, m2, m3])
        assert r.n_cells == 3


class TestEdgeCases:
    def test_empty_list(self):
        with pytest.raises(ValueError, match="empty"):
            merge_meshes([])

    def test_does_not_modify_original(self):
        m = _single_hex()
        nc_orig = m.n_cells
        merge_meshes([m])
        assert m.n_cells == nc_orig

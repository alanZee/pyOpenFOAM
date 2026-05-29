"""Tests for stitch_mesh — stitch two boundary patches together."""
from __future__ import annotations
import pytest, torch
from pyfoam.core.dtype import INDEX_DTYPE
from pyfoam.mesh.fv_mesh import FvMesh
from pyfoam.tools.stitch_mesh import stitch_mesh


def _two_hex_with_matching_patches():
    """两个相邻立方体，各有面对应 'left' 和 'right' patch。

    Cell 0: x=0..1, Cell 1: x=1..2
    共享面在 x=1 处。
    为了测试 stitch，不预先设置 shared internal face，而是用两个独立 mesh 的面。
    """
    # 使用一个有 2 cells 但 boundary 面可配对的 mesh
    pts = [[0,0,0],[1,0,0],[1,1,0],[0,1,0],  # 0-3
           [0,0,1],[1,0,1],[1,1,1],[0,1,1],  # 4-7
           [2,0,0],[2,1,0],[2,0,1],[2,1,1]]  # 8-11
    # Internal face: face at x=1 (between cell 0 and cell 1)
    # Cell 0: 0..1, Cell 1: 1..2
    faces = [
        [1,2,6,5],    # 0: internal face (x=1), owner=0, nbr=1
        [0,3,2,1],    # 1: bottom cell 0
        [0,1,5,4],    # 2: front cell 0
        [3,7,6,2],    # 3: back cell 0
        [0,4,7,3],    # 4: left cell 0 (x=0)
        [4,5,6,7],    # 5: top cell 0
        [5,8,11,6],   # 6: front cell 1  (y=0 face)
        [1,5,10,8],   # 7: bottom cell 1
        [6,11,10,9],  # 8: back cell 1
        [9,8,11,10],  # 9: right cell 1 (x=2)
        [2,9,10,6],   # 10: top cell 1
    ]
    owner = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    neighbour = [1]
    boundary = [
        {"name": "patch_bottom", "type": "wall", "startFace": 1, "nFaces": 5},
        {"name": "patch_top", "type": "wall", "startFace": 6, "nFaces": 5},
    ]
    m = FvMesh(points=torch.tensor(pts, dtype=torch.float64),
               faces=[torch.tensor(f, dtype=INDEX_DTYPE) for f in faces],
               owner=torch.tensor(owner, dtype=INDEX_DTYPE),
               neighbour=torch.tensor(neighbour, dtype=INDEX_DTYPE),
               boundary=boundary, validate=False)
    m.compute_geometry()
    return m


class TestStitchBasic:
    def test_stitch_returns_fv_mesh(self):
        m = _two_hex_with_matching_patches()
        # 没有匹配的 patch 对，应该返回类似原始 mesh
        r = stitch_mesh(m, "patch_bottom", "patch_top", tolerance=1e-6)
        assert isinstance(r, FvMesh)

    def test_invalid_patch_name(self):
        m = _two_hex_with_matching_patches()
        with pytest.raises(ValueError, match="not found"):
            stitch_mesh(m, "nonexistent", "patch_top")


class TestStitchGeometry:
    def test_preserves_cell_count(self):
        """stitch 不改变 cell 数量。"""
        m = _two_hex_with_matching_patches()
        r = stitch_mesh(m, "patch_bottom", "patch_top", tolerance=1e-6)
        assert r.n_cells == m.n_cells

    def test_preserves_volume(self):
        m = _two_hex_with_matching_patches()
        v0 = m.total_volume.item()
        r = stitch_mesh(m, "patch_bottom", "patch_top", tolerance=1e-6)
        r.compute_geometry()
        assert abs(r.total_volume.item() - v0) < 1e-8


class TestStitchWithCoincidentFaces:
    def test_matching_faces_stitched(self):
        """创建有两个匹配边界面的 mesh，验证 stitch 减少了 boundary faces。"""
        # 构造一个简单场景：2 cells, 有匹配的 boundary 对
        pts = [[0,0,0],[1,0,0],[1,1,0],[0,1,0],
               [0,0,1],[1,0,1],[1,1,1],[0,1,1]]
        # 两个 cell，不设 internal face，而是用两对面表示
        faces = [
            [0,1,2,3],  # 0: bottom
            [4,5,6,7],  # 1: top
            [0,1,5,4],  # 2: front cell0
            [3,7,6,2],  # 3: back cell0
            [0,4,7,3],  # 4: left cell0
            [1,2,6,5],  # 5: right cell0
            [0,1,5,4],  # 6: front cell1 (same geometry)
            [3,7,6,2],  # 7: back cell1
            [0,4,7,3],  # 8: left cell1
            [1,2,6,5],  # 9: right cell1
        ]
        owner = [0,0,0,0,0,0,1,1,1,1]
        neighbour = torch.tensor([], dtype=INDEX_DTYPE)
        boundary = [
            {"name": "patch_a", "type": "wall", "startFace": 0, "nFaces": 5},
            {"name": "patch_b", "type": "wall", "startFace": 5, "nFaces": 5},
        ]
        m = FvMesh(points=torch.tensor(pts, dtype=torch.float64),
                   faces=[torch.tensor(f, dtype=INDEX_DTYPE) for f in faces],
                   owner=torch.tensor(owner, dtype=INDEX_DTYPE),
                   neighbour=neighbour,
                   boundary=boundary, validate=False)
        m.compute_geometry()
        n_faces_before = m.n_faces
        r = stitch_mesh(m, "patch_a", "patch_b", tolerance=1e-6)
        # 应该减少了总面数（stitched 面变 internal）
        assert r.n_faces < n_faces_before

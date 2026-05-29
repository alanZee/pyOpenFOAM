"""Tests for create_baffles_enhanced_2 — enhanced baffle creation v2."""
from __future__ import annotations
import pytest
import torch
from pyfoam.core.dtype import INDEX_DTYPE
from pyfoam.mesh.fv_mesh import FvMesh
from pyfoam.tools.create_baffles_enhanced_2 import BaffleEnhanced2Result, create_baffles_enhanced_2


def _two_cell_mesh():
    """2-cell hex mesh with 1 internal face."""
    pts = [
        [0,0,0],[1,0,0],[1,1,0],[0,1,0],
        [0,0,1],[1,0,1],[1,1,1],[0,1,1],
        [2,0,0],[2,1,0],[2,0,1],[2,1,1],
    ]
    int_faces = [[1,2,6,5]]
    bnd_faces = [
        [0,3,2,1], [4,5,6,7], [0,1,5,4], [2,3,7,6], [0,4,7,3],
        [1,8,10,5], [6,7,11,10], [8,9,11,10],
    ]
    all_faces = [torch.tensor(f, dtype=INDEX_DTYPE) for f in int_faces + bnd_faces]
    owner = torch.tensor([0,0,0,0,0,0,1,1,1], dtype=INDEX_DTYPE)
    neighbour = torch.tensor([1], dtype=INDEX_DTYPE)
    boundary = [
        {"name": "bottom", "type": "wall", "startFace": 1, "nFaces": 1},
        {"name": "top", "type": "wall", "startFace": 2, "nFaces": 1},
        {"name": "front", "type": "wall", "startFace": 3, "nFaces": 1},
        {"name": "back", "type": "wall", "startFace": 4, "nFaces": 1},
        {"name": "left", "type": "wall", "startFace": 5, "nFaces": 1},
        {"name": "front1", "type": "wall", "startFace": 6, "nFaces": 1},
        {"name": "back1", "type": "wall", "startFace": 7, "nFaces": 1},
        {"name": "right", "type": "wall", "startFace": 8, "nFaces": 1},
    ]
    m = FvMesh(points=torch.tensor(pts, dtype=torch.float64),
               faces=all_faces, owner=owner, neighbour=neighbour,
               boundary=boundary, validate=False)
    m.compute_geometry()
    return m


class TestCreateBafflesEnhanced2:
    def test_returns_result_type(self):
        m = _two_cell_mesh()
        r = create_baffles_enhanced_2(m, face_indices=[0])
        assert isinstance(r, BaffleEnhanced2Result)

    def test_single_baffle(self):
        m = _two_cell_mesh()
        r = create_baffles_enhanced_2(m, face_indices=[0])
        assert r.n_baffles == 1

    def test_dual_patches(self):
        m = _two_cell_mesh()
        r = create_baffles_enhanced_2(m, face_indices=[0], dual_patches=True)
        assert len(r.baffle_patches) == 2

    def test_no_indices_raises(self):
        m = _two_cell_mesh()
        with pytest.raises(ValueError, match="One of"):
            create_baffles_enhanced_2(m)

    def test_n_filtered_default(self):
        m = _two_cell_mesh()
        r = create_baffles_enhanced_2(m, face_indices=[0])
        assert r.n_filtered == 0

    def test_min_area_filter(self):
        m = _two_cell_mesh()
        # Very large threshold should filter everything out
        r = create_baffles_enhanced_2(m, face_indices=[0], min_area=1e6)
        assert r.n_baffles == 0
        assert r.n_filtered > 0

    def test_source_patches(self):
        m = _two_cell_mesh()
        r = create_baffles_enhanced_2(m, source_patches=["left"])
        # left has 1 boundary face; baffles need internal faces, so this may be empty
        assert isinstance(r, BaffleEnhanced2Result)

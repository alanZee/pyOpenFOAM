"""Tests for create_patch_enhanced_2 — enhanced patch creation v2."""
from __future__ import annotations
import pytest
import torch
from pyfoam.core.dtype import INDEX_DTYPE
from pyfoam.mesh.fv_mesh import FvMesh
from pyfoam.tools.create_patch_enhanced_2 import PatchEnhanced2Result, create_patch_enhanced_2


def _simple_mesh():
    """2-cell hex mesh for patch creation tests."""
    pts = [
        [0,0,0],[1,0,0],[1,1,0],[0,1,0],
        [0,0,1],[1,0,1],[1,1,1],[0,1,1],
        [2,0,0],[2,1,0],[2,0,1],[2,1,1],
    ]
    int_faces = [[1,2,6,5]]
    bnd_faces = [
        [0,3,2,1],[4,5,6,7],[0,1,5,4],[2,3,7,6],[0,4,7,3],
        [1,8,10,5],[6,7,11,10],[8,9,11,10],
    ]
    all_f = [torch.tensor(f, dtype=INDEX_DTYPE) for f in int_faces + bnd_faces]
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
               faces=all_f, owner=owner, neighbour=neighbour,
               boundary=boundary, validate=False)
    m.compute_geometry()
    return m


class TestCreatePatchEnhanced2:
    def test_returns_result_type(self):
        m = _simple_mesh()
        r = create_patch_enhanced_2(m, face_indices=[1], patch_name="new")
        assert isinstance(r, PatchEnhanced2Result)

    def test_patches_created(self):
        m = _simple_mesh()
        r = create_patch_enhanced_2(m, face_indices=[1], patch_name="new")
        assert "new" in r.patches_created

    def test_n_faces_moved(self):
        m = _simple_mesh()
        r = create_patch_enhanced_2(m, face_indices=[1], patch_name="new")
        assert r.n_faces_moved >= 1

    def test_box_selection(self):
        m = _simple_mesh()
        r = create_patch_enhanced_2(
            m, box_min=(-1, -1, -1), box_max=(0.5, 0.5, 0.5),
            patch_name="box_patch",
        )
        assert isinstance(r, PatchEnhanced2Result)

    def test_normal_selection(self):
        m = _simple_mesh()
        r = create_patch_enhanced_2(
            m, normal_dir=(0, 0, 1), normal_tol=45.0,
            patch_name="normal_patch",
        )
        assert isinstance(r, PatchEnhanced2Result)

    def test_duplicate_name_raises(self):
        m = _simple_mesh()
        with pytest.raises(ValueError, match="already exists"):
            create_patch_enhanced_2(m, face_indices=[1], patch_name="bottom")

    def test_multi_patch(self):
        m = _simple_mesh()
        r = create_patch_enhanced_2(
            m,
            multi_patch=[([1], "patch_a", "wall"), ([2], "patch_b", "wall")],
        )
        assert len(r.patches_created) >= 1

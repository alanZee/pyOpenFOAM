"""Tests for create_patch_enhanced — enhanced patch creation."""
from __future__ import annotations
import pytest
import torch
from pyfoam.core.dtype import INDEX_DTYPE
from pyfoam.mesh.fv_mesh import FvMesh
from pyfoam.tools.create_patch_enhanced import PatchEnhancedResult, create_patch_enhanced


def _two_cell_mesh():
    pts = [[0,0,0],[1,0,0],[1,1,0],[0,1,0],
           [0,0,1],[1,0,1],[1,1,1],[0,1,1],
           [0,0,2],[1,0,2],[1,1,2],[0,1,2]]
    fc = [[4,5,6,7],[0,3,2,1],[0,1,5,4],[2,3,7,6],[0,4,7,3],[1,2,6,5],
          [8,9,10,11],[4,5,9,8],[7,11,10,6],[4,8,11,7],[5,6,10,9]]
    m = FvMesh(
        points=torch.tensor(pts, dtype=torch.float64),
        faces=[torch.tensor(f, dtype=INDEX_DTYPE) for f in fc],
        owner=torch.tensor([0,0,0,0,0,0,1,1,1,1,1], dtype=INDEX_DTYPE),
        neighbour=torch.tensor([1], dtype=INDEX_DTYPE),
        boundary=[{"name":"bottom","type":"wall","startFace":1,"nFaces":5},
                  {"name":"top","type":"wall","startFace":6,"nFaces":5}],
        validate=False,
    )
    m.compute_geometry()
    return m


class TestCreatePatchEnhanced:
    def test_returns_result_type(self):
        m = _two_cell_mesh()
        r = create_patch_enhanced(m, face_indices=[0], patch_name="new_p")
        assert isinstance(r, PatchEnhancedResult)

    def test_creates_new_patch(self):
        m = _two_cell_mesh()
        r = create_patch_enhanced(m, face_indices=[0], patch_name="new_p")
        assert "new_p" in r.patches_created
        assert r.n_faces_moved >= 1

    def test_duplicate_name_raises(self):
        m = _two_cell_mesh()
        with pytest.raises(ValueError, match="already exists"):
            create_patch_enhanced(m, face_indices=[0], patch_name="bottom")

    def test_cell_based_selection(self):
        m = _two_cell_mesh()
        r = create_patch_enhanced(m, cells=[0], patch_name="cell_faces")
        assert r.n_faces_moved >= 1

    def test_source_patch_selection(self):
        m = _two_cell_mesh()
        r = create_patch_enhanced(m, source_patches=["top"], patch_name="moved_top")
        assert r.n_faces_moved >= 1
        assert "moved_top" in r.patches_created

    def test_multi_patch_mode(self):
        m = _two_cell_mesh()
        r = create_patch_enhanced(m, multi_patch=[
            ([0], "patch_a", "wall"),
            ([1], "patch_b", "wall"),
        ])
        # Face 1 is boundary, face 0 is internal — both should work
        assert len(r.patches_created) >= 1

    def test_result_mesh_has_new_patch(self):
        m = _two_cell_mesh()
        r = create_patch_enhanced(m, face_indices=[0], patch_name="inlet")
        names = {p["name"] for p in r.mesh.boundary}
        assert "inlet" in names

    def test_custom_type(self):
        m = _two_cell_mesh()
        r = create_patch_enhanced(m, face_indices=[0], patch_name="inlet", patch_type="patch")
        for p in r.mesh.boundary:
            if p["name"] == "inlet":
                assert p["type"] == "patch"

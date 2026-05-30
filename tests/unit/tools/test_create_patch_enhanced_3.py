"""Tests for create_patch_enhanced_3 — enhanced patch creation v3."""
from __future__ import annotations
import pytest
import torch
from pyfoam.core.dtype import INDEX_DTYPE
from pyfoam.mesh.fv_mesh import FvMesh
from pyfoam.tools.create_patch_enhanced_3 import PatchEnhanced3Result, create_patch_enhanced_3


def _single_hex_with_boundary():
    pts = [
        [0,0,0],[1,0,0],[1,1,0],[0,1,0],
        [0,0,1],[1,0,1],[1,1,1],[0,1,1],
    ]
    fc = [[0,3,2,1],[4,5,6,7],[0,1,5,4],[2,3,7,6],[0,4,7,3],[1,2,6,5]]
    m = FvMesh(points=torch.tensor(pts, dtype=torch.float64),
               faces=[torch.tensor(f, dtype=INDEX_DTYPE) for f in fc],
               owner=torch.zeros(6, dtype=INDEX_DTYPE),
               neighbour=torch.tensor([], dtype=INDEX_DTYPE),
               boundary=[
                   {"name": "bottom", "type": "wall", "startFace": 0, "nFaces": 1},
                   {"name": "top", "type": "wall", "startFace": 1, "nFaces": 1},
                   {"name": "front", "type": "wall", "startFace": 2, "nFaces": 1},
                   {"name": "back", "type": "wall", "startFace": 3, "nFaces": 1},
                   {"name": "left", "type": "wall", "startFace": 4, "nFaces": 1},
                   {"name": "right", "type": "wall", "startFace": 5, "nFaces": 1},
               ],
               validate=False)
    m.compute_geometry()
    return m


class TestCreatePatchEnhanced3:
    def test_returns_result_type(self):
        m = _single_hex_with_boundary()
        r = create_patch_enhanced_3(m, face_indices=[0], patch_name="new")
        assert isinstance(r, PatchEnhanced3Result)

    def test_patch_created(self):
        m = _single_hex_with_boundary()
        r = create_patch_enhanced_3(m, face_indices=[0], patch_name="new")
        assert "new" in r.patches_created
        assert r.n_faces_moved > 0

    def test_patch_face_counts(self):
        m = _single_hex_with_boundary()
        r = create_patch_enhanced_3(m, face_indices=[0], patch_name="new")
        assert "new" in r.patch_face_counts
        assert r.patch_face_counts["new"] == 1

    def test_patch_areas_reported(self):
        m = _single_hex_with_boundary()
        r = create_patch_enhanced_3(m, face_indices=[0], patch_name="new")
        assert "new" in r.patch_areas
        assert r.patch_areas["new"] > 0

    def test_plane_selection(self):
        m = _single_hex_with_boundary()
        r = create_patch_enhanced_3(
            m, patch_name="plane_sel",
            plane_point=(0.5, 0.5, 0.5),
            plane_normal=(0, 0, 1),
            plane_distance=0.6,
        )
        assert isinstance(r, PatchEnhanced3Result)

    def test_name_conflict_raises(self):
        m = _single_hex_with_boundary()
        with pytest.raises(ValueError, match="already exists"):
            create_patch_enhanced_3(m, face_indices=[0], patch_name="bottom")

"""Tests for create_patch_enhanced_5 — enhanced patch creation v5."""
from __future__ import annotations
import pytest
import torch
from pyfoam.core.dtype import INDEX_DTYPE
from pyfoam.mesh.fv_mesh import FvMesh
from pyfoam.tools.create_patch_enhanced_5 import PatchEnhanced5Result, create_patch_enhanced_5


def _single_hex():
    pts = [[0,0,0],[1,0,0],[1,1,0],[0,1,0],
           [0,0,1],[1,0,1],[1,1,1],[0,1,1]]
    fc = [[0,3,2,1],[4,5,6,7],[0,1,5,4],[2,3,7,6],[0,4,7,3],[1,2,6,5]]
    m = FvMesh(points=torch.tensor(pts, dtype=torch.float64),
               faces=[torch.tensor(f, dtype=INDEX_DTYPE) for f in fc],
               owner=torch.zeros(6, dtype=INDEX_DTYPE),
               neighbour=torch.tensor([], dtype=INDEX_DTYPE),
               boundary=[
                   {"name": "bottom", "type": "wall", "startFace": 0, "nFaces": 1},
                   {"name": "top", "type": "wall", "startFace": 1, "nFaces": 1},
                   {"name": "rest", "type": "wall", "startFace": 2, "nFaces": 4},
               ],
               validate=False)
    m.compute_geometry()
    return m


class TestPatchEnhanced5:
    def test_returns_result_type(self):
        m = _single_hex()
        r = create_patch_enhanced_5(m, face_indices=[0], patch_name="new")
        assert isinstance(r, PatchEnhanced5Result)

    def test_parent_patch(self):
        m = _single_hex()
        r = create_patch_enhanced_5(
            m, face_indices=[0], patch_name="child",
            parent_patch="bottom",
        )
        assert "child" in r.patch_hierarchy
        assert r.patch_hierarchy["child"] == "bottom"

    def test_auto_renumber(self):
        m = _single_hex()
        r = create_patch_enhanced_5(
            m, face_indices=[0], patch_name="new",
            auto_renumber=True,
        )
        assert isinstance(r.renumbered, bool)

    def test_boolean_expression(self):
        m = _single_hex()
        r = create_patch_enhanced_5(
            m, patch_name="bool_patch",
            boolean_expression="box",
            box_min=(0, 0, 0), box_max=(2, 2, 2),
        )
        assert isinstance(r, PatchEnhanced5Result)

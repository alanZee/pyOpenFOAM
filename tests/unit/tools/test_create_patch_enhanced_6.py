"""Tests for create_patch_enhanced_6 — enhanced patch creation v6."""
from __future__ import annotations
import pytest
import torch
from pyfoam.core.dtype import INDEX_DTYPE
from pyfoam.mesh.fv_mesh import FvMesh
from pyfoam.tools.create_patch_enhanced_6 import PatchEnhanced6Result, create_patch_enhanced_6


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


class TestPatchEnhanced6:
    def test_returns_result_type(self):
        m = _single_hex()
        r = create_patch_enhanced_6(m, face_indices=[0], patch_name="new")
        assert isinstance(r, PatchEnhanced6Result)

    def test_template_wall(self):
        m = _single_hex()
        r = create_patch_enhanced_6(
            m, face_indices=[0], patch_name="wall_patch",
            template="wall",
        )
        assert r.template_used == "wall"

    def test_template_inlet(self):
        m = _single_hex()
        r = create_patch_enhanced_6(
            m, face_indices=[0], patch_name="inlet",
            template="inlet",
        )
        assert r.template_used == "inlet"

    def test_template_outlet(self):
        m = _single_hex()
        r = create_patch_enhanced_6(
            m, face_indices=[0], patch_name="outlet",
            template="outlet",
        )
        assert r.template_used == "outlet"

    def test_unknown_template_ignored(self):
        m = _single_hex()
        r = create_patch_enhanced_6(
            m, face_indices=[0], patch_name="new",
            template="nonexistent",
        )
        assert r.template_used is None

    def test_conflict_detection(self):
        m = _single_hex()
        r = create_patch_enhanced_6(
            m, face_indices=[0], patch_name="new",
            detect_conflicts=True,
        )
        assert r.n_conflicts >= 0

    def test_undo_available(self):
        m = _single_hex()
        r = create_patch_enhanced_6(
            m, face_indices=[0], patch_name="new",
            enable_undo=True,
        )
        assert r.undo_available is True

    def test_undo_not_available(self):
        m = _single_hex()
        r = create_patch_enhanced_6(
            m, face_indices=[0], patch_name="new",
            enable_undo=False,
        )
        assert r.undo_available is False

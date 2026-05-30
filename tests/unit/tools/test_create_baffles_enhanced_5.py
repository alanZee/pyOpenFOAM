"""Tests for create_baffles_enhanced_5 — enhanced baffle creation v5."""
from __future__ import annotations
import pytest
import torch
from pyfoam.core.dtype import INDEX_DTYPE
from pyfoam.mesh.fv_mesh import FvMesh
from pyfoam.tools.create_baffles_enhanced_5 import (
    BaffleEnhanced5Result, BaffleNetwork, create_baffles_enhanced_5,
)


def _two_cell_hex():
    """Two hex cells sharing one internal face."""
    pts = [[0,0,0],[1,0,0],[1,1,0],[0,1,0],
           [0,0,1],[1,0,1],[1,1,1],[0,1,1],
           [2,0,0],[2,1,0],[2,0,1],[2,1,1]]
    # 9 faces: 1 internal + 8 boundary
    faces = [
        [1,2,6,5],      # 0: internal (between cell 0 and cell 1)
        [0,3,2,1],      # 1: bottom cell0
        [4,5,6,7],      # 2: top cell0
        [0,1,5,4],      # 3: front cell0
        [3,7,6,2],      # 4: back cell0
        [0,4,7,3],      # 5: left cell0
        [8,9,11,10],    # 6: bottom-right
        [1,5,6,2],      # already used -- use: bottom cell1
        [8,10,11,9],    # 8: top cell1
    ]
    # Use simpler: 1 internal + 10 boundary faces
    all_faces = [
        [1,2,6,5],       # face 0: internal
        [0,3,2,1],       # face 1: boundary
        [4,5,6,7],       # face 2: boundary
        [0,1,5,4],       # face 3: boundary
        [3,7,6,2],       # face 4: boundary
        [0,4,7,3],       # face 5: boundary
        [8,10,11,9],     # face 6: boundary
        [1,5,10,8],      # face 7: boundary
        [2,9,11,6],      # face 8: boundary
        [8,9,11,10],     # face 9: boundary (was wrong -- fix)
        [1,8,9,2],       # face 10: boundary (front cell1)
    ]
    # Simpler approach: just use the conftest 2-cell fixture approach
    pts2 = torch.tensor([
        [0,0,0],[1,0,0],[1,1,0],[0,1,0],
        [0,0,1],[1,0,1],[1,1,1],[0,1,1],
        [2,0,0],[2,1,0],[2,0,1],[2,1,1],
    ], dtype=torch.float64)
    faces2 = [
        torch.tensor([1,2,6,5], dtype=INDEX_DTYPE),    # 0: internal
        torch.tensor([0,3,2,1], dtype=INDEX_DTYPE),    # 1
        torch.tensor([4,5,6,7], dtype=INDEX_DTYPE),    # 2
        torch.tensor([0,1,5,4], dtype=INDEX_DTYPE),    # 3
        torch.tensor([3,7,6,2], dtype=INDEX_DTYPE),    # 4
        torch.tensor([0,4,7,3], dtype=INDEX_DTYPE),    # 5
        torch.tensor([1,5,10,8], dtype=INDEX_DTYPE),   # 6
        torch.tensor([2,9,11,6], dtype=INDEX_DTYPE),   # 7
        torch.tensor([1,8,9,2], dtype=INDEX_DTYPE),    # 8
        torch.tensor([8,10,11,9], dtype=INDEX_DTYPE),  # 9
    ]
    owner2 = torch.tensor([0,0,0,0,0,0,1,1,1,1], dtype=INDEX_DTYPE)
    neighbour2 = torch.tensor([1], dtype=INDEX_DTYPE)
    boundary2 = [
        {"name": "bottom", "type": "wall", "startFace": 1, "nFaces": 1},
        {"name": "top", "type": "wall", "startFace": 2, "nFaces": 1},
        {"name": "front0", "type": "wall", "startFace": 3, "nFaces": 1},
        {"name": "back0", "type": "wall", "startFace": 4, "nFaces": 1},
        {"name": "left", "type": "wall", "startFace": 5, "nFaces": 1},
        {"name": "front1", "type": "wall", "startFace": 6, "nFaces": 1},
        {"name": "back1", "type": "wall", "startFace": 7, "nFaces": 1},
        {"name": "face1", "type": "wall", "startFace": 8, "nFaces": 1},
        {"name": "right", "type": "wall", "startFace": 9, "nFaces": 1},
    ]
    m = FvMesh(points=pts2, faces=faces2, owner=owner2,
               neighbour=neighbour2, boundary=boundary2, validate=False)
    m.compute_geometry()
    return m


class TestBaffleEnhanced5:
    def test_returns_result_type(self):
        m = _two_cell_hex()
        r = create_baffles_enhanced_5(m, face_indices=[0])
        assert isinstance(r, BaffleEnhanced5Result)

    def test_network_analysis(self):
        m = _two_cell_hex()
        r = create_baffles_enhanced_5(m, face_indices=[0], analyze_networks=True)
        assert isinstance(r.networks, list)

    def test_flow_resistance(self):
        m = _two_cell_hex()
        r = create_baffles_enhanced_5(
            m, face_indices=[0],
            flow_resistance=True, porosity=0.3,
        )
        assert r.porosity == 0.3

    def test_quality_degradation(self):
        m = _two_cell_hex()
        r = create_baffles_enhanced_5(m, face_indices=[0])
        assert 0.0 <= r.quality_degradation <= 1.0

    def test_no_face_indices_raises(self):
        m = _two_cell_hex()
        with pytest.raises(ValueError):
            create_baffles_enhanced_5(m)

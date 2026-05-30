"""Tests for create_baffles_enhanced_4 — enhanced baffle creation v4."""
from __future__ import annotations
import pytest
import torch
from pyfoam.core.dtype import INDEX_DTYPE
from pyfoam.mesh.fv_mesh import FvMesh
from pyfoam.tools.create_baffles_enhanced_4 import BaffleEnhanced4Result, create_baffles_enhanced_4


def _two_cell_mesh():
    """Two cells sharing one internal face."""
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


class TestBaffleEnhanced4:
    def test_returns_result_type(self):
        m = _two_cell_mesh()
        r = create_baffles_enhanced_4(m, face_indices=[0])
        assert isinstance(r, BaffleEnhanced4Result)

    def test_porosity_stored(self):
        m = _two_cell_mesh()
        r = create_baffles_enhanced_4(m, face_indices=[0], porosity=0.3)
        assert r.porosity == 0.3

    def test_pressure_drop_stored(self):
        m = _two_cell_mesh()
        r = create_baffles_enhanced_4(m, face_indices=[0], pressure_drop_coefficient=1.5)
        assert r.pressure_drop_coefficient == 1.5

    def test_thermal_resistance_stored(self):
        m = _two_cell_mesh()
        r = create_baffles_enhanced_4(m, face_indices=[0], thermal_resistance=0.1)
        assert r.thermal_resistance == 0.1

    def test_auto_thickness(self):
        m = _two_cell_mesh()
        r = create_baffles_enhanced_4(m, face_indices=[0], auto_thickness=True)
        assert r.mean_thickness > 0

    def test_no_face_indices_raises(self):
        m = _two_cell_mesh()
        with pytest.raises(ValueError, match="must be provided"):
            create_baffles_enhanced_4(m)

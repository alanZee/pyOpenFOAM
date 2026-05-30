"""Tests for create_baffles_enhanced_6 — enhanced baffle creation v6."""
from __future__ import annotations
import pytest
import torch
from pyfoam.core.dtype import INDEX_DTYPE
from pyfoam.mesh.fv_mesh import FvMesh
from pyfoam.tools.create_baffles_enhanced_6 import (
    BaffleEnhanced6Result, BaffleScheduleEntry, create_baffles_enhanced_6,
)


def _two_cell_hex():
    pts2 = torch.tensor([
        [0,0,0],[1,0,0],[1,1,0],[0,1,0],
        [0,0,1],[1,0,1],[1,1,1],[0,1,1],
        [2,0,0],[2,1,0],[2,0,1],[2,1,1],
    ], dtype=torch.float64)
    faces2 = [
        torch.tensor([1,2,6,5], dtype=INDEX_DTYPE),
        torch.tensor([0,3,2,1], dtype=INDEX_DTYPE),
        torch.tensor([4,5,6,7], dtype=INDEX_DTYPE),
        torch.tensor([0,1,5,4], dtype=INDEX_DTYPE),
        torch.tensor([3,7,6,2], dtype=INDEX_DTYPE),
        torch.tensor([0,4,7,3], dtype=INDEX_DTYPE),
        torch.tensor([1,5,10,8], dtype=INDEX_DTYPE),
        torch.tensor([2,9,11,6], dtype=INDEX_DTYPE),
        torch.tensor([1,8,9,2], dtype=INDEX_DTYPE),
        torch.tensor([8,10,11,9], dtype=INDEX_DTYPE),
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


class TestBaffleEnhanced6:
    def test_returns_result_type(self):
        m = _two_cell_hex()
        r = create_baffles_enhanced_6(m, face_indices=[0])
        assert isinstance(r, BaffleEnhanced6Result)

    def test_thermal_conductance(self):
        m = _two_cell_hex()
        r = create_baffles_enhanced_6(
            m, face_indices=[0], thermal_conductance=5.0,
        )
        assert r.thermal_conductance >= 0

    def test_optimize_placement(self):
        m = _two_cell_hex()
        r = create_baffles_enhanced_6(
            m, face_indices=[0], optimize_placement=True,
        )
        assert r.n_optimised >= 0

    def test_time_schedule(self):
        m = _two_cell_hex()
        r = create_baffles_enhanced_6(
            m, face_indices=[0],
            time_schedule=[(0.0, 0.3, 1.0), (1.0, 0.5, 0.5)],
        )
        assert len(r.schedule) == 2
        assert isinstance(r.schedule[0], BaffleScheduleEntry)

    def test_dict_snippet_with_conductance(self):
        m = _two_cell_hex()
        r = create_baffles_enhanced_6(
            m, face_indices=[0],
            thermal_conductance=5.0, thermal_resistance=0.1,
        )
        assert isinstance(r.dict_snippet, str)

    def test_quality_degradation(self):
        m = _two_cell_hex()
        r = create_baffles_enhanced_6(m, face_indices=[0])
        assert 0.0 <= r.quality_degradation <= 1.0

    def test_no_face_indices_raises(self):
        m = _two_cell_hex()
        with pytest.raises(ValueError):
            create_baffles_enhanced_6(m)

"""Tests for create_baffles_enhanced_4 — enhanced baffle creation v4."""
from __future__ import annotations
import pytest
import torch
from pyfoam.core.dtype import INDEX_DTYPE
from pyfoam.mesh.fv_mesh import FvMesh
from pyfoam.tools.create_baffles_enhanced_4 import BaffleEnhanced4Result, create_baffles_enhanced_4


def _single_hex():
    pts = [[0,0,0],[1,0,0],[1,1,0],[0,1,0],
           [0,0,1],[1,0,1],[1,1,1],[0,1,1]]
    fc = [[0,3,2,1],[4,5,6,7],[0,1,5,4],[2,3,7,6],[0,4,7,3],[1,2,6,5]]
    m = FvMesh(points=torch.tensor(pts, dtype=torch.float64),
               faces=[torch.tensor(f, dtype=INDEX_DTYPE) for f in fc],
               owner=torch.zeros(6, dtype=INDEX_DTYPE),
               neighbour=torch.tensor([], dtype=INDEX_DTYPE),
               boundary=[{"name": "all", "type": "wall", "startFace": 0, "nFaces": 6}],
               validate=False)
    m.compute_geometry()
    return m


class TestBaffleEnhanced4:
    def test_returns_result_type(self):
        m = _single_hex()
        r = create_baffles_enhanced_4(m, face_indices=[0])
        assert isinstance(r, BaffleEnhanced4Result)

    def test_porosity_stored(self):
        m = _single_hex()
        r = create_baffles_enhanced_4(m, face_indices=[0], porosity=0.3)
        assert r.porosity == 0.3

    def test_pressure_drop_stored(self):
        m = _single_hex()
        r = create_baffles_enhanced_4(m, face_indices=[0], pressure_drop_coefficient=1.5)
        assert r.pressure_drop_coefficient == 1.5

    def test_thermal_resistance_stored(self):
        m = _single_hex()
        r = create_baffles_enhanced_4(m, face_indices=[0], thermal_resistance=0.1)
        assert r.thermal_resistance == 0.1

    def test_auto_thickness(self):
        m = _single_hex()
        r = create_baffles_enhanced_4(m, face_indices=[0], auto_thickness=True)
        assert isinstance(r.mean_thickness, float)

    def test_no_face_indices_raises(self):
        m = _single_hex()
        with pytest.raises(ValueError, match="must be provided"):
            create_baffles_enhanced_4(m)

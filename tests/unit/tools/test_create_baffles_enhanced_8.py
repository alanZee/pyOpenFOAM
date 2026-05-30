"""Tests for create_baffles_enhanced_8 — enhanced baffle creation v8."""
from __future__ import annotations
import pytest
import torch
from pyfoam.core.dtype import INDEX_DTYPE
from pyfoam.mesh.fv_mesh import FvMesh
from pyfoam.tools.create_baffles_enhanced_8 import (
    BaffleEnhanced8Result, PermeabilityCoefficients, ThermalCouplingValidation,
    create_baffles_enhanced_8,
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


class TestBaffleEnhanced8:
    def test_returns_result_type(self):
        m = _two_cell_hex()
        r = create_baffles_enhanced_8(m, face_indices=[0])
        assert isinstance(r, BaffleEnhanced8Result)

    def test_permeability(self):
        m = _two_cell_hex()
        r = create_baffles_enhanced_8(m, face_indices=[0], permeability_model=True, porosity=0.5)
        assert isinstance(r.permeability, PermeabilityCoefficients)
        assert r.permeability.porosity == 0.5

    def test_thermal_validation(self):
        m = _two_cell_hex()
        r = create_baffles_enhanced_8(m, face_indices=[0], validate_thermal_coupling=True)
        assert isinstance(r.thermal_validation, ThermalCouplingValidation)

    def test_optimisation_feedback(self):
        m = _two_cell_hex()
        r = create_baffles_enhanced_8(m, face_indices=[0], compute_feedback=True)
        assert r.optimisation_feedback.effectiveness_score >= 0

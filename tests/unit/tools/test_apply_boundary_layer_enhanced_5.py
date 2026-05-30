"""Tests for apply_boundary_layer_enhanced_5 — enhanced BL application v5."""
from __future__ import annotations
import numpy as np
import pytest
import torch
from pyfoam.core.dtype import INDEX_DTYPE
from pyfoam.mesh.fv_mesh import FvMesh
from pyfoam.tools.apply_boundary_layer_enhanced_5 import (
    EnhancedBL5Properties, EnhancedBL5Result, apply_boundary_layer_enhanced_5,
)


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


class TestApplyBoundaryLayerEnhanced5:
    def test_returns_result_type(self):
        m = _single_hex()
        velocity = np.ones((1, 3)) * 10.0
        bl = EnhancedBL5Properties(delta=0.5, nu=1e-5)
        r = apply_boundary_layer_enhanced_5(m, velocity, bl)
        assert isinstance(r, EnhancedBL5Result)

    def test_skin_friction(self):
        m = _single_hex()
        velocity = np.ones((1, 3)) * 10.0
        bl = EnhancedBL5Properties(delta=0.5, nu=1e-5)
        r = apply_boundary_layer_enhanced_5(m, velocity, bl)
        assert isinstance(r.skin_friction_coefficient, float)

    def test_transition_model(self):
        m = _single_hex()
        velocity = np.ones((1, 3)) * 10.0
        bl = EnhancedBL5Properties(
            delta=0.5, nu=1e-5,
            transition_model=True, Re_x_transition=1e5,
        )
        r = apply_boundary_layer_enhanced_5(m, velocity, bl)
        assert isinstance(r.n_transition_cells, int)

    def test_separation_detection(self):
        m = _single_hex()
        velocity = np.ones((1, 3)) * 10.0
        bl = EnhancedBL5Properties(delta=0.5, nu=1e-5, detect_separation=True)
        r = apply_boundary_layer_enhanced_5(m, velocity, bl)
        assert isinstance(r.n_separation_cells, int)

"""Tests for apply_boundary_layer_enhanced_4 — enhanced BL application v4."""
from __future__ import annotations
import numpy as np
import pytest
import torch
from pyfoam.core.dtype import INDEX_DTYPE
from pyfoam.mesh.fv_mesh import FvMesh
from pyfoam.tools.apply_boundary_layer_enhanced_4 import (
    EnhancedBL4Properties,
    EnhancedBL4Result,
    apply_boundary_layer_enhanced_4,
)


def _hex_mesh():
    pts = [[0,0,0],[1,0,0],[1,1,0],[0,1,0],
           [0,0,1],[1,0,1],[1,1,1],[0,1,1]]
    fc = [[0,3,2,1],[4,5,6,7],[0,1,5,4],[2,3,7,6],[0,4,7,3],[1,2,6,5]]
    m = FvMesh(points=torch.tensor(pts, dtype=torch.float64),
               faces=[torch.tensor(f, dtype=INDEX_DTYPE) for f in fc],
               owner=torch.zeros(6, dtype=INDEX_DTYPE),
               neighbour=torch.tensor([], dtype=INDEX_DTYPE),
               boundary=[{"name": "wall", "type": "wall", "startFace": 0, "nFaces": 6}],
               validate=False)
    m.compute_geometry()
    return m


class TestBLEnhanced4:
    def test_returns_result_type(self):
        m = _hex_mesh()
        U = np.ones((1, 3)) * 10.0
        bl = EnhancedBL4Properties(delta=0.5, nu=1e-5)
        r = apply_boundary_layer_enhanced_4(m, U, bl)
        assert isinstance(r, EnhancedBL4Result)

    def test_temperature_field(self):
        m = _hex_mesh()
        U = np.ones((1, 3)) * 10.0
        bl = EnhancedBL4Properties(delta=0.5, nu=1e-5, T_wall=310.0, T_inf=300.0)
        r = apply_boundary_layer_enhanced_4(m, U, bl)
        assert r.temperature is not None
        assert r.temperature.shape == (1,)

    def test_wall_heat_flux(self):
        m = _hex_mesh()
        U = np.ones((1, 3)) * 10.0
        bl = EnhancedBL4Properties(delta=0.5, nu=1e-5, T_wall=310.0, T_inf=300.0)
        r = apply_boundary_layer_enhanced_4(m, U, bl)
        assert isinstance(r.wall_heat_flux, float)

    def test_nusselt_number(self):
        m = _hex_mesh()
        U = np.ones((1, 3)) * 10.0
        bl = EnhancedBL4Properties(delta=0.5, nu=1e-5, T_wall=310.0, T_inf=300.0)
        r = apply_boundary_layer_enhanced_4(m, U, bl)
        assert isinstance(r.nusselt_number, float)

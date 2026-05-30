"""Tests for set_atm_boundary_layer_enhanced_4 — enhanced ABL v4."""
from __future__ import annotations
import numpy as np
import pytest
import torch
from pyfoam.core.dtype import INDEX_DTYPE
from pyfoam.mesh.fv_mesh import FvMesh
from pyfoam.tools.set_atm_boundary_layer_enhanced_4 import (
    EnhancedABL4Properties,
    EnhancedABL4Result,
    set_atm_boundary_layer_enhanced_4,
)


def _hex_mesh():
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


class TestABLenhanced4:
    def test_returns_result_type(self):
        m = _hex_mesh()
        abl = EnhancedABL4Properties(u_star=0.5, z0=0.01)
        r = set_atm_boundary_layer_enhanced_4(m, abl)
        assert isinstance(r, EnhancedABL4Result)

    def test_temperature_field(self):
        m = _hex_mesh()
        abl = EnhancedABL4Properties(u_star=0.5, z0=0.01, surface_temperature=300.0)
        r = set_atm_boundary_layer_enhanced_4(m, abl)
        assert r.temperature.shape == (1,)

    def test_mixing_length(self):
        m = _hex_mesh()
        abl = EnhancedABL4Properties(u_star=0.5, z0=0.01)
        r = set_atm_boundary_layer_enhanced_4(m, abl)
        assert r.mixing_length.shape == (1,)

    def test_canopy_model(self):
        m = _hex_mesh()
        abl = EnhancedABL4Properties(
            u_star=0.5, z0=0.01, canopy_height=0.5, canopy_drag_coefficient=0.3,
        )
        r = set_atm_boundary_layer_enhanced_4(m, abl)
        assert r.canopy_top_height == 0.5

    def test_bulk_richardson_number(self):
        m = _hex_mesh()
        abl = EnhancedABL4Properties(u_star=0.5, z0=0.01, surface_temperature=300.0)
        r = set_atm_boundary_layer_enhanced_4(m, abl)
        assert isinstance(r.bulk_richardson_number, float)

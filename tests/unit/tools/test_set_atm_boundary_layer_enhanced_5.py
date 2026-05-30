"""Tests for set_atm_boundary_layer_enhanced_5 — enhanced ABL v5."""
from __future__ import annotations
import pytest
import torch
from pyfoam.core.dtype import INDEX_DTYPE
from pyfoam.mesh.fv_mesh import FvMesh
from pyfoam.tools.set_atm_boundary_layer_enhanced_5 import (
    EnhancedABL5Properties, EnhancedABL5Result, set_atm_boundary_layer_enhanced_5,
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


class TestSetAtmBoundaryLayerEnhanced5:
    def test_returns_result_type(self):
        m = _single_hex()
        abl = EnhancedABL5Properties(u_star=0.5, z0=0.01, model="neutral")
        r = set_atm_boundary_layer_enhanced_5(m, abl)
        assert isinstance(r, EnhancedABL5Result)

    def test_roughness_sublayer(self):
        m = _single_hex()
        abl = EnhancedABL5Properties(
            u_star=0.5, z0=0.01, model="neutral",
            roughness_sublayer=True,
        )
        r = set_atm_boundary_layer_enhanced_5(m, abl)
        assert r.roughness_sublayer_correction >= 0

    def test_spectral_von_karman(self):
        m = _single_hex()
        abl = EnhancedABL5Properties(
            u_star=0.5, z0=0.01, model="neutral",
            spectral_model="von_karman",
            turbulence_length_scale=50.0,
        )
        r = set_atm_boundary_layer_enhanced_5(m, abl)
        assert r.spectral_coefficients is not None
        assert r.spectral_coefficients.shape[0] == 1  # 1 cell
        assert r.spectral_coefficients.shape[2] == 3  # 3 components

    def test_spectral_none(self):
        m = _single_hex()
        abl = EnhancedABL5Properties(u_star=0.5, z0=0.01, model="neutral")
        r = set_atm_boundary_layer_enhanced_5(m, abl)
        assert r.spectral_coefficients is None

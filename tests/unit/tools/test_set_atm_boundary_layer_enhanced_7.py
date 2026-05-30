"""Tests for set_atm_boundary_layer_enhanced_7 — enhanced ABL v7."""
from __future__ import annotations
import pytest
import torch
import numpy as np
from pyfoam.core.dtype import INDEX_DTYPE
from pyfoam.mesh.fv_mesh import FvMesh
from pyfoam.tools.set_atm_boundary_layer_enhanced_7 import EnhancedABL7Properties, EnhancedABL7Result, set_atm_boundary_layer_enhanced_7


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


class TestABL7:
    def test_returns_result_type(self):
        m = _single_hex()
        abl = EnhancedABL7Properties(u_star=0.5, z0=0.01)
        r = set_atm_boundary_layer_enhanced_7(m, abl)
        assert isinstance(r, EnhancedABL7Result)

    def test_pollution_dispersion(self):
        m = _single_hex()
        abl = EnhancedABL7Properties(u_star=0.5, z0=0.01, pollution_dispersion=True)
        r = set_atm_boundary_layer_enhanced_7(m, abl)
        assert r.schmidt_number_used == 0.7
        assert r.n_dispersion_cells >= 0

    def test_urban_canopy(self):
        m = _single_hex()
        abl = EnhancedABL7Properties(
            u_star=0.5, z0=0.01,
            urban_canopy=True, building_density=0.3, building_height=20.0,
        )
        r = set_atm_boundary_layer_enhanced_7(m, abl)
        assert r.canopy_drag_cells >= 0

    def test_stability_diagnostics(self):
        m = _single_hex()
        abl = EnhancedABL7Properties(u_star=0.5, z0=0.01, surface_heat_flux=0.01)
        r = set_atm_boundary_layer_enhanced_7(m, abl)
        assert r.stability_regime in ("stable", "neutral", "unstable")
        assert isinstance(r.obukhov_length, float)

    def test_neutral_stability(self):
        m = _single_hex()
        abl = EnhancedABL7Properties(u_star=0.5, z0=0.01, surface_heat_flux=0.0)
        r = set_atm_boundary_layer_enhanced_7(m, abl)
        assert r.stability_regime == "neutral"

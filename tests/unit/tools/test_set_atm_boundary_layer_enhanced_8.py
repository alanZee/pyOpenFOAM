"""Tests for set_atm_boundary_layer_enhanced_8 — enhanced ABL v8."""
from __future__ import annotations
import pytest
import torch
import numpy as np
from pyfoam.core.dtype import INDEX_DTYPE
from pyfoam.mesh.fv_mesh import FvMesh
from pyfoam.tools.set_atm_boundary_layer_enhanced_8 import (
    EnhancedABL8Properties, EnhancedABL8Result, WindEnergyMetrics, SiteClassification,
    set_atm_boundary_layer_enhanced_8,
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


class TestABL8:
    def test_returns_result_type(self):
        m = _single_hex()
        abl = EnhancedABL8Properties(u_star=0.5, z0=0.01)
        r = set_atm_boundary_layer_enhanced_8(m, abl)
        assert isinstance(r, EnhancedABL8Result)

    def test_co2_dispersion(self):
        m = _single_hex()
        abl = EnhancedABL8Properties(
            u_star=0.5, z0=0.01,
            co2_dispersion=True, co2_emission_rate=0.1,
        )
        r = set_atm_boundary_layer_enhanced_8(m, abl)
        assert r.n_co2_cells >= 0

    def test_wind_energy(self):
        m = _single_hex()
        abl = EnhancedABL8Properties(
            u_star=0.5, z0=0.01,
            renewable_assessment=True, hub_height=80.0,
        )
        r = set_atm_boundary_layer_enhanced_8(m, abl)
        assert isinstance(r.wind_energy, WindEnergyMetrics)

    def test_site_classification(self):
        m = _single_hex()
        abl = EnhancedABL8Properties(u_star=0.5, z0=0.03, classify_site=True)
        r = set_atm_boundary_layer_enhanced_8(m, abl)
        assert isinstance(r.site_class, SiteClassification)
        assert r.site_class.eurocode_category != ""

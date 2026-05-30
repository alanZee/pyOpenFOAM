"""Tests for set_atm_boundary_layer_enhanced_6 — enhanced ABL v6."""
from __future__ import annotations
import pytest
import torch
from pyfoam.core.dtype import INDEX_DTYPE
from pyfoam.mesh.fv_mesh import FvMesh
from pyfoam.tools.set_atm_boundary_layer_enhanced_6 import (
    EnhancedABL6Properties, EnhancedABL6Result, set_atm_boundary_layer_enhanced_6,
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


class TestABL6:
    def test_returns_result_type(self):
        m = _single_hex()
        abl = EnhancedABL6Properties(u_star=0.5, z0=0.01)
        r = set_atm_boundary_layer_enhanced_6(m, abl)
        assert isinstance(r, EnhancedABL6Result)

    def test_heterogeneous_z0(self):
        m = _single_hex()
        abl = EnhancedABL6Properties(
            u_star=0.5, z0=0.01,
            heterogeneous_z0={0.01: ["urban"], 0.001: ["water"]},
        )
        r = set_atm_boundary_layer_enhanced_6(m, abl)
        assert r.n_heterogeneous_patches == 2

    def test_geostrophic_wind(self):
        m = _single_hex()
        abl = EnhancedABL6Properties(
            u_star=0.5, z0=0.01,
            geostrophic_wind=(10.0, 0.0, 0.0),
        )
        r = set_atm_boundary_layer_enhanced_6(m, abl)
        assert isinstance(r.mesoscale_balance, float)

    def test_coriolis_latitude(self):
        m = _single_hex()
        abl = EnhancedABL6Properties(u_star=0.5, z0=0.01, coriolis_latitude=60.0)
        r = set_atm_boundary_layer_enhanced_6(m, abl)
        assert r.latitude_used == 60.0

    def test_time_varying(self):
        m = _single_hex()
        abl = EnhancedABL6Properties(
            u_star=0.5, z0=0.01,
            time_varying=[],
        )
        r = set_atm_boundary_layer_enhanced_6(m, abl)
        assert r.n_time_steps == 0

    def test_default_no_heterogeneity(self):
        m = _single_hex()
        abl = EnhancedABL6Properties(u_star=0.5, z0=0.01)
        r = set_atm_boundary_layer_enhanced_6(m, abl)
        assert r.n_heterogeneous_patches == 0

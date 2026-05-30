"""Tests for set_atm_boundary_layer_enhanced_9 — enhanced ABL v9."""
from __future__ import annotations
import pytest
import torch
import numpy as np
from pyfoam.core.dtype import INDEX_DTYPE
from pyfoam.mesh.fv_mesh import FvMesh
from pyfoam.tools.set_atm_boundary_layer_enhanced_9 import (
    EnhancedABL9Properties, EnhancedABL9Result, MultiLayerABL,
    SourceApportionment, StabilityTransition,
    set_atm_boundary_layer_enhanced_9,
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


class TestSetAtmBoundaryLayerEnhanced9:
    def test_returns_result_type(self):
        m = _single_hex()
        abl = EnhancedABL9Properties(u_star=0.5, z0=0.01)
        r = set_atm_boundary_layer_enhanced_9(m, abl)
        assert isinstance(r, EnhancedABL9Result)

    def test_multi_layer(self):
        m = _single_hex()
        abl = EnhancedABL9Properties(u_star=0.5, z0=0.01, multi_layer=True)
        r = set_atm_boundary_layer_enhanced_9(m, abl)
        assert isinstance(r.multi_layer, MultiLayerABL)

    def test_source_apportionment(self):
        m = _single_hex()
        sources = [
            {"height": 10.0, "rate": 1.0, "name": "stack_A"},
            {"height": 20.0, "rate": 2.0, "name": "stack_B"},
        ]
        abl = EnhancedABL9Properties(
            u_star=0.5, z0=0.01,
            source_apportionment=True, pollutant_sources=sources,
        )
        r = set_atm_boundary_layer_enhanced_9(m, abl)
        assert isinstance(r.apportionment, SourceApportionment)
        assert r.apportionment.n_sources == 2

    def test_stability_transition(self):
        m = _single_hex()
        abl = EnhancedABL9Properties(
            u_star=0.5, z0=0.01,
            stability_transition=True, hour_of_day=14.0,
            surface_heat_flux=200.0,
        )
        r = set_atm_boundary_layer_enhanced_9(m, abl)
        assert isinstance(r.stability, StabilityTransition)
        assert r.stability.hour_of_day == 14.0

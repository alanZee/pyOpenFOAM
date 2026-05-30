"""Tests for apply_boundary_layer_enhanced_7 — enhanced BL application v7."""
from __future__ import annotations
import pytest
import torch
import numpy as np
from pyfoam.core.dtype import INDEX_DTYPE
from pyfoam.mesh.fv_mesh import FvMesh
from pyfoam.tools.apply_boundary_layer_enhanced_7 import EnhancedBL7Properties, EnhancedBL7Result, apply_boundary_layer_enhanced_7


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


class TestBL7:
    def test_returns_result_type(self):
        m = _single_hex()
        U = np.ones((1, 3), dtype=np.float64)
        bl = EnhancedBL7Properties(delta=0.1, nu=1e-5)
        r = apply_boundary_layer_enhanced_7(m, U, bl)
        assert isinstance(r, EnhancedBL7Result)

    def test_separation_bubble(self):
        m = _single_hex()
        U = np.ones((1, 3), dtype=np.float64)
        bl = EnhancedBL7Properties(
            delta=0.1, nu=1e-5,
            separation_bubble=True, bubble_reattachment_factor=0.6,
        )
        r = apply_boundary_layer_enhanced_7(m, U, bl)
        assert r.n_separation_bubbles >= 0

    def test_transition_dynamics(self):
        m = _single_hex()
        U = np.ones((1, 3), dtype=np.float64)
        bl = EnhancedBL7Properties(
            delta=0.1, nu=1e-5,
            transition_dynamics=True, intermittency_onset=200.0,
        )
        r = apply_boundary_layer_enhanced_7(m, U, bl)
        assert 0.0 <= r.intermittency <= 1.0

    def test_heat_transfer_enhancement(self):
        m = _single_hex()
        U = np.ones((1, 3), dtype=np.float64)
        bl = EnhancedBL7Properties(
            delta=0.1, nu=1e-5,
            heat_transfer_enhancement=True,
            nusselt_correlation="gnielinski",
        )
        r = apply_boundary_layer_enhanced_7(m, U, bl)
        assert r.enhanced_nusselt >= 0.0
        assert r.heat_transfer_coefficient >= 0.0

    def test_dittus_boelter(self):
        m = _single_hex()
        U = np.ones((1, 3), dtype=np.float64)
        bl = EnhancedBL7Properties(
            delta=0.1, nu=1e-5,
            heat_transfer_enhancement=True,
            nusselt_correlation="dittus_boelter",
        )
        r = apply_boundary_layer_enhanced_7(m, U, bl)
        assert r.enhanced_nusselt >= 0.0

    def test_defaults(self):
        m = _single_hex()
        U = np.ones((1, 3), dtype=np.float64)
        bl = EnhancedBL7Properties(delta=0.1, nu=1e-5)
        r = apply_boundary_layer_enhanced_7(m, U, bl)
        assert r.n_separation_bubbles == 0
        assert r.intermittency == 0.0
        assert r.enhanced_nusselt == 0.0

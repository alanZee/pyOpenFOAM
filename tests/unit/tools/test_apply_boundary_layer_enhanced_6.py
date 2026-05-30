"""Tests for apply_boundary_layer_enhanced_6 — enhanced BL application v6."""
from __future__ import annotations
import pytest
import torch
import numpy as np
from pyfoam.core.dtype import INDEX_DTYPE
from pyfoam.mesh.fv_mesh import FvMesh
from pyfoam.tools.apply_boundary_layer_enhanced_6 import (
    EnhancedBL6Properties, EnhancedBL6Result, apply_boundary_layer_enhanced_6,
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


class TestBL6:
    def test_returns_result_type(self):
        m = _single_hex()
        U = np.ones((1, 3)) * 10.0
        bl = EnhancedBL6Properties(delta=0.1, nu=1e-5)
        r = apply_boundary_layer_enhanced_6(m, U, bl)
        assert isinstance(r, EnhancedBL6Result)

    def test_compressible(self):
        m = _single_hex()
        U = np.ones((1, 3)) * 10.0
        bl = EnhancedBL6Properties(delta=0.1, nu=1e-5, compressible=True, Mach=0.5)
        r = apply_boundary_layer_enhanced_6(m, U, bl)
        assert r.compressibility_factor >= 1.0

    def test_unsteady(self):
        m = _single_hex()
        U = np.ones((1, 3)) * 10.0
        bl = EnhancedBL6Properties(delta=0.1, nu=1e-5, unsteady=True, lag_time_scale=0.1)
        r = apply_boundary_layer_enhanced_6(m, U, bl)
        assert r.n_unsteady_cells >= 0

    def test_roughness_dynamics(self):
        m = _single_hex()
        U = np.ones((1, 3)) * 10.0
        bl = EnhancedBL6Properties(
            delta=0.1, nu=1e-5,
            z0_rough=0.01, roughness_growth_rate=0.001,
        )
        r = apply_boundary_layer_enhanced_6(m, U, bl)
        assert r.roughness_z0_effective >= bl.z0_rough

    def test_recovery_temperature(self):
        m = _single_hex()
        U = np.ones((1, 3)) * 10.0
        bl = EnhancedBL6Properties(
            delta=0.1, nu=1e-5,
            compressible=True, Mach=0.5, T_wall=300.0, T_inf=293.15,
        )
        r = apply_boundary_layer_enhanced_6(m, U, bl)
        assert r.recovery_temperature >= 293.15

    def test_no_compressibility(self):
        m = _single_hex()
        U = np.ones((1, 3)) * 10.0
        bl = EnhancedBL6Properties(delta=0.1, nu=1e-5)
        r = apply_boundary_layer_enhanced_6(m, U, bl)
        assert r.compressibility_factor == 1.0

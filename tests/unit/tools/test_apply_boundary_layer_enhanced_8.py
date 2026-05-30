"""Tests for apply_boundary_layer_enhanced_8 — enhanced BL application v8."""
from __future__ import annotations
import pytest
import torch
import numpy as np
from pyfoam.core.dtype import INDEX_DTYPE
from pyfoam.mesh.fv_mesh import FvMesh
from pyfoam.tools.apply_boundary_layer_enhanced_8 import (
    EnhancedBL8Properties, EnhancedBL8Result, ThermalBLCoupling, FoulingPrediction,
    NoiseEstimate, apply_boundary_layer_enhanced_8,
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


class TestBL8:
    def test_returns_result_type(self):
        m = _single_hex()
        vel = np.zeros((1, 3), dtype=np.float64)
        bl = EnhancedBL8Properties(delta=0.1, nu=1e-5)
        r = apply_boundary_layer_enhanced_8(m, vel, bl)
        assert isinstance(r, EnhancedBL8Result)

    def test_thermal_coupling(self):
        m = _single_hex()
        vel = np.zeros((1, 3), dtype=np.float64)
        bl = EnhancedBL8Properties(delta=0.1, nu=1e-5, thermal_coupling=True)
        r = apply_boundary_layer_enhanced_8(m, vel, bl)
        assert isinstance(r.thermal_bl, ThermalBLCoupling)

    def test_fouling_prediction(self):
        m = _single_hex()
        vel = np.zeros((1, 3), dtype=np.float64)
        bl = EnhancedBL8Properties(delta=0.1, nu=1e-5, fouling_prediction=True)
        r = apply_boundary_layer_enhanced_8(m, vel, bl)
        assert isinstance(r.fouling, FoulingPrediction)

    def test_noise_modelling(self):
        m = _single_hex()
        vel = np.zeros((1, 3), dtype=np.float64)
        bl = EnhancedBL8Properties(delta=0.1, nu=1e-5, noise_modelling=True)
        r = apply_boundary_layer_enhanced_8(m, vel, bl)
        assert isinstance(r.noise, NoiseEstimate)

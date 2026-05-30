"""Tests for set_waves_enhanced_9 — enhanced wave initialisation v9."""
from __future__ import annotations
import pytest
import torch
import numpy as np
from pyfoam.core.dtype import INDEX_DTYPE
from pyfoam.mesh.fv_mesh import FvMesh
from pyfoam.tools.set_waves_enhanced_9 import (
    EnhancedWave9Properties, EnhancedWave9Result, WaveClimateStats,
    WECPowerMatrix, PropagationResult, set_waves_enhanced_9,
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


class TestSetWavesEnhanced9:
    def test_returns_result_type(self):
        m = _single_hex()
        wave = EnhancedWave9Properties(water_depth=1.0)
        r = set_waves_enhanced_9(m, wave)
        assert isinstance(r, EnhancedWave9Result)

    def test_climate_statistics(self):
        m = _single_hex()
        wave = EnhancedWave9Properties(water_depth=1.0, climate_statistics=True)
        r = set_waves_enhanced_9(m, wave)
        assert isinstance(r.climate, WaveClimateStats)
        assert r.climate.n_samples > 0

    def test_wec_power(self):
        m = _single_hex()
        wave = EnhancedWave9Properties(water_depth=1.0, energy_extraction=True)
        r = set_waves_enhanced_9(m, wave)
        assert isinstance(r.wec_power, WECPowerMatrix)

    def test_propagation(self):
        m = _single_hex()
        bathy = np.array([5.0, 4.0, 3.0, 2.0, 1.0, 0.5, 3.0, 4.0])
        wave = EnhancedWave9Properties(water_depth=5.0, propagation=True, bathymetry_depths=bathy)
        r = set_waves_enhanced_9(m, wave)
        assert isinstance(r.propagation, PropagationResult)
        assert r.propagation.n_cells_transformed >= 0

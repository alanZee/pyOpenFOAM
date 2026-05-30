"""Tests for set_waves_enhanced_7 — enhanced wave initialisation v7."""
from __future__ import annotations
import pytest
import torch
import numpy as np
from pyfoam.core.dtype import INDEX_DTYPE
from pyfoam.mesh.fv_mesh import FvMesh
from pyfoam.tools.set_waves_enhanced_7 import EnhancedWave7Properties, EnhancedWave7Result, set_waves_enhanced_7


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


class TestWaveEnhanced7:
    def test_returns_result_type(self):
        m = _single_hex()
        wave = EnhancedWave7Properties(water_depth=10.0, wave_height=1.0, wave_period=2.0)
        r = set_waves_enhanced_7(m, wave)
        assert isinstance(r, EnhancedWave7Result)

    def test_doppler_shift(self):
        m = _single_hex()
        wave = EnhancedWave7Properties(
            water_depth=10.0, wave_height=1.0, wave_period=2.0,
            current_interaction=True, current_velocity=(1.0, 0.0, 0.0),
        )
        r = set_waves_enhanced_7(m, wave)
        assert isinstance(r.doppler_shift, float)

    def test_energy_analysis(self):
        m = _single_hex()
        wave = EnhancedWave7Properties(water_depth=10.0, wave_height=1.0, wave_period=2.0)
        r = set_waves_enhanced_7(m, wave)
        assert r.energy_flux >= 0.0
        assert r.group_velocity >= 0.0
        assert r.radiation_stress_xx >= 0.0

    def test_sediment_coupling(self):
        m = _single_hex()
        wave = EnhancedWave7Properties(
            water_depth=10.0, wave_height=1.0, wave_period=2.0,
            sediment_coupling=True, sediment_d50=0.0003,
        )
        r = set_waves_enhanced_7(m, wave)
        assert r.shields_parameter >= 0.0
        assert r.sediment_transport_rate >= 0.0

    def test_no_sediment_by_default(self):
        m = _single_hex()
        wave = EnhancedWave7Properties(water_depth=10.0, wave_height=1.0, wave_period=2.0)
        r = set_waves_enhanced_7(m, wave)
        assert r.shields_parameter == 0.0

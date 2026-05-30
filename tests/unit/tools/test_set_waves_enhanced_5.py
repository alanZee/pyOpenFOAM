"""Tests for set_waves_enhanced_5 — enhanced wave initialisation v5."""
from __future__ import annotations
import pytest
import torch
from pyfoam.core.dtype import INDEX_DTYPE
from pyfoam.mesh.fv_mesh import FvMesh
from pyfoam.tools.set_waves_enhanced_5 import (
    EnhancedWave5Properties, EnhancedWave5Result, set_waves_enhanced_5,
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


class TestSetWavesEnhanced5:
    def test_returns_result_type(self):
        m = _single_hex()
        wave = EnhancedWave5Properties(water_depth=10.0, wave_height=0.5, wave_period=2.0)
        r = set_waves_enhanced_5(m, wave)
        assert isinstance(r, EnhancedWave5Result)

    def test_current_interaction(self):
        m = _single_hex()
        wave = EnhancedWave5Properties(
            water_depth=10.0, wave_height=0.5, wave_period=2.0,
            current_velocity=(1.0, 0.0, 0.0),
        )
        r = set_waves_enhanced_5(m, wave)
        assert r.velocity.shape[0] == 1

    def test_multi_directional(self):
        m = _single_hex()
        wave = EnhancedWave5Properties(
            water_depth=10.0, wave_height=0.5, wave_period=2.0,
            wave_type="irregular",
            n_directions=8, spreading_exponent=4.0,
            seed=42,
        )
        r = set_waves_enhanced_5(m, wave)
        assert r.directional_spread > 0

    def test_spectral_diagnostics(self):
        m = _single_hex()
        wave = EnhancedWave5Properties(
            water_depth=10.0, wave_height=0.5, wave_period=2.0,
            wave_type="irregular",
            n_components=10, seed=42,
        )
        r = set_waves_enhanced_5(m, wave)
        assert isinstance(r.peakedness, float)
        assert isinstance(r.groupiness_factor, float)

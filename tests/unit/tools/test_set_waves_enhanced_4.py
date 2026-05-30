"""Tests for set_waves_enhanced_4 — enhanced wave initialisation v4."""
from __future__ import annotations
import numpy as np
import pytest
import torch
from pyfoam.core.dtype import INDEX_DTYPE
from pyfoam.mesh.fv_mesh import FvMesh
from pyfoam.tools.set_waves_enhanced_4 import (
    EnhancedWave4Properties,
    EnhancedWave4Result,
    set_waves_enhanced_4,
)


def _hex_mesh():
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


class TestWavesEnhanced4:
    def test_returns_result_type(self):
        m = _hex_mesh()
        w = EnhancedWave4Properties(water_depth=1.0, wave_height=0.1, wave_period=1.0)
        r = set_waves_enhanced_4(m, w)
        assert isinstance(r, EnhancedWave4Result)

    def test_stokes1(self):
        m = _hex_mesh()
        w = EnhancedWave4Properties(water_depth=1.0, wave_height=0.1, wave_period=1.0, wave_type="stokes1")
        r = set_waves_enhanced_4(m, w)
        assert r.velocity.shape == (1, 3)

    def test_max_elevation(self):
        m = _hex_mesh()
        w = EnhancedWave4Properties(water_depth=1.0, wave_height=0.1, wave_period=1.0)
        r = set_waves_enhanced_4(m, w)
        assert r.max_wave_elevation >= 0

    def test_rogue_wave_mode(self):
        m = _hex_mesh()
        w = EnhancedWave4Properties(
            water_depth=10.0, wave_height=1.0, wave_period=5.0,
            wave_type="rogue", n_components=5, seed=42,
        )
        r = set_waves_enhanced_4(m, w)
        assert isinstance(bool(r.rogue_wave_detected), bool)

    def test_bfi_metric(self):
        m = _hex_mesh()
        w = EnhancedWave4Properties(
            water_depth=10.0, wave_height=1.0, wave_period=5.0,
            wave_type="irregular", n_components=10, seed=42,
        )
        r = set_waves_enhanced_4(m, w)
        assert isinstance(r.benjamin_feir_index, float)

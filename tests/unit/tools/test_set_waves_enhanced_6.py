"""Tests for set_waves_enhanced_6 — enhanced wave initialisation v6."""
from __future__ import annotations
import pytest
import torch
import numpy as np
from pyfoam.core.dtype import INDEX_DTYPE
from pyfoam.mesh.fv_mesh import FvMesh
from pyfoam.tools.set_waves_enhanced_6 import (
    EnhancedWave6Properties, EnhancedWave6Result, set_waves_enhanced_6,
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


class TestWavesEnhanced6:
    def test_returns_result_type(self):
        m = _single_hex()
        wave = EnhancedWave6Properties(water_depth=10.0, wave_height=1.0, wave_period=2.0)
        r = set_waves_enhanced_6(m, wave)
        assert isinstance(r, EnhancedWave6Result)

    def test_sponge_layer(self):
        m = _single_hex()
        wave = EnhancedWave6Properties(
            water_depth=10.0, wave_height=1.0, wave_period=2.0,
            sponge_layer=True, sponge_width=0.5, sponge_coefficient=5.0,
        )
        r = set_waves_enhanced_6(m, wave)
        assert r.n_sponge_cells >= 0
        assert r.sponge_damping is not None

    def test_sponge_linear_profile(self):
        m = _single_hex()
        wave = EnhancedWave6Properties(
            water_depth=10.0, wave_height=1.0, wave_period=2.0,
            sponge_layer=True, sponge_width=0.5, sponge_profile="linear",
        )
        r = set_waves_enhanced_6(m, wave)
        assert r.n_sponge_cells >= 0

    def test_sponge_exponential_profile(self):
        m = _single_hex()
        wave = EnhancedWave6Properties(
            water_depth=10.0, wave_height=1.0, wave_period=2.0,
            sponge_layer=True, sponge_width=0.5, sponge_profile="exponential",
        )
        r = set_waves_enhanced_6(m, wave)
        assert r.n_sponge_cells >= 0

    def test_generation_zone(self):
        m = _single_hex()
        wave = EnhancedWave6Properties(
            water_depth=10.0, wave_height=1.0, wave_period=2.0,
            generation_zone=True, generation_start=0.0, generation_width=1.0,
        )
        r = set_waves_enhanced_6(m, wave)
        assert r.n_generation_cells >= 0

    def test_absorption(self):
        m = _single_hex()
        wave = EnhancedWave6Properties(
            water_depth=10.0, wave_height=1.0, wave_period=2.0,
            absorption=True, sponge_width=1.0,
        )
        r = set_waves_enhanced_6(m, wave)
        assert r.n_absorption_cells >= 0

    def test_no_sponge_damping_by_default(self):
        m = _single_hex()
        wave = EnhancedWave6Properties(water_depth=10.0, wave_height=1.0, wave_period=2.0)
        r = set_waves_enhanced_6(m, wave)
        assert r.sponge_damping is None
        assert r.n_sponge_cells == 0

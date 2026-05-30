"""Tests for set_waves_enhanced_8 — enhanced wave initialisation v8."""
from __future__ import annotations
import pytest
import torch
import numpy as np
from pyfoam.core.dtype import INDEX_DTYPE
from pyfoam.mesh.fv_mesh import FvMesh
from pyfoam.tools.set_waves_enhanced_8 import (
    EnhancedWave8Properties, EnhancedWave8Result, set_waves_enhanced_8,
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


class TestSetWaves8:
    def test_returns_result_type(self):
        m = _single_hex()
        wave = EnhancedWave8Properties(water_depth=1.0, wave_height=0.5, wave_period=1.0)
        r = set_waves_enhanced_8(m, wave)
        assert isinstance(r, EnhancedWave8Result)

    def test_structure_interaction(self):
        m = _single_hex()
        wave = EnhancedWave8Properties(
            water_depth=10.0, wave_height=1.0, wave_period=2.0,
            structure_interaction=True, structure_diameter=5.0,
        )
        r = set_waves_enhanced_8(m, wave)
        assert r.morison_force >= 0

    def test_morphodynamics(self):
        m = _single_hex()
        wave = EnhancedWave8Properties(
            water_depth=10.0, wave_height=1.0, wave_period=2.0,
            sediment_coupling=True, morphodynamics=True,
        )
        r = set_waves_enhanced_8(m, wave)
        assert isinstance(r.bed_level_change, float)

    def test_overtopping(self):
        m = _single_hex()
        wave = EnhancedWave8Properties(
            water_depth=10.0, wave_height=2.0, wave_period=3.0,
            estimate_overtopping=True, crest_height=3.0,
        )
        r = set_waves_enhanced_8(m, wave)
        assert r.overtopping_rate >= 0

    def test_default_properties(self):
        p = EnhancedWave8Properties()
        assert p.structure_interaction is False
        assert p.morphodynamics is False

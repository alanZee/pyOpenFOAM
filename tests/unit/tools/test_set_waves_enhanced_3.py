"""Tests for set_waves_enhanced_3 — enhanced wave initialisation v3."""
from __future__ import annotations
import math
import numpy as np
import pytest
from pyfoam.tools.set_waves_enhanced_3 import (
    EnhancedWave3Properties,
    EnhancedWave3Result,
    set_waves_enhanced_3,
)


class TestEnhancedWave3Properties:
    def test_default_values(self):
        w = EnhancedWave3Properties()
        assert w.water_depth == 10.0
        assert w.wave_type == "stokes1"
        assert w.stream_N == 5
        assert w.beach_slope == 0.05

    def test_wave_type_options(self):
        for wt in ["stokes1", "stokes2", "stokes5", "cnoidal", "irregular", "stream_function"]:
            w = EnhancedWave3Properties(wave_type=wt)
            assert w.wave_type == wt

    def test_wave_number_deep_water(self):
        w = EnhancedWave3Properties(water_depth=1000.0, wave_period=4.0)
        k = w.wave_number(9.81)
        expected = (2 * math.pi / 4.0) ** 2 / 9.81
        assert abs(k - expected) / expected < 0.01


class TestSetWavesEnhanced3:
    def test_returns_result_type(self, fv_mesh):
        w = EnhancedWave3Properties(water_depth=10.0, wave_height=0.5, wave_period=2.0)
        r = set_waves_enhanced_3(fv_mesh, w)
        assert isinstance(r, EnhancedWave3Result)

    def test_alpha_shape(self, fv_mesh):
        w = EnhancedWave3Properties(water_depth=10.0, wave_height=0.5, wave_period=2.0)
        r = set_waves_enhanced_3(fv_mesh, w)
        assert r.alpha.shape == (fv_mesh.n_cells,)

    def test_ursell_number(self, fv_mesh):
        w = EnhancedWave3Properties(water_depth=10.0, wave_height=0.5, wave_period=2.0)
        r = set_waves_enhanced_3(fv_mesh, w)
        assert r.ursell_number > 0

    def test_iribarren_number(self, fv_mesh):
        w = EnhancedWave3Properties(
            water_depth=10.0, wave_height=0.5, wave_period=2.0, beach_slope=0.1,
        )
        r = set_waves_enhanced_3(fv_mesh, w)
        assert r.iribarren_number > 0

    def test_is_breaking_check(self, fv_mesh):
        w = EnhancedWave3Properties(water_depth=10.0, wave_height=0.5, wave_period=2.0)
        r = set_waves_enhanced_3(fv_mesh, w)
        assert isinstance(r.is_breaking, bool)

    def test_stream_function_mode(self, fv_mesh):
        w = EnhancedWave3Properties(
            water_depth=10.0, wave_height=0.5, wave_period=2.0,
            wave_type="stream_function", stream_N=3,
        )
        r = set_waves_enhanced_3(fv_mesh, w)
        assert r.alpha.shape == (fv_mesh.n_cells,)
        assert r.stream_coefficients is not None

    def test_compute_potential(self, fv_mesh):
        w = EnhancedWave3Properties(water_depth=10.0, wave_height=0.5, wave_period=2.0)
        r = set_waves_enhanced_3(fv_mesh, w, compute_potential=True)
        assert r.potential is not None
        assert r.potential.shape == (fv_mesh.n_cells,)

    def test_stokes5_mode(self, fv_mesh):
        w = EnhancedWave3Properties(
            water_depth=10.0, wave_height=0.5, wave_period=2.0, wave_type="stokes5",
        )
        r = set_waves_enhanced_3(fv_mesh, w)
        assert r.alpha.shape == (fv_mesh.n_cells,)

    def test_irregular_waves(self, fv_mesh):
        w = EnhancedWave3Properties(
            water_depth=10.0, wave_height=0.5, wave_period=2.0,
            wave_type="irregular", n_components=5, seed=42,
        )
        r = set_waves_enhanced_3(fv_mesh, w)
        assert r.spectrum_frequencies is not None
        assert len(r.spectrum_frequencies) == 5

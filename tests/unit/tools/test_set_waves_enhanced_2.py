"""Tests for set_waves_enhanced_2 — enhanced wave initialisation v2."""
from __future__ import annotations
import math
import numpy as np
import pytest
from pyfoam.tools.set_waves_enhanced_2 import (
    EnhancedWave2Properties,
    EnhancedWave2Result,
    set_waves_enhanced_2,
)


class TestEnhancedWave2Properties:
    def test_default_values(self):
        w = EnhancedWave2Properties()
        assert w.water_depth == 10.0
        assert w.wave_height == 1.0
        assert w.wave_type == "stokes1"

    def test_angular_frequency(self):
        w = EnhancedWave2Properties(wave_period=4.0)
        assert abs(w.angular_frequency() - math.pi / 2.0) < 1e-10

    def test_wave_number_deep_water(self):
        w = EnhancedWave2Properties(water_depth=1000.0, wave_period=4.0)
        k = w.wave_number(9.81)
        expected = (2 * math.pi / 4.0) ** 2 / 9.81
        assert abs(k - expected) / expected < 0.01

    def test_wave_type_options(self):
        for wt in ["stokes1", "stokes2", "stokes5", "cnoidal", "irregular"]:
            w = EnhancedWave2Properties(wave_type=wt)
            assert w.wave_type == wt

    def test_current_velocity(self):
        w = EnhancedWave2Properties(current_velocity=(1.0, 0.0, 0.0))
        assert w.current_velocity == (1.0, 0.0, 0.0)


class TestSetWavesEnhanced2:
    def test_returns_result_type(self, fv_mesh):
        w = EnhancedWave2Properties(water_depth=10.0, wave_height=0.5, wave_period=2.0)
        r = set_waves_enhanced_2(fv_mesh, w)
        assert isinstance(r, EnhancedWave2Result)

    def test_alpha_shape(self, fv_mesh):
        w = EnhancedWave2Properties(water_depth=10.0, wave_height=0.5, wave_period=2.0)
        r = set_waves_enhanced_2(fv_mesh, w)
        assert r.alpha.shape == (fv_mesh.n_cells,)

    def test_pressure_shape(self, fv_mesh):
        w = EnhancedWave2Properties(water_depth=10.0, wave_height=0.5, wave_period=2.0)
        r = set_waves_enhanced_2(fv_mesh, w)
        assert r.pressure.shape == (fv_mesh.n_cells,)

    def test_velocity_shape(self, fv_mesh):
        w = EnhancedWave2Properties(water_depth=10.0, wave_height=0.5, wave_period=2.0)
        r = set_waves_enhanced_2(fv_mesh, w)
        assert r.velocity.shape == (fv_mesh.n_cells, 3)

    def test_stokes5_mode(self, fv_mesh):
        w = EnhancedWave2Properties(
            water_depth=10.0, wave_height=0.5, wave_period=2.0, wave_type="stokes5",
        )
        r = set_waves_enhanced_2(fv_mesh, w)
        assert r.alpha.shape == (fv_mesh.n_cells,)

    def test_compute_potential(self, fv_mesh):
        w = EnhancedWave2Properties(water_depth=10.0, wave_height=0.5, wave_period=2.0)
        r = set_waves_enhanced_2(fv_mesh, w, compute_potential=True)
        assert r.potential is not None
        assert r.potential.shape == (fv_mesh.n_cells,)

    def test_irregular_waves(self, fv_mesh):
        w = EnhancedWave2Properties(
            water_depth=10.0, wave_height=0.5, wave_period=2.0,
            wave_type="irregular", n_components=5, seed=42,
        )
        r = set_waves_enhanced_2(fv_mesh, w)
        assert r.alpha.shape == (fv_mesh.n_cells,)
        assert r.spectrum_frequencies is not None
        assert len(r.spectrum_frequencies) == 5

    def test_still_water(self, fv_mesh):
        w = EnhancedWave2Properties(water_depth=10.0, wave_height=0.0, wave_period=2.0)
        r = set_waves_enhanced_2(fv_mesh, w, free_surface_z=0.5)
        cc = fv_mesh.cell_centres.detach().cpu().numpy()
        for ci in range(fv_mesh.n_cells):
            z = cc[ci, 2]
            if z < 0.5:
                assert r.alpha[ci] == 1.0

    def test_cnoidal_mode(self, fv_mesh):
        w = EnhancedWave2Properties(
            water_depth=10.0, wave_height=0.5, wave_period=2.0, wave_type="cnoidal",
        )
        r = set_waves_enhanced_2(fv_mesh, w)
        assert r.alpha.shape == (fv_mesh.n_cells,)

"""Tests for set_waves_enhanced — enhanced wave initialisation."""
from __future__ import annotations
import math
import numpy as np
import pytest
from pyfoam.tools.set_waves_enhanced import (
    EnhancedWaveProperties,
    EnhancedWaveResult,
    set_waves_enhanced,
)


class TestEnhancedWaveProperties:
    def test_default_values(self):
        w = EnhancedWaveProperties()
        assert w.water_depth == 10.0
        assert w.wave_height == 1.0
        assert w.wave_type == "stokes1"

    def test_angular_frequency(self):
        w = EnhancedWaveProperties(wave_period=4.0)
        assert abs(w.angular_frequency() - math.pi / 2.0) < 1e-10

    def test_wave_number_deep_water(self):
        w = EnhancedWaveProperties(water_depth=1000.0, wave_period=4.0)
        k = w.wave_number(9.81)
        expected = (2 * math.pi / 4.0) ** 2 / 9.81
        assert abs(k - expected) / expected < 0.01

    def test_explicit_wave_length(self):
        w = EnhancedWaveProperties(wave_length=10.0)
        k = w.wave_number()
        assert abs(k - 2 * math.pi / 10.0) < 1e-10

    def test_wave_type_options(self):
        for wt in ["stokes1", "stokes2", "cnoidal"]:
            w = EnhancedWaveProperties(wave_type=wt)
            assert w.wave_type == wt


class TestSetWavesEnhanced:
    def test_returns_result_type(self, fv_mesh):
        w = EnhancedWaveProperties(water_depth=10.0, wave_height=0.5, wave_period=2.0)
        r = set_waves_enhanced(fv_mesh, w)
        assert isinstance(r, EnhancedWaveResult)

    def test_alpha_shape(self, fv_mesh):
        w = EnhancedWaveProperties(water_depth=10.0, wave_height=0.5, wave_period=2.0)
        r = set_waves_enhanced(fv_mesh, w)
        assert r.alpha.shape == (fv_mesh.n_cells,)

    def test_pressure_shape(self, fv_mesh):
        w = EnhancedWaveProperties(water_depth=10.0, wave_height=0.5, wave_period=2.0)
        r = set_waves_enhanced(fv_mesh, w)
        assert r.pressure.shape == (fv_mesh.n_cells,)

    def test_velocity_shape(self, fv_mesh):
        w = EnhancedWaveProperties(water_depth=10.0, wave_height=0.5, wave_period=2.0)
        r = set_waves_enhanced(fv_mesh, w)
        assert r.velocity.shape == (fv_mesh.n_cells, 3)

    def test_wave_number_returned(self, fv_mesh):
        w = EnhancedWaveProperties(water_depth=10.0, wave_height=0.5, wave_period=2.0)
        r = set_waves_enhanced(fv_mesh, w)
        assert r.wave_number > 0
        assert r.wave_length > 0

    def test_stokes2_mode(self, fv_mesh):
        w = EnhancedWaveProperties(
            water_depth=10.0, wave_height=0.5, wave_period=2.0, wave_type="stokes2",
        )
        r = set_waves_enhanced(fv_mesh, w)
        assert r.alpha.shape == (fv_mesh.n_cells,)

    def test_cnoidal_mode(self, fv_mesh):
        w = EnhancedWaveProperties(
            water_depth=10.0, wave_height=0.5, wave_period=2.0, wave_type="cnoidal",
        )
        r = set_waves_enhanced(fv_mesh, w)
        assert r.alpha.shape == (fv_mesh.n_cells,)

    def test_still_water(self, fv_mesh):
        w = EnhancedWaveProperties(water_depth=10.0, wave_height=0.0, wave_period=2.0)
        r = set_waves_enhanced(fv_mesh, w, free_surface_z=0.5)
        cc = fv_mesh.cell_centres.detach().cpu().numpy()
        for ci in range(fv_mesh.n_cells):
            z = cc[ci, 2]
            if z < 0.5:
                assert r.alpha[ci] == 1.0

    def test_pressure_non_negative(self, fv_mesh):
        w = EnhancedWaveProperties(water_depth=10.0, wave_height=0.5, wave_period=2.0)
        r = set_waves_enhanced(fv_mesh, w)
        assert np.all(r.pressure >= 0.0)

    def test_gravity_vector(self, fv_mesh):
        w = EnhancedWaveProperties(water_depth=10.0, wave_height=0.5, wave_period=2.0)
        r = set_waves_enhanced(fv_mesh, w, g=[0, 0, -9.81])
        assert r.alpha.shape == (fv_mesh.n_cells,)

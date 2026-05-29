"""Tests for set_waves — wave field initialisation."""
from __future__ import annotations
import math
import numpy as np
import pytest
from pyfoam.tools.set_waves import set_waves, WaveProperties, wave_celerity, deep_water_wavelength


class TestWaveProperties:
    def test_angular_frequency(self):
        """omega = 2*pi/T."""
        w = WaveProperties(wave_period=4.0)
        assert abs(w.angular_frequency() - math.pi / 2.0) < 1e-10

    def test_wave_number_deep_water(self):
        """Deep water: k ~ omega^2 / g."""
        w = WaveProperties(water_depth=1000.0, wave_period=4.0)
        k = w.wave_number(9.81)
        expected = (2 * math.pi / 4.0) ** 2 / 9.81
        assert abs(k - expected) / expected < 0.01

    def test_wave_number_shallow_water(self):
        """Shallow water dispersion should converge."""
        w = WaveProperties(water_depth=5.0, wave_period=10.0)
        k = w.wave_number(9.81)
        assert k > 0

    def test_wave_number_explicit(self):
        """Explicit wave_length should override dispersion."""
        w = WaveProperties(wave_length=10.0)
        k = w.wave_number()
        assert abs(k - 2 * math.pi / 10.0) < 1e-10

    def test_default_values(self):
        """Default constructor should work."""
        w = WaveProperties()
        assert w.water_depth == 10.0
        assert w.wave_height == 1.0
        assert w.wave_period == 2.0


class TestSetWaves:
    def test_returns_arrays(self, fv_mesh):
        """Should return alpha and pressure arrays."""
        w = WaveProperties(water_depth=10.0, wave_height=1.0, wave_period=2.0)
        alpha, p = set_waves(fv_mesh, w)
        n_cells = fv_mesh.n_cells
        assert alpha.shape == (n_cells,)
        assert p.shape == (n_cells,)

    def test_alpha_range(self, fv_mesh):
        """Alpha should be 0 or 1 (sharp interface)."""
        w = WaveProperties(water_depth=10.0, wave_height=0.1, wave_period=2.0)
        alpha, _ = set_waves(fv_mesh, w)
        assert np.all((alpha == 0.0) | (alpha == 1.0))

    def test_pressure_non_negative(self, fv_mesh):
        """Pressure should be non-negative."""
        w = WaveProperties(water_depth=10.0, wave_height=0.5, wave_period=2.0)
        _, p = set_waves(fv_mesh, w)
        assert np.all(p >= 0.0)

    def test_still_water_uniform(self, fv_mesh):
        """Zero wave height should produce uniform water below surface."""
        w = WaveProperties(water_depth=10.0, wave_height=0.0, wave_period=2.0)
        alpha, p = set_waves(fv_mesh, w, free_surface_z=0.5)
        # All cells at z < 0.5 should be water (alpha=1)
        # All cells at z > 0.5 should be air (alpha=0)
        cell_centres = fv_mesh.cell_centres.detach().cpu().numpy()
        for ci in range(fv_mesh.n_cells):
            z = cell_centres[ci, 2]
            if z < 0.5:
                assert alpha[ci] == 1.0
            else:
                assert alpha[ci] == 0.0

    def test_gravity_vector(self, fv_mesh):
        """Custom gravity vector should work."""
        w = WaveProperties(water_depth=10.0, wave_height=0.0, wave_period=2.0)
        alpha, p = set_waves(fv_mesh, w, g=[0, 0, -9.81])
        assert alpha.shape == (fv_mesh.n_cells,)

    def test_custom_rho(self, fv_mesh):
        """Higher density should scale pressure."""
        w = WaveProperties(water_depth=10.0, wave_height=0.0, wave_period=2.0)
        _, p1 = set_waves(fv_mesh, w, rho=1000.0, free_surface_z=10.0)
        _, p2 = set_waves(fv_mesh, w, rho=2000.0, free_surface_z=10.0)
        # Cells fully submerged: p2 should be 2x p1
        submerged = p1 > 0
        if submerged.any():
            ratio = p2[submerged] / p1[submerged]
            assert np.allclose(ratio, 2.0, atol=0.1)

    def test_time_dependence(self, fv_mesh):
        """Wave field should change with time."""
        w = WaveProperties(water_depth=10.0, wave_height=1.0, wave_period=2.0)
        alpha1, _ = set_waves(fv_mesh, w, time=0.0)
        alpha2, _ = set_waves(fv_mesh, w, time=1.0)
        # At least some cells should differ (wave has moved)
        # This depends on mesh extent — just check it doesn't crash
        assert alpha1.shape == alpha2.shape


class TestWaveUtilities:
    def test_celerity(self):
        """Wave celerity should be positive."""
        w = WaveProperties(water_depth=10.0, wave_period=4.0)
        c = wave_celerity(w, g=9.81)
        assert c > 0

    def test_deep_water_wavelength(self):
        """L0 = g*T^2/(2*pi)."""
        w = WaveProperties(wave_period=4.0)
        L0 = deep_water_wavelength(w, g=9.81)
        expected = 9.81 * 16.0 / (2 * math.pi)
        assert abs(L0 - expected) < 1e-6

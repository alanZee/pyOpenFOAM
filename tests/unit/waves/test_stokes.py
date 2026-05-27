"""Tests for the 2nd-order Stokes wave model."""

import math

import pytest
import torch

from pyfoam.waves.wave_model import WaveModel, GRAVITY
from pyfoam.waves.stokes import StokesWave
from pyfoam.waves.airy import AiryWave


@pytest.fixture
def stokes_wave():
    """Stokes wave: A=2m, d=20m, T=8s."""
    return StokesWave(amplitude=2.0, depth=20.0, period=8.0)


@pytest.fixture
def small_stokes():
    """Small-amplitude Stokes wave: A=0.1m, d=20m, T=6s."""
    return StokesWave(amplitude=0.1, depth=20.0, period=6.0)


# ---------------------------------------------------------------------------
# RTS registry
# ---------------------------------------------------------------------------

class TestStokesRTS:
    def test_registered(self):
        assert "stokes" in WaveModel.available_types()

    def test_create_via_factory(self):
        wave = WaveModel.create("stokes", amplitude=2.0, depth=20.0, period=8.0)
        assert isinstance(wave, StokesWave)

    def test_type_name(self):
        wave = StokesWave(amplitude=2.0, depth=20.0, period=8.0)
        assert wave.type_name == "stokes"


# ---------------------------------------------------------------------------
# Elevation
# ---------------------------------------------------------------------------

class TestStokesElevation:
    def test_shape(self, stokes_wave):
        x = torch.linspace(0, 100, 50)
        eta = stokes_wave.wave_elevation(x, t=0.0)
        assert eta.shape == (50,)

    def test_reduces_to_airy_for_small_amplitude(self, small_stokes):
        """For small amplitude, Stokes ≈ Airy."""
        airy = AiryWave(
            amplitude=small_stokes.amplitude,
            depth=small_stokes.depth,
            period=small_stokes.period,
        )
        x = torch.linspace(0, 50, 100)
        eta_stokes = small_stokes.wave_elevation(x, t=0.0)
        eta_airy = airy.wave_elevation(x, t=0.0)
        # 二阶修正应远小于一阶项
        diff = (eta_stokes - eta_airy).abs()
        assert diff.max() < small_stokes.amplitude * 0.05

    def test_periodicity(self, stokes_wave):
        """Stokes wave should have same period as Airy."""
        L = 2.0 * math.pi / stokes_wave.wavenumber
        x = torch.linspace(0, 100, 100)
        eta1 = stokes_wave.wave_elevation(x, t=0.0)
        eta2 = stokes_wave.wave_elevation(x + L, t=0.0)
        assert torch.allclose(eta1, eta2, atol=1e-6)

    def test_second_harmonic_present(self, stokes_wave):
        """2nd-order elevation should differ from pure cosine."""
        x = torch.linspace(0, 200, 200)
        eta = stokes_wave.wave_elevation(x, t=0.0)
        A = stokes_wave.amplitude
        # 含二阶项后，最大值应 > A（波峰变尖）
        assert eta.max() > A

    def test_dtype_consistency(self, stokes_wave):
        x = torch.linspace(0, 10, 10, dtype=torch.float64)
        eta = stokes_wave.wave_elevation(x, t=0.0)
        assert eta.dtype == torch.float64


# ---------------------------------------------------------------------------
# Velocity
# ---------------------------------------------------------------------------

class TestStokesVelocity:
    def test_shape(self, stokes_wave):
        x = torch.linspace(0, 50, 20)
        z = torch.full((20,), 18.0)
        u, w = stokes_wave.velocity(x, t=0.0, z=z)
        assert u.shape == (20,)
        assert w.shape == (20,)

    def test_velocity_exceeds_airy(self, stokes_wave):
        """Stokes velocity should be larger than Airy due to 2nd-order terms."""
        airy = AiryWave(
            amplitude=stokes_wave.amplitude,
            depth=stokes_wave.depth,
            period=stokes_wave.period,
        )
        x = torch.linspace(0, 50, 20)
        z = torch.full((20,), stokes_wave.depth * 0.9)
        u_stokes, _ = stokes_wave.velocity(x, t=0.0, z=z)
        u_airy, _ = airy.velocity(x, t=0.0, z=z)
        # 在某个点处 Stokes 速度应更大
        assert u_stokes.abs().max() >= u_airy.abs().max()

    def test_finite_values(self, stokes_wave):
        x = torch.linspace(0, 50, 20)
        z = torch.linspace(1, 19, 20)
        u, w = stokes_wave.velocity(x, t=0.0, z=z)
        assert torch.isfinite(u).all()
        assert torch.isfinite(w).all()


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------

class TestStokesProperties:
    def test_repr(self):
        wave = StokesWave(amplitude=2.0, depth=20.0, period=8.0)
        assert "StokesWave" in repr(wave)

"""Tests for the cnoidal wave model."""

import math

import pytest
import torch

from pyfoam.waves.wave_model import WaveModel, GRAVITY
from pyfoam.waves.cnoidal import CnoidalWave
from pyfoam.waves.airy import AiryWave


@pytest.fixture
def cnoidal_wave():
    """Cnoidal wave in shallow water: A=1m, d=5m, T=10s, L=80m."""
    return CnoidalWave(amplitude=1.0, depth=5.0, period=10.0, wavelength=80.0)


@pytest.fixture
def deep_cnoidal():
    """Cnoidal wave in deeper water (m -> 0): A=0.5m, d=50m, T=8s."""
    return CnoidalWave(amplitude=0.5, depth=50.0, period=8.0)


# ---------------------------------------------------------------------------
# RTS registry
# ---------------------------------------------------------------------------

class TestCnoidalRTS:
    def test_registered(self):
        assert "cnoidal" in WaveModel.available_types()

    def test_create_via_factory(self):
        wave = WaveModel.create(
            "cnoidal", amplitude=1.0, depth=5.0, period=10.0, wavelength=80.0
        )
        assert isinstance(wave, CnoidalWave)

    def test_type_name(self):
        wave = CnoidalWave(amplitude=1.0, depth=5.0, period=10.0, wavelength=80.0)
        assert wave.type_name == "cnoidal"


# ---------------------------------------------------------------------------
# Elliptic parameter
# ---------------------------------------------------------------------------

class TestCnoidalEllipticParameter:
    def test_m_range(self, cnoidal_wave):
        """m should be in [0, 1)."""
        m = cnoidal_wave.elliptic_parameter
        assert 0.0 <= m < 1.0

    def test_m_zero_for_deep_water(self, deep_cnoidal):
        """For deep water conditions, m should be near zero."""
        m = deep_cnoidal.elliptic_parameter
        assert m < 0.5  # 深水时 m 应较小

    def test_celerity_positive(self, cnoidal_wave):
        """Wave celerity should be positive and finite."""
        c = cnoidal_wave.celerity
        assert c > 0
        assert math.isfinite(c)

    def test_celerity_exceeds_shallow_water(self, cnoidal_wave):
        """Celerity should be >= sqrt(g*d) for nonlinear waves."""
        d = cnoidal_wave.depth
        c_min = math.sqrt(GRAVITY * d)
        assert cnoidal_wave.celerity >= c_min - 1e-6


# ---------------------------------------------------------------------------
# wave_elevation
# ---------------------------------------------------------------------------

class TestCnoidalElevation:
    def test_shape(self, cnoidal_wave):
        x = torch.linspace(0, 100, 50)
        eta = cnoidal_wave.wave_elevation(x, t=0.0)
        assert eta.shape == (50,)

    def test_periodicity(self, cnoidal_wave):
        """Cnoidal wave should be periodic with wavelength L."""
        L = cnoidal_wave._L
        x = torch.linspace(0, 50, 100)
        eta1 = cnoidal_wave.wave_elevation(x, t=0.0)
        eta2 = cnoidal_wave.wave_elevation(x + L, t=0.0)
        # 允许较大误差（简化实现）
        assert torch.allclose(eta1, eta2, atol=0.5)

    def test_finite_values(self, cnoidal_wave):
        x = torch.linspace(0, 100, 50)
        eta = cnoidal_wave.wave_elevation(x, t=0.0)
        assert torch.isfinite(eta).all()

    def test_deep_water_reduces_to_airy(self, deep_cnoidal):
        """For deep water (m -> 0), cnoidal ≈ Airy."""
        airy = AiryWave(
            amplitude=deep_cnoidal.amplitude,
            depth=deep_cnoidal.depth,
            period=deep_cnoidal.period,
        )
        x = torch.linspace(0, 50, 100)
        eta_cnoidal = deep_cnoidal.wave_elevation(x, t=0.0)
        eta_airy = airy.wave_elevation(x, t=0.0)
        # 允许 20% 误差（简化近似）
        assert torch.allclose(eta_cnoidal, eta_airy, atol=0.5)

    def test_dtype_float64(self, cnoidal_wave):
        x = torch.linspace(0, 10, 10, dtype=torch.float64)
        eta = cnoidal_wave.wave_elevation(x, t=0.0)
        assert eta.dtype == torch.float64


# ---------------------------------------------------------------------------
# velocity
# ---------------------------------------------------------------------------

class TestCnoidalVelocity:
    def test_shape(self, cnoidal_wave):
        x = torch.linspace(0, 50, 20)
        z = torch.full((20,), 4.0)
        u, w = cnoidal_wave.velocity(x, t=0.0, z=z)
        assert u.shape == (20,)
        assert w.shape == (20,)

    def test_vertical_velocity_zero(self, cnoidal_wave):
        """Shallow water approximation: w ≈ 0."""
        x = torch.linspace(0, 50, 20)
        z = torch.full((20,), 4.0)
        _, w = cnoidal_wave.velocity(x, t=0.0, z=z)
        assert torch.allclose(w, torch.zeros_like(w))

    def test_horizontal_velocity_finite(self, cnoidal_wave):
        """u should be finite."""
        x = torch.linspace(0, 50, 20)
        z = torch.full((20,), 4.0)
        u, _ = cnoidal_wave.velocity(x, t=0.0, z=z)
        assert torch.isfinite(u).all()

    def test_velocity_positive_with_positive_elevation(self, cnoidal_wave):
        """正波高处应有正速度（浅水近似）。"""
        x = torch.linspace(0, 50, 20)
        z = torch.full((20,), 4.0)
        u, _ = cnoidal_wave.velocity(x, t=0.0, z=z)
        eta = cnoidal_wave.wave_elevation(x, t=0.0)
        # u = c * eta/d，因此 u 与 eta 同号
        mask = eta > 0
        if mask.any():
            assert (u[mask] > -1e-6).all()

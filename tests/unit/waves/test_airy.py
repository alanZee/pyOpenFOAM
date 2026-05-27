"""Tests for the Airy wave model."""

import math

import pytest
import torch

from pyfoam.waves.wave_model import WaveModel, GRAVITY
from pyfoam.waves.airy import AiryWave


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def deep_wave():
    """Deep-water Airy wave: A=1m, d=100m, T=8s."""
    return AiryWave(amplitude=1.0, depth=100.0, period=8.0)


@pytest.fixture
def shallow_wave():
    """Shallow-water Airy wave: A=0.5m, d=10m, T=10s."""
    return AiryWave(amplitude=0.5, depth=10.0, period=10.0)


# ---------------------------------------------------------------------------
# RTS registry
# ---------------------------------------------------------------------------

class TestAiryRTS:
    def test_registered(self):
        assert "airy" in WaveModel.available_types()

    def test_create_via_factory(self):
        wave = WaveModel.create("airy", amplitude=1.0, depth=10.0, period=8.0)
        assert isinstance(wave, AiryWave)
        assert wave.amplitude == 1.0

    def test_type_name(self):
        wave = AiryWave(amplitude=1.0, depth=10.0, period=8.0)
        assert wave.type_name == "airy"


# ---------------------------------------------------------------------------
# Dispersion relation
# ---------------------------------------------------------------------------

class TestAiryDispersion:
    def test_deep_water_dispersion(self, deep_wave):
        """In deep water, k ~ omega^2/g."""
        omega = deep_wave.angular_frequency
        k_expected = omega**2 / GRAVITY
        k_actual = deep_wave.wavenumber
        assert abs(k_actual - k_expected) / k_expected < 1e-5

    def test_dispersion_relation_satisfied(self, deep_wave):
        """omega^2 = g*k*tanh(k*d) must hold."""
        k = deep_wave.wavenumber
        omega = deep_wave.angular_frequency
        d = deep_wave.depth
        lhs = omega**2
        rhs = GRAVITY * k * math.tanh(k * d)
        assert abs(lhs - rhs) / lhs < 1e-10

    def test_shallow_water_dispersion(self, shallow_wave):
        """In shallow water, k ~ omega/sqrt(g*d)."""
        omega = shallow_wave.angular_frequency
        d = shallow_wave.depth
        k_approx = omega / math.sqrt(GRAVITY * d)
        k_actual = shallow_wave.wavenumber
        # 浅水近似有偏差，允许 10% 误差
        assert abs(k_actual - k_approx) / k_approx < 0.10


# ---------------------------------------------------------------------------
# wave_elevation
# ---------------------------------------------------------------------------

class TestAiryElevation:
    def test_shape(self, deep_wave):
        x = torch.linspace(0, 100, 50)
        eta = deep_wave.wave_elevation(x, t=0.0)
        assert eta.shape == (50,)

    def test_amplitude_range(self, deep_wave):
        """Elevation should stay within [-A, A]."""
        x = torch.linspace(0, 200, 200)
        for t in [0.0, 2.0, 4.0, 6.0]:
            eta = deep_wave.wave_elevation(x, t=t)
            assert eta.max() <= deep_wave.amplitude + 1e-10
            assert eta.min() >= -deep_wave.amplitude - 1e-10

    def test_periodicity(self, deep_wave):
        """eta(x, t) should be periodic: eta(x + L, t) = eta(x, t)."""
        L = 2.0 * math.pi / deep_wave.wavenumber
        x = torch.linspace(0, 50, 100)
        eta1 = deep_wave.wave_elevation(x, t=0.0)
        eta2 = deep_wave.wave_elevation(x + L, t=0.0)
        # 浮点 L 计算引入截断误差，放宽至 1e-6
        assert torch.allclose(eta1, eta2, atol=1e-6)

    def test_time_shift(self, deep_wave):
        """Shifting x by c*dt should be same as shifting t by dt."""
        T = deep_wave.period
        x = torch.linspace(0, 50, 100)
        dt = T / 4.0
        c = deep_wave.angular_frequency / deep_wave.wavenumber
        eta_t = deep_wave.wave_elevation(x, t=dt)
        eta_x = deep_wave.wave_elevation(x - c * dt, t=0.0)
        assert torch.allclose(eta_t, eta_x, atol=1e-6)

    def test_at_t_zero(self, deep_wave):
        """At t=0, eta = A*cos(k*x); at x=0 should be A."""
        x_zero = torch.tensor([0.0])
        eta = deep_wave.wave_elevation(x_zero, t=0.0)
        assert torch.allclose(eta, torch.tensor([deep_wave.amplitude]), atol=1e-10)

    def test_dtype_float64(self, deep_wave):
        x = torch.linspace(0, 10, 10, dtype=torch.float64)
        eta = deep_wave.wave_elevation(x, t=0.0)
        assert eta.dtype == torch.float64


# ---------------------------------------------------------------------------
# velocity
# ---------------------------------------------------------------------------

class TestAiryVelocity:
    def test_shape(self, deep_wave):
        x = torch.linspace(0, 50, 20)
        z = torch.full((20,), 90.0)  # 接近水面
        u, w = deep_wave.velocity(x, t=0.0, z=z)
        assert u.shape == (20,)
        assert w.shape == (20,)

    def test_velocity_decreases_with_depth(self, deep_wave):
        """深水中速度幅值随深度递减：海底 u 小于水面 u。"""
        x = torch.tensor([0.0])
        z_bed = torch.tensor([0.0])
        z_surface = torch.tensor([deep_wave.depth])

        u_bed, _ = deep_wave.velocity(x, t=0.0, z=z_bed)
        u_surf, _ = deep_wave.velocity(x, t=0.0, z=z_surface)
        # cosh(k*z) 在 z=d 处最大，z=0 处最小，所以 u_bed <= u_surf
        assert abs(u_bed.item()) <= abs(u_surf.item()) + 1e-10

    def test_velocity_at_seabed(self, deep_wave):
        """海底 z=0 处：w=0（不可穿透），u=A*omega/sinh(kd)（指数衰减）。"""
        x = torch.tensor([0.0])
        z_bed = torch.tensor([0.0])
        u, w = deep_wave.velocity(x, t=0.0, z=z_bed)
        # 海底不可穿透条件：w(z=0) = 0
        assert abs(w.item()) < 1e-10
        # u(z=0) = A*omega*cosh(0)/sinh(kd) = A*omega/sinh(kd)
        k = deep_wave.wavenumber
        omega = deep_wave.angular_frequency
        d = deep_wave.depth
        expected_u = deep_wave.amplitude * omega / math.sinh(k * d)
        assert abs(u.item() - expected_u) / abs(expected_u) < 1e-6

    def test_divergence_free(self, deep_wave):
        """速度场应近似满足连续性：du/dx + dw/dz ≈ 0。"""
        dx = 1e-4
        x = torch.linspace(1, 50, 50)
        z = torch.full((50,), deep_wave.depth - 1.0)

        u1, _ = deep_wave.velocity(x - dx / 2, t=0.0, z=z)
        u2, _ = deep_wave.velocity(x + dx / 2, t=0.0, z=z)
        du_dx = (u2 - u1) / dx

        z_lo = z - dx / 2
        z_hi = z + dx / 2
        _, w_lo = deep_wave.velocity(x, t=0.0, z=z_lo)
        _, w_hi = deep_wave.velocity(x, t=0.0, z=z_hi)
        dw_dz = (w_hi - w_lo) / dx

        div = du_dx + dw_dz
        # 散度应接近零（允许有限差分法数值误差）
        assert div.abs().max() < 0.15

    def test_shallow_wave_amplitude(self, shallow_wave):
        """速度幅值应为有限正值。"""
        x = torch.linspace(0, 50, 20)
        z = torch.full((20,), shallow_wave.depth * 0.9)
        u, w = shallow_wave.velocity(x, t=0.0, z=z)
        assert u.abs().max() > 0
        assert torch.isfinite(u).all()
        assert torch.isfinite(w).all()


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------

class TestAiryProperties:
    def test_angular_frequency(self):
        wave = AiryWave(amplitude=1.0, depth=10.0, period=8.0)
        assert abs(wave.angular_frequency - 2.0 * math.pi / 8.0) < 1e-12

    def test_repr(self):
        wave = AiryWave(amplitude=1.0, depth=10.0, period=8.0)
        assert "AiryWave" in repr(wave)
        assert "A=1.0" in repr(wave)

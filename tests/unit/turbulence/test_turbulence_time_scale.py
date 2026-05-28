"""
Unit tests for turbulence time scale models.

Tests cover:
- KolmogorovTimeScale: tau = sqrt(nu/epsilon)
- IntegralTimeScale: tau = k/epsilon
"""

from __future__ import annotations

import pytest
import torch

from pyfoam.core.dtype import CFD_DTYPE


class TestKolmogorovTimeScale:
    """Tests for Kolmogorov (micro) time scale."""

    def test_init(self):
        from pyfoam.turbulence.turbulence_time_scale import KolmogorovTimeScale

        model = KolmogorovTimeScale(nu=1e-5)
        assert model.nu == 1e-5

    def test_compute_shape(self):
        from pyfoam.turbulence.turbulence_time_scale import KolmogorovTimeScale

        model = KolmogorovTimeScale(nu=1e-5)
        k = torch.full((10,), 1.0, dtype=CFD_DTYPE)
        epsilon = torch.full((10,), 100.0, dtype=CFD_DTYPE)

        tau = model.compute(k, epsilon)
        assert tau.shape == (10,)
        assert torch.isfinite(tau).all()

    def test_compute_positive(self):
        """Time scale is always positive."""
        from pyfoam.turbulence.turbulence_time_scale import KolmogorovTimeScale

        model = KolmogorovTimeScale(nu=1e-5)
        k = torch.full((10,), 1.0, dtype=CFD_DTYPE)
        epsilon = torch.full((10,), 100.0, dtype=CFD_DTYPE)

        tau = model.compute(k, epsilon)
        assert (tau > 0).all()

    def test_compute_formula(self):
        """tau = sqrt(nu / epsilon)."""
        from pyfoam.turbulence.turbulence_time_scale import KolmogorovTimeScale

        nu = 1e-5
        model = KolmogorovTimeScale(nu=nu)
        k = torch.full((5,), 1.0, dtype=CFD_DTYPE)
        epsilon = torch.full((5,), 100.0, dtype=CFD_DTYPE)

        tau = model.compute(k, epsilon)
        expected = (nu / 100.0) ** 0.5
        assert torch.allclose(tau, torch.full((5,), expected, dtype=CFD_DTYPE), rtol=1e-10)

    def test_compute_scales_with_epsilon(self):
        """Higher epsilon -> smaller time scale."""
        from pyfoam.turbulence.turbulence_time_scale import KolmogorovTimeScale

        model = KolmogorovTimeScale(nu=1e-5)
        k = torch.full((5,), 1.0, dtype=CFD_DTYPE)

        eps_low = torch.full((5,), 10.0, dtype=CFD_DTYPE)
        eps_high = torch.full((5,), 1000.0, dtype=CFD_DTYPE)

        tau_low = model.compute(k, eps_low)
        tau_high = model.compute(k, eps_high)

        assert tau_low.mean() > tau_high.mean()

    def test_length_scale(self):
        """Kolmogorov length scale is computed correctly."""
        from pyfoam.turbulence.turbulence_time_scale import KolmogorovTimeScale

        nu = 1e-5
        model = KolmogorovTimeScale(nu=nu)
        k = torch.full((5,), 1.0, dtype=CFD_DTYPE)
        epsilon = torch.full((5,), 100.0, dtype=CFD_DTYPE)

        eta = model.length_scale(k, epsilon)
        expected = (nu ** 3 / 100.0) ** 0.25
        assert torch.allclose(eta, torch.full((5,), expected, dtype=CFD_DTYPE), rtol=1e-10)

    def test_handles_zero_epsilon(self):
        """Handles zero epsilon without division by zero."""
        from pyfoam.turbulence.turbulence_time_scale import KolmogorovTimeScale

        model = KolmogorovTimeScale(nu=1e-5)
        k = torch.full((5,), 1.0, dtype=CFD_DTYPE)
        epsilon = torch.zeros(5, dtype=CFD_DTYPE)

        tau = model.compute(k, epsilon)
        assert torch.isfinite(tau).all()


class TestIntegralTimeScale:
    """Tests for integral (large-eddy) time scale."""

    def test_init(self):
        from pyfoam.turbulence.turbulence_time_scale import IntegralTimeScale

        model = IntegralTimeScale(C_T=1.0)
        assert model.C_T == 1.0

    def test_compute_shape(self):
        from pyfoam.turbulence.turbulence_time_scale import IntegralTimeScale

        model = IntegralTimeScale()
        k = torch.full((10,), 1.0, dtype=CFD_DTYPE)
        epsilon = torch.full((10,), 100.0, dtype=CFD_DTYPE)

        tau = model.compute(k, epsilon)
        assert tau.shape == (10,)
        assert torch.isfinite(tau).all()

    def test_compute_positive(self):
        """Time scale is always positive."""
        from pyfoam.turbulence.turbulence_time_scale import IntegralTimeScale

        model = IntegralTimeScale()
        k = torch.full((10,), 1.0, dtype=CFD_DTYPE)
        epsilon = torch.full((10,), 100.0, dtype=CFD_DTYPE)

        tau = model.compute(k, epsilon)
        assert (tau > 0).all()

    def test_compute_formula(self):
        """tau = C_T * k / epsilon."""
        from pyfoam.turbulence.turbulence_time_scale import IntegralTimeScale

        C_T = 0.5
        model = IntegralTimeScale(C_T=C_T)
        k = torch.full((5,), 2.0, dtype=CFD_DTYPE)
        epsilon = torch.full((5,), 100.0, dtype=CFD_DTYPE)

        tau = model.compute(k, epsilon)
        expected = C_T * 2.0 / 100.0
        assert torch.allclose(tau, torch.full((5,), expected, dtype=CFD_DTYPE), rtol=1e-10)

    def test_compute_scales_with_k(self):
        """Higher k -> larger time scale."""
        from pyfoam.turbulence.turbulence_time_scale import IntegralTimeScale

        model = IntegralTimeScale()
        epsilon = torch.full((5,), 100.0, dtype=CFD_DTYPE)

        k_low = torch.full((5,), 0.1, dtype=CFD_DTYPE)
        k_high = torch.full((5,), 10.0, dtype=CFD_DTYPE)

        tau_low = model.compute(k_low, epsilon)
        tau_high = model.compute(k_high, epsilon)

        assert tau_high.mean() > tau_low.mean()

    def test_length_scale(self):
        """Integral length scale: L = k^1.5 / epsilon."""
        from pyfoam.turbulence.turbulence_time_scale import IntegralTimeScale

        model = IntegralTimeScale()
        k = torch.full((5,), 4.0, dtype=CFD_DTYPE)
        epsilon = torch.full((5,), 8.0, dtype=CFD_DTYPE)

        L = model.length_scale(k, epsilon)
        expected = 4.0 ** 1.5 / 8.0  # = 8/8 = 1.0
        assert torch.allclose(L, torch.full((5,), expected, dtype=CFD_DTYPE), rtol=1e-10)

    def test_velocity_scale(self):
        """Velocity scale: u' = sqrt(2/3 * k)."""
        from pyfoam.turbulence.turbulence_time_scale import IntegralTimeScale

        model = IntegralTimeScale()
        k = torch.full((5,), 3.0, dtype=CFD_DTYPE)

        u_prime = model.velocity_scale(k)
        expected = (2.0 / 3.0 * 3.0) ** 0.5  # = sqrt(2)
        assert torch.allclose(u_prime, torch.full((5,), expected, dtype=CFD_DTYPE), rtol=1e-10)

    def test_handles_zero_epsilon(self):
        """Handles zero epsilon without division by zero."""
        from pyfoam.turbulence.turbulence_time_scale import IntegralTimeScale

        model = IntegralTimeScale()
        k = torch.full((5,), 1.0, dtype=CFD_DTYPE)
        epsilon = torch.zeros(5, dtype=CFD_DTYPE)

        tau = model.compute(k, epsilon)
        assert torch.isfinite(tau).all()

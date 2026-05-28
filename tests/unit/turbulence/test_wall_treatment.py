"""Tests for turbulence wall treatment models."""

import math
import pytest
import torch

from pyfoam.turbulence.wall_treatment import (
    WallTreatment,
    StandardWallTreatment,
    AutomaticWallTreatment,
)


class TestWallTreatmentRTS:
    """Test the RTS (Run-Time Selection) registry."""

    def test_standard_registered(self):
        """StandardWallTreatment is registered."""
        assert "standard" in WallTreatment.available_types()

    def test_automatic_registered(self):
        """AutomaticWallTreatment is registered."""
        assert "automatic" in WallTreatment.available_types()

    def test_factory_create_standard(self):
        """Factory creates StandardWallTreatment."""
        wt = WallTreatment.create("standard", nu=1e-5)
        assert isinstance(wt, StandardWallTreatment)
        assert wt.nu == 1e-5

    def test_factory_create_automatic(self):
        """Factory creates AutomaticWallTreatment."""
        wt = WallTreatment.create("automatic", nu=1e-5)
        assert isinstance(wt, AutomaticWallTreatment)
        assert wt.nu == 1e-5

    def test_factory_unknown_raises(self):
        """Unknown name raises KeyError."""
        with pytest.raises(KeyError, match="Unknown wall treatment"):
            WallTreatment.create("nonexistent")


class TestStandardWallTreatment:
    """Test the StandardWallTreatment model."""

    def test_init_defaults(self):
        """Default parameters are set correctly."""
        wt = StandardWallTreatment(nu=1.5e-5)
        assert wt.nu == 1.5e-5
        assert wt.kappa == 0.41
        assert wt.E == 9.8
        assert wt.C_mu == 0.09

    def test_custom_params(self):
        """Custom parameters are stored correctly."""
        wt = StandardWallTreatment(nu=1e-5, kappa=0.38, E=9.0, C_mu=0.08)
        assert wt.nu == 1e-5
        assert wt.kappa == 0.38
        assert wt.E == 9.0
        assert wt.C_mu == 0.08

    def test_compute_u_tau(self):
        """u_tau = C_mu^{1/4} * sqrt(k)."""
        wt = StandardWallTreatment()
        k = torch.tensor([1.0, 4.0, 9.0], dtype=torch.float64)
        u_tau = wt.compute_u_tau(k)
        expected = 0.09 ** 0.25 * torch.sqrt(k)
        assert torch.allclose(u_tau, expected, rtol=1e-10)

    def test_compute_y_plus(self):
        """y+ = u_tau * y / nu."""
        wt = StandardWallTreatment(nu=1e-5)
        u_tau = torch.tensor([0.1, 0.2], dtype=torch.float64)
        y = torch.tensor([1e-4, 2e-4], dtype=torch.float64)
        y_plus = wt.compute_y_plus(u_tau, y)
        expected = u_tau * y / 1e-5
        assert torch.allclose(y_plus, expected, rtol=1e-10)

    def test_compute_nut_positive(self):
        """nut is always non-negative."""
        wt = StandardWallTreatment(nu=1.5e-5)
        k = torch.tensor([1.0, 10.0, 100.0], dtype=torch.float64)
        y = torch.tensor([1e-3, 1e-2, 1e-1], dtype=torch.float64)
        nut = wt.compute_nut(k, y)
        assert (nut >= 0).all()

    def test_compute_nut_shape(self):
        """nut has correct shape."""
        wt = StandardWallTreatment()
        n = 5
        k = torch.ones(n, dtype=torch.float64) * 10.0
        y = torch.ones(n, dtype=torch.float64) * 1e-3
        nut = wt.compute_nut(k, y)
        assert nut.shape == (n,)

    def test_compute_k_local_equilibrium(self):
        """k = u_tau^2 / sqrt(C_mu)."""
        wt = StandardWallTreatment(C_mu=0.09)
        u_tau = torch.tensor([0.1, 0.5, 1.0], dtype=torch.float64)
        k = wt.compute_k(u_tau)
        expected = u_tau.pow(2) / math.sqrt(0.09)
        assert torch.allclose(k, expected, rtol=1e-10)

    def test_compute_epsilon_local_equilibrium(self):
        """epsilon = C_mu^{3/4} * k^{3/2} / (kappa * y)."""
        wt = StandardWallTreatment(nu=1.5e-5, kappa=0.41, C_mu=0.09)
        k = torch.tensor([1.0, 4.0], dtype=torch.float64)
        y = torch.tensor([1e-3, 1e-2], dtype=torch.float64)
        eps = wt.compute_epsilon(k, y)
        expected = 0.09 ** 0.75 * k.pow(1.5) / (0.41 * y)
        assert torch.allclose(eps, expected, rtol=1e-10)

    def test_compute_epsilon_positive(self):
        """epsilon is always positive."""
        wt = StandardWallTreatment()
        k = torch.tensor([0.01, 1.0, 100.0], dtype=torch.float64)
        y = torch.tensor([1e-4, 1e-3, 1e-2], dtype=torch.float64)
        eps = wt.compute_epsilon(k, y)
        assert (eps > 0).all()

    def test_compute_omega_formula(self):
        """omega = sqrt(k) / (C_mu^{1/4} * kappa * y)."""
        wt = StandardWallTreatment(nu=1.5e-5, kappa=0.41, C_mu=0.09)
        k = torch.tensor([1.0, 4.0], dtype=torch.float64)
        y = torch.tensor([1e-3, 1e-2], dtype=torch.float64)
        omega = wt.compute_omega(k, y)
        expected = torch.sqrt(k) / (0.09 ** 0.25 * 0.41 * y)
        assert torch.allclose(omega, expected, rtol=1e-10)

    def test_compute_omega_positive(self):
        """omega is always positive."""
        wt = StandardWallTreatment()
        k = torch.tensor([0.01, 1.0], dtype=torch.float64)
        y = torch.tensor([1e-4, 1e-3], dtype=torch.float64)
        omega = wt.compute_omega(k, y)
        assert (omega > 0).all()


class TestAutomaticWallTreatment:
    """Test the AutomaticWallTreatment model."""

    def test_init_defaults(self):
        """Default parameters are set correctly."""
        wt = AutomaticWallTreatment(nu=1.5e-5)
        assert wt.y_plus_low == 5.0
        assert wt.y_plus_high == 30.0

    def test_custom_params(self):
        """Custom y+ thresholds."""
        wt = AutomaticWallTreatment(y_plus_low=3.0, y_plus_high=20.0)
        assert wt.y_plus_low == 3.0
        assert wt.y_plus_high == 20.0

    def test_blending_factor_low_re(self):
        """Blend = 0 for y+ < y_plus_low."""
        wt = AutomaticWallTreatment(y_plus_low=5.0, y_plus_high=30.0)
        y_plus = torch.tensor([0.1, 1.0, 3.0, 4.9], dtype=torch.float64)
        blend = wt._blending_factor(y_plus)
        assert torch.allclose(blend, torch.zeros(4, dtype=torch.float64), atol=1e-10)

    def test_blending_factor_high_re(self):
        """Blend = 1 for y+ > y_plus_high."""
        wt = AutomaticWallTreatment(y_plus_low=5.0, y_plus_high=30.0)
        y_plus = torch.tensor([30.1, 50.0, 100.0, 1000.0], dtype=torch.float64)
        blend = wt._blending_factor(y_plus)
        assert torch.allclose(blend, torch.ones(4, dtype=torch.float64), atol=1e-10)

    def test_blending_factor_mid_range(self):
        """Blend is between 0 and 1 in transition region."""
        wt = AutomaticWallTreatment(y_plus_low=5.0, y_plus_high=30.0)
        y_plus = torch.tensor([10.0, 15.0, 20.0, 25.0], dtype=torch.float64)
        blend = wt._blending_factor(y_plus)
        assert (blend > 0).all()
        assert (blend < 1).all()

    def test_blending_factor_monotonic(self):
        """Blending factor increases monotonically."""
        wt = AutomaticWallTreatment(y_plus_low=5.0, y_plus_high=30.0)
        y_plus = torch.linspace(0.1, 100.0, 50, dtype=torch.float64)
        blend = wt._blending_factor(y_plus)
        for i in range(len(blend) - 1):
            assert blend[i] <= blend[i + 1] + 1e-10

    def test_compute_nut_low_re_goes_to_zero(self):
        """nut approaches zero in the low-Re region."""
        wt = AutomaticWallTreatment(nu=1e-5)
        # Very small k and y -> small y+
        k = torch.tensor([0.01], dtype=torch.float64)
        y = torch.tensor([1e-5], dtype=torch.float64)
        nut = wt.compute_nut(k, y)
        assert nut.item() < 1e-5  # very small

    def test_compute_nut_high_re_matches_standard(self):
        """nut in high-Re region matches standard wall treatment."""
        nu = 1.5e-5
        wt_auto = AutomaticWallTreatment(nu=nu)
        wt_std = StandardWallTreatment(nu=nu)

        # Large k and y -> large y+
        k = torch.tensor([100.0, 200.0], dtype=torch.float64)
        y = torch.tensor([0.01, 0.02], dtype=torch.float64)

        nut_auto = wt_auto.compute_nut(k, y)
        nut_std = wt_std.compute_nut(k, y)

        # Should be close (within blending accuracy)
        assert torch.allclose(nut_auto, nut_std, rtol=0.1)

    def test_compute_k_same_as_standard(self):
        """compute_k is the same as standard wall treatment."""
        wt = AutomaticWallTreatment()
        u_tau = torch.tensor([0.1, 0.5, 1.0], dtype=torch.float64)
        k = wt.compute_k(u_tau)

        wt_std = StandardWallTreatment()
        k_std = wt_std.compute_k(u_tau)

        assert torch.allclose(k, k_std, rtol=1e-10)

    def test_compute_epsilon_low_re_formula(self):
        """Low-Re epsilon uses 2*nu*k/y^2."""
        nu = 1e-5
        wt = AutomaticWallTreatment(nu=nu, y_plus_low=5.0, y_plus_high=30.0)
        # Small k and y to ensure low y+
        k = torch.tensor([0.001], dtype=torch.float64)
        y = torch.tensor([1e-5], dtype=torch.float64)
        eps = wt.compute_epsilon(k, y)
        # Should be close to low-Re formula: 2*nu*k/y^2
        expected_low_re = 2.0 * nu * k / y.pow(2)
        # Since y+ is very low, blend should be ~0, so eps ≈ eps_low_re
        assert torch.allclose(eps, expected_low_re, rtol=0.1)

    def test_compute_omega_positive(self):
        """omega is always positive."""
        wt = AutomaticWallTreatment(nu=1e-5)
        k = torch.tensor([0.01, 1.0, 100.0], dtype=torch.float64)
        y = torch.tensor([1e-4, 1e-3, 1e-2], dtype=torch.float64)
        omega = wt.compute_omega(k, y)
        assert (omega > 0).all()

    def test_batch_processing(self):
        """Works correctly with batched tensors."""
        wt = AutomaticWallTreatment()
        n = 50
        k = torch.rand(n, dtype=torch.float64) * 10.0 + 0.01
        y = torch.rand(n, dtype=torch.float64) * 0.01 + 1e-5

        nut = wt.compute_nut(k, y)
        assert nut.shape == (n,)
        assert (nut >= 0).all()

        eps = wt.compute_epsilon(k, y)
        assert eps.shape == (n,)
        assert (eps > 0).all()

        omega = wt.compute_omega(k, y)
        assert omega.shape == (n,)
        assert (omega > 0).all()

"""Tests for incompressible drift-flux model.

Tests cover:
- IncompressibleDriftFlux mixture properties (density, viscosity)
- Slip velocity computation (hindered settling, direction)
- Drift flux computation
- solve_timestep convenience method
"""

import pytest
import torch

from pyfoam.multiphase.incompressible_drift_flux import IncompressibleDriftFlux


class TestIncompressibleDriftFluxInit:
    """Constructor and default parameter tests."""

    def test_defaults(self):
        model = IncompressibleDriftFlux()
        assert model.rho_d == pytest.approx(1000.0)
        assert model.rho_c == pytest.approx(1.225)
        assert model.mu_d == pytest.approx(1.002e-3)
        assert model.mu_c == pytest.approx(1.8e-5)
        assert model.particle_diameter == pytest.approx(1e-3)
        assert model.richardson_zaki_n == pytest.approx(2.4)
        assert model.alpha_max == pytest.approx(0.63)

    def test_custom_params(self):
        model = IncompressibleDriftFlux(
            rho_d=2500.0, rho_c=1000.0,
            mu_d=1e-3, mu_c=1e-3,
            particle_diameter=5e-4,
            richardson_zaki_n=3.0,
            alpha_max=0.5,
        )
        assert model.rho_d == pytest.approx(2500.0)
        assert model.richardson_zaki_n == pytest.approx(3.0)


class TestMixtureProperties:
    """Mixture density and viscosity tests."""

    def test_mixture_density_pure_continuous(self):
        model = IncompressibleDriftFlux(rho_d=1000.0, rho_c=1.0)
        alpha = torch.zeros(5, dtype=torch.float64)
        rho_m = model.mixture_density(alpha)
        assert torch.allclose(rho_m, torch.full((5,), 1.0, dtype=torch.float64))

    def test_mixture_density_pure_dispersed(self):
        model = IncompressibleDriftFlux(rho_d=1000.0, rho_c=1.0)
        alpha = torch.ones(5, dtype=torch.float64)
        rho_m = model.mixture_density(alpha)
        assert torch.allclose(rho_m, torch.full((5,), 1000.0, dtype=torch.float64))

    def test_mixture_density_mixed(self):
        model = IncompressibleDriftFlux(rho_d=1000.0, rho_c=1.0)
        alpha = torch.full((5,), 0.5, dtype=torch.float64)
        rho_m = model.mixture_density(alpha)
        expected = 0.5 * 1000.0 + 0.5 * 1.0
        assert torch.allclose(rho_m, torch.full((5,), expected, dtype=torch.float64))

    def test_mixture_viscosity_formula(self):
        model = IncompressibleDriftFlux(mu_d=2e-3, mu_c=1e-3)
        alpha = torch.full((5,), 0.4, dtype=torch.float64)
        mu_m = model.mixture_viscosity(alpha)
        expected = 0.4 * 2e-3 + 0.6 * 1e-3
        assert torch.allclose(mu_m, torch.full((5,), expected, dtype=torch.float64))

    def test_mixture_density_clamped_alpha(self):
        """Alpha values outside [0,1] should be clamped."""
        model = IncompressibleDriftFlux(rho_d=1000.0, rho_c=1.0)
        alpha = torch.tensor([-0.1, 1.5], dtype=torch.float64)
        rho_m = model.mixture_density(alpha)
        assert rho_m[0] == pytest.approx(1.0)
        assert rho_m[1] == pytest.approx(1000.0)


class TestSlipVelocity:
    """Slip velocity computation tests."""

    def test_shape(self):
        model = IncompressibleDriftFlux()
        alpha = torch.zeros(10, dtype=torch.float64)
        U = model.compute_slip_velocity(alpha)
        assert U.shape == (10, 3)

    def test_direction_heavier_dispersed(self):
        """Heavier dispersed phase settles in gravity direction."""
        model = IncompressibleDriftFlux(rho_d=2000.0, rho_c=1.0)
        alpha = torch.zeros(5, dtype=torch.float64)
        U = model.compute_slip_velocity(alpha)
        assert (U[:, 2] < 0).all()  # downward for default gravity

    def test_magnitude_decreases_with_alpha(self):
        """Hindered settling: slip velocity decreases with increasing alpha."""
        model = IncompressibleDriftFlux(rho_d=2000.0, rho_c=1.0)
        gravity = torch.tensor([0.0, 0.0, -9.81], dtype=torch.float64)
        alpha_low = torch.full((1,), 0.01, dtype=torch.float64)
        alpha_high = torch.full((1,), 0.5, dtype=torch.float64)
        U_low = model.compute_slip_velocity(alpha_low, gravity)
        U_high = model.compute_slip_velocity(alpha_high, gravity)
        assert abs(U_low[0, 2].item()) > abs(U_high[0, 2].item())

    def test_finite_output(self):
        model = IncompressibleDriftFlux()
        alpha = torch.rand(20, dtype=torch.float64)
        U = model.compute_slip_velocity(alpha)
        assert torch.isfinite(U).all()

    def test_custom_gravity(self):
        model = IncompressibleDriftFlux(rho_d=2000.0, rho_c=1.0)
        alpha = torch.zeros(5, dtype=torch.float64)
        g = torch.tensor([9.81, 0.0, 0.0], dtype=torch.float64)
        U = model.compute_slip_velocity(alpha, gravity=g)
        # Should have positive x-component
        assert (U[:, 0] > 0).all()
        assert torch.allclose(U[:, 1], torch.zeros(5, dtype=torch.float64))


class TestDriftFlux:
    """Drift flux computation tests."""

    def test_shape(self):
        model = IncompressibleDriftFlux()
        alpha = torch.rand(10, dtype=torch.float64)
        U_slip = torch.randn(10, 3, dtype=torch.float64)
        J = model.compute_drift_flux(alpha, U_slip)
        assert J.shape == (10, 3)

    def test_zero_at_alpha_zero(self):
        model = IncompressibleDriftFlux()
        alpha = torch.zeros(5, dtype=torch.float64)
        U_slip = torch.ones(5, 3, dtype=torch.float64)
        J = model.compute_drift_flux(alpha, U_slip)
        assert torch.allclose(J, torch.zeros_like(J))

    def test_drift_flux_factor_peaks_at_half_alpha(self):
        """Drift flux factor alpha*(1-alpha) peaks at alpha=0.5."""
        model = IncompressibleDriftFlux(alpha_max=1.0)
        U_slip = torch.ones(1, 3, dtype=torch.float64)
        # Use different alpha values to find peak
        J_05 = model.compute_drift_flux(torch.tensor([0.5], dtype=torch.float64), U_slip)
        J_03 = model.compute_drift_flux(torch.tensor([0.3], dtype=torch.float64), U_slip)
        J_07 = model.compute_drift_flux(torch.tensor([0.7], dtype=torch.float64), U_slip)
        assert J_05.norm() > J_03.norm()
        assert J_05.norm() > J_07.norm()


class TestSolveTimestep:
    """Convenience solve_timestep tests."""

    def test_returns_all_keys(self):
        model = IncompressibleDriftFlux()
        alpha = torch.rand(10, dtype=torch.float64)
        result = model.solve_timestep(alpha)
        assert set(result.keys()) == {"rho_m", "mu_m", "U_slip", "J"}

    def test_shapes(self):
        model = IncompressibleDriftFlux()
        alpha = torch.rand(10, dtype=torch.float64)
        result = model.solve_timestep(alpha)
        assert result["rho_m"].shape == (10,)
        assert result["mu_m"].shape == (10,)
        assert result["U_slip"].shape == (10, 3)
        assert result["J"].shape == (10, 3)

    def test_all_finite(self):
        model = IncompressibleDriftFlux()
        alpha = torch.rand(20, dtype=torch.float64)
        result = model.solve_timestep(alpha)
        for v in result.values():
            assert torch.isfinite(v).all()

"""Tests for drift-flux models.

Tests cover:
- DriftFluxModel RTS registry
- SimpleDriftFlux: Stokes settling, drift flux, alpha clamping
- GeneralDriftFlux: Richardson-Zaki, Zuber-Findlay, distribution parameter
"""

from __future__ import annotations

import pytest
import torch

from pyfoam.multiphase.drift_flux_models import (
    DriftFluxModel,
    SimpleDriftFlux,
    GeneralDriftFlux,
)


# ===========================================================================
# Registry tests
# ===========================================================================


class TestDriftFluxModelRegistry:
    """RTS registry tests for DriftFluxModel."""

    def test_simple_registered(self):
        assert "simple" in DriftFluxModel.available_types()

    def test_general_registered(self):
        assert "general" in DriftFluxModel.available_types()

    def test_factory_create_simple(self):
        model = DriftFluxModel.create("simple", d=0.002)
        assert isinstance(model, SimpleDriftFlux)
        assert model.d == pytest.approx(0.002)

    def test_factory_create_general(self):
        model = DriftFluxModel.create("general", V0=[0, 0, -0.2], n_exp=3.0)
        assert isinstance(model, GeneralDriftFlux)
        assert model.n_exp == pytest.approx(3.0)

    def test_factory_unknown_raises(self):
        with pytest.raises(KeyError, match="Unknown DriftFluxModel"):
            DriftFluxModel.create("nonexistentModel")

    def test_available_types_sorted(self):
        types = DriftFluxModel.available_types()
        assert types == sorted(types)

    def test_duplicate_registration_raises(self):
        with pytest.raises(ValueError, match="already registered"):
            @DriftFluxModel.register("simple")
            class _Duplicate:
                pass


# ===========================================================================
# SimpleDriftFlux tests
# ===========================================================================


class TestSimpleDriftFlux:
    """Tests for the simple algebraic slip model."""

    def test_default_parameters(self):
        model = SimpleDriftFlux()
        assert model.d == pytest.approx(0.001)
        assert model.alpha_max == pytest.approx(0.6)

    def test_custom_parameters(self):
        model = SimpleDriftFlux(d=0.005, alpha_max=0.5)
        assert model.d == pytest.approx(0.005)
        assert model.alpha_max == pytest.approx(0.5)

    def test_slip_velocity_shape(self):
        model = SimpleDriftFlux(d=0.001)
        alpha = torch.zeros(10, dtype=torch.float64)
        gravity = torch.tensor([0.0, 0.0, -9.81], dtype=torch.float64)
        U_slip = model.compute_slip_velocity(alpha, 1000.0, 1.0, 1e-3, gravity, 10)
        assert U_slip.shape == (10, 3)

    def test_slip_velocity_direction(self):
        """Slip velocity should be in the gravity direction (positive for heavier particles)."""
        model = SimpleDriftFlux(d=0.001)
        alpha = torch.zeros(5, dtype=torch.float64)
        gravity = torch.tensor([0.0, 0.0, -9.81], dtype=torch.float64)
        U_slip = model.compute_slip_velocity(alpha, 1000.0, 1.0, 1e-3, gravity, 5)
        # For rho_d > rho_c, slip is in gravity direction (negative z)
        assert (U_slip[:, 2] < 0).all()

    def test_slip_velocity_magnitude(self):
        """Slip velocity magnitude matches Stokes law."""
        model = SimpleDriftFlux(d=0.002)
        alpha = torch.zeros(3, dtype=torch.float64)
        gravity = torch.tensor([0.0, 0.0, -9.81], dtype=torch.float64)
        U_slip = model.compute_slip_velocity(alpha, 1500.0, 1000.0, 1e-3, gravity, 3)
        # Stokes: U = (1500 - 1000) * 9.81 * (0.002)^2 / (18 * 1e-3)
        expected = 500.0 * 9.81 * 4e-6 / 0.018
        assert abs(abs(U_slip[0, 2].item()) - expected) / expected < 0.01

    def test_slip_independent_of_alpha(self):
        """Simple model: slip velocity is uniform (independent of alpha)."""
        model = SimpleDriftFlux(d=0.001)
        alpha_low = torch.full((5,), 0.01, dtype=torch.float64)
        alpha_high = torch.full((5,), 0.5, dtype=torch.float64)
        gravity = torch.tensor([0.0, 0.0, -9.81], dtype=torch.float64)
        U_low = model.compute_slip_velocity(alpha_low, 1000.0, 1.0, 1e-3, gravity, 5)
        U_high = model.compute_slip_velocity(alpha_high, 1000.0, 1.0, 1e-3, gravity, 5)
        assert torch.allclose(U_low, U_high)

    def test_drift_flux_shape(self):
        model = SimpleDriftFlux()
        alpha = torch.rand(10, dtype=torch.float64)
        U_slip = torch.randn(10, 3, dtype=torch.float64)
        J = model.compute_drift_flux(alpha, U_slip, 10)
        assert J.shape == (10, 3)

    def test_drift_flux_zero_at_alpha_zero(self):
        """Drift flux should be zero at alpha=0."""
        model = SimpleDriftFlux()
        alpha = torch.zeros(5, dtype=torch.float64)
        U_slip = torch.ones(5, 3, dtype=torch.float64)
        J = model.compute_drift_flux(alpha, U_slip, 5)
        assert torch.allclose(J, torch.zeros_like(J))

    def test_drift_flux_zero_at_alpha_one(self):
        """Drift flux should be zero at alpha=alpha_max (since clamping gives (1-1)=0 at alpha_max)."""
        # Use alpha_max=1.0 so that alpha=1.0 is not clamped
        model = SimpleDriftFlux(alpha_max=1.0)
        alpha = torch.ones(5, dtype=torch.float64)
        U_slip = torch.ones(5, 3, dtype=torch.float64)
        J = model.compute_drift_flux(alpha, U_slip, 5)
        assert torch.allclose(J, torch.zeros_like(J))

    def test_drift_flux_max_at_half_alpha(self):
        """Drift flux magnitude is maximised at alpha = 0.5."""
        model = SimpleDriftFlux()
        U_slip = torch.ones(1, 3, dtype=torch.float64)

        J_mid = model.compute_drift_flux(torch.tensor([0.5]), U_slip, 1)
        J_low = model.compute_drift_flux(torch.tensor([0.1]), U_slip, 1)
        J_high = model.compute_drift_flux(torch.tensor([0.9]), U_slip, 1)

        assert J_mid.norm() > J_low.norm()
        assert J_mid.norm() > J_high.norm()

    def test_drift_flux_alpha_clamped(self):
        """Alpha is clamped to alpha_max."""
        model = SimpleDriftFlux(alpha_max=0.5)
        alpha = torch.tensor([0.7], dtype=torch.float64)
        U_slip = torch.ones(1, 3, dtype=torch.float64)
        J = model.compute_drift_flux(alpha, U_slip, 1)
        # At alpha_max=0.5: factor = 0.5 * 0.5 = 0.25
        expected = 0.25 * U_slip
        assert torch.allclose(J, expected)


# ===========================================================================
# GeneralDriftFlux tests
# ===========================================================================


class TestGeneralDriftFlux:
    """Tests for the general drift-flux model."""

    def test_default_parameters(self):
        model = GeneralDriftFlux()
        assert model.V0 == [0.0, 0.0, -0.1]
        assert model.n_exp == pytest.approx(2.0)
        assert model.C0 == pytest.approx(1.0)
        assert model.alpha_max == pytest.approx(0.6)

    def test_custom_parameters(self):
        model = GeneralDriftFlux(V0=[0.1, 0, -0.2], n_exp=3.0, C0=1.2, alpha_max=0.5)
        assert model.V0 == [0.1, 0, -0.2]
        assert model.n_exp == pytest.approx(3.0)
        assert model.C0 == pytest.approx(1.2)
        assert model.alpha_max == pytest.approx(0.5)

    def test_slip_velocity_shape(self):
        model = GeneralDriftFlux()
        alpha = torch.zeros(10, dtype=torch.float64)
        gravity = torch.tensor([0.0, 0.0, -9.81], dtype=torch.float64)
        U_slip = model.compute_slip_velocity(alpha, 1000.0, 1.0, 1e-3, gravity, 10)
        assert U_slip.shape == (10, 3)

    def test_slip_at_zero_alpha(self):
        """At zero alpha, slip velocity equals V0."""
        model = GeneralDriftFlux(V0=[0.0, 0.0, -0.15])
        alpha = torch.zeros(5, dtype=torch.float64)
        gravity = torch.tensor([0.0, 0.0, -9.81], dtype=torch.float64)
        U_slip = model.compute_slip_velocity(alpha, 1000.0, 1.0, 1e-3, gravity, 5)
        expected = torch.tensor([0.0, 0.0, -0.15], dtype=torch.float64)
        assert torch.allclose(U_slip[0], expected)

    def test_slip_decreases_with_alpha(self):
        """Slip velocity decreases with increasing alpha (Richardson-Zaki)."""
        model = GeneralDriftFlux(V0=[0.0, 0.0, -0.1], n_exp=2.0)
        gravity = torch.tensor([0.0, 0.0, -9.81], dtype=torch.float64)

        alpha_low = torch.full((1,), 0.1, dtype=torch.float64)
        alpha_high = torch.full((1,), 0.5, dtype=torch.float64)

        U_low = model.compute_slip_velocity(alpha_low, 1000.0, 1.0, 1e-3, gravity, 1)
        U_high = model.compute_slip_velocity(alpha_high, 1000.0, 1.0, 1e-3, gravity, 1)

        # At higher alpha, slip should be smaller (more hindered)
        assert abs(U_low[0, 2].item()) > abs(U_high[0, 2].item())

    def test_slip_richardson_zaki_exponent(self):
        """Slip velocity follows (1-alpha)^n."""
        model = GeneralDriftFlux(V0=[0.0, 0.0, -1.0], n_exp=2.0)
        alpha = torch.tensor([0.3], dtype=torch.float64)
        gravity = torch.tensor([0.0, 0.0, -9.81], dtype=torch.float64)
        U_slip = model.compute_slip_velocity(alpha, 1000.0, 1.0, 1e-3, gravity, 1)
        # (1 - 0.3)^2 = 0.49
        expected_z = -1.0 * 0.49
        assert abs(U_slip[0, 2].item() - expected_z) < 1e-10

    def test_drift_flux_without_mixture_velocity(self):
        """Drift flux without U_m: only alpha*(1-alpha)*U_slip."""
        model = GeneralDriftFlux(C0=0.0)
        alpha = torch.tensor([0.3], dtype=torch.float64)
        U_slip = torch.tensor([[0.0, 0.0, -0.5]], dtype=torch.float64)
        J = model.compute_drift_flux(alpha, U_slip, 1)
        # factor = 0.3 * 0.7 = 0.21
        expected = 0.21 * U_slip
        assert torch.allclose(J, expected)

    def test_drift_flux_with_mixture_velocity(self):
        """Drift flux with U_m includes distribution parameter."""
        model = GeneralDriftFlux(C0=1.2)
        alpha = torch.tensor([0.3], dtype=torch.float64)
        U_slip = torch.tensor([[0.0, 0.0, -0.5]], dtype=torch.float64)
        U_m = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
        J = model.compute_drift_flux(alpha, U_slip, 1, U_m=U_m)
        # J = 0.21 * (-0.5) + 1.2 * 0.3 * 1.0
        expected_z = 0.21 * (-0.5)
        expected_x = 1.2 * 0.3 * 1.0
        assert abs(J[0, 0].item() - expected_x) < 1e-10
        assert abs(J[0, 2].item() - expected_z) < 1e-10

    def test_drift_flux_zero_at_boundaries(self):
        """Drift flux is zero at alpha=0 and at alpha_max."""
        # Use alpha_max=1.0 so alpha=1.0 is not clamped
        model = GeneralDriftFlux(C0=0.0, alpha_max=1.0)
        U_slip = torch.ones(1, 3, dtype=torch.float64)

        J_0 = model.compute_drift_flux(torch.tensor([0.0]), U_slip, 1)
        J_1 = model.compute_drift_flux(torch.tensor([1.0]), U_slip, 1)

        assert torch.allclose(J_0, torch.zeros_like(J_0))
        assert torch.allclose(J_1, torch.zeros_like(J_1))

    def test_slip_velocity_finite(self):
        """All slip velocity values are finite."""
        model = GeneralDriftFlux()
        alpha = torch.rand(20, dtype=torch.float64)
        gravity = torch.tensor([0.0, 0.0, -9.81], dtype=torch.float64)
        U_slip = model.compute_slip_velocity(alpha, 1000.0, 1.0, 1e-3, gravity, 20)
        assert torch.isfinite(U_slip).all()

    def test_drift_flux_finite(self):
        """All drift flux values are finite."""
        model = GeneralDriftFlux()
        alpha = torch.rand(20, dtype=torch.float64)
        U_slip = torch.randn(20, 3, dtype=torch.float64)
        J = model.compute_drift_flux(alpha, U_slip, 20)
        assert torch.isfinite(J).all()

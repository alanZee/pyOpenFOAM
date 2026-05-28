"""
Tests for enhanced turbulence damping2 models.

Tests cover:
- TurbulenceDamping2Model ABC and RTS registry
- WolfhardtDamping registration and factory
- Default and custom parameters
- Interface indicator computation
- k damping with y+ dependence
- epsilon damping with y+ dependence
- omega damping with y+ dependence
- Without y+ falls back to interface-only damping
- Viscous sublayer (small y+) produces strong damping
- Log-law region (large y+) produces weak near-wall damping
"""

import pytest
import torch

from pyfoam.multiphase.turbulence_damping_2 import (
    TurbulenceDamping2Model,
    WolfhardtDamping,
)


class TestTurbulenceDamping2ABC:
    """Abstract base class tests."""

    def test_registry_not_empty(self):
        """wolfhardtDamping is registered."""
        assert "wolfhardtDamping" in TurbulenceDamping2Model.available_types()

    def test_factory_create(self):
        model = TurbulenceDamping2Model.create("wolfhardtDamping")
        assert isinstance(model, WolfhardtDamping)

    def test_factory_unknown_raises(self):
        with pytest.raises(KeyError, match="Unknown turbulence damping2 model"):
            TurbulenceDamping2Model.create("nonexistent")


class TestWolfhardtDamping:
    """WolfhardtDamping model tests."""

    def test_registration(self):
        assert "wolfhardtDamping" in TurbulenceDamping2Model.available_types()

    def test_factory_creation(self):
        model = TurbulenceDamping2Model.create("wolfhardtDamping")
        assert isinstance(model, WolfhardtDamping)

    def test_default_params(self):
        model = WolfhardtDamping()
        assert model.damping_coeff == 5.0
        assert model.alpha_min == 0.01
        assert model.alpha_max == 0.99
        assert model.y_plus_ref == 50.0

    def test_custom_params(self):
        model = WolfhardtDamping(
            damping_coeff=10.0, alpha_min=0.05, alpha_max=0.95, y_plus_ref=100.0,
        )
        assert model.damping_coeff == 10.0
        assert model.alpha_min == 0.05
        assert model.alpha_max == 0.95
        assert model.y_plus_ref == 100.0

    def test_interface_indicator_pure_phases(self):
        model = WolfhardtDamping()
        alpha = torch.tensor([0.0, 1.0], dtype=torch.float64)
        ind = model.compute_interface_indicator(alpha)
        assert torch.allclose(ind, torch.zeros(2, dtype=torch.float64), atol=1e-10)

    def test_interface_indicator_at_interface(self):
        model = WolfhardtDamping()
        alpha = torch.tensor([0.5], dtype=torch.float64)
        ind = model.compute_interface_indicator(alpha)
        assert torch.allclose(ind, torch.tensor([1.0], dtype=torch.float64), atol=1e-10)

    def test_damp_k_pure_phases_unchanged(self):
        """Pure phases (alpha=0 or 1) have no interface damping."""
        model = WolfhardtDamping(damping_coeff=5.0)
        alpha = torch.tensor([0.0, 1.0], dtype=torch.float64)
        k = torch.tensor([100.0, 100.0], dtype=torch.float64)
        k_damped = model.damp_k(alpha, k)
        assert torch.allclose(k_damped, k, atol=1e-10)

    def test_damp_k_at_interface_no_yplus(self):
        """Interface region damped even without y+."""
        model = WolfhardtDamping(damping_coeff=5.0)
        alpha = torch.tensor([0.5], dtype=torch.float64)
        k = torch.tensor([100.0], dtype=torch.float64)
        k_damped = model.damp_k(alpha, k)
        # 4 * 0.5 * 0.5 * 5 = 5, exp(-5) ~ 0.0067
        assert k_damped[0] < 1.0

    def test_damp_k_stronger_near_wall(self):
        """Small y+ produces stronger damping than large y+."""
        model = WolfhardtDamping(damping_coeff=5.0, y_plus_ref=50.0)
        alpha = torch.tensor([0.5, 0.5], dtype=torch.float64)
        k = torch.tensor([100.0, 100.0], dtype=torch.float64)
        y_plus = torch.tensor([1.0, 200.0], dtype=torch.float64)

        k_damped = model.damp_k(alpha, k, y_plus=y_plus)
        # Small y+: exp(-(1/50)^2) ~ 1.0 (wall damping ~ 1)
        # Large y+: exp(-(200/50)^2) = exp(-16) ~ 1.1e-7 (wall damping ~ 0)
        # Total f = coeff * f_interface * f_wall
        # For small y+: f = 5 * 1 * ~1 = 5 => exp(-5) ~ 0.0067
        # For large y+: f = 5 * 1 * ~0 = ~0 => exp(0) ~ 1
        assert k_damped[0] < k_damped[1]

    def test_damp_k_large_yplus_no_wall_damping(self):
        """Large y+ eliminates wall damping, leaving interface damping."""
        model = WolfhardtDamping(damping_coeff=5.0, y_plus_ref=50.0)
        alpha = torch.tensor([0.5], dtype=torch.float64)
        k = torch.tensor([100.0], dtype=torch.float64)
        y_plus = torch.tensor([1000.0], dtype=torch.float64)
        k_damped = model.damp_k(alpha, k, y_plus=y_plus)
        # Wall damping factor: exp(-(1000/50)^2) ~ 0
        # Total f ~ 0, so k_damped ~ k
        assert k_damped[0] > 99.0

    def test_damp_epsilon_reduces_at_interface(self):
        model = WolfhardtDamping(damping_coeff=5.0)
        alpha = torch.tensor([0.5, 0.0, 1.0], dtype=torch.float64)
        eps = torch.tensor([50.0, 50.0, 50.0], dtype=torch.float64)
        eps_damped = model.damp_epsilon(alpha, eps)
        assert eps_damped[0] < 1.0
        assert torch.allclose(eps_damped[1], torch.tensor(50.0, dtype=torch.float64))
        assert torch.allclose(eps_damped[2], torch.tensor(50.0, dtype=torch.float64))

    def test_damp_omega_reduces_at_interface(self):
        model = WolfhardtDamping(damping_coeff=5.0)
        alpha = torch.tensor([0.5, 0.0], dtype=torch.float64)
        omega = torch.tensor([200.0, 200.0], dtype=torch.float64)
        omega_damped = model.damp_omega(alpha, omega)
        # Interface is damped (less than pure phase)
        assert omega_damped[0] < omega_damped[1]
        # Pure phase unchanged
        assert torch.allclose(omega_damped[1], torch.tensor(200.0, dtype=torch.float64))

    def test_damp_epsilon_with_yplus(self):
        """epsilon damping also uses y+."""
        model = WolfhardtDamping(damping_coeff=5.0, y_plus_ref=50.0)
        alpha = torch.tensor([0.5, 0.5], dtype=torch.float64)
        eps = torch.tensor([50.0, 50.0], dtype=torch.float64)
        y_plus = torch.tensor([1.0, 200.0], dtype=torch.float64)
        eps_damped = model.damp_epsilon(alpha, eps, y_plus=y_plus)
        # Small y+: strong damping
        assert eps_damped[0] < eps_damped[1]

    def test_damp_omega_with_yplus(self):
        """omega damping also uses y+."""
        model = WolfhardtDamping(damping_coeff=5.0, y_plus_ref=50.0)
        alpha = torch.tensor([0.5, 0.5], dtype=torch.float64)
        omega = torch.tensor([200.0, 200.0], dtype=torch.float64)
        y_plus = torch.tensor([1.0, 200.0], dtype=torch.float64)
        omega_damped = model.damp_omega(alpha, omega, y_plus=y_plus)
        assert omega_damped[0] < omega_damped[1]

    def test_zero_damping_coeff_no_effect(self):
        model = WolfhardtDamping(damping_coeff=0.0)
        alpha = torch.tensor([0.5, 0.3, 0.7], dtype=torch.float64)
        k = torch.tensor([100.0, 200.0, 300.0], dtype=torch.float64)
        k_damped = model.damp_k(alpha, k)
        assert torch.allclose(k_damped, k, atol=1e-10)

    def test_batch_processing(self):
        model = WolfhardtDamping(damping_coeff=5.0)
        n = 50
        alpha = torch.rand(n, dtype=torch.float64)
        k = torch.ones(n, dtype=torch.float64) * 10.0
        y_plus = torch.rand(n, dtype=torch.float64) * 100.0
        k_damped = model.damp_k(alpha, k, y_plus=y_plus)
        assert k_damped.shape == (n,)
        assert (k_damped <= k + 1e-10).all()

    def test_outside_alpha_range_no_damping(self):
        """Alpha outside [alpha_min, alpha_max] should not be damped."""
        model = WolfhardtDamping(damping_coeff=5.0, alpha_min=0.1, alpha_max=0.9)
        alpha = torch.tensor([0.001, 0.999], dtype=torch.float64)
        k = torch.tensor([100.0, 100.0], dtype=torch.float64)
        y_plus = torch.tensor([1.0, 1.0], dtype=torch.float64)
        k_damped = model.damp_k(alpha, k, y_plus=y_plus)
        assert torch.allclose(k_damped, k, atol=1e-10)

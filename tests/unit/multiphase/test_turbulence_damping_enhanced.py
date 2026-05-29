"""Tests for enhanced turbulence damping models.

Tests cover:
- TurbulenceDampingEnhancedModel RTS registry
- GradientDamping: gradient-based damping and fallback
- ExponentialBlendedDamping: bell-shaped profile
"""

import pytest
import torch

from pyfoam.multiphase.turbulence_damping_enhanced import (
    TurbulenceDampingEnhancedModel,
    GradientDamping,
    ExponentialBlendedDamping,
)


class TestTurbulenceDampingEnhancedRegistry:
    """RTS registration tests."""

    def test_gradient_damping_registered(self):
        assert "gradientDamping" in TurbulenceDampingEnhancedModel.available_types()

    def test_exponential_blended_registered(self):
        assert "exponentialBlendedDamping" in TurbulenceDampingEnhancedModel.available_types()

    def test_factory_create_gradient(self):
        model = TurbulenceDampingEnhancedModel.create("gradientDamping", damping_coeff=8.0)
        assert isinstance(model, GradientDamping)
        assert model.damping_coeff == pytest.approx(8.0)

    def test_factory_create_blended(self):
        model = TurbulenceDampingEnhancedModel.create("exponentialBlendedDamping")
        assert isinstance(model, ExponentialBlendedDamping)

    def test_factory_unknown_raises(self):
        with pytest.raises(KeyError, match="Unknown enhanced damping"):
            TurbulenceDampingEnhancedModel.create("nonexistent")

    def test_duplicate_registration_raises(self):
        with pytest.raises(ValueError, match="already registered"):
            @TurbulenceDampingEnhancedModel.register("gradientDamping")
            class _Dup:
                pass

    def test_available_types_sorted(self):
        types = TurbulenceDampingEnhancedModel.available_types()
        assert types == sorted(types)


class TestGradientDamping:
    """Tests for GradientDamping."""

    def test_default_params(self):
        model = GradientDamping()
        assert model.damping_coeff == pytest.approx(10.0)
        assert model.alpha_min == pytest.approx(0.01)
        assert model.alpha_max == pytest.approx(0.99)
        assert model.grad_threshold == pytest.approx(0.1)
        assert model.grad_max == pytest.approx(1.0)

    def test_damp_k_no_gradient(self):
        """Without grad_alpha, falls back to alpha-based proxy."""
        model = GradientDamping(damping_coeff=5.0, alpha_min=0.01, alpha_max=0.99)
        alpha = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
        k = torch.ones(3, dtype=torch.float64)
        k_damped = model.damp_k(alpha, k)
        assert k_damped.shape == (3,)
        # At alpha=0 (pure phase), no damping
        assert float(k_damped[0].item()) == pytest.approx(1.0, abs=1e-6)
        # At alpha=1 (pure phase), no damping
        assert float(k_damped[2].item()) == pytest.approx(1.0, abs=1e-6)
        # At alpha=0.5 (interface), should be damped
        assert float(k_damped[1].item()) < 1.0

    def test_damp_k_with_gradient(self):
        """With grad_alpha, uses gradient-based damping."""
        model = GradientDamping(damping_coeff=5.0, alpha_min=0.01, alpha_max=0.99)
        alpha = torch.tensor([0.1, 0.5, 0.9], dtype=torch.float64)
        k = torch.ones(3, dtype=torch.float64)
        grad_alpha = torch.tensor([
            [0.01, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.01, 0.0, 0.0],
        ], dtype=torch.float64)
        k_damped = model.damp_k(alpha, k, grad_alpha=grad_alpha)
        assert k_damped.shape == (3,)
        # Middle cell with large gradient: most damped
        assert k_damped[1] < k_damped[0]
        assert k_damped[1] < k_damped[2]

    def test_damp_epsilon(self):
        model = GradientDamping(damping_coeff=5.0, alpha_min=0.01, alpha_max=0.99)
        alpha = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
        eps = torch.ones(3, dtype=torch.float64) * 100.0
        eps_damped = model.damp_epsilon(alpha, eps)
        assert eps_damped.shape == (3,)
        assert float(eps_damped[0].item()) == pytest.approx(100.0, abs=1e-3)

    def test_damp_omega(self):
        model = GradientDamping(damping_coeff=5.0, alpha_min=0.01, alpha_max=0.99)
        alpha = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
        omega = torch.ones(3, dtype=torch.float64) * 50.0
        omega_damped = model.damp_omega(alpha, omega)
        assert omega_damped.shape == (3,)
        assert float(omega_damped[0].item()) == pytest.approx(50.0, abs=1e-3)

    def test_pure_phases_unaffected(self):
        """Pure phases should not be damped."""
        model = GradientDamping(damping_coeff=100.0)
        alpha = torch.tensor([0.0, 1.0], dtype=torch.float64)
        k = torch.ones(2, dtype=torch.float64) * 10.0
        k_damped = model.damp_k(alpha, k)
        assert torch.allclose(k_damped, k, atol=1e-6)

    def test_damping_factor_range(self):
        """Damping factor should be in [0, damping_coeff]."""
        model = GradientDamping(damping_coeff=8.0)
        alpha = torch.rand(50, dtype=torch.float64).clamp(0.01, 0.99)
        f = model.compute_damping_factor(alpha)
        assert (f >= 0).all()
        assert (f <= 8.0 + 1e-6).all()


class TestExponentialBlendedDamping:
    """Tests for ExponentialBlendedDamping."""

    def test_default_params(self):
        model = ExponentialBlendedDamping()
        assert model.damping_coeff == pytest.approx(10.0)
        assert model.width == pytest.approx(0.1)

    def test_custom_params(self):
        model = ExponentialBlendedDamping(damping_coeff=5.0, width=0.2)
        assert model.damping_coeff == pytest.approx(5.0)
        assert model.width == pytest.approx(0.2)

    def test_bell_shape(self):
        """Damping should peak at the interface (alpha ~ 0.5)."""
        model = ExponentialBlendedDamping(damping_coeff=10.0, alpha_min=0.01, alpha_max=0.99, width=0.2)
        alpha = torch.linspace(0.0, 1.0, 101, dtype=torch.float64)
        f = model.compute_damping_factor(alpha)
        # Maximum should be near alpha=0.5
        max_idx = f.argmax()
        alpha_peak = alpha[max_idx]
        assert abs(float(alpha_peak.item()) - 0.5) < 0.1

    def test_pure_phases_unaffected(self):
        """At alpha=0 and alpha=1, damping should be near zero."""
        model = ExponentialBlendedDamping(damping_coeff=10.0, width=0.05)
        alpha = torch.tensor([0.0, 1.0], dtype=torch.float64)
        k = torch.ones(2, dtype=torch.float64) * 10.0
        k_damped = model.damp_k(alpha, k)
        # Should be nearly unchanged at pure phases
        assert float(k_damped[0].item()) > 9.0
        assert float(k_damped[1].item()) > 9.0

    def test_damp_k(self):
        model = ExponentialBlendedDamping(damping_coeff=10.0, width=0.2)
        alpha = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
        k = torch.ones(3, dtype=torch.float64) * 100.0
        k_damped = model.damp_k(alpha, k)
        assert k_damped.shape == (3,)
        # Interface (alpha=0.5) should be damped most
        assert k_damped[1] < k_damped[0]
        assert k_damped[1] < k_damped[2]

    def test_damp_epsilon(self):
        model = ExponentialBlendedDamping(damping_coeff=10.0, width=0.2)
        alpha = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
        eps = torch.ones(3, dtype=torch.float64) * 100.0
        eps_damped = model.damp_epsilon(alpha, eps)
        assert eps_damped.shape == (3,)
        assert eps_damped[1] < eps_damped[0]

    def test_damp_omega(self):
        model = ExponentialBlendedDamping(damping_coeff=10.0, width=0.2)
        alpha = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
        omega = torch.ones(3, dtype=torch.float64) * 100.0
        omega_damped = model.damp_omega(alpha, omega)
        assert omega_damped.shape == (3,)
        assert omega_damped[1] < omega_damped[0]

    def test_damping_factor_range(self):
        """Damping factor should be in [0, damping_coeff]."""
        model = ExponentialBlendedDamping(damping_coeff=10.0)
        alpha = torch.rand(50, dtype=torch.float64)
        f = model.compute_damping_factor(alpha)
        assert (f >= 0).all()
        assert (f <= 10.0 + 1e-6).all()

    def test_wider_width_more_damping(self):
        """Wider width should apply damping over a broader range."""
        alpha = torch.tensor([0.15], dtype=torch.float64)
        model_narrow = ExponentialBlendedDamping(damping_coeff=10.0, width=0.05)
        model_wide = ExponentialBlendedDamping(damping_coeff=10.0, width=0.3)
        f_narrow = model_narrow.compute_damping_factor(alpha)
        f_wide = model_wide.compute_damping_factor(alpha)
        assert f_wide > f_narrow

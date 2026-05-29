"""Tests for enhanced turbulence damping v4 models.

Tests cover:
- TurbulenceDamping4EnhancedModel abstract base and registry
- ReynoldsAdaptiveDamping: Reynolds-adaptive damping
- TwoLayerDamping: two-layer damping model
- AlphaGradientLimiter: gradient-limited damping
"""

import pytest
import torch

from pyfoam.multiphase.turbulence_damping_enhanced_4 import (
    TurbulenceDamping4EnhancedModel,
    ReynoldsAdaptiveDamping,
    TwoLayerDamping,
    AlphaGradientLimiter,
)


class TestTurbulenceDamping4EnhancedModel:
    """Tests for abstract base."""

    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            TurbulenceDamping4EnhancedModel()

    def test_registry(self):
        types = TurbulenceDamping4EnhancedModel.available_types()
        assert "reynoldsAdaptive" in types
        assert "twoLayer" in types
        assert "alphaGradientLimiter" in types

    def test_factory_create(self):
        model = TurbulenceDamping4EnhancedModel.create("reynoldsAdaptive")
        assert isinstance(model, ReynoldsAdaptiveDamping)

    def test_factory_unknown(self):
        with pytest.raises(KeyError):
            TurbulenceDamping4EnhancedModel.create("nonexistent")


class TestReynoldsAdaptiveDamping:
    """Tests for ReynoldsAdaptiveDamping."""

    def test_default_params(self):
        model = ReynoldsAdaptiveDamping()
        assert model.damping_coeff == pytest.approx(10.0)
        assert model.Re_low == pytest.approx(100.0)
        assert model.Re_high == pytest.approx(10000.0)

    def test_damp_k_shape(self):
        model = ReynoldsAdaptiveDamping()
        alpha = torch.tensor([0.5, 0.8], dtype=torch.float64)
        k = torch.tensor([1.0, 1.0], dtype=torch.float64)
        k_damped = model.damp_k(alpha, k)
        assert k_damped.shape == (2,)
        assert (k_damped <= k).all()

    def test_damp_with_reynolds_number(self):
        model = ReynoldsAdaptiveDamping(Re_low=100, Re_high=10000)
        alpha = torch.tensor([0.5, 0.5], dtype=torch.float64)
        k = torch.tensor([1.0, 1.0], dtype=torch.float64)
        Re = torch.tensor([50.0, 50000.0], dtype=torch.float64)
        k_damped = model.damp_k(alpha, k, Re_loc=Re)
        assert k_damped.shape == (2,)
        assert (k_damped <= k).all()

    def test_no_damping_at_alpha_zero(self):
        model = ReynoldsAdaptiveDamping()
        alpha = torch.tensor([0.0], dtype=torch.float64)
        f = model.compute_damping_factor(alpha)
        assert float(f[0].item()) == pytest.approx(0.0)

    def test_damp_epsilon(self):
        model = ReynoldsAdaptiveDamping()
        alpha = torch.tensor([0.5], dtype=torch.float64)
        eps = torch.tensor([1.0], dtype=torch.float64)
        eps_damped = model.damp_epsilon(alpha, eps)
        assert float(eps_damped[0].item()) < 1.0

    def test_damp_omega(self):
        model = ReynoldsAdaptiveDamping()
        alpha = torch.tensor([0.5], dtype=torch.float64)
        omega = torch.tensor([1.0], dtype=torch.float64)
        omega_damped = model.damp_omega(alpha, omega)
        assert float(omega_damped[0].item()) < 1.0


class TestTwoLayerDamping:
    """Tests for TwoLayerDamping."""

    def test_default_params(self):
        model = TwoLayerDamping()
        assert model.y_plus_switch == pytest.approx(11.0)
        assert model.wall_damping_coeff == pytest.approx(8.0)

    def test_damp_k_shape(self):
        model = TwoLayerDamping()
        alpha = torch.tensor([0.5, 0.8], dtype=torch.float64)
        k = torch.tensor([1.0, 1.0], dtype=torch.float64)
        k_damped = model.damp_k(alpha, k)
        assert k_damped.shape == (2,)

    def test_near_wall_stronger_damping(self):
        """Near-wall cells (low y+) should have more wall damping than far cells."""
        model = TwoLayerDamping(wall_damping_coeff=20.0, damping_coeff=10.0)
        # Use alpha=0.2 to reduce interface (outer) damping contribution
        alpha = torch.tensor([0.2, 0.2], dtype=torch.float64)
        k = torch.tensor([1.0, 1.0], dtype=torch.float64)
        y_plus = torch.tensor([1.0, 100.0], dtype=torch.float64)
        k_damped = model.damp_k(alpha, k, y_plus=y_plus)
        # y+=1 (near wall): inner layer dominates -> more damping
        # y+=100 (far from wall): outer layer with low alpha -> less damping
        assert float(k_damped[0].item()) < float(k_damped[1].item())

    def test_no_y_plus_fallback(self):
        model = TwoLayerDamping()
        alpha = torch.tensor([0.5], dtype=torch.float64)
        k = torch.tensor([1.0], dtype=torch.float64)
        k_damped = model.damp_k(alpha, k)
        assert k_damped.shape == (1,)
        assert float(k_damped[0].item()) < 1.0


class TestAlphaGradientLimiter:
    """Tests for AlphaGradientLimiter."""

    def test_default_params(self):
        model = AlphaGradientLimiter()
        assert model.grad_max == pytest.approx(1.0)

    def test_damp_k_shape(self):
        model = AlphaGradientLimiter()
        alpha = torch.tensor([0.5, 0.8], dtype=torch.float64)
        k = torch.tensor([1.0, 1.0], dtype=torch.float64)
        k_damped = model.damp_k(alpha, k)
        assert k_damped.shape == (2,)

    def test_damp_with_grad_alpha(self):
        model = AlphaGradientLimiter()
        alpha = torch.tensor([0.5], dtype=torch.float64)
        k = torch.tensor([1.0], dtype=torch.float64)
        grad_alpha = torch.tensor([[0.3, 0.0, 0.0]], dtype=torch.float64)
        k_damped = model.damp_k(alpha, k, grad_alpha=grad_alpha)
        assert k_damped.shape == (1,)
        assert float(k_damped[0].item()) < 1.0

    def test_higher_gradient_more_damping(self):
        """Higher gradient should produce more damping."""
        model = AlphaGradientLimiter(grad_max=1.0)
        alpha = torch.tensor([0.5, 0.5], dtype=torch.float64)
        k = torch.tensor([1.0, 1.0], dtype=torch.float64)
        grad_low = torch.tensor([[0.1, 0.0, 0.0]], dtype=torch.float64)
        grad_high = torch.tensor([[0.8, 0.0, 0.0]], dtype=torch.float64)
        k_damped_low = model.damp_k(alpha[:1], k[:1], grad_alpha=grad_low)
        k_damped_high = model.damp_k(alpha[:1], k[:1], grad_alpha=grad_high)
        assert float(k_damped_low[0].item()) > float(k_damped_high[0].item())

    def test_no_gradient_fallback(self):
        model = AlphaGradientLimiter()
        alpha = torch.tensor([0.5], dtype=torch.float64)
        k = torch.tensor([1.0], dtype=torch.float64)
        k_damped = model.damp_k(alpha, k)
        assert k_damped.shape == (1,)

"""Tests for enhanced turbulence damping v5 models.

Tests cover:
- TurbulenceDamping5EnhancedModel abstract base and registry
- NearWallAnisotropicDamping: anisotropic near-wall damping
- BetaDampedModel: beta-distribution damping
- MultiScaleDamping: multi-scale damping
"""

import pytest
import torch

from pyfoam.multiphase.turbulence_damping_enhanced_5 import (
    TurbulenceDamping5EnhancedModel,
    NearWallAnisotropicDamping,
    BetaDampedModel,
    MultiScaleDamping,
)


class TestTurbulenceDamping5EnhancedModel:
    """Tests for abstract base."""

    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            TurbulenceDamping5EnhancedModel()

    def test_registry(self):
        types = TurbulenceDamping5EnhancedModel.available_types()
        assert "nearWallAnisotropic" in types
        assert "betaDamped" in types
        assert "multiScale" in types

    def test_factory_create(self):
        model = TurbulenceDamping5EnhancedModel.create("nearWallAnisotropic")
        assert isinstance(model, NearWallAnisotropicDamping)

    def test_factory_unknown(self):
        with pytest.raises(KeyError):
            TurbulenceDamping5EnhancedModel.create("nonexistent")


class TestNearWallAnisotropicDamping:
    """Tests for NearWallAnisotropicDamping."""

    def test_default_params(self):
        model = NearWallAnisotropicDamping()
        assert model.damping_coeff == pytest.approx(10.0)
        assert model.aniso_factor == pytest.approx(0.3)
        assert model.y_plus_visc == pytest.approx(5.0)

    def test_damp_k_shape(self):
        model = NearWallAnisotropicDamping()
        alpha = torch.tensor([0.5, 0.8], dtype=torch.float64)
        k = torch.tensor([1.0, 1.0], dtype=torch.float64)
        k_damped = model.damp_k(alpha, k)
        assert k_damped.shape == (2,)
        assert (k_damped <= k).all()

    def test_damp_with_aniso_ratio(self):
        model = NearWallAnisotropicDamping(aniso_factor=0.5)
        alpha = torch.tensor([0.5], dtype=torch.float64)
        k = torch.tensor([1.0], dtype=torch.float64)
        # High aniso ratio (more isotropic) -> less damping modification
        k_iso = model.damp_k(alpha, k, aniso_ratio=torch.tensor([0.9], dtype=torch.float64))
        # Low aniso ratio (more anisotropic) -> more damping modification
        k_aniso = model.damp_k(alpha, k, aniso_ratio=torch.tensor([0.1], dtype=torch.float64))
        # Anisotropic should have different damping than isotropic
        assert float(k_iso[0].item()) != float(k_aniso[0].item())

    def test_no_damping_at_alpha_zero(self):
        model = NearWallAnisotropicDamping()
        alpha = torch.tensor([0.0], dtype=torch.float64)
        f = model.compute_damping_factor(alpha)
        assert float(f[0].item()) == pytest.approx(0.0)


class TestBetaDampedModel:
    """Tests for BetaDampedModel."""

    def test_default_params(self):
        model = BetaDampedModel()
        assert model.beta_a == pytest.approx(2.0)
        assert model.beta_b == pytest.approx(2.0)

    def test_damp_k_shape(self):
        model = BetaDampedModel()
        alpha = torch.tensor([0.5, 0.8], dtype=torch.float64)
        k = torch.tensor([1.0, 1.0], dtype=torch.float64)
        k_damped = model.damp_k(alpha, k)
        assert k_damped.shape == (2,)

    def test_peak_damping_at_alpha_half(self):
        """Beta(2,2) peaks at alpha=0.5, so damping should be strongest there."""
        model = BetaDampedModel(beta_a=2.0, beta_b=2.0)
        alpha_half = torch.tensor([0.5], dtype=torch.float64)
        alpha_edge = torch.tensor([0.1], dtype=torch.float64)
        f_half = model.compute_damping_factor(alpha_half)
        f_edge = model.compute_damping_factor(alpha_edge)
        assert float(f_half[0].item()) > float(f_edge[0].item())

    def test_no_damping_at_extremes(self):
        """Damping should vanish at alpha=0 or alpha=1."""
        model = BetaDampedModel()
        alpha_zero = torch.tensor([0.0], dtype=torch.float64)
        alpha_one = torch.tensor([1.0], dtype=torch.float64)
        f_zero = model.compute_damping_factor(alpha_zero)
        f_one = model.compute_damping_factor(alpha_one)
        assert float(f_zero[0].item()) == pytest.approx(0.0)
        assert float(f_one[0].item()) == pytest.approx(0.0, abs=1e-6)

    def test_custom_beta_params(self):
        model = BetaDampedModel(beta_a=3.0, beta_b=1.5)
        alpha = torch.tensor([0.5], dtype=torch.float64)
        k = torch.tensor([1.0], dtype=torch.float64)
        k_damped = model.damp_k(alpha, k)
        assert float(k_damped[0].item()) < 1.0


class TestMultiScaleDamping:
    """Tests for MultiScaleDamping."""

    def test_default_weights(self):
        model = MultiScaleDamping()
        w_large, w_medium, w_small = model.weights
        total = w_large + w_medium + w_small
        assert total == pytest.approx(1.0, abs=0.01)

    def test_damp_k_shape(self):
        model = MultiScaleDamping()
        alpha = torch.tensor([0.5, 0.8], dtype=torch.float64)
        k = torch.tensor([1.0, 1.0], dtype=torch.float64)
        k_damped = model.damp_k(alpha, k)
        assert k_damped.shape == (2,)

    def test_damp_with_epsilon_and_gradient(self):
        model = MultiScaleDamping()
        alpha = torch.tensor([0.5, 0.5], dtype=torch.float64)
        k = torch.tensor([1.0, 1.0], dtype=torch.float64)
        epsilon = torch.tensor([0.01, 0.1], dtype=torch.float64)
        grad_alpha = torch.tensor([0.1, 0.5], dtype=torch.float64)
        k_damped = model.damp_k(
            alpha, k,
            epsilon=epsilon,
            grad_alpha_mag=grad_alpha,
        )
        assert k_damped.shape == (2,)
        assert (k_damped <= k).all()

    def test_custom_weights(self):
        model = MultiScaleDamping(
            large_scale_weight=0.7,
            medium_scale_weight=0.2,
            small_scale_weight=0.1,
        )
        w_large, w_medium, w_small = model.weights
        assert w_large == pytest.approx(0.7, abs=0.01)

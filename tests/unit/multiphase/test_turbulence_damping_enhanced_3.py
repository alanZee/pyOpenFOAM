"""Tests for enhanced turbulence damping v3 (near-wall models).

Tests cover:
- TurbulenceDamping3EnhancedModel abstract base and registry
- WallDampedDamping: wall-aware damping
- SpaldingDamping: Spalding wall law damping
- BlendedWallInterfaceDamping: wall-interface blending
"""

import pytest
import torch

from pyfoam.multiphase.turbulence_damping_enhanced_3 import (
    TurbulenceDamping3EnhancedModel,
    WallDampedDamping,
    SpaldingDamping,
    BlendedWallInterfaceDamping,
)


class TestTurbulenceDamping3EnhancedModel:
    """Tests for abstract base."""

    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            TurbulenceDamping3EnhancedModel()

    def test_registry(self):
        types = TurbulenceDamping3EnhancedModel.available_types()
        assert "wallDampedDamping" in types
        assert "spaldingDamping" in types
        assert "blendedWallInterface" in types

    def test_factory_create(self):
        model = TurbulenceDamping3EnhancedModel.create("wallDampedDamping")
        assert isinstance(model, WallDampedDamping)

    def test_factory_unknown(self):
        with pytest.raises(KeyError):
            TurbulenceDamping3EnhancedModel.create("nonexistent")


class TestWallDampedDamping:
    """Tests for WallDampedDamping."""

    def test_default_params(self):
        model = WallDampedDamping()
        assert model.damping_coeff == pytest.approx(10.0)
        assert model.y_plus_switch == pytest.approx(30.0)

    def test_damp_k_shape(self):
        model = WallDampedDamping()
        alpha = torch.tensor([0.5, 0.8], dtype=torch.float64)
        k = torch.tensor([1.0, 1.0], dtype=torch.float64)
        k_damped = model.damp_k(alpha, k)
        assert k_damped.shape == (2,)
        assert (k_damped <= k).all()
        assert (k_damped > 0).all()

    def test_damp_with_y_plus(self):
        model = WallDampedDamping()
        alpha = torch.tensor([0.5], dtype=torch.float64)
        k = torch.tensor([1.0], dtype=torch.float64)
        y_plus = torch.tensor([5.0], dtype=torch.float64)
        k_damped = model.damp_k(alpha, k, y_plus=y_plus)
        # Should be damped more with y+ = 5 (near wall) than without
        k_damped_no_yp = model.damp_k(alpha, k)
        assert float(k_damped[0].item()) < float(k_damped_no_yp[0].item())

    def test_no_damping_at_alpha_zero(self):
        model = WallDampedDamping()
        alpha = torch.tensor([0.0], dtype=torch.float64)
        k = torch.tensor([1.0], dtype=torch.float64)
        f = model.compute_damping_factor(alpha)
        # Interface factor should be 0 at alpha=0 (no interface damping)
        assert float(f[0].item()) == pytest.approx(0.0)

    def test_damp_epsilon(self):
        model = WallDampedDamping()
        alpha = torch.tensor([0.5], dtype=torch.float64)
        eps = torch.tensor([1.0], dtype=torch.float64)
        eps_damped = model.damp_epsilon(alpha, eps)
        assert eps_damped.shape == (1,)
        assert float(eps_damped[0].item()) < 1.0

    def test_damp_omega(self):
        model = WallDampedDamping()
        alpha = torch.tensor([0.5], dtype=torch.float64)
        omega = torch.tensor([1.0], dtype=torch.float64)
        omega_damped = model.damp_omega(alpha, omega)
        assert omega_damped.shape == (1,)
        assert float(omega_damped[0].item()) < 1.0


class TestSpaldingDamping:
    """Tests for SpaldingDamping."""

    def test_default_params(self):
        model = SpaldingDamping()
        assert model.kappa == pytest.approx(0.41)
        assert model.B == pytest.approx(5.5)

    def test_damp_k_shape(self):
        model = SpaldingDamping()
        alpha = torch.tensor([0.5, 0.8], dtype=torch.float64)
        k = torch.tensor([1.0, 1.0], dtype=torch.float64)
        k_damped = model.damp_k(alpha, k)
        assert k_damped.shape == (2,)

    def test_stronger_damping_at_low_y_plus(self):
        model = SpaldingDamping()
        alpha = torch.tensor([0.5, 0.5], dtype=torch.float64)
        k = torch.tensor([1.0, 1.0], dtype=torch.float64)
        y_plus = torch.tensor([1.0, 100.0], dtype=torch.float64)
        k_damped = model.damp_k(alpha, k, y_plus=y_plus)
        # Low y+ should have more damping
        assert float(k_damped[0].item()) < float(k_damped[1].item())

    def test_no_y_plus_fallback(self):
        model = SpaldingDamping()
        alpha = torch.tensor([0.5], dtype=torch.float64)
        k = torch.tensor([1.0], dtype=torch.float64)
        k_damped = model.damp_k(alpha, k)
        assert k_damped.shape == (1,)
        assert float(k_damped[0].item()) < 1.0


class TestBlendedWallInterfaceDamping:
    """Tests for BlendedWallInterfaceDamping."""

    def test_default_params(self):
        model = BlendedWallInterfaceDamping()
        assert model.d_ref == pytest.approx(1e-3)
        assert model.wall_damping_coeff == pytest.approx(8.0)

    def test_damp_k_with_wall_distance(self):
        model = BlendedWallInterfaceDamping()
        alpha = torch.tensor([0.5], dtype=torch.float64)
        k = torch.tensor([1.0], dtype=torch.float64)
        d_wall = torch.tensor([1e-4], dtype=torch.float64)
        k_damped = model.damp_k(alpha, k, wall_distance=d_wall)
        assert k_damped.shape == (1,)
        assert float(k_damped[0].item()) < 1.0

    def test_near_wall_stronger_damping(self):
        model = BlendedWallInterfaceDamping(d_ref=1e-3)
        alpha = torch.tensor([0.5, 0.5], dtype=torch.float64)
        k = torch.tensor([1.0, 1.0], dtype=torch.float64)
        d_wall = torch.tensor([1e-5, 1.0], dtype=torch.float64)
        k_damped = model.damp_k(alpha, k, wall_distance=d_wall)
        # Closer to wall -> more damping
        assert float(k_damped[0].item()) < float(k_damped[1].item())

    def test_with_grad_alpha(self):
        model = BlendedWallInterfaceDamping()
        alpha = torch.tensor([0.5], dtype=torch.float64)
        k = torch.tensor([1.0], dtype=torch.float64)
        grad_alpha = torch.tensor([[0.5, 0.0, 0.0]], dtype=torch.float64)
        k_damped = model.damp_k(alpha, k, grad_alpha=grad_alpha)
        assert k_damped.shape == (1,)
        assert float(k_damped[0].item()) < 1.0

    def test_no_wall_info_fallback(self):
        model = BlendedWallInterfaceDamping()
        alpha = torch.tensor([0.5], dtype=torch.float64)
        k = torch.tensor([1.0], dtype=torch.float64)
        k_damped = model.damp_k(alpha, k)
        assert k_damped.shape == (1,)

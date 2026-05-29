"""
Unit tests for enhanced turbulence damping models — Phase 2.
"""

from __future__ import annotations

import pytest
import torch

from pyfoam.multiphase.turbulence_damping_enhanced_2 import (
    TurbulenceDamping2EnhancedModel,
    LopezDeBertodanoDamping,
    KataokaDamping,
)


# ======================================================================
# RTS Registry
# ======================================================================

class TestTurbulenceDamping2Registry:
    """RTS registration tests."""

    def test_lopez_registered(self):
        assert "lopezDeBertodano" in TurbulenceDamping2EnhancedModel.available_types()

    def test_kataoka_registered(self):
        assert "kataoka" in TurbulenceDamping2EnhancedModel.available_types()

    def test_factory_create_lopez(self):
        model = TurbulenceDamping2EnhancedModel.create(
            "lopezDeBertodano", damping_coeff=2.0,
        )
        assert isinstance(model, LopezDeBertodanoDamping)
        assert model.damping_coeff == pytest.approx(2.0)

    def test_factory_create_kataoka(self):
        model = TurbulenceDamping2EnhancedModel.create(
            "kataoka", damping_coeff=3.0,
        )
        assert isinstance(model, KataokaDamping)
        assert model.damping_coeff == pytest.approx(3.0)

    def test_factory_unknown_raises(self):
        with pytest.raises(KeyError, match="Unknown damping"):
            TurbulenceDamping2EnhancedModel.create("nonexistent")

    def test_duplicate_registration_raises(self):
        with pytest.raises(ValueError, match="already registered"):
            @TurbulenceDamping2EnhancedModel.register("lopezDeBertodano")
            class _Dup:
                pass

    def test_available_types_sorted(self):
        types = TurbulenceDamping2EnhancedModel.available_types()
        assert types == sorted(types)


# ======================================================================
# LopezDeBertodanoDamping
# ======================================================================

class TestLopezDeBertodanoDamping:
    """Tests for Lopez de Bertodano damping model."""

    def test_default_params(self):
        model = LopezDeBertodanoDamping()
        assert model.damping_coeff == pytest.approx(1.0)
        assert model.d_bubble == pytest.approx(1e-3)
        assert model.rho_c == pytest.approx(1000.0)
        assert model.mu_c == pytest.approx(1e-3)

    def test_custom_params(self):
        model = LopezDeBertodanoDamping(
            damping_coeff=2.0, d_bubble=5e-3, rho_c=1.225,
        )
        assert model.damping_coeff == pytest.approx(2.0)
        assert model.d_bubble == pytest.approx(5e-3)

    def test_d_bubble_must_be_positive(self):
        with pytest.raises(ValueError, match="d_bubble"):
            LopezDeBertodanoDamping(d_bubble=0.0)

    def test_rho_c_must_be_positive(self):
        with pytest.raises(ValueError, match="rho_c"):
            LopezDeBertodanoDamping(rho_c=0.0)

    def test_mu_c_must_be_positive(self):
        with pytest.raises(ValueError, match="mu_c"):
            LopezDeBertodanoDamping(mu_c=0.0)

    def test_pure_phases_no_damping(self):
        """Pure phases (alpha=0, alpha=1) should not be damped."""
        model = LopezDeBertodanoDamping()
        alpha = torch.tensor([0.0, 1.0], dtype=torch.float64)
        k = torch.ones(2, dtype=torch.float64) * 10.0
        k_damped = model.damp_k(alpha, k)
        assert float(k_damped[0].item()) == pytest.approx(10.0, abs=1e-6)
        assert float(k_damped[1].item()) == pytest.approx(10.0, abs=1e-6)

    def test_interface_damped(self):
        """Interface should be damped."""
        model = LopezDeBertodanoDamping(damping_coeff=5.0)
        alpha = torch.tensor([0.0, 0.3, 0.5, 0.7, 1.0], dtype=torch.float64)
        k = torch.ones(5, dtype=torch.float64) * 10.0
        k_damped = model.damp_k(alpha, k)
        # Middle values should be damped
        assert k_damped[1] < 10.0
        assert k_damped[2] < 10.0
        assert k_damped[3] < 10.0

    def test_damp_k_with_k_epsilon(self):
        """Providing tke and epsilon fields should use physics-based damping."""
        model = LopezDeBertodanoDamping(damping_coeff=2.0)
        alpha = torch.tensor([0.1, 0.3, 0.5], dtype=torch.float64)
        k = torch.ones(3, dtype=torch.float64) * 5.0
        eps = torch.ones(3, dtype=torch.float64) * 100.0
        k_damped = model.damp_k(alpha, k, tke=k, epsilon=eps)
        assert k_damped.shape == (3,)
        assert (k_damped <= 5.0 + 1e-6).all()

    def test_damp_epsilon(self):
        model = LopezDeBertodanoDamping()
        alpha = torch.tensor([0.0, 0.3, 1.0], dtype=torch.float64)
        eps = torch.ones(3, dtype=torch.float64) * 50.0
        eps_damped = model.damp_epsilon(alpha, eps)
        assert eps_damped.shape == (3,)
        assert float(eps_damped[0].item()) == pytest.approx(50.0, abs=1e-3)

    def test_damp_omega(self):
        model = LopezDeBertodanoDamping()
        alpha = torch.tensor([0.0, 0.3, 1.0], dtype=torch.float64)
        omega = torch.ones(3, dtype=torch.float64) * 20.0
        omega_damped = model.damp_omega(alpha, omega)
        assert omega_damped.shape == (3,)

    def test_damping_factor_range(self):
        model = LopezDeBertodanoDamping(damping_coeff=5.0)
        alpha = torch.rand(50, dtype=torch.float64).clamp(0.01, 0.99)
        f = model.compute_damping_factor(alpha)
        assert (f >= 0).all()
        assert (f <= 5.0 + 1e-6).all()

    def test_repr(self):
        model = LopezDeBertodanoDamping(d_bubble=5e-3)
        r = repr(model)
        assert "LopezDeBertodano" in r
        assert "5e-3" in r or "0.005" in r


# ======================================================================
# KataokaDamping
# ======================================================================

class TestKataokaDamping:
    """Tests for Kataoka damping model."""

    def test_default_params(self):
        model = KataokaDamping()
        assert model.damping_coeff == pytest.approx(1.0)
        assert model.d_bubble == pytest.approx(1e-3)

    def test_custom_params(self):
        model = KataokaDamping(damping_coeff=3.0, d_bubble=2e-3)
        assert model.damping_coeff == pytest.approx(3.0)
        assert model.d_bubble == pytest.approx(2e-3)

    def test_d_bubble_must_be_positive(self):
        with pytest.raises(ValueError, match="d_bubble"):
            KataokaDamping(d_bubble=0.0)

    def test_pure_phases_no_damping(self):
        """Pure phases should not be damped."""
        model = KataokaDamping()
        alpha = torch.tensor([0.0, 1.0], dtype=torch.float64)
        k = torch.ones(2, dtype=torch.float64) * 10.0
        k_damped = model.damp_k(alpha, k)
        assert float(k_damped[0].item()) == pytest.approx(10.0, abs=1e-6)
        assert float(k_damped[1].item()) == pytest.approx(10.0, abs=1e-6)

    def test_interface_damped(self):
        """Interface should be damped."""
        model = KataokaDamping(damping_coeff=5.0)
        alpha = torch.tensor([0.0, 0.3, 0.5, 0.7, 1.0], dtype=torch.float64)
        k = torch.ones(5, dtype=torch.float64) * 10.0
        k_damped = model.damp_k(alpha, k)
        assert k_damped[1] < 10.0
        assert k_damped[2] < 10.0
        assert k_damped[3] < 10.0

    def test_damp_k_with_k_epsilon(self):
        """Providing tke and epsilon should use physics-based damping."""
        model = KataokaDamping(damping_coeff=2.0, d_bubble=1e-3)
        alpha = torch.tensor([0.1, 0.3, 0.5], dtype=torch.float64)
        k = torch.ones(3, dtype=torch.float64) * 5.0
        eps = torch.ones(3, dtype=torch.float64) * 100.0
        k_damped = model.damp_k(alpha, k, tke=k, epsilon=eps)
        assert k_damped.shape == (3,)

    def test_damp_epsilon(self):
        model = KataokaDamping()
        alpha = torch.tensor([0.0, 0.3, 1.0], dtype=torch.float64)
        eps = torch.ones(3, dtype=torch.float64) * 50.0
        eps_damped = model.damp_epsilon(alpha, eps)
        assert eps_damped.shape == (3,)

    def test_damp_omega(self):
        model = KataokaDamping()
        alpha = torch.tensor([0.0, 0.3, 1.0], dtype=torch.float64)
        omega = torch.ones(3, dtype=torch.float64) * 20.0
        omega_damped = model.damp_omega(alpha, omega)
        assert omega_damped.shape == (3,)

    def test_damping_factor_range(self):
        model = KataokaDamping(damping_coeff=5.0)
        alpha = torch.rand(50, dtype=torch.float64).clamp(0.01, 0.99)
        f = model.compute_damping_factor(alpha)
        assert (f >= 0).all()
        assert (f <= 5.0 + 1e-6).all()

    def test_bell_shape_profile(self):
        """Kataoka damping should peak at intermediate alpha (bell shape).

        alpha*(1-alpha) peaks at alpha=0.5
        """
        model = KataokaDamping(damping_coeff=10.0, d_bubble=1e-3)
        alpha = torch.linspace(0.01, 0.99, 99, dtype=torch.float64)
        f = model.compute_damping_factor(alpha)
        max_idx = f.argmax()
        alpha_peak = alpha[max_idx]
        # Peak should be near alpha=0.5 (within 0.1)
        assert abs(float(alpha_peak.item()) - 0.5) < 0.1

    def test_repr(self):
        model = KataokaDamping(d_bubble=2e-3)
        r = repr(model)
        assert "Kataoka" in r
        assert "2e-3" in r or "0.002" in r

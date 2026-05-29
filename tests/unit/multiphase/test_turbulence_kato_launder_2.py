"""Tests for enhanced Kato-Launder damping model (version 2).

Tests cover KatoLaunderDamping2:
- Init with default and custom parameters
- Interface damping factor (parabolic)
- Phase-weighted viscosity
- Gradient-based damping
- damp_production with Kato-Launder rotation
- damp_k_source
"""

import pytest
import torch

from pyfoam.multiphase.turbulence_kato_launder_2 import KatoLaunderDamping2


class TestKatoLaunderDamping2Init:
    """Initialisation tests."""

    def test_default_params(self):
        model = KatoLaunderDamping2()
        assert model.damping_strength == pytest.approx(0.9)
        assert model.alpha_min == pytest.approx(0.01)
        assert model.alpha_max == pytest.approx(0.99)
        assert model.alpha_cutoff == pytest.approx(0.01)
        assert model.beta == pytest.approx(0.0)
        assert model.use_rotation is True

    def test_custom_params(self):
        model = KatoLaunderDamping2(
            damping_strength=0.5,
            alpha_min=0.05,
            alpha_max=0.95,
            alpha_cutoff=0.001,
            beta=10.0,
            use_rotation=False,
        )
        assert model.damping_strength == pytest.approx(0.5)
        assert model.alpha_min == pytest.approx(0.05)
        assert model.alpha_max == pytest.approx(0.95)
        assert model.alpha_cutoff == pytest.approx(0.001)
        assert model.beta == pytest.approx(10.0)
        assert model.use_rotation is False


class TestKatoLaunderDamping2Interface:
    """Interface damping factor tests."""

    def test_pure_phases_no_damping(self):
        model = KatoLaunderDamping2(damping_strength=0.9)
        alpha = torch.tensor([0.0, 1.0, 0.0, 1.0], dtype=torch.float64)
        f = model.compute_interface_factor(alpha)
        assert torch.allclose(f, torch.ones(4, dtype=torch.float64), atol=1e-10)

    def test_interface_maximum_damping(self):
        """alpha=0.5 → maximum damping."""
        model = KatoLaunderDamping2(damping_strength=0.9)
        alpha = torch.tensor([0.5], dtype=torch.float64)
        f = model.compute_interface_factor(alpha)
        # f = 1 - 0.9 * 4 * 0.5 * 0.5 = 1 - 0.9 = 0.1
        assert f[0].item() == pytest.approx(0.1)

    def test_zero_damping_strength(self):
        """damping_strength=0 → no damping regardless of alpha."""
        model = KatoLaunderDamping2(damping_strength=0.0)
        alpha = torch.tensor([0.3, 0.5, 0.7], dtype=torch.float64)
        f = model.compute_interface_factor(alpha)
        assert torch.allclose(f, torch.ones(3, dtype=torch.float64))

    def test_higher_damping_more_reduction(self):
        alpha = torch.tensor([0.5], dtype=torch.float64)
        model_low = KatoLaunderDamping2(damping_strength=0.1)
        model_high = KatoLaunderDamping2(damping_strength=0.9)
        f_low = model_low.compute_interface_factor(alpha)
        f_high = model_high.compute_interface_factor(alpha)
        assert f_low > f_high

    def test_outside_alpha_threshold(self):
        """Outside [alpha_min, alpha_max]: no damping."""
        model = KatoLaunderDamping2(damping_strength=0.9, alpha_min=0.1, alpha_max=0.9)
        alpha = torch.tensor([0.001, 0.999], dtype=torch.float64)
        f = model.compute_interface_factor(alpha)
        assert torch.allclose(f, torch.ones(2, dtype=torch.float64), atol=1e-10)


class TestKatoLaunderDamping2PhaseWeight:
    """Phase weight (viscosity scaling) tests."""

    def test_pure_continuous_phase(self):
        """alpha=0 (all continuous) → weight=1."""
        model = KatoLaunderDamping2()
        alpha = torch.tensor([0.0], dtype=torch.float64)
        w = model.compute_phase_weight(alpha)
        assert w[0].item() == pytest.approx(1.0)

    def test_pure_dispersed_phase(self):
        """alpha=1 (all dispersed) → weight=alpha_cutoff."""
        model = KatoLaunderDamping2(alpha_cutoff=0.01)
        alpha = torch.tensor([1.0], dtype=torch.float64)
        w = model.compute_phase_weight(alpha)
        assert w[0].item() == pytest.approx(0.01)

    def test_mixed_phase(self):
        """alpha=0.3 → weight=0.7."""
        model = KatoLaunderDamping2()
        alpha = torch.tensor([0.3], dtype=torch.float64)
        w = model.compute_phase_weight(alpha)
        assert w[0].item() == pytest.approx(0.7)

    def test_monotone_decreasing(self):
        """Higher alpha → lower weight."""
        model = KatoLaunderDamping2()
        alpha = torch.tensor([0.0, 0.2, 0.5, 0.8, 1.0], dtype=torch.float64)
        w = model.compute_phase_weight(alpha)
        for i in range(len(w) - 1):
            assert w[i] >= w[i + 1]


class TestKatoLaunderDamping2GradientDamping:
    """Gradient-based damping tests."""

    def test_zero_beta_no_damping(self):
        model = KatoLaunderDamping2(beta=0.0)
        grad = torch.tensor([0.0, 10.0, 100.0], dtype=torch.float64)
        f = model.compute_gradient_damping(grad)
        assert torch.allclose(f, torch.ones(3, dtype=torch.float64))

    def test_zero_gradient_no_damping(self):
        model = KatoLaunderDamping2(beta=5.0)
        grad = torch.tensor([0.0], dtype=torch.float64)
        f = model.compute_gradient_damping(grad)
        assert f[0].item() == pytest.approx(1.0)

    def test_larger_gradient_more_damping(self):
        model = KatoLaunderDamping2(beta=1.0)
        grad = torch.tensor([0.1, 1.0, 10.0], dtype=torch.float64)
        f = model.compute_gradient_damping(grad)
        for i in range(len(f) - 1):
            assert f[i] > f[i + 1]

    def test_output_range(self):
        model = KatoLaunderDamping2(beta=5.0)
        grad = torch.rand(50, dtype=torch.float64) * 100
        f = model.compute_gradient_damping(grad)
        assert (f > 0).all()
        assert (f <= 1.0 + 1e-10).all()


class TestKatoLaunderDamping2EffectiveViscosity:
    """Effective viscosity tests."""

    def test_pure_continuous(self):
        """alpha=0 → nu_t_eff = nu_t."""
        model = KatoLaunderDamping2()
        alpha = torch.tensor([0.0, 0.0], dtype=torch.float64)
        nu_t = torch.tensor([0.1, 0.2], dtype=torch.float64)
        nu_t_eff = model.effective_viscosity(alpha, nu_t)
        assert torch.allclose(nu_t_eff, nu_t)

    def test_pure_dispersed(self):
        """alpha=1 → nu_t_eff = nu_t * alpha_cutoff."""
        model = KatoLaunderDamping2(alpha_cutoff=0.01)
        alpha = torch.tensor([1.0], dtype=torch.float64)
        nu_t = torch.tensor([0.1], dtype=torch.float64)
        nu_t_eff = model.effective_viscosity(alpha, nu_t)
        assert nu_t_eff[0].item() == pytest.approx(0.001)


class TestKatoLaunderDamping2Production:
    """damp_production tests."""

    def test_rotation_based_formula(self):
        """Standard Kato-Launder: P = 2 * nu_t_eff * S * Omega * f."""
        model = KatoLaunderDamping2(damping_strength=0.0, alpha_cutoff=0.0)
        alpha = torch.tensor([0.0], dtype=torch.float64)
        S_mag = torch.tensor([100.0], dtype=torch.float64)
        Omega_mag = torch.tensor([50.0], dtype=torch.float64)
        nu_t = torch.tensor([0.01], dtype=torch.float64)
        P = model.damp_production(alpha, S_mag, Omega_mag, nu_t)
        # P = 2 * 0.01 * 1.0 * 100 * 50 * 1.0 = 100
        assert P[0].item() == pytest.approx(100.0)

    def test_stagnation_reduced(self):
        """At stagnation (Omega ~ 0): production is near zero."""
        model = KatoLaunderDamping2(damping_strength=0.0, alpha_cutoff=0.0)
        alpha = torch.tensor([0.0], dtype=torch.float64)
        S_mag = torch.tensor([100.0], dtype=torch.float64)
        Omega_mag = torch.tensor([0.001], dtype=torch.float64)
        nu_t = torch.tensor([0.01], dtype=torch.float64)
        P = model.damp_production(alpha, S_mag, Omega_mag, nu_t)
        assert P[0].item() < 1.0

    def test_interface_damping_reduces_production(self):
        """Interface damping reduces production near alpha=0.5."""
        model_no_damp = KatoLaunderDamping2(damping_strength=0.0, alpha_cutoff=0.0)
        model_damp = KatoLaunderDamping2(damping_strength=0.9, alpha_cutoff=0.0)
        alpha = torch.tensor([0.5], dtype=torch.float64)
        S_mag = torch.tensor([100.0], dtype=torch.float64)
        Omega_mag = torch.tensor([50.0], dtype=torch.float64)
        nu_t = torch.tensor([0.01], dtype=torch.float64)
        P_no = model_no_damp.damp_production(alpha, S_mag, Omega_mag, nu_t)
        P_damp = model_damp.damp_production(alpha, S_mag, Omega_mag, nu_t)
        assert P_damp[0].item() < P_no[0].item()

    def test_phase_weight_reduces_in_dispersed(self):
        """Phase weighting reduces production in dispersed phase."""
        model = KatoLaunderDamping2(damping_strength=0.0, alpha_cutoff=0.01)
        alpha_continuous = torch.tensor([0.0], dtype=torch.float64)
        alpha_dispersed = torch.tensor([0.99], dtype=torch.float64)
        S_mag = torch.tensor([100.0], dtype=torch.float64)
        Omega_mag = torch.tensor([50.0], dtype=torch.float64)
        nu_t = torch.tensor([0.01], dtype=torch.float64)
        P_c = model.damp_production(alpha_continuous, S_mag, Omega_mag, nu_t)
        P_d = model.damp_production(alpha_dispersed, S_mag, Omega_mag, nu_t)
        assert P_d[0].item() < P_c[0].item()

    def test_gradient_damping(self):
        """Alpha gradient damping further reduces production."""
        model = KatoLaunderDamping2(damping_strength=0.0, alpha_cutoff=0.0, beta=10.0)
        alpha = torch.tensor([0.0], dtype=torch.float64)
        S_mag = torch.tensor([100.0], dtype=torch.float64)
        Omega_mag = torch.tensor([50.0], dtype=torch.float64)
        nu_t = torch.tensor([0.01], dtype=torch.float64)
        grad_zero = torch.tensor([0.0], dtype=torch.float64)
        grad_large = torch.tensor([100.0], dtype=torch.float64)
        P_no_grad = model.damp_production(alpha, S_mag, Omega_mag, nu_t, grad_zero)
        P_grad = model.damp_production(alpha, S_mag, Omega_mag, nu_t, grad_large)
        assert P_grad[0].item() < P_no_grad[0].item()

    def test_without_rotation(self):
        """use_rotation=False: standard |S|^2 with alpha damping."""
        model = KatoLaunderDamping2(
            damping_strength=0.0, alpha_cutoff=0.0, use_rotation=False,
        )
        alpha = torch.tensor([0.0], dtype=torch.float64)
        S_mag = torch.tensor([100.0], dtype=torch.float64)
        Omega_mag = torch.tensor([50.0], dtype=torch.float64)
        nu_t = torch.tensor([0.01], dtype=torch.float64)
        P = model.damp_production(alpha, S_mag, Omega_mag, nu_t)
        # P = 2 * 0.01 * 100^2 = 200
        assert P[0].item() == pytest.approx(200.0)

    def test_finite_output(self):
        """Output is always finite."""
        model = KatoLaunderDamping2(damping_strength=0.9, alpha_cutoff=0.01, beta=5.0)
        n = 100
        alpha = torch.rand(n, dtype=torch.float64)
        S_mag = torch.rand(n, dtype=torch.float64) * 100 + 1.0
        Omega_mag = torch.rand(n, dtype=torch.float64) * 100 + 0.01
        nu_t = torch.rand(n, dtype=torch.float64) * 0.1
        grad = torch.rand(n, dtype=torch.float64) * 50
        P = model.damp_production(alpha, S_mag, Omega_mag, nu_t, grad)
        assert torch.isfinite(P).all()


class TestKatoLaunderDamping2KSource:
    """damp_k_source tests."""

    def test_dissipation_subtracted(self):
        """Source = P_damped - epsilon."""
        model = KatoLaunderDamping2(damping_strength=0.0, alpha_cutoff=0.0)
        alpha = torch.tensor([0.0], dtype=torch.float64)
        k = torch.tensor([1.0], dtype=torch.float64)
        S_mag = torch.tensor([100.0], dtype=torch.float64)
        Omega_mag = torch.tensor([50.0], dtype=torch.float64)
        nu_t = torch.tensor([0.01], dtype=torch.float64)
        eps = torch.tensor([0.5], dtype=torch.float64)
        source = model.damp_k_source(alpha, k, S_mag, Omega_mag, nu_t, eps)
        # P = 100, source = 100 - 0.5 = 99.5
        assert source[0].item() == pytest.approx(99.5)

    def test_auto_dissipation(self):
        """Without explicit epsilon: uses C_mu * k^1.5."""
        model = KatoLaunderDamping2(damping_strength=0.0, alpha_cutoff=0.0)
        alpha = torch.tensor([0.0], dtype=torch.float64)
        k = torch.tensor([1.0], dtype=torch.float64)
        S_mag = torch.tensor([0.1], dtype=torch.float64)
        Omega_mag = torch.tensor([0.1], dtype=torch.float64)
        nu_t = torch.tensor([0.01], dtype=torch.float64)
        source = model.damp_k_source(alpha, k, S_mag, Omega_mag, nu_t)
        # P = 2 * 0.01 * 0.1 * 0.1 = 0.0002
        # eps = 0.09 * 1.0^1.5 = 0.09
        # source = 0.0002 - 0.09 = -0.0898
        assert source[0].item() < 0.0

    def test_repr(self):
        model = KatoLaunderDamping2(damping_strength=0.8, alpha_cutoff=0.001, beta=5.0)
        r = repr(model)
        assert "0.8" in r
        assert "0.001" in r
        assert "5.0" in r

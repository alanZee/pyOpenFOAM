"""Tests for generalised Newtonian viscosity models (RTS-registered).

Tests cover:
- RTS registration and factory creation
- CassonModel
- HerschelBulkleyModel
- BinghamModel
- QuemadaModel
- StrainRateFunctionModel
- mu_zero() and mu_inf() limits
- Bingham = HerschelBulkley with n=1 identity
"""

import pytest
import torch

from pyfoam.turbulence.generalized_newtonian import (
    GeneralizedNewtonianViscosity,
    CassonModel,
    HerschelBulkleyModel,
    BinghamModel,
    QuemadaModel,
    StrainRateFunctionModel,
)


# ---------------------------------------------------------------------------
# RTS tests
# ---------------------------------------------------------------------------


class TestGeneralizedNewtonianRTS:
    """Test the RTS registry for generalised Newtonian viscosity models."""

    def test_available_types(self):
        """All five models are registered."""
        types = GeneralizedNewtonianViscosity.available_types()
        assert "Casson" in types
        assert "HerschelBulkley" in types
        assert "Bingham" in types
        assert "Quemada" in types
        assert "strainRateFunction" in types

    def test_factory_casson(self):
        """Factory creates CassonModel."""
        model = GeneralizedNewtonianViscosity.create(
            "Casson", tau_y=0.1, mu_inf=0.001,
        )
        assert isinstance(model, CassonModel)

    def test_factory_herschel_bulkley(self):
        """Factory creates HerschelBulkleyModel."""
        model = GeneralizedNewtonianViscosity.create(
            "HerschelBulkley", tau_y=0.1, K=0.5, n=0.7,
        )
        assert isinstance(model, HerschelBulkleyModel)

    def test_factory_bingham(self):
        """Factory creates BinghamModel."""
        model = GeneralizedNewtonianViscosity.create(
            "Bingham", tau_y=0.1, mu_inf=0.001,
        )
        assert isinstance(model, BinghamModel)

    def test_factory_quemada(self):
        """Factory creates QuemadaModel."""
        model = GeneralizedNewtonianViscosity.create(
            "Quemada", phi=0.3, k0=2.5, k_inf=0.5,
        )
        assert isinstance(model, QuemadaModel)

    def test_factory_strain_rate_function(self):
        """Factory creates StrainRateFunctionModel."""
        func = lambda gd: gd * 0.01  # noqa: E731
        model = GeneralizedNewtonianViscosity.create(
            "strainRateFunction", func=func,
        )
        assert isinstance(model, StrainRateFunctionModel)

    def test_factory_unknown_raises(self):
        """Unknown model name raises KeyError."""
        with pytest.raises(KeyError, match="Unknown generalised Newtonian"):
            GeneralizedNewtonianViscosity.create("nonexistent")

    def test_duplicate_registration_raises(self):
        """Duplicate registration raises ValueError."""
        with pytest.raises(ValueError, match="already registered"):
            @GeneralizedNewtonianViscosity.register("Casson")
            class _Duplicate:
                pass


# ---------------------------------------------------------------------------
# CassonModel
# ---------------------------------------------------------------------------


class TestCassonModel:
    """Tests for CassonModel."""

    def test_high_shear_limit(self):
        """At high gamma_dot, mu -> mu_inf (plastic viscosity)."""
        vm = CassonModel(tau_y=1.0, mu_inf=0.001)
        gd = torch.tensor([1e10], dtype=torch.float64)
        mu = vm.mu(gd)
        # Should approach mu_inf = 0.001
        assert torch.allclose(
            mu, torch.tensor([0.001], dtype=torch.float64), atol=1e-4,
        )

    def test_formula(self):
        """Exact formula: mu = (sqrt(tau_y/gd) + sqrt(mu_inf))^2."""
        vm = CassonModel(tau_y=0.5, mu_inf=0.004)
        gd = torch.tensor([2.0], dtype=torch.float64)
        expected = ((0.5 / 2.0) ** 0.5 + 0.004 ** 0.5) ** 2
        assert torch.allclose(
            vm.mu(gd),
            torch.tensor([expected], dtype=torch.float64),
            atol=1e-10,
        )

    def test_shear_thinning(self):
        """Viscosity decreases with increasing shear rate."""
        vm = CassonModel(tau_y=1.0, mu_inf=0.001)
        gd_low = torch.tensor([0.1], dtype=torch.float64)
        gd_high = torch.tensor([100.0], dtype=torch.float64)
        assert vm.mu(gd_low) > vm.mu(gd_high)

    def test_yield_stress_effect(self):
        """Higher tau_y gives higher viscosity at same shear rate."""
        vm_high = CassonModel(tau_y=1.0, mu_inf=0.001)
        vm_low = CassonModel(tau_y=0.01, mu_inf=0.001)
        gd = torch.tensor([1.0], dtype=torch.float64)
        assert vm_high.mu(gd) > vm_low.mu(gd)

    def test_mu_inf(self):
        """mu_inf returns plastic viscosity."""
        vm = CassonModel(tau_y=0.5, mu_inf=0.003)
        assert vm.mu_inf() == 0.003

    def test_batch(self):
        """Works with batch tensors."""
        vm = CassonModel(tau_y=0.1, mu_inf=0.001)
        gd = torch.rand(100, dtype=torch.float64) * 10.0 + 0.01
        mu = vm.mu(gd)
        assert mu.shape == (100,)
        assert torch.isfinite(mu).all()
        assert (mu > 0).all()

    def test_repr(self):
        """repr includes class and parameters."""
        r = repr(CassonModel(tau_y=0.5, mu_inf=0.003))
        assert "CassonModel" in r
        assert "tau_y=0.5" in r
        assert "mu_inf=0.003" in r


# ---------------------------------------------------------------------------
# HerschelBulkleyModel
# ---------------------------------------------------------------------------


class TestHerschelBulkleyModel:
    """Tests for HerschelBulkleyModel."""

    def test_power_law_limit(self):
        """With tau_y=0, reduces to power-law."""
        vm = HerschelBulkleyModel(tau_y=0.0, K=1.0, n=0.5)
        gd = torch.tensor([4.0], dtype=torch.float64)
        expected = 1.0 * 4.0 ** (-0.5)
        assert torch.allclose(
            vm.mu(gd),
            torch.tensor([expected], dtype=torch.float64),
            atol=1e-10,
        )

    def test_yield_stress_increases_viscosity(self):
        """Nonzero tau_y increases viscosity at low strain rates."""
        vm_yield = HerschelBulkleyModel(tau_y=1.0, K=0.01, n=0.5)
        vm_no_yield = HerschelBulkleyModel(tau_y=0.0, K=0.01, n=0.5)
        gd = torch.tensor([0.1], dtype=torch.float64)
        assert vm_yield.mu(gd) > vm_no_yield.mu(gd)

    def test_formula(self):
        """Exact formula: mu = tau_y/gd + K * gd^(n-1)."""
        vm = HerschelBulkleyModel(tau_y=0.5, K=0.8, n=0.6)
        gd = torch.tensor([3.0], dtype=torch.float64)
        expected = 0.5 / 3.0 + 0.8 * 3.0 ** (0.6 - 1.0)
        assert torch.allclose(
            vm.mu(gd),
            torch.tensor([expected], dtype=torch.float64),
            atol=1e-10,
        )

    def test_shear_thinning(self):
        """For n < 1, viscosity decreases with shear rate (at high gd)."""
        vm = HerschelBulkleyModel(tau_y=0.0, K=1.0, n=0.5)
        gd_low = torch.tensor([1.0], dtype=torch.float64)
        gd_high = torch.tensor([100.0], dtype=torch.float64)
        assert vm.mu(gd_low) > vm.mu(gd_high)

    def test_mu_inf_n_less_1(self):
        """mu_inf for n < 1 returns small value."""
        vm = HerschelBulkleyModel(tau_y=0.1, K=0.5, n=0.5)
        assert vm.mu_inf() == pytest.approx(1e-6)

    def test_mu_inf_n_equals_1(self):
        """mu_inf for n=1 returns K (Bingham limit)."""
        vm = HerschelBulkleyModel(tau_y=0.1, K=0.5, n=1.0)
        assert vm.mu_inf() == 0.5

    def test_mu_zero(self):
        """mu_zero returns large value (yield stress dominated)."""
        vm = HerschelBulkleyModel(tau_y=0.1, K=0.5, n=0.5)
        assert vm.mu_zero() > 1e10

    def test_batch(self):
        """Works with batch tensors."""
        vm = HerschelBulkleyModel(tau_y=0.1, K=0.5, n=0.7)
        gd = torch.rand(100, dtype=torch.float64) * 10.0 + 0.01
        mu = vm.mu(gd)
        assert mu.shape == (100,)
        assert torch.isfinite(mu).all()
        assert (mu > 0).all()

    def test_repr(self):
        """repr includes class and parameters."""
        r = repr(HerschelBulkleyModel(tau_y=0.5, K=0.8, n=0.6))
        assert "HerschelBulkleyModel" in r
        assert "tau_y=0.5" in r
        assert "K=0.8" in r
        assert "n=0.6" in r


# ---------------------------------------------------------------------------
# BinghamModel
# ---------------------------------------------------------------------------


class TestBinghamModel:
    """Tests for BinghamModel."""

    def test_formula(self):
        """Exact formula: mu = tau_y/gd + mu_inf."""
        vm = BinghamModel(tau_y=0.5, mu_inf=0.003)
        gd = torch.tensor([2.0], dtype=torch.float64)
        expected = 0.5 / 2.0 + 0.003
        assert torch.allclose(
            vm.mu(gd),
            torch.tensor([expected], dtype=torch.float64),
            atol=1e-10,
        )

    def test_high_shear_limit(self):
        """At high gamma_dot, mu -> mu_inf."""
        vm = BinghamModel(tau_y=0.5, mu_inf=0.003)
        gd = torch.tensor([1e10], dtype=torch.float64)
        mu = vm.mu(gd)
        assert torch.allclose(
            mu, torch.tensor([0.003], dtype=torch.float64), atol=1e-6,
        )

    def test_yield_stress_effect(self):
        """Higher tau_y gives higher viscosity."""
        vm_high = BinghamModel(tau_y=1.0, mu_inf=0.003)
        vm_low = BinghamModel(tau_y=0.01, mu_inf=0.003)
        gd = torch.tensor([1.0], dtype=torch.float64)
        assert vm_high.mu(gd) > vm_low.mu(gd)

    def test_mu_inf(self):
        """mu_inf returns plastic viscosity."""
        vm = BinghamModel(tau_y=0.5, mu_inf=0.003)
        assert vm.mu_inf() == 0.003

    def test_mu_zero(self):
        """mu_zero returns large value (yield stress dominated)."""
        vm = BinghamModel(tau_y=0.1, mu_inf=0.003)
        assert vm.mu_zero() > 1e10

    def test_batch(self):
        """Works with batch tensors."""
        vm = BinghamModel(tau_y=0.1, mu_inf=0.003)
        gd = torch.rand(100, dtype=torch.float64) * 10.0 + 0.01
        mu = vm.mu(gd)
        assert mu.shape == (100,)
        assert torch.isfinite(mu).all()
        assert (mu > 0).all()

    def test_repr(self):
        """repr includes class and parameters."""
        r = repr(BinghamModel(tau_y=0.5, mu_inf=0.003))
        assert "BinghamModel" in r
        assert "tau_y=0.5" in r
        assert "mu_inf=0.003" in r


# ---------------------------------------------------------------------------
# Bingham = HerschelBulkley(n=1) identity
# ---------------------------------------------------------------------------


class TestBinghamHerschelBulkleyIdentity:
    """Bingham is a special case of Herschel-Bulkley with n=1."""

    def test_same_viscosity_values(self):
        """Bingham and H-B (n=1, K=mu_inf) give same viscosity."""
        tau_y = 0.3
        mu_inf = 0.005
        bing = BinghamModel(tau_y=tau_y, mu_inf=mu_inf)
        hb = HerschelBulkleyModel(tau_y=tau_y, K=mu_inf, n=1.0)

        gd = torch.tensor([0.1, 1.0, 10.0, 100.0], dtype=torch.float64)
        assert torch.allclose(bing.mu(gd), hb.mu(gd), atol=1e-10)


# ---------------------------------------------------------------------------
# QuemadaModel
# ---------------------------------------------------------------------------


class TestQuemadaModel:
    """Tests for QuemadaModel."""

    def test_formula(self):
        """Exact formula check."""
        vm = QuemadaModel(phi=0.3, k0=2.5, k_inf=0.5, gamma_dot_ref=1.0, mu_inf=0.001)
        gd = torch.tensor([4.0], dtype=torch.float64)
        k = 2.5 + 0.5 * (4.0 / 1.0) ** 0.5
        denom = 1.0 - 0.5 * k * 0.3
        expected = 0.001 * denom ** (-2)
        assert torch.allclose(
            vm.mu(gd),
            torch.tensor([expected], dtype=torch.float64),
            atol=1e-10,
        )

    def test_increasing_with_shear(self):
        """Viscosity increases with shear rate for concentrated suspensions."""
        vm = QuemadaModel(phi=0.4, k0=2.5, k_inf=1.0, gamma_dot_ref=1.0, mu_inf=0.001)
        gd_low = torch.tensor([0.1], dtype=torch.float64)
        gd_high = torch.tensor([10.0], dtype=torch.float64)
        # Higher shear -> higher k -> higher viscosity
        assert vm.mu(gd_low) < vm.mu(gd_high)

    def test_phi_effect(self):
        """Higher volume fraction gives higher viscosity."""
        vm_high = QuemadaModel(phi=0.4, mu_inf=0.001)
        vm_low = QuemadaModel(phi=0.1, mu_inf=0.001)
        gd = torch.tensor([1.0], dtype=torch.float64)
        assert vm_high.mu(gd) > vm_low.mu(gd)

    def test_zero_phi(self):
        """With phi=0, viscosity equals mu_inf (pure solvent)."""
        vm = QuemadaModel(phi=0.0, mu_inf=0.005)
        gd = torch.tensor([1.0, 10.0, 100.0], dtype=torch.float64)
        expected = torch.full_like(gd, 0.005)
        assert torch.allclose(vm.mu(gd), expected, atol=1e-10)

    def test_mu_zero(self):
        """mu_zero returns finite value for moderate phi."""
        vm = QuemadaModel(phi=0.3, k0=2.5, mu_inf=0.001)
        val = vm.mu_zero()
        assert val > 0
        assert val < float("inf")

    def test_batch(self):
        """Works with batch tensors."""
        vm = QuemadaModel(phi=0.3, k0=2.5, k_inf=0.5, mu_inf=0.001)
        gd = torch.rand(100, dtype=torch.float64) * 10.0 + 0.01
        mu = vm.mu(gd)
        assert mu.shape == (100,)
        assert torch.isfinite(mu).all()
        assert (mu > 0).all()

    def test_repr(self):
        """repr includes class and parameters."""
        r = repr(QuemadaModel(phi=0.3, k0=2.5))
        assert "QuemadaModel" in r
        assert "phi=0.3" in r
        assert "k0=2.5" in r


# ---------------------------------------------------------------------------
# StrainRateFunctionModel
# ---------------------------------------------------------------------------


class TestStrainRateFunctionModel:
    """Tests for StrainRateFunctionModel."""

    def test_linear_function(self):
        """Wrapping a linear function: mu = a * gd."""
        a = 0.05
        func = lambda gd: a * gd  # noqa: E731
        vm = StrainRateFunctionModel(func, mu_zero_val=0.0, mu_inf_val=1.0)
        gd = torch.tensor([1.0, 2.0, 10.0], dtype=torch.float64)
        expected = a * gd
        assert torch.allclose(vm.mu(gd), expected, atol=1e-10)

    def test_constant_function(self):
        """Wrapping a constant function gives Newtonian behaviour."""
        mu_val = 0.003
        func = lambda gd: torch.full_like(gd, mu_val)  # noqa: E731
        vm = StrainRateFunctionModel(func, mu_zero_val=mu_val, mu_inf_val=mu_val)
        gd = torch.tensor([0.1, 1.0, 100.0], dtype=torch.float64)
        assert torch.allclose(
            vm.mu(gd), torch.full_like(gd, mu_val), atol=1e-10,
        )

    def test_power_law_function(self):
        """Wrapping a power-law: mu = K * gd^(n-1)."""
        K, n = 1.0, 0.5
        func = lambda gd: K * gd.clamp(min=1e-30).pow(n - 1.0)  # noqa: E731
        vm = StrainRateFunctionModel(func, mu_zero_val=1e6, mu_inf_val=1e-6)
        gd = torch.tensor([4.0], dtype=torch.float64)
        expected = K * 4.0 ** (n - 1.0)
        assert torch.allclose(
            vm.mu(gd),
            torch.tensor([expected], dtype=torch.float64),
            atol=1e-10,
        )

    def test_mu_zero(self):
        """mu_zero returns the user-specified value."""
        func = lambda gd: gd  # noqa: E731
        vm = StrainRateFunctionModel(func, mu_zero_val=0.5)
        assert vm.mu_zero() == 0.5

    def test_mu_inf(self):
        """mu_inf returns the user-specified value."""
        func = lambda gd: gd  # noqa: E731
        vm = StrainRateFunctionModel(func, mu_inf_val=0.002)
        assert vm.mu_inf() == 0.002

    def test_batch(self):
        """Works with batch tensors."""
        func = lambda gd: gd ** 0.5  # noqa: E731
        vm = StrainRateFunctionModel(func)
        gd = torch.rand(100, dtype=torch.float64) * 10.0 + 0.01
        mu = vm.mu(gd)
        assert mu.shape == (100,)
        assert torch.isfinite(mu).all()
        assert (mu > 0).all()

    def test_repr(self):
        """repr includes function name."""
        def my_viscosity(gd):
            return gd * 0.01

        vm = StrainRateFunctionModel(my_viscosity)
        r = repr(vm)
        assert "StrainRateFunctionModel" in r
        assert "my_viscosity" in r

    def test_repr_lambda(self):
        """repr handles lambda functions gracefully."""
        func = lambda gd: gd  # noqa: E731
        vm = StrainRateFunctionModel(func)
        r = repr(vm)
        assert "<lambda>" in r
